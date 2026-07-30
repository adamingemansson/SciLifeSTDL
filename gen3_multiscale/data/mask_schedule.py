"""Stratified mask-schedule generator -- makes
`configs/architectureN.yaml`'s `masking.strata` block (Phase 0 item 7:
predeclared hole-size/shape strata, "The schedule must cover predeclared
hole-size/shape strata; report results by the same strata") actually
functional.

Confirmed real, concrete bug (2nd Codex re-audit of commit 547f51e): the
reused `mask_bank.py::make_split` (verbatim copy, CONTRACT.md section 2)
expects `masking_cfg.strategy`/`masking_cfg.params` -- calling it with
the configs' literal `masking.strata` list (as-is, with no consuming
code) would raise `ValueError("unknown masking strategy None")`, exactly
as the audit predicted. This module is the missing piece, not a change
to `mask_bank.py` itself: it converts one stratum entry (`name`,
`radius_range`, `radius_unit`, `shape`) into a real `masking_cfg` dict
`make_split`/`build_mask_bank` already understand, then calls
`build_mask_bank` once per stratum (with per-stratum seed offsets so
seeds never collide across strata) and merges the results into one
combined, stratum-tagged schedule.

This is genuinely different in kind from the still-missing real
per-sample HEST-1k data builder (CONTRACT.md section 21): mask
scheduling operates purely on coordinate/slice-id arrays, exactly like
Phase 2's boundary extraction, so it is fully testable on synthetic
coordinates today, with no dependency on real loaded GEX/GigaPath
features.
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.boundary_graph import EmptyBoundaryError, extract_boundary_and_local_context

# Keeps every stratum's seed range from ever colliding with another
# stratum's, for any split_seeds a caller supplies (default validation=
# 700_000/test=900_000 leaves ample room below this stride).
_STRATUM_SEED_STRIDE = 1_000_000

# Bumped whenever the mask/seed-GENERATION ALGORITHM itself changes (not
# when strata/split content changes -- that's already covered by
# strata_fingerprint/dataset_fingerprint/spatial_fingerprint). Recorded
# in every bank and validated on load/ensure, so an on-disk schedule
# built by an OLDER, buggy version of this algorithm gets correctly
# rejected and regenerated rather than silently reused -- fixes a real
# gap (5th Codex re-audit of commit c02a5d1): without this, fixing
# build_stratified_training_seed_bank's seed-collision bug below would
# have left any already-persisted (buggy) schedule looking identical by
# every OTHER fingerprint field, since none of dataset/spatial/strata
# content actually changed -- only the algorithm did.
_MASK_GENERATION_VERSION = "3"
_MAX_MASK_ATTEMPTS = 1000


def validate_mask_has_observed_boundary(
    coords3d: np.ndarray,
    obs_names: Iterable[str],
    context_obs_names: Iterable[str],
    query_obs_names: Iterable[str],
    *,
    k_neighbors: int = 6,
) -> dict:
    """Require a realized mask to represent a hole with observed tissue around it.

    A mask that removes an entire disconnected tissue fragment can have a
    non-empty context and query while still having no observed boundary.
    Such a mask is not a missing-tissue reconstruction example and makes
    the model's mandatory boundary-attention branch undefined.
    """
    names = np.asarray([str(x) for x in obs_names])
    position_by_name = {name: pos for pos, name in enumerate(names)}
    try:
        context_pos = np.asarray([position_by_name[str(x)] for x in context_obs_names], dtype=int)
        query_pos = np.asarray([position_by_name[str(x)] for x in query_obs_names], dtype=int)
    except KeyError as exc:
        raise ValueError(f"mask references unknown observation {exc.args[0]!r}") from exc
    result = extract_boundary_and_local_context(
        np.asarray(coords3d)[context_pos, :2], np.asarray(coords3d)[query_pos, :2],
        k_neighbors=k_neighbors, local_k=1, max_rings=1,
    )
    return result.diagnostic


def stratum_to_masking_cfg(stratum: dict, default_n_patches: int = 1) -> dict:
    """Convert one `masking.strata[i]` entry into a `masking_cfg` dict
    `mask_bank.make_split`/`build_mask_bank` actually understand
    (`strategy="random_dropout_patches"`, matching
    `gen2_architectures/data/masking.py`'s existing hole-size/shape
    stratification support -- CONTRACT.md section 4's plan)."""
    required = {"radius_range", "radius_unit", "shape"}
    missing = required - set(stratum)
    if missing:
        raise ValueError(f"stratum {stratum.get('name', '?')!r} is missing required fields: {sorted(missing)}")
    return {
        "strategy": "random_dropout_patches",
        "params": {
            "n_patches": int(stratum.get("n_patches", default_n_patches)),
            "radius_range": list(stratum["radius_range"]),
            "radius_unit": str(stratum["radius_unit"]),
            "shape": str(stratum["shape"]),
        },
    }


def strata_fingerprint(strata: list[dict], split_counts: dict[str, int] | None = None, split_seeds: dict[str, int] | None = None) -> str:
    """A single combined fingerprint covering the ORDERED strata
    definition (plus split counts/seeds) -- fixes a real, confirmed gap
    (3rd Codex re-audit of commit ca7cf53): "it has no combined
    fingerprint covering the ordered strata definition", only
    `per_stratum_masking_fingerprint` entries an unordered dict could
    reshuffle without changing. Converts every stratum to its real
    `masking_cfg` (via `stratum_to_masking_cfg`) first, so this
    fingerprint changes if a stratum's radius/shape/unit changes even if
    its `name` doesn't."""
    payload = {
        "strata": [{"name": s.get("name"), "masking_cfg": stratum_to_masking_cfg(s)} for s in strata],
        "split_counts": dict(split_counts or {"validation": 4, "test": 8}),
        "split_seeds": dict(split_seeds or {"validation": 700_000, "test": 900_000}),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_stratified_mask_bank(
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    obs_names: Iterable[str],
    strata: list[dict],
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    """Build ONE combined mask bank covering every predeclared hole-size/
    shape stratum, so a training/validation/test schedule genuinely
    stratifies by hole size/shape rather than blending them into one
    distribution. Every record is tagged with its `stratum` name;
    `stratum_records` below is the per-stratum equivalent of
    `mask_bank.split_records`.

    Each stratum's own masking_fingerprint (from calling
    `mask_bank.build_mask_bank` once per stratum internally) is preserved
    under `per_stratum_masking_fingerprint`, so an existing audited
    per-stratum staleness check still has something to compare against;
    this function does not attempt to reproduce `load_mask_bank`'s
    on-disk staleness semantics itself.
    """
    if not strata:
        raise ValueError("strata must be a non-empty list")
    stratum_names = [s.get("name") for s in strata]
    if len(set(stratum_names)) != len(strata) or any(name is None for name in stratum_names):
        raise ValueError("every stratum must have a unique, non-null 'name'")

    split_counts = dict(split_counts or {"validation": 4, "test": 8})
    split_seeds = dict(split_seeds or {"validation": 700_000, "test": 900_000})

    names_arr = np.asarray([str(x) for x in obs_names])
    all_records = []
    per_stratum_fingerprints = {}
    accepted_query_sets_by_split: dict[str, set[tuple[str, ...]]] = {
        split: set() for split in split_counts
    }
    for i, stratum in enumerate(strata):
        stratum_name = stratum["name"]
        masking_cfg = stratum_to_masking_cfg(stratum)
        stratum_seeds = {split: int(seed) + i * _STRATUM_SEED_STRIDE for split, seed in split_seeds.items()}
        max_context = mask_bank._cfg_get(masking_cfg, "max_context_points", None)
        context_selection = str(mask_bank._cfg_get(masking_cfg, "context_selection", "random"))
        for split, raw_count in split_counts.items():
            count = int(raw_count)
            if count < 0:
                raise ValueError(f"split_counts[{split!r}] must be non-negative, got {count}")
            for record_index in range(count):
                accepted = None
                for attempt in range(_MAX_MASK_ATTEMPTS):
                    # Each record gets a disjoint deterministic retry stream:
                    # index + attempt*count.  Retrying record i can therefore
                    # never consume record j's seed.
                    seed = int(stratum_seeds[split]) + record_index + attempt * max(count, 1)
                    context, query = mask_bank.make_split(coords3d, slice_ids, masking_cfg, seed)
                    context = mask_bank.cap_context_mask(
                        context, max_context, seed, coords3d=coords3d,
                        query_mask=query, selection=context_selection,
                    )
                    if not query.any() or not context.any():
                        continue
                    context_names = names_arr[context].tolist()
                    query_names = names_arr[query].tolist()
                    try:
                        validate_mask_has_observed_boundary(
                            coords3d, names_arr, context_names, query_names,
                        )
                    except EmptyBoundaryError:
                        continue
                    query_identity = tuple(sorted(query_names))
                    if query_identity in accepted_query_sets_by_split[split]:
                        continue
                    accepted = {
                        "split": split,
                        "index": record_index,
                        "seed": seed,
                        "seed_attempt": attempt,
                        "context_obs_names": context_names,
                        "query_obs_names": query_names,
                        "stratum": stratum_name,
                    }
                    accepted_query_sets_by_split[split].add(query_identity)
                    break
                if accepted is None:
                    raise ValueError(
                        f"could not realize a non-empty, boundary-valid, unique {split!r} mask "
                        f"for stratum {stratum_name!r}, index {record_index} within "
                        f"{_MAX_MASK_ATTEMPTS} deterministic attempts"
                    )
                all_records.append(accepted)
        per_stratum_fingerprints[stratum_name] = mask_bank.masking_fingerprint(
            masking_cfg, split_counts, stratum_seeds,
        )

    return {
        "version": 1,
        "mask_generation_version": _MASK_GENERATION_VERSION,
        "kind": "stratified_mask_bank",
        "dataset_fingerprint": mask_bank.dataset_fingerprint(names_arr),
        "spatial_fingerprint": mask_bank.spatial_fingerprint(coords3d, slice_ids),
        "strata_fingerprint": strata_fingerprint(strata, split_counts, split_seeds),
        "strata": stratum_names,
        "per_stratum_masking_fingerprint": per_stratum_fingerprints,
        "split_counts": split_counts,
        "split_seeds": split_seeds,
        "n_obs": int(len(names_arr)),
        "records": all_records,
    }


def stratum_records(bank: dict, split: str, stratum: str) -> list[dict]:
    """Records for one (split, stratum) pair, sorted by index -- the
    per-stratum equivalent of `mask_bank.split_records`, so evaluation
    results can be reported per stratum (CONTRACT.md section 4)."""
    records = [r for r in bank["records"] if r["split"] == split and r["stratum"] == stratum]
    return sorted(records, key=lambda r: int(r["index"]))


def save_stratified_mask_bank(bank: dict, path: str | Path) -> Path:
    """Atomic write, mirroring `mask_bank.save_mask_bank` exactly (same
    process-specific-temp-file-then-os.replace discipline, for the same
    reason: concurrent single-GPU jobs may all request the same immutable
    stratified bank at once)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(bank, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_stratified_mask_bank(
    path: str | Path,
    obs_names: Iterable[str],
    *,
    coords3d: np.ndarray | None = None,
    slice_ids: np.ndarray | None = None,
    strata: list[dict] | None = None,
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    """Fail-closed staleness validation, mirroring `mask_bank.load_mask_bank`
    exactly, but checking `strata_fingerprint` (this module's OWN combined
    ordered-strata fingerprint) rather than a single `masking_fingerprint`
    -- a stratified bank has no single masking config to fingerprint
    against."""
    path = Path(path)
    bank = json.loads(path.read_text())
    expected_dataset = mask_bank.dataset_fingerprint(obs_names)
    if bank.get("dataset_fingerprint") != expected_dataset:
        raise ValueError(
            f"stratified mask bank {path} was built for a different observation set; "
            "regenerate it for this exact QC/alignment result"
        )
    supplied = [coords3d is not None, slice_ids is not None, strata is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("coords3d, slice_ids and strata must be supplied together")
    if all(supplied):
        expected_spatial = mask_bank.spatial_fingerprint(coords3d, slice_ids)
        expected_strata = strata_fingerprint(strata, split_counts, split_seeds)
        if bank.get("spatial_fingerprint") != expected_spatial:
            raise ValueError(
                f"stratified mask bank {path} was built for different coordinates or slice IDs; "
                "regenerate it"
            )
        if bank.get("strata_fingerprint") != expected_strata:
            raise ValueError(
                f"stratified mask bank {path} was built with different strata, split counts, "
                "or split seeds; use a distinct path or regenerate it"
            )
        if bank.get("mask_generation_version") != _MASK_GENERATION_VERSION:
            raise ValueError(
                f"stratified mask bank {path} was built with mask-generation algorithm version "
                f"{bank.get('mask_generation_version')!r}, but this code is version "
                f"{_MASK_GENERATION_VERSION!r}; regenerate it -- the underlying generation logic "
                "changed even though the strata/split content did not"
            )
    return bank


def build_stratified_training_seed_bank(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    n_items: int, base_seed: int, strata: list[dict],
    unique_masks_per_stratum: int | None = None,
    require_full_seed_pool: bool = False,
) -> dict:
    """Stratified equivalent of `mask_bank.build_training_seed_bank`: a
    lossless, deterministic (stratum, seed) schedule for `n_items`
    training draws, round-robining across strata so the immutable
    training schedule genuinely covers every predeclared hole-size/shape
    stratum -- fixes a real, confirmed gap (3rd Codex re-audit of commit
    ca7cf53): "its default schedule contains validation and test masks
    only, not training masks." (`build_stratified_mask_bank` above
    correctly only ever covered validation/test -- explicit per-record
    storage -- matching `mask_bank.py`'s own existing split between
    explicit validation/test records and a lossless TRAINING seed
    schedule; this is that same split's stratified counterpart, not a
    naive third split_counts entry bolted onto the explicit-record path,
    which would reproduce the exact multi-gigabyte-JSON problem
    `build_training_seed_bank`'s own docstring already explains.)

    Two real, confirmed gaps fixed here (4th Codex re-audit of commit
    0fd46e5):
    - **Misleading parameter name.** The previous `unique_mask_count`
      name suggested a GLOBAL unique-mask total, but the value actually
      meant "unique seeds PER STRATUM" -- e.g. with 4 strata and the old
      name set to 1, the schedule still produced 4 distinct (stratum,
      seed) combinations, not 1, because each stratum's seed range is
      independently offset. Renamed to `unique_masks_per_stratum`, which
      states the actual semantics directly; no behavior changed.
    - **Coverage claim without an enforced precondition.** Round-robin
      (item i -> stratum i % n_strata) genuinely covers every stratum
      only when `n_items >= len(strata)` -- fewer items than strata
      deterministically skips the remainder, which is a real
      precondition, not "starving one by chance" as the previous
      docstring implied. Now raises instead of silently producing partial
      coverage a caller didn't ask for.
    - **No spatial fingerprint.** The schedule used to fingerprint only
      `obs_names` -- the same barcodes with altered coordinates would
      silently reuse a schedule whose generated spatial holes would
      actually differ. `coords3d`/`slice_ids` are now required inputs,
      fingerprinted via the same `mask_bank.spatial_fingerprint`
      `build_stratified_mask_bank` already uses.

    Each stratum's seed range is offset by the same `_STRATUM_SEED_STRIDE`
    `build_stratified_mask_bank` uses, so a training draw's seed can
    never collide with a different stratum's draw.

    `unique_masks_per_stratum` is a CAP on the per-stratum seed-cycling
    pool, not automatically a promise that every seed in that pool is
    actually drawn -- round-robin assignment only visits a given stratum
    `n_items // n_strata` times (give or take one), so a caller asking
    for a larger pool than that (e.g. `unique_masks_per_stratum=99` with
    only 4 items total) legitimately just doesn't exhaust it; this is
    intentional, matching the corrected semantics from the 5th Codex
    re-audit of commit c02a5d1 (see
    `test_build_stratified_training_seed_bank_allows_unique_masks_per_stratum_larger_than_n_items`).
    The output always reports `realized_unique_seeds_per_stratum` (the
    ACTUAL count achieved per stratum) so a caller can verify their
    coverage intent was met without guessing from `n_items`/`n_strata`
    arithmetic. `require_full_seed_pool=True` (a real, confirmed gap --
    6th Codex re-audit of commit 06f5cce: "the function does not
    guarantee that the entire requested pool is realized") turns that
    same check into a fail-closed precondition: it raises unless every
    stratum actually realizes the full `unique_masks_per_stratum` seeds,
    i.e. `n_items >= n_strata * unique_masks_per_stratum`. Defaults to
    False rather than becoming the unconditional behavior, because
    flipping the DEFAULT would silently break the 5th-round decision
    that oversized pools are legitimate, not an error -- callers that
    genuinely need the guarantee (e.g. "validation must show exactly N
    distinct masks per stratum") now have an explicit, opt-in way to
    demand it instead.
    """
    n_items = int(n_items)
    base_seed = int(base_seed)
    if n_items < 1:
        raise ValueError("training seed bank requires at least one item")
    if not strata:
        raise ValueError("strata must be a non-empty list")
    stratum_names = [s.get("name") for s in strata]
    if len(set(stratum_names)) != len(strata) or any(name is None for name in stratum_names):
        raise ValueError("every stratum must have a unique, non-null 'name'")
    n_strata = len(strata)
    if n_items < n_strata:
        raise ValueError(
            f"n_items ({n_items}) must be >= the number of strata ({n_strata}) to guarantee every "
            "stratum is covered at least once -- round-robin assignment cannot cover more strata "
            "than there are items"
        )

    if unique_masks_per_stratum is None:
        # Real, confirmed gap (7th Codex re-audit of commit 2782ff0):
        # defaulting to n_items unconditionally means
        # require_full_seed_pool=True combined with the default is
        # MATHEMATICALLY IMPOSSIBLE to satisfy whenever n_strata > 1
        # (a stratum is only ever visited n_items // n_strata times, always
        # less than n_items for n_strata > 1) -- a guaranteed-to-raise
        # footgun, not a real precondition. In strict mode, default to the
        # largest value every stratum can equally guarantee -- floor
        # division (n_items // n_strata), NOT a "ceiling" (a wording slip
        # in an earlier version of this comment the 8th audit correctly
        # flagged: floor division rounds DOWN, so it's the maximum
        # UNIFORMLY achievable count, not an upper rounding bound) --
        # instead, so the default trivially satisfies its own guarantee by
        # construction; the permissive (non-strict) default of n_items is
        # unchanged, preserving the 5th round's "oversized pools are
        # legitimate" decision for the common, non-strict case.
        unique_masks_per_stratum = (n_items // n_strata) if require_full_seed_pool else n_items
    else:
        unique_masks_per_stratum = int(unique_masks_per_stratum)
    if unique_masks_per_stratum < 1:
        raise ValueError(f"unique_masks_per_stratum must be positive, got {unique_masks_per_stratum}")
    if unique_masks_per_stratum >= _STRATUM_SEED_STRIDE:
        raise ValueError(
            f"unique_masks_per_stratum ({unique_masks_per_stratum}) must be < the per-stratum seed "
            f"stride ({_STRATUM_SEED_STRIDE}), or a stratum's own local seed range would overflow "
            "into the next stratum's offset block and collide with it"
        )

    # Real, confirmed bug fixed here (5th Codex re-audit of commit
    # c02a5d1), reproduced exactly as a regression test before fixing:
    # the previous formula used `i % unique_masks_per_stratum` (the
    # GLOBAL item index modulo the count) as this stratum's local seed
    # index. But a given stratum is only visited every n_strata-th item,
    # so the actual set of values `i % unique_masks_per_stratum` takes
    # for THAT stratum's i's is a strict subset of
    # {0, ..., unique_masks_per_stratum-1} whenever n_strata shares a
    # common factor with unique_masks_per_stratum (e.g. 4 strata and
    # unique_masks_per_stratum=64 realized only 64/4=16 distinct seeds
    # per stratum, not 64 -- confirmed by direct computation). Fixed by
    # tracking `occurrence_in_stratum` -- how many times THIS stratum has
    # been visited so far, independent of n_strata -- and taking that
    # modulo unique_masks_per_stratum instead; this correctly cycles
    # through every one of the unique_masks_per_stratum seeds regardless
    # of the relationship between n_strata and unique_masks_per_stratum.
    items = []
    seeds_by_stratum: dict[str, set[int]] = {name: set() for name in stratum_names}
    for i in range(n_items):
        stratum_index = i % n_strata
        occurrence_in_stratum = i // n_strata
        local_seed_index = occurrence_in_stratum % unique_masks_per_stratum
        seed = base_seed + local_seed_index + stratum_index * _STRATUM_SEED_STRIDE
        items.append({"stratum": stratum_names[stratum_index], "seed": seed})
        seeds_by_stratum[stratum_names[stratum_index]].add(seed)

    realized_unique_seeds_per_stratum = {name: len(seeds) for name, seeds in seeds_by_stratum.items()}
    if require_full_seed_pool:
        short = {
            name: count for name, count in realized_unique_seeds_per_stratum.items()
            if count < unique_masks_per_stratum
        }
        if short:
            raise ValueError(
                f"require_full_seed_pool=True but these strata did not realize the full "
                f"unique_masks_per_stratum={unique_masks_per_stratum} seed pool: {short} -- "
                f"increase n_items to at least n_strata * unique_masks_per_stratum "
                f"({n_strata} * {unique_masks_per_stratum} = {n_strata * unique_masks_per_stratum}), "
                f"got n_items={n_items}"
            )

    names = [str(x) for x in obs_names]
    return {
        "version": 1,
        "mask_generation_version": _MASK_GENERATION_VERSION,
        "kind": "stratified_training_seed_schedule",
        "dataset_fingerprint": mask_bank.dataset_fingerprint(names),
        "spatial_fingerprint": mask_bank.spatial_fingerprint(coords3d, slice_ids),
        "n_obs": len(names),
        "n_items": n_items,
        "base_seed": base_seed,
        "unique_masks_per_stratum": unique_masks_per_stratum,
        "realized_unique_seeds_per_stratum": realized_unique_seeds_per_stratum,
        "strata_fingerprint": strata_fingerprint(strata),
        "strata": stratum_names,
        "items": items,
    }


def ensure_stratified_training_seed_bank(
    path: str | Path,
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    obs_names: Iterable[str],
    n_items: int,
    base_seed: int,
    strata: list[dict],
    unique_masks_per_stratum: int | None = None,
    require_full_seed_pool: bool = False,
) -> tuple[dict, Path]:
    """Load-or-atomically-create, mirroring `mask_bank.ensure_training_seed_bank`:
    reuse an existing on-disk schedule if every identifying field and the
    exact (stratum, seed) sequence still match, otherwise build and
    persist a fresh one. Fails closed (raises) on any mismatch -- now
    including a changed `spatial_fingerprint` (see
    `build_stratified_training_seed_bank`'s docstring) -- rather than
    silently reusing a stale or altered schedule.

    Returns the freshly-recomputed `expected` dict on a validated reuse,
    NOT the raw on-disk JSON (real, confirmed gap -- 7th Codex re-audit
    of commit 2782ff0): this module's own output schema has grown new
    derived fields over time (e.g. `realized_unique_seeds_per_stratum`,
    added this round); an on-disk bank written by an OLDER version of
    this module would simply lack that key. Bumping
    `_MASK_GENERATION_VERSION` would be the wrong tool here -- that field
    is reserved for changes to the actual (stratum, seed) GENERATION
    algorithm (see its own docstring), and adding a derived reporting
    field changes neither the algorithm nor the resulting seeds. Once
    every identifying field AND the exact `items` sequence are confirmed
    identical between the on-disk bank and a fresh recomputation, the two
    are semantically equivalent by definition -- returning the fresh
    `expected` (which is always complete under the CURRENT schema)
    instead of the possibly-schema-stale on-disk dict closes the gap
    without conflating "the seeds changed" with "the reporting schema
    grew a field".

    Also ATOMICALLY REWRITES the on-disk file itself with the current
    representation whenever it's schema-stale (real, confirmed gap --
    8th Codex re-audit of commit 7b5c267): returning the fresh dict fixes
    what THIS caller sees, but left the stale JSON sitting on disk for
    any other reader -- a different process calling this same function
    concurrently, or any future code that reads the file directly instead
    of through this function -- to still see it. Since the content is
    already proven semantically equivalent by the checks above, rewriting
    it is safe; this makes the on-disk artifact self-healing instead of
    merely working around it in memory."""
    path = Path(path)
    expected = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items, base_seed, strata,
        unique_masks_per_stratum=unique_masks_per_stratum,
        require_full_seed_pool=require_full_seed_pool,
    )
    if path.exists():
        bank = json.loads(path.read_text())
        for key in (
            "kind", "mask_generation_version", "dataset_fingerprint", "spatial_fingerprint", "n_obs",
            "n_items", "base_seed", "unique_masks_per_stratum", "strata_fingerprint",
        ):
            if bank.get(key) != expected.get(key):
                raise ValueError(
                    f"stratified training seed bank {path} does not match the current data/run "
                    f"for field {key!r}; use a new path or remove the stale bank"
                )
        if bank.get("items") != expected["items"]:
            raise ValueError(f"stratified training seed bank {path} contains an altered schedule")
        if bank != expected:
            save_stratified_mask_bank(expected, path)  # self-heal: rewrite the schema-stale file in place
        return expected, path
    save_stratified_mask_bank(expected, path)  # a generic atomic-JSON writer, not mask-bank-specific -- mask_bank.py itself reuses save_mask_bank the same way for its own training seed banks
    return expected, path


def ensure_stratified_mask_bank(
    path: str | Path,
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    obs_names: Iterable[str],
    strata: list[dict],
    split_counts: dict[str, int] | None = None,
    split_seeds: dict[str, int] | None = None,
) -> dict:
    """Load-or-atomically-create, mirroring `mask_bank.ensure_mask_bank`
    exactly: reuse an existing on-disk bank if its fingerprints match,
    otherwise build and persist a fresh one, then re-read and validate
    after the atomic write (so a concurrent job writing a differently
    configured bank to the same path is caught, not silently trusted)."""
    path = Path(path)
    if path.exists():
        return load_stratified_mask_bank(
            path, obs_names, coords3d=coords3d, slice_ids=slice_ids, strata=strata,
            split_counts=split_counts, split_seeds=split_seeds,
        )
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, strata, split_counts=split_counts, split_seeds=split_seeds)
    save_stratified_mask_bank(bank, path)
    return load_stratified_mask_bank(
        path, obs_names, coords3d=coords3d, slice_ids=slice_ids, strata=strata,
        split_counts=split_counts, split_seeds=split_seeds,
    )
