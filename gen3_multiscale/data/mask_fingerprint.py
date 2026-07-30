"""Realized-mask fingerprinting -- Step 3 of the real Gen3 data
builder/trainer (Adam's explicit 9-step implementation order,
CONTRACT.md section 30, item 3: "Fingerprint sorted realized query
identities"; sharpened by the 10th Codex re-audit of commit 9592d9e's
closing instruction: "implement Step 3 using realized composite query
identities rather than seeds"; hardened by the 11th Codex re-audit of
commit 9dab8fe, finding #3).

`mask_bank.py`/`mask_schedule.py`'s training seed banks deliberately
store only a (stratum, seed) SCHEDULE, not every realized context/query
barcode list, to avoid multi-gigabyte JSON (mask_bank.build_training_seed_bank's
own docstring). Their `unique_mask_count`/`unique_masks_per_stratum`
fields count DISTINCT SEED VALUES and implicitly assume distinct seeds
always realize distinct masks. `mask_bank.make_split` is a deterministic
function of (data, masking_cfg, seed), but it is NOT proven injective in
seed OR in masking_cfg: two different seeds (even across two DIFFERENT
strata/configurations, not just within one) are not guaranteed to
produce two different actual context/query splits (a real, repeatedly-
flagged gap across this project's audit history). This module closes
that gap by REALIZING candidate draws and fingerprinting the ACTUAL
resulting query set by real composite (sample_id, spot) identity
(`dataset_manifest.composite_spot_id` -- the true globally-unique spot
identity, since raw Visium barcodes are not globally unique across
samples), rather than trusting the seed integer (or which masking
config produced it) as a stand-in for mask identity.

Also provides a real, positive cross-split leakage check: for one
sample, a training draw's realized query spots must never coincide with
a held-out (validation/test) mask's realized query spots for that same
sample, or the model would be trained to reconstruct exactly what it is
later evaluated on. (Cross-SAMPLE leakage is already structurally
prevented by dataset_manifest's patient-disjoint sample-level split --
this module does not re-check that; it checks WITHIN one sample's own
realized masks, which the sample-level split cannot see.)

**Sample-split awareness (12th Codex re-audit of commit 1bb66d6, finding
#7/#8, CONFIRMED).** An earlier version of this module's report builder
processed train/validation/test masks for one sample as if all three
normally coexist -- but under this project's patient-disjoint SAMPLE-
level split (`dataset_manifest.py`), a TRAINING sample's real evaluation
happens on ENTIRELY DIFFERENT held-out samples, not on its own masks.
`mask_bank.py`/`mask_schedule.py`'s per-sample "validation"/"test"
split_counts concept, when realized on a training sample at all, is at
most a SECONDARY same-sample capacity/early-stopping diagnostic. This
module now has two distinct, sample-role-aware report builders:
`build_training_sample_mask_report` (primary = this sample's training
schedule; same-sample validation/test masks, if any, are reported under
an explicitly separate, clearly-labeled `"same_sample_capacity_diagnostic"`
section) and `build_held_out_sample_mask_report` (primary = the ONE
split -- validation or test -- this held-out sample actually belongs
to; fails closed if its mask bank contains any record for a different
split).

`build_collision_free_training_schedule` (12th Codex re-audit finding
#5, CONFIRMED: detecting a seed/mask collision is not the same as
PREVENTING one -- production needs deterministic avoidance, not a test
that gets lucky searching thousands of seeds) deterministically builds
a training schedule that is guaranteed, by construction, to never query
a reserved (validation/test) composite identity and never repeat an
already-accepted training mask.

`verify_no_duplicate_masks_within_split` (finding #7) rejects two
records within the SAME split (e.g. two "validation" records) realizing
an identical mask -- a gap the prior cross-split-only leakage check
could not see.

Every report is bound to SHA256 fingerprints of the manifest, spatial
lattice, and strata it was built from (finding #6) -- a consumer (Step
8's preflight gate, or the trainer itself) is expected to recompute
these same fingerprints from its own live data and refuse a report
whose fingerprints don't match, rather than trusting a report file's
mere existence.
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.boundary_graph import EmptyBoundaryError
from gen3_multiscale.data.dataset_manifest import composite_spot_id
from gen3_multiscale.data.mask_schedule import (
    prepare_masking_cfg_for_sample,
    _MASK_GENERATION_VERSION, _STRATUM_SEED_STRIDE, strata_fingerprint,
    stratum_to_masking_cfg, validate_mask_has_observed_boundary,
)

_REPORT_VERSION = 4
_SCHEDULE_VERSION = 2


class EmptyMaskRealizationError(ValueError):
    """Raised by `realize_seed_and_fingerprint` ONLY when a
    (masking_cfg, seed) draw produces an empty context or query -- the
    one realization failure `build_collision_free_training_schedule`'s
    seed-retry loop may treat as "try the next seed" (13th Codex
    re-audit finding #8, CONFIRMED: a bare `except ValueError: continue`
    also swallowed structural errors -- a manifest mismatch, a malformed
    masking_cfg -- retrying up to `max_attempts_per_item` times on an
    error that could never succeed on a different seed, instead of
    failing immediately). Subclasses ValueError so any EXISTING `except
    ValueError` caller elsewhere in this codebase is unaffected."""


def sorted_composite_query_fingerprint(sample_id: str, query_obs_names: Iterable[str]) -> str:
    """SHA256 of the SORTED set of composite (sample_id, spot) identities
    for a realized query set -- order-independent (two masks with the
    same query spots in a different iteration order must fingerprint
    identically), and namespaced by sample_id via composite_spot_id so
    fingerprints from different samples can never collide by coincidence
    even if their raw barcodes do.

    Rejects a query list containing the SAME barcode more than once (11th
    Codex re-audit finding #3): a real query set is a SET of distinct
    spots, and silently deduplicating (or worse, letting a duplicate
    inflate `sorted(...)`'s output non-deterministically depending on
    how many times it repeats) would hide a real upstream bug rather
    than fail loudly."""
    query_obs_names = [str(b) for b in query_obs_names]
    if len(query_obs_names) != len(set(query_obs_names)):
        duplicates = sorted({b for b in query_obs_names if query_obs_names.count(b) > 1})
        raise ValueError(
            f"{sample_id}: query_obs_names contains duplicate barcode(s): {duplicates[:10]} -- a "
            "realized query set must be a set of distinct spots"
        )
    composite_ids = sorted(composite_spot_id(sample_id, b) for b in query_obs_names)
    return sha256("\n".join(composite_ids).encode("utf-8")).hexdigest()


def validate_realized_barcodes_against_manifest(manifest: dict, sample_id: str, obs_names: Iterable[str]) -> None:
    """Fail-closed: every one of `obs_names` (context OR query) must be
    a barcode the dataset manifest (dataset_manifest.build_dataset_manifest's
    output) actually declared for `sample_id`. 11th Codex re-audit
    finding #3 ("validate every realized identity against the manifest
    and expected sample"): a realized mask referencing a barcode the
    manifest never declared for this sample -- or a sample_id the
    manifest doesn't even have -- indicates the real data drifted since
    the manifest was built, or a caller passed the wrong sample's
    coordinates/obs_names; either way, this must fail loudly rather than
    silently fingerprint bogus data."""
    if sample_id not in manifest["samples"]:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    declared = set(manifest["samples"][sample_id]["barcodes"])
    obs_names = [str(b) for b in obs_names]
    unexpected = sorted(set(obs_names) - declared)
    if unexpected:
        raise ValueError(
            f"{sample_id}: realized mask references {len(unexpected)} barcode(s) the dataset "
            f"manifest never declared for this sample (examples: {unexpected[:10]}) -- the real "
            "data changed since the manifest was built, or this is the wrong sample's coordinates"
        )


def realize_seed_and_fingerprint(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    sample_id: str, masking_cfg: dict, seed: int, *, manifest: dict | None = None,
    spatial_adjacency: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict:
    """Realize ONE (masking_cfg, seed) draw exactly as
    `mask_bank.build_mask_bank` does internally (same `make_split` call,
    same context capping), and fingerprint its REALIZED query set by
    real composite spot identity. Returns a record with the same
    `context_obs_names`/`query_obs_names` shape as a `mask_bank.py`
    record, plus `query_composite_fingerprint`.

    `manifest` (optional): when supplied, both `context_obs_names` and
    `query_obs_names` are validated against it (see
    validate_realized_barcodes_against_manifest) before fingerprinting."""
    names = np.asarray([str(x) for x in obs_names])
    context, query = mask_bank.make_split(coords3d, slice_ids, masking_cfg, seed)
    max_context = mask_bank._cfg_get(masking_cfg, "max_context_points", None)
    context_selection = str(mask_bank._cfg_get(masking_cfg, "context_selection", "random"))
    context = mask_bank.cap_context_mask(
        context, max_context, seed, coords3d=coords3d, query_mask=query, selection=context_selection,
    )
    if not query.any() or not context.any():
        raise EmptyMaskRealizationError(f"mask seed {seed} produced an empty context or query")
    context_obs_names = names[context].tolist()
    query_obs_names = names[query].tolist()
    validate_mask_has_observed_boundary(
        coords3d, names, context_obs_names, query_obs_names,
        spatial_adjacency=spatial_adjacency,
    )
    if manifest is not None:
        validate_realized_barcodes_against_manifest(manifest, sample_id, context_obs_names)
        validate_realized_barcodes_against_manifest(manifest, sample_id, query_obs_names)
    return {
        "seed": int(seed),
        "context_obs_names": context_obs_names,
        "query_obs_names": query_obs_names,
        "query_composite_fingerprint": sorted_composite_query_fingerprint(sample_id, query_obs_names),
    }


def verify_realized_pool_uniqueness(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    sample_id: str, items: list[dict], *, manifest: dict | None = None,
) -> dict:
    """Fail-closed: realize every `{"label", "masking_cfg", "seed"}`
    entry in `items` and confirm their REALIZED query composite
    fingerprints are pairwise distinct ACROSS THE WHOLE POOL --
    regardless of which masking_cfg/stratum produced them, not merely
    within one config's own seed range. 11th Codex re-audit finding #3
    ("detect identical masks across different strata/configurations,
    not only seeds passed to one masking configuration"): two DIFFERENT
    masking configs (e.g. two strata with overlapping radius ranges)
    could realize an identical query set just as easily as two seeds
    within the same config can. `label` is any caller-chosen identifier
    (e.g. f"{stratum}:{seed}") used only for reporting which entries
    collided."""
    fingerprint_by_label: dict[str, str] = {}
    labels_by_fingerprint: dict[str, list[str]] = {}
    for item in items:
        label = str(item["label"])
        record = realize_seed_and_fingerprint(
            coords3d, slice_ids, obs_names, sample_id, item["masking_cfg"], item["seed"], manifest=manifest,
        )
        fp = record["query_composite_fingerprint"]
        fingerprint_by_label[label] = fp
        labels_by_fingerprint.setdefault(fp, []).append(label)
    collisions = {fp: labels for fp, labels in labels_by_fingerprint.items() if len(labels) > 1}
    if collisions:
        raise ValueError(
            f"{len(collisions)} distinct realized query composite identity group(s) were each "
            f"produced by more than one (config, seed) draw for sample {sample_id!r} -- seed/config "
            f"uniqueness does not guarantee mask uniqueness; colliding groups: "
            f"{list(collisions.values())[:5]}"
        )
    return {
        "n_items": len(items),
        "n_unique_realized_masks": len(labels_by_fingerprint),
        "fingerprint_by_label": fingerprint_by_label,
    }


def verify_realized_seed_uniqueness(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    sample_id: str, masking_cfg: dict, seeds: Iterable[int], *, manifest: dict | None = None,
) -> dict:
    """Convenience wrapper over verify_realized_pool_uniqueness for the
    common single-masking_cfg case (a real, positive check that a seed
    pool -- e.g. one stratum's unique_masks_per_stratum range from
    mask_schedule.build_stratified_training_seed_bank -- realizes
    genuinely distinct masks, not merely distinct seed integers)."""
    seeds = [int(s) for s in seeds]
    items = [{"label": seed, "masking_cfg": masking_cfg, "seed": seed} for seed in seeds]
    pool_result = verify_realized_pool_uniqueness(coords3d, slice_ids, obs_names, sample_id, items, manifest=manifest)
    return {
        "n_seeds": pool_result["n_items"],
        "n_unique_realized_masks": pool_result["n_unique_realized_masks"],
        "fingerprint_by_seed": {int(label): fp for label, fp in pool_result["fingerprint_by_label"].items()},
    }


def realized_query_composite_ids(sample_id: str, records: list[dict]) -> set[str]:
    """Every composite query identity across a list of realized mask
    records (each with a real or realized `query_obs_names` list, e.g.
    from `mask_bank.py`'s explicit validation/test records or this
    module's own `realize_seed_and_fingerprint` output) for ONE sample
    -- the base set `verify_no_cross_split_query_leakage` is built on.
    Rejects duplicate barcodes within any single record's query list
    (via sorted_composite_query_fingerprint's own check)."""
    ids: set[str] = set()
    for record in records:
        sorted_composite_query_fingerprint(sample_id, record["query_obs_names"])  # duplicate check, result unused
        ids.update(composite_spot_id(sample_id, b) for b in record["query_obs_names"])
    return ids


def verify_no_cross_split_query_leakage(realized_composite_ids_by_split: dict[str, set[str]]) -> dict:
    """Fail-closed pairwise-disjointness check across splits (e.g.
    "train" vs "validation" vs "test") for one sample's realized query
    composite identities. A training draw's query spots must never
    coincide with a held-out mask's query spots for the same sample, or
    the model would be evaluated on exactly what it was trained to
    reconstruct -- a leak the sample-level train/validation/test split
    (dataset_manifest.py) cannot see, since it operates WITHIN one
    sample shared across splits (e.g. a training sample that also has
    its own in-sample validation/test masks for early-stopping
    diagnostics)."""
    splits = sorted(realized_composite_ids_by_split)
    overlaps: dict[tuple[str, str], list[str]] = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            shared = realized_composite_ids_by_split[a] & realized_composite_ids_by_split[b]
            if shared:
                overlaps[(a, b)] = sorted(shared)
    if overlaps:
        preview = {f"{a}/{b}": ids[:5] for (a, b), ids in overlaps.items()}
        raise ValueError(f"query composite identity leakage across splits: {preview}")
    return {
        "splits": splits,
        "n_ids_by_split": {s: len(realized_composite_ids_by_split[s]) for s in splits},
    }


def verify_no_duplicate_masks_within_split(sample_id: str, records: list[dict]) -> dict:
    """Fail-closed: within ONE split's records (e.g. all "validation"
    records for one sample), no two DIFFERENT records may realize the
    IDENTICAL query composite fingerprint. 12th Codex re-audit of commit
    1bb66d6, finding #7 (CONFIRMED): the prior cross-split leakage check
    only ever compared DIFFERENT splits against each other -- it never
    checked whether two records inside the SAME split (e.g. validation
    mask #2 and validation mask #5) happened to realize the exact same
    mask, which would silently halve the split's real effective sample
    size while still being counted as two independent evaluation
    draws."""
    fingerprint_by_index: dict[int, str] = {}
    indices_by_fingerprint: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        fp = sorted_composite_query_fingerprint(sample_id, record["query_obs_names"])
        fingerprint_by_index[i] = fp
        indices_by_fingerprint.setdefault(fp, []).append(i)
    duplicates = {fp: idxs for fp, idxs in indices_by_fingerprint.items() if len(idxs) > 1}
    if duplicates:
        raise ValueError(
            f"{sample_id}: {len(duplicates)} duplicate realized mask(s) found within one split -- "
            f"record indices {list(duplicates.values())[:5]} share an identical query composite "
            "fingerprint"
        )
    return {"n_records": len(records), "n_unique_masks": len(indices_by_fingerprint)}


def build_collision_free_training_schedule(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str], sample_id: str,
    strata: list[dict], n_items: int, base_seed: int, reserved_query_composite_ids: set[str],
    *, manifest: dict | None = None, max_attempts_per_item: int = 1000,
    spatial_adjacency: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict:
    """Deterministically build a training schedule whose realized masks
    are GUARANTEED, by construction, to (a) never query a RESERVED
    composite identity -- validation/test query spots, reserved FIRST
    by the caller (typically via `realized_query_composite_ids` over a
    `mask_schedule.build_stratified_mask_bank`'s validation+test
    records) -- and (b) never repeat an already-accepted training
    mask's realized query fingerprint. 12th Codex re-audit of commit
    1bb66d6, finding #5 (CONFIRMED): the previous round's tests proved
    `verify_realized_pool_uniqueness` can DETECT a collision, but
    nothing in this package could actually PRODUCE a collision-free
    schedule other than a test searching thousands of seeds for a lucky
    one -- production needs deterministic avoidance, not luck.

    Round-robins across strata (mirroring
    `mask_schedule.build_stratified_training_seed_bank`'s own coverage
    discipline: item i uses stratum `i % n_strata`). For each item,
    tries candidate seeds starting from a deterministic per-stratum
    counter (using the same `_STRATUM_SEED_STRIDE` offset convention
    `mask_schedule.py` already uses, so seeds from this function never
    collide with that module's own seed ranges), advancing by 1 on
    every rejected attempt (ONLY an `EmptyMaskRealizationError` --
    context/query genuinely empty for this seed -- a reserved-identity
    collision, or a duplicate; any OTHER exception, e.g. a manifest
    mismatch, propagates immediately rather than burning through
    `max_attempts_per_item` retries on an error no seed could fix --
    13th Codex re-audit finding #8, CONFIRMED), up to
    `max_attempts_per_item` -- raises (fail-closed, never silently
    accepts a colliding mask) if no valid seed is found within budget
    for any item.

    The returned schedule records `reserved_composite_ids_fingerprint`
    -- a SHA256 of the sorted `reserved_query_composite_ids` SET itself,
    not merely its count (13th Codex re-audit finding #3, CONFIRMED: a
    count alone cannot detect the reserved set CHANGING while staying
    the same size) -- so `validate_collision_free_training_schedule`
    can later confirm the schedule was built to avoid the SAME reserved
    identities a caller is currently trying to protect, not merely a
    same-sized different set."""
    n_items = int(n_items)
    base_seed = int(base_seed)
    if n_items < 1:
        raise ValueError("training schedule requires at least one item")
    if not strata:
        raise ValueError("strata must be a non-empty list")
    stratum_names = [s.get("name") for s in strata]
    if len(set(stratum_names)) != len(strata) or any(name is None for name in stratum_names):
        raise ValueError("every stratum must have a unique, non-null 'name'")
    n_strata = len(strata)
    if n_items < n_strata:
        raise ValueError(
            f"n_items ({n_items}) must be >= the number of strata ({n_strata}) to guarantee every "
            "stratum is covered at least once"
        )
    masking_cfg_by_stratum = {
        s["name"]: prepare_masking_cfg_for_sample(stratum_to_masking_cfg(s), coords3d, slice_ids)
        for s in strata
    }

    accepted_items: list[dict] = []
    accepted_fingerprints: set[str] = set()
    local_seed_by_stratum: dict[str, int] = {name: 0 for name in stratum_names}
    for i in range(n_items):
        stratum_index = i % n_strata
        stratum_name = stratum_names[stratum_index]
        masking_cfg = masking_cfg_by_stratum[stratum_name]
        accepted = None
        tried_seeds: list[int] = []
        for _ in range(max_attempts_per_item):
            local_seed = local_seed_by_stratum[stratum_name]
            seed = base_seed + local_seed + stratum_index * _STRATUM_SEED_STRIDE
            local_seed_by_stratum[stratum_name] += 1
            tried_seeds.append(seed)
            try:
                record = realize_seed_and_fingerprint(
                    coords3d, slice_ids, obs_names, sample_id, masking_cfg, seed, manifest=manifest,
                    spatial_adjacency=spatial_adjacency,
                )
            except (EmptyMaskRealizationError, EmptyBoundaryError):
                # Empty context/query, or a mask that removes an entire
                # disconnected fragment and has no observed boundary: both
                # are seed-specific invalid realizations, so try the next
                # deterministic seed. Structural errors still propagate.
                continue
            fp = record["query_composite_fingerprint"]
            query_composite_ids = {composite_spot_id(sample_id, b) for b in record["query_obs_names"]}
            if query_composite_ids & reserved_query_composite_ids:
                continue
            if fp in accepted_fingerprints:
                continue
            accepted = {"stratum": stratum_name, "seed": seed, "query_composite_fingerprint": fp}
            break
        if accepted is None:
            raise ValueError(
                f"{sample_id}: could not find a collision-free training mask for item {i} (stratum "
                f"{stratum_name!r}) within {max_attempts_per_item} attempts -- tried seeds "
                f"{tried_seeds[:5]}{'...' if len(tried_seeds) > 5 else ''}"
            )
        accepted_fingerprints.add(accepted["query_composite_fingerprint"])
        accepted_items.append(accepted)

    return {
        "version": _SCHEDULE_VERSION,
        "kind": "collision_free_training_schedule",
        "sample_id": sample_id,
        "n_items": n_items,
        "base_seed": base_seed,
        "strata": stratum_names,
        "strata_fingerprint": strata_fingerprint(strata),
        "n_reserved_composite_ids": len(reserved_query_composite_ids),
        "reserved_composite_ids_fingerprint": _composite_id_set_fingerprint(reserved_query_composite_ids),
        "items": [{"stratum": it["stratum"], "seed": it["seed"]} for it in accepted_items],
        "realized_query_composite_fingerprints": [it["query_composite_fingerprint"] for it in accepted_items],
    }


def _manifest_fingerprint(manifest: dict) -> str:
    return sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _mask_bank_records_fingerprint(records: list[dict]) -> str:
    return sha256(json.dumps(records, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _composite_id_set_fingerprint(composite_ids: Iterable[str]) -> str:
    """Order-independent SHA256 of a SET of composite spot identities --
    used both for a training schedule's `reserved_query_composite_ids`
    (13th Codex re-audit finding #3) and available for any other
    identity-set binding a caller needs."""
    return sha256("\n".join(sorted(str(c) for c in composite_ids)).encode("utf-8")).hexdigest()


def _observation_order_fingerprint(obs_names: Iterable[str]) -> str:
    """SHA256 of the obs_names sequence IN ORDER -- order matters here
    (unlike the composite-id SET fingerprints above), since
    `realize_seed_and_fingerprint`'s `names = np.asarray([str(x) for x
    in obs_names])` indexes by position, so a report is only meaningful
    for the exact ordered observation sequence it was computed against
    (13th Codex re-audit finding #2, CONFIRMED: a prior report bound
    only manifest/spatial/strata fingerprints, never the ordered
    identity sequence realization itself depends on)."""
    return sha256("\n".join(str(b) for b in obs_names).encode("utf-8")).hexdigest()


def _input_fingerprints(
    manifest: dict, coords3d: np.ndarray, slice_ids: np.ndarray, strata: list[dict], obs_names: Iterable[str],
) -> dict:
    """SHA256 fingerprints binding a report to the exact inputs it was
    built from -- 12th Codex re-audit of commit 1bb66d6, finding #6
    (CONFIRMED: a prior report recorded only summary counts, so a stale
    report file could later be silently accepted for different
    underlying data), extended by 13th Codex re-audit finding #2
    (CONFIRMED: the manifest/spatial/strata fingerprints alone say
    nothing about the ORDERED observation sequence a report's realized
    masks actually depend on). A consumer (Step 8's preflight gate, or
    the trainer itself) is expected to recompute these same fingerprints
    from its own live manifest/coords/strata/obs_names and refuse to
    trust a report whose fingerprints don't match -- this module only
    RECORDS them, it does not itself re-verify a loaded report against
    live data (that check belongs to whichever caller has the live data
    to compare against)."""
    return {
        "manifest_fingerprint": _manifest_fingerprint(manifest),
        "spatial_fingerprint": mask_bank.spatial_fingerprint(coords3d, slice_ids),
        "strata_fingerprint": strata_fingerprint(strata),
        "observation_order_fingerprint": _observation_order_fingerprint(obs_names),
    }


def validate_collision_free_training_schedule(
    schedule: dict, coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str], sample_id: str,
    strata: list[dict], reserved_query_composite_ids: set[str], *, manifest: dict | None = None,
    spatial_adjacency: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict:
    """Fail-closed re-verification of a (typically LOADED-from-disk)
    `build_collision_free_training_schedule` output against LIVE data
    (13th Codex re-audit of commit 65611c7, finding #3, CONFIRMED: no
    function existed to re-verify a persisted schedule still matched the
    generation it accompanies -- a report could reference an on-disk
    schedule file whose CONTENT nothing ever re-checked).

    Checks, in order: schema version and kind; `sample_id` match; strata
    identity AND `strata_fingerprint` match; the reserved-set
    fingerprint (a SHA256 of the actual SET, not merely a count --
    finding #3's other half: a same-SIZED but DIFFERENT reserved set
    would pass a count-only check) match; internal item-count
    consistency; then, the strongest check, RE-REALIZES every stored
    (stratum, seed) item from scratch and confirms its freshly-realized
    query composite fingerprint matches the one on record, in order,
    with no reserved-identity collisions and no internal duplicates.
    This is the only way to actually prove a stored schedule still
    matches what `build_collision_free_training_schedule` would produce
    against this exact live data today, rather than trusting the file's
    own self-reported summary fields."""
    if schedule.get("version") != _SCHEDULE_VERSION:
        raise ValueError(
            f"schedule version {schedule.get('version')!r} != expected {_SCHEDULE_VERSION!r} -- "
            "regenerate the schedule"
        )
    if schedule.get("kind") != "collision_free_training_schedule":
        raise ValueError(f"schedule kind {schedule.get('kind')!r} is not 'collision_free_training_schedule'")
    if schedule.get("sample_id") != sample_id:
        raise ValueError(f"schedule was built for sample {schedule.get('sample_id')!r}, not {sample_id!r}")
    stratum_names = [s.get("name") for s in strata]
    if schedule.get("strata") != stratum_names:
        raise ValueError(f"schedule strata {schedule.get('strata')!r} != live strata {stratum_names!r}")
    if schedule.get("strata_fingerprint") != strata_fingerprint(strata):
        raise ValueError("schedule strata_fingerprint does not match the live strata")
    live_reserved_fp = _composite_id_set_fingerprint(reserved_query_composite_ids)
    if schedule.get("reserved_composite_ids_fingerprint") != live_reserved_fp:
        raise ValueError(
            "schedule reserved_composite_ids_fingerprint does not match the live reserved set -- the "
            "validation/test mask bank this schedule was built to avoid colliding with has changed"
        )
    items = schedule.get("items", [])
    fingerprints = schedule.get("realized_query_composite_fingerprints", [])
    if len(items) != len(fingerprints):
        raise ValueError("schedule items/realized_query_composite_fingerprints length mismatch")
    if len(items) != schedule.get("n_items"):
        raise ValueError("schedule items length does not match its own recorded n_items")

    masking_cfg_by_stratum = {
        s["name"]: prepare_masking_cfg_for_sample(stratum_to_masking_cfg(s), coords3d, slice_ids)
        for s in strata
    }
    seen_fingerprints: set[str] = set()
    for i, (item, recorded_fp) in enumerate(zip(items, fingerprints)):
        masking_cfg = masking_cfg_by_stratum[item["stratum"]]
        record = realize_seed_and_fingerprint(
            coords3d, slice_ids, obs_names, sample_id, masking_cfg, item["seed"], manifest=manifest,
            spatial_adjacency=spatial_adjacency,
        )
        fresh_fp = record["query_composite_fingerprint"]
        if fresh_fp != recorded_fp:
            raise ValueError(
                f"schedule item {i} (stratum {item['stratum']!r}, seed {item['seed']}) re-realizes to "
                f"fingerprint {fresh_fp!r}, but the schedule recorded {recorded_fp!r} -- the schedule "
                "no longer matches what it would produce against this live data"
            )
        query_composite_ids = {composite_spot_id(sample_id, b) for b in record["query_obs_names"]}
        if query_composite_ids & reserved_query_composite_ids:
            raise ValueError(f"schedule item {i} queries a reserved composite identity on re-realization")
        if fresh_fp in seen_fingerprints:
            raise ValueError(f"schedule item {i} duplicates an earlier item's realized query fingerprint")
        seen_fingerprints.add(fresh_fp)
    return {"n_items": len(items), "passed": True}


def build_training_sample_mask_report(
    manifest: dict, sample_id: str, coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    strata: list[dict], training_schedule: dict, reserved_query_composite_ids: set[str], *,
    same_sample_diagnostic_mask_bank: dict | None = None,
    spatial_adjacency: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> dict:
    """The primary mask-fingerprint report for a TRAINING sample
    (`dataset_manifest.py`'s `train_sample_ids`). 12th Codex re-audit of
    commit 1bb66d6, finding #7/#8 (CONFIRMED): under this project's
    patient-disjoint SAMPLE-level split, a training sample's real
    evaluation happens on ENTIRELY DIFFERENT held-out samples, not on
    its own masks -- `mask_bank.py`/`mask_schedule.py`'s per-sample
    "validation"/"test" split_counts concept, when realized on a
    TRAINING sample, is at most a SECONDARY same-sample capacity/early-
    stopping diagnostic, never the primary evaluation protocol. This
    function reports that distinction explicitly rather than treating
    all three split types as normally coexisting on one sample.

    `training_schedule`: `build_collision_free_training_schedule`'s
    output for this sample.
    `reserved_query_composite_ids`: the LIVE reserved (validation/test)
    composite identity set for this sample -- required (14th Codex
    re-audit of commit 8d4e276, finding #2, CONFIRMED: this function
    used to re-realize masks only to prove POOL uniqueness among
    themselves; it never had a reserved set to check against at all, so
    it could not detect a training schedule that collides with the
    CURRENT reserved set, only with itself). Passed straight through to
    `validate_collision_free_training_schedule`, which is now called
    BEFORE this function can return -- re-realizing every stored
    `(stratum, seed)` item from scratch and confirming: its fresh query
    composite fingerprint matches the one on record; the schedule's own
    `reserved_composite_ids_fingerprint` matches
    `reserved_query_composite_ids`'s live fingerprint (not merely a
    stale count); and no two items collide. A caller can no longer get
    `passed: true` out of this function while handing it a schedule that
    is stale, tampered, or built against a reserved set that has since
    changed.
    `same_sample_diagnostic_mask_bank` (optional): a
    `mask_schedule.build_stratified_mask_bank`'s validation/test
    records realized on THIS SAME sample -- reported under a clearly
    separate `"same_sample_capacity_diagnostic"` section, cross-checked
    for leakage against the primary training schedule, but never
    conflated with primary held-out evaluation.

    13th Codex re-audit of commit 65611c7, finding #4 (CONFIRMED): this
    now REQUIRES `manifest["samples"][sample_id]["split"] == "train"` --
    a prior version accepted any sample_id the training_schedule itself
    claimed, never cross-checking it against the manifest's own role
    assignment."""
    if sample_id not in manifest["samples"]:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    manifest_split = manifest["samples"][sample_id].get("split")
    if manifest_split != "train":
        raise ValueError(
            f"{sample_id}: dataset manifest assigns split {manifest_split!r}, not 'train' -- refusing "
            "to build a training-sample mask report for a sample the manifest does not consider a "
            "training sample"
        )
    if sample_id != training_schedule["sample_id"]:
        raise ValueError(
            f"training_schedule was built for sample {training_schedule['sample_id']!r}, not {sample_id!r}"
        )

    # Re-realizes every stored item, confirms it matches live data AND
    # the LIVE reserved set (not just internal pool uniqueness), and
    # confirms no duplicates -- raises (fail-closed) on any mismatch.
    validate_collision_free_training_schedule(
        training_schedule, coords3d, slice_ids, obs_names, sample_id, strata, reserved_query_composite_ids,
        manifest=manifest, spatial_adjacency=spatial_adjacency,
    )

    masking_cfg_by_stratum = {
        s["name"]: prepare_masking_cfg_for_sample(stratum_to_masking_cfg(s), coords3d, slice_ids)
        for s in strata
    }
    train_query_ids: set[str] = set()
    for it in training_schedule["items"]:
        record = realize_seed_and_fingerprint(
            coords3d, slice_ids, obs_names, sample_id, masking_cfg_by_stratum[it["stratum"]], it["seed"],
            manifest=manifest, spatial_adjacency=spatial_adjacency,
        )
        train_query_ids.update(composite_spot_id(sample_id, b) for b in record["query_obs_names"])

    report = {
        "version": _REPORT_VERSION,
        "kind": "training_sample_mask_report",
        "sample_id": sample_id,
        "role": "train",
        "input_fingerprints": _input_fingerprints(manifest, coords3d, slice_ids, strata, obs_names),
        "training_schedule_content_fingerprint": _mask_bank_records_fingerprint(training_schedule["items"]),
        "realized_query_composite_fingerprints_fingerprint": _composite_id_set_fingerprint(
            training_schedule["realized_query_composite_fingerprints"]
        ),
        "reserved_composite_ids_fingerprint": _composite_id_set_fingerprint(reserved_query_composite_ids),
        "schedule_validated_against_live_data_and_reserved_set": True,
        "primary": {
            "n_items": len(training_schedule["items"]),
            # Guaranteed == n_items: validate_collision_free_training_schedule above
            # already raised if any two items realized the same fingerprint.
            "n_unique_masks": len(training_schedule["items"]),
        },
    }

    if same_sample_diagnostic_mask_bank is not None:
        diagnostic_ids_by_split: dict[str, set[str]] = {}
        diagnostic_n_records: dict[str, int] = {}
        diagnostic_content_fingerprint_by_split: dict[str, str] = {}
        for split in ("validation", "test"):
            records = [r for r in same_sample_diagnostic_mask_bank["records"] if r["split"] == split]
            if not records:
                continue
            for record in records:
                validate_realized_barcodes_against_manifest(manifest, sample_id, record["context_obs_names"])
                validate_realized_barcodes_against_manifest(manifest, sample_id, record["query_obs_names"])
            verify_no_duplicate_masks_within_split(sample_id, records)
            diagnostic_ids_by_split[split] = realized_query_composite_ids(sample_id, records)
            diagnostic_n_records[split] = len(records)
            diagnostic_content_fingerprint_by_split[split] = _mask_bank_records_fingerprint(records)
        leakage_result = verify_no_cross_split_query_leakage({"train": train_query_ids, **diagnostic_ids_by_split})
        report["same_sample_capacity_diagnostic"] = {
            "note": (
                "SECONDARY diagnostic only -- masks realized on the SAME sample as the primary "
                "training schedule above, not an independent held-out sample. Never substitute this "
                "for cross-sample evaluation on dataset_manifest.py's validation_sample_ids/"
                "test_sample_ids."
            ),
            "n_records_by_split": diagnostic_n_records,
            "n_composite_query_ids_by_split": leakage_result["n_ids_by_split"],
            "content_fingerprint_by_split": diagnostic_content_fingerprint_by_split,
        }

    report["passed"] = True
    return report


def build_held_out_sample_mask_report(
    manifest: dict, sample_id: str, coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    strata: list[dict], split: str, stratified_mask_bank: dict, *,
    expected_split_counts: dict[str, int], expected_split_seeds: dict[str, int],
) -> dict:
    """The primary mask-fingerprint report for a VALIDATION or TEST
    sample (`dataset_manifest.py`'s `validation_sample_ids`/
    `test_sample_ids` -- an entirely DIFFERENT sample from any training
    sample under the patient-disjoint sample-level split). Only
    `split`'s own records are the primary evaluation protocol here.
    Fails closed if `stratified_mask_bank` contains ANY record for a
    DIFFERENT split -- a real methodological error under this project's
    split discipline: a held-out sample must never carry
    training-labeled masks, and a validation sample must never carry
    test-labeled masks or vice versa (12th Codex re-audit finding #7/#8).

    `expected_split_counts`/`expected_split_seeds`: the RESOLVED
    EXPERIMENT CONFIGURATION's own `{"validation": n, "test": m}`-shaped
    dicts (e.g. `evaluation.n_validation_masks`/`n_test_masks` and
    `validation_seed`/`test_seed` from a real config, matching
    `mask_schedule.build_stratified_mask_bank`'s own `split_counts`/
    `split_seeds` parameters) -- REQUIRED, and never read off the
    supplied `stratified_mask_bank` itself (14th Codex re-audit of
    commit 8d4e276, finding #3, CONFIRMED: a prior version compared the
    bank's `strata_fingerprint` against a fingerprint computed FROM the
    bank's own recorded `split_counts`/`split_seeds` -- self-referential,
    and could never catch a bank built with the WRONG counts/seeds for
    this experiment in the first place).

    13th Codex re-audit of commit 65611c7 (CONFIRMED):
    - finding #4: now REQUIRES `manifest["samples"][sample_id]["split"]
      == split` -- a prior version accepted any sample_id/split pair the
      caller claimed, never cross-checking against the manifest's own
      role assignment; also now REJECTS an empty `records` list (a prior
      version let a held-out mask bank with zero records for this split
      silently "pass" with `n_records=0`).
    - finding #5: `stratified_mask_bank`'s own recorded
      `dataset_fingerprint`/`spatial_fingerprint`/`mask_generation_version`
      are validated against the LIVE obs_names/coords3d/slice_ids
      (mirroring `mask_schedule.load_stratified_mask_bank`'s own
      staleness checks, but for an in-memory bank a caller already has
      rather than one re-read from disk) -- a prior version trusted the
      bank's records without ever confirming the bank itself was built
      from the same underlying data.

    14th Codex re-audit of commit 8d4e276, finding #3 (CONFIRMED): the
    prior per-stratum check only required AT LEAST ONE record per
    stratum, and the total-count check alone could not detect a
    misdistributed bank (e.g. 3 records in stratum A + 1 in stratum B
    passing a "4 total, 2 expected per stratum" check). Fixed:
    - every stratum must have EXACTLY `expected_split_counts[split]`
      records for this split -- checked independently per stratum, not
      only in aggregate.
    - each stratum's record `index` values must be EXACTLY
      `{0, ..., expected_split_counts[split] - 1}` -- no gaps, no
      duplicates, no out-of-range indices.
    - each record's `seed` must match the EXACT deterministic seed
      `mask_schedule.build_stratified_mask_bank`/`mask_bank.build_mask_bank`
      would have assigned it: `expected_split_seeds[split] +
      stratum_index * _STRATUM_SEED_STRIDE + record_index`.
    - every record is RE-REALIZED from live coordinates (via
      `realize_seed_and_fingerprint` at its own expected seed) and its
      fresh `context_obs_names`/`query_obs_names` are compared for EXACT
      list equality against what's stored -- not merely re-deriving a
      fingerprint from the STORED barcodes (which would trivially always
      match itself), but proving the stored barcodes are what live data
      actually produces."""
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    if sample_id not in manifest["samples"]:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    manifest_split = manifest["samples"][sample_id].get("split")
    if manifest_split != split:
        raise ValueError(
            f"{sample_id}: dataset manifest assigns split {manifest_split!r}, not the requested "
            f"{split!r} -- refusing to build a held-out report under the wrong sample role"
        )
    if split not in expected_split_counts:
        raise ValueError(f"expected_split_counts is missing an entry for split {split!r}")
    if split not in expected_split_seeds:
        raise ValueError(f"expected_split_seeds is missing an entry for split {split!r}")
    if not strata:
        raise ValueError("strata must be a non-empty list")

    obs_names_list = [str(b) for b in obs_names]
    expected_dataset_fp = mask_bank.dataset_fingerprint(np.asarray(obs_names_list))
    if stratified_mask_bank.get("dataset_fingerprint") != expected_dataset_fp:
        raise ValueError(
            f"{sample_id}: stratified_mask_bank's dataset_fingerprint does not match the live "
            "obs_names -- this mask bank was built for different observation data"
        )
    expected_spatial_fp = mask_bank.spatial_fingerprint(coords3d, slice_ids)
    if stratified_mask_bank.get("spatial_fingerprint") != expected_spatial_fp:
        raise ValueError(
            f"{sample_id}: stratified_mask_bank's spatial_fingerprint does not match the live "
            "coordinates/slice IDs -- this mask bank was built for different spatial data"
        )

    bank_split_counts = stratified_mask_bank.get("split_counts")
    if bank_split_counts != dict(expected_split_counts):
        raise ValueError(
            f"{sample_id}: stratified_mask_bank's split_counts {bank_split_counts!r} does not match "
            f"the resolved experiment's expected_split_counts {dict(expected_split_counts)!r}"
        )
    bank_split_seeds = stratified_mask_bank.get("split_seeds")
    if bank_split_seeds != dict(expected_split_seeds):
        raise ValueError(
            f"{sample_id}: stratified_mask_bank's split_seeds {bank_split_seeds!r} does not match "
            f"the resolved experiment's expected_split_seeds {dict(expected_split_seeds)!r}"
        )
    expected_strata_fp = strata_fingerprint(strata, expected_split_counts, expected_split_seeds)
    if stratified_mask_bank.get("strata_fingerprint") != expected_strata_fp:
        raise ValueError(
            f"{sample_id}: stratified_mask_bank's strata_fingerprint does not match the live strata "
            "under the resolved experiment's expected split_counts/split_seeds"
        )
    if stratified_mask_bank.get("mask_generation_version") != _MASK_GENERATION_VERSION:
        raise ValueError(
            f"{sample_id}: stratified_mask_bank was built with mask-generation algorithm version "
            f"{stratified_mask_bank.get('mask_generation_version')!r}, expected "
            f"{_MASK_GENERATION_VERSION!r} -- regenerate the mask bank"
        )

    other_splits = {r["split"] for r in stratified_mask_bank["records"]} - {split}
    if other_splits:
        raise ValueError(
            f"{sample_id}: this is a {split!r} sample, but its mask bank contains record(s) for "
            f"other split(s) too: {sorted(other_splits)} -- a held-out sample must never carry "
            "masks from a different split"
        )
    records = [r for r in stratified_mask_bank["records"] if r["split"] == split]
    if not records:
        raise ValueError(
            f"{sample_id}: no {split!r} record(s) found in stratified_mask_bank -- an empty held-out "
            "mask bank must never silently pass as a valid evaluation report"
        )

    expected_count_per_stratum = int(expected_split_counts[split])
    if expected_count_per_stratum <= 0:
        raise ValueError(f"expected_split_counts[{split!r}] must be positive, got {expected_count_per_stratum}")
    base_seed = int(expected_split_seeds[split])
    masking_cfg_by_stratum = {s["name"]: stratum_to_masking_cfg(s) for s in strata}

    for stratum_index, stratum in enumerate(strata):
        stratum_name = stratum.get("name")
        stratum_records = [r for r in records if r.get("stratum") == stratum_name]
        if len(stratum_records) != expected_count_per_stratum:
            raise ValueError(
                f"{sample_id}: stratum {stratum_name!r} has {len(stratum_records)} {split!r} "
                f"record(s), expected EXACTLY {expected_count_per_stratum}"
            )
        indices = sorted(int(r["index"]) for r in stratum_records)
        expected_indices = list(range(expected_count_per_stratum))
        if indices != expected_indices:
            raise ValueError(
                f"{sample_id}: stratum {stratum_name!r} {split!r} records have index set {indices}, "
                f"expected exactly {expected_indices}"
            )
        stratum_base_seed = base_seed + stratum_index * _STRATUM_SEED_STRIDE
        record_by_index = {int(r["index"]): r for r in stratum_records}
        for i in range(expected_count_per_stratum):
            record = record_by_index[i]
            seed_attempt = int(record.get("seed_attempt", -1))
            if seed_attempt < 0:
                raise ValueError(
                    f"{sample_id}: stratum {stratum_name!r} {split!r} record index {i} "
                    "has no valid non-negative seed_attempt -- the bank predates deterministic "
                    "boundary-valid mask regeneration"
                )
            expected_seed = stratum_base_seed + i + seed_attempt * expected_count_per_stratum
            if int(record["seed"]) != expected_seed:
                raise ValueError(
                    f"{sample_id}: stratum {stratum_name!r} {split!r} record index {i} has seed "
                    f"{record['seed']}, expected {expected_seed} (base_seed={base_seed}, "
                    f"stratum_index={stratum_index}, seed_attempt={seed_attempt}, "
                    f"_STRATUM_SEED_STRIDE={_STRATUM_SEED_STRIDE})"
                )
            validate_realized_barcodes_against_manifest(manifest, sample_id, record["context_obs_names"])
            validate_realized_barcodes_against_manifest(manifest, sample_id, record["query_obs_names"])
            context_set = set(record["context_obs_names"])
            query_set = set(record["query_obs_names"])
            if not context_set:
                raise ValueError(f"{sample_id}: {split} record index {i} (stratum {stratum_name!r}) has an empty context set")
            if not query_set:
                raise ValueError(f"{sample_id}: {split} record index {i} (stratum {stratum_name!r}) has an empty query set")
            if context_set & query_set:
                raise ValueError(
                    f"{sample_id}: {split} record index {i} (stratum {stratum_name!r}) has overlapping "
                    "context/query barcodes"
                )
            fresh = realize_seed_and_fingerprint(
                coords3d, slice_ids, obs_names, sample_id, masking_cfg_by_stratum[stratum_name], expected_seed,
                manifest=manifest,
            )
            if fresh["context_obs_names"] != list(record["context_obs_names"]):
                raise ValueError(
                    f"{sample_id}: stratum {stratum_name!r} {split!r} record index {i} (seed "
                    f"{expected_seed}) re-realizes to a DIFFERENT context set than stored -- the stored "
                    "record no longer matches what live coordinates/masking would produce"
                )
            if fresh["query_obs_names"] != list(record["query_obs_names"]):
                raise ValueError(
                    f"{sample_id}: stratum {stratum_name!r} {split!r} record index {i} (seed "
                    f"{expected_seed}) re-realizes to a DIFFERENT query set than stored -- the stored "
                    "record no longer matches what live coordinates/masking would produce"
                )

    dedup_result = verify_no_duplicate_masks_within_split(sample_id, records)

    return {
        "version": _REPORT_VERSION,
        "kind": "held_out_sample_mask_report",
        "sample_id": sample_id,
        "role": split,
        "input_fingerprints": _input_fingerprints(manifest, coords3d, slice_ids, strata, obs_names_list),
        "mask_bank_content_fingerprint": _mask_bank_records_fingerprint(records),
        "expected_split_counts": dict(expected_split_counts),
        "expected_split_seeds": dict(expected_split_seeds),
        "primary": {
            "n_records": dedup_result["n_records"],
            "n_unique_masks": dedup_result["n_unique_masks"],
        },
        "passed": True,
    }


def save_collision_free_training_schedule(schedule: dict, path: str | Path) -> Path:
    """Atomic write, mirroring save_dataset_manifest/save_mask_bank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(schedule, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_collision_free_training_schedule(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_mask_fingerprint_report(report: dict, path: str | Path) -> Path:
    """Atomic write, mirroring save_dataset_manifest/save_mask_bank."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_mask_fingerprint_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
