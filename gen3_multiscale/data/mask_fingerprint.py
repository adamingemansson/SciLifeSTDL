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

`build_mask_fingerprint_report`/`save_mask_fingerprint_report`/
`load_mask_fingerprint_report` process a REAL, complete production
schedule (a `mask_schedule.build_stratified_training_seed_bank` +
`mask_schedule.build_stratified_mask_bank` pair, for one sample) and
persist a signed-off leakage/uniqueness report to disk -- the artifact
Step 8's preflight gate is meant to require before a real training run
starts (11th Codex re-audit finding #3: "process the complete
production schedule and persist a leakage report that the trainer must
verify before starting").
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.dataset_manifest import composite_spot_id
from gen3_multiscale.data.mask_schedule import stratum_to_masking_cfg

_REPORT_VERSION = 1


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
        raise ValueError(f"mask seed {seed} produced an empty context or query")
    context_obs_names = names[context].tolist()
    query_obs_names = names[query].tolist()
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


def build_mask_fingerprint_report(
    manifest: dict, sample_id: str, coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    strata: list[dict], stratified_training_seed_bank: dict, stratified_mask_bank: dict,
) -> dict:
    """Process the COMPLETE production schedule for one sample -- every
    unique (stratum, seed) pair the real training schedule actually
    contains, plus every real validation/test record -- and produce one
    signed-off report: realized-mask uniqueness across the whole pool
    (not just within one stratum), manifest/sample validation for every
    realized identity, and cross-split leakage rejection. 11th Codex
    re-audit finding #3 ("process the complete production schedule and
    persist a leakage report that the trainer must verify before
    starting").

    `stratified_training_seed_bank`: mask_schedule.build_stratified_training_seed_bank's
    output for this sample (its `items` list may legitimately repeat the
    same (stratum, seed) pair many times via round-robin cycling --
    deduplicated here before realizing, so a large `n_items` doesn't
    mean redundant work).
    `stratified_mask_bank`: mask_schedule.build_stratified_mask_bank's
    output for this sample (explicit validation/test records; never
    re-realized, since they already store the real context/query
    barcodes directly).

    Raises (fail-closed) on any uniqueness collision, manifest
    validation failure, or cross-split leakage; otherwise returns a
    JSON-serializable report with `"passed": True`."""
    masking_cfg_by_stratum = {s["name"]: stratum_to_masking_cfg(s) for s in strata}
    unique_train_pairs = sorted({
        (item["stratum"], int(item["seed"])) for item in stratified_training_seed_bank["items"]
    })
    train_items = [
        {"label": f"{stratum}:{seed}", "masking_cfg": masking_cfg_by_stratum[stratum], "seed": seed}
        for stratum, seed in unique_train_pairs
    ]
    pool_result = verify_realized_pool_uniqueness(
        coords3d, slice_ids, obs_names, sample_id, train_items, manifest=manifest,
    )
    train_query_ids: set[str] = set()
    for stratum, seed in unique_train_pairs:
        record = realize_seed_and_fingerprint(
            coords3d, slice_ids, obs_names, sample_id, masking_cfg_by_stratum[stratum], seed, manifest=manifest,
        )
        train_query_ids.update(composite_spot_id(sample_id, b) for b in record["query_obs_names"])

    ids_by_split = {"train": train_query_ids}
    n_records_by_split = {"train": len(unique_train_pairs)}
    for split in ("validation", "test"):
        records = [r for r in stratified_mask_bank["records"] if r["split"] == split]
        for record in records:
            validate_realized_barcodes_against_manifest(manifest, sample_id, record["context_obs_names"])
            validate_realized_barcodes_against_manifest(manifest, sample_id, record["query_obs_names"])
        ids_by_split[split] = realized_query_composite_ids(sample_id, records)
        n_records_by_split[split] = len(records)

    leakage_result = verify_no_cross_split_query_leakage(ids_by_split)

    return {
        "version": _REPORT_VERSION,
        "sample_id": sample_id,
        "n_train_pairs_realized": len(unique_train_pairs),
        "n_unique_train_masks": pool_result["n_unique_realized_masks"],
        "n_records_by_split": n_records_by_split,
        "n_composite_query_ids_by_split": leakage_result["n_ids_by_split"],
        "passed": True,
    }


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
