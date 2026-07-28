"""Realized-mask fingerprinting -- Step 3 of the real Gen3 data
builder/trainer (Adam's explicit 9-step implementation order,
CONTRACT.md section 30, item 3: "Fingerprint sorted realized query
identities"; sharpened by the 10th Codex re-audit of commit 9592d9e's
closing instruction: "implement Step 3 using realized composite query
identities rather than seeds").

`mask_bank.py`/`mask_schedule.py`'s training seed banks deliberately
store only a (stratum, seed) SCHEDULE, not every realized context/query
barcode list, to avoid multi-gigabyte JSON (mask_bank.build_training_seed_bank's
own docstring). Their `unique_mask_count`/`unique_masks_per_stratum`
fields count DISTINCT SEED VALUES and implicitly assume distinct seeds
always realize distinct masks. `mask_bank.make_split` is a deterministic
function of (data, masking_cfg, seed), but it is NOT proven injective in
seed: two different seeds are not guaranteed to produce two different
actual context/query splits (a real, repeatedly-flagged gap across this
project's audit history). This module closes that gap by REALIZING a
candidate seed and fingerprinting the ACTUAL resulting query set by real
composite (sample_id, spot) identity (`dataset_manifest.composite_spot_id`
-- the true globally-unique spot identity, since raw Visium barcodes are
not globally unique across samples), rather than trusting the seed
integer as a stand-in for mask identity.

Also provides a real, positive cross-split leakage check: for one
sample, a training draw's realized query spots must never coincide with
a held-out (validation/test) mask's realized query spots for that same
sample, or the model would be trained to reconstruct exactly what it is
later evaluated on. (Cross-SAMPLE leakage is already structurally
prevented by dataset_manifest's patient-disjoint sample-level split --
this module does not re-check that; it checks WITHIN one sample's own
realized masks, which the sample-level split cannot see.)
"""
from __future__ import annotations

from hashlib import sha256
from typing import Iterable

import numpy as np

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.dataset_manifest import composite_spot_id


def sorted_composite_query_fingerprint(sample_id: str, query_obs_names: Iterable[str]) -> str:
    """SHA256 of the SORTED set of composite (sample_id, spot) identities
    for a realized query set -- order-independent (two masks with the
    same query spots in a different iteration order must fingerprint
    identically), and namespaced by sample_id via composite_spot_id so
    fingerprints from different samples can never collide by coincidence
    even if their raw barcodes do."""
    composite_ids = sorted(composite_spot_id(sample_id, b) for b in query_obs_names)
    return sha256("\n".join(composite_ids).encode("utf-8")).hexdigest()


def realize_seed_and_fingerprint(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    sample_id: str, masking_cfg: dict, seed: int,
) -> dict:
    """Realize ONE (masking_cfg, seed) draw exactly as
    `mask_bank.build_mask_bank` does internally (same `make_split` call,
    same context capping), and fingerprint its REALIZED query set by
    real composite spot identity. Returns a record with the same
    `context_obs_names`/`query_obs_names` shape as a `mask_bank.py`
    record, plus `query_composite_fingerprint`."""
    names = np.asarray([str(x) for x in obs_names])
    context, query = mask_bank.make_split(coords3d, slice_ids, masking_cfg, seed)
    max_context = mask_bank._cfg_get(masking_cfg, "max_context_points", None)
    context_selection = str(mask_bank._cfg_get(masking_cfg, "context_selection", "random"))
    context = mask_bank.cap_context_mask(
        context, max_context, seed, coords3d=coords3d, query_mask=query, selection=context_selection,
    )
    if not query.any() or not context.any():
        raise ValueError(f"mask seed {seed} produced an empty context or query")
    query_obs_names = names[query].tolist()
    return {
        "seed": int(seed),
        "context_obs_names": names[context].tolist(),
        "query_obs_names": query_obs_names,
        "query_composite_fingerprint": sorted_composite_query_fingerprint(sample_id, query_obs_names),
    }


def verify_realized_seed_uniqueness(
    coords3d: np.ndarray, slice_ids: np.ndarray, obs_names: Iterable[str],
    sample_id: str, masking_cfg: dict, seeds: Iterable[int],
) -> dict:
    """Fail-closed: realize every seed in `seeds` and confirm their
    REALIZED query composite fingerprints are pairwise distinct.
    Different seed VALUES are not proof of different realized masks --
    this is the direct check "using realized composite query identities
    rather than seeds" both the 9-step order and the 10th Codex
    re-audit's closing instruction specify. Intended to validate a
    `mask_schedule.build_stratified_training_seed_bank`-style pool's
    (e.g. one stratum's `unique_masks_per_stratum` seed range) implicit
    uniqueness promise against what actually gets realized, not merely
    against distinct integer seeds."""
    seeds = [int(s) for s in seeds]
    fingerprint_by_seed: dict[int, str] = {}
    seeds_by_fingerprint: dict[str, list[int]] = {}
    for seed in seeds:
        record = realize_seed_and_fingerprint(coords3d, slice_ids, obs_names, sample_id, masking_cfg, seed)
        fp = record["query_composite_fingerprint"]
        fingerprint_by_seed[seed] = fp
        seeds_by_fingerprint.setdefault(fp, []).append(seed)
    collisions = {fp: s for fp, s in seeds_by_fingerprint.items() if len(s) > 1}
    if collisions:
        raise ValueError(
            f"{len(collisions)} distinct realized query composite identity group(s) were each "
            f"produced by more than one seed for sample {sample_id!r} -- seed uniqueness does not "
            f"guarantee mask uniqueness; colliding seed groups: {list(collisions.values())[:5]}"
        )
    return {
        "n_seeds": len(seeds),
        "n_unique_realized_masks": len(seeds_by_fingerprint),
        "fingerprint_by_seed": fingerprint_by_seed,
    }


def realized_query_composite_ids(sample_id: str, records: list[dict]) -> set[str]:
    """Every composite query identity across a list of realized mask
    records (each with a real or realized `query_obs_names` list, e.g.
    from `mask_bank.py`'s explicit validation/test records or this
    module's own `realize_seed_and_fingerprint` output) for ONE sample
    -- the base set `verify_no_cross_split_query_leakage` is built on."""
    ids: set[str] = set()
    for record in records:
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
