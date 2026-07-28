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

from typing import Iterable

import numpy as np

from gen3_multiscale.data import mask_bank

# Keeps every stratum's seed range from ever colliding with another
# stratum's, for any split_seeds a caller supplies (default validation=
# 700_000/test=900_000 leaves ample room below this stride).
_STRATUM_SEED_STRIDE = 1_000_000


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

    all_records = []
    per_stratum_fingerprints = {}
    for i, stratum in enumerate(strata):
        stratum_name = stratum["name"]
        masking_cfg = stratum_to_masking_cfg(stratum)
        stratum_seeds = {split: int(seed) + i * _STRATUM_SEED_STRIDE for split, seed in split_seeds.items()}
        bank = mask_bank.build_mask_bank(
            coords3d, slice_ids, obs_names, masking_cfg,
            split_counts=split_counts, split_seeds=stratum_seeds,
        )
        for record in bank["records"]:
            record = dict(record)
            record["stratum"] = stratum_name
            all_records.append(record)
        per_stratum_fingerprints[stratum_name] = bank["masking_fingerprint"]

    names_arr = np.asarray([str(x) for x in obs_names])
    return {
        "version": 1,
        "kind": "stratified_mask_bank",
        "dataset_fingerprint": mask_bank.dataset_fingerprint(names_arr),
        "spatial_fingerprint": mask_bank.spatial_fingerprint(coords3d, slice_ids),
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
