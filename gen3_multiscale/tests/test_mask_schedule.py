"""Stratified mask-schedule generator -- fixes a real, confirmed bug (2nd
Codex re-audit of commit 547f51e): configs/architectureN.yaml's
masking.strata block had no consuming code, and the reused
mask_bank.make_split expects strategy/params, not a strata list --
calling it directly with the literal config would raise
ValueError("unknown masking strategy None"). Verified here against
synthetic coordinates (pure geometry, like Phase 2's boundary-extraction
tests -- no dependency on real loaded HEST-1k data) and against the
REAL masking.strata blocks in the four actual config files.
"""
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.mask_schedule import build_stratified_mask_bank, stratum_records, stratum_to_masking_cfg

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _synthetic_slide(n=12):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords3d = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    slice_ids = np.array(["slide_a"] * (n * n))
    obs_names = [f"spot_{i}" for i in range(n * n)]
    return coords3d, slice_ids, obs_names


_STRATA = [
    {"name": "small_compact", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"},
    {"name": "large_irregular", "radius_range": [3.0, 4.0], "radius_unit": "coordinate", "shape": "mixed"},
]


def test_stratum_to_masking_cfg_produces_a_real_random_dropout_patches_config():
    cfg = stratum_to_masking_cfg(_STRATA[0])
    assert cfg == {
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 1, "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"},
    }


def test_stratum_to_masking_cfg_raises_on_a_missing_required_field():
    with pytest.raises(ValueError, match="missing required fields"):
        stratum_to_masking_cfg({"name": "bad", "radius_range": [1.0, 2.0]})


def test_stratum_to_masking_cfg_is_directly_usable_by_the_real_reused_make_split():
    """The core regression proof: what the audit predicted would raise
    ValueError("unknown masking strategy None") now runs cleanly through
    the ACTUAL reused mask_bank.make_split, unmodified."""
    coords3d, slice_ids, _obs_names = _synthetic_slide()
    for stratum in _STRATA:
        masking_cfg = stratum_to_masking_cfg(stratum)
        context, query = mask_bank.make_split(coords3d, slice_ids, masking_cfg, seed=0)
        assert context.any() and query.any()
        assert not np.any(context & query)


def test_build_stratified_mask_bank_tags_every_record_with_its_stratum():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _STRATA, split_counts={"validation": 2, "test": 3},
    )
    assert bank["strata"] == ["small_compact", "large_irregular"]
    assert len(bank["records"]) == (2 + 3) * len(_STRATA)
    for record in bank["records"]:
        assert record["stratum"] in {"small_compact", "large_irregular"}


def test_build_stratified_mask_bank_never_reuses_a_seed_across_strata():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _STRATA, split_counts={"validation": 3, "test": 3},
    )
    seeds_by_stratum = {}
    for record in bank["records"]:
        seeds_by_stratum.setdefault(record["stratum"], set()).add(record["seed"])
    small_seeds, large_seeds = seeds_by_stratum["small_compact"], seeds_by_stratum["large_irregular"]
    assert small_seeds.isdisjoint(large_seeds)


def test_build_stratified_mask_bank_produces_different_masking_fingerprints_per_stratum():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, _STRATA)
    fingerprints = bank["per_stratum_masking_fingerprint"]
    assert fingerprints["small_compact"] != fingerprints["large_irregular"]


def test_build_stratified_mask_bank_rejects_duplicate_or_missing_stratum_names():
    with pytest.raises(ValueError, match="unique, non-null"):
        build_stratified_mask_bank(
            *_synthetic_slide(), strata=[_STRATA[0], _STRATA[0]],
        )


def test_stratum_records_filters_by_split_and_stratum():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _STRATA, split_counts={"validation": 2, "test": 3},
    )
    validation_small = stratum_records(bank, "validation", "small_compact")
    assert len(validation_small) == 2
    assert all(r["split"] == "validation" and r["stratum"] == "small_compact" for r in validation_small)
    assert [r["index"] for r in validation_small] == sorted(r["index"] for r in validation_small)


@pytest.mark.parametrize("name", ["architecture1", "architecture2", "architecture3", "architecture4"])
def test_the_real_config_files_masking_strata_are_directly_usable(name):
    """Ties this fix to the actual regression the audit found: the real
    configs/architectureN.yaml files' masking.strata blocks, loaded
    exactly as a real entrypoint would load them, must build a real
    stratified mask bank without error."""
    config = OmegaConf.to_container(OmegaConf.load(_CONFIG_DIR / f"{name}.yaml"), resolve=True)
    strata = config["masking"]["strata"]
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, strata, split_counts={"validation": 1, "test": 1})
    assert len(bank["records"]) == 2 * len(strata)
