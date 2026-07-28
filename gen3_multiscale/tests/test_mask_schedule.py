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
from gen3_multiscale.data.mask_schedule import (
    build_stratified_mask_bank, build_stratified_training_seed_bank, ensure_stratified_mask_bank,
    ensure_stratified_training_seed_bank, load_stratified_mask_bank, save_stratified_mask_bank,
    strata_fingerprint, stratum_records, stratum_to_masking_cfg,
)

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


# ---------------------------------------------------------------------------
# strata_fingerprint -- fixes a real, confirmed gap (3rd Codex re-audit of
# commit ca7cf53): "it has no combined fingerprint covering the ordered
# strata definition", only per-stratum fingerprints an unordered dict could
# reshuffle without changing.
# ---------------------------------------------------------------------------
def test_strata_fingerprint_is_deterministic_and_content_sensitive():
    assert strata_fingerprint(_STRATA) == strata_fingerprint(_STRATA)
    changed = [dict(_STRATA[0], radius_range=[9.0, 10.0]), _STRATA[1]]
    assert strata_fingerprint(_STRATA) != strata_fingerprint(changed)


def test_strata_fingerprint_is_sensitive_to_stratum_order():
    reordered = [_STRATA[1], _STRATA[0]]
    assert strata_fingerprint(_STRATA) != strata_fingerprint(reordered)


def test_build_stratified_mask_bank_includes_a_strata_fingerprint():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, _STRATA)
    assert bank["strata_fingerprint"] == strata_fingerprint(_STRATA, bank["split_counts"], bank["split_seeds"])


# ---------------------------------------------------------------------------
# save/load/ensure_stratified_mask_bank -- fixes a real, confirmed gap (3rd
# Codex re-audit): "nothing outside its tests calls it" / "no ensure/load/
# save path" / "version 1 incompatible with the existing version-2
# load_mask_bank() staleness validation" (a DIFFERENT, dedicated trio for
# the genuinely different stratified schema, not a forced fit into
# mask_bank.load_mask_bank's single-masking_fingerprint contract).
# ---------------------------------------------------------------------------
def test_save_and_load_stratified_mask_bank_round_trips(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, _STRATA)
    path = save_stratified_mask_bank(bank, tmp_path / "bank.json")
    assert path.is_file()
    loaded = load_stratified_mask_bank(path, obs_names, coords3d=coords3d, slice_ids=slice_ids, strata=_STRATA)
    assert loaded == bank


def test_load_stratified_mask_bank_rejects_a_different_observation_set(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, _STRATA)
    path = save_stratified_mask_bank(bank, tmp_path / "bank.json")
    with pytest.raises(ValueError, match="different observation set"):
        load_stratified_mask_bank(path, [f"other_{n}" for n in obs_names])


def test_load_stratified_mask_bank_rejects_changed_strata(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    bank = build_stratified_mask_bank(coords3d, slice_ids, obs_names, _STRATA)
    path = save_stratified_mask_bank(bank, tmp_path / "bank.json")
    changed_strata = [dict(_STRATA[0], radius_range=[9.0, 10.0]), _STRATA[1]]
    with pytest.raises(ValueError, match="different strata"):
        load_stratified_mask_bank(path, obs_names, coords3d=coords3d, slice_ids=slice_ids, strata=changed_strata)


def test_ensure_stratified_mask_bank_builds_then_reuses(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    path = tmp_path / "bank.json"
    first = ensure_stratified_mask_bank(path, coords3d, slice_ids, obs_names, _STRATA)
    assert path.is_file()
    second = ensure_stratified_mask_bank(path, coords3d, slice_ids, obs_names, _STRATA)
    assert first == second  # reused, not rebuilt with new random records


def test_ensure_stratified_mask_bank_rejects_a_stale_bank_at_the_same_path(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    path = tmp_path / "bank.json"
    ensure_stratified_mask_bank(path, coords3d, slice_ids, obs_names, _STRATA)
    changed_strata = [dict(_STRATA[0], radius_range=[9.0, 10.0]), _STRATA[1]]
    with pytest.raises(ValueError, match="different strata"):
        ensure_stratified_mask_bank(path, coords3d, slice_ids, obs_names, changed_strata)


# ---------------------------------------------------------------------------
# build_stratified_training_seed_bank / ensure_stratified_training_seed_bank
# -- fixes a real, confirmed gap (3rd Codex re-audit of commit ca7cf53):
# "its default schedule contains validation and test masks only, not
# training masks."
# ---------------------------------------------------------------------------
def test_build_stratified_training_seed_bank_round_robins_across_strata():
    _obs_names = [f"o{i}" for i in range(4)]
    bank = build_stratified_training_seed_bank(_obs_names, n_items=6, base_seed=0, strata=_STRATA)
    strata_sequence = [item["stratum"] for item in bank["items"]]
    assert strata_sequence == ["small_compact", "large_irregular", "small_compact", "large_irregular", "small_compact", "large_irregular"]


def test_build_stratified_training_seed_bank_seeds_never_collide_across_strata():
    _obs_names = [f"o{i}" for i in range(4)]
    bank = build_stratified_training_seed_bank(_obs_names, n_items=10, base_seed=0, strata=_STRATA)
    seeds_by_stratum: dict[str, set[int]] = {}
    for item in bank["items"]:
        seeds_by_stratum.setdefault(item["stratum"], set()).add(item["seed"])
    assert seeds_by_stratum["small_compact"].isdisjoint(seeds_by_stratum["large_irregular"])


def test_build_stratified_training_seed_bank_is_deterministic():
    _obs_names = [f"o{i}" for i in range(4)]
    bank_a = build_stratified_training_seed_bank(_obs_names, n_items=8, base_seed=5, strata=_STRATA)
    bank_b = build_stratified_training_seed_bank(_obs_names, n_items=8, base_seed=5, strata=_STRATA)
    assert bank_a == bank_b


def test_build_stratified_training_seed_bank_rejects_an_out_of_range_unique_mask_count():
    _obs_names = [f"o{i}" for i in range(4)]
    with pytest.raises(ValueError, match="unique_mask_count"):
        build_stratified_training_seed_bank(_obs_names, n_items=4, base_seed=0, strata=_STRATA, unique_mask_count=99)


def test_ensure_stratified_training_seed_bank_builds_then_reuses(tmp_path):
    _obs_names = [f"o{i}" for i in range(4)]
    path = tmp_path / "training_seeds.json"
    first, _ = ensure_stratified_training_seed_bank(path, _obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert path.is_file()
    second, _ = ensure_stratified_training_seed_bank(path, _obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert first == second


def test_ensure_stratified_training_seed_bank_rejects_an_altered_schedule_at_the_same_path(tmp_path):
    _obs_names = [f"o{i}" for i in range(4)]
    path = tmp_path / "training_seeds.json"
    ensure_stratified_training_seed_bank(path, _obs_names, n_items=6, base_seed=0, strata=_STRATA)
    with pytest.raises(ValueError, match="does not match"):
        ensure_stratified_training_seed_bank(path, _obs_names, n_items=12, base_seed=0, strata=_STRATA)
