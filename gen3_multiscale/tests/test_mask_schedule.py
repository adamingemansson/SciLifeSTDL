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
import json
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


def test_build_stratified_mask_bank_retries_a_boundaryless_mask_deterministically(monkeypatch):
    """A non-empty mask can still remove an entire disconnected tissue
    fragment.  The fixed held-out schedule must replace it deterministically
    rather than letting training crash at its first validation pass."""
    query_cluster = np.stack(
        [np.arange(7, dtype=np.float64) * 0.01, np.zeros(7)], axis=1,
    )
    observed_cluster = np.stack(
        [10.0 + np.arange(7, dtype=np.float64) * 0.01, np.zeros(7)], axis=1,
    )
    coords2d = np.concatenate([query_cluster, observed_cluster], axis=0)
    coords3d = np.concatenate([coords2d, np.zeros((coords2d.shape[0], 1))], axis=1)
    slice_ids = np.full(coords3d.shape[0], "S0", dtype=object)
    obs_names = np.asarray([f"spot{i}" for i in range(coords3d.shape[0])])

    def fake_make_split(_coords3d, _slice_ids, _masking_cfg, seed):
        query = np.zeros(coords3d.shape[0], dtype=bool)
        if seed == 123:
            query[:7] = True  # whole disconnected fragment: invalid
        else:
            query[0] = True   # surrounded by observed spots: valid
        return ~query, query

    monkeypatch.setattr(mask_bank, "make_split", fake_make_split)
    bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names,
        [{"name": "only", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"}],
        split_counts={"validation": 1}, split_seeds={"validation": 123},
    )
    assert len(bank["records"]) == 1
    assert bank["records"][0]["seed"] == 124
    assert bank["records"][0]["seed_attempt"] == 1
    assert bank["records"][0]["query_obs_names"] == ["spot0"]


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
# training masks." Further fixed (4th Codex re-audit of commit 0fd46e5):
# the old unique_mask_count parameter name was misleading (it meant
# "unique seeds PER STRATUM", not a global total), round-robin coverage
# was only actually guaranteed when n_items >= n_strata but that wasn't
# enforced, and the schedule didn't fingerprint coordinates/slice IDs at
# all (the same barcodes with altered coordinates would silently reuse a
# schedule whose generated holes would actually differ).
# ---------------------------------------------------------------------------
def _training_seed_slide():
    coords3d, slice_ids, obs_names = _synthetic_slide(n=4)
    return coords3d, slice_ids, obs_names


def test_build_stratified_training_seed_bank_round_robins_across_strata():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank = build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    strata_sequence = [item["stratum"] for item in bank["items"]]
    assert strata_sequence == ["small_compact", "large_irregular", "small_compact", "large_irregular", "small_compact", "large_irregular"]


def test_build_stratified_training_seed_bank_seeds_never_collide_across_strata():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank = build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=10, base_seed=0, strata=_STRATA)
    seeds_by_stratum: dict[str, set[int]] = {}
    for item in bank["items"]:
        seeds_by_stratum.setdefault(item["stratum"], set()).add(item["seed"])
    assert seeds_by_stratum["small_compact"].isdisjoint(seeds_by_stratum["large_irregular"])


def test_build_stratified_training_seed_bank_is_deterministic():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank_a = build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=8, base_seed=5, strata=_STRATA)
    bank_b = build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=8, base_seed=5, strata=_STRATA)
    assert bank_a == bank_b


def test_build_stratified_training_seed_bank_allows_unique_masks_per_stratum_larger_than_n_items():
    """Corrected semantics (5th Codex re-audit): unique_masks_per_stratum
    larger than n_items is legitimate -- it just means this particular
    call doesn't exhaust the full per-stratum seed pool, not an error.
    The old validation rejected this based on an incorrect mental model
    (a global item budget) rather than the actual per-stratum occurrence
    semantics."""
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA, unique_masks_per_stratum=99,
    )
    assert len(bank["items"]) == 4


def test_build_stratified_training_seed_bank_rejects_unique_masks_per_stratum_at_or_above_the_seed_stride():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    with pytest.raises(ValueError, match="unique_masks_per_stratum"):
        build_stratified_training_seed_bank(
            coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA,
            unique_masks_per_stratum=1_000_000,
        )


def test_build_stratified_training_seed_bank_realizes_the_full_promised_unique_seed_pool_per_stratum():
    """The exact regression test the audit specified: 4 strata, 64 unique
    masks per stratum, >= 256 items -- must realize exactly 64 unique
    seeds per stratum and exactly 256 unique (stratum, seed) combinations
    (reproducing the audit's own worked example, which the old formula
    failed: it only ever realized 16 of the 64 promised seeds per
    stratum, confirmed by direct computation before this fix)."""
    coords3d, slice_ids, obs_names = _synthetic_slide(n=4)
    four_strata = [
        {"name": f"stratum_{i}", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"}
        for i in range(4)
    ]
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=256, base_seed=0, strata=four_strata, unique_masks_per_stratum=64,
    )
    seeds_by_stratum: dict[str, set[int]] = {}
    for item in bank["items"]:
        seeds_by_stratum.setdefault(item["stratum"], set()).add(item["seed"])
    for stratum_name, seeds in seeds_by_stratum.items():
        assert len(seeds) == 64, f"{stratum_name} realized {len(seeds)} unique seeds, expected 64"
    all_combinations = {(item["stratum"], item["seed"]) for item in bank["items"]}
    assert len(all_combinations) == 256


def test_build_stratified_training_seed_bank_reports_realized_unique_seeds_per_stratum():
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): unique_masks_per_stratum is a CAP on the seed pool,
    not a promise every seed in it is actually drawn -- the bank must
    report what was actually realized so a caller can check their
    coverage intent without redoing the n_items/n_strata arithmetic."""
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA, unique_masks_per_stratum=99,
    )
    # 2 strata, 4 items -> 2 occurrences per stratum -> only 2 of the 99
    # promised seeds are ever actually realized per stratum.
    assert bank["realized_unique_seeds_per_stratum"] == {"small_compact": 2, "large_irregular": 2}


def test_build_stratified_training_seed_bank_require_full_seed_pool_rejects_an_under_provisioned_schedule():
    """Regression test for the 6th audit's core complaint: nothing
    enforced that n_items was large enough to realize the full promised
    unique_masks_per_stratum pool. require_full_seed_pool=True turns
    that into a fail-closed precondition for callers that need the
    guarantee, without changing the permissive default (preserving the
    5th round's deliberate "oversized pools are legitimate" decision)."""
    coords3d, slice_ids, obs_names = _training_seed_slide()
    with pytest.raises(ValueError, match="did not realize the full"):
        build_stratified_training_seed_bank(
            coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA,
            unique_masks_per_stratum=99, require_full_seed_pool=True,
        )


def test_build_stratified_training_seed_bank_require_full_seed_pool_accepts_a_sufficiently_provisioned_schedule():
    coords3d, slice_ids, obs_names = _synthetic_slide(n=4)
    four_strata = [
        {"name": f"stratum_{i}", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"}
        for i in range(4)
    ]
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=256, base_seed=0, strata=four_strata,
        unique_masks_per_stratum=64, require_full_seed_pool=True,
    )
    assert all(count == 64 for count in bank["realized_unique_seeds_per_stratum"].values())


def test_build_stratified_training_seed_bank_require_full_seed_pool_with_default_unique_masks_per_stratum_does_not_always_raise():
    """Regression test for a real, confirmed gap (7th Codex re-audit of
    commit 2782ff0): "Strict mode combined with the default
    unique_masks_per_stratum=n_items is mathematically impossible when
    there is more than one stratum." Confirmed -- the permissive
    default (n_items) can never be exhausted once n_strata > 1, so
    require_full_seed_pool=True previously raised UNCONDITIONALLY
    whenever a caller left unique_masks_per_stratum unset, which is a
    guaranteed-to-fail footgun, not a real precondition. Strict mode now
    defaults unique_masks_per_stratum to the naturally achievable
    ceiling (n_items // n_strata) instead, so the default trivially
    satisfies its own guarantee."""
    coords3d, slice_ids, obs_names = _training_seed_slide()  # 2 strata
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=10, base_seed=0, strata=_STRATA, require_full_seed_pool=True,
    )
    assert bank["unique_masks_per_stratum"] == 5  # 10 items // 2 strata
    assert all(count == 5 for count in bank["realized_unique_seeds_per_stratum"].values())


def test_ensure_stratified_training_seed_bank_backfills_a_schema_field_missing_from_an_older_on_disk_bank(tmp_path):
    """Regression test for a real, confirmed gap (7th Codex re-audit of
    commit 2782ff0): this module's output schema gained a new derived
    field (realized_unique_seeds_per_stratum) without a
    _MASK_GENERATION_VERSION bump, since the actual (stratum, seed)
    GENERATION algorithm didn't change -- correctly, per that field's own
    documented purpose. But that meant a validated reuse of an on-disk
    bank written before this field existed would return the raw,
    field-missing dict as-is. ensure_stratified_training_seed_bank now
    returns the freshly-recomputed (schema-current) dict whenever a
    reuse is validated, not the raw on-disk JSON, so this can never
    surface as a missing key after a legitimate reuse."""
    coords3d, slice_ids, obs_names = _training_seed_slide()
    path = tmp_path / "training_seeds.json"
    first, _ = ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert "realized_unique_seeds_per_stratum" in first

    # Simulate an older on-disk artifact written before this field existed.
    stale = json.loads(path.read_text())
    del stale["realized_unique_seeds_per_stratum"]
    path.write_text(json.dumps(stale, indent=2, sort_keys=True))

    reused, _ = ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert "realized_unique_seeds_per_stratum" in reused
    assert reused["items"] == first["items"]  # still the exact same schedule, just schema-complete

    # Real, confirmed gap (8th Codex re-audit of commit 7b5c267):
    # returning the fresh dict to THIS caller fixed what THIS caller
    # sees, but left the stale JSON sitting on disk for any OTHER
    # reader. The on-disk file itself must now be self-healed too.
    on_disk = json.loads(path.read_text())
    assert "realized_unique_seeds_per_stratum" in on_disk
    assert on_disk == build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)


def test_build_stratified_training_seed_bank_with_one_unique_mask_per_stratum_still_produces_distinct_combinations():
    """Regression test locking in the exact scenario the audit gave:
    unique_masks_per_stratum=1 must NOT mean "1 unique mask overall" --
    each stratum gets its own offset seed range, so n_items distinct
    (stratum, seed) combinations still result."""
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA, unique_masks_per_stratum=1,
    )
    combinations = {(item["stratum"], item["seed"]) for item in bank["items"]}
    assert len(combinations) == 2  # 2 strata x 1 unique seed each -- 4 items round-robin onto these 2 combos


def test_build_stratified_training_seed_bank_requires_at_least_one_item_per_stratum_for_coverage():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    with pytest.raises(ValueError, match="n_items"):
        build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=1, base_seed=0, strata=_STRATA)


def test_build_stratified_training_seed_bank_fingerprints_coordinates():
    coords3d, slice_ids, obs_names = _training_seed_slide()
    bank_a = build_stratified_training_seed_bank(coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA)
    moved_coords3d = coords3d.copy()
    moved_coords3d[:, 0] += 1000.0  # same barcodes, different spatial layout
    bank_b = build_stratified_training_seed_bank(moved_coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_STRATA)
    assert bank_a["dataset_fingerprint"] == bank_b["dataset_fingerprint"]  # same barcodes
    assert bank_a["spatial_fingerprint"] != bank_b["spatial_fingerprint"]  # but different coordinates


def test_ensure_stratified_training_seed_bank_builds_then_reuses(tmp_path):
    coords3d, slice_ids, obs_names = _training_seed_slide()
    path = tmp_path / "training_seeds.json"
    first, _ = ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert path.is_file()
    second, _ = ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    assert first == second


def test_ensure_stratified_training_seed_bank_rejects_an_altered_schedule_at_the_same_path(tmp_path):
    coords3d, slice_ids, obs_names = _training_seed_slide()
    path = tmp_path / "training_seeds.json"
    ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    with pytest.raises(ValueError, match="does not match"):
        ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=12, base_seed=0, strata=_STRATA)


def test_ensure_stratified_training_seed_bank_rejects_changed_coordinates_at_the_same_path(tmp_path):
    coords3d, slice_ids, obs_names = _training_seed_slide()
    path = tmp_path / "training_seeds.json"
    ensure_stratified_training_seed_bank(path, coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
    moved_coords3d = coords3d.copy()
    moved_coords3d[:, 0] += 1000.0
    with pytest.raises(ValueError, match="does not match"):
        ensure_stratified_training_seed_bank(path, moved_coords3d, slice_ids, obs_names, n_items=6, base_seed=0, strata=_STRATA)
