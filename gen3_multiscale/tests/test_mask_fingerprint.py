"""Tests for realized-mask fingerprinting (Step 3 of the real Gen3 data
builder/trainer, CONTRACT.md section 33)."""
import numpy as np
import pytest

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.dataset_manifest import composite_spot_id
from gen3_multiscale.data.mask_fingerprint import (
    build_collision_free_training_schedule, build_held_out_sample_mask_report,
    build_training_sample_mask_report, load_collision_free_training_schedule, load_mask_fingerprint_report,
    realize_seed_and_fingerprint, realized_query_composite_ids, save_collision_free_training_schedule,
    save_mask_fingerprint_report, sorted_composite_query_fingerprint,
    validate_realized_barcodes_against_manifest, verify_no_cross_split_query_leakage,
    verify_no_duplicate_masks_within_split, verify_realized_pool_uniqueness, verify_realized_seed_uniqueness,
)
from gen3_multiscale.data.mask_schedule import build_stratified_mask_bank, stratum_to_masking_cfg

_STRATUM = {"name": "small_compact", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"}


def _synthetic_slide(n=12):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords3d = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    slice_ids = np.array(["slide_a"] * (n * n))
    obs_names = [f"spot_{i}" for i in range(n * n)]
    return coords3d, slice_ids, obs_names


def test_sorted_composite_query_fingerprint_is_order_independent_and_sample_namespaced():
    fp_forward = sorted_composite_query_fingerprint("S0", ["a", "b", "c"])
    fp_reversed = sorted_composite_query_fingerprint("S0", ["c", "b", "a"])
    assert fp_forward == fp_reversed  # order independent

    fp_other_sample = sorted_composite_query_fingerprint("S1", ["a", "b", "c"])
    assert fp_forward != fp_other_sample  # namespaced by sample_id, not just raw barcodes

    fp_different_content = sorted_composite_query_fingerprint("S0", ["a", "b", "d"])
    assert fp_forward != fp_different_content


def test_realize_seed_and_fingerprint_matches_a_direct_make_split_call():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    masking_cfg = stratum_to_masking_cfg(_STRATUM)
    seed = 42

    record = realize_seed_and_fingerprint(coords3d, slice_ids, obs_names, "S0", masking_cfg, seed)

    names = np.asarray([str(x) for x in obs_names])
    context, query = mask_bank.make_split(coords3d, slice_ids, masking_cfg, seed)
    context = mask_bank.cap_context_mask(context, None, seed, coords3d=coords3d, query_mask=query)
    assert record["context_obs_names"] == names[context].tolist()
    assert record["query_obs_names"] == names[query].tolist()
    assert record["query_composite_fingerprint"] == sorted_composite_query_fingerprint(
        "S0", names[query].tolist(),
    )


def test_verify_realized_seed_uniqueness_passes_for_genuinely_distinct_seeds():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    masking_cfg = stratum_to_masking_cfg(_STRATUM)
    seeds = list(range(10))

    result = verify_realized_seed_uniqueness(coords3d, slice_ids, obs_names, "S0", masking_cfg, seeds)
    assert result["n_seeds"] == 10
    assert result["n_unique_realized_masks"] == 10


def test_verify_realized_seed_uniqueness_raises_on_a_genuine_realized_collision():
    """The core regression this module exists for: two DIFFERENT seed
    values can realize the IDENTICAL query set (seed uniqueness is not
    proof of mask uniqueness). Engineered deterministically -- not
    flaky -- using hold_out_slice with two candidate slices: np.random's
    rng.choice over only two options means many different seeds
    coincidentally pick the SAME held-out slice, producing byte-identical
    masks. A dry run (never a hardcoded guess about the RNG) finds two
    such colliding seeds first."""
    n = 6
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    half = (n * n) // 2
    coords3d = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    slice_ids = np.array(["slide_a"] * half + ["slide_b"] * (n * n - half))
    obs_names = [f"spot_{i}" for i in range(n * n)]
    masking_cfg = {"strategy": "hold_out_slice"}

    choice_by_seed = {
        seed: np.random.default_rng(seed).choice(np.unique(slice_ids)) for seed in range(50)
    }
    seeds_by_choice: dict[str, list[int]] = {}
    for seed, choice in choice_by_seed.items():
        seeds_by_choice.setdefault(str(choice), []).append(seed)
    colliding_seeds = next(s for s in seeds_by_choice.values() if len(s) >= 2)[:2]

    with pytest.raises(ValueError, match="does not guarantee mask uniqueness"):
        verify_realized_seed_uniqueness(
            coords3d, slice_ids, obs_names, "S0", masking_cfg, colliding_seeds,
        )


def test_realized_query_composite_ids_and_cross_split_leakage_check():
    train_records = [{"query_obs_names": ["spot_1", "spot_2"]}]
    validation_records = [{"query_obs_names": ["spot_3", "spot_4"]}]

    train_ids = realized_query_composite_ids("S0", train_records)
    validation_ids = realized_query_composite_ids("S0", validation_records)
    result = verify_no_cross_split_query_leakage({"train": train_ids, "validation": validation_ids})
    assert result["splits"] == ["train", "validation"]
    assert result["n_ids_by_split"] == {"train": 2, "validation": 2}


def test_verify_no_cross_split_query_leakage_raises_on_a_real_overlap():
    train_ids = realized_query_composite_ids("S0", [{"query_obs_names": ["spot_1", "spot_2"]}])
    validation_ids = realized_query_composite_ids("S0", [{"query_obs_names": ["spot_2", "spot_3"]}])
    with pytest.raises(ValueError, match="leakage across splits"):
        verify_no_cross_split_query_leakage({"train": train_ids, "validation": validation_ids})


def test_realized_query_composite_ids_are_namespaced_so_different_samples_never_collide():
    """The exact reason composite identity (not raw barcode) is used:
    two DIFFERENT samples sharing a literal barcode string must never be
    treated as an overlap."""
    sample_a_ids = realized_query_composite_ids("SAMPLE_A", [{"query_obs_names": ["SPOT1"]}])
    sample_b_ids = realized_query_composite_ids("SAMPLE_B", [{"query_obs_names": ["SPOT1"]}])
    result = verify_no_cross_split_query_leakage({"a": sample_a_ids, "b": sample_b_ids})
    assert result["n_ids_by_split"] == {"a": 1, "b": 1}


def test_sorted_composite_query_fingerprint_rejects_duplicate_barcodes():
    """11th Codex re-audit of commit 9dab8fe, finding #3: a realized
    query list must be a SET of distinct spots -- a duplicate indicates
    a real upstream bug and must fail loudly, not silently collapse."""
    with pytest.raises(ValueError, match="duplicate barcode"):
        sorted_composite_query_fingerprint("S0", ["a", "b", "a"])


def test_realized_query_composite_ids_rejects_duplicate_within_a_record():
    with pytest.raises(ValueError, match="duplicate barcode"):
        realized_query_composite_ids("S0", [{"query_obs_names": ["spot_1", "spot_1"]}])


def test_validate_realized_barcodes_against_manifest_accepts_declared_barcodes():
    manifest = {"samples": {"S0": {"barcodes": ["spot_1", "spot_2", "spot_3"]}}}
    validate_realized_barcodes_against_manifest(manifest, "S0", ["spot_1", "spot_2"])  # no raise


def test_validate_realized_barcodes_against_manifest_rejects_an_undeclared_barcode():
    manifest = {"samples": {"S0": {"barcodes": ["spot_1", "spot_2"]}}}
    with pytest.raises(ValueError, match="never declared"):
        validate_realized_barcodes_against_manifest(manifest, "S0", ["spot_1", "NOT_DECLARED"])


def test_validate_realized_barcodes_against_manifest_rejects_an_unknown_sample():
    manifest = {"samples": {"S0": {"barcodes": ["spot_1"]}}}
    with pytest.raises(ValueError, match="is not a sample"):
        validate_realized_barcodes_against_manifest(manifest, "WRONG_SAMPLE", ["spot_1"])


def test_realize_seed_and_fingerprint_validates_against_a_supplied_manifest():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    masking_cfg = stratum_to_masking_cfg(_STRATUM)
    good_manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    realize_seed_and_fingerprint(coords3d, slice_ids, obs_names, "S0", masking_cfg, 0, manifest=good_manifest)  # no raise

    bad_manifest = {"samples": {"S0": {"barcodes": ["spot_0"]}}}  # missing almost every real barcode
    with pytest.raises(ValueError, match="never declared"):
        realize_seed_and_fingerprint(coords3d, slice_ids, obs_names, "S0", masking_cfg, 0, manifest=bad_manifest)


def test_verify_realized_pool_uniqueness_detects_a_cross_config_collision():
    """11th Codex re-audit finding #3 ("detect identical masks across
    different strata/configurations, not only seeds passed to one
    masking configuration"): two DIFFERENT masking_cfg dicts can realize
    an IDENTICAL mask for the SAME seed. Engineered deterministically:
    mask_bank.make_split's hold_out_slice branch never reads
    masking_cfg['params'] at all, so two configs differing only in an
    ignored params field realize byte-identical masks for the same
    seed."""
    n = 6
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    half = (n * n) // 2
    coords3d = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(np.float64)
    slice_ids = np.array(["slide_a"] * half + ["slide_b"] * (n * n - half))
    obs_names = [f"spot_{i}" for i in range(n * n)]
    cfg_a = {"strategy": "hold_out_slice", "params": {"tag": "config_A"}}
    cfg_b = {"strategy": "hold_out_slice", "params": {"tag": "config_B"}}
    items = [
        {"label": "stratum_a:0", "masking_cfg": cfg_a, "seed": 0},
        {"label": "stratum_b:0", "masking_cfg": cfg_b, "seed": 0},
    ]
    with pytest.raises(ValueError, match="does not guarantee mask uniqueness"):
        verify_realized_pool_uniqueness(coords3d, slice_ids, obs_names, "S0", items)


def test_verify_realized_pool_uniqueness_passes_for_genuinely_distinct_cross_config_draws():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    strata = [
        {"name": "small", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"},
        {"name": "large", "radius_range": [3.0, 4.0], "radius_unit": "coordinate", "shape": "mixed"},
    ]
    items = [
        {"label": f"{s['name']}:{seed}", "masking_cfg": stratum_to_masking_cfg(s), "seed": seed}
        for s in strata for seed in range(3)
    ]
    result = verify_realized_pool_uniqueness(coords3d, slice_ids, obs_names, "S0", items)
    assert result["n_items"] == 6
    assert result["n_unique_realized_masks"] == 6


_REPORT_STRATA = [
    {"name": "small", "radius_range": [1.0, 2.0], "radius_unit": "coordinate", "shape": "circle"},
    {"name": "large", "radius_range": [3.0, 4.0], "radius_unit": "coordinate", "shape": "mixed"},
]


def test_verify_no_duplicate_masks_within_split_passes_for_distinct_masks():
    records = [{"query_obs_names": ["a"]}, {"query_obs_names": ["b"]}]
    result = verify_no_duplicate_masks_within_split("S0", records)
    assert result == {"n_records": 2, "n_unique_masks": 2}


def test_verify_no_duplicate_masks_within_split_raises_on_a_real_duplicate():
    """12th Codex re-audit of commit 1bb66d6, finding #7: the prior
    cross-split leakage check never caught two records WITHIN the same
    split realizing an identical mask (here, the same query set in a
    different iteration order)."""
    records = [{"query_obs_names": ["a", "b"]}, {"query_obs_names": ["b", "a"]}]
    with pytest.raises(ValueError, match="duplicate realized mask"):
        verify_no_duplicate_masks_within_split("S0", records)


def _reserved_ids_from_mask_bank(sample_id: str, bank: dict) -> set[str]:
    records = [r for r in bank["records"] if r["split"] in ("validation", "test")]
    return realized_query_composite_ids(sample_id, records)


def test_build_collision_free_training_schedule_avoids_reserved_and_duplicate_masks():
    """The core regression this function exists for (12th Codex re-audit
    finding #5): on a SMALL grid with LARGE-radius strata -- exactly the
    configuration that made cross-split overlap common by chance alone
    in this file's earlier tests -- this function must succeed
    DETERMINISTICALLY, on the first call, with zero retries from the
    caller, never leaking into a reserved identity and never repeating
    an accepted mask."""
    coords3d, slice_ids, obs_names = _synthetic_slide()  # n=12, small -- collisions are common here by chance
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 1},
    )
    reserved_ids = _reserved_ids_from_mask_bank("S0", mask_bank_bank)

    schedule = build_collision_free_training_schedule(
        coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=6, base_seed=0,
        reserved_query_composite_ids=reserved_ids, manifest=manifest,
    )
    assert schedule["n_items"] == 6
    assert len(schedule["items"]) == 6
    fingerprints = schedule["realized_query_composite_fingerprints"]
    assert len(set(fingerprints)) == len(fingerprints)  # pairwise distinct, guaranteed by construction

    for item in schedule["items"]:
        cfg = next(stratum_to_masking_cfg(s) for s in _REPORT_STRATA if s["name"] == item["stratum"])
        record = realize_seed_and_fingerprint(coords3d, slice_ids, obs_names, "S0", cfg, item["seed"])
        query_ids = {composite_spot_id("S0", b) for b in record["query_obs_names"]}
        assert query_ids.isdisjoint(reserved_ids)


def test_build_collision_free_training_schedule_raises_when_impossible_within_budget():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    reserved_ids = {composite_spot_id("S0", b) for b in obs_names}  # reserve EVERY spot -- every draw must collide
    with pytest.raises(ValueError, match="could not find a collision-free training mask"):
        build_collision_free_training_schedule(
            coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=2, base_seed=0,
            reserved_query_composite_ids=reserved_ids, max_attempts_per_item=5,
        )


def test_save_and_load_collision_free_training_schedule_round_trips(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    schedule = build_collision_free_training_schedule(
        coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=4, base_seed=0,
        reserved_query_composite_ids=set(),
    )
    path = tmp_path / "schedule.json"
    save_collision_free_training_schedule(schedule, path)
    assert load_collision_free_training_schedule(path) == schedule


def test_build_training_sample_mask_report_end_to_end_and_persists(tmp_path):
    """The complete PRIMARY training-sample report: a collision-free
    schedule (deterministic, not a lucky search) reserved against the
    same sample's validation/test masks, processed and persisted to
    disk -- the artifact Step 8's preflight gate will require before
    training starts."""
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 1},
    )
    reserved_ids = _reserved_ids_from_mask_bank("S0", mask_bank_bank)
    training_schedule = build_collision_free_training_schedule(
        coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=6, base_seed=0,
        reserved_query_composite_ids=reserved_ids, manifest=manifest,
    )

    report = build_training_sample_mask_report(
        manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, training_schedule,
    )
    assert report["passed"] is True
    assert report["role"] == "train"
    assert report["primary"]["n_items"] == 6
    assert report["primary"]["n_unique_masks"] == 6
    assert "manifest_fingerprint" in report["input_fingerprints"]
    assert "same_sample_capacity_diagnostic" not in report

    path = tmp_path / "report.json"
    save_mask_fingerprint_report(report, path)
    assert load_mask_fingerprint_report(path) == report


def test_build_training_sample_mask_report_includes_a_clearly_labeled_same_sample_diagnostic():
    """12th Codex re-audit finding #8: same-sample validation/test masks
    must be reported under an explicitly separate section, never
    conflated with the primary training schedule.

    Uses only a "validation"-labeled diagnostic bank here (test:0):
    mask_bank.py's own validation/test draws are independently seeded
    and, on this file's small 12x12 grid, can coincidentally overlap
    EACH OTHER (a separate, already-documented characteristic of the
    underlying per-sample mask-drawing scheme, unrelated to what this
    test verifies -- the labeling/separation of the diagnostic section
    itself)."""
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 0},
    )
    reserved_ids = _reserved_ids_from_mask_bank("S0", mask_bank_bank)
    training_schedule = build_collision_free_training_schedule(
        coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=6, base_seed=0,
        reserved_query_composite_ids=reserved_ids, manifest=manifest,
    )

    report = build_training_sample_mask_report(
        manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, training_schedule,
        same_sample_diagnostic_mask_bank=mask_bank_bank,
    )
    diagnostic = report["same_sample_capacity_diagnostic"]
    assert diagnostic["n_records_by_split"] == {"validation": 2}  # 1 per stratum * 2 strata
    assert "SECONDARY diagnostic" in diagnostic["note"]


def test_build_training_sample_mask_report_raises_on_a_sample_id_mismatch():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    schedule = build_collision_free_training_schedule(
        coords3d, slice_ids, obs_names, "S0", _REPORT_STRATA, n_items=4, base_seed=0,
        reserved_query_composite_ids=set(),
    )
    with pytest.raises(ValueError, match="was built for sample"):
        build_training_sample_mask_report(
            manifest, "WRONG_SAMPLE", coords3d, slice_ids, obs_names, _REPORT_STRATA, schedule,
        )


def test_build_held_out_sample_mask_report_end_to_end():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 3, "test": 0},
    )
    report = build_held_out_sample_mask_report(
        manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, "validation", mask_bank_bank,
    )
    assert report["passed"] is True
    assert report["role"] == "validation"
    assert report["primary"]["n_records"] == 6  # 3 per stratum * 2 strata
    assert report["primary"]["n_unique_masks"] == 6


def test_build_held_out_sample_mask_report_raises_when_bank_contains_a_different_split():
    """12th Codex re-audit finding #7/#8: a held-out sample must never
    carry masks from a different split -- e.g. a validation sample's
    mask bank must not also contain test-labeled records."""
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 1},
    )
    with pytest.raises(ValueError, match="must never carry masks from a different split"):
        build_held_out_sample_mask_report(
            manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, "validation", mask_bank_bank,
        )


def test_build_held_out_sample_mask_report_raises_on_duplicate_masks_within_the_split():
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 2, "test": 0},
    )
    validation_records = [r for r in mask_bank_bank["records"] if r["split"] == "validation"]
    validation_records[1]["query_obs_names"] = list(validation_records[0]["query_obs_names"])
    validation_records[1]["context_obs_names"] = [
        b for b in obs_names if b not in validation_records[1]["query_obs_names"]
    ]
    with pytest.raises(ValueError, match="duplicate realized mask"):
        build_held_out_sample_mask_report(
            manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, "validation", mask_bank_bank,
        )


def test_report_input_fingerprints_change_when_the_manifest_changes():
    """12th Codex re-audit finding #6: reports must be bound to the
    exact inputs they were built from."""
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest_a = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    manifest_b = {"samples": {"S0": {"barcodes": list(obs_names)}, "extra_key": "changes the fingerprint"}}
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 0},
    )
    report_a = build_held_out_sample_mask_report(
        manifest_a, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, "validation", mask_bank_bank,
    )
    report_b = build_held_out_sample_mask_report(
        manifest_b, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, "validation", mask_bank_bank,
    )
    assert report_a["input_fingerprints"]["manifest_fingerprint"] != report_b["input_fingerprints"]["manifest_fingerprint"]
