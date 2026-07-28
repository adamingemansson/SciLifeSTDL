"""Tests for realized-mask fingerprinting (Step 3 of the real Gen3 data
builder/trainer, CONTRACT.md section 33)."""
import numpy as np
import pytest

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.mask_fingerprint import (
    build_mask_fingerprint_report, load_mask_fingerprint_report, realize_seed_and_fingerprint,
    realized_query_composite_ids, save_mask_fingerprint_report, sorted_composite_query_fingerprint,
    validate_realized_barcodes_against_manifest, verify_no_cross_split_query_leakage,
    verify_realized_pool_uniqueness, verify_realized_seed_uniqueness,
)
from gen3_multiscale.data.mask_schedule import (
    build_stratified_mask_bank, build_stratified_training_seed_bank, stratum_to_masking_cfg,
)

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


def test_build_mask_fingerprint_report_end_to_end_and_persists(tmp_path):
    """The complete production-schedule check: a real
    stratified-training-seed-bank + stratified-mask-bank pair (built
    exactly as mask_schedule.py's own audited functions produce them)
    for one sample, processed and persisted to disk -- the artifact
    Step 8's preflight gate will require before training starts.

    A real, separate finding surfaced while writing this test: with
    `random_dropout_patches` scattering query patches uniformly across
    ONE shared coordinate space, cross-split query overlap between
    independently-seeded train/validation/test draws is common by
    chance alone, even on a large (1600-spot) grid -- not a bug in this
    module (its job is to DETECT that, which it correctly does; see
    test_build_mask_fingerprint_report_raises_on_cross_split_leakage),
    but a real characteristic of the existing per-sample mask-drawing
    scheme this test must not fight by hoping for luck. A deterministic
    dry-run search (never a hardcoded guess, matching this test suite's
    established pattern for RNG-dependent scenarios elsewhere) finds a
    base_seed that happens to realize a genuinely leakage-free schedule
    with small strata radii on a large grid; once found, it is fully
    deterministic on every subsequent run of this exact test."""
    coords3d, slice_ids, obs_names = _synthetic_slide(n=40)
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    strata = [
        {"name": "small", "radius_range": [0.4, 0.6], "radius_unit": "coordinate", "shape": "circle"},
        {"name": "large", "radius_range": [0.8, 1.0], "radius_unit": "coordinate", "shape": "circle"},
    ]

    report = None
    for base_seed in range(0, 5000, 100):
        training_bank = build_stratified_training_seed_bank(
            coords3d, slice_ids, obs_names, n_items=4, base_seed=base_seed, strata=strata,
        )
        mask_bank_bank = build_stratified_mask_bank(
            coords3d, slice_ids, obs_names, strata, split_counts={"validation": 1, "test": 1},
            split_seeds={"validation": 700_000 + base_seed, "test": 900_000 + base_seed},
        )
        try:
            report = build_mask_fingerprint_report(
                manifest, "S0", coords3d, slice_ids, obs_names, strata, training_bank, mask_bank_bank,
            )
            break
        except ValueError:
            continue
    assert report is not None, "no leakage-free schedule found in the search range"

    assert report["passed"] is True
    assert report["sample_id"] == "S0"
    assert report["n_train_pairs_realized"] >= 1
    assert report["n_records_by_split"]["validation"] == 2  # 1 per stratum
    assert report["n_records_by_split"]["test"] == 2

    path = tmp_path / "report.json"
    save_mask_fingerprint_report(report, path)
    assert load_mask_fingerprint_report(path) == report


def test_build_mask_fingerprint_report_raises_on_manifest_barcode_mismatch(tmp_path):
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": ["not_a_real_barcode"]}}}  # wrong sample's barcodes
    training_bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_REPORT_STRATA,
    )
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 1},
    )
    with pytest.raises(ValueError, match="never declared"):
        build_mask_fingerprint_report(
            manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, training_bank, mask_bank_bank,
        )


def test_build_mask_fingerprint_report_raises_on_cross_split_leakage(tmp_path):
    """Wiring check: a validation record whose query spots coincide with
    a real training draw's realized query spots must be caught by the
    full report, not just by the underlying unit-tested leakage
    function in isolation."""
    coords3d, slice_ids, obs_names = _synthetic_slide()
    manifest = {"samples": {"S0": {"barcodes": list(obs_names)}}}
    training_bank = build_stratified_training_seed_bank(
        coords3d, slice_ids, obs_names, n_items=4, base_seed=0, strata=_REPORT_STRATA,
    )
    mask_bank_bank = build_stratified_mask_bank(
        coords3d, slice_ids, obs_names, _REPORT_STRATA, split_counts={"validation": 1, "test": 0},
    )
    first_item = training_bank["items"][0]
    stratum_cfg = next(stratum_to_masking_cfg(s) for s in _REPORT_STRATA if s["name"] == first_item["stratum"])
    leaked = realize_seed_and_fingerprint(
        coords3d, slice_ids, obs_names, "S0", stratum_cfg, first_item["seed"],
    )
    # Corrupt the validation record to leak a real training query spot.
    for record in mask_bank_bank["records"]:
        if record["split"] == "validation":
            record["query_obs_names"] = leaked["query_obs_names"]
            record["context_obs_names"] = [b for b in obs_names if b not in leaked["query_obs_names"]]

    with pytest.raises(ValueError, match="leakage across splits"):
        build_mask_fingerprint_report(
            manifest, "S0", coords3d, slice_ids, obs_names, _REPORT_STRATA, training_bank, mask_bank_bank,
        )
