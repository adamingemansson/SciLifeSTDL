"""Tests for realized-mask fingerprinting (Step 3 of the real Gen3 data
builder/trainer, CONTRACT.md section 33)."""
import numpy as np
import pytest

from gen3_multiscale.data import mask_bank
from gen3_multiscale.data.mask_fingerprint import (
    realize_seed_and_fingerprint, realized_query_composite_ids,
    sorted_composite_query_fingerprint, verify_no_cross_split_query_leakage,
    verify_realized_seed_uniqueness,
)
from gen3_multiscale.data.mask_schedule import stratum_to_masking_cfg

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
