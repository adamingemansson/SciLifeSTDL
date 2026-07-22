"""
Regression tests for the 2026-07-20 held-out-spot generalization test
(see masking.held_out_mask's own docstring, and docs/results_log.md's
2026-07-20 entry): does the model just recall spots it saw as a
supervised training TARGET many times over a training run, rather than
genuinely reconstructing from context? Every training step redraws
masking randomly over the same slide, and a query spot's real
coordinates/image are always given as input while the network is trained
to output that spot's real expression -- across thousands of steps, most
spots get used as a training target repeatedly, and the existing "held-out
masking draw" eval only guarantees a fresh HOLE PLACEMENT, not spots the
model has never been trained to predict.

held_out_mask + make_context_query_split's heldout_mask param fix this:
a FIXED (seeded) subset of spots that are structurally guaranteed to
NEVER be placed in a training query set for the whole training run.

Run with: python -m tests.test_heldout_generalization
"""
import numpy as np
from omegaconf import OmegaConf

from src.data import masking
from src.training.train import make_context_query_split, MaskedContextQueryDataset


def test_held_out_mask_deterministic_and_matches_fraction():
    m1 = masking.held_out_mask(1000, 0.15, seed=42)
    m2 = masking.held_out_mask(1000, 0.15, seed=42)
    assert np.array_equal(m1, m2), "same seed must produce the identical mask"
    assert 0.10 < m1.mean() < 0.20, f"mean {m1.mean()} far from requested 0.15"
    m3 = masking.held_out_mask(1000, 0.15, seed=43)
    assert not np.array_equal(m1, m3), "different seeds must produce different masks"
    print("[heldout_generalization] OK — held_out_mask is deterministic per-seed, matches requested fraction")


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    coords3d = rng.uniform(0, 500, size=(n, 3))
    slice_ids = np.zeros(n, dtype=int)
    return coords3d, slice_ids


def test_heldout_mask_never_appears_in_query_across_many_draws():
    """The core correctness property: no matter how many random masking
    draws happen, a held-out spot must NEVER be assigned to the query set."""
    coords3d, slice_ids = _synthetic()
    heldout = masking.held_out_mask(300, 0.2, seed=7)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 3, "radius_range": [30, 80]},
    })
    for seed in range(50):
        context_mask, query_mask = make_context_query_split(
            coords3d, slice_ids, masking_cfg, seed, heldout_mask=heldout
        )
        assert not (query_mask & heldout).any(), f"seed {seed}: held-out spot leaked into query set"
        assert (context_mask | query_mask).all(), "every spot must still be covered"
        assert not (context_mask & query_mask).any(), "context/query must stay disjoint"
    print("[heldout_generalization] OK — across 50 random draws, held-out spots never appear in query")


def test_without_guard_heldout_spots_do_get_drawn_as_query():
    """Sanity check that the guard is doing real work, not vacuously
    passing because held-out spots were never going to be drawn anyway."""
    coords3d, slice_ids = _synthetic()
    heldout = masking.held_out_mask(300, 0.2, seed=7)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 3, "radius_range": [30, 80]},
    })
    touched = any(
        (make_context_query_split(coords3d, slice_ids, masking_cfg, seed)[1] & heldout).any()
        for seed in range(50)
    )
    assert touched, "expected at least one draw to touch a held-out spot without the guard"
    print("[heldout_generalization] OK — confirmed the guard is doing real work (not a vacuous no-op)")


def test_heldout_mask_none_is_true_noop():
    coords3d, slice_ids = _synthetic()
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 3, "radius_range": [30, 80]},
    })
    for seed in range(5):
        a = make_context_query_split(coords3d, slice_ids, masking_cfg, seed)
        b = make_context_query_split(coords3d, slice_ids, masking_cfg, seed, heldout_mask=None)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    print("[heldout_generalization] OK — heldout_mask=None (default) is a true no-op")


def test_masked_context_query_dataset_respects_heldout_mask():
    coords3d, slice_ids = _synthetic(n=200)
    expr = np.random.default_rng(1).random((200, 10)).astype("float32")
    heldout = masking.held_out_mask(200, 0.25, seed=3)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 2, "radius_range": [30, 80]},
    })
    ds = MaskedContextQueryDataset(
        coords3d, expr, slice_ids, masking_cfg, n_items=20, base_seed=0, heldout_mask=heldout,
    )
    heldout_coords = coords3d[heldout]
    for i in range(len(ds)):
        item = ds[i]
        query_coords = item["query"]["coords"].numpy()
        # no query point should exactly match a held-out coordinate
        for hc in heldout_coords:
            assert not np.any(np.all(np.isclose(query_coords, hc), axis=-1)), (
                "a held-out spot's coordinates appeared in a training query set"
            )
    print("[heldout_generalization] OK — MaskedContextQueryDataset never draws held-out spots as query")


if __name__ == "__main__":
    test_held_out_mask_deterministic_and_matches_fraction()
    test_heldout_mask_never_appears_in_query_across_many_draws()
    test_without_guard_heldout_spots_do_get_drawn_as_query()
    test_heldout_mask_none_is_true_noop()
    test_masked_context_query_dataset_respects_heldout_mask()
    print("\nAll held-out generalization tests passed.")
