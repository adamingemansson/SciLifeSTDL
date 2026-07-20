"""
Smoke test for MaskedContextQueryDataset (src/training/train.py) — tiny
synthetic data, no real dataset needed. Checks shapes, and that different
indices actually draw different masks (the real point of the redesign).

Run with: python -m tests.test_masked_dataset
"""
import numpy as np
from omegaconf import OmegaConf

from src.training.train import MaskedContextQueryDataset


def _make_synthetic(n_points=200, n_genes=30, n_slices=3, seed=0):
    rng = np.random.default_rng(seed)
    coords3d = rng.uniform(0, 500, size=(n_points, 3))
    expr = rng.random((n_points, n_genes))
    slice_ids = rng.integers(0, n_slices, size=n_points)
    return coords3d, expr, slice_ids


def test_random_dropout_patches_variety():
    coords3d, expr, slice_ids = _make_synthetic()
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 2, "radius_range": [30, 80]},
    })
    ds = MaskedContextQueryDataset(coords3d, expr, slice_ids, masking_cfg, n_items=5, base_seed=0)

    query_sizes = []
    for i in range(len(ds)):
        item = ds[i]
        n_context = item["context"]["coords"].shape[0]
        n_query = item["query"]["coords"].shape[0]
        assert n_context + n_query == coords3d.shape[0]
        assert item["context"]["expression"].shape == (n_context, expr.shape[1])
        assert item["target_expression"].shape == (n_query, expr.shape[1])
        query_sizes.append(n_query)

    assert len(set(query_sizes)) > 1, (
        f"all {len(ds)} draws produced the same query size {query_sizes} — "
        "masking isn't actually varying per index"
    )
    print(f"[random_dropout_patches] OK — query sizes across draws: {query_sizes}")


def test_hold_out_slice_variety():
    coords3d, expr, slice_ids = _make_synthetic(n_slices=4)
    masking_cfg = OmegaConf.create({"strategy": "hold_out_slice", "params": {}})
    ds = MaskedContextQueryDataset(coords3d, expr, slice_ids, masking_cfg, n_items=8, base_seed=1)

    # Compare actual query coordinate content across draws, not just size —
    # two different held-out slices could coincidentally have equal point
    # counts, which would make a size-only check flaky.
    distinct_draws = set()
    for i in range(len(ds)):
        item = ds[i]
        coords_np = item["query"]["coords"].numpy()
        assert coords_np.shape[0] > 0
        distinct_draws.add(tuple(map(tuple, np.round(coords_np, 6))))

    assert len(distinct_draws) > 1, "all draws held out the identical slice — no variety across indices"
    print(f"[hold_out_slice] OK — {len(distinct_draws)} distinct held-out slices across {len(ds)} draws")


def test_max_context_points_caps_context_size():
    """2026-07-20 real-hardware OOM fix: random_dropout_patches leaves
    EVERY non-query point as context, unbounded — on tkdgx1, two jobs
    running the identical config differed only by seed and ended up at
    ~20GB vs ~39.5GB of a 40GB card purely from which slice draw they got.
    masking.max_context_points bounds this deterministically."""
    coords3d, expr, slice_ids = _make_synthetic(n_points=200, n_slices=1)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 1, "radius_range": [10, 20]},
        "max_context_points": 50,
    })
    ds = MaskedContextQueryDataset(coords3d, expr, slice_ids, masking_cfg, n_items=5, base_seed=0)
    for i in range(len(ds)):
        item = ds[i]
        n_context = item["context"]["coords"].shape[0]
        assert n_context <= 50, f"context size {n_context} exceeds max_context_points=50"
    print("[max_context_points] OK — context size capped at 50 across all draws")


def test_max_context_points_none_preserves_prior_behavior():
    """max_context_points absent/None must behave exactly as before — the
    cap is purely opt-in."""
    coords3d, expr, slice_ids = _make_synthetic(n_points=200, n_slices=1)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 1, "radius_range": [10, 20]},
    })
    ds = MaskedContextQueryDataset(coords3d, expr, slice_ids, masking_cfg, n_items=3, base_seed=0)
    for i in range(len(ds)):
        item = ds[i]
        n_context = item["context"]["coords"].shape[0]
        n_query = item["query"]["coords"].shape[0]
        assert n_context + n_query == coords3d.shape[0], (
            "without max_context_points, context+query must still cover every point"
        )
    print("[max_context_points] OK — absent cap leaves context/query split unchanged")


if __name__ == "__main__":
    test_random_dropout_patches_variety()
    test_hold_out_slice_variety()
    test_max_context_points_caps_context_size()
    test_max_context_points_none_preserves_prior_behavior()
    print("\nAll MaskedContextQueryDataset smoke tests passed.")
