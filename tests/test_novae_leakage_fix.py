"""
Regression tests for the 2026-07-20 Novae leakage fix (see
docs/results_log.md's 2026-07-20 entry, "CRITICAL -- Novae features leak
masked query information"). get_novae_features (src/training/train.py)
computes Novae over the FULL intact sample before any masking exists --
since Novae's representations are graph-propagated, a context spot's
embedding could carry graph-diffused information from its masked/query
neighbors' real expression. get_novae_features_context_only fixes this by
running Novae's own graph construction on a context-ONLY AnnData, so
query spots never exist in the graph at all.

Uses a mocked precompute_novae_features (real `novae` package not
required) since the point being tested is the LEAK-FREE SUBSETTING LOGIC
(what AnnData gets passed in, how the result gets placed back), not
Novae's own real representations.

Run with: python -m tests.test_novae_leakage_fix
"""
import tempfile

import anndata as ad
import numpy as np
from omegaconf import OmegaConf

from src.training.train import get_novae_features_context_only


def _make_adata(n=10, n_genes=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_genes)).astype("float32")
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"spot{i}" for i in range(n)]
    return adata


def _cfg(cache_dir):
    return OmegaConf.create({
        "data": {"hest_data_dir": "unused", "hest_cache_dir": cache_dir, "sample_id": "TESTSAMPLE"},
    })


def test_precompute_only_ever_sees_context_rows():
    """The real leak-free property: precompute_novae_features must be
    called with an AnnData containing ONLY the context spots -- proves
    query rows never enter Novae's graph construction at all, not just
    that they're excluded from the OUTPUT."""
    adata = _make_adata(10)
    context_mask = np.array([True, True, False, True, False, False, True, False, False, True])
    expected_context_names = set(np.array(adata.obs_names)[context_mask])

    seen = []

    def fake_precompute(sub_adata):
        seen.append(set(sub_adata.obs_names))
        return np.arange(sub_adata.n_obs, dtype="float32")[:, None] * np.ones((1, 3), dtype="float32")

    with tempfile.TemporaryDirectory() as tmp:
        import src.training.train as train_mod
        orig = None
        import src.models.conditioning as cond_mod
        orig = cond_mod.precompute_novae_features
        cond_mod.precompute_novae_features = fake_precompute
        try:
            full = get_novae_features_context_only(_cfg(tmp), adata, context_mask)
        finally:
            cond_mod.precompute_novae_features = orig

    assert len(seen) == 1
    assert seen[0] == expected_context_names, (
        "precompute_novae_features was called with the wrong spot set -- "
        f"saw {seen[0]}, expected only context spots {expected_context_names}"
    )
    print("[novae_leakage_fix] OK — precompute_novae_features only ever sees context-only rows")


def test_full_length_array_correct_placement_and_zeros_elsewhere():
    adata = _make_adata(8)
    context_mask = np.array([True, False, True, True, False, False, True, False])

    def fake_precompute(sub_adata):
        return np.arange(1, sub_adata.n_obs + 1, dtype="float32")[:, None] * np.ones((1, 4), dtype="float32")

    with tempfile.TemporaryDirectory() as tmp:
        import src.models.conditioning as cond_mod
        orig = cond_mod.precompute_novae_features
        cond_mod.precompute_novae_features = fake_precompute
        try:
            full = get_novae_features_context_only(_cfg(tmp), adata, context_mask)
        finally:
            cond_mod.precompute_novae_features = orig

    assert full.shape == (8, 4)
    # context rows (indices 0,2,3,6) get the fake per-context-row values 1,2,3,4 in order
    assert np.array_equal(full[0], [1, 1, 1, 1])
    assert np.array_equal(full[2], [2, 2, 2, 2])
    assert np.array_equal(full[3], [3, 3, 3, 3])
    assert np.array_equal(full[6], [4, 4, 4, 4])
    # non-context rows must be exactly zero (never read by any current caller, see docstring)
    for i in (1, 4, 5, 7):
        assert np.array_equal(full[i], [0, 0, 0, 0]), f"non-context row {i} must be zero"
    print("[novae_leakage_fix] OK — context rows correctly placed, non-context rows are zero")


def test_cache_hit_avoids_recompute_for_same_context_set():
    adata = _make_adata(6)
    context_mask = np.array([True, True, False, True, False, False])
    call_count = [0]

    def fake_precompute(sub_adata):
        call_count[0] += 1
        return np.ones((sub_adata.n_obs, 2), dtype="float32")

    with tempfile.TemporaryDirectory() as tmp:
        import src.models.conditioning as cond_mod
        orig = cond_mod.precompute_novae_features
        cond_mod.precompute_novae_features = fake_precompute
        try:
            cfg = _cfg(tmp)
            full1 = get_novae_features_context_only(cfg, adata, context_mask)
            full2 = get_novae_features_context_only(cfg, adata, context_mask)  # same mask -> cache hit
        finally:
            cond_mod.precompute_novae_features = orig

    assert call_count[0] == 1, f"expected exactly 1 real computation (2nd call should hit cache), got {call_count[0]}"
    assert np.array_equal(full1, full2)
    print("[novae_leakage_fix] OK — identical context set hits the cache, no recomputation")


def test_different_context_masks_use_different_cache_entries():
    """A DIFFERENT masking draw must NOT reuse another draw's cached
    entry -- two different context sets over the same sample are
    genuinely different graphs."""
    adata = _make_adata(6)
    mask_a = np.array([True, True, False, True, False, False])
    mask_b = np.array([False, True, True, False, True, False])  # different context set
    call_count = [0]

    def fake_precompute(sub_adata):
        call_count[0] += 1
        return np.full((sub_adata.n_obs, 2), call_count[0], dtype="float32")

    with tempfile.TemporaryDirectory() as tmp:
        import src.models.conditioning as cond_mod
        orig = cond_mod.precompute_novae_features
        cond_mod.precompute_novae_features = fake_precompute
        try:
            cfg = _cfg(tmp)
            full_a = get_novae_features_context_only(cfg, adata, mask_a)
            full_b = get_novae_features_context_only(cfg, adata, mask_b)
        finally:
            cond_mod.precompute_novae_features = orig

    assert call_count[0] == 2, f"different context masks must both trigger real computation, got {call_count[0]} calls"
    assert not np.array_equal(full_a[mask_a], full_b[mask_b] if mask_a.sum() == mask_b.sum() else full_a[mask_a])
    print("[novae_leakage_fix] OK — different context masks use separate cache entries, no collision")


if __name__ == "__main__":
    test_precompute_only_ever_sees_context_rows()
    test_full_length_array_correct_placement_and_zeros_elsewhere()
    test_cache_hit_avoids_recompute_for_same_context_set()
    test_different_context_masks_use_different_cache_entries()
    print("\nAll Novae leakage fix tests passed.")
