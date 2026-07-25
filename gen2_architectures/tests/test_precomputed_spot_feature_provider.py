import tempfile
from pathlib import Path

import anndata as ad
import numpy as np

from gen2_architectures.data.context_features import PrecomputedSpotFeatureProvider


def _fake_adata(n_obs: int = 10, n_vars: int = 5) -> ad.AnnData:
    rng = np.random.default_rng(0)
    adata = ad.AnnData(X=rng.standard_normal((n_obs, n_vars)).astype(np.float32))
    adata.obs_names = [f"spot{i}" for i in range(n_obs)]
    adata.obsm["spatial"] = rng.uniform(0, 1000, size=(n_obs, 2))
    return adata


def test_computes_the_whole_sample_once_and_slices_by_context_mask():
    calls = []

    def feature_fn(a):
        calls.append(a.n_obs)
        return np.arange(a.n_obs * 3, dtype=np.float32).reshape(a.n_obs, 3)

    adata = _fake_adata(n_obs=10)
    provider = PrecomputedSpotFeatureProvider(adata, feature_fn=feature_fn)

    mask_a = np.zeros(10, dtype=bool)
    mask_a[[0, 1, 2]] = True
    mask_b = np.zeros(10, dtype=bool)
    mask_b[[5, 6]] = True

    out_a = provider(mask_a)
    out_b = provider(mask_b)

    # feature_fn called on the WHOLE sample exactly once, regardless of
    # how many different context masks are subsequently queried
    assert calls == [10]
    assert out_a.shape == (3, 3)
    assert out_b.shape == (2, 3)
    assert provider.output_dim == 3


def test_disk_cache_is_one_file_per_sample_not_per_mask():
    """Real bug this class fixes: the old (Novae-oriented) provider wrote
    a new cache file per DISTINCT context mask. This provider must write
    exactly one file per sample, regardless of how many different masks
    are ever queried against it."""
    def feature_fn(a):
        return np.ones((a.n_obs, 4), dtype=np.float32)

    adata = _fake_adata(n_obs=10)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        provider = PrecomputedSpotFeatureProvider(adata, cache_dir=cache_dir, sample_id="S1", feature_fn=feature_fn)
        for i in range(20):
            mask = np.zeros(10, dtype=bool)
            mask[i % 10] = True
            provider(mask)
        files = list(cache_dir.glob("*.npz"))
        assert len(files) == 1, f"expected exactly one cache file, found {files}"
        assert files[0].name == "S1.npz"


def test_second_provider_instance_hits_the_disk_cache_without_recomputing():
    calls = []

    def feature_fn(a):
        calls.append(1)
        return np.full((a.n_obs, 2), 7.0, dtype=np.float32)

    adata = _fake_adata(n_obs=6)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        p1 = PrecomputedSpotFeatureProvider(adata, cache_dir=cache_dir, sample_id="S1", feature_fn=feature_fn)
        p1(np.ones(6, dtype=bool))
        assert len(calls) == 1

        p2 = PrecomputedSpotFeatureProvider(adata, cache_dir=cache_dir, sample_id="S1", feature_fn=feature_fn)
        out = p2(np.ones(6, dtype=bool))
        assert len(calls) == 1, "second provider instance should hit the disk cache, not recompute"
        assert np.array_equal(out, np.full((6, 2), 7.0, dtype=np.float32))


def test_stale_cache_is_recomputed_when_the_feature_signature_changes():
    def feature_fn(a):
        return np.ones((a.n_obs, 2), dtype=np.float32)

    adata_v1 = _fake_adata(n_obs=6, n_vars=5)
    adata_v2 = _fake_adata(n_obs=6, n_vars=8)  # different var count -> different feature_signature
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        p1 = PrecomputedSpotFeatureProvider(adata_v1, cache_dir=cache_dir, sample_id="S1", feature_fn=feature_fn)
        p1(np.ones(6, dtype=bool))

        calls = []

        def feature_fn_v2(a):
            calls.append(1)
            return np.full((a.n_obs, 2), 9.0, dtype=np.float32)

        p2 = PrecomputedSpotFeatureProvider(adata_v2, cache_dir=cache_dir, sample_id="S1", feature_fn=feature_fn_v2)
        out = p2(np.ones(6, dtype=bool))
        assert len(calls) == 1, "a changed feature_signature must trigger a real recompute, not a stale cache hit"
        assert np.array_equal(out, np.full((6, 2), 9.0, dtype=np.float32))
