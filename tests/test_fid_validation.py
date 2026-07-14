"""
Validation for the custom FID-style metric (docs/metrics_notes.md SS2,
task #14). The actual deliverable here is running the validation plan the
docs already prescribe before trusting the number, not just picking an
embedding function:
  (a) monotonic sanity gradient: FID should worsen as a candidate degrades
      from ground truth -> blurred approximation -> pure noise.
  (b) correlates with existing pointwise metrics on cases they agree on.
  (c) sensitive to a failure mode pointwise metrics miss: correct marginal
      values, wrong spatial arrangement.
Synthetic spatially-clustered data — no real dataset needed for this.

Run with: python -m tests.test_fid_validation
"""
import numpy as np

from src.evaluation import metrics as ev


def _make_synthetic_tissue(n_points=300, n_genes=30, n_clusters=4, seed=0):
    """Spatially clustered synthetic ST-like data: points in 2D space,
    grouped into spatial clusters, each cluster with a distinct mean
    expression profile — a minimal stand-in for real tissue's spatial
    correlation between structure and biology."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10, size=(n_clusters, 2))
    cluster_id = rng.integers(0, n_clusters, size=n_points)
    coords = centers[cluster_id] + rng.normal(scale=0.5, size=(n_points, 2))

    gene_means = rng.uniform(0, 3, size=(n_clusters, n_genes))
    expression = gene_means[cluster_id] + rng.normal(scale=0.3, size=(n_points, n_genes))
    expression = np.clip(expression, 0, None)
    return coords, expression, cluster_id


def _fit_pca_embed(real_expression, n_components=10):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components).fit(real_expression)
    return lambda expr: ev.embed_pca(expr, pca)


def test_monotonic_sanity_gradient():
    coords, expression, _ = _make_synthetic_tissue(seed=0)
    embed = _fit_pca_embed(expression)

    blurred = expression * 0.5 + expression.mean(axis=0) * 0.5  # shrink toward the global mean
    noise = np.random.default_rng(1).normal(
        loc=expression.mean(), scale=expression.std(), size=expression.shape
    )

    real_emb = embed(expression)
    fid_truth = ev.st_fid(real_emb, embed(expression))
    fid_blur = ev.st_fid(real_emb, embed(blurred))
    fid_noise = ev.st_fid(real_emb, embed(noise))

    assert fid_truth < fid_blur < fid_noise, (
        f"expected fid_truth < fid_blur < fid_noise, got "
        f"{fid_truth:.4f}, {fid_blur:.4f}, {fid_noise:.4f}"
    )
    print(f"[monotonic_sanity_gradient] OK — truth={fid_truth:.4f} < "
          f"blur={fid_blur:.4f} < noise={fid_noise:.4f}")


def test_correlates_with_pointwise_metrics():
    coords, expression, _ = _make_synthetic_tissue(seed=0)
    embed = _fit_pca_embed(expression)

    blurred = expression * 0.5 + expression.mean(axis=0) * 0.5
    noise = np.random.default_rng(1).normal(
        loc=expression.mean(), scale=expression.std(), size=expression.shape
    )

    real_emb = embed(expression)
    fid_blur = ev.st_fid(real_emb, embed(blurred))
    fid_noise = ev.st_fid(real_emb, embed(noise))
    pcc_blur = np.nanmean(ev.pearson_per_gene(blurred, expression))
    pcc_noise = np.nanmean(ev.pearson_per_gene(noise, expression))

    assert fid_noise > fid_blur, "FID should rank noise worse than blur"
    assert pcc_noise < pcc_blur, "PCC should also rank noise worse than blur"
    print(f"[correlates_with_pointwise_metrics] OK — FID and PCC agree on "
          f"the noise-vs-blur ranking (fid: {fid_blur:.4f} < {fid_noise:.4f}, "
          f"pcc: {pcc_noise:.4f} < {pcc_blur:.4f})")


def test_sensitive_to_spatial_arrangement_only_corruption():
    """The actual selling point (docs/metrics_notes.md SS2): a corruption
    that keeps each point's marginal expression value (same multiset of
    real profiles) but reassigns which profile sits at which location.
    Plain per-point embeddings are provably blind to this — permuting an
    unordered embedding set never changes its mean/covariance — so this
    also validates the "unit of comparison" fix (pool_knn_neighborhood)."""
    coords, expression, _ = _make_synthetic_tissue(seed=0)

    rng = np.random.default_rng(2)
    shuffled_idx = rng.permutation(expression.shape[0])
    shuffled_expression = expression[shuffled_idx]  # same multiset, wrong locations

    embed_point = _fit_pca_embed(expression)
    fid_point = ev.st_fid(embed_point(expression), embed_point(shuffled_expression))
    assert fid_point < 1e-6, (
        f"expected ~0 (permutation-invariance of per-point FID), got {fid_point:.6f}"
    )

    from sklearn.decomposition import PCA
    real_patches = ev.pool_knn_neighborhood(coords, expression, k=8)
    shuffled_patches = ev.pool_knn_neighborhood(coords, shuffled_expression, k=8)
    pca = PCA(n_components=10).fit(real_patches)
    fid_patch = ev.st_fid(pca.transform(real_patches), pca.transform(shuffled_patches))

    assert fid_patch > 0.05, (
        f"expected the patch-level embedding to detect the spatial-arrangement "
        f"corruption (fid_patch={fid_patch:.4f}), it did not"
    )
    print(f"[sensitive_to_spatial_arrangement_only_corruption] OK — "
          f"per-point FID blind to shuffling ({fid_point:.6f}), "
          f"patch-level FID catches it ({fid_patch:.4f})")


if __name__ == "__main__":
    test_monotonic_sanity_gradient()
    test_correlates_with_pointwise_metrics()
    test_sensitive_to_spatial_arrangement_only_corruption()
    print("\nAll FID/MMD validation checks passed.")
