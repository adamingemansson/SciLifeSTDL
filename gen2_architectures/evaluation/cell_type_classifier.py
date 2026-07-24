"""
Independent spatial-domain-plausibility classifier.

The generated-expression diagnostic in this module is deliberately called a
*spatial-domain* score, not a cell-type score, unless the caller supplies a
curated label column.  When curated labels are unavailable, deterministic
unsupervised expression clusters are used only as a secondary plausibility
check; they are not ground-truth biology.
"""
from __future__ import annotations

import numpy as np
import anndata as ad


def _dense_expression(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    return np.asarray(x if isinstance(x, np.ndarray) else x.toarray(), dtype=np.float32)


def cluster_pseudo_labels(
    adata: ad.AnnData,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    seed: int = 0,
    method: str = "kmeans",
    n_clusters: int | None = None,
) -> np.ndarray:
    """Create deterministic unsupervised expression-domain labels.

    ``method="kmeans"`` is the default because it is deterministic, lightweight
    and avoids a first-run numba/graph-compilation cost during every benchmark
    process.  ``method="leiden"`` preserves the historical Scanpy diagnostic.
    Neither method produces curated cell types.
    """
    method = str(method).lower()
    if adata.n_obs == 0:
        return np.empty(0, dtype=str)
    if adata.n_obs == 1:
        return np.asarray(["0"], dtype=str)

    if method == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA

        x = _dense_expression(adata)
        max_components = min(50, x.shape[0] - 1, x.shape[1])
        if max_components >= 2:
            x = PCA(n_components=max_components, random_state=seed).fit_transform(x)
        if n_clusters is None:
            # A modest, data-size-aware domain count.  The resolution multiplier
            # provides a familiar control without pretending it is identical to
            # Leiden's resolution parameter.
            n_clusters = int(round(max(2.0, np.sqrt(adata.n_obs / 2.0)) * float(resolution)))
        n_clusters = max(2, min(int(n_clusters), adata.n_obs))
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
            batch_size=min(1024, max(32, adata.n_obs)),
        ).fit_predict(x)
        return labels.astype(str)

    if method == "leiden":
        import scanpy as sc

        work = adata.copy()
        n_comps = min(50, work.n_vars - 1, work.n_obs - 1)
        if n_comps >= 2:
            sc.pp.pca(work, n_comps=n_comps)
        sc.pp.neighbors(work, n_neighbors=min(int(n_neighbors), work.n_obs - 1), random_state=seed)
        sc.tl.leiden(
            work,
            resolution=float(resolution),
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        return work.obs["leiden"].to_numpy().astype(str)

    raise ValueError("pseudo-domain method must be 'kmeans' or 'leiden'")


class SpatialDomainPlausibilityClassifier:
    """Classify generated expression into labels learned from real context."""

    def __init__(self, n_estimators: int = 100, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(seed),
            n_jobs=1,
        )

    def fit(self, expression: np.ndarray, labels: np.ndarray) -> "SpatialDomainPlausibilityClassifier":
        self.model.fit(expression, labels)
        return self

    def predict(self, expression: np.ndarray) -> np.ndarray:
        return self.model.predict(expression)

    def plausibility_accuracy(self, generated_expression: np.ndarray,
                               true_labels: np.ndarray) -> float:
        pred = self.predict(generated_expression)
        return float(np.mean(pred == true_labels))


# Backward-compatible import used by the existing test suite and old scripts.
CellTypePlausibilityClassifier = SpatialDomainPlausibilityClassifier
