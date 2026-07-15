"""
Independent cell-type-plausibility classifier — the validation plan for
the "no explicit cell-type conditioning" design decision
(docs/architecture_plan.md), reusing the ARI/NMI downstream-task-
preservation idea already planned in docs/metrics_notes.md SS1.

HEST-1k's INT1 sample has no curated per-spot cell-type labels in the
downloaded .h5ad (confirmed 2026-07-14 by inspecting adata.obs.columns
directly — only QC/spatial columns, no cell_type). Substituting the
fallback already documented in our own metrics plan: Leiden clustering
(Traag et al. 2019, Scientific Reports, "From Louvain to Leiden") on real
expression as pseudo cell-type labels — the same "cluster the data,
compare clusterings" idea docs/metrics_notes.md already specifies,
adapted into a *trained classifier* (rather than a second independent
clustering run) so a single held-out generated location can be scored on
its own — a clustering algorithm needs many points at once, a trained
classifier does not.

Deliberately plain scikit-learn, not a PyTorch/Lightning model —
architecturally independent of every model family in
src/models/registry.py, so a shared bias between generator and evaluator
can't inflate the plausibility score.
"""
from __future__ import annotations

import numpy as np
import anndata as ad


def cluster_pseudo_labels(adata: ad.AnnData, resolution: float = 1.0,
                           n_neighbors: int = 15, seed: int = 0) -> np.ndarray:
    """Leiden clustering on real expression as a cell-type-like label
    substitute when no curated labels exist (see module docstring)."""
    import scanpy as sc
    adata = adata.copy()
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
    # flavor="igraph" (+ n_iterations=2, directed=False) pins to scanpy's
    # own documented future default rather than the current
    # leidenalg-backed one, silencing its FutureWarning (2026-07-15, real
    # warning seen on a run) — same clustering algorithm, different
    # backend implementation.
    sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                 flavor="igraph", n_iterations=2, directed=False)
    return adata.obs["leiden"].to_numpy().astype(str)


class CellTypePlausibilityClassifier:
    """Train on real (expression -> pseudo cell type) pairs held out from
    generation training; use to check whether generated expression at a
    real held-out location still looks like the right type."""

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    def fit(self, expression: np.ndarray, labels: np.ndarray) -> "CellTypePlausibilityClassifier":
        self.model.fit(expression, labels)
        return self

    def predict(self, expression: np.ndarray) -> np.ndarray:
        return self.model.predict(expression)

    def plausibility_accuracy(self, generated_expression: np.ndarray,
                               true_labels: np.ndarray) -> float:
        """Fraction of generated locations classified into their real
        held-out location's true (pseudo) type — the plausibility score."""
        pred = self.predict(generated_expression)
        return float(np.mean(pred == true_labels))
