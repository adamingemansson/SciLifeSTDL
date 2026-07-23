"""Leak-safe niche (spatial-domain) labels for the transport model's
``use_niche_candidate`` (see ``HierarchicalGeneTransportRegressor``'s own
docstring in src/models/registry.py for the modeling side).

Reimplements BANKSY's (Singhal et al., *Nature Methods* 2024) own core
neighbor-augmented feature construction directly with scanpy/sklearn rather
than depending on the ``banksy`` package's own interactive multi-parameter
sweep/plotting pipeline (built for exploratory analysis across several
lambda/k combinations, not "give me one niche id per spot" inside a
training loop). The published mechanism itself is simple and reproduced
here faithfully: for each spot, L2-normalize its own expression and the
mean expression of its ``k_geom`` nearest spatial neighbors separately,
then concatenate them weighted by ``sqrt(1 - lambda)`` /
``sqrt(lambda)``. ``lambda_param=0.8`` matches the paper's own recommended
setting for domain/niche segmentation (as opposed to ``lambda=0.2`` for
cell-typing, which this project has no use for). Clustering reuses
``src.evaluation.cell_type_classifier.cluster_pseudo_labels``'s existing,
already-tested "kmeans by default, deterministic, data-size-aware cluster
count" pattern rather than inventing a second clustering utility.

Just like ``src/data/context_features.py``'s Novae features, niche labels
computed on the FULL slide (including hidden query rows) would leak hidden
query expression through the neighbor-averaging step. This module must
only ever be called through ``ContextOnlyFeatureProvider`` (the generic
engine in ``context_features.py``, reused as-is) on the observed context
subgraph, one training draw's mask at a time.
"""
from __future__ import annotations

import numpy as np


def compute_banksy_augmented_niche_labels(
    adata,
    lambda_param: float = 0.8,
    k_geom: int = 6,
    resolution: float = 1.0,
    min_spots_for_clustering: int = 20,
) -> np.ndarray:
    """Return ``[n_obs, 1]`` float32 niche-cluster ids for every row of
    ``adata`` (a context-only subset -- see module docstring).

    Below ``min_spots_for_clustering``, this training draw's context is too
    small to form a meaningful cluster structure, so every spot is assigned
    the single niche 0 -- an explicit, documented degradation to
    "one niche covering everyone" (the same shape of fallback
    ``use_global_candidate`` always uses), not a silent failure or a
    fabricated cluster boundary drawn from noise.
    """
    from sklearn.neighbors import NearestNeighbors
    from src.evaluation.cell_type_classifier import cluster_pseudo_labels
    import anndata as ad

    x = adata.X
    x = np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)
    n_obs = x.shape[0]
    if n_obs < int(min_spots_for_clustering):
        return np.zeros((n_obs, 1), dtype=np.float32)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    k = min(int(k_geom), n_obs - 1)
    neighbor_idx = (
        NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords, return_distance=False)[:, 1:]
    )
    neighbor_mean = x[neighbor_idx].mean(axis=1)

    own_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    nbr_norm = neighbor_mean / (np.linalg.norm(neighbor_mean, axis=1, keepdims=True) + 1e-8)
    lam = float(lambda_param)
    augmented = np.concatenate(
        [np.sqrt(1.0 - lam) * own_norm, np.sqrt(lam) * nbr_norm], axis=1
    ).astype(np.float32)

    labels = cluster_pseudo_labels(ad.AnnData(augmented), resolution=float(resolution), method="kmeans")
    return labels.astype(np.int64).reshape(-1, 1).astype(np.float32)


def model_uses_niche_candidate(model_params: dict) -> bool:
    return bool(model_params.get("use_niche_candidate", False))


def niche_input_mode(cfg, model_params: dict) -> str:
    """Validate and return the requested niche-candidate safety mode.

    Mirrors ``context_features.novae_input_mode``'s fail-closed contract
    exactly: requesting the niche candidate without an explicit
    ``data.niche_mode=context_only`` raises, so a config can never silently
    end up computing niche labels on the full (leaking) slide graph.
    """
    if not model_uses_niche_candidate(model_params):
        return "disabled"
    mode = str(cfg.data.get("niche_mode", "disabled"))
    if mode == "disabled":
        raise ValueError(
            "This model requests use_niche_candidate, but data.niche_mode is 'disabled'. "
            "Full-slide niche clustering leaks hidden query expression through its "
            "neighbor-averaging step, exactly like full-slide Novae features. "
            "Set data.niche_mode=context_only."
        )
    if mode != "context_only":
        raise ValueError(f"unknown data.niche_mode {mode!r}")
    return mode
