"""Diagnostics for spatial displacement and cross-gene template reuse.

These functions are deliberately model-agnostic.  They consume one held-out
whole-slide prediction at a time and never expose its target to the model.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from gen3_multiscale.evaluation.metrics import pearson_per_gene
from gen3_multiscale.evaluation.structured_field_metrics import coexpression_agreement


def knn_indices(coords: np.ndarray, *, k_neighbors: int = 6) -> np.ndarray:
    """Directed k-nearest-neighbour rows including each spot itself."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must have shape [n_spots >= 2, 2]")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    count = min(int(k_neighbors) + 1, coords.shape[0])
    _, indices = cKDTree(coords).query(coords, k=count)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices


def _smooth(values: np.ndarray, neighbors: np.ndarray, passes: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    for _ in range(int(passes)):
        result = result[neighbors].mean(axis=1)
    return result


def multiscale_pcc_per_gene(
    predicted: np.ndarray, target: np.ndarray, coords: np.ndarray, *,
    k_neighbors: int = 6, chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """Exact PCC and PCC after one/two local graph-averaging passes.

    Both prediction and target are blurred.  Improvement therefore means the
    model recovered the right coarse region but missed fine spot placement;
    it is not a post-processing boost to the prediction alone.
    """
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("predicted and target must match [n_spots, n_genes]")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("predicted and target must be finite")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    neighbors = knn_indices(coords, k_neighbors=k_neighbors)
    exact = pearson_per_gene(predicted, target)
    blur1 = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    blur2 = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    for start in range(0, predicted.shape[1], chunk_size):
        end = min(start + chunk_size, predicted.shape[1])
        pred1 = _smooth(predicted[:, start:end], neighbors, 1)
        true1 = _smooth(target[:, start:end], neighbors, 1)
        blur1[start:end] = pearson_per_gene(pred1, true1)
        blur2[start:end] = pearson_per_gene(
            _smooth(pred1, neighbors, 1), _smooth(true1, neighbors, 1),
        )
    return {"exact": exact, "blur1": blur1, "blur2": blur2}


def variance_diagnostics(predicted: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("predicted and target must match [n_spots, n_genes]")
    target_std = target.std(axis=0)
    predicted_std = predicted.std(axis=0)
    ratio = np.full(target_std.shape, np.nan, dtype=np.float64)
    eligible = target_std >= 1e-8
    ratio[eligible] = predicted_std[eligible] / target_std[eligible]
    return {
        "target_std": target_std,
        "predicted_std": predicted_std,
        "predicted_to_target_std_ratio": ratio,
    }


def _standardize_gene_maps(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0)
    standardized = np.divide(
        centered, np.maximum(scale, 1e-12),
        out=np.zeros_like(centered), where=np.maximum(scale, 1e-12) > 0,
    )
    standardized[:, scale < 1e-8] = 0.0
    return standardized


def effective_rank(values: np.ndarray) -> dict[str, float]:
    """Entropy and participation effective rank across standardized gene maps."""
    standardized = _standardize_gene_maps(values)
    gram = standardized.T @ standardized
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    total = float(eigenvalues.sum())
    if total <= 0:
        return {"entropy_effective_rank": 0.0, "participation_rank": 0.0}
    probabilities = eigenvalues / total
    positive = probabilities > 0
    entropy_rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    participation = float(1.0 / np.sum(np.square(probabilities)))
    return {
        "entropy_effective_rank": entropy_rank,
        "participation_rank": participation,
    }


def template_retrieval(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Ask whether each predicted gene map retrieves its matching target map."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != target.shape or predicted.shape[1] < 2:
        raise ValueError("template retrieval needs matching [spots, genes>=2] arrays")
    pred_z = _standardize_gene_maps(predicted)
    target_z = _standardize_gene_maps(target)
    correlation = pred_z.T @ target_z / float(predicted.shape[0])
    n_genes = correlation.shape[0]
    # Stable sort makes ties deterministic. Rank 1 is best.
    order = np.argsort(-correlation, axis=1, kind="mergesort")
    ranks = np.empty(n_genes, dtype=np.int64)
    for index in range(n_genes):
        ranks[index] = int(np.flatnonzero(order[index] == index)[0]) + 1
    top_targets = order[:, 0]
    counts = np.bincount(top_targets, minlength=n_genes)
    off_diagonal = correlation.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    return {
        "same_gene_top1_fraction": float(np.mean(ranks == 1)),
        "same_gene_top5_fraction": float(np.mean(ranks <= min(5, n_genes))),
        "median_same_gene_rank": float(np.median(ranks)),
        "mean_same_gene_correlation": float(np.mean(np.diag(correlation))),
        "mean_best_wrong_gene_correlation": float(np.mean(np.max(off_diagonal, axis=1))),
        "unique_top_target_fraction": float(np.count_nonzero(counts) / n_genes),
        "maximum_target_reuse_count": int(counts.max()),
        "n_genes": int(n_genes),
    }


def selected_template_diagnostics(
    predicted: np.ndarray, target: np.ndarray, *, max_genes: int = 256,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Template diagnostics on truth-variable genes selected by target variance.

    Selection is diagnostic-only and slide-specific; these values must not be
    presented as a pre-registered benchmark score.
    """
    if max_genes < 2:
        raise ValueError("max_genes must be at least two")
    target = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if target.ndim != 2 or predicted.shape != target.shape:
        raise ValueError("predicted and target must match [n_spots, n_genes]")
    variance = target.var(axis=0)
    eligible = np.flatnonzero(variance >= 1e-8)
    if eligible.size < 2:
        raise ValueError("fewer than two truth-variable genes")
    order = eligible[np.argsort(-variance[eligible], kind="mergesort")]
    selected = order[:min(int(max_genes), order.size)]
    target_selected = target[:, selected]
    predicted_selected = predicted[:, selected]
    pred_rank = effective_rank(predicted_selected)
    true_rank = effective_rank(target_selected)
    retrieval = template_retrieval(predicted_selected, target_selected)
    coexpression = coexpression_agreement(predicted_selected, target_selected)
    result: dict[str, float | int] = {
        "n_selected_genes": int(selected.size),
        "predicted_entropy_effective_rank": pred_rank["entropy_effective_rank"],
        "target_entropy_effective_rank": true_rank["entropy_effective_rank"],
        "entropy_effective_rank_ratio": (
            pred_rank["entropy_effective_rank"] / true_rank["entropy_effective_rank"]
            if true_rank["entropy_effective_rank"] > 0 else float("nan")
        ),
        "predicted_participation_rank": pred_rank["participation_rank"],
        "target_participation_rank": true_rank["participation_rank"],
        **retrieval,
        "coexpression_matrix_pcc": coexpression["correlation_matrix_pcc"],
        "coexpression_matrix_mae": coexpression["correlation_matrix_mae"],
    }
    return selected, result
