"""Evaluation metrics for the multiscale spatial-field architectures --
Phase 7 of the handoff ("Add losses, metrics, and diagnostic
interventions").

The eight pointwise/distributional functions below (through
pool_knn_neighborhood) are copied VERBATIM from
gen2_architectures/evaluation/metrics.py (itself identical to
src/evaluation/metrics.py) -- same provenance discipline as every other
reused module in this package (CONTRACT.md section 2): "do not let this
drift without a deliberate reason." They are pure functions with no
gen2-specific dependencies (only numpy/scipy/sklearn), so nothing about
the copy needed adaptation.

Everything from resolve_gene_panels onward is NEW for gen3_multiscale,
built to satisfy the handoff's Phase 7 requirements that have no existing
implementation anywhere in the repo (confirmed by direct search, not
assumed): patient-level aggregation ("aggregate primary results first
within held-out patients and then macro-average across patients"),
boundary-to-interior and hole-size performance bins, query-edge gradient
agreement decomposed into normal/tangential components, and spatial
variogram / graph-Laplacian agreement. resolve_gene_panels itself IS
adapted from gen2_architectures/evaluation/audit_evaluation.py's
_resolve_gene_panels (same logic, renamed to a public name since this
module doesn't have that file's private-helper context) -- named
evaluation-only gene panels already exist as a real, audited concept
there.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import linalg
from scipy.stats import pearsonr, t as _student_t


# ---------------------------------------------------------------------------
# Verbatim copy from gen2_architectures/evaluation/metrics.py -- do not let
# this drift from that file without a deliberate, documented reason.
# ---------------------------------------------------------------------------
def pearson_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-gene PCC with eligibility defined from ground truth only.

    A truth-constant gene is not identifiable within the evaluated region and
    is returned as NaN. A constant prediction for a truth-variable gene is a
    model failure and receives 0 instead of disappearing from the mean.
    """
    G = pred.shape[1]
    out = np.zeros(G)
    for g in range(G):
        if np.std(true[:, g]) < 1e-8:
            out[g] = np.nan
            continue
        if np.std(pred[:, g]) < 1e-8:
            out[g] = 0.0
            continue
        out[g] = pearsonr(pred[:, g], true[:, g])[0]
    return out


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nonzero_auc(pred: np.ndarray, true: np.ndarray) -> float:
    """AUC for distinguishing zero vs. non-zero true expression using
    predicted magnitude as the score. Flatten across genes/cells."""
    from sklearn.metrics import roc_auc_score
    y = (true.flatten() > 0).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, pred.flatten()))


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Standard Fréchet distance between two Gaussians, same formula as FID."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def st_fid(real_embeddings: np.ndarray, gen_embeddings: np.ndarray) -> float:
    """
    'ST-FID': Fréchet distance between embeddings of real vs. generated
    ST samples (cells, spots, or patches).

    real_embeddings / gen_embeddings: [N, D] arrays from whatever embedding
    function is chosen -- this function is agnostic to that choice by
    design, so the embedding function can be swapped and compared.
    """
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def st_mmd(real_embeddings: np.ndarray, gen_embeddings: np.ndarray,
           gamma: float | None = None) -> float:
    """Alternative to st_fid that drops the Gaussian assumption -- RBF-kernel
    Maximum Mean Discrepancy."""
    from sklearn.metrics.pairwise import rbf_kernel
    if gamma is None:
        gamma = 1.0 / real_embeddings.shape[1]
    Kxx = rbf_kernel(real_embeddings, real_embeddings, gamma=gamma)
    Kyy = rbf_kernel(gen_embeddings, gen_embeddings, gamma=gamma)
    Kxy = rbf_kernel(real_embeddings, gen_embeddings, gamma=gamma)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def embed_pca(expression: np.ndarray, pca_model) -> np.ndarray:
    """Simplest baseline embedding function for st_fid/st_mmd: a PCA fit on
    real reference data. pca_model = a fitted sklearn PCA (fit on real data
    only, then applied to both real held-out and generated samples)."""
    return pca_model.transform(expression)


def pool_knn_neighborhood(coords: np.ndarray, expression: np.ndarray, k: int = 8) -> np.ndarray:
    """Mean-pool each point's expression over its k nearest spatial
    neighbours (including itself) -- a lightweight "patch"/neighborhood
    unit of comparison for st_fid/st_mmd."""
    from sklearn.neighbors import NearestNeighbors
    k = min(k, coords.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    return expression[idx].mean(axis=1)


# ---------------------------------------------------------------------------
# New for gen3_multiscale, Phase 7.
# ---------------------------------------------------------------------------
def resolve_gene_panels(
    gene_names: list[str],
    gene_panels: dict[str, list[str]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Map prespecified evaluation-only gene-name panels onto model output
    columns. Adapted from gen2_architectures/evaluation/audit_evaluation.py's
    _resolve_gene_panels (identical logic; renamed to a public name here
    since this module has no surrounding private-helper context). Panels
    are evaluation views only: they never alter the model vocabulary,
    targets, loss, or decoder. Missing names are recorded rather than
    silently changing the requested panel size."""
    name_to_idx = {str(name): idx for idx, name in enumerate(gene_names)}
    indices: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for raw_name, raw_genes in (gene_panels or {}).items():
        panel_name = str(raw_name)
        if not re.fullmatch(r"[A-Za-z0-9_]+", panel_name):
            raise ValueError(
                f"gene panel name {panel_name!r} must contain only letters, digits and underscores"
            )
        requested = list(dict.fromkeys(str(gene) for gene in raw_genes))
        present = [gene for gene in requested if gene in name_to_idx]
        missing = [gene for gene in requested if gene not in name_to_idx]
        if not present:
            raise ValueError(
                f"gene panel {panel_name!r} has no genes in the model output vocabulary"
            )
        indices[panel_name] = np.asarray([name_to_idx[gene] for gene in present], dtype=np.int64)
        metadata[panel_name] = {
            "requested_count": len(requested),
            "evaluated_count": len(present),
            "genes": present,
            "missing_genes": missing,
        }
    return indices, metadata


def gene_panel_metrics(
    pred: np.ndarray, true: np.ndarray, gene_names: list[str], gene_panels: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """PCC (nanmean over genes) and RMSE restricted to each named
    evaluation-only panel -- "PCC/RMSE on named evaluation-only gene
    panels" (Phase 7)."""
    indices, _metadata = resolve_gene_panels(gene_names, gene_panels)
    out: dict[str, dict[str, float]] = {}
    for panel_name, idx in indices.items():
        panel_pred, panel_true = pred[:, idx], true[:, idx]
        out[panel_name] = {
            "pcc": float(np.nanmean(pearson_per_gene(panel_pred, panel_true))),
            "rmse": rmse(panel_pred, panel_true),
        }
    return out


def aggregate_patient_metrics(per_item_metrics: list[dict[str, float]], patient_ids: list[str]) -> dict[str, dict[str, float]]:
    """"Aggregate primary results first within held-out patients and then
    macro-average across patients. Also report the pooled descriptive
    value, but do not use masks or spots from one patient as if they were
    independent sample replicates. Include confidence intervals at the
    patient level when the number of held-out patients permits them;
    otherwise state that uncertainty is not estimable from one patient."

    per_item_metrics: one dict of metric_name -> value per evaluated item
    (e.g. one mask/hole), patient_ids: the same length, the patient each
    item belongs to. Returns, per metric: patient_mean (mean of
    within-patient means -- the primary reportable number), pooled_mean
    (flat descriptive mean over all items, explicitly NOT the primary
    number), n_patients, patient_ci95 (None with a reason when
    n_patients < 2, since a single patient carries no between-patient
    variance to estimate an interval from).

    Codex re-audit of commit 90f853e, launch blocker #9 ("patient-level
    paired bootstrap or t-based CIs"): the interval is a t-distribution
    interval (`patient_means.std(ddof=1)/sqrt(n) * t.ppf(0.975, df=n-1)`),
    not the previous fixed-z (1.96) normal approximation -- with the
    small held-out patient counts this evaluator realistically runs
    against, the normal approximation understates interval width; the
    t-distribution's wider critical value at low degrees of freedom is
    the honest correction and converges to the same 1.96 as n_patients
    grows large.
    """
    if len(per_item_metrics) != len(patient_ids):
        raise ValueError("per_item_metrics and patient_ids must have the same length")
    if not per_item_metrics:
        raise ValueError("per_item_metrics is empty -- nothing to aggregate")

    metric_names = sorted({key for row in per_item_metrics for key in row})
    patients = sorted(set(patient_ids))
    out: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        pooled_values = np.asarray(
            [row.get(metric_name, np.nan) for row in per_item_metrics], dtype=float,
        )
        patient_means = []
        for patient in patients:
            item_values = np.asarray(
                [row.get(metric_name, np.nan) for row, pid in zip(per_item_metrics, patient_ids) if pid == patient],
                dtype=float,
            )
            finite = item_values[np.isfinite(item_values)]
            if finite.size:
                patient_means.append(float(finite.mean()))
        patient_means_arr = np.asarray(patient_means, dtype=float)
        n_patients = int(patient_means_arr.size)

        pooled_finite = pooled_values[np.isfinite(pooled_values)]
        entry: dict[str, float] = {
            "patient_mean": float(patient_means_arr.mean()) if n_patients else float("nan"),
            "pooled_mean": float(pooled_finite.mean()) if pooled_finite.size else float("nan"),
            "n_patients": n_patients,
            "n_items": int(pooled_finite.size),
        }
        if n_patients >= 2:
            se = float(patient_means_arr.std(ddof=1)) / np.sqrt(n_patients)
            t_crit = float(_student_t.ppf(0.975, df=n_patients - 1))
            entry["patient_ci95_low"] = entry["patient_mean"] - t_crit * se
            entry["patient_ci95_high"] = entry["patient_mean"] + t_crit * se
            entry["ci_estimable"] = 1.0
        else:
            entry["patient_ci95_low"] = float("nan")
            entry["patient_ci95_high"] = float("nan")
            entry["ci_estimable"] = 0.0  # "uncertainty is not estimable from one patient"
        out[metric_name] = entry
    return out


def _bin_by_value(values: np.ndarray, bin_key: np.ndarray, bin_edges: np.ndarray) -> dict[str, dict[str, float]]:
    if values.shape[0] != bin_key.shape[0]:
        raise ValueError("values and bin_key must have the same length")
    bin_idx = np.digitize(bin_key, bin_edges)
    out: dict[str, dict[str, float]] = {}
    for b in range(len(bin_edges) + 1):
        mask = bin_idx == b
        if b == 0:
            label = f"<{bin_edges[0]:g}"
        elif b == len(bin_edges):
            label = f">={bin_edges[-1]:g}"
        else:
            label = f"[{bin_edges[b - 1]:g},{bin_edges[b]:g})"
        selected = values[mask]
        finite = selected[np.isfinite(selected)]
        out[label] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "n": int(mask.sum()),
        }
    return out


def boundary_interior_bins(
    per_query_metric: np.ndarray, query_depth_to_boundary: np.ndarray, bin_edges: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """"Boundary-to-interior performance bins": bins a per-query metric
    (e.g. per-query squared error or per-query correlation contribution)
    by boundary_graph.py's own query_depth_to_boundary BFS hop count --
    depth 0 = directly touches the boundary, larger depth = deeper
    interior. Default edges (0.5, 1.5, 2.5) give bins {0}, {1}, {2},
    {>=3} for the common max_rings=3 depth range."""
    if bin_edges is None:
        bin_edges = np.array([0.5, 1.5, 2.5])
    return _bin_by_value(np.asarray(per_query_metric, dtype=float), np.asarray(query_depth_to_boundary, dtype=float), np.asarray(bin_edges, dtype=float))


def hole_size_bins(
    per_item_metric: np.ndarray, hole_sizes: np.ndarray, bin_edges: np.ndarray,
) -> dict[str, dict[str, float]]:
    """"Hole-size bins": bins a per-item (per-mask/per-hole) metric by
    hole size (e.g. query spot count), matching CONTRACT.md section 4's
    small/medium/large hole-size strata so results can be reported per
    stratum."""
    return _bin_by_value(np.asarray(per_item_metric, dtype=float), np.asarray(hole_sizes, dtype=float), np.asarray(bin_edges, dtype=float))


def edge_gradient_agreement(
    predicted_expression: np.ndarray,
    target_expression: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int = 6,
) -> dict[str, float]:
    """"Query-edge gradient agreement", decomposed into components normal
    and tangential to the hole boundary (Phase 7's explicit evaluation-
    only diagnostic decomposition -- "tests whether expression trends
    continue plausibly into the missing region without adding another
    trainable branch").

    For each query-query graph edge (i, j) (the same k-NN graph
    losses.py's spatial_gradient_loss builds), the edge direction is
    classified as boundary-NORMAL (radially aligned with the hole's own
    centroid -- the direction evidence would flow in from the boundary)
    or boundary-TANGENTIAL (perpendicular to that) by whichever the edge
    direction's absolute dot product with the query midpoint's own radial
    unit vector is larger. This mirrors geometry_utils.py's
    compute_hole_geometry radial-distance-to-centroid convention -- the
    same notion of "hole geometry" used everywhere else in this project,
    not a new one invented for this metric. Reports the Pearson
    correlation between predicted and true per-edge mean expression
    differences within each component, plus the edge count in each.
    """
    from gen3_multiscale.data.boundary_graph import build_knn_adjacency

    n_query = query_coords.shape[0]
    if predicted_expression.shape[0] != n_query or target_expression.shape[0] != n_query:
        raise ValueError("predicted_expression/target_expression must have one row per query coordinate")

    adjacency = build_knn_adjacency(query_coords, k_neighbors=k_neighbors)
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            j = int(j)
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
    if not edges:
        return {
            "normal_pcc": float("nan"), "normal_n_edges": 0,
            "tangential_pcc": float("nan"), "tangential_n_edges": 0,
        }

    centroid = query_coords.mean(axis=0)
    edges_arr = np.asarray(edges, dtype=int)
    i_idx, j_idx = edges_arr[:, 0], edges_arr[:, 1]

    edge_vec = query_coords[j_idx] - query_coords[i_idx]
    edge_len = np.linalg.norm(edge_vec, axis=1, keepdims=True)
    edge_dir = np.divide(edge_vec, edge_len, out=np.zeros_like(edge_vec), where=edge_len > 1e-9)

    midpoint = 0.5 * (query_coords[i_idx] + query_coords[j_idx])
    radial_vec = midpoint - centroid[None, :]
    radial_len = np.linalg.norm(radial_vec, axis=1, keepdims=True)
    radial_dir = np.divide(radial_vec, radial_len, out=np.zeros_like(radial_vec), where=radial_len > 1e-9)

    normal_alignment = np.abs(np.sum(edge_dir * radial_dir, axis=1))
    is_normal = normal_alignment >= 0.5  # closer to radially-aligned than to perpendicular

    pred_diff_mean = (predicted_expression[j_idx] - predicted_expression[i_idx]).mean(axis=1)
    true_diff_mean = (target_expression[j_idx] - target_expression[i_idx]).mean(axis=1)

    def _pcc_or_nan(pred_sub: np.ndarray, true_sub: np.ndarray) -> float:
        if pred_sub.size < 2 or np.std(true_sub) < 1e-8 or np.std(pred_sub) < 1e-8:
            return float("nan")
        return float(pearsonr(pred_sub, true_sub)[0])

    return {
        "normal_pcc": _pcc_or_nan(pred_diff_mean[is_normal], true_diff_mean[is_normal]),
        "normal_n_edges": int(is_normal.sum()),
        "tangential_pcc": _pcc_or_nan(pred_diff_mean[~is_normal], true_diff_mean[~is_normal]),
        "tangential_n_edges": int((~is_normal).sum()),
    }


def spatial_variogram_agreement(
    predicted_expression: np.ndarray,
    target_expression: np.ndarray,
    query_coords: np.ndarray,
    n_bins: int = 8,
    max_pairs: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Empirical semivariance (mean squared expression difference,
    averaged over genes) binned by pairwise spatial distance, computed
    SEPARATELY for the predicted and true fields, then compared --
    "spatial variogram ... agreement" (Phase 7). A model that reconstructs
    the right per-gene values but the wrong spatial arrangement can still
    look good on flat PCC/RMSE; this metric is sensitive to that failure
    mode because it depends on which query points are near which other
    query points; a reasonable fallback given the same detectability gap
    pool_knn_neighborhood's own docstring already documents for
    ST-FID/ST-MMD's plain per-point embeddings above.

    Subsamples pairs (max_pairs, seeded) when the query set is large --
    the full pair count grows quadratically and this is a diagnostic, not
    a loss.
    """
    n_query = query_coords.shape[0]
    if n_query < 2:
        raise ValueError("spatial_variogram_agreement needs at least 2 query points")
    rng = np.random.default_rng(seed)
    all_pairs = np.array([(i, j) for i in range(n_query) for j in range(i + 1, n_query)], dtype=int)
    if all_pairs.shape[0] > max_pairs:
        chosen = rng.choice(all_pairs.shape[0], size=max_pairs, replace=False)
        all_pairs = all_pairs[chosen]
    i_idx, j_idx = all_pairs[:, 0], all_pairs[:, 1]

    distances = np.linalg.norm(query_coords[i_idx] - query_coords[j_idx], axis=1)
    pred_semivar = 0.5 * ((predicted_expression[i_idx] - predicted_expression[j_idx]) ** 2).mean(axis=1)
    true_semivar = 0.5 * ((target_expression[i_idx] - target_expression[j_idx]) ** 2).mean(axis=1)

    bin_edges = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    bin_idx = np.clip(np.digitize(distances, bin_edges[1:-1]), 0, len(bin_edges) - 2)

    pred_curve = np.full(len(bin_edges) - 1, np.nan)
    true_curve = np.full(len(bin_edges) - 1, np.nan)
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if mask.any():
            pred_curve[b] = pred_semivar[mask].mean()
            true_curve[b] = true_semivar[mask].mean()

    valid = np.isfinite(pred_curve) & np.isfinite(true_curve)
    if valid.sum() >= 2 and np.std(pred_curve[valid]) > 1e-8 and np.std(true_curve[valid]) > 1e-8:
        agreement_pcc = float(pearsonr(pred_curve[valid], true_curve[valid])[0])
    else:
        agreement_pcc = float("nan")

    return {
        "agreement_pcc": agreement_pcc,
        "bin_edges": bin_edges.tolist(),
        "predicted_curve": pred_curve.tolist(),
        "true_curve": true_curve.tolist(),
        "n_pairs_used": int(all_pairs.shape[0]),
    }


def graph_laplacian_agreement(
    predicted_expression: np.ndarray, target_expression: np.ndarray, query_coords: np.ndarray, k_neighbors: int = 6,
) -> dict[str, float]:
    """Dirichlet-energy alternative to the variogram above ("spatial
    variogram OR graph-Laplacian agreement" -- the handoff asks for
    either, this project provides both since each is cheap given the
    k-NN graph already built elsewhere): per-gene sum-of-squared-
    differences across the SAME query-query k-NN graph
    losses.py/edge_gradient_agreement use, for the predicted vs true
    fields, reported as a ratio (1.0 = identical smoothness) rather than
    absolute energy so it is comparable across items of different size."""
    from gen3_multiscale.data.boundary_graph import build_knn_adjacency

    adjacency = build_knn_adjacency(query_coords, k_neighbors=k_neighbors)
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            j = int(j)
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
    if not edges:
        return {"predicted_energy": float("nan"), "true_energy": float("nan"), "energy_ratio": float("nan")}
    edges_arr = np.asarray(edges, dtype=int)
    i_idx, j_idx = edges_arr[:, 0], edges_arr[:, 1]
    predicted_energy = float(((predicted_expression[i_idx] - predicted_expression[j_idx]) ** 2).sum())
    true_energy = float(((target_expression[i_idx] - target_expression[j_idx]) ** 2).sum())
    ratio = predicted_energy / true_energy if true_energy > 1e-8 else float("nan")
    return {"predicted_energy": predicted_energy, "true_energy": true_energy, "energy_ratio": ratio}
