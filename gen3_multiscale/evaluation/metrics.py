"""Evaluation metrics for the multiscale spatial-field architectures --
Phase 7 of the handoff ("Add losses, metrics, and diagnostic
interventions").

The eight pointwise/distributional functions below (through
pool_knn_neighborhood) originate from gen2_architectures/evaluation/
metrics.py.  `pearson_per_gene` deliberately uses an equivalent
vectorized implementation here: full-panel Gen3 evaluation otherwise
made millions of scalar scipy calls and emitted one warning per constant
gene.  The eligibility/failure semantics remain unchanged.

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
from scipy.stats import pearsonr, rankdata, t as _student_t


# ---------------------------------------------------------------------------
# Adapted from gen2_architectures/evaluation/metrics.py.  Pearson is
# deliberately vectorized; the remaining functions retain the reused logic.
# ---------------------------------------------------------------------------
def pearson_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-gene PCC with eligibility defined from ground truth only.

    A truth-constant gene is not identifiable within the evaluated region and
    is returned as NaN. A constant prediction for a truth-variable gene is a
    model failure and receives 0 instead of disappearing from the mean.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(
            f"pred and true must be matching 2-D [n_items, n_genes] arrays; "
            f"got {pred.shape} and {true.shape}"
        )
    if pred.shape[0] == 0:
        raise ValueError("cannot compute per-gene PCC over zero items")

    # Computing one scipy.stats.pearsonr call per gene made full-panel
    # evaluation perform millions of Python/scipy calls.  The centered-dot
    # formula is the same Pearson correlation, evaluated for every gene in
    # two vectorized passes.  Float64 accumulation prevents large-offset
    # float32 values from being misclassified as constant through rounding.
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    true_centered = true - true.mean(axis=0, keepdims=True)
    pred_ss = np.einsum("ng,ng->g", pred_centered, pred_centered)
    true_ss = np.einsum("ng,ng->g", true_centered, true_centered)
    pred_std = np.sqrt(pred_ss / pred.shape[0])
    true_std = np.sqrt(true_ss / true.shape[0])

    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    truth_variable = true_std >= 1e-8
    prediction_constant = truth_variable & (pred_std < 1e-8)
    out[prediction_constant] = 0.0

    eligible = truth_variable & ~prediction_constant
    if np.any(eligible):
        covariance = np.einsum(
            "ng,ng->g", pred_centered[:, eligible], true_centered[:, eligible]
        )
        denominator = np.sqrt(pred_ss[eligible] * true_ss[eligible])
        out[eligible] = np.clip(covariance / denominator, -1.0, 1.0)
    return out


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error in the evaluator's declared expression space."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must be matching non-empty arrays")
    return float(np.mean(np.square(pred - true)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error in the evaluator's declared expression space."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must be matching non-empty arrays")
    return float(np.mean(np.abs(pred - true)))


def spearman_per_gene(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 256,
) -> np.ndarray:
    """Per-gene Spearman correlation with PCC-compatible failure semantics.

    Ranking is performed in gene chunks so whole-slide/full-panel evaluation
    does not materialize two additional ``[n_spots, n_genes]`` float64 arrays.
    Truth-constant genes are ineligible (NaN); a constant prediction for a
    truth-variable gene is a model failure (0), matching ``pearson_per_gene``.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
        raise ValueError(
            f"pred and true must be matching 2-D [n_items, n_genes] arrays; "
            f"got {pred.shape} and {true.shape}"
        )
    if pred.shape[0] == 0:
        raise ValueError("cannot compute per-gene Spearman over zero items")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for start in range(0, pred.shape[1], chunk_size):
        end = min(start + chunk_size, pred.shape[1])
        pred_rank = rankdata(pred[:, start:end], axis=0, method="average")
        true_rank = rankdata(true[:, start:end], axis=0, method="average")
        out[start:end] = pearson_per_gene(pred_rank, true_rank)
    return out


def r2_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-gene coefficient of determination across spots.

    A truth-constant gene has no defined R2 and is returned as NaN. Negative
    values are retained because they are important evidence that the model is
    worse than predicting the held-out gene mean.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
        raise ValueError("pred and true must be matching 2-D arrays")
    residual = np.square(true - pred).sum(axis=0)
    centered = true - true.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(axis=0)
    out = np.full(true.shape[1], np.nan, dtype=np.float64)
    valid = total > 1e-12
    out[valid] = 1.0 - residual[valid] / total[valid]
    return out


def gene_correlation_summary(per_gene_pcc: np.ndarray) -> dict[str, float]:
    """Flat per-gene PCC distribution summary for report aggregation.

    Mean PCC alone can hide a model that predicts a small subset of genes
    extremely well while failing broadly. These fields deliberately stay
    scalar/flat so the existing patient-macro aggregator can consume them.
    """
    values = np.asarray(per_gene_pcc, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "median_gene_pcc": float("nan"),
            "gene_pcc_q25": float("nan"),
            "gene_pcc_q75": float("nan"),
            "fraction_gene_pcc_gt_0": float("nan"),
            "fraction_gene_pcc_gt_0_1": float("nan"),
            "fraction_gene_pcc_gt_0_2": float("nan"),
            "fraction_gene_pcc_gt_0_3": float("nan"),
            "n_valid_genes": 0,
        }
    return {
        "median_gene_pcc": float(np.median(finite)),
        "gene_pcc_q25": float(np.quantile(finite, 0.25)),
        "gene_pcc_q75": float(np.quantile(finite, 0.75)),
        "fraction_gene_pcc_gt_0": float(np.mean(finite > 0.0)),
        "fraction_gene_pcc_gt_0_1": float(np.mean(finite > 0.1)),
        "fraction_gene_pcc_gt_0_2": float(np.mean(finite > 0.2)),
        "fraction_gene_pcc_gt_0_3": float(np.mean(finite > 0.3)),
        "n_valid_genes": int(finite.size),
    }


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Return a compact mean/median/IQR/count summary for a metric vector."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"mean_{prefix}": float("nan"),
            f"median_{prefix}": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"n_valid_{prefix}": 0,
        }
    return {
        f"mean_{prefix}": float(np.mean(finite)),
        f"median_{prefix}": float(np.median(finite)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"n_valid_{prefix}": int(finite.size),
    }


def pooled_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """PCC after flattening spots and genes, without a full centered copy.

    This is deliberately a secondary diagnostic: it is dominated by gene
    abundance/mean differences and must not replace mean per-gene PCC.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape or pred.size == 0:
        raise ValueError("pred and true must be matching non-empty arrays")
    n = float(pred.size)
    sum_pred = float(np.sum(pred, dtype=np.float64))
    sum_true = float(np.sum(true, dtype=np.float64))
    pred_ss = float(np.sum(np.square(pred, dtype=np.float64), dtype=np.float64)) - sum_pred ** 2 / n
    true_ss = float(np.sum(np.square(true, dtype=np.float64), dtype=np.float64)) - sum_true ** 2 / n
    if true_ss <= 1e-12:
        return float("nan")
    if pred_ss <= 1e-12:
        return 0.0
    cross = float(np.sum(np.multiply(pred, true, dtype=np.float64), dtype=np.float64))
    covariance = cross - sum_pred * sum_true / n
    return float(np.clip(covariance / np.sqrt(pred_ss * true_ss), -1.0, 1.0))


def spot_profile_pcc_per_spot(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 256,
) -> np.ndarray:
    """PCC across genes inside every spot.

    This complements per-gene PCC across spots: it asks whether the relative
    expression profile within a location is correct, not whether a gene is
    spatially localized correctly across the tissue.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return np.concatenate([
        pearson_per_gene(pred[start:end].T, true[start:end].T)
        for start in range(0, pred.shape[0], chunk_size)
        for end in [min(start + chunk_size, pred.shape[0])]
    ])


def normalized_mutual_information_per_gene(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 64,
) -> np.ndarray:
    """HEtoSGEBench-style per-gene normalized mutual information.

    Prediction and truth are discretized separately into equal-frequency
    bins. MI is normalized by the geometric mean of their entropies. A
    truth-constant gene is ineligible; a constant prediction scores zero.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    n_items, n_genes = pred.shape
    n_bins = min(n_items, max(10, int(n_items ** (1.0 / 3.0))))
    out = np.full(n_genes, np.nan, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        pred_rank = rankdata(pred[:, start:end], axis=0, method="average")
        true_rank = rankdata(true[:, start:end], axis=0, method="average")
        pred_bin = np.minimum(
            ((pred_rank - 1.0) * n_bins / n_items).astype(np.int64), n_bins - 1,
        )
        true_bin = np.minimum(
            ((true_rank - 1.0) * n_bins / n_items).astype(np.int64), n_bins - 1,
        )
        pred_hot = np.eye(n_bins, dtype=np.float64)[pred_bin]
        true_hot = np.eye(n_bins, dtype=np.float64)[true_bin]
        joint = np.einsum("ncb,ncd->cbd", pred_hot, true_hot) / n_items
        px = joint.sum(axis=2)
        py = joint.sum(axis=1)
        expected = px[:, :, None] * py[:, None, :]
        nonzero = joint > 0
        ratio = np.divide(joint, expected, out=np.ones_like(joint), where=nonzero)
        mi = np.sum(np.where(nonzero, joint * np.log(ratio), 0.0), axis=(1, 2))
        hx = -np.sum(np.where(px > 0, px * np.log(np.maximum(px, 1e-300)), 0.0), axis=1)
        hy = -np.sum(np.where(py > 0, py * np.log(np.maximum(py, 1e-300)), 0.0), axis=1)
        denominator = np.sqrt(hx * hy)
        block = np.full(end - start, np.nan, dtype=np.float64)
        truth_variable = np.ptp(true[:, start:end], axis=0) >= 1e-8
        pred_variable = np.ptp(pred[:, start:end], axis=0) >= 1e-8
        valid = truth_variable & pred_variable & (denominator > 1e-12)
        block[valid] = np.clip(mi[valid] / denominator[valid], 0.0, 1.0)
        block[truth_variable & ~pred_variable] = 0.0
        out[start:end] = block
    return out


def jensen_shannon_divergence_per_gene(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 256,
) -> np.ndarray:
    """Per-gene Jensen-Shannon divergence across spots (base 2, 0--1)."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for start in range(0, pred.shape[1], chunk_size):
        end = min(start + chunk_size, pred.shape[1])
        p = np.maximum(np.asarray(pred[:, start:end], dtype=np.float64), 0.0)
        q = np.maximum(np.asarray(true[:, start:end], dtype=np.float64), 0.0)
        p_sum = p.sum(axis=0, keepdims=True)
        q_sum = q.sum(axis=0, keepdims=True)
        p = np.divide(p, p_sum, out=np.full_like(p, 1.0 / len(p)), where=p_sum > 0)
        q = np.divide(q, q_sum, out=np.full_like(q, 1.0 / len(q)), where=q_sum > 0)
        midpoint = 0.5 * (p + q)
        p_ratio = np.divide(p, midpoint, out=np.ones_like(p), where=p > 0)
        q_ratio = np.divide(q, midpoint, out=np.ones_like(q), where=q > 0)
        p_term = np.where(p > 0, p * np.log2(np.maximum(p_ratio, 1e-300)), 0.0)
        q_term = np.where(q > 0, q * np.log2(np.maximum(q_ratio, 1e-300)), 0.0)
        out[start:end] = np.clip(0.5 * (p_term.sum(axis=0) + q_term.sum(axis=0)), 0.0, 1.0)
    return out


def normalized_rmse_per_gene(
    pred: np.ndarray, true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene RMSE normalized by truth range and truth standard deviation."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    error = np.sqrt(np.mean(np.square(pred - true), axis=0))
    value_range = np.ptp(true, axis=0)
    standard_deviation = np.std(true, axis=0, ddof=1) if len(true) > 1 else np.zeros(true.shape[1])
    by_range = np.full(true.shape[1], np.nan, dtype=np.float64)
    by_sd = np.full(true.shape[1], np.nan, dtype=np.float64)
    np.divide(error, value_range, out=by_range, where=value_range >= 1e-8)
    np.divide(error, standard_deviation, out=by_sd, where=standard_deviation >= 1e-8)
    return by_range, by_sd


def benchmark_vector_ssim_per_gene(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 256,
) -> np.ndarray:
    """Coordinate-free vector SSIM compatible with HEtoSGEBench.

    This is intentionally named separately from our coordinate-aware Visium
    raster SSIM. Each gene vector is independently mapped to 256 intensity
    bins before the global SSIM formula is applied across spots.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    for start in range(0, pred.shape[1], chunk_size):
        end = min(start + chunk_size, pred.shape[1])
        p = np.asarray(pred[:, start:end], dtype=np.float64)
        q = np.asarray(true[:, start:end], dtype=np.float64)
        # Mirror the reference implementation's first normalization step.
        # The subsequent equal-width discretization makes this redundant for
        # ordinary non-negative expression, but retaining it makes the metric
        # contract explicit and reproducible.
        p_max = p.max(axis=0)
        q_max = q.max(axis=0)
        p = np.divide(p, p_max, out=np.zeros_like(p), where=np.abs(p_max) >= 1e-12)
        q = np.divide(q, q_max, out=np.zeros_like(q), where=np.abs(q_max) >= 1e-12)
        p_range = np.ptp(p, axis=0)
        q_range = np.ptp(q, axis=0)
        p_scaled = np.divide(
            p - p.min(axis=0), p_range,
            out=np.zeros_like(p), where=p_range >= 1e-8,
        ) * 255.0
        q_scaled = np.divide(
            q - q.min(axis=0), q_range,
            out=np.zeros_like(q), where=q_range >= 1e-8,
        ) * 255.0
        p_scaled = np.floor(p_scaled)
        q_scaled = np.floor(q_scaled)
        p_mean, q_mean = p_scaled.mean(axis=0), q_scaled.mean(axis=0)
        ddof = 1 if len(p_scaled) > 1 else 0
        p_var = p_scaled.var(axis=0, ddof=ddof)
        q_var = q_scaled.var(axis=0, ddof=ddof)
        covariance = np.mean(
            (p_scaled - p_mean) * (q_scaled - q_mean), axis=0,
        ) * (len(p_scaled) / max(len(p_scaled) - 1, 1))
        score = (
            (2.0 * p_mean * q_mean + c1) * (2.0 * covariance + c2)
            / ((p_mean ** 2 + q_mean ** 2 + c1) * (p_var + q_var + c2))
        )
        score[q_range < 1e-8] = np.nan
        out[start:end] = np.clip(score, -1.0, 1.0)
    return out


def nonzero_auc_per_gene(
    pred: np.ndarray, true: np.ndarray, *, chunk_size: int = 256,
) -> np.ndarray:
    """Per-gene zero/nonzero AUC using a vectorized Mann--Whitney statistic."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.ndim != 2 or pred.shape != true.shape or pred.shape[0] == 0:
        raise ValueError("pred and true must be matching non-empty 2-D arrays")
    out = np.full(pred.shape[1], np.nan, dtype=np.float64)
    for start in range(0, pred.shape[1], chunk_size):
        end = min(start + chunk_size, pred.shape[1])
        positive = true[:, start:end] > 0
        n_positive = positive.sum(axis=0).astype(np.float64)
        n_negative = len(true) - n_positive
        ranks = rankdata(pred[:, start:end], axis=0, method="average")
        rank_sum = np.sum(ranks * positive, axis=0)
        valid = (n_positive > 0) & (n_negative > 0)
        block = np.full(end - start, np.nan, dtype=np.float64)
        block[valid] = (
            rank_sum[valid] - n_positive[valid] * (n_positive[valid] + 1.0) / 2.0
        ) / (n_positive[valid] * n_negative[valid])
        out[start:end] = np.clip(block, 0.0, 1.0)
    return out


def comparable_expression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Shared literature-facing point metric contract.

    PCC and Spearman are gene-wise across spots. RMSE/MAE are element-wise in
    the caller's declared target space. The returned distribution fields make
    broad versus few-gene performance visible without storing a large vector
    in every mask/slide record.
    """
    per_gene_pcc = pearson_per_gene(pred, true)
    per_gene_spearman = spearman_per_gene(pred, true)
    per_gene_r2 = r2_per_gene(pred, true)
    per_spot_pcc = spot_profile_pcc_per_spot(pred, true)
    per_gene_nmi = normalized_mutual_information_per_gene(pred, true)
    per_gene_js = jensen_shannon_divergence_per_gene(pred, true)
    per_gene_nrmse_range, per_gene_nrmse_sd = normalized_rmse_per_gene(pred, true)
    per_gene_benchmark_ssim = benchmark_vector_ssim_per_gene(pred, true)
    per_gene_auc = nonzero_auc_per_gene(pred, true)
    valid_pcc = np.isfinite(per_gene_pcc)
    valid_spearman = np.isfinite(per_gene_spearman)
    mean_profile_pred = np.mean(pred, axis=0).reshape(-1, 1)
    mean_profile_true = np.mean(true, axis=0).reshape(-1, 1)
    expression_profile_pcc = pearson_per_gene(mean_profile_pred, mean_profile_true)[0]
    expression_profile_spearman = pearson_per_gene(
        rankdata(mean_profile_pred, axis=0), rankdata(mean_profile_true, axis=0),
    )[0]
    gene_pcc = float(np.mean(per_gene_pcc[valid_pcc])) if valid_pcc.any() else float("nan")
    gene_spearman = (
        float(np.mean(per_gene_spearman[valid_spearman]))
        if valid_spearman.any() else float("nan")
    )
    return {
        # Backward-compatible names: these have always meant the mean of
        # per-gene correlations across spots.
        "pcc": gene_pcc,
        "spearman": gene_spearman,
        "mean_gene_pcc": gene_pcc,
        "mean_gene_spearman": gene_spearman,
        "pooled_pcc": pooled_pearson(pred, true),
        "mean_expression_profile_pcc": float(expression_profile_pcc),
        "mean_expression_profile_spearman": float(expression_profile_spearman),
        "rmse": rmse(pred, true),
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "r2": float(np.nanmean(per_gene_r2)) if np.isfinite(per_gene_r2).any() else float("nan"),
        "median_gene_r2": (
            float(np.nanmedian(per_gene_r2))
            if np.isfinite(per_gene_r2).any() else float("nan")
        ),
        "fraction_gene_r2_gt_0": (
            float(np.mean(per_gene_r2[np.isfinite(per_gene_r2)] > 0.0))
            if np.isfinite(per_gene_r2).any() else float("nan")
        ),
        **_finite_summary(per_spot_pcc, "spot_profile_pcc"),
        **_finite_summary(per_gene_nmi, "gene_nmi"),
        **_finite_summary(per_gene_js, "gene_js_divergence"),
        **_finite_summary(per_gene_nrmse_range, "gene_nrmse_range"),
        **_finite_summary(per_gene_nrmse_sd, "gene_nrmse_sd"),
        **_finite_summary(per_gene_benchmark_ssim, "benchmark_gene_ssim"),
        **_finite_summary(per_gene_auc, "gene_nonzero_auc"),
        **_finite_summary(per_gene_spearman, "gene_spearman"),
        **gene_correlation_summary(per_gene_pcc),
    }


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
    """Comparable point metrics restricted to each evaluation-only panel."""
    indices, _metadata = resolve_gene_panels(gene_names, gene_panels)
    out: dict[str, dict[str, float]] = {}
    for panel_name, idx in indices.items():
        panel_pred, panel_true = pred[:, idx], true[:, idx]
        out[panel_name] = comparable_expression_metrics(panel_pred, panel_true)
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
