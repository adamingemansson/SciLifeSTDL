"""
Post-hoc uncertainty calibration for gene-expression predictions
(2026-07-20), inspired by TISSUE (Nature Methods, 2024, bioRxiv
2023.04.25.538326 — confirmed real via direct web search 2026-07-20:
"Transcript Imputation with Spatial Single-cell Uncertainty Estimation",
model-agnostic split-conformal calibration for spatial transcriptomics
predictions).

HONEST GROUNDING NOTE: TISSUE's paper was not directly read — the search
summary describes it as building on split conformal inference "with
several key modifications" for spatial gene expression specifically,
without giving those modifications' exact mechanism. This module
implements STANDARD split conformal regression (Vovk, Gammerman & Shafer
2005, "Algorithmic Learning in a Random World"; Lei, G'Sell, Rinaldo,
Tibshirani & Wasserman 2018, "Distribution-Free Predictive Inference for
Regression", JASA) — a real, well-established, independently verifiable
statistical method, not a guess at TISSUE's own specific spatial
refinements. Same "grounded in a verified general mechanism, not a
guessed literal port" practice already used for _GNNBlock/
_MoMETransformerBlock when a paper's exact internals weren't independently
confirmed.

UNLIKE every other addition this project has made recently, this is
POST-HOC: it wraps an ALREADY-TRAINED model's point predictions with
calibrated intervals. It requires no new training run, no architecture
change, and no config — it consumes (true, predicted) pairs from a
calibration split and (new) predictions to calibrate, computed however
the caller likes (e.g. via an existing trained checkpoint's sample()
calls on two separate held-out masking draws).

Run tests with: python -m tests.test_uncertainty_calibration
"""
import numpy as np


def conformal_calibrate(cal_true: np.ndarray, cal_pred: np.ndarray,
                         alpha: float = 0.1, per_gene: bool = True) -> np.ndarray:
    """Split-conformal calibration margin from a CALIBRATION set (真实值
    and predictions the model has NEVER been directly optimized against
    reconstructing, e.g. a held-out masking draw disjoint from training).

    cal_true, cal_pred: [n_cal, n_genes] real (measured) vs. predicted
    expression on the calibration set.
    alpha: target miscoverage rate (0.1 -> 90% target coverage).
    per_gene: True (default) computes a SEPARATE margin per gene — genes
    have very different expression scales/variance, so one shared margin
    across all genes would systematically under-cover high-variance genes
    and over-cover low-variance ones. False collapses to one shared
    margin (simpler, valid on average, but not per-gene calibrated).

    Returns the margin: array [n_genes] (per_gene=True) or a 0-d scalar
    array (per_gene=False). Use with conformal_predict_interval.

    The quantile level uses the standard finite-sample-exact correction
    ceil((n+1)(1-alpha))/n (Lei et al. 2018, Eq. 2-3), not the naive
    (1-alpha) empirical quantile — the naive version under-covers on
    finite calibration sets; this correction is what makes split
    conformal's coverage guarantee EXACT (under exchangeability) rather
    than merely asymptotic."""
    assert cal_true.shape == cal_pred.shape, (cal_true.shape, cal_pred.shape)
    n = cal_true.shape[0]
    assert n >= 1, "conformal_calibrate requires at least 1 calibration point"
    resid = np.abs(cal_true - cal_pred)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    axis = 0 if per_gene else None
    return np.quantile(resid, q_level, axis=axis)


def conformal_predict_interval(pred: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Applies a margin (from conformal_calibrate) to new predictions,
    forming [lower, upper] intervals. margin broadcasts against pred's
    last axis (per-gene margin) or is a scalar (shared margin)."""
    return pred - margin, pred + margin


def empirical_coverage(true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values actually falling inside [lower, upper] —
    the real diagnostic for whether calibration hit its target. Should
    land close to (1 - alpha) on a held-out TEST set (not the calibration
    set itself, which would trivially/circularly satisfy coverage)."""
    inside = (true >= lower) & (true <= upper)
    return float(inside.mean())
