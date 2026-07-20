"""
Regression tests for src/evaluation/uncertainty.py (2026-07-20) — post-hoc
split-conformal calibration for gene-expression predictions, inspired by
TISSUE (Nature Methods 2024, bioRxiv 2023.04.25.538326, confirmed real via
direct web search). See uncertainty.py's own module docstring for the
honest grounding note: this implements standard split conformal
(Vovk/Gammerman/Shafer 2005; Lei et al. 2018), not a guess at TISSUE's own
unverified spatial-specific refinements.

Run with: python -m tests.test_uncertainty_calibration
"""
import numpy as np

from src.evaluation.uncertainty import (
    conformal_calibrate, conformal_predict_interval, empirical_coverage,
)


def test_conformal_calibrate_achieves_target_coverage_on_held_out_test_set():
    """The real statistical property split conformal guarantees: margins
    fit on a CALIBRATION set, applied to a SEPARATE TEST set drawn from
    the same distribution, should cover close to (1-alpha) of the time —
    not exactly (finite-sample noise), but close, and NOT dramatically
    off (which would mean the implementation is wrong, e.g. missing the
    finite-sample quantile correction)."""
    rng = np.random.default_rng(0)
    n_cal, n_test, n_genes = 2000, 2000, 5
    true_mean = rng.uniform(1, 10, size=n_genes)
    noise_std = rng.uniform(0.5, 3.0, size=n_genes)  # genes have DIFFERENT variance

    def draw(n):
        true = true_mean + rng.normal(0, 1, size=(n, n_genes)) * 0  # true "signal" constant per gene for this synthetic test
        true = np.broadcast_to(true_mean, (n, n_genes)).copy()
        pred = true + rng.normal(0, 1, size=(n, n_genes)) * noise_std
        return true, pred

    cal_true, cal_pred = draw(n_cal)
    test_true, test_pred = draw(n_test)

    alpha = 0.1
    margin = conformal_calibrate(cal_true, cal_pred, alpha=alpha, per_gene=True)
    assert margin.shape == (n_genes,)
    lower, upper = conformal_predict_interval(test_pred, margin)
    coverage = empirical_coverage(test_true, lower, upper)
    # target 90% -- allow real finite-sample slack, but must be close, not wildly off
    assert 0.85 <= coverage <= 0.95, f"coverage {coverage} far from target 0.90"
    print(f"[uncertainty] OK — per-gene split conformal achieves {coverage:.3f} coverage "
          f"(target 0.90) on a held-out test set")


def test_conformal_calibrate_per_gene_differs_from_shared():
    """Genes with very different residual scales must get DIFFERENT
    per-gene margins — proves per_gene=True isn't silently collapsing to
    one shared value."""
    rng = np.random.default_rng(1)
    n_cal = 500
    cal_true = np.zeros((n_cal, 2))
    cal_pred = np.stack([
        rng.normal(0, 0.1, size=n_cal),   # tiny-residual gene
        rng.normal(0, 5.0, size=n_cal),   # huge-residual gene
    ], axis=1)
    margin = conformal_calibrate(cal_true, cal_pred, alpha=0.1, per_gene=True)
    assert margin.shape == (2,)
    assert margin[1] > 10 * margin[0], (
        f"high-variance gene's margin ({margin[1]}) should be much larger than "
        f"low-variance gene's ({margin[0]}) -- per-gene calibration isn't working"
    )
    print(f"[uncertainty] OK — per-gene margins genuinely differ by scale: {margin}")


def test_conformal_calibrate_shared_margin_is_scalar():
    rng = np.random.default_rng(2)
    cal_true = np.zeros((300, 3))
    cal_pred = rng.normal(0, 1, size=(300, 3))
    margin = conformal_calibrate(cal_true, cal_pred, alpha=0.1, per_gene=False)
    assert margin.shape == (), margin.shape
    print("[uncertainty] OK — per_gene=False returns a single shared scalar margin")


def test_empirical_coverage_trivial_cases():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 0.5, 0.5, 0.5])
    upper = np.array([1.5, 1.5, 1.5, 1.5])  # only the first point (1.0) is covered
    cov = empirical_coverage(true, lower, upper)
    assert abs(cov - 0.25) < 1e-9
    print("[uncertainty] OK — empirical_coverage counts correctly on a hand-checked case")


def test_finite_sample_quantile_correction_matches_lei_et_al_formula():
    """Direct check that conformal_calibrate uses the EXACT finite-sample
    correction ceil((n+1)(1-alpha))/n (Lei et al. 2018), not the naive
    (1-alpha) empirical quantile -- the naive version systematically
    under-covers, which is the real, well-known bug this correction
    exists to avoid."""
    rng = np.random.default_rng(3)
    n, alpha = 19, 0.1
    cal_true = np.zeros((n, 1))
    cal_pred = rng.normal(0, 1, size=(n, 1))
    resid = np.abs(cal_true - cal_pred)

    expected_q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # = ceil(18)/19 = 18/19
    expected_margin = np.quantile(resid, expected_q_level, axis=0)

    margin = conformal_calibrate(cal_true, cal_pred, alpha=alpha, per_gene=True)
    assert np.allclose(margin, expected_margin)
    print(f"[uncertainty] OK — quantile level matches Lei et al. 2018's exact "
          f"finite-sample correction (q_level={expected_q_level:.4f})")


if __name__ == "__main__":
    test_conformal_calibrate_achieves_target_coverage_on_held_out_test_set()
    test_conformal_calibrate_per_gene_differs_from_shared()
    test_conformal_calibrate_shared_margin_is_scalar()
    test_empirical_coverage_trivial_cases()
    test_finite_sample_quantile_correction_matches_lei_et_al_formula()
    print("\nAll uncertainty calibration tests passed.")
