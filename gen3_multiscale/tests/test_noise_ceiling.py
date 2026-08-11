"""Tests for the per-gene noise ceiling.

This number will be used to divide reported scores, so a wrong ceiling
silently rescales every result. The gates are therefore about correctness of
the estimate itself, not just that it runs.
"""
import numpy as np
import pytest

from gen3_multiscale.evaluation.noise_ceiling import (
    gene_noise_ceiling,
    normalized_score,
    spearman_brown,
    split_counts,
)


def _log1p_cpm(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    scaled = counts / np.where(totals > 0, totals, 1.0) * target_sum
    return np.log1p(scaled)


def test_split_preserves_the_counts_exactly():
    rng = np.random.default_rng(0)
    counts = rng.poisson(4.0, size=(200, 30)).astype(np.float64)
    first, second = split_counts(counts, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(first + second, counts)
    assert (first >= 0).all() and (second >= 0).all()


def test_split_refuses_already_normalized_input():
    """Thinning a normalized matrix has no Poisson justification, and the
    resulting 'ceiling' would be meaningless."""
    rng = np.random.default_rng(0)
    normalized = _log1p_cpm(rng.poisson(4.0, size=(50, 10)).astype(np.float64))
    with pytest.raises(ValueError, match="RAW integer counts"):
        split_counts(normalized, rng=np.random.default_rng(1))


def test_pure_poisson_noise_gives_a_ceiling_near_zero():
    """Every spot drawn from the SAME rate: there is no spatial signal at all,
    so no model could score above chance and the ceiling must say so."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(5.0, size=(400, 40)).astype(np.float64)
    result = gene_noise_ceiling(counts, normalize=_log1p_cpm, rng=np.random.default_rng(1))
    assert np.nanmean(result["ceiling"]) < 0.25


def test_strong_signal_gives_a_high_ceiling():
    """Strong COMPOSITIONAL signal must yield a ceiling near 1.

    The signal has to be compositional, not depth. Per-spot normalization
    divides each spot by its own total, so across-spot variation that lives
    purely in sequencing depth is removed before the ceiling ever sees it --
    which is also why the pipeline's normalize_total step protects the model
    from learning library size, and why a depth-only fixture here scores
    moderately no matter how strong its rates look.
    """
    rng = np.random.default_rng(0)
    n_spots, n_genes, depth = 400, 40, 4000.0
    profile_a = np.concatenate([np.full(20, 8.0), np.full(20, 1.0)])
    profile_b = np.concatenate([np.full(20, 1.0), np.full(20, 8.0)])
    weights = rng.uniform(size=(n_spots, 1))
    composition = weights * profile_a + (1 - weights) * profile_b
    composition = composition / composition.sum(axis=1, keepdims=True)
    counts = rng.poisson(composition * depth).astype(np.float64)
    result = gene_noise_ceiling(counts, normalize=_log1p_cpm, rng=np.random.default_rng(1))
    assert np.nanmean(result["ceiling"]) > 0.9


def test_ceiling_falls_as_sequencing_depth_falls():
    """The same underlying biology measured at lower depth must yield a lower
    ceiling -- that monotonicity is the whole claim being made."""
    rng = np.random.default_rng(0)
    shape = np.exp(rng.normal(0, 1.0, size=(400, 30)))
    deep = gene_noise_ceiling(
        rng.poisson(shape * 50.0).astype(np.float64),
        normalize=_log1p_cpm, rng=np.random.default_rng(1),
    )
    shallow = gene_noise_ceiling(
        rng.poisson(shape * 1.0).astype(np.float64),
        normalize=_log1p_cpm, rng=np.random.default_rng(1),
    )
    assert np.nanmean(deep["ceiling"]) > np.nanmean(shallow["ceiling"]) + 0.15


def test_spearman_brown_corrects_upward_and_stays_bounded():
    half = np.array([0.0, 0.2, 0.5, 1.0])
    corrected = spearman_brown(half)
    assert np.all(corrected >= half)
    assert corrected[0] == pytest.approx(0.0)
    assert corrected[2] == pytest.approx(2 / 3)
    assert corrected[3] == pytest.approx(1.0)
    # A negative split-half correlation is noise, not evidence of reliability.
    assert spearman_brown(np.array([-0.3]))[0] == pytest.approx(0.0)


def test_normalized_score_refuses_to_divide_by_an_unusable_ceiling():
    observed = np.array([0.02, 0.30, 0.40])
    ceiling = np.array([0.01, 0.60, 0.80])
    scores = normalized_score(observed, ceiling, minimum_ceiling=0.05)
    assert np.isnan(scores[0]), "0.02/0.01 would report 200% of achievable"
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(0.5)
