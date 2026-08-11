"""How high could ANY model score, given how noisy the measurement is?

Per-gene Pearson against a noisy target is bounded well below 1. If the truth
is ``y = s + e`` with measurement noise ``e`` independent of the signal ``s``,
then even a model that predicts ``s`` perfectly only achieves

    corr(y_hat, y) = corr(y_hat, s) * sqrt( Var(s) / Var(y) )

so the achievable ceiling is ``sqrt(reliability)``, not 1.0. Without that
number a PCC of 0.05 is unreadable: it could be a model failing badly, or a
model at the ceiling on a gene nobody could predict. Measured case that
prompted this: LCN2 is non-zero in 0.75% of INT14's 4,552 spots, and its
whole-slide "target" map is a flat field.

Reliability is estimated by COUNT SPLITTING, which needs no replicates. For a
raw count ``X ~ Poisson(lambda)``, draw ``X1 ~ Binomial(X, 0.5)`` and set
``X2 = X - X1``. Binomial thinning of a Poisson yields ``X1`` and ``X2`` that
are INDEPENDENT given ``lambda`` -- that independence is exact, not an
approximation -- so any correlation between them across spots is shared
signal, never shared noise.

Each half is then pushed through this project's own normalization
(``normalize_total`` + ``log1p``, with the manifest's recorded parameters) so
the estimate lives in the same space the model is scored in. Per gene, the
split-half correlation is Spearman-Brown corrected from half depth back to
full depth, and the ceiling is its square root.

Deliberate detail: the spot set is fixed from the FULL data before splitting,
and no per-cell filter is re-applied to the halves. Splitting halves every
spot's depth, so re-running a ``min_genes`` filter would drop different spots
from each half and the two would no longer describe the same spots.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def split_counts(counts, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Binomial(0.5) thinning of raw counts into two independent halves."""
    dense = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    integral = np.rint(dense)
    if not np.allclose(dense, integral, atol=1e-6):
        raise ValueError(
            "count splitting requires RAW integer counts; these look already "
            "normalized, and thinning a normalized matrix has no Poisson "
            "justification"
        )
    if (integral < 0).any():
        raise ValueError("raw counts must be non-negative")
    first = rng.binomial(integral.astype(np.int64), 0.5)
    return first.astype(np.float64), (integral - first).astype(np.float64)


def _pearson_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=0)
    right_norm = np.linalg.norm(right_centered, axis=0)
    valid = (left_norm > 1e-12) & (right_norm > 1e-12)
    out = np.full(left.shape[1], np.nan, dtype=np.float64)
    out[valid] = (
        (left_centered[:, valid] * right_centered[:, valid]).sum(axis=0)
        / (left_norm[valid] * right_norm[valid])
    )
    return out


def spearman_brown(half_correlation: np.ndarray, *, n_splits: int = 2) -> np.ndarray:
    """Correct a split-half correlation up to full-length reliability.

    Each half carries half the counts and is therefore noisier than the full
    measurement. Reporting the raw split-half correlation as the reliability
    would understate the ceiling.
    """
    corrected = n_splits * half_correlation / (1.0 + (n_splits - 1) * half_correlation)
    return np.clip(corrected, 0.0, 1.0)


def gene_noise_ceiling(counts, *, normalize, rng: np.random.Generator) -> dict:
    """Per-gene achievable-PCC ceiling for one slide's raw count matrix.

    ``normalize(array) -> array`` applies the project's target-space transform
    to a raw count matrix, so this function never re-implements it.
    """
    first, second = split_counts(counts, rng=rng)
    half_correlation = _pearson_columns(normalize(first), normalize(second))
    reliability = spearman_brown(half_correlation)
    ceiling = np.sqrt(reliability)
    scored = ~np.isnan(half_correlation)
    return {
        "split_half_correlation": half_correlation,
        "reliability": reliability,
        "ceiling": ceiling,
        "n_scored_genes": int(scored.sum()),
    }


def normalized_score(observed: np.ndarray, ceiling: np.ndarray, *,
                     minimum_ceiling: float = 0.05) -> np.ndarray:
    """``observed / ceiling``: the fraction of achievable performance reached.

    Genes whose ceiling is below ``minimum_ceiling`` are returned as NaN rather
    than as an enormous ratio. Dividing 0.02 by a ceiling of 0.01 would report
    200% of achievable, which is noise divided by noise -- exactly the kind of
    number that looks like a result and is not one.
    """
    observed = np.asarray(observed, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    if observed.shape != ceiling.shape:
        raise ValueError("observed and ceiling must have the same shape")
    usable = ceiling >= float(minimum_ceiling)
    out = np.full(observed.shape, np.nan, dtype=np.float64)
    out[usable] = observed[usable] / ceiling[usable]
    return out
