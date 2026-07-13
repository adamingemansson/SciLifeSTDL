"""
Evaluation metrics for reconstructed ST data.

Includes standard pointwise metrics plus a skeleton for the custom
FID-style distributional fidelity metric (see docs/metrics_notes.md for the
design rationale — read that before extending this).
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from scipy.stats import pearsonr


def pearson_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """[G] array of per-gene Pearson r between predicted and true expression."""
    G = pred.shape[1]
    out = np.zeros(G)
    for g in range(G):
        if np.std(pred[:, g]) < 1e-8 or np.std(true[:, g]) < 1e-8:
            out[g] = np.nan
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


# ---------------------------------------------------------------------------
# Custom FID-style metric
# ---------------------------------------------------------------------------
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
    ST samples (cells, spots, or patches — see metrics_notes.md for the
    "unit of comparison" decision).

    real_embeddings / gen_embeddings: [N, D] arrays from whatever embedding
    function is chosen (PCA/scVI latent / pretrained foundation model /
    custom-trained encoder — this function is agnostic to that choice by
    design, so the embedding function can be swapped and compared).

    IMPORTANT: before trusting this number, run the validation plan in
    docs/metrics_notes.md (monotonicity sanity check, correlation with
    pointwise metrics, sensitivity to spatial-arrangement-only corruption).
    """
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def st_mmd(real_embeddings: np.ndarray, gen_embeddings: np.ndarray,
           gamma: float | None = None) -> float:
    """Alternative to st_fid that drops the Gaussian assumption — RBF-kernel
    Maximum Mean Discrepancy. Compare against st_fid empirically; see
    docs/metrics_notes.md."""
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
