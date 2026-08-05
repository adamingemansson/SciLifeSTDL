"""Diagnostic-only summary of a FiLM-conditioned encoder's validation
cohort. Never used for loss or checkpoint selection -- purely for
TensorBoard collapse-watching (posterior spread, gamma/beta activity,
predictive diversity)."""
from __future__ import annotations

import numpy as np
import torch

from gen3_multiscale.conditional_wae.model import ConditionalWAE


def compute_film_diagnostics(
    model: ConditionalWAE, posterior_z: torch.Tensor, context: torch.Tensor,
    predictive_std: torch.Tensor, predictive_mean: torch.Tensor,
    conditional_mean_expression: torch.Tensor,
) -> dict:
    if model.encoder_conditioning != "film":
        raise ValueError("film diagnostics require encoder_conditioning='film'")
    if posterior_z.shape[0] < 2:
        raise ValueError("film diagnostics require at least two posterior_z rows")

    z = posterior_z.detach().cpu().numpy().astype(np.float64)
    centered = z - z.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / (z.shape[0] - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total_variance = float(eigenvalues.sum())
    if total_variance > 0:
        probabilities = eigenvalues / total_variance
        nonzero = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    else:
        effective_rank = 0.0
    active_dimensions = int(np.sum(eigenvalues > 1e-6 * max(float(eigenvalues.max()), 1e-12)))

    encoder = model.expression_encoder
    gamma_beta: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.no_grad():
        if getattr(encoder, "film_first", None) is not None:
            gamma, beta = encoder.film_first(context)
            gamma_beta["first"] = (gamma.detach().cpu().numpy(), beta.detach().cpu().numpy())
        if getattr(encoder, "film_second", None) is not None:
            gamma, beta = encoder.film_second(context)
            gamma_beta["second"] = (gamma.detach().cpu().numpy(), beta.detach().cpu().numpy())

    diff = predictive_mean.detach() - conditional_mean_expression.detach()
    return {
        "effective_rank": effective_rank,
        "active_dimensions": active_dimensions,
        "latent_dim": int(z.shape[1]),
        "latent_dim_mean": z.mean(axis=0),
        "latent_dim_std": z.std(axis=0),
        "gamma_beta": gamma_beta,
        "predictive_std_mean": float(predictive_std.detach().mean().cpu()),
        "stochastic_vs_conditional_mean_diff": float(diff.norm(dim=-1).mean().cpu()),
    }
