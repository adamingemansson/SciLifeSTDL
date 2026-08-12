"""Whole-slide (every-spot) H&E-to-GEX inference for MK validation diagnostics.

H&E stays fully visible; query/surrounding GEX never enters the predictor
(``include_observed_gex=False`` is hardcoded here -- this module is only
meaningful for Task I / he_to_st arms). Every tissue spot is predicted
exactly once. The image conditioner's context is computed ONCE for the
whole slide (correctness requires the full slide's attention neighborhood
together, and the existing dense/sparse-attention switch already bounds its
memory). Decoder work is chunked, but each complete slide prediction is
reassembled BEFORE spatial refinement so whole-slide and masked inference
use the same graph and post-processing. Draws are accumulated online, which
avoids retaining ``n_samples * n_spots * n_genes`` values in GPU memory.
"""
from __future__ import annotations

import numpy as np
import torch

from gen3_multiscale.conditional_wae.data import build_conditional_wae_example
from gen3_multiscale.conditional_wae.model import ConditionalWAE
from gen3_multiscale.evaluation.metrics import (
    nonzero_auc,
    pearson_per_gene,
    resolve_gene_panels,
    rmse,
)


@torch.no_grad()
def predict_whole_slide(
    model: ConditionalWAE, sample, *, chunk_size: int = 2048,
    n_samples: int | None = None, seed: int = 0,
    normalized_coords: np.ndarray | None = None,
    padded_adjacency: tuple[np.ndarray, np.ndarray] | None = None,
    z_mean: torch.Tensor | None = None, z_std: torch.Tensor | None = None,
    latent_spatial_correlation: float = 0.0,
) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    inputs, target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
        normalized_coords=normalized_coords, padded_adjacency=padded_adjacency,
    )
    context = model.image_conditioner(inputs)
    n_rows = context.shape[0]
    count = int(n_samples or model.n_inference_samples)
    if count < 1:
        raise ValueError("n_samples must be positive")
    if model.distributional_head is not None:
        raise ValueError(
            "whole-slide inference currently requires likelihood='gaussian_mse'; "
            "refinement semantics for the distributional head are not defined"
        )

    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    predictive_sum = torch.zeros(
        n_rows, model.n_genes, dtype=context.dtype, device=context.device,
    )
    predictive_sum_squared = torch.zeros_like(predictive_sum)
    for _ in range(count):
        z = model.sample_inference_latent(
            context, generator=generator, z_mean=z_mean, z_std=z_std,
            latent_spatial_correlation=latent_spatial_correlation,
        )
        decoded_chunks = []
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            decoded_chunks.append(model.decode(z[start:end], context[start:end])[0])
        complete_draw = torch.cat(decoded_chunks, dim=0)
        if complete_draw.shape != predictive_sum.shape:
            raise RuntimeError("whole-slide chunking did not cover every row exactly once")
        complete_draw = model.refine_prediction(complete_draw, context, inputs)
        predictive_sum.add_(complete_draw)
        predictive_sum_squared.add_(complete_draw.square())

    predictive_mean = predictive_sum / count
    predictive_variance = (predictive_sum_squared / count - predictive_mean.square()).clamp_min(0.0)
    conditional_mean_chunks = [
        model.conditional_mean_head(context[start:min(start + chunk_size, n_rows)])
        for start in range(0, n_rows, chunk_size)
    ]
    point_prediction = model.refine_prediction(
        torch.cat(conditional_mean_chunks, dim=0), context, inputs,
    )

    return {
        "sample_id": inputs.sample_id,
        "coords": np.asarray(sample.full_sample_coords, dtype=np.float32),
        "target": target,
        "expression": point_prediction,
        "point_prediction": point_prediction,
        "conditional_mean_expression": point_prediction,
        "wae_predictive_mean": predictive_mean,
        "predictive_mean": predictive_mean,
        "predictive_std": predictive_variance.sqrt(),
        "image_context": context,
        "context": context,
        "n_spots": int(n_rows),
    }


def _arm_panel_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "pcc": float(np.nanmean(pearson_per_gene(pred, true))),
        "rmse": rmse(pred, true),
        "auc": nonzero_auc(pred, true),
    }


def whole_slide_metrics(
    prediction: dict, gene_names: list[str], gene_panels: dict[str, list[str]] | None = None,
) -> dict:
    """PCC/RMSE/AUC for one whole-slide prediction, restricted to all genes
    and each configured panel (e.g. CCRCC-50, HVG-50, HVG-200), for both the
    WAE-prior arm (predictive_mean) and deterministic point-prediction arm --
    the same two-arm split conditional_wae_evaluator.py already reports."""
    true = np.asarray(prediction["target"], dtype=np.float32)
    arms = {
        "model": prediction["predictive_mean"].detach().cpu().numpy().astype(np.float32),
        "conditional_mean": prediction["conditional_mean_expression"].detach().cpu().numpy().astype(np.float32),
    }
    panel_indices, panel_metadata = (
        resolve_gene_panels(gene_names, gene_panels) if gene_panels else ({}, {})
    )
    per_arm = {}
    for arm, pred in arms.items():
        entry = {"all_genes": _arm_panel_metrics(pred, true)}
        for panel_name, idx in panel_indices.items():
            entry[panel_name] = _arm_panel_metrics(pred[:, idx], true[:, idx])
        per_arm[arm] = entry
    return {
        "sample_id": prediction["sample_id"],
        "n_spots": prediction["n_spots"],
        "per_arm": per_arm,
        "prediction_roles": {
            "primary_point_prediction": "conditional_mean",
            "wae_prior_predictive_mean": "model",
        },
        "gene_panel_metadata": panel_metadata,
    }
