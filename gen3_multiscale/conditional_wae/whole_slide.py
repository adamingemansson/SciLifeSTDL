"""Whole-slide (every-spot) H&E-to-GEX inference for MK validation diagnostics.

H&E stays fully visible; query/surrounding GEX never enters the predictor
(``include_observed_gex=False`` is hardcoded here -- this module is only
meaningful for Task I / he_to_st arms). Every tissue spot is predicted
exactly once. The image conditioner's context is computed ONCE for the
whole slide (correctness requires the full slide's attention neighborhood
together, and the existing dense/sparse-attention switch already bounds its
memory); only the per-spot decode step -- which duplicates across the
`n_samples` stochastic draws -- is split into memory-safe chunks. The
latent noise for every spot is drawn in one canonical pass BEFORE chunking,
so results are exactly independent of `chunk_size`.
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

    generator = torch.Generator(device=context.device).manual_seed(int(seed))
    z_all = torch.randn(
        count, n_rows, model.latent_dim,
        dtype=context.dtype, device=context.device, generator=generator,
    )

    predictive_mean_chunks: list[torch.Tensor] = []
    predictive_std_chunks: list[torch.Tensor] = []
    conditional_mean_chunks: list[torch.Tensor] = []
    covered = np.zeros(n_rows, dtype=bool)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        context_chunk = context[start:end]
        draws = [
            model.decode(z_all[draw_index, start:end], context_chunk)[0]
            for draw_index in range(count)
        ]
        stacked = torch.stack(draws)
        predictive_mean_chunks.append(stacked.mean(0))
        predictive_std_chunks.append(stacked.std(0, unbiased=False))
        conditional_mean_chunks.append(model.conditional_mean_head(context_chunk))
        covered[start:end] = True
    if not covered.all():
        raise RuntimeError("whole-slide chunking did not cover every row exactly once")

    return {
        "sample_id": inputs.sample_id,
        "coords": np.asarray(sample.full_sample_coords, dtype=np.float32),
        "target": target,
        "predictive_mean": torch.cat(predictive_mean_chunks, dim=0),
        "predictive_std": torch.cat(predictive_std_chunks, dim=0),
        "conditional_mean_expression": torch.cat(conditional_mean_chunks, dim=0),
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
    stochastic-model arm (predictive_mean) and the conditional-mean arm --
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
        "gene_panel_metadata": panel_metadata,
    }
