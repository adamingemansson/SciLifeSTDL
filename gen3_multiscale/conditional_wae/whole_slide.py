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
from gen3_multiscale.evaluation.metrics import (
    comparable_expression_metrics,
    nonzero_auc,
    resolve_gene_panels,
)
from gen3_multiscale.evaluation.structured_field_metrics import structured_field_metrics


def _whole_slide_inputs_and_context(model: torch.nn.Module, sample):
    """Build the fail-closed H&E-only full-slide input exactly once."""
    inputs, target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    return inputs, target, model.image_conditioner(inputs)


@torch.no_grad()
def predict_deterministic_refinement_stages(
    model: torch.nn.Module, sample, *, chunk_size: int = 2048,
) -> dict:
    """Expose the four deterministic refinement paths on one complete slide.

    ``base`` is the row-wise image decoder, ``within_only`` applies only the
    centered gene-program correction, ``between_only`` applies only the
    spot-graph refiner, and ``full`` uses the trained composition including
    its learned gates.  This is a checkpoint audit, not four separately
    trained models.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if bool(getattr(model, "has_latent_model", True)):
        raise ValueError("refinement-stage inference requires a deterministic model")
    for name in ("decode_base_from_context", "_apply_within", "_apply_between",
                 "refine_prediction"):
        if not hasattr(model, name):
            raise ValueError(f"model does not expose deterministic refinement stage {name}")
    inputs, target, context = _whole_slide_inputs_and_context(model, sample)
    n_rows = int(context.shape[0])
    base = torch.cat([
        model.decode_base_from_context(context[start:min(start + chunk_size, n_rows)])
        for start in range(0, n_rows, chunk_size)
    ], dim=0)
    if base.shape[0] != n_rows:
        raise RuntimeError("whole-slide chunking did not cover every row exactly once")
    within = model._apply_within(base)
    between = model._apply_between(base, context, inputs)
    full = model.refine_prediction(base, context, inputs)
    gates = model.composition_gates() if hasattr(model, "composition_gates") else None
    return {
        "sample_id": inputs.sample_id,
        "coords": np.asarray(sample.full_sample_coords, dtype=np.float32),
        "target": target,
        "n_spots": n_rows,
        "stages": {
            "base": base,
            "within_only": within,
            "between_only": between,
            "full": full,
        },
        "composition_gates": None if gates is None else gates.detach().clone(),
    }


@torch.no_grad()
def predict_whole_slide(
    model: torch.nn.Module, sample, *, chunk_size: int = 2048,
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
    decode_base = getattr(model, "decode_base_from_context", model.conditional_mean_head)
    conditional_mean_chunks = [
        decode_base(context[start:min(start + chunk_size, n_rows)])
        for start in range(0, n_rows, chunk_size)
    ]
    raw_point = torch.cat(conditional_mean_chunks, dim=0)
    point_prediction = (
        model.refine_prediction(raw_point, context, inputs)
        if hasattr(model, "refine_prediction") else raw_point
    )
    if getattr(model, "has_latent_model", True):
        predictive_sum = torch.zeros_like(point_prediction)
        predictive_sum_squared = torch.zeros_like(point_prediction)
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
        predictive_variance = (
            predictive_sum_squared / count - predictive_mean.square()
        ).clamp_min(0.0)
    else:
        predictive_mean = point_prediction
        predictive_variance = torch.zeros_like(point_prediction)

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
        **comparable_expression_metrics(pred, true),
        "auc": nonzero_auc(pred, true),
    }


def whole_slide_metrics(
    prediction: dict, gene_names: list[str], gene_panels: dict[str, list[str]] | None = None,
    *, per_gene_scale: np.ndarray | torch.Tensor | None = None,
    structured_field_config: dict | None = None,
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
    result = {
        "sample_id": prediction["sample_id"],
        "n_spots": prediction["n_spots"],
        "per_arm": per_arm,
        "prediction_roles": {
            "primary_point_prediction": "conditional_mean",
            "wae_prior_predictive_mean": "model",
        },
        "gene_panel_metadata": panel_metadata,
    }
    structured = dict(structured_field_config or {})
    if structured.get("enabled", False):
        if per_gene_scale is None:
            raise ValueError(
                "structured whole-slide metrics require a training-only per_gene_scale"
            )
        if isinstance(per_gene_scale, torch.Tensor):
            per_gene_scale = per_gene_scale.detach().cpu().numpy()
        result["structured_field"] = structured_field_metrics(
            arms["conditional_mean"], true, prediction["coords"], per_gene_scale,
            panel_indices=panel_indices,
            local_k=int(structured.get("local_k", 6)),
            wide_k=int(structured.get("wide_k", 18)),
            nontrivial_threshold=float(
                structured.get("nontrivial_gradient_threshold_training_sd", 0.25)
            ),
        )
    return result
