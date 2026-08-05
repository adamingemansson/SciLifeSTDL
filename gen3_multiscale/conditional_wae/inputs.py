"""Leakage-safe inputs for the separate full-H&E-to-GEX task."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FullImageExpressionInputs:
    """One slide or same-slide spot minibatch visible at inference.

    Query expression is deliberately absent. Task II may carry observed GEX
    from non-query rows as a compact ``[M, G]`` matrix plus slide-row indices.
    During training the real query-GEX target is passed separately to
    ``ConditionalWAE.compute_generator_losses``.
    """

    sample_id: str
    image_features: np.ndarray | torch.Tensor  # [N, image_feature_dim]
    coords: np.ndarray | torch.Tensor  # [N, 2], slide-relative/normalized
    image_available: np.ndarray | torch.Tensor  # [N] bool
    query_mask: np.ndarray | torch.Tensor  # [N] bool; locations whose GEX is predicted
    observed_expression: np.ndarray | torch.Tensor | None = None  # [M, G], observed rows only
    observed_expression_indices: np.ndarray | torch.Tensor | None = None  # [M] slide-row indices
    expression_available: np.ndarray | torch.Tensor | None = None  # [N] bool
    neighbor_indices: np.ndarray | torch.Tensor | None = None  # [N, K] cached geometry graph
    neighbor_mask: np.ndarray | torch.Tensor | None = None  # [N, K] bool
    histology_features: np.ndarray | torch.Tensor | None = None  # [N, histology_features.FEATURE_DIM]


def validate_full_image_expression_inputs(inputs: FullImageExpressionInputs) -> None:
    features = torch.as_tensor(inputs.image_features)
    coords = torch.as_tensor(inputs.coords)
    available = torch.as_tensor(inputs.image_available)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("image_features must be a non-empty [N, image_feature_dim] tensor")
    n = features.shape[0]
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be [{n}, 2], got {tuple(coords.shape)}")
    if available.shape != (n,):
        raise ValueError(f"image_available must be [{n}], got {tuple(available.shape)}")
    if available.dtype != torch.bool:
        raise ValueError("image_available must be boolean")
    query_mask = torch.as_tensor(inputs.query_mask)
    if query_mask.shape != (n,) or query_mask.dtype != torch.bool or not query_mask.any():
        raise ValueError("query_mask must be boolean [N] with at least one query")
    if not torch.isfinite(features).all() or not torch.isfinite(coords).all():
        raise ValueError("image_features and coords must be finite")
    if inputs.histology_features is not None:
        histology_features = torch.as_tensor(inputs.histology_features)
        if histology_features.ndim != 2 or histology_features.shape[0] != n:
            raise ValueError(f"histology_features must be [{n}, histology_feature_dim]")
        if not torch.isfinite(histology_features).all():
            raise ValueError("histology_features must be finite")
    if (inputs.neighbor_indices is None) != (inputs.neighbor_mask is None):
        raise ValueError("neighbor_indices and neighbor_mask must be supplied together")
    if inputs.neighbor_indices is not None:
        neighbor_indices = torch.as_tensor(inputs.neighbor_indices)
        neighbor_mask = torch.as_tensor(inputs.neighbor_mask)
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != n:
            raise ValueError("neighbor_indices must be [N, K]")
        if neighbor_mask.shape != neighbor_indices.shape or neighbor_mask.dtype != torch.bool:
            raise ValueError("neighbor_mask must be boolean with neighbor_indices' shape")
        if neighbor_indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("neighbor_indices must be integer")
        active_indices = neighbor_indices[neighbor_mask]
        if active_indices.numel() == 0 or active_indices.min() < 0 or active_indices.max() >= n:
            raise ValueError("active neighbor indices must be non-empty, in-range slide rows")
    unavailable = ~available
    if unavailable.any() and not torch.equal(
        features[unavailable], torch.zeros_like(features[unavailable]),
    ):
        raise ValueError(
            "image_features must be exactly zero where image_available=False; "
            "refusing hidden image content in unavailable rows"
        )
    if inputs.observed_expression is None:
        if inputs.observed_expression_indices is not None:
            raise ValueError("observed_expression_indices require observed_expression")
        if inputs.expression_available is not None and torch.as_tensor(
            inputs.expression_available, dtype=torch.bool,
        ).any():
            raise ValueError("expression_available cannot be true without observed_expression")
        return
    if inputs.expression_available is None:
        raise ValueError("observed_expression requires expression_available")
    if inputs.observed_expression_indices is None:
        raise ValueError("observed_expression requires observed_expression_indices")
    expression = torch.as_tensor(inputs.observed_expression)
    expression_indices = torch.as_tensor(inputs.observed_expression_indices)
    expression_available = torch.as_tensor(inputs.expression_available)
    if expression.ndim != 2:
        raise ValueError("observed_expression must be [M, n_genes]")
    if expression_indices.shape != (expression.shape[0],) or expression_indices.dtype not in {
        torch.int32, torch.int64,
    }:
        raise ValueError("observed_expression_indices must be integer [M]")
    if expression_available.shape != (n,) or expression_available.dtype != torch.bool:
        raise ValueError("expression_available must be boolean [N]")
    if not torch.isfinite(expression).all():
        raise ValueError("observed_expression must be finite")
    if expression_indices.numel() == 0:
        raise ValueError("observed_expression cannot be empty when it is supplied")
    if expression_indices.min() < 0 or expression_indices.max() >= n:
        raise ValueError("observed_expression_indices contain an out-of-range slide row")
    if torch.unique(expression_indices).numel() != expression_indices.numel():
        raise ValueError("observed_expression_indices must be unique")
    if (expression_available & query_mask).any():
        raise ValueError("query GEX cannot be marked available; target-expression leakage detected")
    declared_indices = torch.where(expression_available)[0].to(expression_indices.dtype)
    if not torch.equal(torch.sort(expression_indices).values, declared_indices):
        raise ValueError(
            "observed_expression_indices must exactly match expression_available=True rows"
        )
