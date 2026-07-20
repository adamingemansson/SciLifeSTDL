"""Deterministic spatial completion anchors used by baselines and residual models."""
from __future__ import annotations

import torch


def _validate(context_coords: torch.Tensor, context_expression: torch.Tensor,
              query_coords: torch.Tensor) -> None:
    if context_coords.ndim != 2 or query_coords.ndim != 2:
        raise ValueError("coordinates must be rank-2 tensors")
    if context_expression.ndim != 2:
        raise ValueError("context_expression must be [n_context, n_genes]")
    if context_coords.shape[0] != context_expression.shape[0]:
        raise ValueError("context coordinate/expression row counts differ")
    if context_coords.shape[0] == 0:
        raise ValueError("at least one context point is required")


def global_mean_interpolate(context_coords: torch.Tensor, context_expression: torch.Tensor,
                            query_coords: torch.Tensor) -> torch.Tensor:
    _validate(context_coords, context_expression, query_coords)
    return context_expression.mean(dim=0, keepdim=True).expand(query_coords.shape[0], -1)


def nearest_interpolate(context_coords: torch.Tensor, context_expression: torch.Tensor,
                        query_coords: torch.Tensor) -> torch.Tensor:
    _validate(context_coords, context_expression, query_coords)
    nearest = torch.cdist(query_coords, context_coords).argmin(dim=1)
    return context_expression[nearest]


def local_mean_interpolate(context_coords: torch.Tensor, context_expression: torch.Tensor,
                           query_coords: torch.Tensor, k: int = 8) -> torch.Tensor:
    _validate(context_coords, context_expression, query_coords)
    k = max(1, min(int(k), context_coords.shape[0]))
    idx = torch.topk(torch.cdist(query_coords, context_coords), k=k, largest=False, dim=1).indices
    return context_expression[idx].mean(dim=1)


def idw_interpolate(context_coords: torch.Tensor, context_expression: torch.Tensor,
                    query_coords: torch.Tensor, k: int = 8, power: float = 1.0,
                    eps: float = 1e-6) -> torch.Tensor:
    _validate(context_coords, context_expression, query_coords)
    k = max(1, min(int(k), context_coords.shape[0]))
    dist, idx = torch.topk(torch.cdist(query_coords, context_coords), k=k, largest=False, dim=1)
    weights = (dist + eps).pow(-float(power))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.einsum("nk,nkg->ng", weights, context_expression[idx])


def harmonic_interpolate(
    context_coords: torch.Tensor,
    context_expression: torch.Tensor,
    query_coords: torch.Tensor,
    k: int = 8,
    ridge: float = 1e-4,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Graph-harmonic extension with observed context values as boundaries.

    A symmetric k-nearest-neighbour graph is built over context and query
    locations together. Query values solve ``L_qq X_q = -L_qc X_c``. A small
    ridge stabilizes disconnected or nearly singular query components. IDW is
    used only as a numerical fallback.
    """
    _validate(context_coords, context_expression, query_coords)
    n_c, n_q = context_coords.shape[0], query_coords.shape[0]
    if n_q == 0:
        return context_expression.new_empty((0, context_expression.shape[1]))
    all_coords = torch.cat([context_coords, query_coords], dim=0)
    n = all_coords.shape[0]
    k = max(1, min(int(k), n - 1))
    dist = torch.cdist(all_coords, all_coords)
    dist.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = torch.topk(dist, k=k, largest=False, dim=1)

    finite = knn_dist[torch.isfinite(knn_dist)]
    scale = finite.median().clamp_min(eps) if finite.numel() else dist.new_tensor(1.0)
    weights = torch.exp(-((knn_dist / scale) ** 2)).clamp_min(eps)
    W = dist.new_zeros((n, n))
    rows = torch.arange(n, device=dist.device)[:, None].expand_as(knn_idx)
    W[rows, knn_idx] = weights
    W = torch.maximum(W, W.T)
    L = torch.diag(W.sum(dim=1)) - W
    L_qq = L[n_c:, n_c:]
    L_qc = L[n_c:, :n_c]
    eye = torch.eye(n_q, dtype=L.dtype, device=L.device)
    try:
        return torch.linalg.solve(L_qq + float(ridge) * eye, -L_qc @ context_expression)
    except RuntimeError:
        return idw_interpolate(context_coords, context_expression, query_coords, k=k)


def interpolate(mode: str, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
    mode = str(mode).lower()
    if mode in {"global", "global_mean", "mean"}:
        return global_mean_interpolate(context_coords, context_expression, query_coords)
    if mode in {"nearest", "nn"}:
        return nearest_interpolate(context_coords, context_expression, query_coords)
    if mode in {"local", "local_mean", "knn_mean"}:
        return local_mean_interpolate(context_coords, context_expression, query_coords, k=k)
    if mode in {"idw", "inverse_distance"}:
        return idw_interpolate(
            context_coords, context_expression, query_coords, k=k,
            power=float(kwargs.get("power", 1.0)), eps=float(kwargs.get("eps", 1e-6)),
        )
    if mode in {"harmonic", "laplacian"}:
        return harmonic_interpolate(
            context_coords, context_expression, query_coords, k=k,
            ridge=float(kwargs.get("ridge", 1e-4)), eps=float(kwargs.get("eps", 1e-6)),
        )
    raise ValueError(f"unknown interpolation mode {mode!r}")
