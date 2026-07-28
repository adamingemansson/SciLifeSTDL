"""Shared deterministic training objective -- Phase 7 of the multiscale
spatial-field handoff ("Add losses, metrics, and diagnostic
interventions").

"Use the same primary reconstruction objective for all four. Add one weak
spatial-gradient objective on query graph edges that compares predicted
differences with true differences ... Use a small common weight, initially
0.05 after scale normalization. This loss must match gradients rather than
force neighbouring spots to be identical." Both pieces are deliberately
architecture-agnostic plain functions (predicted/target tensors in,
scalar out) -- every one of Architectures 1-4 calls the SAME two
functions, never a per-architecture reimplementation, matching the
handoff's "Keep this deliberately simple" instruction.

The primary objective is plain MSE between predicted and target
full-gene expression, matching the only loss actually exercised so far
(Phase 6's gradient-flow tests) -- CONTRACT.md flags this explicitly as
"No losses beyond a plain MSE" prior to this phase.
"""
from __future__ import annotations

import torch

from gen3_multiscale.data.boundary_graph import build_knn_adjacency


def primary_reconstruction_loss(predicted_expression: torch.Tensor, target_expression: torch.Tensor) -> torch.Tensor:
    """Plain MSE over the query spot-by-gene matrix -- the shared
    deterministic objective every architecture's transport-head output is
    trained against."""
    if predicted_expression.shape != target_expression.shape:
        raise ValueError(
            f"predicted_expression {tuple(predicted_expression.shape)} and target_expression "
            f"{tuple(target_expression.shape)} must have the same shape"
        )
    return torch.nn.functional.mse_loss(predicted_expression, target_expression)


def _query_graph_edges(query_coords: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    """[E, 2] int64 tensor of (i, j) query-query graph edges (i < j, each
    unordered pair listed once) from the same k-NN adjacency
    boundary_graph.py uses everywhere else in this project, so "query
    graph edges" means the identical graph in the loss, the metrics, and
    the model's own query-query self-attention -- never three different
    ad hoc notions of adjacency."""
    coords_np = query_coords.detach().cpu().numpy()
    adjacency = build_knn_adjacency(coords_np, k_neighbors=k_neighbors)
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            j = int(j)
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long, device=query_coords.device)
    return torch.as_tensor(edges, dtype=torch.long, device=query_coords.device)


def spatial_gradient_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    query_coords: torch.Tensor,
    k_neighbors: int = 6,
    per_gene_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted and true STANDARDIZED expression differences
    across query-query graph edges: "compares predicted differences with
    true differences", i.e. matches gradients, never absolute levels --
    two adjacent queries with a genuine sharp biological boundary are not
    penalized for disagreeing, only for disagreeing about how MUCH and in
    which direction expression changes across the edge.

    per_gene_scale: [n_genes] positive per-gene standardization scale
    (e.g. training-set expression std -- "after scale normalization").
    When None (the common case until a real data builder supplies a
    training-fit scale), this function falls back to the per-gene std of
    target_expression WITHIN THIS CALL -- a documented simplification
    (mirrors harmonic.py's/target_gene_scale's identical "training-only
    fitting must happen outside this function" limitation), not a claim
    that this is the training-set scale.
    """
    if predicted_expression.shape != target_expression.shape:
        raise ValueError(
            f"predicted_expression {tuple(predicted_expression.shape)} and target_expression "
            f"{tuple(target_expression.shape)} must have the same shape"
        )
    n_query = query_coords.shape[0]
    if predicted_expression.shape[0] != n_query:
        raise ValueError("predicted_expression/target_expression must have one row per query coordinate")

    edges = _query_graph_edges(query_coords, k_neighbors)
    if edges.shape[0] == 0:
        return torch.zeros((), device=predicted_expression.device, dtype=predicted_expression.dtype)

    if per_gene_scale is None:
        scale = target_expression.std(dim=0).clamp_min(1e-6)
    else:
        scale = per_gene_scale.clamp_min(1e-6)

    i_idx, j_idx = edges[:, 0], edges[:, 1]
    pred_diff = (predicted_expression[i_idx] - predicted_expression[j_idx]) / scale
    true_diff = (target_expression[i_idx] - target_expression[j_idx]) / scale
    return torch.nn.functional.mse_loss(pred_diff, true_diff)


def combined_reconstruction_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    query_coords: torch.Tensor,
    gradient_weight: float = 0.05,
    k_neighbors: int = 6,
    per_gene_scale: torch.Tensor | None = None,
) -> dict:
    """The shared deterministic objective, assembled: primary + a small
    weight (default 0.05, the handoff's stated initial value) times the
    spatial-gradient term. Returns every component (not just the total)
    so a training loop can log them separately without recomputing."""
    if gradient_weight < 0:
        raise ValueError("gradient_weight must be non-negative")
    primary = primary_reconstruction_loss(predicted_expression, target_expression)
    gradient = spatial_gradient_loss(
        predicted_expression, target_expression, query_coords, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )
    total = primary + gradient_weight * gradient
    return {"total": total, "primary": primary, "gradient": gradient}
