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


def rmse_pcc_reconstruction_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    *,
    pcc_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-panel RMSE plus a bounded mean per-gene correlation penalty.

    Constant genes contribute zero correlation rather than NaN.  This is a
    training objective only; reported PCC/RMSE still come from the shared
    evaluator and are not replaced by these differentiable approximations.
    """
    if predicted_expression.shape != target_expression.shape:
        raise ValueError("predicted_expression and target_expression must have matching shapes")
    if pcc_weight < 0:
        raise ValueError("pcc_weight must be non-negative")
    mse = torch.nn.functional.mse_loss(predicted_expression, target_expression)
    rmse = torch.sqrt(mse + 1e-8)
    pred_centered = predicted_expression - predicted_expression.mean(dim=0, keepdim=True)
    true_centered = target_expression - target_expression.mean(dim=0, keepdim=True)
    numerator = (pred_centered * true_centered).sum(dim=0)
    pred_energy = pred_centered.square().sum(dim=0)
    true_energy = true_centered.square().sum(dim=0)
    valid = (pred_energy > 1e-8) & (true_energy > 1e-8)
    if valid.any():
        # Select before sqrt: sqrt(0)'s undefined derivative can produce
        # NaN gradients even when a later torch.where masks its value.
        denominator = torch.sqrt(pred_energy[valid] * true_energy[valid])
        correlation = numerator[valid] / denominator
        pcc_loss = 1.0 - correlation.mean()
    else:
        pcc_loss = rmse.new_zeros(())
    return rmse + pcc_weight * pcc_loss, rmse, pcc_loss


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


def field_amplitude_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    *,
    per_gene_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match gene-wise field level and within-mask spatial amplitude.

    Both terms are expressed in training-fit per-gene scale units.  The
    amplitude term compares ``log(std)`` so a uniformly shrunken prediction
    is penalized directly instead of being largely tolerated by RMSE.
    """
    if predicted_expression.shape != target_expression.shape:
        raise ValueError("predicted_expression and target_expression must match")
    if predicted_expression.ndim != 2 or predicted_expression.shape[0] < 2:
        zero = predicted_expression.new_zeros(())
        return zero, zero
    scale = torch.as_tensor(
        per_gene_scale, dtype=predicted_expression.dtype,
        device=predicted_expression.device,
    )
    if scale.shape != (predicted_expression.shape[1],):
        raise ValueError("per_gene_scale must be [n_genes]")
    scale = scale.clamp_min(1e-6)
    mean_loss = torch.nn.functional.smooth_l1_loss(
        predicted_expression.mean(dim=0) / scale,
        target_expression.mean(dim=0) / scale,
    )
    pred_std = predicted_expression.std(dim=0, unbiased=False) / scale
    true_std = target_expression.std(dim=0, unbiased=False) / scale
    amplitude_loss = torch.nn.functional.smooth_l1_loss(
        torch.log(pred_std + 1e-3), torch.log(true_std + 1e-3),
    )
    return mean_loss, amplitude_loss


def gene_map_identity_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    *,
    gene_indices: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric contrastive identity loss between spatial gene maps.

    A predicted map for gene *g* must retrieve the target map for the same
    gene among a bounded train-selected panel.  This directly detects the
    repeated-template failure where several output genes receive nearly the
    same spatial pattern.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if predicted_expression.shape != target_expression.shape:
        raise ValueError("predicted_expression and target_expression must match")
    if predicted_expression.shape[0] < 2:
        return predicted_expression.new_zeros(())
    indices = torch.as_tensor(
        gene_indices, dtype=torch.long, device=predicted_expression.device,
    )
    if indices.ndim != 1 or indices.numel() < 2:
        return predicted_expression.new_zeros(())
    predicted = predicted_expression.index_select(1, indices)
    target = target_expression.index_select(1, indices)
    predicted = predicted - predicted.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    pred_energy = predicted.square().sum(dim=0)
    true_energy = target.square().sum(dim=0)
    valid = (pred_energy > 1e-8) & (true_energy > 1e-8)
    if int(valid.sum()) < 2:
        return predicted_expression.new_zeros(())
    predicted = predicted[:, valid] / pred_energy[valid].sqrt().unsqueeze(0)
    target = target[:, valid] / true_energy[valid].sqrt().unsqueeze(0)
    logits = predicted.T @ target / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, labels)
        + torch.nn.functional.cross_entropy(logits.T, labels)
    )


def gene_map_spectrum_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    *,
    gene_indices: torch.Tensor,
    max_rank: int = 64,
) -> torch.Tensor:
    """Match the normalized singular-value spectrum of selected gene maps.

    The normalization deliberately removes total amplitude (handled by
    :func:`field_amplitude_loss`) and isolates lost spatial/gene diversity.
    """
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    if predicted_expression.shape != target_expression.shape:
        raise ValueError("predicted_expression and target_expression must match")
    indices = torch.as_tensor(
        gene_indices, dtype=torch.long, device=predicted_expression.device,
    )
    if predicted_expression.shape[0] < 2 or indices.numel() < 2:
        return predicted_expression.new_zeros(())
    predicted = predicted_expression.index_select(1, indices)
    target = target_expression.index_select(1, indices)
    predicted = predicted - predicted.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    rank = min(max_rank, predicted.shape[0] - 1, predicted.shape[1])
    if rank < 1:
        return predicted_expression.new_zeros(())
    pred_singular = torch.linalg.svdvals(predicted.float())[:rank]
    true_singular = torch.linalg.svdvals(target.float())[:rank]
    pred_singular = pred_singular / pred_singular.norm().clamp_min(1e-8)
    true_singular = true_singular / true_singular.norm().clamp_min(1e-8)
    return torch.nn.functional.smooth_l1_loss(
        torch.log(pred_singular + 1e-5), torch.log(true_singular + 1e-5),
    ).to(predicted_expression.dtype)


def combined_reconstruction_loss(
    predicted_expression: torch.Tensor,
    target_expression: torch.Tensor,
    query_coords: torch.Tensor,
    gradient_weight: float = 0.05,
    k_neighbors: int = 6,
    per_gene_scale: torch.Tensor | None = None,
    primary_mode: str = "mse",
    pcc_weight: float = 0.1,
) -> dict:
    """The shared deterministic objective, assembled: primary + a small
    weight (default 0.05, the handoff's stated initial value) times the
    spatial-gradient term. Returns every component (not just the total)
    so a training loop can log them separately without recomputing."""
    if gradient_weight < 0:
        raise ValueError("gradient_weight must be non-negative")
    if primary_mode == "mse":
        primary = primary_reconstruction_loss(predicted_expression, target_expression)
        extra = {}
    elif primary_mode == "rmse_pcc":
        primary, rmse, pcc_loss = rmse_pcc_reconstruction_loss(
            predicted_expression, target_expression, pcc_weight=pcc_weight,
        )
        extra = {"rmse_loss": rmse, "pcc_loss": pcc_loss}
    else:
        raise ValueError("primary_mode must be 'mse' or 'rmse_pcc'")
    gradient = spatial_gradient_loss(
        predicted_expression, target_expression, query_coords, k_neighbors=k_neighbors, per_gene_scale=per_gene_scale,
    )
    total = primary + gradient_weight * gradient
    return {"total": total, "primary": primary, "gradient": gradient, **extra}
