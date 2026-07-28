"""Graph-harmonic interpolation -- Architecture 2's ONLY anchor input, and
the exact-mask external baseline every architecture is compared against
(handoff: "Harmonic, inverse-distance, and nearest-neighbour must also be
computed as exact-mask external baselines for every arm").

Deliberately NOT an nn.Module and does not touch torch autograd at all --
plain numpy in, numpy out. The handoff is explicit: "The harmonic solver
must be outside the trainable neural input path. It supplies only the
explicit anchor in the final equation." Keeping this function entirely
outside torch makes that structurally true, not just a convention a
caller has to remember to respect (no `.detach()` to forget).
"""
from __future__ import annotations

import numpy as np

from gen3_multiscale.data.boundary_graph import build_knn_adjacency


def harmonic_interpolation(
    observed_coords: np.ndarray,
    observed_expression: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int = 6,
    n_iterations: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    """Solve the discrete Laplace/harmonic equation on the geometry-only
    k-NN graph (the same graph construction boundary_graph.py's Phase 2
    boundary-ring extraction uses): every query node's value converges to
    the mean of its current neighbors' values, with observed nodes fixed
    at their real measured expression throughout. Solved by vectorized
    Jacobi relaxation (no sparse-linear-solver dependency) -- correct
    regardless of iteration count, since it's a genuine iterative fixed-
    point solver, not a fixed-depth approximation; n_iterations/tol only
    trade off runtime against how close to the true fixed point the
    result gets.

    A query node with zero graph neighbors (isolated, only possible for a
    pathological/disconnected mask) keeps its initial value (the mean of
    ALL observed expression) rather than dividing by zero or crashing.

    Returns [n_query, n_genes] float32 -- never touches torch/autograd.
    """
    n_observed = observed_coords.shape[0]
    n_query = query_coords.shape[0]
    if n_observed == 0:
        raise ValueError("observed_coords is empty -- no boundary condition to interpolate from")
    if n_query == 0:
        raise ValueError("query_coords is empty -- nothing to interpolate")
    if observed_expression.shape[0] != n_observed:
        raise ValueError(
            f"observed_expression has {observed_expression.shape[0]} rows, expected {n_observed}"
        )

    all_coords = np.concatenate([observed_coords, query_coords], axis=0)
    adjacency = build_knn_adjacency(all_coords, k_neighbors=k_neighbors)

    n_genes = observed_expression.shape[1]
    n_total = n_observed + n_query
    max_degree = max((len(a) for a in adjacency), default=0)
    neighbor_idx = np.zeros((n_total, max(max_degree, 1)), dtype=np.int64)
    neighbor_mask = np.zeros((n_total, max(max_degree, 1)), dtype=bool)
    for i, neighbors in enumerate(adjacency):
        n = len(neighbors)
        if n:
            neighbor_idx[i, :n] = neighbors
            neighbor_mask[i, :n] = True
    degree = neighbor_mask.sum(axis=1, keepdims=True).astype(np.float64)

    values = np.zeros((n_total, n_genes), dtype=np.float64)
    values[:n_observed] = observed_expression
    values[n_observed:] = np.asarray(observed_expression, dtype=np.float64).mean(axis=0, keepdims=True)

    query_neighbor_idx = neighbor_idx[n_observed:]
    query_neighbor_mask = neighbor_mask[n_observed:]
    query_degree = degree[n_observed:]
    isolated = query_degree[:, 0] == 0
    safe_degree = np.clip(query_degree, 1.0, None)

    for _ in range(n_iterations):
        gathered = values[query_neighbor_idx] * query_neighbor_mask[..., None]
        new_query_values = gathered.sum(axis=1) / safe_degree
        if isolated.any():
            new_query_values[isolated] = values[n_observed:][isolated]
        delta = float(np.abs(new_query_values - values[n_observed:]).max())
        values[n_observed:] = new_query_values
        if delta < tol:
            break

    return values[n_observed:].astype(np.float32)
