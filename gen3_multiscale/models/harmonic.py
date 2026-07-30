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
    boundary-ring extraction uses): every query node equals the mean of
    its current neighbors, with observed nodes fixed at their real
    measured expression throughout.

    The solve is reduced to the query nodes and performed exactly, one
    connected query component at a time.  This is mathematically the
    fixed point that the former Jacobi implementation approximated, but
    avoids repeating a full [n_query, n_genes] update hundreds of times.
    ``n_iterations`` and ``tol`` remain accepted for API/checkpoint
    compatibility; an exact linear solve has no iteration budget.

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

    observed_values = np.asarray(observed_expression, dtype=np.float64)
    n_genes = observed_values.shape[1]
    observed_mean = observed_values.mean(axis=0)
    result = np.empty((n_query, n_genes), dtype=np.float64)

    # Components are found only over query-query edges.  Solving them
    # separately keeps the dense system small and lets us preserve the
    # former, explicit behavior for a pathological query component that
    # has no observed boundary: it receives the global observed mean.
    query_adjacency: list[list[int]] = [[] for _ in range(n_query)]
    for query_pos in range(n_query):
        full_pos = n_observed + query_pos
        query_adjacency[query_pos] = [
            int(neighbor - n_observed)
            for neighbor in adjacency[full_pos]
            if neighbor >= n_observed
        ]

    unseen = set(range(n_query))
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component: list[int] = []
        while stack:
            query_pos = stack.pop()
            component.append(query_pos)
            for neighbor in query_adjacency[query_pos]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        component.sort()

        local_by_query = {query_pos: local for local, query_pos in enumerate(component)}
        system = np.eye(len(component), dtype=np.float64)
        rhs = np.zeros((len(component), n_genes), dtype=np.float64)
        has_observed_boundary = False

        for local, query_pos in enumerate(component):
            neighbors = adjacency[n_observed + query_pos]
            degree = len(neighbors)
            if degree == 0:
                rhs[local] = observed_mean
                continue
            inv_degree = 1.0 / float(degree)
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor < n_observed:
                    rhs[local] += observed_values[neighbor] * inv_degree
                    has_observed_boundary = True
                else:
                    system[local, local_by_query[neighbor - n_observed]] -= inv_degree

        component_idx = np.asarray(component, dtype=int)
        if not has_observed_boundary:
            result[component_idx] = observed_mean
            continue
        try:
            result[component_idx] = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "harmonic query system is singular despite having an observed boundary; "
                "the geometry graph is not a valid missing-tissue interpolation domain"
            ) from exc

    if not np.isfinite(result).all():
        raise ValueError("harmonic interpolation produced non-finite values")
    return result.astype(np.float32)
