"""Geometry-only spot graph, boundary-ring (Rings 1-3) BFS extraction, and
true per-query local-neighbor selection -- Phase 2 of the multiscale
spatial-field handoff ("Implement and test boundary extraction").

Everything here operates on plain coordinate arrays, never on gene
expression or H&E content -- the graph itself is geometry-only, exactly
as the handoff specifies ("Build a geometry-only spot graph separately
for each slide"). Uses scipy.spatial.cKDTree throughout, matching this
repo's existing convention (masking.py's radius_unit="spot_spacing",
mask_bank.py's cap_context_mask nearest_query mode) rather than adding a
new graph-library dependency (no networkx anywhere in this repo).

No random sampling or hard-capping anywhere in this module -- the
handoff is explicit ("Do not randomly sample or hard-cap this boundary to
80 spots... Fail closed if a configured safety maximum is exceeded; never
silently truncate it"). extract_boundary_and_local_context raises if
max_boundary_size is exceeded rather than truncating.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


def build_knn_adjacency(coords: np.ndarray, k_neighbors: int = 6) -> list[np.ndarray]:
    """Geometry-only adjacency: up to k_neighbors nearest OTHER spots for
    every spot in `coords`. Default k_neighbors=6 matches Visium's
    hexagonal spot lattice (every interior spot has exactly 6 equidistant
    neighbors) -- edge/corner spots on the tissue boundary still get up to
    6 candidates from cKDTree, some of which may be considerably farther
    than a true hex neighbor; this is the real geometry of a tissue
    boundary, not something to special-case away here.

    Returns a list of length len(coords); adjacency[i] is an int array of
    neighbor positions (never includes i itself)."""
    n = coords.shape[0]
    if n < 2:
        return [np.array([], dtype=int) for _ in range(n)]
    coords64 = np.asarray(coords, dtype=np.float64)
    tree = cKDTree(coords64)
    k = min(k_neighbors + 1, n)  # +1: a point is always its own nearest neighbor
    _, idx = tree.query(coords64, k=k)
    idx = np.atleast_2d(idx)
    adjacency = []
    for i in range(n):
        neighbors = idx[i]
        neighbors = neighbors[neighbors != i]
        adjacency.append(neighbors.astype(int))
    return adjacency


@dataclass(frozen=True)
class BoundaryExtractionResult:
    query_local_neighbor_idx: np.ndarray  # [n_query, local_k] int, positions within observed_coords
    boundary_idx: np.ndarray  # [n_boundary] int, positions within observed_coords, deduplicated
    boundary_ring: np.ndarray  # [n_boundary] int in {1, 2, 3}, aligned with boundary_idx
    query_depth_to_boundary: np.ndarray  # [n_query] int, BFS hops to the nearest observed spot
    diagnostic: dict = field(default_factory=dict)  # Phase 2 item 6: diagnostic manifest


def extract_boundary_and_local_context(
    observed_coords: np.ndarray,
    query_coords: np.ndarray,
    k_neighbors: int = 6,
    local_k: int = 32,
    max_rings: int = 3,
    max_boundary_size: int | None = None,
) -> BoundaryExtractionResult:
    """Build the geometry-only slide spot graph over observed_coords UNION
    query_coords, then derive:

    - query_local_neighbor_idx: for each query, its TRUE local_k nearest
      observed spots by real distance (a per-query independent k-NN over
      the FULL observed set, not the graph adjacency -- this is "the
      high-resolution local context", distinct from the boundary below).
    - boundary_idx / boundary_ring: Rings 1-3, by breadth-first expansion
      through observed-observed graph edges, seeded by Ring 1 = every
      observed spot that is a direct graph-neighbor of any query spot.
      Every observed spot appears in AT MOST one ring (the first/nearest
      ring it's reached at), never duplicated across rings.
    - query_depth_to_boundary: for each query spot, the shortest graph-hop
      path (through query-query edges) to a query spot that itself
      directly touches an observed spot (i.e. Ring-1-adjacent). A query
      spot directly touching the boundary gets depth 0; a query spot one
      hop deeper gets depth 1; and so on. Query spots not connected to any
      boundary-touching query spot (a possible but pathological case for
      a genuinely contiguous hole) get depth = max_rings + 1 as a
      sentinel "unreachable, treat as maximally interior" value, not NaN
      or a crash.

    Raises ValueError (never silently truncates) if max_boundary_size is
    set and Rings 1-3's true observed count exceeds it -- callers that hit
    this should switch to chunked/memory-efficient attention over the
    full boundary, per the handoff, not shrink the boundary itself.
    """
    n_observed = observed_coords.shape[0]
    n_query = query_coords.shape[0]
    if n_observed == 0:
        raise ValueError("observed_coords is empty -- no context to build a graph over")
    if n_query == 0:
        raise ValueError("query_coords is empty -- nothing to predict")

    all_coords = np.concatenate([observed_coords, query_coords], axis=0)
    # In this concatenated array, positions [0, n_observed) are observed
    # spots (same order as observed_coords) and [n_observed, n_observed+n_query)
    # are query spots (same order as query_coords) -- purely a local
    # convenience for building one adjacency graph; never exposed outside
    # this function.
    adjacency = build_knn_adjacency(all_coords, k_neighbors=k_neighbors)

    def is_observed(pos: int) -> bool:
        return pos < n_observed

    # ---- True local_k nearest OBSERVED spots per query (independent of
    # the graph above -- a real k-NN search over the full observed set). ----
    observed_tree = cKDTree(np.asarray(observed_coords, dtype=np.float64))
    k = min(local_k, n_observed)
    _, local_idx = observed_tree.query(np.asarray(query_coords, dtype=np.float64), k=k)
    local_idx = np.atleast_2d(local_idx)
    if k < local_k:
        # Fewer observed spots exist than local_k requests -- pad by
        # repeating the nearest neighbor rather than silently returning a
        # narrower array a caller might not expect (fail-visible: the
        # diagnostic manifest below records this so it isn't invisible).
        pad_width = local_k - k
        local_idx = np.concatenate([local_idx, np.repeat(local_idx[:, :1], pad_width, axis=1)], axis=1)
    query_local_neighbor_idx = local_idx.astype(int)

    # ---- Boundary rings: BFS over observed-observed edges, seeded by
    # Ring 1 = observed neighbors of any query node. ----
    ring_of_observed = -np.ones(n_observed, dtype=int)  # -1 = not yet assigned a ring
    frontier: set[int] = set()
    for q in range(n_observed, n_observed + n_query):
        for neighbor in adjacency[q]:
            if is_observed(neighbor):
                frontier.add(int(neighbor))
    for pos in frontier:
        ring_of_observed[pos] = 1

    current_ring_positions = frontier
    for ring in range(2, max_rings + 1):
        next_ring_positions: set[int] = set()
        for pos in current_ring_positions:
            for neighbor in adjacency[pos]:
                if is_observed(neighbor) and ring_of_observed[neighbor] == -1:
                    next_ring_positions.add(int(neighbor))
        for pos in next_ring_positions:
            ring_of_observed[pos] = ring
        current_ring_positions = next_ring_positions
        if not current_ring_positions:
            break

    boundary_idx = np.flatnonzero(ring_of_observed >= 1)
    boundary_ring = ring_of_observed[boundary_idx]
    if max_boundary_size is not None and boundary_idx.shape[0] > max_boundary_size:
        raise ValueError(
            f"boundary size {boundary_idx.shape[0]} exceeds max_boundary_size={max_boundary_size} "
            "-- use chunked/memory-efficient attention over the full boundary instead of shrinking "
            "it (the handoff explicitly forbids silent truncation here)"
        )

    # ---- Query depth-to-boundary: BFS over query-query edges, seeded by
    # query nodes that directly touch an observed spot (depth 0). ----
    depth = np.full(n_query, -1, dtype=int)
    boundary_touching_queries: set[int] = set()
    for q in range(n_observed, n_observed + n_query):
        if any(is_observed(neighbor) for neighbor in adjacency[q]):
            boundary_touching_queries.add(q)
    for q in boundary_touching_queries:
        depth[q - n_observed] = 0

    frontier_q = boundary_touching_queries
    hop = 0
    while frontier_q:
        next_frontier: set[int] = set()
        for pos in frontier_q:
            for neighbor in adjacency[pos]:
                if not is_observed(neighbor) and depth[neighbor - n_observed] == -1:
                    next_frontier.add(int(neighbor))
        hop += 1
        for pos in next_frontier:
            depth[pos - n_observed] = hop
        frontier_q = next_frontier

    unreachable = depth == -1
    n_unreachable = int(unreachable.sum())
    if n_unreachable:
        depth[unreachable] = max_rings + 1  # sentinel: disconnected from any boundary-touching query
    query_depth_to_boundary = depth

    diagnostic = {
        "n_observed": int(n_observed),
        "n_query": int(n_query),
        "n_boundary": int(boundary_idx.shape[0]),
        "n_boundary_ring_1": int(np.sum(boundary_ring == 1)),
        "n_boundary_ring_2": int(np.sum(boundary_ring == 2)),
        "n_boundary_ring_3": int(np.sum(boundary_ring == 3)),
        "local_k_requested": int(local_k),
        "local_k_effective": int(k),
        "local_k_padded": bool(k < local_k),
        "n_query_touching_boundary_directly": len(boundary_touching_queries),
        "n_query_unreachable_from_boundary": n_unreachable,
        "max_query_depth_to_boundary": int(query_depth_to_boundary.max()) if n_query else 0,
    }

    return BoundaryExtractionResult(
        query_local_neighbor_idx=query_local_neighbor_idx,
        boundary_idx=boundary_idx,
        boundary_ring=boundary_ring,
        query_depth_to_boundary=query_depth_to_boundary,
        diagnostic=diagnostic,
    )
