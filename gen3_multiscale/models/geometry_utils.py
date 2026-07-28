"""Small shared geometry helpers used by every architecture wrapper
(Phase 6) to turn SpatialFieldInputs' plain coordinate arrays into the
relative-geometry and hole-geometry tensors tokens.py/attention.py/
transport_head.py expect. Kept separate and independently tested so each
architecture wrapper calls the SAME function rather than re-deriving
relative geometry slightly differently per architecture.
"""
from __future__ import annotations

import torch


def compute_relative_geometry(query_coords: torch.Tensor, candidate_coords: torch.Tensor) -> torch.Tensor:
    """(dx, dy, distance) from each query to each candidate.

    candidate_coords may be:
    - [n_query, C, 2] -- already gathered PER QUERY (e.g. each query's own
      local_k nearest observed neighbors) -> returns [n_query, C, 3].
    - [C, 2] -- one SHARED candidate set every query attends to (e.g. the
      boundary, regional tokens, global-GEX inducing tokens) -> returns
      [n_query, C, 3] by broadcasting query_coords against it.
    """
    if candidate_coords.ndim == 3:
        if candidate_coords.shape[0] != query_coords.shape[0]:
            raise ValueError(
                f"per-query candidate_coords must have {query_coords.shape[0]} rows (one per "
                f"query), got {candidate_coords.shape[0]}"
            )
        delta = candidate_coords - query_coords[:, None, :]
    elif candidate_coords.ndim == 2:
        delta = candidate_coords[None, :, :] - query_coords[:, None, :]
    else:
        raise ValueError(f"candidate_coords must be 2-D or 3-D, got shape {tuple(candidate_coords.shape)}")
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    return torch.cat([delta, distance], dim=-1)


def compute_hole_geometry(query_coords: torch.Tensor) -> torch.Tensor:
    """[n_query, 2] hole-level geometry, broadcast identically to every
    query in the same item: (normalized query count as a size proxy for
    "area", normalized distance from this query to the hole's own
    centroid). log1p-compresses the count so hole size doesn't dominate
    on a wildly different scale than the (already roughly unit-scale)
    distance-to-centroid term."""
    n_query = query_coords.shape[0]
    if n_query == 0:
        raise ValueError("query_coords is empty -- no hole to describe")
    centroid = query_coords.mean(dim=0, keepdim=True)
    distance_to_centroid = torch.linalg.norm(query_coords - centroid, dim=-1, keepdim=True)
    scale = distance_to_centroid.max().clamp_min(1e-6)
    normalized_distance = distance_to_centroid / scale
    area_proxy = torch.log1p(torch.tensor(float(n_query))) / 10.0
    area_feature = area_proxy.expand(n_query, 1)
    return torch.cat([area_feature, normalized_distance], dim=-1)


def scatter_boundary_ring(n_observed: int, boundary_idx: torch.Tensor, boundary_ring: torch.Tensor) -> torch.Tensor:
    """[n_observed] int tensor: 0 for every observed spot not in
    boundary_idx, else its real ring value (1/2/3) -- the full per-
    observed-spot ring identity SpotTokenProjection expects, expanded
    from boundary_graph.py's sparse (boundary_idx, boundary_ring) pair."""
    full = torch.zeros(n_observed, dtype=torch.long)
    if boundary_idx.numel():
        full[boundary_idx] = boundary_ring.to(torch.long)
    return full
