"""The one shared data object consumed by all four multiscale spatial-field
architectures (handoff doc, "Phase 1: Build one shared example object").

Two separate frozen dataclasses, not one dataclass with a "target" field
a caller could accidentally pass through -- the handoff is explicit that
"Model-forward inputs must be separated from target-only fields by type
and API, not merely by convention." A model's forward() should type-
annotate its argument as SpatialFieldInputs; SpatialFieldTargets exists
only for the loss/metrics call sites, and nothing in this module ever
constructs one from the other.

This module defines the SCHEMA and its invariants (Phase 1). It does not
yet build these objects from real HEST-1k data -- boundary-ring
extraction (Phase 2) and the mask-aware WSI path (Phase 3) are separate,
not-yet-implemented stages. validate_spatial_field_example enforces the
structural contract so Phase 2/3's real builder has something concrete to
be checked against from the very first commit that touches them, and so
this module is independently testable with synthetic examples now.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SpatialFieldTargets:
    """Loss/metric-ONLY fields. Never pass this to a model's forward().

    query_expression: [n_query, n_genes] float32, the model's native
    (library-size-normalized log1p) space -- the same space
    query_target_expression has always meant in gen2_architectures'
    masked_item.py, preserved here for continuity across pipelines.
    query_raw_counts: optional [n_query, n_genes] float32, log1p(raw
    counts) -- kept alongside for STPath-space/oracle-metric parity with
    gen2_architectures' oracle_library_size_pcc_raw_log1p, not used by
    the primary loss.
    """
    query_expression: np.ndarray
    query_raw_counts: np.ndarray | None = None


@dataclass(frozen=True)
class SpatialFieldInputs:
    """Everything a model's forward() may legally see for one training or
    evaluation item (one slide, one contiguous hole). No query GEX, no
    query H&E -- enforced by validate_spatial_field_example, not merely
    by this dataclass never holding such a field (a builder bug could
    still populate the wrong array under the right name; the validator
    catches shape/index-level contamination the type system can't).

    Indices (observed_idx, query_idx, boundary_idx, query_local_neighbor_idx)
    are all indices into the SAME per-slide spot ordering -- the one
    boundary_graph.py's future BFS will build coordinates/adjacency from
    (Phase 2). observed_idx and query_idx partition that ordering; every
    other index array indexes INTO observed_idx's members (never raw
    per-slide row numbers), so a caller never needs the full per-slide
    arrays to interpret them -- see the *_dense_idx cross-references
    documented per field below.

    coords are slide-relative, already normalized (never raw physical
    micrometers/pixels that could identify a specific sample's layout --
    see the handoff's "Coordinate memorization" risk).
    """
    sample_id: str
    patient_id: str

    # Per-slide spot ordering this example was built from -- every index
    # field below is either a position in this ordering (observed_idx,
    # query_idx) or a position WITHIN observed_idx (everything else,
    # documented per-field).
    observed_idx: np.ndarray  # [n_observed] int, positions in the per-slide ordering
    query_idx: np.ndarray  # [n_query] int, positions in the per-slide ordering, disjoint from observed_idx
    coords: np.ndarray  # [n_observed + n_query, 2] float32, normalized/relative, indexed like the per-slide ordering

    # Observed-spot content -- length n_observed, aligned with observed_idx
    # (row i here describes the spot at per-slide position observed_idx[i]).
    observed_gex_conditioning: np.ndarray  # [n_observed, gex_dim] compact feature for scoring/gating
    observed_full_gene_expression: np.ndarray  # [n_observed, n_genes] untouched real values, transported not decoded
    observed_gigapath_features: np.ndarray  # [n_observed, 1536] frozen local H&E tile embeddings

    # Boundary/local structure -- all values are positions WITHIN
    # observed_idx (i.e. in [0, n_observed)), not per-slide positions.
    query_local_neighbor_idx: np.ndarray  # [n_query, local_k] int, true nearest observed_idx positions per query
    boundary_idx: np.ndarray  # [n_boundary] int, observed_idx positions in Rings 1-3, deduplicated
    boundary_ring: np.ndarray  # [n_boundary] int in {1, 2, 3}, aligned with boundary_idx
    query_depth_to_boundary: np.ndarray  # [n_query] int, BFS hops from the nearest boundary ring

    # Optional dense WSI context (Phase 3) -- tiles whose footprint does
    # NOT overlap the hole. None until Phase 3 wires the mask-aware path.
    wsi_tile_features: np.ndarray | None = None  # [n_tiles, 1536]
    wsi_tile_coords: np.ndarray | None = None  # [n_tiles, 2], same normalization as coords

    # Free-form, not consumed by any forward() -- fingerprints/labels a
    # model must never read but a training/eval harness needs (mirrors
    # gen2_architectures' mask/cache provenance discipline).
    provenance: dict = field(default_factory=dict)


def validate_spatial_field_example(inputs: SpatialFieldInputs, targets: SpatialFieldTargets) -> None:
    """Structural contract every real builder (Phase 2/3) and every
    synthetic test example must satisfy. Raises ValueError with a
    specific, actionable message on the first violation found -- fail
    loud, never silently accept a malformed example, matching this
    project's established discipline (mask_bank.record_masks's identical
    context/query-overlap check, checkpoint.verify_gene_names's identical
    fail-closed stance)."""
    observed_idx = np.asarray(inputs.observed_idx)
    query_idx = np.asarray(inputs.query_idx)
    n_observed = observed_idx.shape[0]
    n_query = query_idx.shape[0]

    if n_observed == 0:
        raise ValueError("SpatialFieldInputs.observed_idx is empty -- no context to condition on")
    if n_query == 0:
        raise ValueError("SpatialFieldInputs.query_idx is empty -- nothing to predict")
    if np.intersect1d(observed_idx, query_idx).size > 0:
        raise ValueError(
            "observed_idx and query_idx overlap -- a query spot cannot also be an observed "
            "context spot (this would leak the hidden target through the context path)"
        )
    if len(np.unique(observed_idx)) != n_observed:
        raise ValueError("observed_idx contains duplicate per-slide positions")
    if len(np.unique(query_idx)) != n_query:
        raise ValueError("query_idx contains duplicate per-slide positions")

    n_total = n_observed + n_query
    if inputs.coords.shape[0] != n_total:
        raise ValueError(
            f"coords has {inputs.coords.shape[0]} rows but observed_idx+query_idx cover "
            f"{n_total} per-slide positions"
        )

    for name, arr in (
        ("observed_gex_conditioning", inputs.observed_gex_conditioning),
        ("observed_full_gene_expression", inputs.observed_full_gene_expression),
        ("observed_gigapath_features", inputs.observed_gigapath_features),
    ):
        if arr.shape[0] != n_observed:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected n_observed={n_observed}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    if inputs.query_local_neighbor_idx.shape[0] != n_query:
        raise ValueError(
            f"query_local_neighbor_idx has {inputs.query_local_neighbor_idx.shape[0]} rows, "
            f"expected n_query={n_query}"
        )
    local_flat = inputs.query_local_neighbor_idx.reshape(-1)
    if local_flat.size and (local_flat.min() < 0 or local_flat.max() >= n_observed):
        raise ValueError(
            "query_local_neighbor_idx contains a position outside [0, n_observed) -- must index "
            "into observed_idx, not the per-slide ordering"
        )

    n_boundary = inputs.boundary_idx.shape[0]
    if inputs.boundary_ring.shape[0] != n_boundary:
        raise ValueError("boundary_idx and boundary_ring must have the same length")
    if n_boundary and (inputs.boundary_idx.min() < 0 or inputs.boundary_idx.max() >= n_observed):
        raise ValueError(
            "boundary_idx contains a position outside [0, n_observed) -- must index into "
            "observed_idx, not the per-slide ordering"
        )
    if n_boundary and len(np.unique(inputs.boundary_idx)) != n_boundary:
        raise ValueError("boundary_idx contains duplicates -- each observed spot must appear once")
    if n_boundary and not np.all(np.isin(inputs.boundary_ring, [1, 2, 3])):
        raise ValueError("boundary_ring must only contain values in {1, 2, 3}")

    if inputs.query_depth_to_boundary.shape[0] != n_query:
        raise ValueError("query_depth_to_boundary must have one entry per query")
    if np.any(inputs.query_depth_to_boundary < 0):
        raise ValueError("query_depth_to_boundary must be non-negative (BFS hop count)")

    if inputs.wsi_tile_features is not None:
        if inputs.wsi_tile_coords is None:
            raise ValueError("wsi_tile_features is set but wsi_tile_coords is None")
        if inputs.wsi_tile_features.shape[0] != inputs.wsi_tile_coords.shape[0]:
            raise ValueError("wsi_tile_features and wsi_tile_coords must have the same row count")

    if targets.query_expression.shape[0] != n_query:
        raise ValueError(
            f"targets.query_expression has {targets.query_expression.shape[0]} rows, "
            f"expected n_query={n_query}"
        )
    if targets.query_expression.shape[1] != inputs.observed_full_gene_expression.shape[1]:
        raise ValueError(
            "targets.query_expression and observed_full_gene_expression have different gene "
            "panel widths -- the target and the transport-candidate pool must share one panel"
        )
    if not np.all(np.isfinite(targets.query_expression)):
        raise ValueError("targets.query_expression contains non-finite values")
    if targets.query_raw_counts is not None and targets.query_raw_counts.shape != targets.query_expression.shape:
        raise ValueError("targets.query_raw_counts must have the same shape as targets.query_expression")
