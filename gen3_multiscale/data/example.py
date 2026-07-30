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
yet build these objects from real HEST-1k data -- the mask-aware WSI path
(Phase 3) is a separate, not-yet-implemented stage; boundary-ring
extraction (Phase 2) is implemented in boundary_graph.py and produces the
query_local_neighbor_idx/boundary_idx/boundary_ring/query_depth_to_boundary
fields below. validate_spatial_field_example enforces the structural
contract so Phase 2/3's real builder has something concrete to be checked
against, and so this module is independently testable with synthetic
examples now.

2026-07-28 revision (caught while implementing Phase 2's boundary BFS):
the first version of this schema described observed_idx/query_idx as
"positions in a shared per-slide ordering" while sizing `coords` to only
`n_observed + n_query` rows -- inconsistent, since a slide's real observed
set can be a large fraction of the whole slide (thousands of spots) while
`coords` can't be both "indexed by raw per-slide position" and "only
n_observed+n_query rows long" at the same time. Fixed by dropping raw
per-slide indexing entirely: observed_barcodes/query_barcodes are
provenance-only identifiers (original spot names, never used to index
any array here), and observed_coords/query_coords are separate arrays
POSITION-aligned with the other observed_*/query_* content arrays, the
same convention every other field already used. No caller ever needs a
full per-slide array to interpret anything in this dataclass.
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

    Every observed_*/query_* array pair below is POSITION-aligned: row i
    of observed_coords describes the same spot as row i of
    observed_full_gene_expression and observed_gigapath_features (and
    observed_barcodes[i] is that spot's
    original identity, for provenance/debugging only -- never used to
    index anything). query_local_neighbor_idx and boundary_idx are
    positions WITHIN the observed_* arrays (i.e. in [0, n_observed)),
    never raw per-slide positions -- there is no full-per-slide array
    anywhere in this object to index into.

    Coordinates are slide-relative, already normalized (never raw
    physical micrometers/pixels that could identify a specific sample's
    layout -- see the handoff's "Coordinate memorization" risk).
    """
    sample_id: str
    patient_id: str

    # Provenance only -- original per-slide spot identity, never used to
    # index any array in this dataclass.
    observed_barcodes: np.ndarray  # [n_observed] str/int
    query_barcodes: np.ndarray  # [n_query] str/int

    observed_coords: np.ndarray  # [n_observed, 2] float32, normalized/relative
    query_coords: np.ndarray  # [n_query, 2] float32, same normalization as observed_coords

    # Observed-spot content -- position-aligned with observed_coords/observed_barcodes.
    #
    # No separate precomputed "conditioning" field: an earlier revision
    # had one (observed_gex_conditioning), but a 3rd-round Codex re-audit
    # (of commit ca7cf53) correctly flagged it as a dangerous dead input
    # once _SharedFieldArchitecture stopped reading it -- "later code may
    # accidentally start consuming them again." Removed entirely rather
    # than kept-but-optional, per this project's own standing principle
    # (delete what's genuinely unused rather than leave a vestigial
    # field). The real, trainable "weighted_linear" gene conditioning
    # encoder (models/gene_encoder.py) is owned and called by the model
    # itself, from observed_full_gene_expression below -- the only gene
    # array this dataclass carries.
    observed_full_gene_expression: np.ndarray  # [n_observed, n_genes] untouched real values, transported not decoded
    observed_gigapath_features: np.ndarray  # [n_observed, 1536] frozen local H&E tile embeddings, zeroed where unavailable

    # 15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED real:
    # a prior version had NO per-spot H&E-availability signal at all --
    # SpotTokenProjection's modality_flags input (models/tokens.py) was
    # hardcoded to "always available" in _SharedFieldArchitecture, and
    # example_builder.py DROPPED an entire context spot (both GEX and
    # image) whenever its H&E patch physically overlapped the query hole,
    # even though that spot's GEX is real, measured, and available. GEX
    # availability and H&E availability are two INDEPENDENT real-world
    # failure modes (imaging can fail locally while transcriptomics
    # stays readable) and must be tracked as two independent per-spot
    # facts, not conflated into one drop decision. observed_* arrays now
    # include EVERY GEX-available context spot; observed_image_available
    # is the explicit per-spot flag SpotTokenProjection's modality_flags
    # input is built from -- a spot with image_available=False ALWAYS has
    # its observed_gigapath_features row zeroed (never garbage, never a
    # real patch feature the deployment scenario wouldn't actually have),
    # and the flag -- not the zero value -- is what tells the model so.
    observed_image_available: np.ndarray  # [n_observed] bool, True iff this spot's H&E patch does not overlap the hole

    # Boundary/local structure -- all index values are positions WITHIN
    # the observed_* arrays (i.e. in [0, n_observed)).
    query_local_neighbor_idx: np.ndarray  # [n_query, local_k] int, true nearest observed positions per query
    boundary_idx: np.ndarray  # [n_boundary] int, observed positions in Rings 1-3, deduplicated
    boundary_ring: np.ndarray  # [n_boundary] int in {1, 2, 3}, aligned with boundary_idx
    query_depth_to_boundary: np.ndarray  # [n_query] int, BFS hops from the nearest observed spot

    # Dense WSI context (Phase 3 / Step 5) -- tiles whose footprint does
    # NOT overlap the hole (real deployment-visible tiles only; already
    # filtered by slide_context.visible_slide_context before reaching
    # here). All FIVE of the fields below are set together (all or none)
    # until a caller wires in a real slide_context.load_slide_context
    # result.
    #
    # 16th Codex re-audit (Step 5 Part 2 acceptance criteria), CONFIRMED
    # real: a prior version had exactly ONE wsi_tile_coords field, used
    # for BOTH real GigaPath LongNet inference (which needs its own
    # native target-MPP tile coordinates for positional encoding -- see
    # models.slide_encoder.FrozenGigaPathSlideEncoder.forward) AND
    # regional-token spatial pooling / relative-geometry-to-query
    # (which needs the SAME centered, spot-spacing-normalized frame as
    # observed_coords/query_coords, or a query-relative attention bias
    # would be computing distances in the wrong units entirely).
    # Feeding LongNet a normalized, per-example-centered coordinate would
    # silently corrupt its real positional encoding (LongNet was
    # pretrained on true physical tile positions, not a value that
    # rescales and re-centers per example); feeding the regional pooler
    # raw LongNet-frame coordinates would make relative geometry to
    # query_coords meaningless.
    #
    # 17th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED
    # real: a prior version derived wsi_tile_regional_coords from THIS
    # SAME LongNet-frame array (`visible["coords"]`) minus the ST-level-0
    # reference/scale -- physically invalid whenever the cache's source
    # MPP differs from GigaPath's target MPP (0.5 um/px), since `coords`
    # and `mask_coords`/spot_coords are then genuinely different
    # physical frames, not just different units of the same one.
    # wsi_tile_longnet_coords MUST stay slide_context.visible_slide_context's
    # `coords` (GigaPath's own frame, UNNORMALIZED, fed to LongNet
    # unchanged); wsi_tile_regional_coords MUST be derived from that same
    # function's `level0_coords` (the level-0/HEST-aligned tile CENTER
    # coordinates, the SAME physical frame observed_coords/query_coords
    # are already expressed in) via the documented
    # `(level0_coords - reference) / scale` transform -- the two source
    # arrays are never interchangeable, hence the deliberately
    # unambiguous field name (renamed from wsi_tile_native_coords, which
    # did not make clear which of the two real physical frames it was).
    wsi_tile_longnet_coords: np.ndarray | None = None  # [n_visible_tiles, 2], GigaPath LongNet's own target-MPP coords, UNNORMALIZED
    wsi_tile_regional_coords: np.ndarray | None = None  # [n_visible_tiles, 2], level-0/HEST-aligned frame, same centered/normalized transform as observed_coords
    wsi_tile_features: np.ndarray | None = None  # [n_visible_tiles, 1536]
    # (xmin, xmax, ymin, ymax) computed from the COMPLETE tile set BEFORE
    # hole filtering, in the SAME normalized frame as
    # wsi_tile_regional_coords -- models.slide_encoder.pool_regional_tokens's
    # own required input, so regional grid cell (i, j) refers to the
    # SAME physical region across every example on this slide regardless
    # of which hole was cut for this particular item (15th Codex
    # re-audit's "regional-grid bounds must come from the complete slide
    # before masking" requirement).
    full_slide_coord_bounds: tuple[float, float, float, float] | None = None
    # Real cache-key material for the frozen LongNet global vector (15th
    # Codex re-audit's "the existing coordinate-only in-memory LongNet
    # cache key must be strengthened" requirement) -- binds the ACTUAL
    # tile-cache content hash, the real tile coordinates, and the
    # specific VISIBLE-tile set this example's hole produced
    # (slide_context.visible_slide_context's own context_id, itself now
    # bound to real content -- see that module). A caller combines this
    # with the loaded GigaPath checkpoint's own SHA256 (a property of
    # WHICH model is running, not of this example) to form the complete
    # cache namespace; this dataclass never assumes a specific checkpoint.
    # REQUIRED whenever the other WSI fields are set (16th Codex
    # re-audit, CONFIRMED: a prior version's all-or-none check did not
    # include this field at all, so WSI context could pass validation
    # with slide_cache_namespace=None).
    slide_cache_namespace: str | None = None

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
    n_observed = inputs.observed_coords.shape[0]
    n_query = inputs.query_coords.shape[0]

    if n_observed == 0:
        raise ValueError("SpatialFieldInputs.observed_coords is empty -- no context to condition on")
    if n_query == 0:
        raise ValueError("SpatialFieldInputs.query_coords is empty -- nothing to predict")

    observed_barcodes = np.asarray(inputs.observed_barcodes)
    query_barcodes = np.asarray(inputs.query_barcodes)
    if observed_barcodes.shape[0] != n_observed:
        raise ValueError("observed_barcodes must have one entry per observed spot")
    if query_barcodes.shape[0] != n_query:
        raise ValueError("query_barcodes must have one entry per query spot")
    if np.intersect1d(observed_barcodes, query_barcodes).size > 0:
        raise ValueError(
            "observed_barcodes and query_barcodes overlap -- a query spot cannot also be an "
            "observed context spot (this would leak the hidden target through the context path)"
        )
    if len(np.unique(observed_barcodes)) != n_observed:
        raise ValueError("observed_barcodes contains duplicate spot identities")
    if len(np.unique(query_barcodes)) != n_query:
        raise ValueError("query_barcodes contains duplicate spot identities")

    for name, arr in (
        ("observed_full_gene_expression", inputs.observed_full_gene_expression),
        ("observed_gigapath_features", inputs.observed_gigapath_features),
    ):
        if arr.shape[0] != n_observed:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected n_observed={n_observed}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    image_available = np.asarray(inputs.observed_image_available)
    if image_available.shape != (n_observed,):
        raise ValueError(
            f"observed_image_available must be [n_observed]=[{n_observed}], got shape "
            f"{image_available.shape}"
        )
    if not np.array_equal(image_available, image_available.astype(bool)):
        raise ValueError("observed_image_available must be a boolean (0/1) array")
    unavailable = ~image_available.astype(bool)
    if unavailable.any() and not np.all(inputs.observed_gigapath_features[unavailable] == 0.0):
        raise ValueError(
            "observed_gigapath_features has non-zero value(s) for a spot marked "
            "observed_image_available=False -- an unavailable image must be an explicit zero, "
            "never a real (or garbage) feature the model could learn to read"
        )

    if inputs.query_local_neighbor_idx.shape[0] != n_query:
        raise ValueError(
            f"query_local_neighbor_idx has {inputs.query_local_neighbor_idx.shape[0]} rows, "
            f"expected n_query={n_query}"
        )
    local_flat = inputs.query_local_neighbor_idx.reshape(-1)
    if local_flat.size and (local_flat.min() < 0 or local_flat.max() >= n_observed):
        raise ValueError(
            "query_local_neighbor_idx contains a position outside [0, n_observed) -- must index "
            "into the observed_* arrays"
        )

    n_boundary = inputs.boundary_idx.shape[0]
    if n_boundary == 0:
        raise ValueError(
            "boundary_idx is empty -- every spatial-field example must have at least one "
            "observed boundary spot"
        )
    if inputs.boundary_ring.shape[0] != n_boundary:
        raise ValueError("boundary_idx and boundary_ring must have the same length")
    if n_boundary and (inputs.boundary_idx.min() < 0 or inputs.boundary_idx.max() >= n_observed):
        raise ValueError(
            "boundary_idx contains a position outside [0, n_observed) -- must index into the "
            "observed_* arrays"
        )
    if n_boundary and len(np.unique(inputs.boundary_idx)) != n_boundary:
        raise ValueError("boundary_idx contains duplicates -- each observed spot must appear once")
    if n_boundary and not np.all(np.isin(inputs.boundary_ring, [1, 2, 3])):
        raise ValueError("boundary_ring must only contain values in {1, 2, 3}")

    if inputs.query_depth_to_boundary.shape[0] != n_query:
        raise ValueError("query_depth_to_boundary must have one entry per query")
    if np.any(inputs.query_depth_to_boundary < 0):
        raise ValueError("query_depth_to_boundary must be non-negative (BFS hop count)")

    wsi_fields_set = (
        inputs.wsi_tile_longnet_coords is not None, inputs.wsi_tile_regional_coords is not None,
        inputs.wsi_tile_features is not None, inputs.full_slide_coord_bounds is not None,
        inputs.slide_cache_namespace is not None,
    )
    if any(wsi_fields_set) and not all(wsi_fields_set):
        raise ValueError(
            "wsi_tile_longnet_coords, wsi_tile_regional_coords, wsi_tile_features, "
            "full_slide_coord_bounds, and slide_cache_namespace must be set together (all five or "
            "none) -- regional/global GigaPath context requires all of them"
        )
    if inputs.wsi_tile_features is not None:
        # 18th Codex re-audit (Step 5 Part 2, "Other real gaps"),
        # CONFIRMED: ndim must be checked BEFORE reading shape[0] -- a
        # scalar (0-d) array has shape () and shape[0] raises a raw
        # IndexError, not the intended, actionable ValueError.
        if inputs.wsi_tile_features.ndim != 2:
            raise ValueError(f"wsi_tile_features must be [N, F], got shape {inputs.wsi_tile_features.shape}")
        n_visible_tiles = inputs.wsi_tile_features.shape[0]
        # 17th Codex re-audit (Step 5 Part 2), CONFIRMED: a prior version
        # only checked ROW counts against wsi_tile_features -- never that
        # the coordinate arrays were actually [N, 2] (a [N, 3] or [N]
        # array with a matching row count would have silently passed).
        for coord_name, coord_arr in (
            ("wsi_tile_longnet_coords", inputs.wsi_tile_longnet_coords),
            ("wsi_tile_regional_coords", inputs.wsi_tile_regional_coords),
        ):
            if coord_arr.shape != (n_visible_tiles, 2):
                raise ValueError(
                    f"{coord_name} must be [{n_visible_tiles}, 2] aligned with wsi_tile_features, "
                    f"got shape {coord_arr.shape}"
                )
            # 17th Codex re-audit, CONFIRMED: no check existed for
            # duplicate coordinates in EITHER WSI frame independently --
            # load_slide_context only rejects duplicates in the
            # UNFILTERED cache's `coords` field, never `mask_coords`
            # (the two are independently sourced fields in a real cache
            # and could disagree), and never after mask-filtering either.
            if np.unique(coord_arr, axis=0).shape[0] != n_visible_tiles:
                raise ValueError(f"{coord_name} contains duplicate tile coordinates")
        if n_visible_tiles == 0:
            raise ValueError("wsi_tile_features is set but has zero rows -- pass None instead of an empty array")
        if not np.all(np.isfinite(inputs.wsi_tile_features)):
            raise ValueError("wsi_tile_features contains non-finite values")
        if not np.all(np.isfinite(inputs.wsi_tile_longnet_coords)) or not np.all(np.isfinite(inputs.wsi_tile_regional_coords)):
            raise ValueError("wsi_tile_longnet_coords/wsi_tile_regional_coords contain non-finite values")
        if not isinstance(inputs.slide_cache_namespace, str) or not inputs.slide_cache_namespace.strip():
            raise ValueError("slide_cache_namespace must be a non-empty string when WSI context is set")
        xmin, xmax, ymin, ymax = inputs.full_slide_coord_bounds
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"full_slide_coord_bounds {inputs.full_slide_coord_bounds} is degenerate/invalid")
        # full_slide_coord_bounds is documented as the SAME normalized
        # frame as wsi_tile_regional_coords (models.slide_encoder.
        # pool_regional_tokens's own required input) -- checked against
        # the regional, not native, coordinates.
        regional_xy = np.asarray(inputs.wsi_tile_regional_coords, dtype=np.float64)
        outside = (
            (regional_xy[:, 0] < xmin) | (regional_xy[:, 0] > xmax)
            | (regional_xy[:, 1] < ymin) | (regional_xy[:, 1] > ymax)
        )
        if outside.any():
            raise ValueError(
                f"{int(outside.sum())} wsi_tile_regional_coords row(s) fall outside "
                f"full_slide_coord_bounds {inputs.full_slide_coord_bounds} -- the visible tile set "
                "must be a subset of the complete slide the bounds were computed from"
            )

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
