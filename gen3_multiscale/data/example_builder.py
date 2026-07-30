"""Real per-example builder for the Gen3 spatial-field schema -- Step 2
of the real Gen3 data builder/trainer (Adam's explicit instruction and
9-step implementation order, CONTRACT.md section 30, item 2: "Build
training examples from realized masks. Query spots must be physically
absent from both input GEX and target-region H&E. Context may contain
surrounding H&E and observed GEX only.").

Given one sample's real, QC'd, gene-panel-aligned data (from
load_sample_for_examples below) and a context/query barcode split (from
a realized mask -- e.g. a mask_bank.py record's context_obs_names/
query_obs_names), builds a real SpatialFieldInputs/SpatialFieldTargets
pair.

Physical H&E safety goes further than barcode-level safety: a context
spot's H&E patch can PHYSICALLY OVERLAP the query hole even though its
own barcode is disjoint from every query barcode -- the patch footprint
is a real square of pixels centered on the spot's coordinate, not a
point. 15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED real:
an earlier version DROPPED such a spot from the observed set entirely
(both GEX and image) -- but GEX and H&E are independent real-world
failure modes (imaging can fail locally while transcriptomics stays
readable), and dropping the GEX too throws away real, available signal
for no reason. Every GEX-available context spot is now RETAINED;
`slide_context.nonoverlapping_context_patch_mask` (already-audited
overlap geometry, reused, not reimplemented) instead produces an
explicit per-spot `observed_image_available` flag -- a spot with that
flag False gets its `observed_gigapath_features` row EXPLICITLY zeroed
(the real patch is never even fed to `image_feature_fn`, modeling the
actual deployment scenario where that image would not exist) and the
flag itself (not the zero value) is what SpotTokenProjection's
`modality_flags` input reads. Query spots are already structurally
absent from every observed_* array (validate_spatial_field_example
enforces disjoint observed/query barcodes, unconditionally); this module
never even reads a query spot's expression or patch for anything other
than the physical-overlap geometry check and SpatialFieldTargets itself.

Coordinates are normalized before being stored -- SpatialFieldInputs'
own docstring is explicit ("Coordinates are slide-relative, already
normalized... never raw physical micrometers/pixels that could identify
a specific sample's layout") and this has been repeated as an explicit
audit recommendation (6th Codex re-audit of commit 06f5cce: "Use
coordinates relative to the hole or slide centre in units of median
spot spacing. Do not feed raw pixel coordinates."). Centered on this
example's own observed+query centroid, scaled by the sample's median
nearest-neighbor spot spacing (computed once from the WHOLE real spot
lattice, not just this example's subset, for a stable per-sample unit
that doesn't fluctuate mask-to-mask).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from gen3_multiscale.data import loaders
from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example
from gen3_multiscale.data.slide_context import (
    nonoverlapping_context_patch_mask, tile_centers, visible_slide_context,
)

# 17th Codex re-audit (Step 5 Part 2 launch blocker #4): image intervention
# semantics are only implemented consistently for "target_zero" today --
# "all_zero" removes WSI context but leaves spot H&E features populated
# (a real inconsistency: use_regional_he/use_global_slide would then fail
# on the resulting example instead of cleanly degrading), "shuffled" does
# not actually shuffle WSI features (a silent no-op alias of "full"), and
# "full" still removes spot patches overlapping the hole even though
# nothing else about the item was damaged. Until those modes are made
# consistent, the real builder accepts only "target_zero".
_SUPPORTED_IMAGE_MODES = frozenset({"target_zero"})


def load_expression_for_model_target_space(
    manifest: dict, sample_id: str, hest_data_dir: str | Path | None = None,
):
    """Load one sample in the exact Gen3 target space without touching H&E.

    This is the authoritative expression half of ``load_sample_for_examples``
    and is also used for train-only variance panels.  Keeping it here avoids
    either loading gigabytes of irrelevant patches or reimplementing the
    normalization/QC transform in the evaluator.
    """
    hest_data_dir = Path(hest_data_dir) if hest_data_dir is not None else Path(manifest["hest_data_dir"])
    record = manifest["samples"][sample_id]
    args = manifest["build_args"]
    import scanpy as sc

    adata = loaders.load_hest_sample(hest_data_dir, sample_id, organ=record["organ"], tech=record["tech"])
    if args["gene_min_genes_per_spot"] > 0:
        sc.pp.filter_cells(adata, min_genes=args["gene_min_genes_per_spot"])
    adata = loaders.basic_qc_and_normalize(
        adata, min_genes=0, min_cells=0,
        transform=args["expression_transform"], target_sum=args["expression_target_sum"],
    )
    missing_genes = [g for g in manifest["gene_panel"] if g not in adata.var_names]
    if missing_genes:
        raise ValueError(
            f"{sample_id} is missing {len(missing_genes)} genes from the manifest's declared "
            f"gene_panel (examples: {missing_genes[:5]}) -- rebuild the manifest"
        )
    adata = adata[:, manifest["gene_panel"]].copy()
    expected_barcodes = set(map(str, record["barcodes"]))
    actual_barcodes = set(map(str, adata.obs_names))
    if actual_barcodes != expected_barcodes:
        raise ValueError(
            f"{sample_id}: expression spot identities differ from the immutable manifest "
            f"(missing={sorted(expected_barcodes - actual_barcodes)[:5]}, "
            f"unexpected={sorted(actual_barcodes - expected_barcodes)[:5]}); rebuild the manifest"
        )
    return adata


def load_sample_for_examples(
    manifest: dict, sample_id: str, hest_data_dir: str | Path | None = None,
) -> tuple["ad.AnnData", np.ndarray, np.ndarray]:  # noqa: F821 -- anndata imported lazily below
    """Real, QC'd, gene-panel-aligned expression AND H&E patches for one
    manifest sample, ready for build_spatial_field_example.

    Reads every QC/normalization parameter FROM the manifest's own
    `build_args` -- never re-specified by the caller -- so the spot/gene
    universe this function produces is guaranteed identical to what the
    manifest already declared exists for this sample. One authoritative
    source of truth (the manifest), not two independently-configured
    ones that could silently drift apart.

    18th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED
    real: a prior version let `loaders.align_patches_to_adata` SILENTLY
    DROP any spot with no matching H&E patch (a normal ~4.5% HEST-1k
    gap) -- but realized masks (mask_bank.py) are generated against the
    MANIFEST's expression-QC spot set (dataset_manifest.py), which never
    accounts for H&E-patch availability at all. A mask could therefore
    reference a barcode this function used to silently drop, making
    `build_spatial_field_example` raise (the barcode is absent from
    `adata.obs_names`) far downstream of manifest/mask construction --
    or worse, contradicting the whole point of `observed_image_available`
    (a spot with GEX but no H&E should be RETAINED as GEX-only, not
    dropped outright). Fixed: `align_patches_to_adata` no longer drops
    ANY spot -- the third return value, `image_source_available`, is a
    real per-spot boolean (aligned with the returned `adata`/`patches`)
    marking which spots actually have a real H&E patch on disk; the
    barcode universe this function returns is now EXACTLY the manifest's
    declared set (checked both directions below), never a silently-
    shrunk subset."""
    hest_data_dir = Path(hest_data_dir) if hest_data_dir is not None else Path(manifest["hest_data_dir"])
    record = manifest["samples"][sample_id]
    adata = load_expression_for_model_target_space(manifest, sample_id, hest_data_dir)

    patches, patch_barcodes = loaders.load_hest_patches(hest_data_dir, sample_id)
    adata, patches, image_source_available = loaders.align_patches_to_adata(adata, patches, patch_barcodes)

    manifest_barcodes = set(record["barcodes"])
    unexpected = set(adata.obs_names) - manifest_barcodes
    if unexpected:
        raise ValueError(
            f"{sample_id}: real data now has {len(unexpected)} spot(s) the manifest never "
            f"declared (examples: {sorted(unexpected)[:5]}) -- the real data changed since the "
            "manifest was built; rebuild the manifest"
        )
    missing = manifest_barcodes - set(adata.obs_names)
    if missing:
        raise ValueError(
            f"{sample_id}: real data is now missing {len(missing)} spot(s) the manifest "
            f"declared (examples: {sorted(missing)[:5]}) -- the real data changed since the "
            "manifest was built; rebuild the manifest"
        )
    return adata, patches, image_source_available


def _median_nearest_neighbor_spacing(coords: np.ndarray) -> float:
    """Robust per-sample spot-spacing unit -- the median nearest-OTHER-
    spot distance across the WHOLE real spot lattice (not just one
    example's observed/query subset, which would fluctuate mask-to-mask
    for no biological reason)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return 1.0
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)  # k=2: self (distance 0) + nearest other spot
    return float(np.median(dist[:, 1]))


def build_spatial_field_example(
    adata: "ad.AnnData",  # noqa: F821
    patches: np.ndarray,
    context_barcodes: list[str],
    query_barcodes: list[str],
    image_feature_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    sample_id: str,
    patient_id: str,
    full_sample_coords: np.ndarray | None = None,
    require_full_sample_coords: bool = True,
    patch_size_fullres: float = 224.0,
    k_neighbors: int = 6,
    local_k: int = 32,
    max_rings: int = 3,
    max_boundary_size: int | None = None,
    expected_feature_width: int | None = None,
    slide_context: dict | None = None,
    image_mode: str = "target_zero",
    image_source_available: np.ndarray | None = None,
    precomputed_spot_features: np.ndarray | None = None,
) -> tuple[SpatialFieldInputs, SpatialFieldTargets]:
    """Build one real SpatialFieldInputs/SpatialFieldTargets pair from
    one realized (context, query) barcode split.

    `adata`/`patches` must already be QC'd, gene-panel-aligned, and
    barcode-aligned to each other (load_sample_for_examples's contract).

    EXACTLY ONE of `image_feature_fn` or `precomputed_spot_features` must
    be given -- mutually exclusive, never both, never neither (21st
    Codex re-audit, Step 5/6 boundary #2, CONFIRMED real: "the spot
    cache is not yet cleanly consumable -- build_spatial_field_example()
    still receives image_feature_fn(patches). That callback receives
    neither barcodes nor aligned positions, so it cannot safely slice
    the newly cached feature matrix"). `image_feature_fn` computes
    per-spot image features from raw patches (e.g. a real GigaPath
    forward pass) -- injected rather than called directly here, so this
    module has no hard dependency on a real GigaPath checkpoint and is
    fully testable with a cheap stub, matching every other pluggable-
    feature-function pattern already established in this codebase
    (models/slide_encoder.py, gen2_architectures' context_features.py).
    `precomputed_spot_features` is a `[adata.n_obs, feature_dim]` array
    aligned EXACTLY with `adata.obs_names` (the same row order/contract
    `spot_feature_cache.load_gen3_spot_features` already returns its
    `features` in) -- the production path: this function selects
    `precomputed_spot_features[context_pos[available_pos]]` directly,
    never calling any encoder, satisfying "never invoke the frozen
    GigaPath tile encoder per training example" (CONTRACT.md section 44)
    concretely rather than merely by convention. The real trainer (Step
    6) must use `precomputed_spot_features`, never `image_feature_fn` --
    `image_feature_fn` remains for tests/smoke scripts that have no
    precomputed cache to load.

    `full_sample_coords` (default: this example's own observed+query
    union) lets a caller pass the sample's COMPLETE coordinate lattice
    for a stable, mask-independent spot-spacing unit -- the recommended
    real usage. Falling back to the example's own subset is a
    mask-dependent spacing unit that fluctuates example-to-example for
    no biological reason (10th Codex re-audit of commit 9592d9e, finding
    #5), so it is REQUIRED (raises if omitted) unless the caller
    explicitly passes `require_full_sample_coords=False` -- a
    deliberate, named escape hatch reserved for small/synthetic tests
    where the whole-sample lattice isn't conveniently available; the
    real trainer must never set it.

    `expected_feature_width` (default: no check) lets a caller assert
    `image_feature_fn`'s real output width (e.g. GigaPath's 1536) as an
    extra fail-closed guard.

    `slide_context` (default: None -- no regional/global WSI context) is
    a real `slide_context.load_slide_context(...)` result -- the whole,
    UNMASKED tissue-wide GigaPath tile cache for this sample.
    `slide_context.visible_slide_context` removes every tile whose
    footprint physically overlaps this example's query hole (16th Codex
    re-audit, Step 5 Part 2: "Every WSI tile physically overlapping the
    hole is excluded"), so architectures.py's regional-token pooling and
    LongNet global-vector computation only ever see tiles that could
    still exist after the same physical damage query spots model.

    Two GENUINELY DIFFERENT physical coordinate frames come out of
    `visible_slide_context`, never mixed (17th Codex re-audit, Step 5
    Part 2 launch blocker #1, CONFIRMED real: an earlier version derived
    regional coordinates from `visible["coords"]` -- GigaPath's own
    target-MPP LongNet frame -- minus the ST-level-0 reference/scale,
    which is physically invalid whenever the cache's source MPP differs
    from GigaPath's target MPP): `wsi_tile_longnet_coords` keeps
    `visible["coords"]` UNCHANGED, fed to real GigaPath/LongNet inference
    as-is; `wsi_tile_regional_coords` is derived from
    `visible["level0_coords"]` (the level-0/HEST-aligned tile CENTER
    coordinates -- the SAME physical frame observed_coords/query_coords
    are already expressed in) via the documented `(level0_coords -
    reference) / scale` transform, so regional spatial attention shares
    an honest, physically consistent frame with every other coordinate
    in this example. `full_slide_coord_bounds` is computed from the
    COMPLETE (pre-hole-filtering) level-0 tile-center set
    (`slide_context.tile_centers`), in that same transform, so a
    regional grid cell's spatial meaning stays stable across different
    holes on the same slide (16th Codex re-audit: "Regional-grid bounds
    come from the complete slide before masking"). `slide_cache_namespace`
    is `visible_slide_context`'s own `context_id` -- already bound to
    real tile-cache content, the visible tile set, and this example's
    hole (15th/16th Codex re-audits).

    `image_mode` accepts only `"target_zero"` today -- `all_zero`/
    `shuffled`/`full` are not yet implemented consistently across the
    WSI-context and spot-H&E-availability paths (e.g. `all_zero`
    currently removes WSI context while leaving spot H&E features
    populated); any other value raises rather than silently producing
    an inconsistent example.

    `image_source_available` (default: None -- every spot assumed to
    have a real H&E patch on disk) is `loaders.align_patches_to_adata`'s
    real per-spot boolean, aligned with `adata`/`patches` (18th Codex
    re-audit, Step 5 Part 2 launch blocker #1): `load_sample_for_examples`
    now RETAINS every manifest-declared spot even when it has no
    matching H&E patch (never silently drops it -- see that function's
    own docstring for why silently dropping broke the manifest/mask
    contract), so a context spot's real H&E availability is now TWO
    independent facts ANDed together --
    `observed_image_available = image_source_available[context] &
    nonoverlapping_context_patch_mask(...)` -- missing on disk, or
    physically overlapping the hole. Either reason zeroes
    `observed_gigapath_features` and excludes the spot from
    `image_feature_fn`'s input identically; the None default (every
    source available) is safe only because every REAL caller
    (`load_sample_for_examples`) always supplies the true array
    explicitly -- reserved for test fixtures that build patches with no
    missing rows.
    """
    if full_sample_coords is None and require_full_sample_coords:
        raise ValueError(
            f"{sample_id}: full_sample_coords is required (the complete aligned sample "
            "lattice, for a stable mask-independent spot-spacing unit) -- pass it explicitly, "
            "or require_full_sample_coords=False for small/synthetic tests only"
        )
    if slide_context is not None and full_sample_coords is None:
        # 17th Codex re-audit, Step 5 Part 2 launch blocker #2, CONFIRMED
        # real: regional-grid stability across masks fundamentally
        # requires a coordinate reference derived from the COMPLETE
        # sample lattice, never a per-example subset -- see the
        # reference/scale computation below. Without full_sample_coords
        # there is no complete lattice to derive it from, so WSI context
        # cannot be safely combined with the require_full_sample_coords=False
        # escape hatch.
        raise ValueError(
            f"{sample_id}: slide_context requires full_sample_coords (regional-grid stability "
            "across masks depends on a coordinate reference derived from the complete sample "
            "lattice, not a per-example subset) -- pass full_sample_coords explicitly"
        )
    # 18th Codex re-audit (Step 5 Part 2, "Other real gaps"), CONFIRMED
    # real: this check used to run ONLY when slide_context was given --
    # a caller building an Architecture 1/2 example (no WSI context at
    # all) could pass image_mode="all_zero"/"full"/"shuffled" and it
    # would be silently ignored (image_mode is never even read when
    # slide_context is None), giving the false impression the mode had
    # some effect. Unconditional now: an unsupported image_mode always
    # raises, regardless of whether slide_context happens to be set.
    if image_mode not in _SUPPORTED_IMAGE_MODES:
        raise ValueError(
            f"{sample_id}: image_mode={image_mode!r} is not yet implemented consistently -- "
            f"only {sorted(_SUPPORTED_IMAGE_MODES)} is supported until all_zero/shuffled/full are "
            "made consistent across the WSI-context and spot-H&E-availability paths"
        )

    obs_names = np.asarray(adata.obs_names, dtype=str)
    barcode_to_pos = {b: i for i, b in enumerate(obs_names)}

    missing_context = [b for b in context_barcodes if b not in barcode_to_pos]
    missing_query = [b for b in query_barcodes if b not in barcode_to_pos]
    if missing_context or missing_query:
        raise ValueError(
            f"{sample_id}: mask references barcodes absent from the aligned sample data -- "
            f"context missing {missing_context[:5]}, query missing {missing_query[:5]}"
        )
    if set(context_barcodes) & set(query_barcodes):
        raise ValueError(f"{sample_id}: context and query barcodes overlap")
    if not context_barcodes:
        raise ValueError(f"{sample_id}: context_barcodes is empty -- no context to condition on")
    if not query_barcodes:
        raise ValueError(f"{sample_id}: query_barcodes is empty -- nothing to predict")
    if patches.shape[0] != adata.n_obs:
        raise ValueError(
            f"{sample_id}: patches has {patches.shape[0]} rows but adata has {adata.n_obs} spots -- "
            "patches must already be aligned to adata (load_sample_for_examples's contract)"
        )
    if image_source_available is None:
        image_source_available_arr = np.ones(adata.n_obs, dtype=bool)
    else:
        image_source_available_arr = np.asarray(image_source_available, dtype=bool)
        if image_source_available_arr.shape != (adata.n_obs,):
            raise ValueError(
                f"{sample_id}: image_source_available has shape {image_source_available_arr.shape}, "
                f"expected ({adata.n_obs},) aligned with adata/patches"
            )

    # 21st Codex re-audit, Step 5/6 boundary #2, CONFIRMED real: the
    # production path needs a way to slice a precomputed, barcode-
    # aligned feature matrix directly -- image_feature_fn only ever
    # receives raw pixel patches, never barcodes or absolute positions,
    # so it cannot safely index into spot_feature_cache's cached array.
    # Mutually exclusive with image_feature_fn: never both (ambiguous
    # which one is authoritative), never neither (nothing would ever
    # populate observed_gigapath_features).
    if (image_feature_fn is None) == (precomputed_spot_features is None):
        raise ValueError(
            f"{sample_id}: exactly one of image_feature_fn or precomputed_spot_features must be "
            "given -- never both, never neither"
        )
    precomputed_spot_features_arr = None
    if precomputed_spot_features is not None:
        precomputed_spot_features_arr = np.asarray(precomputed_spot_features, dtype=np.float32)
        if precomputed_spot_features_arr.ndim != 2 or precomputed_spot_features_arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"{sample_id}: precomputed_spot_features must be [N, feature_dim] aligned with "
                f"adata.obs_names ({adata.n_obs} rows), got shape {precomputed_spot_features_arr.shape}"
            )
        if not np.isfinite(precomputed_spot_features_arr).all():
            raise ValueError(f"{sample_id}: precomputed_spot_features contains non-finite values")

    all_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if all_coords.ndim != 2 or all_coords.shape[1] != 2:
        raise ValueError(f"{sample_id}: adata.obsm['spatial'] must be [N, 2], got shape {all_coords.shape}")
    if not np.isfinite(all_coords).all():
        raise ValueError(f"{sample_id}: adata.obsm['spatial'] contains non-finite coordinates")
    if np.unique(all_coords, axis=0).shape[0] != all_coords.shape[0]:
        raise ValueError(
            f"{sample_id}: adata.obsm['spatial'] has duplicate spot coordinates -- corrupted or "
            "misaligned real data"
        )

    query_pos = np.asarray([barcode_to_pos[b] for b in query_barcodes], dtype=int)
    query_coords_raw = all_coords[query_pos]

    context_pos_all = np.asarray([barcode_to_pos[b] for b in context_barcodes], dtype=int)
    context_coords_all_raw = all_coords[context_pos_all]

    # 15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED real:
    # a prior version DROPPED an entire context spot (both GEX and H&E)
    # whenever its H&E patch FOOTPRINT physically overlapped the query
    # hole -- but that spot's GEX is still real, measured, and available
    # (GEX and imaging are independent real-world failure modes; imaging
    # can fail locally while transcriptomics stays readable). Every
    # GEX-available context spot is now RETAINED; H&E availability is
    # tracked as an explicit per-spot flag instead
    # (SpatialFieldInputs.observed_image_available), consumed by
    # SpotTokenProjection's modality_flags input -- never silently
    # inferred from a zero-valued image feature.
    context_pos = context_pos_all
    # 18th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED
    # real: H&E availability now has TWO independent real-world causes
    # -- no matching patch on disk at all (image_source_available,
    # real per-spot data from load_sample_for_examples/
    # align_patches_to_adata) and physical overlap with THIS example's
    # hole (nonoverlapping_context_patch_mask, geometry-only, mask-
    # dependent) -- ANDed together; either reason alone must zero the
    # feature and exclude the spot from image_feature_fn identically.
    context_image_source_available = image_source_available_arr[context_pos]
    physically_nonoverlapping = nonoverlapping_context_patch_mask(
        context_coords_all_raw, query_coords_raw, patch_size_fullres,
    )
    observed_image_available = context_image_source_available & physically_nonoverlapping
    n_image_source_unavailable = int((~context_image_source_available).sum())
    n_physical_overlap_unavailable = int((~physically_nonoverlapping).sum())
    n_image_unavailable = int((~observed_image_available).sum())

    observed_barcodes = obs_names[context_pos]
    observed_coords_raw = all_coords[context_pos]

    if full_sample_coords is not None:
        # 11th Codex re-audit of commit 9dab8fe, finding #3 ("validate
        # full_sample_coords itself -- shape, finiteness, uniqueness,
        # row count and agreement with the aligned sample. Currently
        # only adata.obsm['spatial'] receives those checks."): a caller
        # can pass ANY array here -- it must be held to the same
        # fail-closed standard as adata.obsm['spatial'] itself, plus an
        # explicit check that it actually IS the aligned sample's own
        # complete lattice (a superset containing every one of
        # all_coords' rows), not some unrelated or mismatched array.
        full_coords_arr = np.asarray(full_sample_coords, dtype=np.float64)
        if full_coords_arr.ndim != 2 or full_coords_arr.shape[1] != 2:
            raise ValueError(
                f"{sample_id}: full_sample_coords must be [M, 2], got shape {full_coords_arr.shape}"
            )
        if not np.isfinite(full_coords_arr).all():
            raise ValueError(f"{sample_id}: full_sample_coords contains non-finite coordinates")
        if np.unique(full_coords_arr, axis=0).shape[0] != full_coords_arr.shape[0]:
            raise ValueError(f"{sample_id}: full_sample_coords has duplicate rows")
        if full_coords_arr.shape[0] < all_coords.shape[0]:
            raise ValueError(
                f"{sample_id}: full_sample_coords has {full_coords_arr.shape[0]} rows, fewer than "
                f"the aligned sample's {all_coords.shape[0]} spots -- it must be the COMPLETE lattice"
            )
        full_coords_set = {tuple(row) for row in full_coords_arr}
        n_missing = sum(1 for row in all_coords if tuple(row) not in full_coords_set)
        if n_missing:
            raise ValueError(
                f"{sample_id}: full_sample_coords does not agree with the aligned sample -- "
                f"{n_missing} of the sample's own coordinates are absent from it"
            )
        spacing_source = full_coords_arr
        # 17th Codex re-audit, Step 5 Part 2 launch blocker #2, CONFIRMED
        # real: a prior version derived `reference` from ONLY this
        # example's own observed+query subset -- if context is capped,
        # reserved, filtered, or otherwise incomplete, the coordinate
        # ORIGIN itself shifts between masks on the identical sample,
        # so the SAME physical WSI tile would land at different
        # regional coordinates (and potentially a different regional
        # grid cell) depending purely on which mask happened to be
        # realized -- contradicting the slide-stable regional-grid
        # contract "Regional-grid bounds come from the complete slide
        # before masking, so regions remain spatially stable across
        # holes." Deriving BOTH reference and scale from the SAME
        # complete, validated sample lattice makes the whole coordinate
        # system (observed_coords/query_coords AND the WSI regional
        # frame) mask-independent, not just the scale unit -- and is a
        # more literal reading of the original 6th Codex re-audit
        # recommendation ("coordinates relative to the hole OR SLIDE
        # CENTRE") than a per-example subset centroid ever was.
        reference_source = full_coords_arr
    else:
        spacing_source = np.concatenate([context_coords_all_raw, query_coords_raw], axis=0)
        reference_source = spacing_source
    scale = max(_median_nearest_neighbor_spacing(spacing_source), 1e-6)
    reference = reference_source.mean(axis=0)
    observed_coords = ((observed_coords_raw - reference) / scale).astype(np.float32)
    query_coords = ((query_coords_raw - reference) / scale).astype(np.float32)

    wsi_tile_longnet_coords = None
    wsi_tile_regional_coords = None
    wsi_tile_features = None
    full_slide_coord_bounds = None
    slide_cache_namespace = None
    if slide_context is not None:
        visible = visible_slide_context(slide_context, query_coords_raw, image_mode, patch_size_fullres)
        if visible["available"]:
            # GigaPath's own target-MPP frame -- fed to LongNet UNCHANGED,
            # never combined with the level-0/HEST-aligned reference/scale.
            wsi_tile_longnet_coords = np.asarray(visible["coords"], dtype=np.float32)
            # The level-0/HEST-aligned tile CENTER coordinates -- the
            # SAME physical frame observed_coords/query_coords/
            # spacing_source are already expressed in, so this transform
            # is physically valid regardless of the cache's source MPP.
            level0_visible = np.asarray(visible["level0_coords"], dtype=np.float32)
            wsi_tile_regional_coords = ((level0_visible - reference) / scale).astype(np.float32)
            wsi_tile_features = np.asarray(visible["features"], dtype=np.float32)
            # Bounds come from the COMPLETE (unmasked) level-0 tile-center
            # set, in the same normalized frame, so a regional grid cell
            # keeps the same spatial meaning regardless of which hole
            # this particular example carries (16th Codex re-audit).
            full_slide_level0 = tile_centers(slide_context).astype(np.float32)
            full_slide_regional = (full_slide_level0 - reference) / scale
            full_slide_coord_bounds = (
                float(full_slide_regional[:, 0].min()), float(full_slide_regional[:, 0].max()),
                float(full_slide_regional[:, 1].min()), float(full_slide_regional[:, 1].max()),
            )
            slide_cache_namespace = str(visible["context_id"])

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    observed_full_gene_expression = np.asarray(X[context_pos], dtype=np.float32)
    query_expression = np.asarray(X[query_pos], dtype=np.float32)

    # Only feed AVAILABLE patches/rows to image_feature_fn/
    # precomputed_spot_features -- a spot whose H&E overlaps the
    # synthetic hole models a REAL deployment scenario where that patch
    # would not exist; its real (undamaged, in this training setup)
    # pixels/cached feature must never reach the model just because they
    # happen to still be present on disk/in the cache.
    available_pos = np.flatnonzero(observed_image_available)
    if precomputed_spot_features_arr is not None:
        # Production path: a direct slice into a barcode-aligned,
        # already-encoded feature matrix -- no encoder call of any kind.
        # `context_pos[available_pos]` are absolute row indices into
        # `adata`/`patches`/`precomputed_spot_features_arr`'s SHARED
        # alignment (all three are row-aligned to adata.obs_names), so
        # this selects exactly the same physical spots image_feature_fn
        # would otherwise have been asked to encode from raw pixels --
        # never a query-spot row (query spots never appear in
        # context_pos) and never an unavailable spot's row (excluded by
        # available_pos).
        computed_features = precomputed_spot_features_arr[context_pos[available_pos]]
        feature_width = precomputed_spot_features_arr.shape[1]
    elif available_pos.size > 0:
        computed_features = np.asarray(image_feature_fn(patches[context_pos[available_pos]]), dtype=np.float32)
        if computed_features.ndim != 2:
            raise ValueError(
                f"{sample_id}: image_feature_fn must return a 2D [N, feature_dim] array, got shape "
                f"{computed_features.shape}"
            )
        if computed_features.shape[0] != available_pos.shape[0]:
            raise ValueError(
                f"{sample_id}: image_feature_fn returned {computed_features.shape[0]} rows for "
                f"{available_pos.shape[0]} available observed patches"
            )
        if not np.isfinite(computed_features).all():
            raise ValueError(f"{sample_id}: image_feature_fn returned non-finite feature values")
        feature_width = computed_features.shape[1]
    elif expected_feature_width is not None:
        computed_features = np.zeros((0, expected_feature_width), dtype=np.float32)
        feature_width = expected_feature_width
    else:
        raise ValueError(
            f"{sample_id}: every context spot's H&E patch overlaps the query hole (all "
            f"{context_pos.shape[0]} image-unavailable) -- pass expected_feature_width so a "
            "correctly-shaped all-zero observed_gigapath_features can be built"
        )
    if expected_feature_width is not None and feature_width != expected_feature_width:
        raise ValueError(
            f"{sample_id}: image feature width {feature_width} (from "
            f"{'precomputed_spot_features' if precomputed_spot_features_arr is not None else 'image_feature_fn'}), "
            f"expected {expected_feature_width}"
        )
    observed_gigapath_features = np.zeros((context_pos.shape[0], feature_width), dtype=np.float32)
    observed_gigapath_features[available_pos] = computed_features

    boundary = extract_boundary_and_local_context(
        observed_coords, query_coords, k_neighbors=k_neighbors, local_k=local_k,
        max_rings=max_rings, max_boundary_size=max_boundary_size,
    )

    inputs = SpatialFieldInputs(
        sample_id=sample_id,
        patient_id=patient_id,
        observed_barcodes=observed_barcodes,
        query_barcodes=np.asarray(query_barcodes, dtype=str),
        observed_coords=observed_coords,
        query_coords=query_coords,
        observed_full_gene_expression=observed_full_gene_expression,
        observed_gigapath_features=observed_gigapath_features,
        observed_image_available=observed_image_available,
        query_local_neighbor_idx=boundary.query_local_neighbor_idx,
        boundary_idx=boundary.boundary_idx,
        boundary_ring=boundary.boundary_ring,
        query_depth_to_boundary=boundary.query_depth_to_boundary,
        wsi_tile_longnet_coords=wsi_tile_longnet_coords,
        wsi_tile_regional_coords=wsi_tile_regional_coords,
        wsi_tile_features=wsi_tile_features,
        full_slide_coord_bounds=full_slide_coord_bounds,
        slide_cache_namespace=slide_cache_namespace,
        provenance={
            "n_context_requested": len(context_barcodes),
            # 18th Codex re-audit (Step 5 Part 2 launch blocker #1): now
            # two independent, separately-tracked reasons -- missing on
            # disk (image_source_available) vs. physical hole overlap
            # (mask-dependent geometry) -- plus their combined total.
            "n_context_image_unavailable_for_missing_source_patch": n_image_source_unavailable,
            "n_context_image_unavailable_for_physical_he_overlap": n_physical_overlap_unavailable,
            "n_context_image_unavailable_total": n_image_unavailable,
            "spot_spacing_scale": scale,
            # 17th Codex re-audit, Step 5 Part 2 launch blocker #2: the
            # coordinate reference is now recorded explicitly (previously
            # implicit and mask-dependent) so a caller/test can verify
            # two examples on the same sample share the identical origin.
            "coordinate_reference": reference.tolist(),
            "wsi_context_available": wsi_tile_features is not None,
            **boundary.diagnostic,
        },
    )
    targets = SpatialFieldTargets(query_expression=query_expression)
    validate_spatial_field_example(inputs, targets)
    return inputs, targets
