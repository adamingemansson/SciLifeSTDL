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
point. Such spots are EXCLUDED from the observed set entirely here (not
merely from the barcode-level check), via
slide_context.nonoverlapping_context_patch_mask -- already-audited
overlap geometry, reused, not reimplemented. Query spots are already
structurally absent from every observed_* array (validate_spatial_field_example
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
from gen3_multiscale.data.slide_context import nonoverlapping_context_patch_mask


def load_sample_for_examples(
    manifest: dict, sample_id: str, hest_data_dir: str | Path | None = None,
) -> tuple["ad.AnnData", np.ndarray]:  # noqa: F821 -- anndata imported lazily below
    """Real, QC'd, gene-panel-aligned expression AND H&E patches for one
    manifest sample, ready for build_spatial_field_example.

    Reads every QC/normalization parameter FROM the manifest's own
    `build_args` -- never re-specified by the caller -- so the spot/gene
    universe this function produces is guaranteed identical to what the
    manifest already declared exists for this sample. One authoritative
    source of truth (the manifest), not two independently-configured
    ones that could silently drift apart.

    KNOWN, DELIBERATE SCOPE BOUNDARY: the manifest's recorded spot
    barcodes (dataset_manifest.py) reflect the real EXPRESSION data's
    per-spot QC filter only -- it never loads H&E patches (a
    deliberately lightweight Step-1 artifact). `loaders.align_patches_to_adata`
    below can drop additional spots that have no matching H&E patch (a
    normal ~4.5% HEST-1k gap, not an error -- see that function's own
    docstring). The barcode universe this function actually returns can
    therefore be a SUBSET of the manifest's declared one; this is
    checked (never silently grows beyond the manifest, only shrinks for
    an already-documented reason) but not treated as an error."""
    hest_data_dir = Path(hest_data_dir) if hest_data_dir is not None else Path(manifest["hest_data_dir"])
    record = manifest["samples"][sample_id]
    args = manifest["build_args"]

    import scanpy as sc

    adata = loaders.load_hest_sample(hest_data_dir, sample_id, organ=record["organ"], tech=record["tech"])
    if args["gene_min_genes_per_spot"] > 0:
        sc.pp.filter_cells(adata, min_genes=args["gene_min_genes_per_spot"])
    # min_genes=0/min_cells=0 here: per-spot QC was just applied above
    # (matching dataset_manifest.py's identical filter exactly); gene-level
    # QC is already baked into manifest["gene_panel"] -- basic_qc_and_normalize
    # is called only for its normalize/log1p transform and raw-counts-layer
    # bookkeeping, not to filter anything a second time.
    adata = loaders.basic_qc_and_normalize(
        adata, min_genes=0, min_cells=0,
        transform=args["expression_transform"], target_sum=args["expression_target_sum"],
    )
    missing_genes = [g for g in manifest["gene_panel"] if g not in adata.var_names]
    if missing_genes:
        raise ValueError(
            f"{sample_id} is missing {len(missing_genes)} genes from the manifest's declared "
            f"gene_panel (examples: {missing_genes[:5]}) -- the real data changed since the "
            "manifest was built; rebuild the manifest"
        )
    adata = adata[:, manifest["gene_panel"]].copy()

    patches, patch_barcodes = loaders.load_hest_patches(hest_data_dir, sample_id)
    adata, patches = loaders.align_patches_to_adata(adata, patches, patch_barcodes)

    manifest_barcodes = set(record["barcodes"])
    unexpected = set(adata.obs_names) - manifest_barcodes
    if unexpected:
        raise ValueError(
            f"{sample_id}: real data now has {len(unexpected)} spot(s) the manifest never "
            f"declared (examples: {sorted(unexpected)[:5]}) -- the real data changed since the "
            "manifest was built; rebuild the manifest"
        )
    return adata, patches


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
    image_feature_fn: Callable[[np.ndarray], np.ndarray],
    *,
    sample_id: str,
    patient_id: str,
    full_sample_coords: np.ndarray | None = None,
    patch_size_fullres: float = 224.0,
    k_neighbors: int = 6,
    local_k: int = 32,
    max_rings: int = 3,
    max_boundary_size: int | None = None,
) -> tuple[SpatialFieldInputs, SpatialFieldTargets]:
    """Build one real SpatialFieldInputs/SpatialFieldTargets pair from
    one realized (context, query) barcode split.

    `adata`/`patches` must already be QC'd, gene-panel-aligned, and
    barcode-aligned to each other (load_sample_for_examples's contract).
    `image_feature_fn` computes per-spot image features from raw patches
    (e.g. the real GigaPath tile encoder) -- injected rather than called
    directly here, so this module has no hard dependency on a real
    GigaPath checkpoint and is fully testable with a cheap stub, matching
    every other pluggable-feature-function pattern already established
    in this codebase (models/slide_encoder.py, gen2_architectures'
    context_features.py).

    `full_sample_coords` (default: this example's own observed+query
    union) lets a caller pass the sample's COMPLETE coordinate lattice
    for a stable, mask-independent spot-spacing unit -- the recommended
    real usage; falling back to the example's own subset is only for
    small/synthetic tests where the whole-sample lattice isn't
    conveniently available.
    """
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

    all_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    query_pos = np.asarray([barcode_to_pos[b] for b in query_barcodes], dtype=int)
    query_coords_raw = all_coords[query_pos]

    context_pos_all = np.asarray([barcode_to_pos[b] for b in context_barcodes], dtype=int)
    context_coords_all_raw = all_coords[context_pos_all]

    # Physical H&E safety: exclude context spots whose patch FOOTPRINT
    # overlaps the query hole, not just ones whose barcode happens to be
    # a query barcode.
    safe_mask = nonoverlapping_context_patch_mask(context_coords_all_raw, query_coords_raw, patch_size_fullres)
    n_excluded_for_overlap = int((~safe_mask).sum())
    context_pos = context_pos_all[safe_mask]
    if context_pos.size == 0:
        raise ValueError(
            f"{sample_id}: every context spot's H&E patch overlaps the query hole -- no safe "
            "observed context remains"
        )

    observed_barcodes = obs_names[context_pos]
    observed_coords_raw = all_coords[context_pos]

    spacing_source = full_sample_coords if full_sample_coords is not None else np.concatenate(
        [context_coords_all_raw, query_coords_raw], axis=0,
    )
    scale = max(_median_nearest_neighbor_spacing(spacing_source), 1e-6)
    reference = np.concatenate([observed_coords_raw, query_coords_raw], axis=0).mean(axis=0)
    observed_coords = ((observed_coords_raw - reference) / scale).astype(np.float32)
    query_coords = ((query_coords_raw - reference) / scale).astype(np.float32)

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    observed_full_gene_expression = np.asarray(X[context_pos], dtype=np.float32)
    query_expression = np.asarray(X[query_pos], dtype=np.float32)

    observed_patches = patches[context_pos]
    observed_gigapath_features = np.asarray(image_feature_fn(observed_patches), dtype=np.float32)
    if observed_gigapath_features.shape[0] != context_pos.shape[0]:
        raise ValueError(
            f"{sample_id}: image_feature_fn returned {observed_gigapath_features.shape[0]} rows "
            f"for {context_pos.shape[0]} observed patches"
        )

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
        query_local_neighbor_idx=boundary.query_local_neighbor_idx,
        boundary_idx=boundary.boundary_idx,
        boundary_ring=boundary.boundary_ring,
        query_depth_to_boundary=boundary.query_depth_to_boundary,
        provenance={
            "n_context_requested": len(context_barcodes),
            "n_context_excluded_for_physical_he_overlap": n_excluded_for_overlap,
            "spot_spacing_scale": scale,
            **boundary.diagnostic,
        },
    )
    targets = SpatialFieldTargets(query_expression=query_expression)
    validate_spatial_field_example(inputs, targets)
    return inputs, targets
