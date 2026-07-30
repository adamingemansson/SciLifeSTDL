"""Gen4's extended input dataclass and example builder.

`Gen4SpatialFieldInputs` is a frozen-dataclass SUBCLASS of
`gen3_multiscale.data.example.SpatialFieldInputs` that adds exactly one new
optional field, `context_gex_embedding`, defaulted to `None`. It is purely
additive: `SpatialFieldInputs` itself is never edited, every existing Gen3
consumer of a plain `SpatialFieldInputs` is unaffected, and any code path
that never sets `context_gex_embedding` behaves identically to Gen3 (see
GEN4_CONTRACT.md section 5).

`build_gen4_spatial_field_example` does not reimplement example
construction -- it calls the existing, unmodified
`example_builder.build_spatial_field_example` for everything (coordinates,
boundary/local structure, image features, WSI context, targets) and then
attaches the GEX-context embedding as a barcode-keyed lookup against the
REAL, final `observed_barcodes` that function returned -- never by
replicating its internal barcode selection/reordering logic separately,
which would be a second, driftable source of truth for the same alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets
from gen3_multiscale.data.example_builder import build_spatial_field_example


@dataclass(frozen=True)
class Gen4SpatialFieldInputs(SpatialFieldInputs):
    """SpatialFieldInputs plus one optional frozen-GEX-context-provider
    field. `context_gex_embedding` is `[n_observed, gex_context_dim]`,
    position-aligned with every other `observed_*` array (same convention
    the base class already documents), or `None` for arms that use the
    ordinary trainable `WeightedGeneExpressionEncoder` path instead (arms
    A and D -- see GEN4_CONTRACT.md section 4)."""
    context_gex_embedding: np.ndarray | None = None
    context_gex_embedding_provenance: dict | None = None


def validate_gen4_spatial_field_example(inputs: Gen4SpatialFieldInputs, targets: SpatialFieldTargets) -> None:
    """Structural checks specific to the one new field; the entire base
    contract is re-verified by delegating to the existing, unmodified
    `data.example.validate_spatial_field_example` first."""
    from gen3_multiscale.data.example import validate_spatial_field_example

    validate_spatial_field_example(inputs, targets)
    if inputs.context_gex_embedding is None:
        return
    n_observed = inputs.observed_coords.shape[0]
    embedding = np.asarray(inputs.context_gex_embedding)
    if embedding.ndim != 2 or embedding.shape[0] != n_observed:
        raise ValueError(
            f"context_gex_embedding must be [n_observed={n_observed}, D], got shape {embedding.shape}"
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("context_gex_embedding contains non-finite values")


def _select_by_barcode(barcode_to_row: dict[str, np.ndarray], barcodes: np.ndarray, *, field_name: str) -> np.ndarray:
    missing = [str(b) for b in barcodes if str(b) not in barcode_to_row]
    if missing:
        raise KeyError(
            f"{field_name}: {len(missing)} observed barcode(s) have no entry in the provided "
            f"embedding lookup (examples: {missing[:5]}) -- rebuild the cache for this sample"
        )
    return np.stack([np.asarray(barcode_to_row[str(b)]) for b in barcodes], axis=0)


def build_gen4_spatial_field_example(
    adata: "ad.AnnData",  # noqa: F821
    patches: np.ndarray,
    context_barcodes: list[str],
    query_barcodes: list[str],
    *,
    sample_id: str,
    patient_id: str,
    precomputed_spot_features: np.ndarray,
    gex_context_embedding: dict[str, np.ndarray] | None = None,
    gex_context_provenance: dict | None = None,
    full_sample_coords: np.ndarray | None = None,
    require_full_sample_coords: bool = True,
    patch_size_fullres: float = 224.0,
    k_neighbors: int = 6,
    local_k: int = 32,
    max_rings: int = 3,
    max_boundary_size: int | None = None,
    expected_feature_width: int | None = None,
    slide_context: dict | None = None,
    image_source_available: np.ndarray | None = None,
) -> tuple[Gen4SpatialFieldInputs, SpatialFieldTargets]:
    """Build one Gen4 example. `precomputed_spot_features` is exactly the
    Gen3 contract (a `[adata.n_obs, image_feature_dim]` array aligned to
    `adata.obs_names`, e.g. UNI2's or GigaPath's own cache, or a STPath
    context-only encoder's output already reduced to per-spot rows by the
    caller -- see gen4/conditioner.py for which arms populate this from
    which encoder). `gex_context_embedding`, when given, is a
    `{barcode: row}` mapping (e.g. `dict(zip(cached["barcodes"],
    cached["features"]))`) covering at least every barcode this example's
    realized mask will place in `context_barcodes` -- arms without a frozen
    GEX-context provider (A, D) simply omit it, and `context_gex_embedding`
    on the returned object stays `None`.
    """
    inputs, targets = build_spatial_field_example(
        adata, patches, context_barcodes, query_barcodes,
        image_feature_fn=None,
        sample_id=sample_id, patient_id=patient_id,
        full_sample_coords=full_sample_coords, require_full_sample_coords=require_full_sample_coords,
        patch_size_fullres=patch_size_fullres, k_neighbors=k_neighbors, local_k=local_k, max_rings=max_rings,
        max_boundary_size=max_boundary_size, expected_feature_width=expected_feature_width,
        slide_context=slide_context, image_mode="target_zero",
        image_source_available=image_source_available, precomputed_spot_features=precomputed_spot_features,
    )

    context_embedding = None
    if gex_context_embedding is not None:
        context_embedding = _select_by_barcode(
            gex_context_embedding, inputs.observed_barcodes, field_name="context_gex_embedding",
        )

    base_fields = {field: getattr(inputs, field) for field in inputs.__dataclass_fields__}
    gen4_inputs = Gen4SpatialFieldInputs(
        **base_fields,
        context_gex_embedding=context_embedding,
        context_gex_embedding_provenance=gex_context_provenance,
    )
    validate_gen4_spatial_field_example(gen4_inputs, targets)
    return gen4_inputs, targets
