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
    # Codex audit finding #4: `wsi_tile_features` (inherited from the base
    # class) is populated by Gen3's existing dense-WSI tile cache, which is
    # GigaPath-encoded, not UNI2-encoded -- no real UNI2 dense-WSI cache
    # builder exists yet. Arm A/C's `global_context_source="uni2_pool"`
    # branch would otherwise silently consume those GigaPath-shaped
    # features as if they were UNI2 features. `Gen4Conditioner` requires
    # this field to equal exactly "uni2" before it will read
    # `wsi_tile_features` for that branch (gen4/conditioner.py's
    # `_global_slide_vector`) -- Gen3's builder never sets it, so the
    # pathway fails closed until a real, provenance-tagged UNI2 dense-WSI
    # cache exists.
    wsi_tile_feature_provenance: str | None = None
    # Arm 4 (intended-design hybrid, GEN4_CONTRACT.md-adjacent scope
    # extension): `observed_gigapath_features` (inherited) already serves
    # arm D/3's STPath image-tokenizer input (GigaPath-shaped, per its own
    # convention) and arm C/1's per-spot UNI2 cache when THAT arm alone
    # owns the field. The hybrid arm needs BOTH a GigaPath-shaped array to
    # feed STPath's own tokenizer AND a genuinely separate UNI2-shaped
    # per-spot morphology token at once -- one generic field cannot hold
    # both simultaneously. This field is that second, UNI2-specific
    # source, used ONLY by `image_feature_source="hybrid_context"`;
    # `None` for every other arm.
    observed_uni2_features: np.ndarray | None = None
    # Audit finding: STPath's organ token was fixed at construction time
    # ("Kidney" default), silently wrong for every non-Kidney sample in a
    # multi-organ QC2 dataset (Lung/Liver/Bowel). This is the real,
    # per-sample manifest organ -- one example is always exactly one
    # sample, so one organ. Required (fails closed) whenever arm D/3 or
    # arm 4's STPath-consuming path is used; `None` for every other arm.
    sample_organ: str | None = None


def validate_gen4_spatial_field_example(inputs: Gen4SpatialFieldInputs, targets: SpatialFieldTargets) -> None:
    """Structural checks specific to the new fields; the entire base
    contract is re-verified by delegating to the existing, unmodified
    `data.example.validate_spatial_field_example` first."""
    from gen3_multiscale.data.example import validate_spatial_field_example

    validate_spatial_field_example(inputs, targets)
    n_observed = inputs.observed_coords.shape[0]
    if inputs.context_gex_embedding is not None:
        embedding = np.asarray(inputs.context_gex_embedding)
        if embedding.ndim != 2 or embedding.shape[0] != n_observed:
            raise ValueError(
                f"context_gex_embedding must be [n_observed={n_observed}, D], got shape {embedding.shape}"
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError("context_gex_embedding contains non-finite values")
    if inputs.observed_uni2_features is not None:
        uni2_features = np.asarray(inputs.observed_uni2_features)
        if uni2_features.ndim != 2 or uni2_features.shape[0] != n_observed:
            raise ValueError(
                f"observed_uni2_features must be [n_observed={n_observed}, D], got shape {uni2_features.shape}"
            )
        if not np.all(np.isfinite(uni2_features)):
            raise ValueError("observed_uni2_features contains non-finite values")


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
    uni2_spot_embedding: dict[str, np.ndarray] | None = None,
    wsi_tile_feature_provenance: str | None = None,
    sample_organ: str | None = None,
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
    on the returned object stays `None`. This is also where a real
    scFoundation cache's output belongs (frozen_context/hybrid_context
    arms B/C/4) -- scFoundation has no separate parameter because its
    output is context_gex_embedding, exactly like every other frozen
    GEX-context provider.

    `uni2_spot_embedding` (audit finding #1: "the hybrid arm receives
    observed_uni2_features=None"), when given, is a `{barcode: row}`
    mapping (same shape/contract as `gex_context_embedding`, e.g. a real
    `gen4.uni2_spot_cache.load_uni2_spot_features` record turned into a
    dict) covering every realized context barcode -- used ONLY by arm 4
    (`image_feature_source="hybrid_context"`). Query rows are
    structurally absent from `inputs.observed_barcodes` (this function
    delegates realization entirely to `build_spatial_field_example`), so
    those are excluded the same way every other observed_* field is.

    Integration-audit finding #4 (CONFIRMED real bug, fixed): a context
    row's H&E can PHYSICALLY overlap the query hole even though its own
    barcode is disjoint from every query barcode (the patch footprint is
    a real square of pixels, not a point -- see example_builder.py's own
    docstring). `build_spatial_field_example` already computes exactly
    this per-row exclusion as `inputs.observed_image_available` and
    applies it when building `precomputed_spot_features` (a spot whose
    patch overlaps the hole gets an explicit zero row there, never its
    real cached feature). The FIRST version of this function selected
    `observed_uni2_features` directly from `uni2_spot_embedding` with NO
    reference to `observed_image_available` at all -- a context spot
    physically inside the hole's footprint would leak its real, cached
    UNI2 morphology feature into the hybrid arm's input even though the
    SAME spot's `precomputed_spot_features`/STPath-tokenizer row was
    correctly zeroed. `observed_uni2_features` is now explicitly zeroed
    at every row where `observed_image_available` is False, exactly
    mirroring `precomputed_spot_features`'s own contract -- see
    `test_build_gen4_spatial_field_example_zeroes_uni2_features_for_
    physically_overlapping_context_rows` for the adversarial proof.
    `None` for every other arm, leaving `observed_uni2_features` `None`
    on the returned object.

    `wsi_tile_feature_provenance` (audit finding #1) tags which encoder
    produced `slide_context`'s tiles -- pass `"uni2"` when `slide_context`
    came from `gen4.uni2_dense_wsi_cache.load_uni2_dense_wsi_context`,
    `None` (the default) for GigaPath's own `data.slide_context.
    load_slide_context` or when no dense-WSI context is used at all. See
    `Gen4SpatialFieldInputs.wsi_tile_feature_provenance`'s own docstring
    for why this field exists (arm A/C's `global_context_source=
    "uni2_pool"` must never silently consume GigaPath-encoded tiles).

    `sample_organ` (audit finding #3) is this sample's real manifest
    organ (e.g. `manifest["samples"][sample_id]["organ"]") -- required
    (fails closed in `Gen4Conditioner`) whenever arm D/3 or arm 4's
    STPath-consuming path is used; `None` for every other arm.
    """
    if wsi_tile_feature_provenance is not None and wsi_tile_feature_provenance != "uni2":
        raise ValueError(
            f"wsi_tile_feature_provenance must be 'uni2' or None, got {wsi_tile_feature_provenance!r}"
        )
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
    uni2_features = None
    if uni2_spot_embedding is not None:
        uni2_features = _select_by_barcode(
            uni2_spot_embedding, inputs.observed_barcodes, field_name="observed_uni2_features",
        )
        # Integration-audit finding #4: zero every row whose H&E is not
        # actually available (physically overlaps the query hole, or was
        # never available in the first place) -- the same exclusion
        # precomputed_spot_features already gets, applied here too since
        # this field is selected from a separate, unmasked source.
        unavailable = ~np.asarray(inputs.observed_image_available, dtype=bool)
        if unavailable.any():
            uni2_features = uni2_features.copy()
            uni2_features[unavailable] = 0.0

    base_fields = {field: getattr(inputs, field) for field in inputs.__dataclass_fields__}
    gen4_inputs = Gen4SpatialFieldInputs(
        **base_fields,
        context_gex_embedding=context_embedding,
        context_gex_embedding_provenance=gex_context_provenance,
        observed_uni2_features=uni2_features,
        wsi_tile_feature_provenance=wsi_tile_feature_provenance,
        sample_organ=sample_organ,
    )
    validate_gen4_spatial_field_example(gen4_inputs, targets)
    return gen4_inputs, targets
