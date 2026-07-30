"""Arm D (STPath context-only) example construction -- GEN4_CONTRACT.md
section 8. Unlike arms A-C (whose per-spot image representation is a
row-independent, precomputed-once-per-sample cache), STPath's contextual
mixing genuinely depends on which spots are in a given mask's context set,
so its representation is computed per-mask, here, at example-construction
time -- never inside `Gen4Conditioner.forward()` and never cached
independent of the mask.
"""
from __future__ import annotations

import numpy as np
import torch

from gen3_multiscale.gen4.inputs import Gen4SpatialFieldInputs, validate_gen4_spatial_field_example
from gen3_multiscale.data.example import SpatialFieldTargets
from gen3_multiscale.data.example_builder import build_spatial_field_example


def build_gen4_stpath_example(
    adata: "ad.AnnData",  # noqa: F821
    patches: np.ndarray,
    context_barcodes: list[str],
    query_barcodes: list[str],
    *,
    sample_id: str,
    patient_id: str,
    gigapath_tile_features: np.ndarray,
    stpath_encoder,
    full_sample_coords: np.ndarray | None = None,
    require_full_sample_coords: bool = True,
    image_source_available: np.ndarray | None = None,
) -> tuple[Gen4SpatialFieldInputs, SpatialFieldTargets]:
    """`gigapath_tile_features` is a `[adata.n_obs, 1536]` array aligned to
    `adata.obs_names` (STPath's own image tokenizer input -- the SAME
    per-spot GigaPath cache Gen3/arm B already use; STPath needs no
    separate image encoder of its own). `stpath_encoder` is a
    `gen4.stpath_context.Gen4STPathContextEncoder` (or
    `tests._gen4_fixtures.Gen4STPathStub`), exposing `encode_context_only`.

    First builds an ordinary base example (via the unmodified
    `example_builder.build_spatial_field_example`) purely to obtain the
    real, final `observed_*` arrays in their correct order -- its
    `observed_gigapath_features` field is then REPLACED with STPath's
    context-only representation, never with the raw GigaPath tile features
    a caller of this function would otherwise see in that slot (matching
    GEN4_CONTRACT.md section 4 divergence 3: STPath's output stands in for
    the whole image branch for arm D)."""
    base_inputs, targets = build_spatial_field_example(
        adata, patches, context_barcodes, query_barcodes,
        image_feature_fn=None, sample_id=sample_id, patient_id=patient_id,
        full_sample_coords=full_sample_coords, require_full_sample_coords=require_full_sample_coords,
        image_mode="target_zero", image_source_available=image_source_available,
        precomputed_spot_features=gigapath_tile_features,
    )

    context_coords = torch.as_tensor(base_inputs.observed_coords, dtype=torch.float32)
    context_expression = torch.as_tensor(base_inputs.observed_full_gene_expression, dtype=torch.float32)
    context_image_features = torch.as_tensor(base_inputs.observed_gigapath_features, dtype=torch.float32)
    context_image_available = torch.as_tensor(base_inputs.observed_image_available, dtype=torch.bool)
    with torch.no_grad():
        stpath_embedding = stpath_encoder.encode_context_only(
            context_coords, context_expression, context_image_features, context_image_available,
        )
    stpath_embedding = stpath_embedding.detach().cpu().numpy().astype(np.float32)
    if stpath_embedding.shape[0] != base_inputs.observed_coords.shape[0]:
        raise ValueError(
            f"{sample_id}: STPath context-only output has {stpath_embedding.shape[0]} rows, expected "
            f"{base_inputs.observed_coords.shape[0]} (one per observed spot)"
        )

    base_fields = {field: getattr(base_inputs, field) for field in base_inputs.__dataclass_fields__}
    base_fields["observed_gigapath_features"] = stpath_embedding
    # STPath's per-context-spot output is always a real, defined embedding
    # (its own internal missing_image_token already stands in for a spot
    # with no real H&E patch -- see Gen4STPathContextEncoder.encode_context_only)
    # -- never a garbage row -- so the base "unavailable image => explicit
    # zero row" invariant no longer applies once this field holds STPath's
    # output rather than raw per-spot tile features. Every context row is
    # therefore marked available for arm D; the ORIGINAL per-spot
    # image-availability signal is still what determined which spots got
    # `missing_image_token` inside STPath itself, one layer up.
    base_fields["observed_image_available"] = np.ones_like(base_inputs.observed_image_available, dtype=bool)
    gen4_inputs = Gen4SpatialFieldInputs(**base_fields, context_gex_embedding=None, context_gex_embedding_provenance=None)
    validate_gen4_spatial_field_example(gen4_inputs, targets)
    return gen4_inputs, targets
