"""Gen4Conditioner -- GEN4_CONTRACT.md sections 2-6.

A thin, structurally-minimal subclass of the existing, unmodified
`gen3_multiscale.models.architectures.Architecture3`. Every module Gen3
already audited (token projections, `SpatialFieldBackbone`,
`GeneValueTransportHead`, `InducedGlobalGEXPool`, `pool_regional_tokens`)
is inherited verbatim; only two forward-time decision points are
overridden, matching GEN4_CONTRACT.md section 4's own description of what
actually differs between arms:

1. `_observed_tokens`'s `gex_features` source (the ordinary trainable
   `WeightedGeneExpressionEncoder` on raw expression, or a frozen
   GEX-context embedding projected down to the same width).
2. `_global_slide_vector`'s source (GigaPath's real frozen LongNet vector,
   unchanged, or `MaskAwareCoordinateAttentionPool`'s trainable pool over
   UNI2 tiles, or no global branch at all for arm D).

`observed_gigapath_features` (image slot) needs no override at all: every
arm populates that SAME field name at the data layer (cache builders for
arms A-C, `gen4/stpath_example.py` for arm D) with whatever encoder's
per-spot representation that arm uses -- `_SharedFieldArchitecture` never
assumed the array in that field came specifically from GigaPath.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.models.architectures import Architecture3
from gen3_multiscale.models.geometry_utils import scatter_boundary_ring

_GEX_SOURCES = frozenset({"weighted_linear", "frozen_context"})
_GLOBAL_SOURCES = frozenset({"none", "gigapath", "uni2_pool"})

# Sentinel identity string used ONLY to satisfy _SharedFieldArchitecture's
# own constructor-time "gigapath_checkpoint_sha256 must match
# slide_encoder.checkpoint_sha256" identity check when the real global
# context source is a trainable UNI2 pool, not a frozen GigaPath
# checkpoint -- that check exists to stop a caller silently mismatching a
# checkpoint claim against reality; a trainable module has no checkpoint
# to claim at all, so both sides of the check are set to this same fixed,
# clearly-named constant rather than bypassing the check (which would
# require touching architectures.py).
_UNI2_POOL_SENTINEL_IDENTITY = "gen4-trainable-uni2-global-pool-not-a-frozen-checkpoint"


class Gen4Conditioner(Architecture3):
    def __init__(
        self,
        *,
        n_genes: int,
        gex_feature_dim: int,
        image_feature_dim: int = 1536,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        max_boundary_size: int | None = None,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        gene_gate_mode: str = "per_gene",
        use_query_gate: bool = True,
        use_residual: bool = False,
        residual_rank: int = 32,
        target_gene_scale: torch.Tensor | None = None,
        use_regional_he: bool = False,
        use_global_gex: bool = True,
        regional_grid_size: int = 4,
        n_gex_inducing: int = 16,
        harmonic_k_neighbors: int = 6,
        gex_feature_source: str = "weighted_linear",
        gex_context_embedding_dim: int | None = None,
        global_context_source: str = "none",
        global_slide_dim: int = 768,
        slide_encoder=None,
        gigapath_checkpoint_sha256: str | None = None,
        uni2_global_pool: nn.Module | None = None,
        model_architecture_version: str = "gen4-conditioner-v1",
    ):
        if gex_feature_source not in _GEX_SOURCES:
            raise ValueError(f"gex_feature_source must be one of {sorted(_GEX_SOURCES)}, got {gex_feature_source!r}")
        if global_context_source not in _GLOBAL_SOURCES:
            raise ValueError(f"global_context_source must be one of {sorted(_GLOBAL_SOURCES)}, got {global_context_source!r}")
        if gex_feature_source == "frozen_context" and not gex_context_embedding_dim:
            raise ValueError("gex_feature_source='frozen_context' requires a positive gex_context_embedding_dim")

        effective_slide_encoder = None
        effective_checkpoint_sha256 = None
        if global_context_source == "gigapath":
            effective_slide_encoder = slide_encoder
            effective_checkpoint_sha256 = gigapath_checkpoint_sha256
        elif global_context_source == "uni2_pool":
            if uni2_global_pool is None:
                raise ValueError("global_context_source='uni2_pool' requires a real uni2_global_pool module")
            # See _UNI2_POOL_SENTINEL_IDENTITY docstring above.
            uni2_global_pool.checkpoint_sha256 = _UNI2_POOL_SENTINEL_IDENTITY
            effective_slide_encoder = uni2_global_pool
            effective_checkpoint_sha256 = _UNI2_POOL_SENTINEL_IDENTITY

        super().__init__(
            n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale,
            use_regional_he=use_regional_he, use_global_gex=use_global_gex,
            use_global_slide=(global_context_source != "none"),
            global_slide_dim=global_slide_dim, n_gex_inducing=n_gex_inducing,
            harmonic_k_neighbors=harmonic_k_neighbors, regional_grid_size=regional_grid_size,
            slide_encoder=effective_slide_encoder, gigapath_checkpoint_sha256=effective_checkpoint_sha256,
            model_architecture_version=model_architecture_version,
        )
        self.gex_feature_source = gex_feature_source
        self.global_context_source = global_context_source

        if gex_feature_source == "frozen_context":
            # self.gene_encoder is still constructed by Architecture3's own
            # __init__ (reused unmodified -- see this module's docstring)
            # but is never called in _observed_tokens below for this arm.
            # It is left in place, rather than deleted, so the inherited
            # `_SharedFieldArchitecture.forward`'s own
            # `next(self.gene_encoder.parameters()).device` device-source
            # logic keeps working unmodified; its gradient is disabled so
            # it is genuinely inert (no optimizer update, excluded from
            # "trainable" counts by gen4/param_report.py's requires_grad
            # check) rather than merely unused-but-still-trainable.
            for parameter in self.gene_encoder.parameters():
                parameter.requires_grad_(False)
            self.gex_context_proj = nn.Sequential(
                nn.LayerNorm(gex_context_embedding_dim), nn.Linear(gex_context_embedding_dim, gex_feature_dim),
            )
        else:
            self.gex_context_proj = None

    def _observed_tokens(self, inputs, device: torch.device) -> torch.Tensor:
        if self.gex_feature_source == "weighted_linear":
            return super()._observed_tokens(inputs, device)

        context_embedding = getattr(inputs, "context_gex_embedding", None)
        if context_embedding is None:
            raise ValueError(
                "gex_feature_source='frozen_context' requires Gen4SpatialFieldInputs.context_gex_embedding "
                "to be set -- build the example with gen4.inputs.build_gen4_spatial_field_example and a "
                "real gex_context_embedding lookup"
            )
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed, torch.as_tensor(inputs.boundary_idx, device=device), torch.as_tensor(inputs.boundary_ring, device=device),
        )
        modality_flags = torch.as_tensor(inputs.observed_image_available, dtype=torch.float32, device=device).unsqueeze(-1)
        embedding_tensor = torch.as_tensor(context_embedding, dtype=torch.float32, device=device)
        gex_features = self.gex_context_proj(embedding_tensor)
        return self.spot_token(
            image_features=torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32, device=device),
            gex_features=gex_features,
            coords=torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device),
            boundary_ring=full_ring,
            modality_flags=modality_flags,
        )

    def _global_slide_vector(self, inputs, device: torch.device) -> torch.Tensor:
        if self.global_context_source == "gigapath":
            return super()._global_slide_vector(inputs, device)
        if self.global_context_source == "uni2_pool":
            self._require_wsi_context(inputs, "global_context_source='uni2_pool'")
            tile_features = torch.as_tensor(inputs.wsi_tile_features, dtype=torch.float32, device=device)
            tile_coords = torch.as_tensor(inputs.wsi_tile_regional_coords, dtype=torch.float32, device=device)
            return self.slide_encoder(tile_features, tile_coords)
        raise RuntimeError(f"_global_slide_vector called with global_context_source={self.global_context_source!r}")
