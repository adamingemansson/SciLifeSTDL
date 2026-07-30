"""Gen4Conditioner -- GEN4_CONTRACT.md sections 2-6.

A thin, structurally-minimal subclass of the existing, unmodified
`gen3_multiscale.models.architectures.Architecture3`. Every module Gen3
already audited (token projections, `SpatialFieldBackbone`,
`GeneValueTransportHead`, `InducedGlobalGEXPool`, `pool_regional_tokens`)
is inherited verbatim; only three forward-time decision points are
overridden, matching GEN4_CONTRACT.md section 4's own description of what
actually differs between arms:

1. `_observed_tokens`'s `image_features` source (a precomputed per-spot
   array already sitting in `inputs.observed_gigapath_features` -- arms
   A/B/C -- or STPath's own context-only encoder, called LIVE inside this
   forward pass -- arm D).
2. `_observed_tokens`'s `gex_features` source (the ordinary trainable
   `WeightedGeneExpressionEncoder` on raw expression, a frozen
   GEX-context embedding projected down to the same width, or a small
   trainable projection of the SAME live STPath embedding used for the
   image slot -- arm D, since STPath's joint representation is meant to
   replace both encoders, not just the image one).
3. `_global_slide_vector`'s source (GigaPath's real frozen LongNet vector,
   unchanged, or `MaskAwareCoordinateAttentionPool`'s trainable pool over
   UNI2 tiles, or no global branch at all for arm D).

Codex audit finding #3 (of the first Gen4 push), confirmed real and fixed
here: the previous design PRECOMPUTED STPath's context-only embedding
outside the model (`gen4/stpath_example.py`, now removed), wrapped in
`torch.no_grad()` and baked into a plain numpy array before the model
ever saw it -- since that happened outside any training step's autograd
graph, STPath's own trainable `proj`/`embedding_norm` layers
(`gen4/stpath_context.py::Gen4STPathContextEncoder`) could never receive
a gradient no matter how the resulting conditioner was trained; they
would have stayed at their random initialization forever. The audit also
correctly flagged that the previous design kept a SEPARATE trainable
`WeightedGeneExpressionEncoder` for arm D's GEX slot, when the design
intent (STPath's representation stands in for BOTH per-spot encoders,
not just the image one) meant no such second encoder should exist. Both
are fixed together below: `image_feature_source='stpath_context'` holds a
real `stpath_encoder` submodule (so its parameters appear in
`self.parameters()`, receive real gradients, and are checkpointed/
reported like any other trainable module) and is called INSIDE
`_observed_tokens`, live, every forward pass -- never precomputed,
never wrapped in `no_grad()` here. `gene_encoder` is never constructed
as a live conditioning path for this arm; the single STPath embedding is
projected into `gex_features` by a small dedicated trainable layer
instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.models.architectures import Architecture3
from gen3_multiscale.models.geometry_utils import scatter_boundary_ring

_GEX_SOURCES = frozenset({"weighted_linear", "frozen_context", "stpath_joint", "hybrid_context"})
_GLOBAL_SOURCES = frozenset({"none", "gigapath", "uni2_pool"})
_IMAGE_SOURCES = frozenset({"precomputed", "stpath_context", "hybrid_context"})

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
        image_feature_source: str = "precomputed",
        stpath_encoder: nn.Module | None = None,
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
        if image_feature_source not in _IMAGE_SOURCES:
            raise ValueError(f"image_feature_source must be one of {sorted(_IMAGE_SOURCES)}, got {image_feature_source!r}")
        if gex_feature_source == "frozen_context" and not gex_context_embedding_dim:
            raise ValueError("gex_feature_source='frozen_context' requires a positive gex_context_embedding_dim")
        if (image_feature_source == "stpath_context") != (gex_feature_source == "stpath_joint"):
            raise ValueError(
                "image_feature_source='stpath_context' and gex_feature_source='stpath_joint' must be used "
                "together -- STPath's single joint representation stands in for both per-spot encoders, "
                "never just one of them"
            )
        if (image_feature_source == "hybrid_context") != (gex_feature_source == "hybrid_context"):
            raise ValueError(
                "image_feature_source='hybrid_context' and gex_feature_source='hybrid_context' must be used "
                "together -- the hybrid arm fuses STPath+UNI2+scFoundation into both per-spot slots at once"
            )
        if image_feature_source == "stpath_context" and stpath_encoder is None:
            raise ValueError("image_feature_source='stpath_context' requires a real stpath_encoder module")
        if image_feature_source == "hybrid_context":
            if stpath_encoder is None:
                raise ValueError("image_feature_source='hybrid_context' requires a real stpath_encoder module")
            if not gex_context_embedding_dim:
                raise ValueError(
                    "image_feature_source='hybrid_context' requires a positive gex_context_embedding_dim "
                    "(the frozen scFoundation cache width) -- the hybrid arm's GEX slot fuses scFoundation "
                    "context embeddings with STPath's joint token"
                )

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
        self.image_feature_source = image_feature_source

        if gex_feature_source in ("frozen_context", "stpath_joint", "hybrid_context"):
            # self.gene_encoder is still constructed by Architecture3's own
            # __init__ (reused unmodified -- see this module's docstring)
            # but is never called in _observed_tokens below for these arms.
            # It is left in place, rather than deleted, so the inherited
            # `_SharedFieldArchitecture.forward`'s own
            # `next(self.gene_encoder.parameters()).device` device-source
            # logic keeps working unmodified; its gradient is disabled so
            # it is genuinely inert (no optimizer update, excluded from
            # "trainable" counts by gen4/param_report.py's requires_grad
            # check) rather than merely unused-but-still-trainable.
            for parameter in self.gene_encoder.parameters():
                parameter.requires_grad_(False)
        if gex_feature_source == "frozen_context":
            self.gex_context_proj = nn.Sequential(
                nn.LayerNorm(gex_context_embedding_dim), nn.Linear(gex_context_embedding_dim, gex_feature_dim),
            )
        else:
            self.gex_context_proj = None

        if image_feature_source == "stpath_context":
            self.stpath_encoder = stpath_encoder  # real trainable submodule -- see forward-time call below
            self.gex_from_stpath_proj = nn.Sequential(
                nn.LayerNorm(image_feature_dim), nn.Linear(image_feature_dim, gex_feature_dim),
            )
        else:
            self.stpath_encoder = None
            self.gex_from_stpath_proj = None

        if image_feature_source == "hybrid_context":
            # Arm 4 (intended-design hybrid): STPath's joint context token,
            # UNI2's own per-spot morphology token (the same precomputed
            # `observed_gigapath_features` field arm C/1 already reuses for
            # real UNI2 features), and scFoundation's observed-GEX context
            # embedding are each projected to a shared width and fused with
            # small trainable Linear layers -- no new transformer, no extra
            # loss, just fusion adapters feeding the SAME `spot_token`/
            # boundary-transformer path every other arm uses.
            self.stpath_encoder = stpath_encoder  # real trainable submodule -- see forward-time call below
            self.hybrid_scf_proj = nn.Sequential(
                nn.LayerNorm(gex_context_embedding_dim), nn.Linear(gex_context_embedding_dim, image_feature_dim),
            )
            self.hybrid_image_fusion = nn.Sequential(
                nn.LayerNorm(2 * image_feature_dim), nn.Linear(2 * image_feature_dim, image_feature_dim),
            )
            self.hybrid_gex_fusion = nn.Sequential(
                nn.LayerNorm(2 * image_feature_dim), nn.Linear(2 * image_feature_dim, gex_feature_dim),
            )
        else:
            self.hybrid_scf_proj = None
            self.hybrid_image_fusion = None
            self.hybrid_gex_fusion = None

    def _observed_tokens(self, inputs, device: torch.device) -> torch.Tensor:
        if self.image_feature_source == "hybrid_context":
            return self._hybrid_observed_tokens(inputs, device)
        if self.image_feature_source == "stpath_context":
            return self._stpath_observed_tokens(inputs, device)
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

    def _stpath_observed_tokens(self, inputs, device: torch.device) -> torch.Tensor:
        """Arm D. Calls `self.stpath_encoder.encode_context_only` LIVE,
        every forward pass, on `inputs.observed_gigapath_features` (the
        RAW per-spot GigaPath tile features STPath's own image tokenizer
        expects -- see GEN4_CONTRACT.md section 8; this arm's examples are
        now built with the exact same `precomputed_spot_features` cache
        arm B uses, never a precomputed STPath output). Deliberately
        NEVER wrapped in `torch.no_grad()` here -- STPath's own frozen
        backbone already scopes its OWN internal `no_grad()` block inside
        `encode_context_only` (`gen4/stpath_context.py`), leaving its
        trainable `proj`/`embedding_norm` layers, and this method's own
        `gex_from_stpath_proj`, on the live autograd graph."""
        if getattr(inputs, "sample_organ", None) is None:
            raise ValueError(
                "image_feature_source='stpath_context' requires Gen4SpatialFieldInputs.sample_organ to be "
                "set -- the real per-sample manifest organ (STPath's organ token was previously fixed at "
                "construction time, silently wrong for every non-default-organ sample)"
            )
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed, torch.as_tensor(inputs.boundary_idx, device=device), torch.as_tensor(inputs.boundary_ring, device=device),
        )
        context_coords = torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device)
        context_expression = torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32, device=device)
        context_image_features = torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32, device=device)
        context_image_available = torch.as_tensor(inputs.observed_image_available, dtype=torch.bool, device=device)

        stpath_embedding = self.stpath_encoder.encode_context_only(
            context_coords, context_expression, context_image_features, context_image_available,
            organ_type=inputs.sample_organ,
        )
        gex_features = self.gex_from_stpath_proj(stpath_embedding)
        # Every context row now has a real, defined STPath representation
        # (its own internal missing_image_token already stands in for a
        # spot with no real H&E patch) -- modality_flags are therefore
        # all-available for this arm, mirroring the previous design's
        # identical reasoning (see git history) for why the base
        # per-spot image-availability signal no longer gates this slot
        # once STPath has already folded it in one layer down.
        modality_flags = torch.ones(n_observed, 1, dtype=torch.float32, device=device)
        return self.spot_token(
            image_features=stpath_embedding, gex_features=gex_features, coords=context_coords,
            boundary_ring=full_ring, modality_flags=modality_flags,
        )

    def _hybrid_observed_tokens(self, inputs, device: torch.device) -> torch.Tensor:
        """Arm 4 (intended-design hybrid): STPath's joint context token
        fused with UNI2's own per-spot morphology token (image slot) and
        with scFoundation's observed-GEX context embedding (GEX slot).
        STPath's `encode_context_only` runs LIVE here, exactly as arm D/3
        does -- never precomputed, never wrapped in `torch.no_grad()`.
        `observed_uni2_features` (a genuinely separate per-spot source from
        whatever feeds STPath's own tokenizer) must be present; this arm
        raises rather than silently falling back to a missing modality."""
        if inputs.observed_uni2_features is None:
            raise ValueError(
                "image_feature_source='hybrid_context' requires Gen4SpatialFieldInputs.observed_uni2_features "
                "to be set -- a real, genuinely UNI2-encoded per-spot array, separate from whatever feeds "
                "STPath's own image tokenizer"
            )
        context_embedding = getattr(inputs, "context_gex_embedding", None)
        if context_embedding is None:
            raise ValueError(
                "image_feature_source='hybrid_context' requires Gen4SpatialFieldInputs.context_gex_embedding "
                "to be set -- a real, frozen scFoundation observed-GEX embedding"
            )
        if getattr(inputs, "sample_organ", None) is None:
            raise ValueError(
                "image_feature_source='hybrid_context' requires Gen4SpatialFieldInputs.sample_organ to be "
                "set -- the real per-sample manifest organ (STPath's organ token was previously fixed at "
                "construction time, silently wrong for every non-default-organ sample)"
            )
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed, torch.as_tensor(inputs.boundary_idx, device=device), torch.as_tensor(inputs.boundary_ring, device=device),
        )
        context_coords = torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device)
        context_expression = torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32, device=device)
        context_image_features = torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32, device=device)
        context_image_available = torch.as_tensor(inputs.observed_image_available, dtype=torch.bool, device=device)

        stpath_embedding = self.stpath_encoder.encode_context_only(
            context_coords, context_expression, context_image_features, context_image_available,
            organ_type=inputs.sample_organ,
        )  # [n_observed, image_feature_dim] -- joint context token
        uni2_embedding = torch.as_tensor(inputs.observed_uni2_features, dtype=torch.float32, device=device)
        scf_embedding = self.hybrid_scf_proj(torch.as_tensor(context_embedding, dtype=torch.float32, device=device))

        image_features = self.hybrid_image_fusion(torch.cat([stpath_embedding, uni2_embedding], dim=-1))
        gex_features = self.hybrid_gex_fusion(torch.cat([stpath_embedding, scf_embedding], dim=-1))
        # STPath's own missing_image_token already stands in for a spot
        # with no real H&E patch (matching arm D/3's identical reasoning);
        # every context row therefore has a real, defined representation.
        modality_flags = torch.ones(n_observed, 1, dtype=torch.float32, device=device)
        return self.spot_token(
            image_features=image_features, gex_features=gex_features, coords=context_coords,
            boundary_ring=full_ring, modality_flags=modality_flags,
        )

    def _global_slide_vector(self, inputs, device: torch.device) -> torch.Tensor:
        if self.global_context_source == "gigapath":
            return super()._global_slide_vector(inputs, device)
        if self.global_context_source == "uni2_pool":
            self._require_wsi_context(inputs, "global_context_source='uni2_pool'")
            # Codex audit finding #4: `wsi_tile_features` is populated by
            # Gen3's existing dense-WSI tile cache, which is GigaPath-
            # encoded -- no real UNI2 dense-WSI cache exists yet. Without
            # this check, arm A/C would silently consume GigaPath-shaped
            # features as if they were UNI2 features. Fails closed until a
            # real, provenance-tagged UNI2 dense-WSI cache builder sets
            # `wsi_tile_feature_provenance == "uni2"` explicitly.
            if getattr(inputs, "wsi_tile_feature_provenance", None) != "uni2":
                raise ValueError(
                    "global_context_source='uni2_pool' requires inputs.wsi_tile_feature_provenance == 'uni2' -- "
                    "Gen3's existing dense-WSI tile cache is GigaPath-encoded, not UNI2-encoded, and no real "
                    "UNI2 dense-WSI cache builder exists yet (Codex audit finding #4). This pathway is disabled "
                    "until a real, provenance-tagged UNI2 dense-WSI cache is built and explicitly marks its "
                    "output with this field."
                )
            tile_features = torch.as_tensor(inputs.wsi_tile_features, dtype=torch.float32, device=device)
            tile_coords = torch.as_tensor(inputs.wsi_tile_regional_coords, dtype=torch.float32, device=device)
            return self.slide_encoder(tile_features, tile_coords)
        raise RuntimeError(f"_global_slide_vector called with global_context_source={self.global_context_source!r}")
