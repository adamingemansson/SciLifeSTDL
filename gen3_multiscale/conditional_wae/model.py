"""Matched conditional WAE-MMD and WAE-GAN models for H&E-to-GEX.

The conditioner reuses Architecture 1's image projection, Fourier coordinates,
relative-geometry attention, and (for Task II only) weighted surrounding-GEX
encoding. It intentionally replaces Architecture 1's gene-transport output
head with the conditional WAE decoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.func import functional_call

from gen3_multiscale.conditional_wae.coexpression import GeneCoexpressionRefinement
from gen3_multiscale.conditional_wae.inputs import (
    FullImageExpressionInputs,
    validate_full_image_expression_inputs,
)
from gen3_multiscale.models.gene_basis import GeneResidualBasis
from gen3_multiscale.gen5.autoencoder import ExpressionEncoder
from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.models.attention import RelativeGeometryBias
from gen3_multiscale.models.losses import rmse_pcc_reconstruction_loss
from gen3_multiscale.models.gene_encoder import WeightedGeneExpressionEncoder
from gen3_multiscale.models.tokens import SpotTokenProjection


class _ImageSpatialBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dense_threshold: int,
                 sparse_k: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.sparse_k = int(sparse_k)
        self.cached_attention = _CachedGeometrySelfAttention(
            hidden_dim, n_heads, dense_threshold=dense_threshold,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, coords: torch.Tensor,
                neighbor_indices: torch.Tensor | None,
                neighbor_mask: torch.Tensor | None) -> torch.Tensor:
        normalized = self.norm_attention(hidden)
        if neighbor_indices is None:
            if len(coords) == 1:
                adjacency = [[0]]
            else:
                adjacency = build_knn_adjacency(
                    coords.detach().cpu().numpy(),
                    k_neighbors=min(self.sparse_k, len(coords) - 1),
                )
            width = max(len(neighbors) for neighbors in adjacency)
            neighbor_indices = torch.zeros(
                len(adjacency), width, dtype=torch.long, device=coords.device,
            )
            neighbor_mask = torch.zeros(
                len(adjacency), width, dtype=torch.bool, device=coords.device,
            )
            for row, neighbors in enumerate(adjacency):
                count = len(neighbors)
                neighbor_indices[row, :count] = torch.as_tensor(neighbors, device=coords.device)
                neighbor_mask[row, :count] = True
        update = self.cached_attention(normalized, coords, neighbor_indices, neighbor_mask)
        hidden = hidden + update
        return hidden + self.ffn(self.norm_ffn(hidden))


class _CachedGeometrySelfAttention(nn.Module):
    """Architecture-1 dense/sparse attention with a cached sparse graph."""

    def __init__(self, hidden_dim: int, n_heads: int, *, dense_threshold: int):
        super().__init__()
        if dense_threshold < 1:
            raise ValueError("dense_threshold must be positive")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dense_threshold = int(dense_threshold)
        self.last_attention_mode: str | None = None
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_bias = RelativeGeometryBias(n_heads)

    def forward(self, hidden: torch.Tensor, coords: torch.Tensor,
                neighbor_indices: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        n, k_neighbors = neighbor_indices.shape
        if hidden.shape[0] != n or coords.shape != (n, 2) or neighbor_mask.shape != (n, k_neighbors):
            raise ValueError("cached adjacency must align with hidden rows and coordinates")
        if n <= self.dense_threshold:
            # Use the same projections/geometry bias as the sparse path,
            # but expose every slide row to every other row. This makes the
            # configured Architecture-1 dense/sparse threshold real while
            # retaining the sample-cached graph for larger slides.
            neighbor_indices = torch.arange(
                n, dtype=torch.long, device=hidden.device,
            )[None, :].expand(n, -1)
            neighbor_mask = torch.ones(
                n, n, dtype=torch.bool, device=hidden.device,
            )
            k_neighbors = n
            self.last_attention_mode = "dense"
        else:
            self.last_attention_mode = "sparse"
        q = self.query_proj(hidden).view(n, self.n_heads, self.head_dim)
        k = self.key_proj(hidden).view(n, self.n_heads, self.head_dim)
        v = self.value_proj(hidden).view(n, self.n_heads, self.head_dim)
        row_indices = torch.arange(n, device=hidden.device)[:, None].expand(-1, k_neighbors)
        delta = coords[neighbor_indices] - coords[row_indices]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        bias = self.geometry_bias(torch.cat([delta, distance], dim=-1)).permute(0, 2, 1)
        gathered_k = k[neighbor_indices]
        gathered_v = v[neighbor_indices]
        logits = torch.einsum("nhd,nkhd->nhk", q, gathered_k) / math.sqrt(self.head_dim)
        logits = (logits + bias).masked_fill(~neighbor_mask[:, None, :], float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        output = torch.einsum("nhk,nkhd->nhd", weights, gathered_v)
        return self.out_proj(output.reshape(n, self.hidden_dim))


class Architecture1ImageConditioner(nn.Module):
    """Architecture-1-derived conditioner with full visible H&E.

    In task I, ``observed_expression`` is absent and every GEX feature is
    zero/unavailable. In task II, only surrounding GEX rows are supplied;
    query GEX is structurally absent. H&E remains visible at query locations
    in both tasks (except genuinely missing source patches).
    """

    def __init__(self, n_genes: int, image_feature_dim: int = 1536,
                 gex_feature_dim: int = 256, hidden_dim: int = 512,
                 n_heads: int = 8, n_blocks: int = 4,
                 dense_threshold: int = 256, sparse_k: int = 10,
                 coord_dim: int = 64, image_proj_dim: int = 256,
                 gex_proj_dim: int = 256, modality_flag_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        self.image_feature_dim = int(image_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_genes = int(n_genes)
        self.gex_feature_dim = int(gex_feature_dim)
        self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, gex_feature_dim)
        self.spot_token = SpotTokenProjection(
            hidden_dim=hidden_dim, image_feature_dim=image_feature_dim,
            image_proj_dim=image_proj_dim, gex_feature_dim=gex_feature_dim,
            gex_proj_dim=gex_proj_dim, coord_dim=coord_dim,
            n_modality_flags=2, modality_flag_dim=modality_flag_dim,
        )
        self.blocks = nn.ModuleList([
            _ImageSpatialBlock(hidden_dim, n_heads, dense_threshold, sparse_k, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, inputs: FullImageExpressionInputs) -> torch.Tensor:
        validate_full_image_expression_inputs(inputs)
        device = next(self.parameters()).device
        image = torch.as_tensor(inputs.image_features, dtype=torch.float32, device=device)
        coords = torch.as_tensor(inputs.coords, dtype=torch.float32, device=device)
        available = torch.as_tensor(
            inputs.image_available, dtype=torch.float32, device=device,
        ).unsqueeze(-1)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=device)
        neighbor_indices = (
            torch.as_tensor(inputs.neighbor_indices, dtype=torch.long, device=device)
            if inputs.neighbor_indices is not None else None
        )
        neighbor_mask = (
            torch.as_tensor(inputs.neighbor_mask, dtype=torch.bool, device=device)
            if inputs.neighbor_mask is not None else None
        )
        if image.shape[1] != self.image_feature_dim:
            raise ValueError(
                f"image feature width {image.shape[1]} != configured {self.image_feature_dim}"
            )
        if inputs.observed_expression is None:
            gex_features = torch.zeros(
                image.shape[0], self.gex_feature_dim, dtype=image.dtype, device=device,
            )
            expression_available = torch.zeros_like(available)
        else:
            expression = torch.as_tensor(
                inputs.observed_expression, dtype=torch.float32, device=device,
            )
            if expression.shape[1] != self.n_genes:
                raise ValueError(
                    f"observed expression width {expression.shape[1]} != configured {self.n_genes}"
                )
            expression_indices = torch.as_tensor(
                inputs.observed_expression_indices, dtype=torch.long, device=device,
            )
            gex_features = torch.zeros(
                image.shape[0], self.gex_feature_dim, dtype=image.dtype, device=device,
            )
            gex_features.index_copy_(0, expression_indices, self.gene_encoder(expression))
            expression_available = torch.as_tensor(
                inputs.expression_available, dtype=torch.float32, device=device,
            ).unsqueeze(-1)
        hidden = self.spot_token(
            image_features=image,
            gex_features=gex_features,
            coords=coords,
            boundary_ring=torch.zeros(image.shape[0], dtype=torch.long, device=device),
            modality_flags=torch.cat([available, expression_available], dim=-1),
        )
        for block in self.blocks:
            hidden = block(hidden, coords, neighbor_indices, neighbor_mask)
        return hidden[query_mask]


def imq_mmd(encoded: torch.Tensor, prior: torch.Tensor,
            scales: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased, non-negative inverse-multiquadratic MMD for WAE training."""
    if encoded.ndim != 2 or encoded.shape != prior.shape:
        raise ValueError("encoded and prior must be matching [N, latent_dim] tensors")
    if encoded.shape[0] < 2:
        raise ValueError("MMD requires at least two rows")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("MMD scales must be positive")

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(left, right).square()
        result = torch.zeros_like(distance)
        for scale in scales:
            constant = 2.0 * encoded.shape[1] * float(scale)
            result = result + constant / (constant + distance)
        return result

    value = kernel(encoded, encoded).mean() + kernel(prior, prior).mean()
    value = value - 2.0 * kernel(encoded, prior).mean()
    return value.clamp_min(0.0)


class _FiLMGenerator(nn.Module):
    """Maps image context to a per-feature (gamma, beta) pair, zero-initialized
    so `gamma=1, beta=0` regardless of context -- FiLM(h) = h at construction,
    identical to no conditioning at all until training moves the weights."""

    def __init__(self, context_dim: int, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.to_gamma_beta = nn.Linear(context_dim, 2 * self.feature_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.to_gamma_beta(context)
        gamma, beta = gamma_beta[..., : self.feature_dim], gamma_beta[..., self.feature_dim :]
        return 1.0 + gamma, beta


class FiLMConditionedExpressionEncoder(nn.Module):
    """Same two-hidden-layer + projection shape as ExpressionEncoder, but the
    requested hidden layer(s) are FiLM-modulated by image context: the
    encoding of target expression becomes a function of both the expression
    AND the image, q(z | expression, context), instead of q(z | expression)
    alone. `film_layers` selects which of {"first", "second"} hidden layers
    receive FiLM; `shared_film_generator` ties the two layers' (gamma, beta)
    generators into one module instead of independent ones."""

    _VALID_LAYERS = frozenset({"first", "second"})

    def __init__(self, n_genes: int, context_dim: int, *, latent_dim: int = 256,
                 hidden_dim: int = 1024, film_layers: tuple[str, ...] = ("first", "second"),
                 shared_film_generator: bool = False):
        super().__init__()
        chosen = frozenset(film_layers)
        if not chosen or not chosen.issubset(self._VALID_LAYERS):
            raise ValueError(f"film_layers must be a non-empty subset of {self._VALID_LAYERS}, got {film_layers}")
        if shared_film_generator and chosen != self._VALID_LAYERS:
            raise ValueError("shared_film_generator requires film_layers to include both layers")
        self.film_layers = chosen
        self.linear1 = nn.Linear(n_genes, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.GELU()

        if shared_film_generator:
            shared = _FiLMGenerator(context_dim, hidden_dim)
            self.film_first, self.film_second = shared, shared
        else:
            self.film_first = _FiLMGenerator(context_dim, hidden_dim) if "first" in chosen else None
            self.film_second = _FiLMGenerator(context_dim, hidden_dim) if "second" in chosen else None

    def forward(self, expression: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(f"expression must be [N, n_genes], got shape {tuple(expression.shape)}")
        if context.shape[0] != expression.shape[0]:
            raise ValueError("context must have one row per expression row")
        hidden = self.norm1(self.linear1(expression))
        if self.film_first is not None:
            gamma, beta = self.film_first(context)
            hidden = gamma * hidden + beta
        hidden = self.activation(hidden)

        hidden = self.norm2(self.linear2(hidden))
        if self.film_second is not None:
            gamma, beta = self.film_second(context)
            hidden = gamma * hidden + beta
        hidden = self.activation(hidden)

        return self.projection(hidden)


class FrozenGeneEmbeddingExpressionEncoder(nn.Module):
    """Same two-hidden-layer + projection shape as ExpressionEncoder/
    FiLMConditionedExpressionEncoder, but the first Linear(n_genes,
    hidden_dim) is replaced by a FROZEN per-gene embedding table
    (real expression @ frozen_table.T -> trainable projection into
    hidden_dim) -- the same "frozen big representation + small trainable
    head" pattern already used by STPathFrozenGeneEncoder
    (src/models/stpath_gene_table.py) and GeneCoexpressionRefinement
    (coexpression.py), applied here to the WAE's OWN posterior encoder
    instead of a decoder-side refinement. `frozen_table` is
    `[embedding_dim, n_genes]` -- the exact orientation
    extract_scfoundation_gene_embedding_table and GeneResidualBasis's own
    `basis` tensor already use, so either can be passed in directly with
    no transpose (in practice this project reuses the SAME saved
    gene-coexpression-basis artifacts -- from-scratch SVD-fit or
    scFoundation-derived -- as the table source here, rather than
    re-deriving a separate table).

    Unlike FiLMConditionedExpressionEncoder (which requires at least one
    FiLM layer), `film_layers` may be EMPTY here: a frozen-table encoder
    with zero FiLM layers is still a real, distinct arm (the encoder
    differs from the from-scratch linear baseline in its gene
    representation; it just isn't ALSO image-conditioned). This lets one
    class serve every {gene-encoder source} x {FiLM on/off} cell of a
    factorial ablation without duplicating the film-application logic."""

    _VALID_LAYERS = frozenset({"first", "second"})

    def __init__(self, frozen_table: torch.Tensor, context_dim: int, *,
                 latent_dim: int = 256, hidden_dim: int = 1024,
                 film_layers: tuple[str, ...] = (), shared_film_generator: bool = False):
        super().__init__()
        table = torch.as_tensor(frozen_table, dtype=torch.float32)
        if table.dim() != 2:
            raise ValueError(f"frozen_table must be [embedding_dim, n_genes], got shape {tuple(table.shape)}")
        chosen = frozenset(film_layers)
        if not chosen.issubset(self._VALID_LAYERS):
            raise ValueError(f"film_layers must be a subset of {self._VALID_LAYERS}, got {film_layers}")
        if shared_film_generator and chosen != self._VALID_LAYERS:
            raise ValueError("shared_film_generator requires film_layers to include both layers")
        self.film_layers = chosen
        self.register_buffer("frozen_table", table)  # [embedding_dim, n_genes], never trained
        self.input_proj = nn.Linear(table.shape[0], hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.GELU()

        if shared_film_generator:
            shared = _FiLMGenerator(context_dim, hidden_dim)
            self.film_first, self.film_second = shared, shared
        else:
            self.film_first = _FiLMGenerator(context_dim, hidden_dim) if "first" in chosen else None
            self.film_second = _FiLMGenerator(context_dim, hidden_dim) if "second" in chosen else None

    def forward(self, expression: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(f"expression must be [N, n_genes], got shape {tuple(expression.shape)}")
        if expression.shape[-1] != self.frozen_table.shape[1]:
            raise ValueError(
                f"expression has {expression.shape[-1]} genes, frozen_table was built for "
                f"{self.frozen_table.shape[1]}"
            )
        needs_context = self.film_first is not None or self.film_second is not None
        if needs_context:
            if context is None:
                raise ValueError("this encoder has FiLM layers configured and requires 'context'")
            if context.shape[0] != expression.shape[0]:
                raise ValueError("context must have one row per expression row")

        pretrained = expression @ self.frozen_table.T  # [B, embedding_dim], frozen
        hidden = self.norm1(self.input_proj(pretrained))
        if self.film_first is not None:
            gamma, beta = self.film_first(context)
            hidden = gamma * hidden + beta
        hidden = self.activation(hidden)

        hidden = self.norm2(self.linear2(hidden))
        if self.film_second is not None:
            gamma, beta = self.film_second(context)
            hidden = gamma * hidden + beta
        hidden = self.activation(hidden)

        return self.projection(hidden)


class ConditionalWAE(nn.Module):
    """Conditional full-expression WAE with either MMD or GAN prior matching.

    Both variants share the exact conditioner, encoder, image-mean head and
    conditional residual decoder. Only the aggregate-posterior regularizer
    differs, making the comparison interpretable.
    """

    def __init__(self, n_genes: int, image_conditioner: Architecture1ImageConditioner,
                 *, regularizer: str, latent_dim: int = 256,
                 autoencoder_hidden_dim: int = 1024,
                 discriminator_hidden_dim: int = 256,
                 regularizer_weight: float = 0.1,
                 conditional_mean_weight: float = 1.0,
                 pcc_weight: float = 0.1,
                 n_inference_samples: int = 8,
                 encoder_conditioning: str = "none",
                 film_layers: tuple[str, ...] = ("first", "second"),
                 film_shared_generator: bool = False,
                 gene_coexpression_basis: GeneResidualBasis | None = None,
                 gene_encoder_table: torch.Tensor | None = None):
        super().__init__()
        if regularizer not in {"mmd", "gan"}:
            raise ValueError("regularizer must be 'mmd' or 'gan'")
        if n_genes < 1 or latent_dim < 1 or n_inference_samples < 1:
            raise ValueError("n_genes, latent_dim and n_inference_samples must be positive")
        if regularizer_weight < 0 or conditional_mean_weight < 0 or pcc_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if encoder_conditioning not in {"none", "film"}:
            raise ValueError("encoder_conditioning must be 'none' or 'film'")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.regularizer = regularizer
        self.regularizer_weight = float(regularizer_weight)
        self.conditional_mean_weight = float(conditional_mean_weight)
        self.pcc_weight = float(pcc_weight)
        self.n_inference_samples = int(n_inference_samples)
        self.encoder_conditioning = encoder_conditioning
        self.image_conditioner = image_conditioner
        if image_conditioner.n_genes != n_genes:
            raise ValueError("image_conditioner and ConditionalWAE must use the same n_genes")
        context_dim = image_conditioner.hidden_dim
        if gene_encoder_table is not None:
            table = torch.as_tensor(gene_encoder_table, dtype=torch.float32)
            if table.shape[1] != n_genes:
                raise ValueError(
                    f"gene_encoder_table must be [embedding_dim, n_genes={n_genes}], "
                    f"got {tuple(table.shape)}"
                )
            self.expression_encoder = FrozenGeneEmbeddingExpressionEncoder(
                table, context_dim, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim,
                film_layers=(film_layers if encoder_conditioning == "film" else ()),
                shared_film_generator=film_shared_generator,
            )
        elif encoder_conditioning == "film":
            self.expression_encoder = FiLMConditionedExpressionEncoder(
                n_genes, context_dim, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim,
                film_layers=film_layers, shared_film_generator=film_shared_generator,
            )
        else:
            self.expression_encoder = ExpressionEncoder(
                n_genes, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim,
            )
        self.conditional_mean_head = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, autoencoder_hidden_dim),
            nn.GELU(), nn.Linear(autoencoder_hidden_dim, n_genes),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, autoencoder_hidden_dim),
            nn.GELU(), nn.LayerNorm(autoencoder_hidden_dim),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim), nn.GELU(),
            nn.Linear(autoencoder_hidden_dim, n_genes),
        )
        self.discriminator = (
            nn.Sequential(
                nn.Linear(latent_dim, discriminator_hidden_dim), nn.GELU(),
                nn.Linear(discriminator_hidden_dim, discriminator_hidden_dim), nn.GELU(),
                nn.Linear(discriminator_hidden_dim, 1),
            )
            if regularizer == "gan" else None
        )
        self.coexpression_refinement = None
        if gene_coexpression_basis is not None:
            if gene_coexpression_basis.n_genes != n_genes:
                raise ValueError("gene_coexpression_basis and ConditionalWAE must use the same n_genes")
            self.coexpression_refinement = GeneCoexpressionRefinement(gene_coexpression_basis)

    def _target(self, inputs: FullImageExpressionInputs, target_expression, *,
                device: torch.device, dtype: torch.dtype, n_rows: int) -> torch.Tensor:
        target = torch.as_tensor(target_expression, dtype=dtype, device=device)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=device)
        if target.shape == (query_mask.shape[0], self.n_genes):
            target = target[query_mask]
        if target.shape != (n_rows, self.n_genes):
            raise ValueError(
                f"target_expression must be [{n_rows}, {self.n_genes}], got {tuple(target.shape)}"
            )
        if not torch.isfinite(target).all():
            raise ValueError("target_expression must be finite")
        return target

    def decode(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.shape != (context.shape[0], self.latent_dim):
            raise ValueError("z must have one configured-width row per image-context row")
        conditional_mean = self.conditional_mean_head(context)
        residual = self.residual_decoder(torch.cat([context, z], dim=-1))
        reconstruction = conditional_mean + residual
        # The coexpression refinement (when enabled) only ever nudges the
        # ordinary full-gene prediction -- conditional_mean, the base
        # image-only prediction reported and compared throughout this
        # project, is deliberately never touched by it.
        if self.coexpression_refinement is not None:
            reconstruction = self.coexpression_refinement(reconstruction)
        return reconstruction, conditional_mean

    def _encode_target(self, target: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if self.encoder_conditioning == "film":
            if context is None:
                raise ValueError("encoder_conditioning='film' requires image context to encode target expression")
            return self.expression_encoder(target, context)
        return self.expression_encoder(target)

    def encode_posterior(self, target_expression, inputs: FullImageExpressionInputs | None = None) -> torch.Tensor:
        """Diagnostic-only q(z | target expression[, image context]). Never
        used for training loss or eval metrics -- exists for TensorBoard
        Projector snapshots, which are allowed to see real target GEX since
        they never feed back into a prediction."""
        device = next(self.expression_encoder.parameters()).device
        target = torch.as_tensor(target_expression, dtype=torch.float32, device=device)
        if self.encoder_conditioning == "film" and inputs is None:
            raise ValueError("encoder_conditioning='film' requires 'inputs' to encode the posterior")
        context = self.image_conditioner(inputs) if self.encoder_conditioning == "film" else None
        return self._encode_target(target, context)

    def compute_generator_losses(self, inputs: FullImageExpressionInputs,
                                 target_expression, *, generator=None) -> dict:
        context = self.image_conditioner(inputs)
        target = self._target(
            inputs, target_expression, device=context.device, dtype=context.dtype,
            n_rows=context.shape[0],
        )
        encoded = self._encode_target(target, context)
        prior = torch.randn(
            encoded.shape, dtype=encoded.dtype, device=encoded.device, generator=generator,
        )
        reconstruction, conditional_mean = self.decode(encoded, context)
        reconstruction_loss, reconstruction_rmse, reconstruction_pcc = (
            rmse_pcc_reconstruction_loss(
                reconstruction, target, pcc_weight=self.pcc_weight,
            )
        )
        conditional_mean_loss, conditional_mean_rmse, conditional_mean_pcc = rmse_pcc_reconstruction_loss(
            conditional_mean, target, pcc_weight=self.pcc_weight,
        )
        if self.regularizer == "mmd":
            prior_loss = imq_mmd(encoded, prior)
        else:
            detached_state = {
                name: value.detach() for name, value in self.discriminator.named_parameters()
            }
            detached_state.update({
                name: value for name, value in self.discriminator.named_buffers()
            })
            logits = functional_call(self.discriminator, detached_state, (encoded,))
            prior_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits),
            )
        total = (
            reconstruction_loss + self.conditional_mean_weight * conditional_mean_loss
            + self.regularizer_weight * prior_loss
        )
        return {
            "total": total,
            "expression": reconstruction,
            "conditional_mean_expression": conditional_mean,
            "latent": encoded,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_rmse": reconstruction_rmse,
            "reconstruction_pcc_loss": reconstruction_pcc,
            "conditional_mean_loss": conditional_mean_loss,
            "conditional_mean_rmse": conditional_mean_rmse,
            "conditional_mean_pcc_loss": conditional_mean_pcc,
            "prior_loss": prior_loss,
        }

    def compute_discriminator_loss(self, target_expression, *, generator=None,
                                   inputs: FullImageExpressionInputs | None = None) -> torch.Tensor:
        if self.discriminator is None:
            raise RuntimeError("discriminator loss is only defined for regularizer='gan'")
        if self.encoder_conditioning == "film" and inputs is None:
            raise ValueError("encoder_conditioning='film' requires 'inputs' to compute the discriminator loss")
        device = next(self.expression_encoder.parameters()).device
        target = torch.as_tensor(target_expression, dtype=torch.float32, device=device)
        if target.ndim != 2 or target.shape[1] != self.n_genes or not torch.isfinite(target).all():
            raise ValueError("target_expression must be finite [N, n_genes]")
        with torch.no_grad():
            context = self.image_conditioner(inputs) if self.encoder_conditioning == "film" else None
            encoded = self._encode_target(target, context)
        prior = torch.randn(
            encoded.shape, dtype=encoded.dtype, device=encoded.device, generator=generator,
        )
        real_logits = self.discriminator(prior)
        encoded_logits = self.discriminator(encoded.detach())
        return (
            nn.functional.binary_cross_entropy_with_logits(
                real_logits, torch.ones_like(real_logits),
            )
            + nn.functional.binary_cross_entropy_with_logits(
                encoded_logits, torch.zeros_like(encoded_logits),
            )
        )

    def forward(self, inputs: FullImageExpressionInputs) -> dict:
        context = self.image_conditioner(inputs)
        conditional_mean = self.conditional_mean_head(context)
        return {"expression": conditional_mean, "image_context": context}

    @torch.no_grad()
    def sample_predictive_distribution(self, inputs: FullImageExpressionInputs,
                                       n_samples: int | None = None,
                                       generator=None) -> dict:
        context = self.image_conditioner(inputs)
        count = int(n_samples or self.n_inference_samples)
        if count < 1:
            raise ValueError("n_samples must be positive")
        samples = []
        for _ in range(count):
            z = torch.randn(
                context.shape[0], self.latent_dim,
                dtype=context.dtype, device=context.device, generator=generator,
            )
            prediction, _ = self.decode(z, context)
            samples.append(prediction)
        stacked = torch.stack(samples)
        conditional_mean = self.conditional_mean_head(context)
        return {
            "expression": stacked.mean(0),
            "predictive_mean": stacked.mean(0),
            "predictive_std": stacked.std(0, unbiased=False),
            "predictive_samples": stacked,
            "conditional_mean_expression": conditional_mean,
            "image_context": context,
        }
