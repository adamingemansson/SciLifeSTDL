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
from gen3_multiscale.models.losses import (
    rmse_pcc_reconstruction_loss,
    spatial_gradient_loss,
)
from gen3_multiscale.models.gene_encoder import WeightedGeneExpressionEncoder
from gen3_multiscale.models.tokens import SpotTokenProjection
from gen3_multiscale.conditional_wae.spatial_refinement import (
    SpatialExpressionRefiner,
    refine_expression,
)
from gen3_multiscale.conditional_wae.structured_field import (
    CenteredGeneStructureArtifact,
    CenteredGeneStructureRefinement,
)
from gen3_multiscale.conditional_wae.distributional import (
    ZeroInflatedGaussianHead,
    zero_inflated_gaussian_moments,
    zero_inflated_gaussian_nll,
)

VALID_LIKELIHOODS = ("gaussian_mse", "zero_inflated_gaussian")
VALID_CONDITIONER_MODES = ("local", "spatial")
VALID_PRIOR_MODES = ("standard", "conditional")
VALID_STRUCTURED_COMPOSITIONS = (
    "within_then_between",
    "between_then_within",
    "within_between_within",
    "parallel_gated",
)


class LocalImageConditioner(nn.Module):
    """Strict per-spot UNI2 conditioner with no spatial information.

    This control deliberately has no coordinate input, neighbour graph,
    attention block, observed-GEX branch, or spatial refiner.  Each query row
    is therefore a function only of its own frozen image embedding and the
    image-availability flag.  Keeping this as a separate module (instead of
    zeroing coordinates inside :class:`Architecture1ImageConditioner`) makes
    the no-spatial claim structural and auditable.
    """

    def __init__(self, n_genes: int, image_feature_dim: int = 1536,
                 hidden_dim: int = 512, image_proj_dim: int = 256,
                 modality_flag_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_genes = int(n_genes)
        self.image_feature_dim = int(image_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.image_branch = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, image_proj_dim),
            nn.GELU(),
        )
        self.availability_branch = nn.Sequential(
            nn.Linear(1, modality_flag_dim), nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(image_proj_dim + modality_flag_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim),
        )

    def forward(self, inputs: FullImageExpressionInputs) -> torch.Tensor:
        validate_full_image_expression_inputs(inputs)
        device = next(self.parameters()).device
        image = torch.as_tensor(inputs.image_features, dtype=torch.float32, device=device)
        available = torch.as_tensor(
            inputs.image_available, dtype=torch.float32, device=device,
        ).unsqueeze(-1)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=device)
        if image.ndim != 2 or image.shape[1] != self.image_feature_dim:
            raise ValueError(
                f"image features must be [N, {self.image_feature_dim}], got {tuple(image.shape)}"
            )
        # A missing image is never allowed to masquerade as a real embedding.
        image = image * available
        context = self.output(torch.cat([
            self.image_branch(image), self.availability_branch(available),
        ], dim=-1))
        return context[query_mask]


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


def conditional_imq_mmd(encoded: torch.Tensor, prior: torch.Tensor,
                        context: torch.Tensor, *, context_weight: float = 1.0) -> torch.Tensor:
    """Joint-MMD match of ``(context, posterior-z)`` and ``(context, prior-z)``.

    A global ``MMD(q(z), p(z))`` would permit a learned prior to ignore the
    image condition completely.  Matching the two joint distributions keeps
    the condition in the discrepancy.  Context is detached and normalized
    per feature so the conditioner cannot reduce the penalty by collapsing or
    rescaling its own representation.
    """
    if context_weight <= 0:
        raise ValueError("context_weight must be positive")
    if context.ndim != 2 or context.shape[0] != encoded.shape[0]:
        raise ValueError("context must be [N, context_dim] and align with latent rows")
    normalized = nn.functional.layer_norm(context.detach(), (context.shape[-1],))
    # Equalize the expected context and latent squared-norm contributions.
    scale = float(context_weight) * math.sqrt(encoded.shape[1] / context.shape[1])
    condition = normalized * scale
    return imq_mmd(
        torch.cat([condition, encoded], dim=-1),
        torch.cat([condition, prior], dim=-1),
    )


class ConditionalGaussianPrior(nn.Module):
    """Image-conditioned diagonal Gaussian ``p(z | context)``."""

    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        # Start at the historical N(0,I) prior; conditioning must earn its use.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = self.network(context)
        mean, log_std = parameters.split(self.latent_dim, dim=-1)
        return mean, log_std.clamp(min=-5.0, max=2.0).exp()


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


class DeterministicSpatialPredictor(nn.Module):
    """UNI2 + spatial conditioner + supervised full-GEX prediction head.

    No expression encoder, latent, prior, MMD term, or residual decoder is
    constructed. Its prediction dictionary matches ``ConditionalWAE`` so
    data, checkpointing and metric code stay identical across the ablation.
    """

    def __init__(self, n_genes: int, image_conditioner: nn.Module, *,
                 hidden_dim: int = 1024, pcc_weight: float = 0.1,
                 n_inference_samples: int = 1,
                 gene_structure_artifact: CenteredGeneStructureArtifact | None = None,
                 gene_structure_hidden_dim: int = 64,
                 n_refinement_steps: int = 0,
                 structured_composition: str = "within_then_between",
                 refinement_k_neighbors: int = 6,
                 refinement_hidden_dim: int = 256,
                 refinement_gex_feature_dim: int = 256,
                 per_gene_scale: torch.Tensor | None = None,
                 local_gradient_weight: float = 0.0,
                 wide_gradient_weight: float = 0.0,
                 local_gradient_k: int = 6,
                 wide_gradient_k: int = 18):
        super().__init__()
        if n_genes < 1 or hidden_dim < 1 or n_inference_samples < 1:
            raise ValueError("n_genes, hidden_dim and n_inference_samples must be positive")
        if pcc_weight < 0 or local_gradient_weight < 0 or wide_gradient_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if n_refinement_steps < 0:
            raise ValueError("n_refinement_steps must be non-negative")
        if structured_composition not in VALID_STRUCTURED_COMPOSITIONS:
            raise ValueError(
                "structured_composition must be one of "
                f"{VALID_STRUCTURED_COMPOSITIONS}"
            )
        if local_gradient_k < 1 or wide_gradient_k < 1:
            raise ValueError("gradient neighbourhood sizes must be positive")
        self.n_genes = int(n_genes)
        self.pcc_weight = float(pcc_weight)
        self.n_inference_samples = int(n_inference_samples)
        self.image_conditioner = image_conditioner
        self.conditional_mean_head = nn.Sequential(
            nn.LayerNorm(image_conditioner.hidden_dim),
            nn.Linear(image_conditioner.hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.coexpression_refinement = (
            CenteredGeneStructureRefinement(
                gene_structure_artifact, hidden_dim=int(gene_structure_hidden_dim),
            )
            if gene_structure_artifact is not None else None
        )
        self.discriminator = None
        self.distributional_head = None
        self.n_refinement_steps = int(n_refinement_steps)
        self.structured_composition = str(structured_composition)
        self.spatial_refiner = (
            SpatialExpressionRefiner(
                n_genes, image_conditioner.hidden_dim,
                gex_feature_dim=int(refinement_gex_feature_dim),
                hidden_dim=int(refinement_hidden_dim),
                k_neighbors=int(refinement_k_neighbors),
            )
            if self.n_refinement_steps > 0 else None
        )
        if self.structured_composition != "within_then_between":
            if self.coexpression_refinement is None or self.spatial_refiner is None:
                raise ValueError(
                    f"structured_composition={self.structured_composition!r} requires "
                    "both centered gene-structure and between-spot refinement"
                )
        self.composition_gate_logits = (
            nn.Parameter(torch.full((2,), math.log(0.1 / 0.9), dtype=torch.float32))
            if self.structured_composition == "parallel_gated" else None
        )
        if per_gene_scale is None:
            if local_gradient_weight > 0 or wide_gradient_weight > 0:
                raise ValueError("gradient supervision requires a training-only per_gene_scale")
            scale = torch.ones(n_genes, dtype=torch.float32)
        else:
            scale = torch.as_tensor(per_gene_scale, dtype=torch.float32)
            if scale.shape != (n_genes,) or not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError("per_gene_scale must be finite, positive, and [n_genes]")
        self.register_buffer("per_gene_scale", scale, persistent=True)
        self.local_gradient_weight = float(local_gradient_weight)
        self.wide_gradient_weight = float(wide_gradient_weight)
        self.local_gradient_k = int(local_gradient_k)
        self.wide_gradient_k = int(wide_gradient_k)
        self.encoder_conditioning = "none"
        self.prior_mode = "none"
        self.has_latent_model = False
        self.latent_dim = 0

    def decode_base_from_context(self, context: torch.Tensor) -> torch.Tensor:
        """Decode rows independently, before either structured correction.

        Whole-slide inference chunks this row-independent operation, then
        reassembles the complete slide before calling ``refine_prediction``.
        Keeping every structured operation in that second stage prevents a
        between-spot branch from accidentally seeing only a decoder chunk.
        """
        return self.conditional_mean_head(context)

    def _apply_within(self, expression: torch.Tensor) -> torch.Tensor:
        if self.coexpression_refinement is None:
            return expression
        return self.coexpression_refinement(expression)

    def _apply_between(self, expression: torch.Tensor, context: torch.Tensor,
                       inputs: FullImageExpressionInputs) -> torch.Tensor:
        if self.spatial_refiner is None or self.n_refinement_steps == 0:
            return expression
        coords = torch.as_tensor(inputs.coords, dtype=expression.dtype, device=expression.device)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=expression.device)
        query_coords = coords[query_mask]
        if query_coords.shape[0] != expression.shape[0]:
            raise ValueError("query coordinates do not align with deterministic predictions")
        return refine_expression(
            self.spatial_refiner, expression, context, query_coords,
            n_steps=self.n_refinement_steps,
        )

    def refine_prediction(self, expression: torch.Tensor, context: torch.Tensor,
                          inputs: FullImageExpressionInputs) -> torch.Tensor:
        """Apply the configured within/between composition identically everywhere."""
        if self.structured_composition == "within_then_between":
            return self._apply_between(self._apply_within(expression), context, inputs)
        if self.structured_composition == "between_then_within":
            return self._apply_within(self._apply_between(expression, context, inputs))
        if self.structured_composition == "within_between_within":
            prediction = self._apply_within(expression)
            prediction = self._apply_between(prediction, context, inputs)
            return self._apply_within(prediction)
        if self.structured_composition == "parallel_gated":
            within = self._apply_within(expression)
            between = self._apply_between(expression, context, inputs)
            gates = torch.sigmoid(self.composition_gate_logits)
            return (
                expression
                + gates[0] * (within - expression)
                + gates[1] * (between - expression)
            )
        raise RuntimeError(f"unhandled structured composition {self.structured_composition!r}")

    def composition_gates(self) -> torch.Tensor | None:
        """Return learned within/between branch weights for diagnostics."""
        if self.composition_gate_logits is None:
            return None
        return torch.sigmoid(self.composition_gate_logits)

    def predict_point_from_context(self, context: torch.Tensor,
                                   inputs: FullImageExpressionInputs) -> torch.Tensor:
        return self.refine_prediction(self.decode_base_from_context(context), context, inputs)

    def forward(self, inputs: FullImageExpressionInputs) -> dict:
        context = self.image_conditioner(inputs)
        point = self.predict_point_from_context(context, inputs)
        return {
            "expression": point, "point_prediction": point,
            "conditional_mean_expression": point, "image_context": context,
        }

    def compute_generator_losses(self, inputs: FullImageExpressionInputs,
                                 target_expression, *, generator=None) -> dict:
        del generator
        context = self.image_conditioner(inputs)
        target = torch.as_tensor(target_expression, dtype=context.dtype, device=context.device)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=context.device)
        if target.shape == (len(query_mask), self.n_genes):
            target = target[query_mask]
        if target.shape != (context.shape[0], self.n_genes):
            raise ValueError("target expression does not align with deterministic query rows")
        point = self.predict_point_from_context(context, inputs)
        total, rmse, pcc = rmse_pcc_reconstruction_loss(
            point, target, pcc_weight=self.pcc_weight,
        )
        query_coords = torch.as_tensor(
            inputs.coords, dtype=point.dtype, device=point.device,
        )[query_mask]
        zero = total.new_zeros(())
        local_gradient = (
            spatial_gradient_loss(
                point, target, query_coords, k_neighbors=self.local_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            if self.local_gradient_weight > 0 else zero
        )
        wide_gradient = (
            spatial_gradient_loss(
                point, target, query_coords, k_neighbors=self.wide_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            if self.wide_gradient_weight > 0 else zero
        )
        primary = total
        total = (
            primary
            + self.local_gradient_weight * local_gradient
            + self.wide_gradient_weight * wide_gradient
        )
        result = {
            "total": total, "expression": point,
            "conditional_mean_expression": point,
            "latent": point.new_empty((point.shape[0], 0)),
            "reconstruction_loss": primary, "reconstruction_rmse": rmse,
            "reconstruction_pcc_loss": pcc, "conditional_mean_loss": total,
            "conditional_mean_rmse": rmse, "conditional_mean_pcc_loss": pcc,
            "prior_loss": zero,
            "local_gradient_loss": local_gradient,
            "wide_gradient_loss": wide_gradient,
        }
        gates = self.composition_gates()
        if gates is not None:
            result.update({
                "composition_gate_within": gates[0],
                "composition_gate_between": gates[1],
            })
        return result

    @torch.no_grad()
    def sample_predictive_distribution(self, inputs: FullImageExpressionInputs,
                                       n_samples: int | None = None, **_kwargs) -> dict:
        context = self.image_conditioner(inputs)
        point = self.predict_point_from_context(context, inputs)
        count = int(n_samples or self.n_inference_samples)
        if count < 1:
            raise ValueError("n_samples must be positive")
        samples = point.unsqueeze(0).expand(count, -1, -1)
        return {
            "expression": point, "point_prediction": point,
            "wae_predictive_mean": point, "predictive_mean": point,
            "predictive_std": torch.zeros_like(point),
            "predictive_samples": samples,
            "conditional_mean_expression": point, "image_context": context,
        }


class ConditionalWAE(nn.Module):
    """Conditional full-expression WAE with either MMD or GAN prior matching.

    Both variants share the exact conditioner, encoder, image-mean head and
    conditional residual decoder. Only the aggregate-posterior regularizer
    differs, making the comparison interpretable.
    """

    def __init__(self, n_genes: int, image_conditioner: nn.Module,
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
                 gene_structure_artifact: CenteredGeneStructureArtifact | None = None,
                 gene_structure_hidden_dim: int = 64,
                 gene_encoder_table: torch.Tensor | None = None,
                 z_noise_std: float = 0.0,
                 n_refinement_steps: int = 0,
                 structured_composition: str = "within_then_between",
                 refinement_k_neighbors: int = 6,
                 refinement_hidden_dim: int = 256,
                 refinement_gex_feature_dim: int = 256,
                 likelihood: str = "gaussian_mse",
                 distributional_weight: float = 1.0,
                 distributional_hidden_dim: int = 1024,
                 prior_mode: str = "standard",
                 latent_residual_mode: str = "free",
                 conditional_prior_hidden_dim: int = 256,
                 conditional_prior_context_weight: float = 1.0,
                 conditional_prior_anchor_weight: float = 0.1,
                 specialist_gene_indices: tuple[int, ...] | None = None,
                 specialist_prior_center_weight: float = 0.0,
                 per_gene_scale: torch.Tensor | None = None,
                 local_gradient_weight: float = 0.0,
                 wide_gradient_weight: float = 0.0,
                 local_gradient_k: int = 6,
                 wide_gradient_k: int = 18):
        super().__init__()
        if likelihood not in VALID_LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {VALID_LIKELIHOODS}")
        if distributional_weight < 0:
            raise ValueError("distributional_weight must be non-negative")
        if regularizer not in {"mmd", "gan"}:
            raise ValueError("regularizer must be 'mmd' or 'gan'")
        if n_genes < 1 or latent_dim < 1 or n_inference_samples < 1:
            raise ValueError("n_genes, latent_dim and n_inference_samples must be positive")
        if (regularizer_weight < 0 or conditional_mean_weight < 0 or pcc_weight < 0
                or local_gradient_weight < 0 or wide_gradient_weight < 0):
            raise ValueError("loss weights must be non-negative")
        if local_gradient_k < 1 or wide_gradient_k < 1:
            raise ValueError("gradient neighbourhood sizes must be positive")
        if z_noise_std < 0:
            raise ValueError("z_noise_std must be non-negative")
        if encoder_conditioning not in {"none", "film"}:
            raise ValueError("encoder_conditioning must be 'none' or 'film'")
        if prior_mode not in VALID_PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {VALID_PRIOR_MODES}")
        if structured_composition not in VALID_STRUCTURED_COMPOSITIONS:
            raise ValueError(
                "structured_composition must be one of "
                f"{VALID_STRUCTURED_COMPOSITIONS}"
            )
        if latent_residual_mode not in {"free", "antithetic_zero_mean"}:
            raise ValueError(
                "latent_residual_mode must be 'free' or 'antithetic_zero_mean'"
            )
        if prior_mode == "conditional" and regularizer != "mmd":
            raise ValueError("conditional prior is currently defined only for WAE-MMD")
        if conditional_prior_context_weight <= 0 or conditional_prior_anchor_weight < 0:
            raise ValueError("conditional-prior context weight must be positive and anchor weight non-negative")
        if specialist_prior_center_weight < 0:
            raise ValueError("specialist_prior_center_weight must be non-negative")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.regularizer = regularizer
        self.regularizer_weight = float(regularizer_weight)
        self.conditional_mean_weight = float(conditional_mean_weight)
        self.pcc_weight = float(pcc_weight)
        self.n_inference_samples = int(n_inference_samples)
        self.z_noise_std = float(z_noise_std)
        self.encoder_conditioning = encoder_conditioning
        self.prior_mode = str(prior_mode)
        self.latent_residual_mode = str(latent_residual_mode)
        self.has_latent_model = True
        self.image_conditioner = image_conditioner
        if image_conditioner.n_genes != n_genes:
            raise ValueError("image_conditioner and ConditionalWAE must use the same n_genes")
        context_dim = image_conditioner.hidden_dim
        self.conditional_prior = (
            ConditionalGaussianPrior(
                context_dim, latent_dim, hidden_dim=int(conditional_prior_hidden_dim),
            )
            if self.prior_mode == "conditional" else None
        )
        self.conditional_prior_context_weight = float(conditional_prior_context_weight)
        self.conditional_prior_anchor_weight = float(conditional_prior_anchor_weight)
        specialist_indices = tuple(int(index) for index in (specialist_gene_indices or ()))
        if specialist_indices:
            if len(set(specialist_indices)) != len(specialist_indices):
                raise ValueError("specialist_gene_indices must be unique")
            if min(specialist_indices) < 0 or max(specialist_indices) >= n_genes:
                raise ValueError("specialist_gene_indices are outside the full gene panel")
            if latent_residual_mode != "free":
                raise ValueError("specialist latent heads require latent_residual_mode='free'")
        elif specialist_prior_center_weight != 0:
            raise ValueError(
                "specialist_prior_center_weight requires specialist_gene_indices"
            )
        self.register_buffer(
            "specialist_gene_indices",
            torch.tensor(specialist_indices, dtype=torch.long),
            # Do not add an empty buffer to the expected state of every
            # historical checkpoint.  Specialist checkpoints persist their
            # immutable panel; ordinary WAE checkpoints remain loadable.
            persistent=bool(specialist_indices),
        )
        self.specialist_prior_center_weight = float(specialist_prior_center_weight)
        self.has_specialist_latent_head = bool(specialist_indices)
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
            nn.Linear(
                autoencoder_hidden_dim,
                len(specialist_indices) if specialist_indices else n_genes,
            ),
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
        # New structured-field path. Unlike the legacy reconstruction-only
        # coexpression refiner above, this exact module is applied by
        # ``refine_prediction`` to posterior reconstruction, deterministic
        # point prediction, and every sampled inference path.
        self.centered_gene_structure_refinement = (
            CenteredGeneStructureRefinement(
                gene_structure_artifact, hidden_dim=int(gene_structure_hidden_dim),
            )
            if gene_structure_artifact is not None else None
        )
        # Iterative spatial refinement (STFlow Eq. 7 analogue): neighbouring
        # PREDICTED expression steers attention. Never reads target GEX, so the
        # query-GEX-invisible contract is unchanged. 0 steps is a strict no-op.
        # Predictive-distribution head. When enabled it supplies an ANALYTIC
        # per-spot/per-gene mean and std, so predictive uncertainty no longer
        # depends on Monte-Carlo scatter over a latent that was measured to
        # contribute noise orthogonal to the error (corr 0.002-0.025).
        self.likelihood = likelihood
        self.distributional_weight = float(distributional_weight)
        self.distributional_head = (
            ZeroInflatedGaussianHead(
                n_genes, context_dim, hidden_dim=int(distributional_hidden_dim),
            )
            if likelihood == "zero_inflated_gaussian" else None
        )
        if n_refinement_steps < 0:
            raise ValueError("n_refinement_steps must be non-negative")
        self.n_refinement_steps = int(n_refinement_steps)
        self.structured_composition = str(structured_composition)
        self.spatial_refiner = (
            SpatialExpressionRefiner(
                n_genes, context_dim,
                gex_feature_dim=refinement_gex_feature_dim,
                hidden_dim=refinement_hidden_dim,
                k_neighbors=refinement_k_neighbors,
            )
            if n_refinement_steps > 0 else None
        )
        if self.structured_composition != "within_then_between":
            if self.centered_gene_structure_refinement is None or self.spatial_refiner is None:
                raise ValueError(
                    f"structured_composition={self.structured_composition!r} requires "
                    "both centered gene-structure and between-spot refinement"
                )
        self.composition_gate_logits = (
            nn.Parameter(torch.full((2,), math.log(0.1 / 0.9), dtype=torch.float32))
            if self.structured_composition == "parallel_gated" else None
        )
        if per_gene_scale is None:
            if local_gradient_weight > 0 or wide_gradient_weight > 0:
                raise ValueError("gradient supervision requires a training-only per_gene_scale")
            scale = torch.ones(n_genes, dtype=torch.float32)
        else:
            scale = torch.as_tensor(per_gene_scale, dtype=torch.float32)
            if (scale.shape != (n_genes,) or not torch.isfinite(scale).all()
                    or not torch.all(scale > 0)):
                raise ValueError("per_gene_scale must be finite, positive, and [n_genes]")
        self.register_buffer("per_gene_scale", scale, persistent=True)
        self.local_gradient_weight = float(local_gradient_weight)
        self.wide_gradient_weight = float(wide_gradient_weight)
        self.local_gradient_k = int(local_gradient_k)
        self.wide_gradient_k = int(wide_gradient_k)

        # New residual-WAE suites may warm-start these exact modules from a
        # trained deterministic structured predictor.  The flag is runtime
        # state, not checkpoint state: the pinned source bundle in the config
        # reconstructs the frozen weights on every train/eval process.
        self._deterministic_backbone_frozen = False

    def _deterministic_backbone_modules(self) -> tuple[nn.Module, ...]:
        modules = [self.image_conditioner, self.conditional_mean_head]
        if self.centered_gene_structure_refinement is not None:
            modules.append(self.centered_gene_structure_refinement)
        if self.spatial_refiner is not None:
            modules.append(self.spatial_refiner)
        return tuple(modules)

    def freeze_deterministic_backbone(self) -> None:
        """Freeze and pin the warm-started deterministic point predictor."""
        for module in self._deterministic_backbone_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.composition_gate_logits is not None:
            self.composition_gate_logits.requires_grad_(False)
        self._deterministic_backbone_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._deterministic_backbone_frozen:
            for module in self._deterministic_backbone_modules():
                module.eval()
        return self

    def _apply_within(self, expression: torch.Tensor) -> torch.Tensor:
        if self.centered_gene_structure_refinement is not None:
            return self.centered_gene_structure_refinement(expression)
        return expression

    def _apply_between(self, expression: torch.Tensor, context: torch.Tensor,
                       inputs: FullImageExpressionInputs) -> torch.Tensor:
        if self.spatial_refiner is None or self.n_refinement_steps == 0:
            return expression
        coords = torch.as_tensor(
            inputs.coords, dtype=expression.dtype, device=expression.device,
        )
        query_mask = torch.as_tensor(
            inputs.query_mask, dtype=torch.bool, device=expression.device,
        )
        if coords.shape[0] != expression.shape[0]:
            # The conditioner emits one row per QUERY spot while `coords`
            # spans every spot in the field; select the query rows so the
            # refinement graph is built over exactly the predicted spots.
            coords = coords[query_mask]
        return refine_expression(
            self.spatial_refiner, expression, context, coords,
            n_steps=self.n_refinement_steps,
        )

    def _refine(self, expression: torch.Tensor, context: torch.Tensor,
                inputs: FullImageExpressionInputs) -> torch.Tensor:
        """Apply the configured within/between composition to one field."""
        if self.structured_composition == "within_then_between":
            return self._apply_between(self._apply_within(expression), context, inputs)
        if self.structured_composition == "between_then_within":
            return self._apply_within(self._apply_between(expression, context, inputs))
        if self.structured_composition == "within_between_within":
            expression = self._apply_within(expression)
            expression = self._apply_between(expression, context, inputs)
            return self._apply_within(expression)
        if self.structured_composition == "parallel_gated":
            within = self._apply_within(expression)
            between = self._apply_between(expression, context, inputs)
            gates = torch.sigmoid(self.composition_gate_logits)
            return (
                expression
                + gates[0] * (within - expression)
                + gates[1] * (between - expression)
            )
        raise RuntimeError(f"unhandled structured composition {self.structured_composition!r}")

    def composition_gates(self) -> torch.Tensor | None:
        if self.composition_gate_logits is None:
            return None
        return torch.sigmoid(self.composition_gate_logits)

    def _prior_mean(self, context: torch.Tensor) -> torch.Tensor:
        if self.conditional_prior is None:
            return torch.zeros(
                context.shape[0], self.latent_dim,
                dtype=context.dtype, device=context.device,
            )
        mean, _ = self.conditional_prior(context)
        return mean

    def _latent_residual(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Decode a free or exactly odd residual around the configured prior mean.

        The odd parameterisation has zero expectation under the symmetric
        Gaussian prior.  It prevents latent sampling from moving the already
        strong deterministic point prediction merely because the residual MLP
        learned a non-zero intercept.
        """
        center = self._prior_mean(context)
        delta = z - center
        if self.latent_residual_mode == "free":
            decoded = self.residual_decoder(torch.cat([context, z], dim=-1))
        else:
            positive = self.residual_decoder(torch.cat([context, delta], dim=-1))
            negative = self.residual_decoder(torch.cat([context, -delta], dim=-1))
            decoded = 0.5 * (positive - negative)
        if not self.has_specialist_latent_head:
            return decoded
        residual = decoded.new_zeros(decoded.shape[0], self.n_genes)
        return residual.index_copy(1, self.specialist_gene_indices, decoded)

    def _specialist_loss_view(self, expression: torch.Tensor) -> torch.Tensor:
        """Restrict trainable reconstruction losses to the specialist panel.

        The full-gene target is still encoded by ``expression_encoder``.  Only
        the decoded residual and its supervised loss are panel-restricted, so
        correlations with genes outside the panel remain available to the
        posterior without letting ~17k frozen outputs dilute a 300-gene loss.
        """
        if not self.has_specialist_latent_head:
            return expression
        return expression.index_select(1, self.specialist_gene_indices)

    def restrict_specialist_prediction(
        self, candidate: torch.Tensor, deterministic_base: torch.Tensor,
    ) -> torch.Tensor:
        """Keep every non-specialist gene exactly equal to the frozen base."""
        if not self.has_specialist_latent_head:
            return candidate
        selected = candidate.index_select(1, self.specialist_gene_indices)
        return deterministic_base.index_copy(1, self.specialist_gene_indices, selected)

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
        residual = self._latent_residual(z, context)
        reconstruction = conditional_mean + residual
        # The coexpression refinement (when enabled) only ever nudges the
        # ordinary full-gene prediction -- conditional_mean, the base
        # image-only prediction reported and compared throughout this
        # project, is deliberately never touched by it.
        if self.coexpression_refinement is not None:
            reconstruction = self.coexpression_refinement(reconstruction)
        return reconstruction, conditional_mean

    def _decode_and_refine(self, z: torch.Tensor, context: torch.Tensor,
                           inputs: FullImageExpressionInputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode one latent while preserving the deterministic mean exactly.

        Structured refiners are nonlinear, so an odd residual before them is
        not sufficient.  The symmetric difference below remains odd after
        any shared within/between composition; paired prior draws therefore
        average exactly to the deterministic refined prediction.
        """
        conditional_base = self.conditional_mean_head(context)
        conditional_mean = self._refine(conditional_base, context, inputs)
        if self.latent_residual_mode == "free":
            reconstruction, _ = self.decode(z, context)
            candidate = self._refine(reconstruction, context, inputs)
            return self.restrict_specialist_prediction(
                candidate, conditional_mean,
            ), conditional_mean
        residual = self._latent_residual(z, context)
        positive = self._refine(conditional_base + residual, context, inputs)
        negative = self._refine(conditional_base - residual, context, inputs)
        candidate = conditional_mean + 0.5 * (positive - negative)
        return self.restrict_specialist_prediction(
            candidate, conditional_mean,
        ), conditional_mean

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
        standard_prior = torch.randn(
            encoded.shape, dtype=encoded.dtype, device=encoded.device, generator=generator,
        )
        if self.conditional_prior is None:
            prior = standard_prior
        else:
            prior_mean, prior_std = self.conditional_prior(context)
            prior = prior_mean + prior_std * standard_prior
        # Real fix (user report, Aug 2026): at inference, z is drawn fresh
        # from N(0,I), never from this encoder -- but the decoder here only
        # ever sees the real ENCODED z during training. If the encoder
        # collapses to a narrower-than-prior region (WAE-MMD only
        # regularizes the AGGREGATE posterior, not each sample, so this is
        # not automatically prevented), the decoder is never forced to be
        # sensitive to genuinely prior-scale z variation, producing severely
        # under-dispersed predictive_std at inference (observed z_std of
        # 9-19 against an ideal of 1.0). Perturbing the decoder's z input
        # with independent noise closes this train/inference mismatch
        # directly. The MMD/GAN regularizer below is still computed on the
        # CLEAN `encoded` -- only the decoder's input is perturbed, so the
        # aggregate-posterior-matching objective itself is unchanged.
        decoder_z = encoded
        if self.z_noise_std > 0:
            z_noise = torch.randn(
                encoded.shape, dtype=encoded.dtype, device=encoded.device, generator=generator,
            )
            decoder_z = encoded + z_noise * self.z_noise_std
        reconstruction, conditional_mean = self._decode_and_refine(
            decoder_z, context, inputs,
        )
        reconstruction_view = self._specialist_loss_view(reconstruction)
        conditional_mean_view = self._specialist_loss_view(conditional_mean)
        target_view = self._specialist_loss_view(target)
        reconstruction_loss, reconstruction_rmse, reconstruction_pcc = (
            rmse_pcc_reconstruction_loss(
                reconstruction_view, target_view, pcc_weight=self.pcc_weight,
            )
        )
        conditional_mean_loss, conditional_mean_rmse, conditional_mean_pcc = rmse_pcc_reconstruction_loss(
            conditional_mean_view, target_view, pcc_weight=self.pcc_weight,
        )
        prior_center_prediction = None
        prior_center_loss = prior_center_rmse = prior_center_pcc = zero = (
            reconstruction_loss.new_zeros(())
        )
        if self.has_specialist_latent_head:
            prior_center_prediction, _ = self._decode_and_refine(
                self._prior_mean(context), context, inputs,
            )
            prior_center_loss, prior_center_rmse, prior_center_pcc = (
                rmse_pcc_reconstruction_loss(
                    self._specialist_loss_view(prior_center_prediction),
                    target_view,
                    pcc_weight=self.pcc_weight,
                )
            )
        query_mask = torch.as_tensor(
            inputs.query_mask, dtype=torch.bool, device=context.device,
        )
        query_coords = torch.as_tensor(
            inputs.coords, dtype=context.dtype, device=context.device,
        )[query_mask]
        if self.local_gradient_weight > 0:
            local_gradient_reconstruction = spatial_gradient_loss(
                reconstruction, target, query_coords,
                k_neighbors=self.local_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            local_gradient_conditional_mean = spatial_gradient_loss(
                conditional_mean, target, query_coords,
                k_neighbors=self.local_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            local_gradient = 0.5 * (
                local_gradient_reconstruction + local_gradient_conditional_mean
            )
        else:
            local_gradient_reconstruction = local_gradient_conditional_mean = local_gradient = zero
        if self.wide_gradient_weight > 0:
            wide_gradient_reconstruction = spatial_gradient_loss(
                reconstruction, target, query_coords,
                k_neighbors=self.wide_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            wide_gradient_conditional_mean = spatial_gradient_loss(
                conditional_mean, target, query_coords,
                k_neighbors=self.wide_gradient_k,
                per_gene_scale=self.per_gene_scale,
            )
            wide_gradient = 0.5 * (
                wide_gradient_reconstruction + wide_gradient_conditional_mean
            )
        else:
            wide_gradient_reconstruction = wide_gradient_conditional_mean = wide_gradient = zero
        conditional_alignment_loss = prior_anchor_loss = None
        if self.regularizer == "mmd":
            if self.conditional_prior is None:
                prior_loss = imq_mmd(encoded, prior)
            else:
                conditional_alignment_loss = conditional_imq_mmd(
                    encoded, prior, context,
                    context_weight=self.conditional_prior_context_weight,
                )
                prior_anchor_loss = imq_mmd(prior, standard_prior)
                prior_loss = (
                    conditional_alignment_loss
                    + self.conditional_prior_anchor_weight * prior_anchor_loss
                )
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
            + self.specialist_prior_center_weight * prior_center_loss
            + self.local_gradient_weight * local_gradient
            + self.wide_gradient_weight * wide_gradient
        )
        distributional_loss = None
        if self.distributional_head is not None:
            zero_logit, head_mean, log_sigma = self.distributional_head(context)
            distributional_loss = zero_inflated_gaussian_nll(
                zero_logit, head_mean, log_sigma, target,
            )
            total = total + self.distributional_weight * distributional_loss
        losses = {
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
            "specialist_prior_center_loss": prior_center_loss,
            "specialist_prior_center_rmse": prior_center_rmse,
            "specialist_prior_center_pcc_loss": prior_center_pcc,
            "local_gradient_loss": local_gradient,
            "wide_gradient_loss": wide_gradient,
            "local_gradient_reconstruction_loss": local_gradient_reconstruction,
            "local_gradient_conditional_mean_loss": local_gradient_conditional_mean,
            "wide_gradient_reconstruction_loss": wide_gradient_reconstruction,
            "wide_gradient_conditional_mean_loss": wide_gradient_conditional_mean,
        }
        # Only present when the likelihood head is enabled. The trainer's
        # accumulator calls .detach() on every scalar entry it finds, so a
        # None placeholder here would break every gaussian_mse run.
        if distributional_loss is not None:
            losses["distributional_loss"] = distributional_loss
        if conditional_alignment_loss is not None:
            losses["conditional_alignment_loss"] = conditional_alignment_loss
            losses["prior_anchor_loss"] = prior_anchor_loss
        if prior_center_prediction is not None:
            losses["specialist_prior_center_expression"] = prior_center_prediction
        return losses

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
        conditional_mean = self.predict_point_from_context(context, inputs)
        point_prediction = conditional_mean
        if self.has_specialist_latent_head:
            point_prediction = self.decode_latent_from_context(
                self._prior_mean(context), context, inputs,
            )
        return {
            "expression": point_prediction,
            "point_prediction": point_prediction,
            "conditional_mean_expression": conditional_mean,
            "image_context": context,
        }


    def predict_point_from_context(
        self, context: torch.Tensor, inputs: FullImageExpressionInputs,
    ) -> torch.Tensor:
        """Deterministic H&E-conditioned point prediction.

        ``expression`` has this meaning in :meth:`forward`.  Keeping the
        operation in one public method prevents masked and whole-slide
        inference from silently applying different post-processing.
        """
        return self.refine_prediction(self.conditional_mean_head(context), context, inputs)

    def refine_prediction(
        self, expression: torch.Tensor, context: torch.Tensor,
        inputs: FullImageExpressionInputs,
    ) -> torch.Tensor:
        """Public inference post-processing shared by every entry point."""
        return self._refine(expression, context, inputs)

    def decode_latent_from_context(
        self, z: torch.Tensor, context: torch.Tensor,
        inputs: FullImageExpressionInputs,
    ) -> torch.Tensor:
        """Decode one latent field and apply the configured refinement."""
        prediction, _ = self._decode_and_refine(z, context, inputs)
        return prediction

    def resolve_inference_prior(
        self, context: torch.Tensor, z_mean: torch.Tensor | None = None,
        z_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and place optional ex-post prior parameters."""
        if z_mean is None:
            z_mean = torch.zeros(self.latent_dim, dtype=context.dtype, device=context.device)
        else:
            z_mean = z_mean.to(dtype=context.dtype, device=context.device)
            if z_mean.shape != (self.latent_dim,):
                raise ValueError(f"z_mean must be [{self.latent_dim}], got {tuple(z_mean.shape)}")
        if z_std is None:
            z_std = torch.ones(self.latent_dim, dtype=context.dtype, device=context.device)
        else:
            z_std = z_std.to(dtype=context.dtype, device=context.device)
            if z_std.shape != (self.latent_dim,):
                raise ValueError(f"z_std must be [{self.latent_dim}], got {tuple(z_std.shape)}")
            if bool((z_std < 0).any()):
                raise ValueError("z_std must be non-negative")
        return z_mean, z_std

    def sample_inference_latent(
        self, context: torch.Tensor, *, generator=None,
        z_mean: torch.Tensor | None = None, z_std: torch.Tensor | None = None,
        latent_spatial_correlation: float = 0.0,
    ) -> torch.Tensor:
        """Draw one latent field using the canonical inference sampler."""
        if not 0.0 <= latent_spatial_correlation <= 1.0:
            raise ValueError("latent_spatial_correlation must be in [0, 1]")
        if self.conditional_prior is not None:
            if z_mean is not None or z_std is not None:
                raise ValueError("ex-post z_mean/z_std overrides are incompatible with a conditional prior")
            z_mean, z_std = self.conditional_prior(context)
        else:
            z_mean, z_std = self.resolve_inference_prior(context, z_mean, z_std)
        noise = torch.randn(
            context.shape[0], self.latent_dim,
            dtype=context.dtype, device=context.device, generator=generator,
        )
        if latent_spatial_correlation > 0:
            shared = torch.randn(
                1, self.latent_dim,
                dtype=context.dtype, device=context.device, generator=generator,
            )
            noise = (
                math.sqrt(latent_spatial_correlation) * shared
                + math.sqrt(1.0 - latent_spatial_correlation) * noise
            )
        if z_mean.ndim == 1:
            z_mean = z_mean.unsqueeze(0)
            z_std = z_std.unsqueeze(0)
        return z_mean + noise * z_std

    @torch.no_grad()
    def sample_predictive_distribution(self, inputs: FullImageExpressionInputs,
                                       n_samples: int | None = None,
                                       generator=None,
                                       z_mean: torch.Tensor | None = None,
                                       z_std: torch.Tensor | None = None,
                                       latent_spatial_correlation: float = 0.0) -> dict:
        context = self.image_conditioner(inputs)
        count = int(n_samples or self.n_inference_samples)
        if count < 1:
            raise ValueError("n_samples must be positive")
        if self.distributional_head is not None:
            # The likelihood head already defines the predictive distribution,
            # so the mean and std are exact rather than estimated from a
            # handful of latent draws. `predictive_samples` keeps its
            # [n_samples, N, n_genes] contract by drawing from that fitted
            # distribution, so every downstream consumer is unchanged.
            zero_logit, head_mean, log_sigma = self.distributional_head(context)
            predictive_mean, predictive_std = zero_inflated_gaussian_moments(
                zero_logit, head_mean, log_sigma,
            )
            positive = torch.sigmoid(-zero_logit)
            draws = []
            for _ in range(count):
                keep = (
                    torch.rand(
                        positive.shape, dtype=positive.dtype, device=positive.device,
                        generator=generator,
                    ) < positive
                ).to(positive.dtype)
                noise = torch.randn(
                    head_mean.shape, dtype=head_mean.dtype, device=head_mean.device,
                    generator=generator,
                )
                draws.append(keep * (head_mean + torch.exp(log_sigma) * noise))
            point_prediction = self.predict_point_from_context(context, inputs)
            return {
                "expression": point_prediction,
                "point_prediction": point_prediction,
                "wae_predictive_mean": predictive_mean,
                "predictive_mean": predictive_mean,
                "predictive_std": predictive_std,
                "predictive_samples": torch.stack(draws),
                "conditional_mean_expression": point_prediction,
                "image_context": context,
            }
        # Ex-post density estimation (Ghosh et al., "From Variational to
        # Deterministic Autoencoders", ICLR 2020): the standard, no-retrain
        # fix for a WAE's aggregate-posterior/prior mismatch is to sample z
        # at inference from a density fit to the encoder's REAL training-set
        # output, rather than the raw prior. z_mean/z_std (per-latent-dim,
        # fit offline by scripts/fit_conditional_wae_ex_post_prior.py) let a
        # caller opt into that without touching training. Both None (the
        # default) reproduces the exact pre-existing N(0,I) behavior.
        # Spatially-coherent latent sampling. Measured on four trained arms,
        # the latent's contribution correlates 0.002-0.025 with the error it
        # would need to explain -- the signature of i.i.d.-per-spot noise,
        # which is what `torch.randn(n_spots, latent_dim)` draws. Real
        # biological variation unexplained by H&E is spatially CORRELATED
        # (tissue niches, clonal regions), so a sample that differs
        # independently at every spot cannot express a coherent alternative
        # hypothesis like "this whole region is tumour".
        #
        # Mixing a per-item shared draw with a per-spot draw as
        #     z_i = sqrt(rho) * z_shared + sqrt(1 - rho) * z_i
        # leaves each z_i EXACTLY marginally N(0,I) -- the distribution the
        # MMD/GAN regularizer actually trained the encoder to match -- while
        # imposing correlation rho between spots. It is therefore a pure
        # inference-time change, valid on already-trained checkpoints, and
        # rho=0 (the default) reproduces the historical sampler bit for bit.
        if self.latent_residual_mode == "antithetic_zero_mean" and (
            z_mean is not None or z_std is not None
        ):
            raise ValueError(
                "antithetic_zero_mean inference uses the model's canonical prior; "
                "ex-post z_mean/z_std overrides are unsupported"
            )
        samples = []
        if self.latent_residual_mode == "antithetic_zero_mean":
            # Paired latent draws make the finite Monte-Carlo predictive mean
            # equal the deterministic point prediction, not merely equal in
            # expectation as the number of samples tends to infinity.
            prior_mean = self._prior_mean(context)
            for _ in range(count // 2):
                z = self.sample_inference_latent(
                    context, generator=generator,
                    latent_spatial_correlation=latent_spatial_correlation,
                )
                mirror = 2.0 * prior_mean - z
                samples.append(self.decode_latent_from_context(z, context, inputs))
                samples.append(self.decode_latent_from_context(mirror, context, inputs))
            if count % 2:
                samples.append(
                    self.decode_latent_from_context(prior_mean, context, inputs)
                )
        else:
            for _ in range(count):
                z = self.sample_inference_latent(
                    context, generator=generator, z_mean=z_mean, z_std=z_std,
                    latent_spatial_correlation=latent_spatial_correlation,
                )
                samples.append(self.decode_latent_from_context(z, context, inputs))
        stacked = torch.stack(samples)
        conditional_mean = self.predict_point_from_context(context, inputs)
        predictive_mean = stacked.mean(0)
        point_prediction = conditional_mean
        if self.has_specialist_latent_head:
            point_prediction = self.decode_latent_from_context(
                self._prior_mean(context), context, inputs,
            )
        return {
            # ``expression`` agrees with forward().  It is the deterministic
            # H&E backbone for broad historical WAEs and the decoded prior
            # center (merged into that backbone) for specialist WAEs.
            # ``predictive_mean`` remains the Monte-Carlo WAE diagnostic.
            "expression": point_prediction,
            "point_prediction": point_prediction,
            "wae_predictive_mean": predictive_mean,
            "predictive_mean": predictive_mean,
            "predictive_std": stacked.std(0, unbiased=False),
            "predictive_samples": stacked,
            "conditional_mean_expression": conditional_mean,
            "image_context": context,
        }
