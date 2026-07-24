"""As-simple-as-possible GigaPath + gene-encoder conditioning (2026-07-24).

Motivated directly by reading STPath's own real architecture (verified
against stpath/model/model.py): its entire multimodal fusion is "project
each modality to d_model, sum the projections, feed a small transformer."
Neither existing context encoder in this project sits at that same point
on the complexity scale -- "builtin"/SpatialContextEncoder and
"storm_lite" (fusion_mode="sum") both still run every context+query spot
through a full nn.TransformerEncoder self-attention pass before returning
the query's row. This module provides the two ends of the minimal
contrast the recovery_suite/lung round wants: SimpleFusionContextEncoder
(sum tokens, uniform mean-pool over the k nearest context spots -- zero
attention parameters) and SimpleCrossAttentionContextEncoder (identical
token construction, but ONE learned cross-attention layer instead of a
fixed uniform average). Comparing the two isolates exactly one variable:
does learned attention over neighbors beat unweighted averaging, holding
the GigaPath encoder, gene encoder, and neighbor set fixed.

Both classes reuse the same tested building blocks as every other context
encoder in this project (GigapathPatchEncoder, MLPGeneEncoder, _knn_indices
-- all from conditioning.py) rather than reimplementing image/gene
encoding from scratch. Query gene expression is never available (that is
the value being predicted) -- both classes use a learned mask_token in its
place, the same real fix StormLiteContextEncoder already uses instead of
STPath's own zero-fill (see stpath's real fusion: a zeroed feature still
passes through a biased nn.Linear and injects a real, wrong signal; a
dedicated learned mask_token has no such failure mode)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditioning import GigapathPatchEncoder, MLPGeneEncoder, _knn_indices


class SimpleFusionContextEncoder(nn.Module):
    """GigaPath image embed + gene-MLP embed, summed per spot (STPath's own
    fusion), aggregated over each query's k nearest context spots by a
    plain, unweighted mean -- no attention, no transformer, no extra
    parameters beyond the two encoders themselves."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, knn_k: int = 16,
                 input_already_log1p: bool = True):
        super().__init__()
        if knn_k < 1:
            raise ValueError("knn_k must be positive")
        self.hidden_dim = int(hidden_dim)
        self.knn_k = int(knn_k)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = MLPGeneEncoder(n_genes, feat_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def _context_tokens(self, context_expression: torch.Tensor,
                         context_images: torch.Tensor | None) -> torch.Tensor:
        n_context = context_expression.shape[0]
        device = context_expression.device
        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            img_embed = self.image_encoder(context_images)
        else:
            img_embed = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        return img_embed + gene_embed

    def _query_own_image(self, n_query: int, query_images: torch.Tensor | None,
                          device: torch.device) -> torch.Tensor:
        if query_images is not None:
            return self.image_encoder(query_images)
        return self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]. context_novae_features/organ/tech
        accepted (matching every other context encoder's call signature in
        _encode_context) and unused -- this encoder is deliberately
        GigaPath+gene-only, no niche/organ conditioning."""
        n_query = query_coords.shape[0]
        device = query_coords.device
        context_tokens = self._context_tokens(context_expression, context_images)
        neighbor_idx = _knn_indices(query_coords[:, :2], context_coords[:, :2], self.knn_k)
        neighbor_tokens = context_tokens[neighbor_idx]  # [n_query, k, hidden_dim]
        pooled = neighbor_tokens.mean(dim=1)
        query_img = self._query_own_image(n_query, query_images, device)
        return pooled + query_img


class _CrossAttnBlock(nn.Module):
    """One pre-norm cross-attention transformer block: query attends over
    context key/value tokens, residual, then a 2-layer FFN, residual. Used
    by SimpleCrossAttentionContextEncoder, stacked n_layers times."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor, kv_source: torch.Tensor) -> torch.Tensor:
        """x: [n_query, 1, hidden_dim] running query representation.
        kv_source: [n_query, k, hidden_dim] the (fixed, same every layer)
        neighbor tokens to attend over. Returns the updated [n_query, 1,
        hidden_dim] representation."""
        q = self.norm_q(x)
        kv = self.norm_kv(kv_source)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SimpleCrossAttentionContextEncoder(nn.Module):
    """Identical token construction to SimpleFusionContextEncoder, but the
    query attends over its k nearest context tokens via ONE standard
    pre-norm cross-attention block (LayerNorm -> multi-head cross-attention
    -> residual -> LayerNorm -> 2-layer FFN -> residual) instead of a fixed
    uniform average, STACKED n_layers times (default 2 -- one layer alone
    only lets the query look at neighbors once; a second layer lets it
    refine that using what the first layer already gathered, the same
    reason every real transformer stacks more than one block). Real
    learned attention weights, real residual connections around both
    sublayers in every layer (removing those would make it untrainable
    past one layer, see this project's own notes on why residual
    connections are structural, not optional complexity)."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, knn_k: int = 16,
                 n_layers: int = 2, input_already_log1p: bool = True):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.hidden_dim = int(hidden_dim)
        self.knn_k = int(knn_k)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = MLPGeneEncoder(n_genes, feat_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

        self.layers = nn.ModuleList([
            _CrossAttnBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def _context_tokens(self, context_expression: torch.Tensor,
                         context_images: torch.Tensor | None) -> torch.Tensor:
        n_context = context_expression.shape[0]
        device = context_expression.device
        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            img_embed = self.image_encoder(context_images)
        else:
            img_embed = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        return img_embed + gene_embed

    def _query_own_image(self, n_query: int, query_images: torch.Tensor | None,
                          device: torch.device) -> torch.Tensor:
        if query_images is not None:
            return self.image_encoder(query_images)
        return self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]."""
        n_query = query_coords.shape[0]
        device = query_coords.device
        context_tokens = self._context_tokens(context_expression, context_images)
        neighbor_idx = _knn_indices(query_coords[:, :2], context_coords[:, :2], self.knn_k)
        neighbor_tokens = context_tokens[neighbor_idx]  # [n_query, k, hidden_dim]

        query_token = self._query_own_image(n_query, query_images, device)
        x = query_token.unsqueeze(1)  # [n_query, 1, hidden_dim]
        for layer in self.layers:
            x = layer(x, neighbor_tokens)
        return x.squeeze(1)


class SimpleFusionSpatialTransformerContextEncoder(nn.Module):
    """Our own GigaPath + gene-MLP token construction (identical to
    SimpleFusionContextEncoder/SimpleCrossAttentionContextEncoder), fed
    through STPath's REAL SpatialTransformer backbone (the same class the
    released model uses internally) instead of a hand-rolled aggregator --
    "basically STPath, but without STPath's own organ/tech tokens and
    fixed-vocabulary gene tokenizer" (2026-07-24 request). Isolates: does
    STPath's real transformer backbone help when paired with simpler,
    from-scratch encoders that don't need STPath's ~39k-gene vocabulary
    lookup or organ/tech conditioning at all.

    Construction verified directly against STPath's real source
    (stpath/model/model.py's get_backbone() and
    stpath/model/encoder/spatial_transformer.py's SpatialTransformer/
    ModelConfig, fetched 2026-07-24): SpatialTransformer takes ONE
    ModelConfig object, not individual kwargs; ModelConfig accepts
    arbitrary extra fields via **kwargs (mlp_ratio isn't one of its
    explicitly named fields but IS read by SpatialTransformer, exactly as
    STFM's own get_backbone() passes it). forward(features, coords,
    batch_idx) where coords is [N, 2] and batch_idx is one integer per
    token (all zeros here -- single sample, no cross-sample batching,
    same convention STPathContextEncoder already uses).

    NOT verified by actually running it (the `stpath` package isn't
    installed in every environment this repo runs in, including the one
    this class was written in) -- unlike every other class in this file,
    this one has only been checked for import-time/construction-time
    correctness, not a real forward+backward pass. Smoke-test this
    config for real before trusting its numbers."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, attn_dropout: float = 0.1,
                 mlp_ratio: float = 2.0, input_already_log1p: bool = True):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        try:
            from stpath.model.encoder.spatial_transformer import SpatialTransformer
            from stpath.model.nn_utils.config import ModelConfig
        except ImportError as exc:
            raise ImportError(
                "SimpleFusionSpatialTransformerContextEncoder requires the external "
                "`stpath` package (git clone Graph-and-Geometric-Learning/STPath + "
                "pip install -e .) for its SpatialTransformer backbone -- not needed "
                "for any pretrained weights or gene vocabulary here, just the backbone "
                "class itself."
            ) from exc

        self.hidden_dim = int(hidden_dim)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = MLPGeneEncoder(n_genes, feat_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.backbone = SpatialTransformer(ModelConfig(
            n_genes=n_genes, d_input=hidden_dim, d_model=hidden_dim,
            n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            attn_dropout=attn_dropout, act="gelu", mlp_ratio=mlp_ratio,
        ))

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]. organ/tech accepted (matching
        every other context encoder's call signature) and unused --
        deliberately no organ/tech conditioning, per the 2026-07-24
        request this class implements."""
        n_context = context_expression.shape[0]
        n_query = query_coords.shape[0]
        n_total = n_context + n_query
        device = query_coords.device

        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            context_img = self.image_encoder(context_images)
        else:
            context_img = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        context_tokens = context_img + gene_embed

        if query_images is not None:
            query_img = self.image_encoder(query_images)
        else:
            query_img = self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)
        query_tokens = query_img + self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

        tokens = torch.cat([context_tokens, query_tokens], dim=0)  # [n_total, hidden_dim]
        coords = torch.cat([context_coords[:, :2], query_coords[:, :2]], dim=0)  # [n_total, 2]
        batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)

        fused = self.backbone(tokens, coords, batch_idx)  # [n_total, hidden_dim]
        return fused[n_context:]
