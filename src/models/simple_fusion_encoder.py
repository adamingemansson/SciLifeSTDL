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


class SimpleCrossAttentionContextEncoder(nn.Module):
    """Identical token construction to SimpleFusionContextEncoder, but the
    query attends over its k nearest context tokens via ONE standard
    pre-norm cross-attention block (LayerNorm -> multi-head cross-attention
    -> residual -> LayerNorm -> 2-layer FFN -> residual) instead of a fixed
    uniform average. The minimal architecture that is still recognizably a
    transformer block: real learned attention weights, real residual
    connections around both sublayers (removing those would make it
    untrainable past one layer, see this project's own notes on why
    residual connections are structural, not optional complexity)."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, knn_k: int = 16,
                 input_already_log1p: bool = True):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = int(hidden_dim)
        self.knn_k = int(knn_k)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = MLPGeneEncoder(n_genes, feat_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

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
        q = self.norm_q(query_token).unsqueeze(1)  # [n_query, 1, hidden_dim]
        kv = self.norm_kv(neighbor_tokens)  # [n_query, k, hidden_dim]
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = query_token.unsqueeze(1) + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x.squeeze(1)
