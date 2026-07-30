"""Mask-aware, coordinate-aware global context pool for UNI2 arms
(GEN4_CONTRACT.md section 6). Deliberately NOT a second slide-level
transformer (the task's own instruction): a single learned inducing query
cross-attends over the item's visible tile features, the same "one learned
query attends over an observed/visible set" shape
`models/global_context.py::InducedGlobalGEXPool` already uses for GEX
tokens, applied here to image tiles instead.

Output is a plain `[global_slide_dim]` vector -- structurally
interchangeable with `FrozenGigaPathSlideEncoder`'s own output from the
backbone's point of view (`MultiscaleBlock`'s `use_global_slide` branch
only ever requires *some* `[global_slide_dim]` vector, never a specific
producer -- see models/backbone.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from gen3_multiscale.models.tokens import FourierCoordinateEncoding


class MaskAwareCoordinateAttentionPool(nn.Module):
    def __init__(
        self,
        tile_feature_dim: int = 1536,
        output_dim: int = 768,
        coord_dim: int = 64,
        hidden_dim: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.coord_encoding = FourierCoordinateEncoding(output_dim=coord_dim)
        self.tile_norm = nn.LayerNorm(tile_feature_dim)
        self.tile_proj = nn.Linear(tile_feature_dim + coord_dim, hidden_dim)

        self.inducing_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, tile_features: torch.Tensor, tile_coords: torch.Tensor) -> torch.Tensor:
        """tile_features: [N, tile_feature_dim] visible-only (already
        hole-filtered, e.g. slide_context.visible_slide_context's own
        output -- this module has no way to enforce that itself, same
        limit models/slide_encoder.py's global-vector path already has).
        tile_coords: [N, 2], same normalized frame as
        wsi_tile_regional_coords. Returns [output_dim]."""
        if tile_features.ndim != 2 or tile_features.shape[0] == 0:
            raise ValueError(f"tile_features must be [N>0, tile_feature_dim], got shape {tuple(tile_features.shape)}")
        if tile_coords.shape != (tile_features.shape[0], 2):
            raise ValueError(
                f"tile_coords must be [{tile_features.shape[0]}, 2] aligned with tile_features, "
                f"got {tuple(tile_coords.shape)}"
            )
        coord_emb = self.coord_encoding(tile_coords)
        tokens = self.tile_proj(torch.cat([self.tile_norm(tile_features), coord_emb], dim=-1))  # [N, hidden_dim]

        q = self.query_proj(self.inducing_query).view(self.n_heads, self.head_dim)
        k = self.key_proj(tokens).view(-1, self.n_heads, self.head_dim)
        v = self.value_proj(tokens).view(-1, self.n_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        logits = torch.einsum("hd,nhd->hn", q, k) * scale  # [n_heads, N]
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("hn,nhd->hd", weights, v).reshape(self.hidden_dim)  # [hidden_dim]
        return self.out_norm(self.out_proj(pooled))
