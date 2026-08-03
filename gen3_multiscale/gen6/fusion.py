"""Small, scalable Gen6 modality-fusion components.

Fusion is performed independently per observed spot (a two-token set: image
and GEX).  Spatial communication remains the responsibility of the existing
audited spatial-field backbone.  This avoids quadratic attention over every
observed spot on large HEST slides while still testing the fusion mechanism.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    def __init__(self, image_dim: int, gene_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(image_dim + gene_dim),
            nn.Linear(image_dim + gene_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image, gene], dim=-1))


class MoMEFusion(nn.Module):
    """Per-spot two-token mixture-of-modality-experts block.

    Image and gene tokens have separate experts; a learned gate mixes their
    contributions after shared two-token self-attention.  Missing image rows
    are replaced before attention and the gate is forced to the gene expert.
    """
    def __init__(self, image_dim: int, gene_dim: int, output_dim: int, n_heads: int = 4):
        super().__init__()
        if output_dim % n_heads:
            raise ValueError("MoME output_dim must be divisible by n_heads")
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.gene_proj = nn.Linear(gene_dim, output_dim)
        self.missing_image = nn.Parameter(torch.zeros(output_dim))
        self.attn = nn.MultiheadAttention(output_dim, n_heads, batch_first=True)
        self.image_expert = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, 2 * output_dim),
            nn.GELU(), nn.Linear(2 * output_dim, output_dim),
        )
        self.gene_expert = nn.Sequential(
            nn.LayerNorm(output_dim), nn.Linear(output_dim, 2 * output_dim),
            nn.GELU(), nn.Linear(2 * output_dim, output_dim),
        )
        self.gate = nn.Sequential(nn.LayerNorm(2 * output_dim), nn.Linear(2 * output_dim, 2))
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, image: torch.Tensor, gene: torch.Tensor,
                image_available: torch.Tensor | None = None) -> torch.Tensor:
        image_token = self.image_proj(image)
        if image_available is not None:
            available = image_available.to(dtype=torch.bool).reshape(-1, 1)
            image_token = torch.where(available, image_token, self.missing_image.expand_as(image_token))
        gene_token = self.gene_proj(gene)
        tokens = torch.stack([image_token, gene_token], dim=1)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        image_value = self.image_expert(tokens[:, 0])
        gene_value = self.gene_expert(tokens[:, 1])
        logits = self.gate(torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1))
        if image_available is not None:
            unavailable = ~image_available.to(dtype=torch.bool).reshape(-1)
            logits = logits.clone()
            logits[unavailable, 0] = torch.finfo(logits.dtype).min
        weights = logits.softmax(dim=-1)
        fused = weights[:, :1] * image_value + weights[:, 1:] * gene_value
        return self.out_norm(fused)


class BidirectionalCrossAttentionFusion(nn.Module):
    """Two directed modality interactions, followed by symmetric fusion."""
    def __init__(self, image_dim: int, gene_dim: int, output_dim: int, n_heads: int = 4):
        super().__init__()
        if output_dim % n_heads:
            raise ValueError("cross-attention output_dim must be divisible by n_heads")
        self.image_proj = nn.Linear(image_dim, output_dim)
        self.gene_proj = nn.Linear(gene_dim, output_dim)
        self.missing_image = nn.Parameter(torch.zeros(output_dim))
        self.image_from_gene = nn.MultiheadAttention(output_dim, n_heads, batch_first=True)
        self.gene_from_image = nn.MultiheadAttention(output_dim, n_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(2 * output_dim), nn.Linear(2 * output_dim, output_dim), nn.GELU(),
        )

    def forward(self, image: torch.Tensor, gene: torch.Tensor,
                image_available: torch.Tensor | None = None) -> torch.Tensor:
        image_token = self.image_proj(image)
        if image_available is not None:
            available = image_available.to(dtype=torch.bool).reshape(-1, 1)
            image_token = torch.where(available, image_token, self.missing_image.expand_as(image_token))
        gene_token = self.gene_proj(gene)
        image_q, gene_q = image_token[:, None], gene_token[:, None]
        image_update, _ = self.image_from_gene(image_q, gene_q, gene_q, need_weights=False)
        gene_update, _ = self.gene_from_image(gene_q, image_q, image_q, need_weights=False)
        return self.out(torch.cat([
            (image_q + image_update).squeeze(1),
            (gene_q + gene_update).squeeze(1),
        ], dim=-1))
