"""The shared spatial-field backbone -- Phase 6's building block, used
identically (with explicit feature flags, never four copy-pasted
implementations) by all four architecture wrappers. Assembles Phase 5's
token/attention modules into the repeated multiscale block Architecture
1's "Model" section specifies:

    1. Query-query self-attention to share information across the hole.
    2. Per-query local cross-attention to the true 32 nearest observed spots.
    3. Boundary cross-attention to every Ring 1-3 spot.
    4. A learned gate combines local and boundary updates.
    5. Feed-forward update with residual connections and normalization.

Architecture 3/4 additionally enable (still the SAME class, via
use_regional_he/use_global_gex/use_global_slide flags, not a subclass or
a separate implementation):

    4. Cross-attention to regional H&E tokens.
    5. Cross-attention to global observed-GEX inducing tokens.
    6. Global LongNet conditioning through FiLM.
    7. A learned modality gate combines ALL updates (local, boundary,
       regional, global-GEX) -- the same gate mechanism, just wider.

"Do not simply add the same global vector to every token before
LayerNorm" (Phase 3's fusion-step warning): global-slide conditioning is
applied via GlobalConditioningFiLM (Phase 5, zero-initialized, an
explicit per-token modulation) at a distinct point in the block, never
summed in with the other branches' plain residual add.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.models.attention import ChunkedCrossAttention, GatheredCrossAttention, QueryQuerySelfAttention
from gen3_multiscale.models.global_context import GlobalConditioningFiLM


class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        max_boundary_size: int | None = None,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_regional_he: bool = False,
        use_global_gex: bool = False,
        use_global_slide: bool = False,
        global_slide_dim: int = 768,
    ):
        super().__init__()
        self.use_regional_he = bool(use_regional_he)
        self.use_global_gex = bool(use_global_gex)
        self.use_global_slide = bool(use_global_slide)

        self.norm_query_self = nn.LayerNorm(hidden_dim)
        self.query_self_attn = QueryQuerySelfAttention(
            hidden_dim=hidden_dim, n_heads=n_heads, dense_threshold=dense_threshold, sparse_k=sparse_k,
        )

        self.norm_cross = nn.LayerNorm(hidden_dim)
        # Local candidates are PER-QUERY (each query has its own 32
        # nearest observed neighbors, boundary_graph.py's
        # query_local_neighbor_idx) -- GatheredCrossAttention, not
        # ChunkedCrossAttention, which assumes one SHARED context set.
        self.local_cross_attn = GatheredCrossAttention(hidden_dim=hidden_dim, n_heads=n_heads)
        # Boundary, regional H&E, and global-GEX inducing tokens are all
        # the SAME shared context every query in this item attends to --
        # ChunkedCrossAttention is the right fit for all three.
        self.boundary_cross_attn = ChunkedCrossAttention(
            hidden_dim=hidden_dim, n_heads=n_heads, chunk_size=chunk_size, max_context_size=max_boundary_size,
        )
        self.regional_cross_attn = (
            ChunkedCrossAttention(hidden_dim=hidden_dim, n_heads=n_heads, chunk_size=chunk_size)
            if use_regional_he else None
        )
        self.global_gex_cross_attn = (
            ChunkedCrossAttention(hidden_dim=hidden_dim, n_heads=n_heads, chunk_size=chunk_size)
            if use_global_gex else None
        )
        n_branches = 2 + int(use_regional_he) + int(use_global_gex)
        self.n_branches = n_branches
        self.branch_gate = nn.Linear(hidden_dim, n_branches)

        self.global_slide_film = (
            GlobalConditioningFiLM(hidden_dim, global_slide_dim) if use_global_slide else None
        )

        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden_dim = ffn_hidden_dim or hidden_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        query_hidden: torch.Tensor,
        query_coords: torch.Tensor,
        local_hidden: torch.Tensor,
        local_geometry: torch.Tensor,
        boundary_hidden: torch.Tensor,
        boundary_geometry: torch.Tensor,
        regional_hidden: torch.Tensor | None = None,
        regional_geometry: torch.Tensor | None = None,
        gex_inducing_hidden: torch.Tensor | None = None,
        gex_inducing_geometry: torch.Tensor | None = None,
        global_slide_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_regional_he and (regional_hidden is None or regional_geometry is None):
            raise ValueError("use_regional_he=True requires regional_hidden and regional_geometry")
        if self.use_global_gex and (gex_inducing_hidden is None or gex_inducing_geometry is None):
            raise ValueError("use_global_gex=True requires gex_inducing_hidden and gex_inducing_geometry")
        if self.use_global_slide and global_slide_vector is None:
            raise ValueError("use_global_slide=True requires global_slide_vector")

        # 1. Query-query self-attention.
        residual = query_hidden
        normed = self.norm_query_self(query_hidden)
        qq_out, _mode = self.query_self_attn(normed, query_coords)
        query_hidden = residual + qq_out

        # 2/3(/4/5). Gated combination of local, boundary(, regional, global-GEX) cross-attention.
        residual = query_hidden
        normed = self.norm_cross(query_hidden)
        branch_outputs = [
            self.local_cross_attn(normed, local_hidden, local_geometry),
            self.boundary_cross_attn(normed, boundary_hidden, boundary_geometry),
        ]
        if self.use_regional_he:
            branch_outputs.append(self.regional_cross_attn(normed, regional_hidden, regional_geometry))
        if self.use_global_gex:
            branch_outputs.append(self.global_gex_cross_attn(normed, gex_inducing_hidden, gex_inducing_geometry))
        stacked = torch.stack(branch_outputs, dim=1)  # [Nq, n_branches, H]
        gate_weights = torch.softmax(self.branch_gate(normed), dim=-1)  # [Nq, n_branches]
        combined = (stacked * gate_weights[..., None]).sum(dim=1)
        query_hidden = residual + combined

        # 6. Global LongNet conditioning (Architecture 3/4 only) -- a
        # distinct FiLM modulation step, never summed into the branch
        # combination above.
        if self.use_global_slide:
            query_hidden = self.global_slide_film(query_hidden, global_slide_vector)

        # 5/7. Feed-forward update with residual connection and normalization.
        residual = query_hidden
        normed = self.norm_ffn(query_hidden)
        query_hidden = residual + self.ffn(normed)

        return query_hidden


class SpatialFieldBackbone(nn.Module):
    """Stacks n_blocks MultiscaleBlocks (default 4-6 per the handoff:
    "Use four to six repeated multiscale blocks, 512 hidden units, and
    eight attention heads"). All blocks share the same feature-flag
    configuration -- this IS the anchor-free conditioner Architectures 1,
    3, and 4 use (Architecture 2 uses the identical backbone, adding the
    harmonic anchor only in the transport head, never in this module)."""

    def __init__(self, n_blocks: int = 4, **block_kwargs):
        super().__init__()
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        self.blocks = nn.ModuleList([MultiscaleBlock(**block_kwargs) for _ in range(n_blocks)])

    def forward(self, query_hidden: torch.Tensor, query_coords: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            query_hidden = block(query_hidden, query_coords, **kwargs)
        return query_hidden
