"""Shared token modules -- Phase 5 items 1-3 of the multiscale spatial-
field handoff ("Build shared token modules"): the common spot-token
projection, query coordinate/depth tokens, and the coordinate encoding
both are built from.

Per the handoff's "Common token schema" (§4): a 512-dimensional hidden
representation throughout, built by concatenating separately-processed
modality features and projecting ONCE to the common width -- "Do not add
unrelated modalities together before normalization, because that can
erase or confound their individual contributions." Every module here
concatenates then projects; none of them ever sums raw modality vectors
of different natural scales together.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierCoordinateEncoding(nn.Module):
    """Relative/normalized [*, 2] coordinates -> Fourier features -> a
    small MLP -> output_dim (~64 per the handoff's token schema).
    Standard NeRF-style log-spaced frequency bands; coordinates are
    assumed already relative/normalized by the caller (never raw
    physical micrometers -- see the handoff's "Coordinate memorization"
    risk, and gen3_multiscale/data/example.py's identical convention)."""

    def __init__(self, output_dim: int = 64, num_frequencies: int = 8, mlp_hidden_dim: int | None = None):
        super().__init__()
        if output_dim < 1 or num_frequencies < 1:
            raise ValueError("output_dim and num_frequencies must be positive")
        freqs = (2.0 ** torch.arange(num_frequencies)) * math.pi
        self.register_buffer("freqs", freqs)
        fourier_dim = 2 * num_frequencies * 2  # 2 coord axes * F frequencies * {sin, cos}
        hidden = mlp_hidden_dim or output_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 2:
            raise ValueError(f"coords must have last dimension 2, got {coords.shape}")
        proj = coords[..., None] * self.freqs  # [..., 2, F]
        proj = proj.reshape(*coords.shape[:-1], -1)  # [..., 2*F]
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 4*F]
        return self.mlp(fourier)


class SpotTokenProjection(nn.Module):
    """Builds one context-spot token from local H&E, compact GEX
    conditioning, coordinates, boundary-ring identity, and modality-
    availability flags -- the exact 5-part schema the handoff specifies
    for a "context spot token" (§4).

    boundary_ring uses 4 categories: 0 = observed, not in Rings 1-3
    (interior of the visible tissue, far from any hole), 1/2/3 = the real
    boundary rings from boundary_graph.py. modality_flags is a small
    (default 1-wide: image_available) float vector projected UP into an
    embedding space, not raw flags concatenated in -- keeps the same
    "project every branch, then concatenate" discipline as every other
    input here."""

    def __init__(
        self,
        hidden_dim: int = 512,
        image_feature_dim: int = 1536,
        image_proj_dim: int = 256,
        gex_feature_dim: int = 256,
        gex_proj_dim: int = 256,
        coord_dim: int = 64,
        n_boundary_rings: int = 4,
        ring_embed_dim: int = 16,
        n_modality_flags: int = 1,
        modality_flag_dim: int = 16,
    ):
        super().__init__()
        self.image_norm = nn.LayerNorm(image_feature_dim)
        self.image_proj = nn.Linear(image_feature_dim, image_proj_dim)
        self.gex_norm = nn.LayerNorm(gex_feature_dim)
        self.gex_proj = nn.Linear(gex_feature_dim, gex_proj_dim)
        self.coord_encoding = FourierCoordinateEncoding(output_dim=coord_dim)
        self.ring_embedding = nn.Embedding(n_boundary_rings, ring_embed_dim)
        self.modality_flag_proj = nn.Linear(n_modality_flags, modality_flag_dim)

        concat_dim = image_proj_dim + gex_proj_dim + coord_dim + ring_embed_dim + modality_flag_dim
        self.output_proj = nn.Linear(concat_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.n_boundary_rings = n_boundary_rings

    def forward(
        self,
        image_features: torch.Tensor,
        gex_features: torch.Tensor,
        coords: torch.Tensor,
        boundary_ring: torch.Tensor,
        modality_flags: torch.Tensor,
    ) -> torch.Tensor:
        n = image_features.shape[0]
        if boundary_ring.min() < 0 or boundary_ring.max() >= self.n_boundary_rings:
            raise ValueError(
                f"boundary_ring must be in [0, {self.n_boundary_rings}), got range "
                f"[{int(boundary_ring.min())}, {int(boundary_ring.max())}]"
            )
        image_emb = self.image_proj(self.image_norm(image_features))
        gex_emb = self.gex_proj(self.gex_norm(gex_features))
        coord_emb = self.coord_encoding(coords)
        ring_emb = self.ring_embedding(boundary_ring)
        flag_emb = self.modality_flag_proj(modality_flags)
        concat = torch.cat([image_emb, gex_emb, coord_emb, ring_emb, flag_emb], dim=-1)
        if concat.shape[0] != n:
            raise ValueError("all SpotTokenProjection inputs must share the same leading spot count")
        return self.output_norm(self.output_proj(concat))


class QueryTokenProjection(nn.Module):
    """Builds one query (missing-spot) token from its coordinate, its
    depth-to-boundary, a single learned "this is a query" identity
    (broadcast to every query token -- distinguishes a query token from a
    context token to any attention module that sees both), and optional
    hole-level geometry (area, normalized distance to the hole centroid).
    Contains no target GEX and no target H&E field anywhere in this
    module -- there is nothing in its signature that COULD carry either.

    depth_to_boundary comes from boundary_graph.py as a non-negative
    integer BFS hop count (0 = touches the boundary directly); embedded
    via a bucketed nn.Embedding (values beyond max_depth_buckets are
    clamped to the last bucket -- deep-interior queries beyond the
    embedding's resolution are still distinguishable from shallow ones,
    just not from each other past that point, a deliberate simplification
    the handoff doesn't prescribe an exact encoding for)."""

    def __init__(
        self,
        hidden_dim: int = 512,
        coord_dim: int = 64,
        max_depth_buckets: int = 16,
        depth_embed_dim: int = 32,
        identity_dim: int = 32,
        hole_geometry_dim: int = 2,
        hole_geometry_embed_dim: int = 16,
        use_hole_geometry: bool = True,
    ):
        super().__init__()
        self.coord_encoding = FourierCoordinateEncoding(output_dim=coord_dim)
        self.max_depth_buckets = int(max_depth_buckets)
        self.depth_embedding = nn.Embedding(max_depth_buckets, depth_embed_dim)
        self.query_identity = nn.Parameter(torch.randn(identity_dim) * 0.02)
        self.use_hole_geometry = bool(use_hole_geometry)
        self.hole_geometry_proj = (
            nn.Linear(hole_geometry_dim, hole_geometry_embed_dim) if use_hole_geometry else None
        )

        concat_dim = coord_dim + depth_embed_dim + identity_dim
        if use_hole_geometry:
            concat_dim += hole_geometry_embed_dim
        self.output_proj = nn.Linear(concat_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coords: torch.Tensor,
        depth_to_boundary: torch.Tensor,
        hole_geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_query = coords.shape[0]
        if torch.any(depth_to_boundary < 0):
            raise ValueError("depth_to_boundary must be non-negative")
        coord_emb = self.coord_encoding(coords)
        depth_clamped = depth_to_boundary.clamp(max=self.max_depth_buckets - 1)
        depth_emb = self.depth_embedding(depth_clamped)
        identity_emb = self.query_identity.expand(n_query, -1)
        parts = [coord_emb, depth_emb, identity_emb]
        if self.use_hole_geometry:
            if hole_geometry is None:
                raise ValueError("this module requires hole_geometry (use_hole_geometry=True)")
            hole_geometry_emb = self.hole_geometry_proj(hole_geometry)
            if hole_geometry_emb.shape[0] == 1 and n_query > 1:
                hole_geometry_emb = hole_geometry_emb.expand(n_query, -1)
            parts.append(hole_geometry_emb)
        elif hole_geometry is not None:
            raise ValueError("hole_geometry was provided but use_hole_geometry=False")
        concat = torch.cat(parts, dim=-1)
        return self.output_norm(self.output_proj(concat))
