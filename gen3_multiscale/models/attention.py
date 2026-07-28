"""Relative-geometry attention bias, chunked boundary cross-attention, and
query-query dense/sparse self-attention -- Phase 5 items 3-5 of the
multiscale spatial-field handoff.

"Relative displacement and distance biases must be available in
query-query and query-context attention" (§5) -- RelativeGeometryBias is
the one shared mechanism both attention modules below use for this,
mirroring the same MLP-on-(dx,dy,distance) pattern
transport_head.py::GeneValueTransportHead already uses for its own
scorer (Phase 4), so relative geometry is treated identically everywhere
it appears in this backbone.

"Implement chunked boundary cross-attention with no silent truncation"
(Phase 5 item 4) -- ChunkedCrossAttention processes the full boundary in
fixed-size chunks using an online (running) softmax, exactly like
FlashAttention's numerical recurrence, so its output is mathematically
identical to a single dense softmax over the WHOLE boundary regardless of
chunk_size (verified by test against an unchunked reference
implementation). max_context_size raises instead of truncating, matching
boundary_graph.py's identical policy from Phase 2.

"Implement query-query dense/sparse switching at a fixed threshold"
(Phase 5 item 5) -- QueryQuerySelfAttention runs full O(n^2) self-attention
when the query count is at or below dense_threshold (default 256, matching
the handoff's own number), and falls back to a geometry-only k-nearest-
query sparse graph (reusing boundary_graph.py's build_knn_adjacency) above
that threshold.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import build_knn_adjacency


class RelativeGeometryBias(nn.Module):
    """(dx, dy, distance) -> one additive attention-logit bias per head.
    The exact same MLP-on-relative-geometry pattern
    transport_head.py::GeneValueTransportHead's scorer already uses
    (Phase 4) -- kept as a small standalone module here so both this
    backbone's attention AND the transport head treat relative geometry
    identically, rather than each attention site inventing its own
    encoding."""

    def __init__(self, n_heads: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_heads),
        )

    def forward(self, relative_geometry: torch.Tensor) -> torch.Tensor:
        if relative_geometry.shape[-1] != 3:
            raise ValueError(f"relative_geometry must have last dimension 3, got {relative_geometry.shape}")
        return self.mlp(relative_geometry)


class ChunkedCrossAttention(nn.Module):
    """Multi-head cross-attention from a (small) query set to a (possibly
    large) context set, processed in fixed-size chunks via an online
    softmax -- mathematically identical to dense full-softmax attention
    over the whole context, never a truncation.

    max_context_size (default None): if set, forward() raises
    ValueError when the context is larger than this, rather than
    silently processing only part of it -- "Fail closed if a configured
    safety maximum is exceeded; never silently truncate it" (handoff §3).
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 8, chunk_size: int = 1024, max_context_size: int | None = None):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.chunk_size = int(chunk_size)
        self.max_context_size = max_context_size

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_bias = RelativeGeometryBias(n_heads)

    def forward(
        self, query_hidden: torch.Tensor, context_hidden: torch.Tensor, relative_geometry: torch.Tensor,
    ) -> torch.Tensor:
        n_query = query_hidden.shape[0]
        n_context = context_hidden.shape[0]
        if relative_geometry.shape != (n_query, n_context, 3):
            raise ValueError(
                f"relative_geometry must be [{n_query}, {n_context}, 3], got {tuple(relative_geometry.shape)}"
            )
        if self.max_context_size is not None and n_context > self.max_context_size:
            raise ValueError(
                f"context size {n_context} exceeds max_context_size={self.max_context_size} -- "
                "increase the limit or process in more chunks; never silently truncate the boundary"
            )
        if n_context == 0:
            raise ValueError("context_hidden is empty -- nothing to attend to")

        q = self.query_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        k_full = self.key_proj(context_hidden)
        v_full = self.value_proj(context_hidden)
        scale = 1.0 / math.sqrt(self.head_dim)

        running_max = torch.full((n_query, self.n_heads), float("-inf"), device=q.device, dtype=q.dtype)
        running_sum = torch.zeros(n_query, self.n_heads, device=q.device, dtype=q.dtype)
        running_output = torch.zeros(n_query, self.n_heads, self.head_dim, device=q.device, dtype=q.dtype)

        for start in range(0, n_context, self.chunk_size):
            end = min(start + self.chunk_size, n_context)
            k_chunk = k_full[start:end].view(-1, self.n_heads, self.head_dim)
            v_chunk = v_full[start:end].view(-1, self.n_heads, self.head_dim)
            geom_chunk = relative_geometry[:, start:end]  # [Nq, chunk, 3]
            bias_chunk = self.geometry_bias(geom_chunk).permute(0, 2, 1)  # [Nq, heads, chunk]

            logits = torch.einsum("qhd,chd->qhc", q, k_chunk) * scale + bias_chunk  # [Nq, heads, chunk]
            chunk_max = logits.max(dim=-1).values  # [Nq, heads]
            new_max = torch.maximum(running_max, chunk_max)
            correction = torch.exp(running_max - new_max)
            correction = torch.where(torch.isfinite(correction), correction, torch.zeros_like(correction))

            exp_logits = torch.exp(logits - new_max[..., None])  # [Nq, heads, chunk]
            running_sum = running_sum * correction + exp_logits.sum(dim=-1)
            chunk_output = torch.einsum("qhc,chd->qhd", exp_logits, v_chunk)
            running_output = running_output * correction[..., None] + chunk_output
            running_max = new_max

        attn_output = running_output / running_sum.clamp_min(1e-12)[..., None]
        return self.out_proj(attn_output.reshape(n_query, self.hidden_dim))


class QueryQuerySelfAttention(nn.Module):
    """Full O(n^2) self-attention among query tokens when the hole has at
    most dense_threshold spots (default 256, matching the handoff's own
    number: "Full query-query self-attention when the hole has at most
    256 query spots"); above that, sparse attention over each query's
    sparse_k (default 8-12 range, default 10 here) nearest OTHER queries
    by real distance -- "For larger holes, sparse query-query attention
    using a geometry-only 8-12 neighbour graph."""

    def __init__(self, hidden_dim: int = 512, n_heads: int = 8, dense_threshold: int = 256, sparse_k: int = 10):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.dense_threshold = int(dense_threshold)
        self.sparse_k = int(sparse_k)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_bias = RelativeGeometryBias(n_heads)

    def _relative_geometry(self, coords: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        delta = coords[j_idx] - coords[i_idx]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        return torch.cat([delta, distance], dim=-1)

    def forward(self, query_hidden: torch.Tensor, query_coords: torch.Tensor) -> tuple[torch.Tensor, str]:
        """Returns (output, mode) where mode is "dense" or "sparse" --
        exposed so a caller/test can confirm which path actually ran,
        rather than only inferring it from the query count."""
        n_query = query_hidden.shape[0]
        q = self.query_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        k = self.key_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        v = self.value_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)

        if n_query <= self.dense_threshold:
            i_idx = torch.arange(n_query, device=q.device).repeat_interleave(n_query)
            j_idx = torch.arange(n_query, device=q.device).repeat(n_query)
            relative_geometry = self._relative_geometry(query_coords, i_idx, j_idx).view(n_query, n_query, 3)
            bias = self.geometry_bias(relative_geometry).permute(0, 2, 1)  # [Nq, heads, Nq]
            logits = torch.einsum("qhd,khd->qhk", q, k) * scale + bias
            weights = torch.softmax(logits, dim=-1)
            output = torch.einsum("qhk,khd->qhd", weights, v)
            return self.out_proj(output.reshape(n_query, self.hidden_dim)), "dense"

        k_neighbors = min(self.sparse_k, n_query - 1)
        adjacency = build_knn_adjacency(query_coords.detach().cpu().numpy(), k_neighbors=k_neighbors)
        max_neighbors = max(len(a) for a in adjacency)
        neighbor_idx = torch.zeros(n_query, max_neighbors, dtype=torch.long, device=q.device)
        neighbor_mask = torch.zeros(n_query, max_neighbors, dtype=torch.bool, device=q.device)
        for i, neighbors in enumerate(adjacency):
            n = len(neighbors)
            if n:
                neighbor_idx[i, :n] = torch.as_tensor(neighbors, device=q.device)
                neighbor_mask[i, :n] = True

        i_idx = torch.arange(n_query, device=q.device)[:, None].expand(-1, max_neighbors).reshape(-1)
        relative_geometry = self._relative_geometry(query_coords, i_idx, neighbor_idx.reshape(-1))
        relative_geometry = relative_geometry.view(n_query, max_neighbors, 3)
        bias = self.geometry_bias(relative_geometry).permute(0, 2, 1)  # [Nq, heads, max_neighbors]

        k_gathered = k[neighbor_idx]  # [Nq, max_neighbors, heads, head_dim]
        v_gathered = v[neighbor_idx]
        logits = torch.einsum("qhd,qkhd->qhk", q, k_gathered) * scale + bias
        logits = logits.masked_fill(~neighbor_mask[:, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)  # an isolated query with zero neighbors -> zero contribution
        output = torch.einsum("qhk,qkhd->qhd", weights, v_gathered)
        return self.out_proj(output.reshape(n_query, self.hidden_dim)), "sparse"
