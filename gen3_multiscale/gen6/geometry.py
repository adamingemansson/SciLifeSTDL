"""Alternative query-query geometry biases for the Gen6 component screen."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.models.attention import QueryQuerySelfAttention
from src.models.conditioning import FrameAveragingBias


class FourierRelativeBias(nn.Module):
    def __init__(self, n_heads: int, n_frequencies: int = 16):
        super().__init__()
        frequencies = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Sequential(
            nn.LayerNorm(3 * 2 * n_frequencies),
            nn.Linear(3 * 2 * n_frequencies, 64), nn.GELU(), nn.Linear(64, n_heads),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        scale = geometry[..., 2:].amax().clamp_min(1e-6)
        phase = geometry[..., None] / scale * self.frequencies
        encoded = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        return self.proj(encoded)


class NormalizedRelativeGeometryBias(nn.Module):
    """Learned relative bias independent of raw pixel/micron units."""
    def __init__(self, n_heads: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, n_heads))

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        scale = geometry[..., 2:].amax().clamp_min(1e-6)
        return self.mlp(geometry / scale)


class Gen6QuerySelfAttention(QueryQuerySelfAttention):
    """Existing dense/sparse attention with an explicitly selected bias.

    The frame mode uses the verified STPath-style frame-averaging module;
    relative mode is the existing Gen3 learned relative bias; Fourier mode
    uses a multiscale sinusoidal encoding of relative geometry.
    """
    def __init__(self, *args, geometry_mode: str, coord_scale: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if geometry_mode not in {"relative_bias", "frame_averaging", "fourier_attention"}:
            raise ValueError(f"unsupported Gen6 geometry_mode {geometry_mode!r}")
        self.geometry_mode = geometry_mode
        self.auto_frame_scale = geometry_mode == "frame_averaging" and coord_scale <= 0
        if geometry_mode == "frame_averaging":
            self.frame_bias = FrameAveragingBias(
                self.n_heads, coord_scale=1.0 if self.auto_frame_scale else coord_scale,
            )
        else:
            self.frame_bias = None
        if geometry_mode == "fourier_attention":
            self.geometry_bias = FourierRelativeBias(self.n_heads)
        elif geometry_mode == "relative_bias":
            self.geometry_bias = NormalizedRelativeGeometryBias(self.n_heads)

    def _full_bias(self, coords: torch.Tensor) -> torch.Tensor:
        n = coords.shape[0]
        if self.frame_bias is not None:
            frame_coords = coords
            if self.auto_frame_scale:
                centered = coords - coords.mean(dim=0, keepdim=True)
                scale = torch.linalg.norm(centered, dim=-1).amax().clamp_min(1e-6)
                frame_coords = centered / scale
            return self.frame_bias(frame_coords).permute(1, 0, 2)
        i_idx = torch.arange(n, device=coords.device).repeat_interleave(n)
        j_idx = torch.arange(n, device=coords.device).repeat(n)
        geometry = self._relative_geometry(coords, i_idx, j_idx).view(n, n, 3)
        return self.geometry_bias(geometry).permute(0, 2, 1)

    def _sparse_bias(
        self, coords: torch.Tensor, neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bias only for selected KNN edges, never an N×N tensor."""
        n, k_neighbors = neighbor_idx.shape
        if self.frame_bias is None:
            i_idx = torch.arange(n, device=coords.device)[:, None].expand(-1, k_neighbors)
            geometry = self._relative_geometry(
                coords, i_idx.reshape(-1), neighbor_idx.reshape(-1),
            ).view(n, k_neighbors, 3)
            return self.geometry_bias(geometry).permute(0, 2, 1)

        frame_coords = coords[:, :2]
        if self.auto_frame_scale:
            centered_coords = frame_coords - frame_coords.mean(dim=0, keepdim=True)
            scale = torch.linalg.norm(centered_coords, dim=-1).amax().clamp_min(1e-6)
            frame_coords = centered_coords / scale
        else:
            frame_coords = frame_coords / float(self.frame_bias.coord_scale)
        radial = frame_coords[:, None] - frame_coords[neighbor_idx]
        radial_norm = radial.norm(dim=-1, keepdim=True)
        centered = radial - radial.mean(dim=1, keepdim=True)
        cov = torch.einsum("nki,nkj->nij", centered, centered) / float(k_neighbors)
        _, eigvecs = torch.linalg.eigh(cov)
        frame_ops = (
            self.frame_bias.ops.view(1, self.frame_bias.n_frames, 1, 2)
            * eigvecs.unsqueeze(1)
        )
        frame_features = torch.einsum("nofd,nkd->nofk", frame_ops, radial)
        frame_features = frame_features.permute(0, 1, 3, 2)
        expanded_norm = radial_norm.unsqueeze(1).expand(
            n, self.frame_bias.n_frames, k_neighbors, 1,
        )
        features = torch.cat([frame_features, expanded_norm], dim=-1)
        return self.frame_bias.edge_bias(features).mean(dim=1).permute(0, 2, 1)

    def forward(self, query_hidden: torch.Tensor, query_coords: torch.Tensor):
        n_query = query_hidden.shape[0]
        q = self.query_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        k = self.key_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        v = self.value_proj(query_hidden).view(n_query, self.n_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        if n_query <= self.dense_threshold:
            full_bias = self._full_bias(query_coords)
            logits = torch.einsum("qhd,khd->qhk", q, k) * scale + full_bias
            weights = logits.softmax(dim=-1)
            output = torch.einsum("qhk,khd->qhd", weights, v)
            return self.out_proj(output.reshape(n_query, self.hidden_dim)), "dense"

        k_neighbors = min(self.sparse_k, n_query - 1)
        adjacency = build_knn_adjacency(query_coords.detach().cpu().numpy(), k_neighbors=k_neighbors)
        max_neighbors = max(len(row) for row in adjacency)
        neighbor_idx = torch.zeros(n_query, max_neighbors, dtype=torch.long, device=q.device)
        neighbor_mask = torch.zeros(n_query, max_neighbors, dtype=torch.bool, device=q.device)
        for index, neighbors in enumerate(adjacency):
            if len(neighbors):
                neighbor_idx[index, :len(neighbors)] = torch.as_tensor(neighbors, device=q.device)
                neighbor_mask[index, :len(neighbors)] = True
        k_gathered, v_gathered = k[neighbor_idx], v[neighbor_idx]
        selected_bias = self._sparse_bias(query_coords, neighbor_idx)
        logits = torch.einsum("qhd,qkhd->qhk", q, k_gathered) * scale + selected_bias
        logits = logits.masked_fill(~neighbor_mask[:, None], float("-inf"))
        weights = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
        output = torch.einsum("qhk,qkhd->qhd", weights, v_gathered)
        return self.out_proj(output.reshape(n_query, self.hidden_dim)), "sparse"


def install_query_geometry(model: nn.Module, geometry_mode: str, *, coord_scale: float = 0.0) -> None:
    """Replace only query self-attention; all other block weights/paths stay matched."""
    if geometry_mode == "fourier_absolute":
        return  # QueryTokenProjection already supplies Fourier absolute coordinates.
    for block in model.backbone.blocks:
        old = block.query_self_attn
        replacement = Gen6QuerySelfAttention(
            hidden_dim=old.hidden_dim, n_heads=old.n_heads,
            dense_threshold=old.dense_threshold, sparse_k=old.sparse_k,
            geometry_mode=geometry_mode, coord_scale=coord_scale,
        )
        block.query_self_attn = replacement
