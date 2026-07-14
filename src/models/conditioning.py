"""
Conditioning encoder — docs/architecture_plan.md layer 1, internals in
docs/model_schematics.md. Turns context (observed cells: coords +
expression) into a fixed conditioning representation per query location,
consumed identically by every generator family (WAE-GAN's decoder, FM-OT's
velocity net, VQ-VAE+AR's transformer).

Design grounded in peer-reviewed, independently-published work (not any
single unreviewed preprint):
  - Random Fourier coordinate encoding: Rahimi & Recht, "Random Features
    for Large-Scale Kernel Machines" (NeurIPS 2007); Tancik et al., "Fourier
    Features Let Networks Learn High Frequency Functions in Low Dimensional
    Domains" (NeurIPS 2020).
  - k-NN graph + attention for spatial neighborhoods in ST specifically:
    SpaGCN (Nature Methods 2021), GraphST (Nature Communications 2023),
    GAAEST (Communications Biology 2024) — all use graph-based spatial
    neighborhood encoding for ST; k is dataset-dependent in this
    literature, commonly in the 6-20 range, which is why it's a
    constructor argument here, not a fixed constant.

Deliberately ST-only for now (no H&E) — see docs/architecture_plan.md
"Known gaps". Kept modular on purpose: everything downstream only ever
consumes this module's output `c`, never its internals, so an image-encoder
branch can be fused in later (concatenated into node_repr/query_feat below)
without changing any generator model.

No torch_geometric dependency — k-NN + attention is implemented directly in
plain PyTorch (cdist/topk, same pattern as InterpolationBaseline in
registry.py), consistent with keeping the dependency footprint light.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """
    Encodes continuous coordinates into a higher-dimensional feature basis
    (Rahimi & Recht 2007; Tancik et al. 2020 — module docstring above).
    Fixed (non-trainable) random projection, so `sigma` is the one
    hyperparameter that matters — controls the encoding's spatial frequency.
    """

    def __init__(self, in_dim: int, num_features: int = 64, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, num_features) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _knn_indices(queries: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """[N_queries, k] indices into `keys` of each query's k nearest neighbours."""
    dists = torch.cdist(queries, keys)
    k = min(k, keys.shape[0])
    _, idx = torch.topk(dists, k=k, largest=False, dim=-1)
    return idx


class _KNNMessageLayer(nn.Module):
    """One round of message passing: each context node attends to its own
    k nearest neighbours. Stacking a few of these lets information from a
    node's local neighbourhood propagate a couple of hops further."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_repr: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        neighbor_repr = node_repr[knn_idx]           # [N, k, hidden_dim]
        query = node_repr.unsqueeze(1)                 # [N, 1, hidden_dim]
        attn_out, _ = self.attn(query, neighbor_repr, neighbor_repr)
        node_repr = self.norm1(node_repr + attn_out.squeeze(1))
        node_repr = self.norm2(node_repr + self.ff(node_repr))
        return node_repr


class SpatialContextEncoder(nn.Module):
    """
    context (coords [N_obs, D], expression [N_obs, G]) + query coords
    [N_query, D]  ->  conditioning vector c [N_query, hidden_dim]

    D is 2 for intra-slice (Track A) or 3 for inter-slice (Track B) — same
    module handles both, since it only ever sees generic coordinates.
    """

    def __init__(self, n_genes: int, coord_dim: int = 3, hidden_dim: int = 256,
                 n_message_layers: int = 2, k_neighbors: int = 10,
                 rff_features: int = 64, rff_sigma: float = 1.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.coord_encoder = RandomFourierFeatures(coord_dim, rff_features, rff_sigma)
        coord_feat_dim = 2 * rff_features  # sin + cos

        self.node_proj = nn.Linear(n_genes + coord_feat_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            _KNNMessageLayer(hidden_dim) for _ in range(n_message_layers)
        )
        self.query_proj = nn.Linear(coord_feat_dim, hidden_dim)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor) -> torch.Tensor:
        # 1. embed each context node from its own expression + coordinate
        context_coord_feat = self.coord_encoder(context_coords)
        node_repr = self.node_proj(torch.cat([context_expression, context_coord_feat], dim=-1))

        # 2. message-pass over the context's own k-NN graph
        context_knn = _knn_indices(context_coords, context_coords, self.k_neighbors)
        for layer in self.message_layers:
            node_repr = layer(node_repr, context_knn)

        # 3. for each query location, attend over its k nearest context nodes
        query_knn = _knn_indices(query_coords, context_coords, self.k_neighbors)
        query_feat = self.query_proj(self.coord_encoder(query_coords))
        neighbor_repr = node_repr[query_knn]                     # [N_query, k, hidden_dim]
        c, _ = self.query_attn(query_feat.unsqueeze(1), neighbor_repr, neighbor_repr)
        return c.squeeze(1)                                        # [N_query, hidden_dim]
