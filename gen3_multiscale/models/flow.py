"""Velocity network, flow-matching loss, and ODE sampling -- Architecture
4's stochastic apparatus, built on top of Architecture 3's frozen
deterministic conditioner.

Velocity network (handoff): "a spatial transformer that has: query-query
attention over the noisy residual field; cross-attention to the frozen
deterministic conditioning tokens from Architecture 3; relative query
geometry; one shared continuous-time value for the whole hole." Built
from the SAME QueryQuerySelfAttention/ChunkedCrossAttention modules every
other architecture uses (Phase 5), not a separate reimplementation --
"the validated gene-aware/transport formulation already present in the
audited branch, if and only if its exact semantics and tests are
preserved. Do not re-create that component from memory based only on its
name" is satisfied by reusing the actual tested modules, not by
reproducing their behavior from memory.

Flow matching: linear (rectified-flow-style) conditional flow matching --
x_t = (1-t)*x0 + t*x1 for x0 ~ N(0, I) noise and x1 = target residual
coefficients, target velocity = x1 - x0 (constant along the straight-line
path), one shared scalar t per hole (per the handoff's "one shared
continuous-time value for the whole hole" -- every query in the item
uses the SAME t and the SAME x0 draw's noise scale, not independently
sampled per spot).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from gen3_multiscale.models.attention import ChunkedCrossAttention, QueryQuerySelfAttention
from gen3_multiscale.models.geometry_utils import compute_relative_geometry


def sinusoidal_time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """Scalar t in [0, 1] -> [dim] sinusoidal embedding (standard
    diffusion/flow-matching time conditioning). t must be a 0-D tensor --
    "one shared continuous-time value for the whole hole", never a
    per-query vector."""
    if t.ndim != 0:
        raise ValueError(f"t must be a 0-D (scalar) tensor, got shape {tuple(t.shape)}")
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float() * freqs
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros(1, device=t.device)], dim=-1)
    return embedding


class VelocityBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dense_threshold: int, sparse_k: int, chunk_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = QueryQuerySelfAttention(hidden_dim=hidden_dim, n_heads=n_heads, dense_threshold=dense_threshold, sparse_k=sparse_k)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = ChunkedCrossAttention(hidden_dim=hidden_dim, n_heads=n_heads, chunk_size=chunk_size)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self, hidden: torch.Tensor, query_coords: torch.Tensor,
        conditioning_hidden: torch.Tensor, conditioning_geometry: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        self_out, _mode = self.self_attn(self.norm_self(hidden), query_coords)
        hidden = residual + self_out

        residual = hidden
        cross_out = self.cross_attn(self.norm_cross(hidden), conditioning_hidden, conditioning_geometry)
        hidden = residual + cross_out

        residual = hidden
        hidden = residual + self.ffn(self.norm_ffn(hidden))
        return hidden


class VelocityNetwork(nn.Module):
    def __init__(
        self,
        residual_rank: int = 64,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 2,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.residual_rank = residual_rank
        self.time_embed_dim = time_embed_dim
        self.residual_in_proj = nn.Linear(residual_rank, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            VelocityBlock(hidden_dim, n_heads, dense_threshold, sparse_k, chunk_size) for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden_dim, residual_rank)

    def forward(
        self, noisy_coefficients: torch.Tensor, t: torch.Tensor,
        query_coords: torch.Tensor, conditioning_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_coefficients.shape[-1] != self.residual_rank:
            raise ValueError(
                f"noisy_coefficients' last dim ({noisy_coefficients.shape[-1]}) must equal "
                f"residual_rank ({self.residual_rank})"
            )
        if conditioning_hidden.shape[0] != noisy_coefficients.shape[0]:
            raise ValueError(
                "conditioning_hidden must have one row per query, matching noisy_coefficients"
            )
        hidden = self.residual_in_proj(noisy_coefficients)
        time_emb = self.time_mlp(sinusoidal_time_embedding(t, self.time_embed_dim))
        hidden = hidden + time_emb[None, :]  # same t for every query -- broadcast add

        conditioning_geometry = compute_relative_geometry(query_coords, query_coords)
        for block in self.blocks:
            hidden = block(hidden, query_coords, conditioning_hidden, conditioning_geometry)
        return self.out_proj(hidden)


def flow_matching_loss(
    velocity_network: VelocityNetwork, target_coefficients: torch.Tensor,
    query_coords: torch.Tensor, conditioning_hidden: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One shared t ~ Uniform(0, 1) and one shared-shape noise draw x0 ~
    N(0, I) for the WHOLE hole (matching "one shared continuous-time
    value for the whole hole" -- x0 has the same per-query independence
    as the target itself, but t is a single scalar). Linear path x_t =
    (1-t)*x0 + t*x1; target velocity is the constant x1 - x0."""
    n_query, rank = target_coefficients.shape
    t = torch.rand((), generator=generator, device=target_coefficients.device)
    x0 = torch.randn(n_query, rank, generator=generator, device=target_coefficients.device)
    x1 = target_coefficients
    x_t = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted_velocity = velocity_network(x_t, t, query_coords, conditioning_hidden)
    return torch.nn.functional.mse_loss(predicted_velocity, target_velocity)


@torch.no_grad()
def sample_residual_coefficients(
    velocity_network: VelocityNetwork, n_query: int, query_coords: torch.Tensor,
    conditioning_hidden: torch.Tensor, n_samples: int = 8, n_steps: int = 20,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Euler-integrate the learned ODE from t=0 (noise) to t=1 (data) for
    n_samples independent draws. Returns [n_samples, n_query, residual_rank].
    Multiple samples must NOT be identical (verified by test) -- each
    draw starts from an independent x0."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rank = velocity_network.residual_rank
    dt = 1.0 / n_steps
    device = query_coords.device

    # Sampling is inference, not training -- dropout must be off (an
    # active dropout mask draws from the global torch RNG on every
    # forward call, which would silently break reproducibility under a
    # fixed `generator` and inject uncontrolled extra noise on top of the
    # flow's own, deliberate x0 draws). Restore the caller's original
    # mode afterward so this function has no side effect on a model still
    # being trained.
    was_training = velocity_network.training
    velocity_network.eval()
    try:
        samples = []
        for _ in range(n_samples):
            x = torch.randn(n_query, rank, generator=generator, device=device)
            t = torch.zeros((), device=device)
            for _step in range(n_steps):
                velocity = velocity_network(x, t, query_coords, conditioning_hidden)
                x = x + velocity * dt
                t = t + dt
            samples.append(x)
        return torch.stack(samples, dim=0)
    finally:
        velocity_network.train(was_training)
