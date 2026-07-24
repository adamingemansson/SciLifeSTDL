"""Architecture 3, Stage A — self-supervised DENOISING transcriptome
autoencoder.

GPT review suggestion #3 (its "biggest suggested modification" to the
original plan): a plain genes->latent->genes autoencoder can collapse
toward a near-identity mapping, since copying the input through is often
the easiest way to minimize reconstruction loss for a wide, redundant
input like gene expression. Corrupting the INPUT (masking a fraction of
genes to zero, and/or adding Gaussian noise) while still reconstructing
the CLEAN target forces the latent to encode actual biological structure
(genes that move together) rather than an near-identity shortcut.

Trained on every observed spot from every training slide, independent of
the spatial task — no context/query split, no image data, just
expression -> expression. Cheap relative to Stage B's spatial training
(no images, no transformer, no k-NN neighborhoods): plan to run this
first, on a fraction of a GPU-day, before Stage B.

Stage B (arch3_stage_b_latent_transformer.py) reuses this class's encoder
and decoder directly (frozen, or fine-tuned at a much smaller LR than the
spatial transformer, per GPT's own suggestion) rather than duplicating the
architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenoisingTranscriptomeAutoencoder(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int = 256,
                 hidden_dims: tuple[int, int] = (4096, 1024)):
        super().__init__()
        h1, h2 = hidden_dims
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, h1), nn.GELU(), nn.LayerNorm(h1),
            nn.Linear(h1, h2), nn.GELU(), nn.LayerNorm(h2),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.GELU(), nn.LayerNorm(h2),
            nn.Linear(h2, h1), nn.GELU(), nn.LayerNorm(h1),
            nn.Linear(h1, n_genes),
        )

    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        return self.encoder(expression)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(expression))


def corrupt_expression(
    expression: torch.Tensor, seed: int,
    mask_fraction: float = 0.2, gaussian_std: float = 0.0,
) -> torch.Tensor:
    """Corrupt clean expression for denoising-autoencoder training.

    mask_fraction (GPT's suggested 20%): independently zeros that fraction
    of genes PER ROW (each spot gets its own random gene subset masked, not
    the same genes across the whole batch — otherwise the encoder could
    just learn "genes at these fixed positions are always garbage, ignore
    them" rather than genuinely reconstructing from partial evidence).
    gaussian_std > 0 additionally adds N(0, gaussian_std) noise on top —
    GPT phrased mask/noise as alternatives ("or"), but nothing stops using
    both; gaussian_std=0.0 (default) skips this entirely, matching the
    simpler "just masking" reading of the suggestion.

    Deterministic given seed, so a fixed validation corruption can be
    reused across epochs for a stable pretraining validation curve."""
    if not 0.0 <= mask_fraction < 1.0:
        raise ValueError(f"mask_fraction must be in [0, 1), got {mask_fraction}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_prob = 1.0 - mask_fraction
    mask = (torch.rand(expression.shape, generator=generator).to(expression.device) < keep_prob).float()
    corrupted = expression * mask
    if gaussian_std > 0.0:
        noise = torch.randn(expression.shape, generator=generator).to(expression.device) * gaussian_std
        corrupted = corrupted + noise
    return corrupted
