"""
VQ-VAE stage 1: reconstruction-only. Building block for the eventual
VQ-VAE + autoregressive transformer registry entry (task #12,
docs/architecture_plan.md "Prioritization" #3) — validated on its own
before the autoregressive stage is added on top (docs/model_schematics.md
build order), per van den Oord et al. 2017, "Neural Discrete
Representation Learning" (NeurIPS) — the original VQ-VAE, peer-reviewed.
The generation-quality precedent for this family (discrete latent +
autoregressive) beating GAN baselines by up to 2 orders of magnitude on
FID/MMD for biological/brain-imaging data is documented in
docs/literature_review.md's Nature Machine Intelligence citation.

Token granularity (open question flagged in docs/model_schematics.md):
one token per cell's whole expression vector, not per-gene-chunk — the
simpler choice, consistent with WAE-GAN/FM-OT's own single-latent-vector
design for this same gene expression vector. Single-token-per-cell VQ-VAE
is also used directly on single-cell data in CASTLE (scATAC-seq) and
CellTok; multi-token/residual VQ (RVQ-Alpha) is a documented possible
upgrade if single-token reconstruction proves too lossy, not built here.

Codebook update: standard (non-EMA) straight-through VQ-VAE formulation
from the original paper — codebook loss + commitment loss via the
stop-gradient trick. EMA codebook updates (VQ-VAE-2, Razavi et al. 2019)
are a documented possible upgrade if the plain version undertrains the
codebook (watch train/codebook_usage below), not implemented here.

No external VQ library (e.g. vector-quantize-pytorch) — the layer itself
is small and well-specified; consistent with this codebase's existing
choice to avoid dependencies for well-understood, small components (see
src/models/conditioning.py's no-torch_geometric rationale).

Deliberately unconditioned (no spatial context c) — this stage only
learns to compress/reconstruct expression vectors; conditioning on
location/context is the autoregressive transformer's job in stage 2, kept
separate so reconstruction quality can be validated in isolation
(docs/model_schematics.md build order: "don't debug both stages'
bugs simultaneously").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl


class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook lookup with straight-through gradients
    (van den Oord et al. 2017, module docstring above)."""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.commitment_weight = commitment_weight

    def forward(self, z_e: torch.Tensor):
        # z_e: [N, latent_dim]
        dists = torch.cdist(z_e, self.codebook.weight)   # [N, codebook_size]
        idx = torch.argmin(dists, dim=-1)                  # [N]
        z_q = self.codebook(idx)                            # [N, latent_dim]

        codebook_loss = nn.functional.mse_loss(z_q, z_e.detach())
        commitment_loss = nn.functional.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()  # straight-through estimator
        return z_q_st, idx, vq_loss


class VQVAEStage1(pl.LightningModule):
    """expression [N, n_genes] -> continuous embedding -> nearest codebook
    token -> reconstructed expression. See module docstring for why this
    is deliberately unconditioned and single-token-per-cell."""

    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dim: int = 256,
                 codebook_size: int = 512, commitment_weight: float = 0.25,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.vq = VectorQuantizer(codebook_size, latent_dim, commitment_weight)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.lr = lr

    def forward(self, expression: torch.Tensor):
        z_e = self.encoder(expression)
        z_q, idx, vq_loss = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, idx, vq_loss

    def training_step(self, batch, batch_idx):
        expression = batch["expression"]
        x_hat, idx, vq_loss = self(expression)
        recon_loss = nn.functional.mse_loss(x_hat, expression)
        loss = recon_loss + vq_loss
        codebook_usage = torch.unique(idx).numel() / self.vq.codebook_size
        self.log_dict({
            "train/recon": recon_loss, "train/vq": vq_loss, "train/loss": loss,
            "train/codebook_usage": codebook_usage,
        })
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
