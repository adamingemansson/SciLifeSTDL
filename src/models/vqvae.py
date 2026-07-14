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

Codebook update: EMA (van den Oord et al. 2017, Appendix A.1; made the
default in Razavi et al. 2019, VQ-VAE-2) rather than the gradient-based
codebook loss from the main text of the original paper. Switched
2026-07-14 after the gradient-based version collapsed to 1/512 codes on
real HEST-1k data (see docs/model_schematics.md) — a well-known failure
mode of that formulation (rich-get-richer: only the current nearest code
gets pushed toward the data, so early winners keep winning). EMA also
comes with dead-code reset (reinitialize codes whose EMA usage falls near
zero to a random real encoder output from the current batch) — standard
engineering practice in most working VQ-VAE implementations (e.g. Jukebox,
EnCodec), not itself a peer-reviewed claim, just how EMA-VQ is normally
deployed.

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
    and an EMA-updated codebook (van den Oord et al. 2017 Appendix A.1;
    Razavi et al. 2019 VQ-VAE-2 — module docstring above). The codebook is
    a buffer, not a trainable nn.Embedding — it's updated directly from
    encoder outputs each training step, not via backprop, so it never
    appears in an optimizer's parameter list."""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_weight: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5, dead_code_threshold: float = 1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(codebook_size, latent_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", embed.clone())

    def forward(self, z_e: torch.Tensor):
        # z_e: [N, latent_dim]
        dists = torch.cdist(z_e, self.embed)   # [N, codebook_size]
        idx = torch.argmin(dists, dim=-1)        # [N]
        z_q = self.embed[idx]                     # [N, latent_dim]

        commitment_loss = self.commitment_weight * nn.functional.mse_loss(z_e, z_q.detach())

        if self.training:
            with torch.no_grad():
                one_hot = nn.functional.one_hot(idx, self.codebook_size).type(z_e.dtype)  # [N, K]
                cluster_size = one_hot.sum(dim=0)                                           # [K]
                embed_sum = one_hot.t() @ z_e                                               # [K, D]

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                smoothed_size = (
                    (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
                )
                self.embed.copy_(self.ema_embed_sum / smoothed_size.unsqueeze(1))

                # dead-code reset: codes the EMA has essentially stopped
                # using get snapped to a random real encoder output from
                # this batch, so they get a chance to re-enter rotation
                dead = self.ema_cluster_size < self.dead_code_threshold
                if dead.any():
                    n_dead = int(dead.sum().item())
                    random_idx = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
                    revived = z_e[random_idx]
                    self.embed[dead] = revived
                    self.ema_embed_sum[dead] = revived
                    self.ema_cluster_size[dead] = 1.0

        z_q_st = z_e + (z_q - z_e).detach()  # straight-through estimator
        return z_q_st, idx, commitment_loss


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


def morton_order(coords: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """Deterministic space-filling-curve (Morton/Z-order) ordering over
    arbitrary-dimensional coordinates. Generalizes the fixed raster
    ordering used for autoregressive token sequences in this family's
    anchor precedent (Tudosiu et al., "Realistic morphology-preserving
    generative modelling of the brain", Nature Machine Intelligence 2024
    — docs/literature_review.md SS3.2b, which orders tokens over a regular
    voxel grid) to an irregular spatial point cloud, by quantizing
    coordinates onto a grid and interleaving their bits (Morton, 1966).

    Returns an index tensor `order` such that coords[order] is sorted
    along the curve.
    """
    coords = coords.detach().cpu()
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    ranges = (maxs - mins).clamp(min=1e-6)
    levels = (1 << bits) - 1
    quantized = ((coords - mins) / ranges * levels).long().clamp(0, levels)

    n, d = quantized.shape
    codes = torch.zeros(n, dtype=torch.long)
    for dim in range(d):
        for b in range(bits):
            bit = (quantized[:, dim] >> b) & 1
            codes = codes | (bit << (b * d + dim))
    return torch.argsort(codes)
