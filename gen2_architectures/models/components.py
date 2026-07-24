"""Shared building blocks used by Architectures 1, 2 and 3's Stage B.

Everything here is genuinely NEW code (not ported from src/), except
RandomFourierFeatures which is reused as-is from
gen2_architectures/models/conditioning.py (copied from src/models/
conditioning.py) — that class already has a real, previously-fixed bug
(sigma=1.0 aliasing at real HEST-1k pixel-scale coordinates, see its own
docstring) worked out, so building a second from-scratch Fourier embedding
here would risk reintroducing the exact same bug rather than reusing the
fix.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen2_architectures.models.conditioning import RandomFourierFeatures


class CoordEmbedding(nn.Module):
    """Relative (dx, dy) -> learned Fourier positional embedding, GPT's
    PDF Component 3. Wraps RandomFourierFeatures (fixed random projection,
    coord_scale-corrected) with a small trainable MLP on top, exactly the
    "FourierFeatures(2->64) -> Linear(64,64) -> GELU -> Linear(64,64)"
    shape from the PDF, generalized to a configurable output width.

    coord_scale MUST be derived from real per-sample coordinate spread
    (the same inject_coord_scale discipline the original codebase already
    established — see conditioning.py's RandomFourierFeatures docstring for
    exactly why a wrong/default coord_scale silently aliases into noise at
    HEST-1k's real pixel-scale coordinates). Never leave this at 1.0 for
    real data.
    """

    def __init__(self, feat_dim: int = 64, num_fourier_features: int = 32,
                 sigma: float = 1.0, coord_scale: float = 1.0):
        super().__init__()
        self.fourier = RandomFourierFeatures(
            in_dim=2, num_features=num_fourier_features, sigma=sigma, coord_scale=coord_scale,
        )
        fourier_dim = 2 * num_fourier_features  # sin+cos concat
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, feat_dim), nn.GELU(), nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, relative_xy: torch.Tensor) -> torch.Tensor:
        """relative_xy: [N, 2] (dx, dy) relative to the query/missing spot.
        Returns [N, feat_dim]."""
        return self.mlp(self.fourier(relative_xy))


class ConfidenceEmbedding(nn.Module):
    """GPT review suggestion #4: every spot token should also carry whether
    it's OBSERVED or MASKED as a binary signal, so zero-expression and
    missing-expression don't look identical to the transformer. Tiny
    lookup table (2 rows), R^16 by default, per GPT's own sizing.

    Every context token passed to the transformer is, by construction,
    observed (query positions never appear as separate transformer input
    tokens in Architectures 1-3 — they're represented by the single
    learnable query token instead, see each architecture's own forward()).
    This embedding matters once contiguous/larger masks make it possible
    for a context NEIGHBOR itself to sit right at a hole's edge with a
    partially-unreliable image patch (context_images/image_available from
    masked_item.py) — appended alongside the mask-embedding-of-that-spot's
    IMAGE availability, not its expression (expression is never partially
    missing for an included context spot; images can be, via
    strict_broken_region's boundary-patch exclusion). Concretely: pass
    context["image_available"] (bool [N]) through this module to get a
    per-spot "was its own H&E patch excluded as unreliable" signal.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(2, embed_dim)

    def forward(self, available: torch.Tensor) -> torch.Tensor:
        """available: bool [N] (True = observed/reliable, False = masked/
        excluded). Returns [N, embed_dim]."""
        return self.embed(available.long())


def build_local_transformer(hidden_dim: int = 512, n_layers: int = 8, n_heads: int = 8,
                             mlp_ratio: float = 4.0, dropout: float = 0.1) -> nn.TransformerEncoder:
    """The shared spatial-reasoning core for Architectures 1/2/3's Stage B —
    a plain nn.TransformerEncoder over a SMALL, capped local neighborhood
    (~80 context tokens + 1 query token, via max_context_points +
    context_selection="nearest_query", see gen2_architectures/data/
    mask_bank.py::cap_context_mask — already a real k-NN cap, not a new
    mechanism). GPT's PDF recommends exactly this shape (8 layers, 512
    hidden, 8 heads, ~30-40M params) and explicitly argues AGAINST anything
    fancier (Perceiver/GNN/hierarchical attention) at this token count —
    full O(n^2) self-attention over ~81 tokens is cheap, nowhere near where
    a more complex architecture would start paying for itself.
    """
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim, nhead=n_heads, dim_feedforward=int(hidden_dim * mlp_ratio),
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class StagedGeneLoss(nn.Module):
    """GPT review's #1 suggestion (the single change it called most
    important): don't use a fixed 0.7*MSE + 0.3*(1-Pearson) loss from step
    one. Pearson gradients are unstable while predictions are still near
    their random initialization; a fixed Pearson weight from the start can
    fight the model before MSE has given it a sensible output scale.

    Three-stage schedule over TRAINING PROGRESS (0.0-1.0, i.e.
    current_step / total_steps — pass this in explicitly rather than
    having the loss module own a step counter, so it stays a pure function
    of (prediction, target, progress) and is trivially testable):

      progress <  stage1_end                  : loss = MSE only
      stage1_end <= progress < stage2_end      : loss = 0.9*MSE + 0.1*Pearson
      progress >= stage2_end                   : loss = 0.7*MSE + 0.3*Pearson

    Defaults (stage1_end=0.2, stage2_end=0.6) match GPT's own suggested
    "epoch 0-20% / 20-60% / 60%+" breakpoints, expressed as fractions of
    total training so the same StagedGeneLoss works regardless of how many
    actual epochs/steps a given architecture's config ends up using.

    Pearson is computed PER GENE across the batch's spots (columns = genes,
    rows = spots in the batch) — this requires more than one spot per
    training step to be meaningful; batch_size must stay well above 1 (see
    each architecture's config, batch_size >= 32).
    """

    def __init__(self, stage1_end: float = 0.2, stage2_end: float = 0.6,
                 stage2_pearson_weight: float = 0.1, stage3_pearson_weight: float = 0.3,
                 eps: float = 1e-8):
        super().__init__()
        if not 0.0 <= stage1_end <= stage2_end <= 1.0:
            raise ValueError("require 0 <= stage1_end <= stage2_end <= 1")
        self.stage1_end = float(stage1_end)
        self.stage2_end = float(stage2_end)
        self.stage2_pearson_weight = float(stage2_pearson_weight)
        self.stage3_pearson_weight = float(stage3_pearson_weight)
        self.eps = float(eps)

    def _mean_gene_pearson(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """1 - mean-over-genes Pearson correlation, batch-of-spots x genes.
        Differentiable (pure tensor ops, no detach) so gradients flow to
        the prediction. Returns a scalar; genes with zero target variance
        contribute 0 to the correlation term (a constant target makes
        Pearson undefined — treated as "no correlation penalty for this
        gene" rather than a NaN that would corrupt the whole batch loss)."""
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=0)
        denominator = (
            pred_centered.square().sum(dim=0).sqrt() * target_centered.square().sum(dim=0).sqrt()
        )
        valid = denominator > self.eps
        pearson_per_gene = torch.zeros_like(numerator)
        pearson_per_gene[valid] = numerator[valid] / denominator[valid].clamp_min(self.eps)
        return 1.0 - pearson_per_gene[valid].mean() if valid.any() else torch.zeros((), device=pred.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, progress: float) -> dict:
        """pred/target: [n_spots, n_genes]. progress: float in [0, 1],
        current_step / total_steps. Returns a dict with "loss" (the
        scalar to backprop) plus "mse"/"pearson_penalty"/"pearson_weight"
        for logging."""
        mse = nn.functional.mse_loss(pred, target)
        if progress < self.stage1_end or pred.shape[0] < 2:
            # Also fall back to pure MSE if the batch happens to have only
            # one spot (Pearson is undefined for a single observation) --
            # correctness fallback, not part of the staged schedule itself.
            pearson_weight = 0.0
        elif progress < self.stage2_end:
            pearson_weight = self.stage2_pearson_weight
        else:
            pearson_weight = self.stage3_pearson_weight
        pearson_penalty = (
            self._mean_gene_pearson(pred, target) if pearson_weight > 0.0
            else torch.zeros((), device=pred.device)
        )
        loss = (1.0 - pearson_weight) * mse + pearson_weight * pearson_penalty
        return {
            "loss": loss, "mse": mse.detach(), "pearson_penalty": pearson_penalty.detach(),
            "pearson_weight": pearson_weight,
        }
