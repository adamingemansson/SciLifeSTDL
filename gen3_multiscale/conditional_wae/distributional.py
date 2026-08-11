"""A real predictive DISTRIBUTION head for H&E->GEX, replacing point regression.

Why this exists (all measured in this project, Aug 2026). The conditional WAE
produced badly miscalibrated uncertainty -- standardized-residual std of 9-19
against an ideal of 1.0, and 68% intervals covering 15-31% of the truth. Two
latent-side fixes failed, and a direct decomposition explained why: the latent
moves the output by 17-33% but its contribution correlates 0.002-0.025 with the
error it would need to explain. Uncertainty was never a latent problem; the
model simply had no way to SAY how uncertain it was, because a squared-error
objective only ever expresses a conditional mean.

Why not NB/ZINB. Hist2ST, HGGEP, STFlow and SHEST all use a (zero-inflated)
negative binomial, and that is right for THEIR targets, which are raw counts.
This project's manifest applies ``normalize_total(1e4)`` then ``log1p``
(``expression_transform="normalize_log1p"``), so targets are continuous
non-negative reals, and an NB likelihood would be misspecified. What survives
the transform is the property that actually matters: ``log1p(0) == 0`` exactly,
so the target keeps a large point mass at zero (measured at 82-99% of spots for
the marker genes we examined). The correct analogue is therefore a HURDLE
(zero-inflated continuous) likelihood:

    P(y = 0)  = pi
    p(y | y>0) = (1 - pi) * Normal(y | mu, sigma^2)

This gives the model three things point regression cannot:

* it can commit to an exact zero, instead of hedging toward a gene's mean --
  the precise failure behind the flat UMOD map, where a gene absent from 12 of
  13 kidney slides was predicted at a uniform moderate level everywhere;
* it emits a per-spot, per-gene sigma, so predictive uncertainty is ANALYTIC
  and needs no Monte-Carlo sampling over a latent; and
* its calibration is trained rather than incidental, since the NLL is a proper
  scoring rule.

No published H&E->ST method reports any calibration or predictive-distribution
metric, so this is the differentiator, not a reimplementation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG_SIGMA_MIN = -7.0
_LOG_SIGMA_MAX = 5.0
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class ZeroInflatedGaussianHead(nn.Module):
    """Per-spot, per-gene (pi, mu, sigma) from the image context.

    Emits three tensors of shape ``[N, n_genes]``: the zero logit, the mean of
    the positive part, and the log standard deviation of the positive part.
    """

    def __init__(self, n_genes: int, context_dim: int, *, hidden_dim: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        if n_genes < 1 or context_dim < 1 or hidden_dim < 1:
            raise ValueError("n_genes, context_dim and hidden_dim must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_genes = int(n_genes)
        self.trunk = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.zero_logit = nn.Linear(hidden_dim, n_genes)
        self.mean = nn.Linear(hidden_dim, n_genes)
        self.log_sigma = nn.Linear(hidden_dim, n_genes)
        # Start near "one shared, moderate sigma per gene" rather than a wildly
        # varying one: a large initial spread makes the NLL flat and the mean
        # head learns nothing for many steps.
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, -1.0)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(context)
        log_sigma = self.log_sigma(hidden).clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        return self.zero_logit(hidden), self.mean(hidden), log_sigma


def zero_inflated_gaussian_nll(zero_logit: torch.Tensor, mean: torch.Tensor,
                               log_sigma: torch.Tensor, target: torch.Tensor,
                               *, zero_tolerance: float = 0.0) -> torch.Tensor:
    """Negative log-likelihood of the hurdle model, averaged over all entries.

    ``zero_tolerance`` treats targets at or below it as exact zeros; the
    default of 0.0 is exact equality, which is correct for ``log1p`` output
    where an unobserved gene is precisely 0.

    Both branches are computed with ``logsigmoid`` rather than ``log(sigmoid)``
    so neither saturates, and the Gaussian branch is evaluated on every entry
    (then masked) to keep the graph dense and free of NaN-producing gathers.
    """
    if not (zero_logit.shape == mean.shape == log_sigma.shape == target.shape):
        raise ValueError("zero_logit, mean, log_sigma and target must have identical shapes")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be non-negative")
    is_zero = target <= zero_tolerance
    # log P(y=0) = logsigmoid(zero_logit); log P(y>0) = logsigmoid(-zero_logit)
    log_zero = F.logsigmoid(zero_logit)
    log_positive = F.logsigmoid(-zero_logit)
    standardized = (target - mean) * torch.exp(-log_sigma)
    gaussian_log_prob = -(_HALF_LOG_2PI + log_sigma + 0.5 * standardized.square())
    log_likelihood = torch.where(is_zero, log_zero, log_positive + gaussian_log_prob)
    return -log_likelihood.mean()


def zero_inflated_gaussian_moments(zero_logit: torch.Tensor, mean: torch.Tensor,
                                   log_sigma: torch.Tensor
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic predictive mean and standard deviation of the hurdle model.

    With ``p = P(y > 0) = sigmoid(-zero_logit)``:

        E[y]   = p * mu
        E[y^2] = p * (sigma^2 + mu^2)
        Var[y] = E[y^2] - E[y]^2

    Returning these directly is what removes the Monte-Carlo sampling over a
    latent that produced the observed under-dispersion: the spread is a trained
    parameter of the model rather than the scatter of decoder outputs.
    """
    positive = torch.sigmoid(-zero_logit)
    variance_positive = torch.exp(2.0 * log_sigma)
    predictive_mean = positive * mean
    second_moment = positive * (variance_positive + mean.square())
    predictive_variance = (second_moment - predictive_mean.square()).clamp_min(0.0)
    return predictive_mean, predictive_variance.sqrt()
