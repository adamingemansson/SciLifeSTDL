"""Global observed-GEX inducing pool and global-slide FiLM conditioning --
Phase 5 items 7-8 of the multiscale spatial-field handoff. Architecture
3/4-specific per the fairness matrix (Architectures 1/2 have neither), but
Phase 5 lists both as shared token/context modules to build now so
Architecture 3/4's wrappers (Phase 6) have something ready to consume.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class InducedGlobalGEXPool(nn.Module):
    """"Use 16 learned inducing queries to cross-attend to all observed
    GEX tokens. Each inducing query produces both a hidden molecular-
    context token and a convex weighted mixture of the corresponding
    untouched observed full-gene vectors... Complexity is linear in the
    number of observed spots for a fixed inducing-token count. Query
    spots must be excluded before this pooling operation."

    Query exclusion is enforced structurally, not by a runtime check:
    forward() only accepts observed_hidden/observed_expression -- there
    is no argument through which a query spot's hidden state or
    expression could ever be passed in.

    The multi-head attention that builds each inducing token's HIDDEN
    (molecular-context) output keeps full per-head expressivity; the
    convex mixture applied to the untouched real observed_expression uses
    ONE distribution per inducing token (the mean of the per-head
    weights, still convex -- a mean of convex combinations is convex),
    since the handoff calls for a single value-preserving candidate per
    inducing query, not one per head."""

    def __init__(self, hidden_dim: int = 512, n_inducing: int = 16, n_heads: int = 8):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.hidden_dim = hidden_dim
        self.n_inducing = int(n_inducing)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.inducing_queries = nn.Parameter(torch.randn(self.n_inducing, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, observed_hidden: torch.Tensor, observed_expression: torch.Tensor) -> dict:
        n_observed = observed_hidden.shape[0]
        if observed_expression.shape[0] != n_observed:
            raise ValueError(
                f"observed_expression has {observed_expression.shape[0]} rows, expected "
                f"n_observed={n_observed}"
            )
        if n_observed == 0:
            raise ValueError("observed_hidden is empty -- nothing to pool")

        q = self.query_proj(self.inducing_queries).view(self.n_inducing, self.n_heads, self.head_dim)
        k = self.key_proj(observed_hidden).view(n_observed, self.n_heads, self.head_dim)
        v = self.value_proj(observed_hidden).view(n_observed, self.n_heads, self.head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)

        logits = torch.einsum("ihd,chd->ihc", q, k) * scale  # [n_inducing, heads, n_observed]
        weights = torch.softmax(logits, dim=-1)  # convex per (inducing token, head)
        hidden_out = torch.einsum("ihc,chd->ihd", weights, v).reshape(self.n_inducing, self.hidden_dim)
        hidden_out = self.out_norm(self.out_proj(hidden_out))

        value_weights = weights.mean(dim=1)  # [n_inducing, n_observed], convex per inducing token
        value_expression = value_weights @ observed_expression  # [n_inducing, G], real values only

        return {
            "hidden": hidden_out,
            "expression": value_expression,
            "value_weights": value_weights,
        }


class GlobalConditioningFiLM(nn.Module):
    """"Global LongNet conditioning through a dedicated gated residual or
    AdaLN/FiLM modulation" (Phase 5 item 8 / Phase 3's fusion step 6).

    FiLM: token = token * (1 + scale(global)) + shift(global). Both
    projections are zero-initialized -- at construction this module is
    the IDENTITY function (contributes nothing) until training moves the
    weights, matching the same zero-init discipline already used for the
    transport head's residual (Phase 4) and giving Architecture 3/4's
    later "slide-token zero/swap must measurably change the prediction"
    diagnostic (Phase 7) a clean, testable starting point: a freshly
    constructed model is UNAFFECTED by the global token by design, so any
    later dependence is something training actually learned, not an
    artifact of initialization."""

    def __init__(self, hidden_dim: int, global_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(global_dim, hidden_dim)
        self.shift_proj = nn.Linear(global_dim, hidden_dim)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, token_hidden: torch.Tensor, global_vector: torch.Tensor) -> torch.Tensor:
        if global_vector.ndim != 1:
            raise ValueError(f"global_vector must be 1-D [global_dim], got shape {tuple(global_vector.shape)}")
        scale = self.scale_proj(global_vector)  # [hidden_dim]
        shift = self.shift_proj(global_vector)  # [hidden_dim]
        return token_hidden * (1.0 + scale) + shift
