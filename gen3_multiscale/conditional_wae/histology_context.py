"""Composition-based histology-structure context injection for the MK
conditional WAE.

`HistologyContextInjector` wraps an EXISTING image conditioner (e.g.
`Architecture1ImageConditioner`) without modifying it at all -- every
other arm keeps constructing and using the plain, unwrapped conditioner
exactly as before. It exposes the identical `forward(inputs) ->
[N, hidden_dim]` interface (plus the `n_genes`/`hidden_dim` attributes
`ConditionalWAE` reads), so `_build_model` only needs to choose WHICH
object to pass as `image_conditioner=` -- no changes anywhere in
`ConditionalWAE`/`Architecture1ImageConditioner` themselves.

The injected signal is `histology_features.py`'s deterministic
multiscale morphology descriptor (never a model, never GEX) -- see that
module's docstring for why this is complementary to, not a duplicate of,
the wrapped conditioner's own opaque geometry-attention mixing. The
projection's final layer is zero-initialized, so this wrapper computes
an EXACT identity (`context + 0 == context`) at construction -- the same
"strict superset" discipline `_FiLMGenerator` (model.py) and
`GeneCoexpressionRefinement` (coexpression.py) already use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.conditional_wae.inputs import FullImageExpressionInputs


class HistologyContextInjector(nn.Module):
    def __init__(self, image_conditioner: nn.Module, histology_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        if histology_feature_dim < 1:
            raise ValueError("histology_feature_dim must be positive")
        self.image_conditioner = image_conditioner
        self.n_genes = int(image_conditioner.n_genes)
        self.context_dim = int(image_conditioner.hidden_dim)
        self.histology_feature_dim = int(histology_feature_dim)
        self.project = nn.Sequential(
            nn.LayerNorm(self.histology_feature_dim), nn.Linear(self.histology_feature_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, self.context_dim),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    @property
    def hidden_dim(self) -> int:
        return self.context_dim

    def forward(self, inputs: FullImageExpressionInputs) -> torch.Tensor:
        context = self.image_conditioner(inputs)
        if inputs.histology_features is None:
            raise ValueError(
                "HistologyContextInjector requires inputs.histology_features -- was "
                "data.use_histology_features left unset when this sample was loaded?"
            )
        # image_conditioner(inputs) returns context for QUERY rows only
        # (Architecture1ImageConditioner.forward's own `hidden[query_mask]`
        # contract) -- histology_features is a full [N, D] slide-row array,
        # so it must be sliced by the same query_mask before adding.
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=context.device)
        histology = torch.as_tensor(inputs.histology_features, dtype=context.dtype, device=context.device)
        if histology.ndim != 2 or histology.shape[0] != query_mask.shape[0]:
            raise ValueError(
                f"histology_features must be [{query_mask.shape[0]}, histology_feature_dim], "
                f"got {tuple(histology.shape)}"
            )
        histology = histology[query_mask]
        if histology.shape[0] != context.shape[0]:
            raise ValueError(
                f"histology_features[query_mask] has {histology.shape[0]} rows, expected "
                f"{context.shape[0]} (one per query row image_conditioner returned)"
            )
        if histology.shape[1] != self.histology_feature_dim:
            raise ValueError(
                f"histology_features has {histology.shape[1]} columns, expected {self.histology_feature_dim}"
            )
        return context + self.project(histology)
