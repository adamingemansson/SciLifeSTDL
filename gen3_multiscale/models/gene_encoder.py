"""The `weighted_linear` gene conditioning encoder -- copied VERBATIM
from `src/models/hierarchical_slide.py::WeightedGeneExpressionEncoder`
(same provenance discipline as every other reused module in this package,
CONTRACT.md section 2: "do not let this drift without a deliberate
reason").

Confirmed real gap, closed by this file (Codex audit finding #7 against
commit 386bcf4, verified before fixing): CONTRACT.md section 10 records
`weighted_linear` as the frozen, evidence-backed gene-encoder choice for
all four architectures, and every `configs/architectureN.yaml` names it
via `model.params.gene_encoder_type` -- but until this file existed, no
module ANYWHERE in `gen3_multiscale/` actually computed
`observed_gex_conditioning` from raw expression; `SpatialFieldInputs`
only ever documented that field as already-encoded, upstream input.
`WeightedGeneExpressionEncoder` is the real, audited, trainable module
that would do that encoding.

This file does NOT close the full gap on its own: nothing in
`gen3_multiscale/` yet CALLS this encoder inside a real per-sample data
builder to actually populate `observed_gex_conditioning` from raw counts
-- that wiring (plus training-only normalization/scale fitting for the
encoder's input) is part of the still-missing real training system
(CONTRACT.md section 21), not something this module can decide on its
own. What this file provides is the previously entirely-absent
implementation itself, ready to be called once that wiring exists.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WeightedGeneExpressionEncoder(nn.Module):
    """STPath-style expression-weighted gene-vocabulary embedding.

    With log-normalized expression ``x`` and one learned vector per gene,
    ``x @ W`` is exactly a weighted bag of gene embeddings. Cross-spot and
    nonlinear processing belongs to the spatial context Transformer, rather
    than an oversized pointwise MLP before spatial information is introduced.
    """

    def __init__(self, n_genes: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(n_genes, output_dim, bias=False)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(f"expression must be [N,G], got {tuple(expression.shape)}")
        return self.projection(expression)
