"""Architecture 1 — "GPT-v1 faithful": small local-neighborhood transformer,
from-scratch MLP gene encoder.

Hypothesis: GPT's own most conservative, concrete recommendation already
recovers a meaningful chunk of the gap to the reference STPath notebook,
with nothing more exotic than components already validated in this project.
See gen2_architectures/README.md for the full spec with dimensions.
"""
from __future__ import annotations

from gen2_architectures.models.local_neighborhood_transformer import LocalNeighborhoodTransformer


class Architecture1(LocalNeighborhoodTransformer):
    def __init__(self, n_genes: int, **kwargs):
        kwargs.pop("gene_encoder_type", None)
        kwargs.pop("scfoundation_dim", None)
        super().__init__(n_genes=n_genes, gene_encoder_type="mlp", **kwargs)
