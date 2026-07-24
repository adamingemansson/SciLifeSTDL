"""Architecture 2 — scFoundation as the gene encoder (GPT's #1-ranked
pretrained option).

Hypothesis: GPT's top recommendation across the whole consultation — swap
the from-scratch MLP gene encoder for a real pretrained single-cell
foundation model. context["expression"] must be populated with precomputed
scFoundation cell embeddings (NOT raw/normalized gene expression) by the
training entrypoint's context_gene_feature_provider before reaching this
model — see gen2_architectures/training/data_prep.py. scFoundation's real
preprocessing contract (verified against its cloned source,
gen2_architectures/models/conditioning.py::precompute_scfoundation_features)
already does library-size-to-1e4 + log1p internally when fed
already_normalized_log1p=True with OUR OWN normalized-log1p expression, so
— unlike STPath (Architecture 4) — no special raw-count feed is needed
here; our project's default preprocessing and scFoundation's own real
formula coincide.

See gen2_architectures/README.md for the full spec with dimensions.
"""
from __future__ import annotations

from gen2_architectures.models.local_neighborhood_transformer import LocalNeighborhoodTransformer


class Architecture2(LocalNeighborhoodTransformer):
    def __init__(self, n_genes: int, scfoundation_dim: int, **kwargs):
        kwargs.pop("gene_encoder_type", None)
        super().__init__(
            n_genes=n_genes, gene_encoder_type="scfoundation",
            scfoundation_dim=scfoundation_dim, **kwargs,
        )
