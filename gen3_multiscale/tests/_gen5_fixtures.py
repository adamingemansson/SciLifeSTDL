"""Shared Gen5 test fixtures. NOT a test file itself. Reuses
gen4._gen4_fixtures's synthetic-example/stub-encoder machinery directly
(imported, not duplicated) and adds an autoencoder fixture."""
from __future__ import annotations

import numpy as np

from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder
from gen3_multiscale.tests._gen4_fixtures import (  # noqa: F401 -- re-exported for Gen5 tests
    GEN4_MODEL_KWARGS, StubSCFoundationEncoder, StubUNI2Encoder, synthetic_gen4_inputs, with_synthetic_wsi_context,
)

GEN5_MODEL_KWARGS = GEN4_MODEL_KWARGS


def tiny_autoencoder(n_genes: int = 6, latent_dim: int = 8, seed: int = 0) -> ExpressionAutoencoder:
    import torch
    gene_names = [f"g{i}" for i in range(n_genes)]
    torch.manual_seed(seed)
    return ExpressionAutoencoder(n_genes, gene_names, latent_dim=latent_dim, hidden_dim=16)


def synthetic_expression(n_rows: int = 40, n_genes: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_rows, n_genes)).astype(np.float32)
