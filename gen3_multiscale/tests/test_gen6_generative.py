from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen6.generative import (
    Gen6ResidualOTFlowModel, Gen6WAEGANModel, sinkhorn_barycentric_ot_pairing,
)
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis


class _Conditioner(nn.Module):
    def __init__(self, n_genes=6, hidden=16):
        super().__init__()
        self.proj = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, n_genes)

    def forward(self, inputs):
        coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
        hidden = self.proj(coords)
        return {"expression": self.out(hidden), "query_hidden": hidden}


def _inputs(n=8):
    return SimpleNamespace(query_coords=np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32))


def test_sinkhorn_pairing_shape_finite_and_deterministic():
    torch.manual_seed(0)
    noise, target = torch.randn(9, 4), torch.randn(9, 4)
    first = sinkhorn_barycentric_ot_pairing(noise, target, epsilon=0.2, n_iters=30)
    second = sinkhorn_barycentric_ot_pairing(noise, target, epsilon=0.2, n_iters=30)
    assert first.shape == target.shape and torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


def test_residual_ot_flow_updates_flow_but_freezes_conditioner():
    genes = [f"g{i}" for i in range(6)]
    basis = fit_gene_residual_basis(np.random.default_rng(1).normal(size=(20, 6)), genes, rank=3)
    model = Gen6ResidualOTFlowModel(
        _Conditioner(), basis, genes, hidden_dim=16, n_heads=4,
        n_flow_blocks=1, dense_threshold=20, n_flow_samples=2, n_ode_steps=2,
    )
    inputs, target = _inputs(), torch.randn(8, 6)
    loss = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(3))["flow_loss"]
    loss.backward()
    assert any(p.grad is not None for p in model.velocity_network.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.conditioner.parameters())
    prediction = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(4))
    assert prediction["predictive_mean"].shape == target.shape


def test_wae_gan_one_optimizer_loss_routes_gradients_correctly():
    model = Gen6WAEGANModel(
        _Conditioner(), 6, latent_dim=4, hidden_dim=16,
        conditioner_hidden_dim=16, discriminator_hidden_dim=8, n_samples=2,
    )
    inputs, target = _inputs(), torch.randn(8, 6)
    losses = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(5))
    losses["total"].backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
    assert any(p.grad is not None for p in model.discriminator.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.conditioner.parameters())
    prediction = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(6))
    assert prediction["predictive_mean"].shape == target.shape
