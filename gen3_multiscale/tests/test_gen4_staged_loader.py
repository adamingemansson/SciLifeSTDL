"""Item 4 (six-launch-blocker audit): a real load -> verify -> freeze ->
eval staged-conditioner loader shared by Gen4 and Gen5 flow training,
mirroring training/train.py's own audited Architecture4 discipline."""
from __future__ import annotations

import numpy as np
import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.flow import Gen4ResidualFlowModel
from gen3_multiscale.gen4.staged_loader import load_and_freeze_deterministic_conditioner
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.training import checkpoint as checkpoint_module

# load_shared_autoencoder (the Gen5-only half of Item 4) is exercised in
# the Gen5 worktree's own test suite -- gen3_multiscale.gen5 does not
# exist in this (Gen4) worktree/branch at all.

N_GENES, GEX_DIM = 6, 4
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]
MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4)


def _gene_basis():
    residuals = np.random.default_rng(0).normal(size=(20, N_GENES))
    return fit_gene_residual_basis(residuals, GENE_NAMES, rank=3)


def _build_flow_model(seed: int) -> Gen4ResidualFlowModel:
    torch.manual_seed(seed)
    return Gen4ResidualFlowModel(
        n_genes=N_GENES, gex_feature_dim=GEX_DIM, gene_basis=_gene_basis(), gene_names=GENE_NAMES,
        image_feature_dim=8, gex_feature_source="weighted_linear", image_feature_source="precomputed",
        global_context_source="none", n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **MODEL_KWARGS,
    )


def test_load_and_freeze_deterministic_conditioner_loads_real_weights_and_freezes(tmp_path):
    trained = _build_flow_model(seed=0)
    checkpoint_dir = tmp_path / "gen4a_conditioner"
    checkpoint_module.save_checkpoint(trained.conditioner, {"arm": "gen4a"}, GENE_NAMES, checkpoint_dir, step=100)

    fresh = _build_flow_model(seed=1)  # a DIFFERENT random init -- proves the load actually overwrites it
    trained_weight = trained.conditioner.state_dict()["gene_encoder.projection.weight"].clone()
    fresh_weight_before = fresh.conditioner.state_dict()["gene_encoder.projection.weight"].clone()
    assert not torch.allclose(trained_weight, fresh_weight_before)

    info = load_and_freeze_deterministic_conditioner(fresh, str(checkpoint_dir), GENE_NAMES)

    assert info["loaded"] is True
    assert info["checkpoint_step"] == 100
    assert info["checkpoint_sha256"] is not None
    fresh_weight_after = fresh.conditioner.state_dict()["gene_encoder.projection.weight"]
    assert torch.allclose(trained_weight, fresh_weight_after)
    assert all(not p.requires_grad for p in fresh.conditioner.parameters())
    assert fresh.conditioner.training is False


def test_load_and_freeze_deterministic_conditioner_can_skip_freezing(tmp_path):
    trained = _build_flow_model(seed=0)
    checkpoint_dir = tmp_path / "gen4a_conditioner"
    checkpoint_module.save_checkpoint(trained.conditioner, {"arm": "gen4a"}, GENE_NAMES, checkpoint_dir, step=1)

    fresh = _build_flow_model(seed=1)
    load_and_freeze_deterministic_conditioner(fresh, str(checkpoint_dir), GENE_NAMES, freeze=False)
    assert all(p.requires_grad for p in fresh.conditioner.parameters())


def test_load_and_freeze_deterministic_conditioner_rejects_gene_panel_mismatch(tmp_path):
    trained = _build_flow_model(seed=0)
    checkpoint_dir = tmp_path / "gen4a_conditioner"
    checkpoint_module.save_checkpoint(trained.conditioner, {"arm": "gen4a"}, GENE_NAMES, checkpoint_dir, step=1)

    fresh = _build_flow_model(seed=1)
    wrong_gene_names = list(reversed(GENE_NAMES))
    import pytest
    with pytest.raises(ValueError, match="different gene panel"):
        load_and_freeze_deterministic_conditioner(fresh, str(checkpoint_dir), wrong_gene_names)
