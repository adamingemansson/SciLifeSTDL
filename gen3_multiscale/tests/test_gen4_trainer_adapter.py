"""Item 5 (six-launch-blocker audit): `training/train.py::build_model_
for_inference` dispatches to `gen4.trainer_adapter` for a Gen4 config
(`model.arm` present) -- exercised here through THAT single shared entry
point (never gen4.trainer_adapter directly-only), since that is exactly
what the trainer/evaluator/overfit-gate/basis-fitter all call."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from gen3_multiscale.gen4.trainer_adapter import is_gen4_config
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, save_gene_residual_basis
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train import build_model_for_inference

_CONFIG_DIR_GEN4 = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "gen4"

N_GENES = 5
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


def _tiny_conditioner_config():
    config = yaml.safe_load((_CONFIG_DIR_GEN4 / "gen4a_conditioner.yaml").read_text())
    config["model"]["params"].update(
        n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
        dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    )
    return config


def _tiny_flow_config():
    config = yaml.safe_load((_CONFIG_DIR_GEN4 / "gen4a_flow.yaml").read_text())
    config["model"]["params"].update(
        n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
        dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, gene_basis_rank=3,
    )
    return config


def test_is_gen4_config_distinguishes_gen3_and_gen4_schemas():
    assert is_gen4_config({"model": {"arm": "gen4a"}}) is True
    assert is_gen4_config({"model": {"architecture": "3"}}) is False


def test_build_model_for_inference_dispatches_to_gen4_conditioner(tmp_path):
    config = _tiny_conditioner_config()
    model, info = build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True)
    from gen3_multiscale.gen4.conditioner import Gen4Conditioner

    assert isinstance(model, Gen4Conditioner)
    assert info["conditioner_info"]["loaded"] is False  # not a flow model -- no staged conditioner to load


def test_build_model_for_inference_gen4_flow_requires_real_conditioner_checkpoint_for_non_smoke():
    config = _tiny_flow_config()
    config["required_fingerprints"]["gene_residual_basis"] = None
    config["required_fingerprints"]["gen4_conditioner_checkpoint"] = None
    with pytest.raises(ValueError, match="gene_residual_basis"):
        build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False)


def test_build_model_for_inference_gen4_flow_stages_and_freezes_the_real_conditioner(tmp_path):
    """End-to-end through the SAME shared entry point the trainer,
    evaluator, and overfit gate all call: a real gen4a conditioner
    checkpoint gets loaded onto the flow model's own conditioner and
    frozen -- Item 4's staged loader, reached via Item 5's dispatch."""
    conditioner_config = _tiny_conditioner_config()
    trained_conditioner, _ = build_model_for_inference(
        conditioner_config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True,
    )
    conditioner_checkpoint_dir = tmp_path / "gen4a_conditioner_checkpoint"
    checkpoint_module.save_checkpoint(
        trained_conditioner, {"arm": "gen4a"}, GENE_NAMES, conditioner_checkpoint_dir, step=50,
    )

    basis_path = tmp_path / "gene_residual_basis.pt"
    residuals = np.random.default_rng(0).normal(size=(20, N_GENES))
    save_gene_residual_basis(fit_gene_residual_basis(residuals, GENE_NAMES, rank=3), basis_path)

    flow_config = _tiny_flow_config()
    flow_config["required_fingerprints"]["gene_residual_basis"] = str(basis_path)
    flow_config["required_fingerprints"]["gen4_conditioner_checkpoint"] = str(conditioner_checkpoint_dir)

    flow_model, info = build_model_for_inference(
        flow_config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False,
    )
    assert info["conditioner_info"]["loaded"] is True
    assert info["conditioner_info"]["checkpoint_step"] == 50
    assert all(not p.requires_grad for p in flow_model.conditioner.parameters())
    assert flow_model.conditioner.training is False
    trained_weight = trained_conditioner.state_dict()["gene_encoder.projection.weight"]
    flow_weight = flow_model.conditioner.state_dict()["gene_encoder.projection.weight"]
    assert torch.allclose(trained_weight, flow_weight)
