"""Integration audit follow-up: `training/train.py::build_model_for_
inference` dispatches Gen5 `latent_flow` configs to
`gen4.trainer_adapter.build_gen5_model_for_inference` (via
`build_gen4_or_gen5_model_for_inference`) -- exercised here through THAT
single shared entry point, same discipline as
tests/test_gen4_trainer_adapter.py."""
from __future__ import annotations

import pathlib

import torch
import yaml

from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder, save_expression_autoencoder_checkpoint
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train import build_model_for_inference

_CONFIG_DIR_GEN5 = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gen5"

N_GENES, LATENT_DIM = 5, 4
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


def _tiny_gen5a_config():
    config = yaml.safe_load((_CONFIG_DIR_GEN5 / "gen5a.yaml").read_text())
    config["model"]["params"].update(
        n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, latent_dim=LATENT_DIM, hidden_dim=16,
        n_heads=2, n_blocks=1, n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2,
        dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    )
    return config


def _real_autoencoder_checkpoint(tmp_path) -> str:
    torch.manual_seed(0)
    autoencoder = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=LATENT_DIM, hidden_dim=16)
    path = tmp_path / "gen5_autoencoder.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp", preprocessing_spec="spec", code_identity="code",
    )
    return str(path), autoencoder


def test_build_model_for_inference_gen5_requires_real_autoencoder_checkpoint_for_non_smoke():
    config = _tiny_gen5a_config()
    config["required_fingerprints"]["expression_autoencoder_checkpoint"] = None
    config["required_fingerprints"]["gen4_conditioner_checkpoint"] = None
    import pytest

    with pytest.raises(ValueError, match="expression_autoencoder_checkpoint"):
        build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False)


def test_build_model_for_inference_gen5_stages_real_autoencoder_and_conditioner(tmp_path):
    """End-to-end through the SAME shared entry point Gen4's own test
    uses: a real Gen5 autoencoder checkpoint (the actual standalone .pt
    format `save_expression_autoencoder_checkpoint` writes -- Integration
    audit finding #8) loads real weights, and a real gen4a-shaped
    conditioner checkpoint stages and freezes onto the flow model's own
    conditioner."""
    autoencoder_path, trained_autoencoder = _real_autoencoder_checkpoint(tmp_path)

    # A real, tiny gen4a conditioner checkpoint (Gen5 reuses Gen4Conditioner
    # unmodified via GEN5_TO_GEN4_ARM).
    conditioner_config = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "configs" / "gen4" / "gen4a_conditioner.yaml").read_text())
    conditioner_config["model"]["params"].update(
        n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
        dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    )
    trained_conditioner, _ = build_model_for_inference(
        conditioner_config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True,
    )
    conditioner_checkpoint_dir = tmp_path / "gen4a_conditioner_checkpoint"
    checkpoint_module.save_checkpoint(trained_conditioner, {"arm": "gen4a"}, GENE_NAMES, conditioner_checkpoint_dir, step=10)

    config = _tiny_gen5a_config()
    config["required_fingerprints"]["expression_autoencoder_checkpoint"] = autoencoder_path
    config["required_fingerprints"]["gen4_conditioner_checkpoint"] = str(conditioner_checkpoint_dir)

    model, info = build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False)

    assert info["kind"] == "latent_flow"
    assert info["autoencoder_info"]["loaded"] is True
    assert info["conditioner_info"]["loaded"] is True
    assert model.autoencoder.latent_dim == LATENT_DIM
    trained_ae_weight = trained_autoencoder.state_dict()["encoder.net.0.weight"]
    model_ae_weight = model.autoencoder.state_dict()["encoder.net.0.weight"]
    assert torch.allclose(trained_ae_weight, model_ae_weight)
    assert all(not p.requires_grad for p in model.autoencoder.parameters())
    assert all(not p.requires_grad for p in model.conditioner.parameters())
