"""Integration-10: construct every one of Gen5's arms through the real,
shared `training/train.py::build_model_for_inference` entry point --
mirrors tests/test_gen4_all_arms_construction.py's discipline, applied
to Gen5's own GEN5_TO_GEN4_ARM-mapped arm needs."""
from __future__ import annotations

import pathlib

import pytest
import torch
import yaml

from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder, save_expression_autoencoder_checkpoint
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train import build_model_for_inference

_CONFIG_DIR_GEN5 = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gen5"
_CONFIG_DIR_GEN4 = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gen4"

N_GENES, LATENT_DIM = 5, 4
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]

_TINY_PARAMS = dict(
    n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, latent_dim=LATENT_DIM, hidden_dim=16,
    n_heads=2, n_blocks=1, n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2,
    dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    gex_context_embedding_dim=6,
)

_TINY_CONDITIONER_PARAMS = dict(
    n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
    dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    gex_context_embedding_dim=6,
)


def _tiny_config(arm: str):
    config = yaml.safe_load((_CONFIG_DIR_GEN5 / f"{arm}.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    return config


def _staged_fingerprints(config: dict, tmp_path) -> dict:
    torch.manual_seed(0)
    autoencoder = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=LATENT_DIM, hidden_dim=16)
    ae_path = tmp_path / f"{config['model']['arm']}_autoencoder.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, ae_path, dataset_manifest_fingerprint="fp", preprocessing_spec="spec", code_identity="code",
    )
    from gen3_multiscale.gen5.model_factory import GEN5_TO_GEN4_ARM

    gen4_arm = GEN5_TO_GEN4_ARM[config["model"]["arm"]]
    gen4_config = yaml.safe_load((_CONFIG_DIR_GEN4 / f"{gen4_arm}_conditioner.yaml").read_text())
    gen4_config["model"]["params"].update(_TINY_CONDITIONER_PARAMS)
    trained_conditioner, _ = build_model_for_inference(gen4_config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True)
    conditioner_dir = tmp_path / f"{config['model']['arm']}_conditioner"
    checkpoint_module.save_checkpoint(trained_conditioner, {"arm": "gen4a"}, GENE_NAMES, conditioner_dir, step=1)
    return {"expression_autoencoder_checkpoint": str(ae_path), "gen4_conditioner_checkpoint": str(conditioner_dir)}


def test_gen5a_and_gen5c_construct_with_no_real_external_package(tmp_path):
    """gen5a/gen5c mirror gen4a/gen4c's own dependency-free needs
    (uni2_pool is self-contained; scFoundation is only a dimension) --
    the only real dependency either has is the staged AE/conditioner,
    both of which are real, constructible artifacts in this sandbox."""
    from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel

    for arm in ("gen5a", "gen5c"):
        config = _tiny_config(arm)
        config["required_fingerprints"].update(_staged_fingerprints(config, tmp_path))
        model, info = build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False)
        assert isinstance(model, Gen5LatentFlowModel)
        assert info["kind"] == "latent_flow"
        assert info["autoencoder_info"]["loaded"] is True
        assert info["conditioner_info"]["loaded"] is True


@pytest.mark.parametrize("arm,missing_encoder", [("gen5b", "slide_encoder"), ("gen5d", "stpath_encoder"), ("gen5e", "stpath_encoder")])
def test_gen5_arms_needing_real_external_weights_fail_closed_on_the_right_guard(arm, missing_encoder, tmp_path):
    config = _tiny_config(arm)
    with pytest.raises(ValueError, match=missing_encoder):
        config["required_fingerprints"].update(_staged_fingerprints(config, tmp_path))
        build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=False)
