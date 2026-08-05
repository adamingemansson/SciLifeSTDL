"""Tests for the use_gene_coexpression_refinement axis added to
conditional_wae/contract.py::static_audit_conditional_wae_config, and the
two new wae_he_gan_coexpression_control/wae_he_gan_coexpression arm
identities."""
import pytest

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config


def _base_config(*, arm: str = "wae_he_gan_control", use_gene_coexpression_refinement: bool = False) -> dict:
    config = {
        "model": {
            "arm": arm, "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
                "use_gene_coexpression_refinement": use_gene_coexpression_refinement,
            },
        },
        "data": {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"},
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    return config


def test_coexpression_defaults_to_disabled_and_is_reported():
    report = static_audit_conditional_wae_config(_base_config())
    assert report["use_gene_coexpression_refinement"] is False


def test_coexpression_requires_a_basis_path_when_enabled():
    config = _base_config(arm="wae_he_gan_coexpression", use_gene_coexpression_refinement=True)
    with pytest.raises(ValueError, match="gene_coexpression_basis_path"):
        static_audit_conditional_wae_config(config)


def test_coexpression_passes_with_a_basis_path():
    config = _base_config(arm="wae_he_gan_coexpression", use_gene_coexpression_refinement=True)
    config["data"]["gene_coexpression_basis_path"] = "/path/to/basis.pt"
    report = static_audit_conditional_wae_config(config)
    assert report["use_gene_coexpression_refinement"] is True


def test_wae_he_gan_coexpression_control_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_coexpression_control")
    static_audit_conditional_wae_config(config)  # must not raise


def test_wae_he_gan_coexpression_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_coexpression", use_gene_coexpression_refinement=True)
    config["data"]["gene_coexpression_basis_path"] = "/path/to/basis.pt"
    static_audit_conditional_wae_config(config)  # must not raise
