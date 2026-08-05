"""Tests for the use_histology_context axis added to conditional_wae/
contract.py::static_audit_conditional_wae_config, and the two new
wae_he_gan_histology_control/wae_he_gan_histology arm identities."""
import pytest

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config


def _base_config(*, arm: str = "wae_he_gan_control", use_histology_context: bool = False) -> dict:
    config = {
        "model": {
            "arm": arm, "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
                "use_histology_context": use_histology_context,
            },
        },
        "data": {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"},
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    return config


def test_histology_context_defaults_to_disabled_and_is_reported():
    report = static_audit_conditional_wae_config(_base_config())
    assert report["use_histology_context"] is False


def test_histology_context_requires_use_histology_features_when_enabled():
    config = _base_config(arm="wae_he_gan_histology", use_histology_context=True)
    with pytest.raises(ValueError, match="use_histology_features"):
        static_audit_conditional_wae_config(config)


def test_histology_context_passes_when_use_histology_features_is_set():
    config = _base_config(arm="wae_he_gan_histology", use_histology_context=True)
    config["data"]["use_histology_features"] = True
    report = static_audit_conditional_wae_config(config)
    assert report["use_histology_context"] is True


def test_wae_he_gan_histology_control_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_histology_control")
    static_audit_conditional_wae_config(config)  # must not raise


def test_wae_he_gan_histology_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_histology", use_histology_context=True)
    config["data"]["use_histology_features"] = True
    static_audit_conditional_wae_config(config)  # must not raise


def test_histology_context_and_coexpression_are_independent_axes():
    """Enabling one ablation's flag must never require or imply the
    other's -- they are declared as three separate arms specifically so
    each component's effect can be isolated."""
    config = _base_config(arm="wae_he_gan_histology", use_histology_context=True)
    config["data"]["use_histology_features"] = True
    report = static_audit_conditional_wae_config(config)
    assert report["use_gene_coexpression_refinement"] is False
