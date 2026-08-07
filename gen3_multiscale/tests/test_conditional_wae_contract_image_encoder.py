"""Tests for the data.image_encoder axis added to
conditional_wae/contract.py::static_audit_conditional_wae_config, and the
two new wae_he_gan_uni2_control/wae_he_gan_uni2 arm identities."""
import pytest

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"


def _base_config(*, arm: str = "wae_he_gan_control", image_encoder: str | None = None) -> dict:
    config = {
        "model": {
            "arm": arm, "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
            },
        },
        "data": {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"},
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    if image_encoder is not None:
        config["data"]["image_encoder"] = image_encoder
    return config


def test_image_encoder_defaults_to_gigapath_and_is_reported():
    report = static_audit_conditional_wae_config(_base_config())
    assert report["image_encoder"] == "gigapath"


def test_image_encoder_gigapath_still_requires_tile_encoder_revision():
    config = _base_config(image_encoder="gigapath")
    del config["data"]["tile_encoder_revision"]
    with pytest.raises(ValueError, match="tile_encoder_revision"):
        static_audit_conditional_wae_config(config)


def test_image_encoder_uni2_requires_uni2_pinned_revision():
    config = _base_config(arm="wae_he_gan_uni2", image_encoder="uni2")
    with pytest.raises(ValueError, match="uni2_pinned_revision"):
        static_audit_conditional_wae_config(config)


def test_image_encoder_uni2_passes_with_uni2_pinned_revision():
    config = _base_config(arm="wae_he_gan_uni2", image_encoder="uni2")
    config["data"]["uni2_pinned_revision"] = _VALID_REVISION
    report = static_audit_conditional_wae_config(config)
    assert report["image_encoder"] == "uni2"


def test_image_encoder_uni2_does_not_require_tile_encoder_revision():
    """A uni2 arm's config never needs GigaPath's pinning field at all."""
    config = _base_config(arm="wae_he_gan_uni2", image_encoder="uni2")
    del config["data"]["tile_encoder_revision"]
    config["data"]["uni2_pinned_revision"] = _VALID_REVISION
    static_audit_conditional_wae_config(config)  # must not raise


def test_image_encoder_rejects_an_unknown_value():
    config = _base_config(image_encoder="resnet50")
    with pytest.raises(ValueError, match="image_encoder"):
        static_audit_conditional_wae_config(config)


def test_wae_he_gan_uni2_control_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_uni2_control")
    static_audit_conditional_wae_config(config)  # must not raise


def test_wae_he_gan_uni2_arm_is_recognized():
    config = _base_config(arm="wae_he_gan_uni2", image_encoder="uni2")
    config["data"]["uni2_pinned_revision"] = _VALID_REVISION
    static_audit_conditional_wae_config(config)  # must not raise


def test_image_encoder_omiclip_requires_omiclip_pinned_revision():
    config = _base_config(image_encoder="omiclip")
    with pytest.raises(ValueError, match="omiclip_pinned_revision"):
        static_audit_conditional_wae_config(config)


def test_image_encoder_omiclip_passes_with_omiclip_pinned_revision():
    config = _base_config(image_encoder="omiclip")
    config["data"]["omiclip_pinned_revision"] = _VALID_REVISION
    report = static_audit_conditional_wae_config(config)
    assert report["image_encoder"] == "omiclip"


def test_image_encoder_omiclip_does_not_require_tile_encoder_revision():
    """An omiclip arm's config never needs GigaPath's/UNI2's own pinning fields."""
    config = _base_config(image_encoder="omiclip")
    del config["data"]["tile_encoder_revision"]
    config["data"]["omiclip_pinned_revision"] = _VALID_REVISION
    static_audit_conditional_wae_config(config)  # must not raise


def test_wae_he_mmd_geneencoder_omiclip_mlp_nofilm_arm_is_recognized():
    config = {
        "model": {
            "arm": "wae_he_mmd_geneencoder_omiclip_mlp_nofilm", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "mmd",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
                "gene_encoder_source": "linear",
            },
        },
        "data": {
            "gen3_manifest_path": "manifest.json", "image_encoder": "omiclip",
            "omiclip_pinned_revision": _VALID_REVISION,
        },
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    report = static_audit_conditional_wae_config(config)
    assert report["image_encoder"] == "omiclip"
    assert report["gene_encoder_source"] == "linear"
