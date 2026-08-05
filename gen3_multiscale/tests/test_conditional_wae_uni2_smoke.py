"""Real-data smoke test: run_conditional_wae_training(..., smoke=True)
reaches one training step and one validation step end-to-end with
data.image_encoder="uni2" -- real synthetic Gen3 data
(_step6_fixtures.build_synthetic_gen3_experiment), a real UNI2
spot-feature cache built from those same real patches, and the real
model/dataset/preflight/optimizer wiring, not a hand-rolled shortcut."""
import numpy as np
import yaml

from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache
from gen3_multiscale.tests._step6_fixtures import STEP6_TRAIN_STRATA, build_synthetic_gen3_experiment
from gen3_multiscale.training import train_conditional_wae as train_module

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
_VALID_SHA256 = "a" * 64
_EXPECTED_SPEC = "uni2_tile_v2:vit_giant_patch14_224:resize224_bicubic_antialias:imagenet_norm"


class _RealShapedStubUNI2Encoder:
    def __init__(self):
        self.identity = EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256=_VALID_SHA256, pinned_revision=_VALID_REVISION,
            package_version="timm-1.0.0", preprocessing_spec=_EXPECTED_SPEC, output_dim=1536,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, 1536)).astype(np.float32)


def _uni2_smoke_config(cfg, manifest_path: str, uni2_cache_root, checkpoint_dir) -> dict:
    return {
        "model": {
            "arm": "wae_he_gan_uni2", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 1536, "gex_feature_dim": 8, "hidden_dim": 16,
                "n_heads": 2, "n_blocks": 1, "dense_threshold": 256, "sparse_k": 10,
                "latent_dim": 5, "autoencoder_hidden_dim": 16, "discriminator_hidden_dim": 16,
                "n_inference_samples": 2,
            },
        },
        "masking": {"strata": STEP6_TRAIN_STRATA},
        "data": {
            "hest_data_dir": str(cfg.data.hest_data_dir), "hest_cache_dir": str(cfg.data.hest_cache_dir),
            "gen3_manifest_path": str(manifest_path),
            "image_encoder": "uni2", "uni2_pinned_revision": _VALID_REVISION,
            "gen3_uni2_spot_feature_cache_dir": str(uni2_cache_root),
            "gex_feature_dim": 8, "n_training_masks_per_sample": 1, "n_validation_masks": 1,
        },
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
        "training": {
            "checkpoint_dir": str(checkpoint_dir), "device": "cpu", "seed": 0, "lr": 1.0e-3,
        },
    }


def test_run_conditional_wae_training_smoke_reaches_one_train_and_validation_step_with_uni2(
    tmp_path, monkeypatch,
):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    uni2_cache_root = tmp_path / "uni2_cache_root"
    encoder = _RealShapedStubUNI2Encoder()
    for sample_id in manifest["samples"]:
        adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        build_uni2_spot_feature_cache(
            uni2_cache_root, sample_id, np.asarray(adata.obs_names), patches, image_source_available, encoder,
        )

    checkpoint_dir = tmp_path / "ckpt_uni2"
    config = _uni2_smoke_config(cfg, manifest_path, uni2_cache_root, checkpoint_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    summary = train_module.run_conditional_wae_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["smoke"] is True
    assert summary["arm"] == "wae_he_gan_uni2"
    assert summary["final_step"] == 1
    assert summary["masks_seen"] == 1
    # smoke mode never writes to the checkpoint dir at all (unlike train.py's
    # own smoke mode, which still writes preflight_report.json/run_manifest.json) --
    # conditional_wae's checkpoint/manifest writes are gated behind `if not smoke`.
    assert not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir())
