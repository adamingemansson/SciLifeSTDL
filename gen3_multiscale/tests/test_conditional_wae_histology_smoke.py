"""Real-data smoke test: run_conditional_wae_training(..., smoke=True)
reaches one training step and one validation step end-to-end with
model.params.use_histology_context=True -- real synthetic Gen3 data
(_step6_fixtures.build_synthetic_gen3_experiment), a real histology-
feature cache built from those same real samples' real patches/
coordinates, and the real model/dataset/preflight/optimizer wiring."""
import numpy as np
import yaml

from gen3_multiscale.conditional_wae.histology_cache import build_histology_feature_cache
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.tests._step6_fixtures import STEP6_TRAIN_STRATA, build_synthetic_gen3_experiment
from gen3_multiscale.training import train_conditional_wae as train_module


def test_run_conditional_wae_training_smoke_reaches_one_train_and_validation_step_with_histology_context(
    tmp_path, monkeypatch,
):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    histology_cache_dir = tmp_path / "histology_cache"
    all_sample_ids = list(manifest["train_sample_ids"]) + list(manifest["validation_sample_ids"])
    for sample_id in all_sample_ids:
        adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        build_histology_feature_cache(
            histology_cache_dir, sample_id, np.asarray(adata.obs_names), coords, patches, image_source_available,
        )

    checkpoint_dir = tmp_path / "ckpt_histology"
    config = {
        "model": {
            "arm": "wae_he_gan_histology", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 1536, "gex_feature_dim": 8, "hidden_dim": 16,
                "n_heads": 2, "n_blocks": 1, "dense_threshold": 256, "sparse_k": 10,
                "latent_dim": 5, "autoencoder_hidden_dim": 16, "discriminator_hidden_dim": 16,
                "n_inference_samples": 2, "use_histology_context": True,
            },
        },
        "masking": {"strata": STEP6_TRAIN_STRATA},
        "data": {
            "hest_data_dir": str(cfg.data.hest_data_dir), "hest_cache_dir": str(cfg.data.hest_cache_dir),
            "gen3_manifest_path": str(manifest_path),
            "tile_encoder_revision": "d072f48609bec7ec4d2c43889262b3029bb1279f",
            "use_histology_features": True, "gen3_histology_feature_cache_dir": str(histology_cache_dir),
            "gex_feature_dim": 8, "n_training_masks_per_sample": 1, "n_validation_masks": 1,
        },
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
        "training": {
            "checkpoint_dir": str(checkpoint_dir), "device": "cpu", "seed": 0, "lr": 1.0e-3,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    summary = train_module.run_conditional_wae_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["smoke"] is True
    assert summary["arm"] == "wae_he_gan_histology"
    assert summary["final_step"] == 1
    assert summary["masks_seen"] == 1
