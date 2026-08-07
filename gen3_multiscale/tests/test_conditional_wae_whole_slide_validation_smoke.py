"""Real-data test: run_conditional_wae_training(..., smoke=False) with
evaluation.whole_slide_validation.enabled=True AND TensorBoard enabled
together -- the exact combination requested for the live UNI2/
coexpression/histology ablation launch (full-slide, every-spot
diagnostic validation, not just the masked query subset), including the
per-gene whole-slide spatial maps (add_whole_slide_spatial_maps's new
gene_indices parameter). NOT run with smoke=True: the trainer
deliberately disables the TensorBoard logger (and therefore the whole-
slide reference-projection path that depends on it) during smoke mode,
so this uses a real, tiny (total_steps=1) non-smoke run instead -- the
only way to actually exercise this code path. Real synthetic Gen3 data
(_step6_fixtures.build_synthetic_gen3_experiment), real model/dataset/
preflight/optimizer wiring, and a real (not mocked) TensorBoard
SummaryWriter + reference GEX projection built from scratch."""
import yaml

from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.tests._step6_fixtures import STEP6_TRAIN_STRATA, build_synthetic_gen3_experiment
from gen3_multiscale.training import train_conditional_wae as train_module


def test_run_conditional_wae_training_reaches_whole_slide_validation_with_tensorboard(
    tmp_path, monkeypatch,
):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    checkpoint_dir = tmp_path / "ckpt_whole_slide"
    tensorboard_dir = tmp_path / "tensorboard"
    reference_projection_path = tmp_path / "shared_reference_gex_projection"
    config = {
        "model": {
            "arm": "wae_he_gan_control", "kind": "conditional_wae",
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
            "tile_encoder_revision": "d072f48609bec7ec4d2c43889262b3029bb1279f",
            "gex_feature_dim": 8, "n_training_masks_per_sample": 1, "n_validation_masks": 1,
        },
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
        "training": {
            "checkpoint_dir": str(checkpoint_dir), "device": "cpu", "seed": 0, "lr": 1.0e-3,
            "total_steps": 1, "eval_every_n_steps": 1, "checkpoint_every_n_steps": 1000,
        },
        "evaluation": {
            "tensorboard": {
                "enabled": True, "log_dir": str(tensorboard_dir), "snapshot_every_n_evals": 1,
                "spatial_gene_count": 2,
            },
            "whole_slide_validation": {
                "enabled": True, "every_n_evals": 1, "max_slides": 1, "chunk_size": 64,
                "reference_projection_path": str(reference_projection_path),
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    summary = train_module.run_conditional_wae_training(str(config_path), smoke=False)
    assert summary["ok"] is True
    assert summary["smoke"] is False
    assert summary["final_step"] == 1

    assert reference_projection_path.with_suffix(".json").is_file()
    assert reference_projection_path.with_suffix(".npz").is_file()
    event_files = list(tensorboard_dir.glob("events.out.tfevents.*"))
    assert event_files, "expected a real TensorBoard event file to be written"
