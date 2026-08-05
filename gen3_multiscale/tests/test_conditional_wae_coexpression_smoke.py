"""Real-data smoke test: run_conditional_wae_training(..., smoke=True)
reaches one training step and one validation step end-to-end with
model.params.use_gene_coexpression_refinement=True -- real synthetic
Gen3 data (_step6_fixtures.build_synthetic_gen3_experiment), a real
gene-coexpression basis fit from those same real training samples' real
expression, and the real model/dataset/preflight/optimizer wiring."""
import yaml

from gen3_multiscale.conditional_wae.coexpression import (
    fit_conditional_wae_gene_coexpression_basis, save_conditional_wae_gene_coexpression_basis,
)
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.tests._step6_fixtures import STEP6_TRAIN_STRATA, build_synthetic_gen3_experiment
from gen3_multiscale.training import train_conditional_wae as train_module


def test_run_conditional_wae_training_smoke_reaches_one_train_and_validation_step_with_coexpression(
    tmp_path, monkeypatch,
):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    gene_names = list(manifest["gene_panel"])
    train_sample_ids = list(manifest["train_sample_ids"])
    expression_by_sample = {}
    for sample_id in train_sample_ids:
        adata, _patches, _availability = example_builder.load_sample_for_examples(manifest, sample_id)
        expression_by_sample[sample_id] = adata.X
    basis, metadata = fit_conditional_wae_gene_coexpression_basis(
        expression_by_sample, train_sample_ids, gene_names, rank=3, seed=0,
    )
    basis_path = tmp_path / "gene_coexpression_basis.pt"
    save_conditional_wae_gene_coexpression_basis(basis, metadata, basis_path)

    checkpoint_dir = tmp_path / "ckpt_coexpression"
    config = {
        "model": {
            "arm": "wae_he_gan_coexpression", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 1536, "gex_feature_dim": 8, "hidden_dim": 16,
                "n_heads": 2, "n_blocks": 1, "dense_threshold": 256, "sparse_k": 10,
                "latent_dim": 5, "autoencoder_hidden_dim": 16, "discriminator_hidden_dim": 16,
                "n_inference_samples": 2, "use_gene_coexpression_refinement": True,
            },
        },
        "masking": {"strata": STEP6_TRAIN_STRATA},
        "data": {
            "hest_data_dir": str(cfg.data.hest_data_dir), "hest_cache_dir": str(cfg.data.hest_cache_dir),
            "gen3_manifest_path": str(manifest_path),
            "tile_encoder_revision": "d072f48609bec7ec4d2c43889262b3029bb1279f",
            "gene_coexpression_basis_path": str(basis_path),
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
    assert summary["arm"] == "wae_he_gan_coexpression"
    assert summary["final_step"] == 1
    assert summary["masks_seen"] == 1
