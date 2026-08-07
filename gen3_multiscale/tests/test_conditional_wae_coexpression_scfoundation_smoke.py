"""Real-data smoke test: run_conditional_wae_training(..., smoke=True)
reaches one training step and one validation step end-to-end with
arm="wae_he_gan_coexpression_scfoundation" -- real synthetic Gen3 data
(_step6_fixtures.build_synthetic_gen3_experiment), a real
GeneCoexpressionRefinement basis built the SAME way the scFoundation CLI
does (extract_scfoundation_gene_embedding_table + fit_conditional_wae_
gene_coexpression_basis_from_scfoundation), just from a faked encoder
(no real scFoundation checkpoint needed for this test -- the encoder's
public contract, not its checkpoint loading, is what _build_model and
the fit function actually consume), and the real model/dataset/
preflight/optimizer wiring. Proves _build_model loads a scFoundation-
sourced basis file exactly like a from-scratch-sourced one, with zero
source-specific branching."""
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from gen3_multiscale.conditional_wae.coexpression import save_conditional_wae_gene_coexpression_basis
from gen3_multiscale.conditional_wae.coexpression_scfoundation import (
    _REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    extract_scfoundation_gene_embedding_table,
    fit_conditional_wae_gene_coexpression_basis_from_scfoundation,
)
from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.tests._step6_fixtures import STEP6_TRAIN_STRATA, build_synthetic_gen3_experiment
from gen3_multiscale.training import train_conditional_wae as train_module


def _fake_scfoundation_encoder(gene_names, embedding_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    pos_emb_weight = torch.as_tensor(
        rng.normal(size=(len(gene_names) + 2, embedding_dim)).astype(np.float32),
    )
    model = SimpleNamespace(pos_emb=torch.nn.Embedding.from_pretrained(pos_emb_weight, freeze=True))
    identity = EncoderIdentity(
        encoder_name="scfoundation", checkpoint_sha256="a" * 64, pinned_revision="b" * 40,
        package_version="git:c" * 8, preprocessing_spec="stub_scfoundation_v1", output_dim=3072,
    )
    return SimpleNamespace(gene_names=tuple(gene_names), scfoundation_vocab=list(gene_names), model=model, identity=identity)


def test_run_conditional_wae_training_smoke_reaches_one_train_and_validation_step_with_scfoundation_coexpression(
    tmp_path, monkeypatch,
):
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)

    gene_names = list(manifest["gene_panel"])
    encoder = _fake_scfoundation_encoder(gene_names, embedding_dim=8)
    embedding_table, report = extract_scfoundation_gene_embedding_table(encoder)
    basis, metadata = fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
        embedding_table, gene_names, encoder.identity, report, rank=3, seed=0,
    )
    basis_path = tmp_path / "gene_coexpression_basis_scfoundation.pt"
    save_conditional_wae_gene_coexpression_basis(
        basis, metadata, basis_path, required_fields=_REQUIRED_METADATA_FIELDS_SCFOUNDATION,
    )

    checkpoint_dir = tmp_path / "ckpt_coexpression_scfoundation"
    config = {
        "model": {
            "arm": "wae_he_gan_coexpression_scfoundation", "kind": "conditional_wae",
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
    assert summary["arm"] == "wae_he_gan_coexpression_scfoundation"
    assert summary["final_step"] == 1
    assert summary["masks_seen"] == 1
