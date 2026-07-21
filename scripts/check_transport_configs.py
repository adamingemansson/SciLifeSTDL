#!/usr/bin/env python3
"""Fail-closed audit for the identity-preserving transport wave."""
from pathlib import Path

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


PATHS = [
    Path("configs/recovery_suite/141_transport_uniform_k32.yaml"),
    Path("configs/recovery_suite/142_transport_geometry_k32.yaml"),
    Path("configs/recovery_suite/143_transport_gex_novae_k32.yaml"),
    Path("configs/recovery_suite/144_transport_full_mome_k32.yaml"),
]
REFERENCE = Path("configs/recovery_suite/131_simple_local_full_novae.yaml")


def main() -> None:
    reference = OmegaConf.load(REFERENCE)
    configs = [OmegaConf.load(path) for path in PATHS]
    ref_groups = _validated_sample_groups(reference)
    ref_masking = OmegaConf.to_container(reference.masking, resolve=True)
    eval_keys = (
        "mask_bank_dir", "n_validation_masks", "n_test_masks",
        "validation_seed", "test_seed", "training_mask_bank_path",
    )
    ref_eval = {key: reference.evaluation[key] for key in eval_keys}

    for path, cfg in zip(PATHS, configs):
        _validate_task_contract(cfg)
        assert _validated_sample_groups(cfg) == ref_groups, path
        assert OmegaConf.to_container(cfg.masking, resolve=True) == ref_masking, path
        assert {key: cfg.evaluation[key] for key in eval_keys} == ref_eval, path
        assert cfg.model.name == "context_transport_regressor", path
        assert int(cfg.model.params.transport_k) == 32, path
        assert float(cfg.model.params.target_scale_floor) == 0.05, path
        assert int(cfg.training.epochs) == 5000, path
        assert int(cfg.training.unique_mask_count) == 256, path
        assert int(cfg.training.mask_seed) == 730000, path
        assert cfg.training.augment_coords is False, path
        assert cfg.training.context_gex_mode == "full", path
        assert cfg.evaluation.context_gex_mode == "full", path
        assert cfg.data.holdout_unit == "sample", path
        assert cfg.data.task_contract == "missing_tissue", path
        assert cfg.validation.require_anchor_improvement is False, path
        assert float(cfg.validation.anchor_min_delta) == 0.001, path
        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "harmonic" not in serialized, path
        assert "highly_variable" not in serialized and "hvg" not in serialized, path

    uniform, geometry, gex, full = configs
    assert uniform.model.params.conditioning_mode == "uniform"
    assert uniform.data.modality_ablation == "gex_only"
    assert uniform.data.novae_mode == "disabled"
    assert uniform.training.image_mode == "all_zero"
    assert uniform.evaluation.primary_image_mode == "all_zero"
    assert geometry.model.params.conditioning_mode == "geometry"
    assert geometry.data.modality_ablation == "gex_only"
    assert geometry.data.novae_mode == "disabled"
    assert geometry.training.image_mode == "all_zero"
    assert geometry.evaluation.primary_image_mode == "all_zero"
    assert gex.model.params.conditioning_mode == "storm_lite"
    assert gex.model.params.gene_encoder_type == "both"
    assert gex.model.params.storm_lite_fusion_mode == "mome"
    assert int(gex.model.params.storm_lite_n_layers) == 2
    assert int(gex.model.params.storm_lite_n_heads) == 4
    assert int(gex.model.params.storm_lite_knn_k) == 32
    assert gex.model.params.storm_lite_qk_norm is True
    assert gex.model.params.storm_lite_use_absolute_coords is False
    assert gex.data.modality_ablation == "gex_only"
    assert gex.data.novae_mode == "context_only"
    assert gex.training.image_mode == "all_zero"
    assert gex.evaluation.primary_image_mode == "all_zero"
    assert full.model.params.conditioning_mode == "storm_lite"
    assert full.model.params.gene_encoder_type == "both"
    assert full.model.params.storm_lite_fusion_mode == "mome"
    assert int(full.model.params.storm_lite_n_layers) == 2
    assert int(full.model.params.storm_lite_n_heads) == 4
    assert int(full.model.params.storm_lite_knn_k) == 32
    assert full.model.params.storm_lite_qk_norm is True
    assert full.model.params.storm_lite_use_absolute_coords is False
    assert full.data.modality_ablation == "both"
    assert full.data.novae_mode == "context_only"
    assert full.training.image_mode == "target_zero"
    assert full.evaluation.primary_image_mode == "target_zero"
    print("Gene-preserving transport config audit: PASS")


if __name__ == "__main__":
    main()
