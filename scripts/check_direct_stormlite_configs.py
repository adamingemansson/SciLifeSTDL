#!/usr/bin/env python3
"""Fail closed if the direct StormLite/MoME wave stops being matched."""
from pathlib import Path

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


PATHS = [
    Path("configs/recovery_suite/137_direct_transformer_neither.yaml"),
    Path("configs/recovery_suite/138_direct_transformer_gex_novae.yaml"),
    Path("configs/recovery_suite/139_direct_transformer_full_novae.yaml"),
    Path("configs/recovery_suite/140_direct_mome_full_novae.yaml"),
]
REFERENCE = Path("configs/recovery_suite/131_simple_local_full_novae.yaml")


def main() -> None:
    reference = OmegaConf.load(REFERENCE)
    configs = [OmegaConf.load(path) for path in PATHS]
    ref_groups = _validated_sample_groups(reference)
    ref_masking = OmegaConf.to_container(reference.masking, resolve=True)
    matched_eval_keys = (
        "mask_bank_dir", "n_validation_masks", "n_test_masks",
        "validation_seed", "test_seed", "training_mask_bank_path",
    )
    ref_eval = {key: reference.evaluation[key] for key in matched_eval_keys}

    for path, cfg in zip(PATHS, configs):
        _validate_task_contract(cfg)
        assert _validated_sample_groups(cfg) == ref_groups, path
        assert OmegaConf.to_container(cfg.masking, resolve=True) == ref_masking, path
        assert {key: cfg.evaluation[key] for key in matched_eval_keys} == ref_eval, path
        assert cfg.model.name == "direct_context_regressor", path
        params = cfg.model.params
        assert params.context_encoder_type == "storm_lite", path
        assert params.storm_lite_fusion_mode in {"sum", "mome"}, path
        assert int(params.storm_lite_n_layers) == 2, path
        assert int(params.storm_lite_n_heads) == 4, path
        assert int(params.storm_lite_knn_k) == 32, path
        assert params.storm_lite_bias_type == "frame_averaging", path
        assert params.storm_lite_use_absolute_coords is False, path
        assert int(cfg.training.epochs) == 5000, path
        assert int(cfg.training.unique_mask_count) == 256, path
        assert int(cfg.training.mask_seed) == 730000, path
        assert cfg.training.augment_coords is False, path
        assert cfg.validation.require_anchor_improvement is False, path
        assert cfg.data.holdout_unit == "sample", path
        assert cfg.data.task_contract == "missing_tissue", path
        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "harmonic_k" not in serialized and "harmonic_ridge" not in serialized, path
        assert "highly_variable" not in serialized and "hvg" not in serialized, path

    neither, gex, full, mome = configs
    assert neither.data.modality_ablation == "neither"
    assert neither.data.novae_mode == "disabled"
    assert neither.training.image_mode == "all_zero"
    assert neither.training.context_gex_mode == "zero"
    assert neither.evaluation.primary_image_mode == "all_zero"
    assert gex.data.modality_ablation == "gex_only"
    assert gex.data.novae_mode == "context_only"
    assert gex.training.image_mode == "all_zero"
    assert gex.training.context_gex_mode == "full"
    assert full.data.modality_ablation == "both"
    assert full.data.novae_mode == "context_only"
    assert full.training.image_mode == "target_zero"
    assert full.training.context_gex_mode == "full"
    assert full.evaluation.primary_image_mode == "target_zero"
    assert mome.data.modality_ablation == "both"
    assert mome.data.novae_mode == "context_only"
    assert mome.model.params.storm_lite_fusion_mode == "mome"
    assert mome.model.params.storm_lite_qk_norm is True
    assert mome.training.image_mode == "target_zero"
    assert mome.training.context_gex_mode == "full"
    print("Direct StormLite/MoME config audit: PASS")


if __name__ == "__main__":
    main()
