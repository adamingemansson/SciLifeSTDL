#!/usr/bin/env python3
"""Fail-closed static audit for the matched local Transformer/MoME wave."""
from pathlib import Path

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


PATHS = [
    Path("configs/recovery_suite/133_local_transformer_neither.yaml"),
    Path("configs/recovery_suite/134_local_transformer_gex_novae.yaml"),
    Path("configs/recovery_suite/135_local_transformer_full_novae.yaml"),
    Path("configs/recovery_suite/136_local_mome_full_novae.yaml"),
]
REFERENCE = Path("configs/recovery_suite/131_simple_local_full_novae.yaml")


def main() -> None:
    reference = OmegaConf.load(REFERENCE)
    configs = [OmegaConf.load(path) for path in PATHS]
    ref_groups = _validated_sample_groups(reference)
    ref_masking = OmegaConf.to_container(reference.masking, resolve=True)
    ref_eval = {
        key: reference.evaluation[key]
        for key in ("mask_bank_dir", "n_validation_masks", "n_test_masks",
                    "validation_seed", "test_seed", "training_mask_bank_path")
    }
    for path, cfg in zip(PATHS, configs):
        _validate_task_contract(cfg)
        assert _validated_sample_groups(cfg) == ref_groups, path
        assert OmegaConf.to_container(cfg.masking, resolve=True) == ref_masking, path
        assert {key: cfg.evaluation[key] for key in ref_eval} == ref_eval, path
        assert cfg.model.name == "harmonic_residual", path
        params = cfg.model.params
        assert params.context_encoder_type == "storm_lite", path
        assert params.storm_lite_fusion_mode in {"sum", "mome"}, path
        assert int(params.storm_lite_n_layers) == 2, path
        assert int(params.storm_lite_n_heads) == 4, path
        assert int(params.storm_lite_knn_k) == 32, path
        assert params.storm_lite_bias_type == "frame_averaging", path
        assert params.storm_lite_use_absolute_coords is False, path
        assert int(params.harmonic_k) == 32, path
        assert int(cfg.training.epochs) == 5000, path
        assert int(cfg.training.unique_mask_count) == 256, path
        assert int(cfg.training.mask_seed) == 730000, path
        assert cfg.training.augment_coords is False, path
        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "highly_variable" not in serialized and "hvg" not in serialized, path

    assert configs[0].data.modality_ablation == "neither"
    assert configs[0].data.novae_mode == "disabled"
    assert configs[1].data.modality_ablation == "gex_only"
    assert configs[1].data.novae_mode == "context_only"
    assert configs[2].data.modality_ablation == "both"
    assert configs[2].data.novae_mode == "context_only"
    assert configs[3].data.modality_ablation == "both"
    assert configs[3].data.novae_mode == "context_only"
    assert configs[3].model.params.storm_lite_fusion_mode == "mome"
    assert configs[3].model.params.storm_lite_qk_norm is True
    print("Local Transformer/MoME config audit: PASS")


if __name__ == "__main__":
    main()
