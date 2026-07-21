#!/usr/bin/env python3
"""Fail-closed static audit for the four simple-local diagnostic configs."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


CONFIG_PATHS = [
    Path("configs/recovery_suite/129_simple_local_harmonic_k32.yaml"),
    Path("configs/recovery_suite/130_simple_local_gex_novae.yaml"),
    Path("configs/recovery_suite/131_simple_local_full_novae.yaml"),
    Path("configs/recovery_suite/132_simple_local_full_mlp.yaml"),
]


def main() -> None:
    configs = [OmegaConf.load(path) for path in CONFIG_PATHS]
    for path, cfg in zip(CONFIG_PATHS, configs):
        _validate_task_contract(cfg)
        train, validation, test = _validated_sample_groups(cfg)
        assert set(train).isdisjoint(validation)
        assert set(train).isdisjoint(test)
        assert set(validation).isdisjoint(test)
        assert cfg.data.expression_transform == "normalize_log1p"
        assert int(cfg.data.min_genes) == 200 and int(cfg.data.min_cells) == 3
        assert cfg.masking.context_selection == "nearest_query"
        assert cfg.masking.params.radius_unit == "spot_spacing"
        assert int(cfg.masking.params.n_patches) == 1
        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "highly_variable" not in serialized and "hvg" not in serialized

    reference = configs[0]
    reference_groups = _validated_sample_groups(reference)
    reference_masking = OmegaConf.to_container(reference.masking, resolve=True)
    reference_eval = {
        key: reference.evaluation[key]
        for key in ("mask_bank_dir", "n_validation_masks", "n_test_masks",
                    "validation_seed", "test_seed")
    }
    for cfg in configs[1:]:
        assert _validated_sample_groups(cfg) == reference_groups
        assert OmegaConf.to_container(cfg.masking, resolve=True) == reference_masking
        assert {
            key: cfg.evaluation[key] for key in reference_eval
        } == reference_eval

    learned = configs[1:]
    schedules = {
        (int(cfg.training.epochs), int(cfg.training.unique_mask_count),
         int(cfg.training.mask_seed), str(cfg.evaluation.training_mask_bank_path))
        for cfg in learned
    }
    assert schedules == {(5000, 256, 730000,
                          "results/mask_banks/training/recovery_suite/simple_local_5k_256_seed730000.json")}
    for cfg in learned:
        assert cfg.model.name == "harmonic_residual"
        assert cfg.model.params.context_encoder_type == "storm_lite"
        assert cfg.model.params.storm_lite_fusion_mode == "local_pool"
        assert int(cfg.model.params.storm_lite_local_k) == 32
        assert cfg.model.params.storm_lite_bias_type == "none"
        assert int(cfg.model.params.harmonic_k) == 32
        assert cfg.training.augment_coords is False

    assert configs[1].data.novae_mode == "context_only"
    assert configs[2].data.novae_mode == "context_only"
    assert configs[1].model.params.gene_encoder_type == "both"
    assert configs[2].model.params.gene_encoder_type == "both"
    assert configs[3].data.novae_mode == "disabled"
    assert configs[3].model.params.gene_encoder_type == "mlp"
    print("Simple-local config audit: PASS")


if __name__ == "__main__":
    main()
