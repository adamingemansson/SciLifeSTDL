#!/usr/bin/env python3
"""Fail-closed static audit for the matched hierarchical experiment suite."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


CONFIGS = (
    "configs/recovery_suite/152_hierarchical_slide_full_20k.yaml",
    "configs/recovery_suite/153_hierarchical_no_slide_20k.yaml",
    "configs/recovery_suite/154_hierarchical_no_novae_20k.yaml",
    "configs/recovery_suite/155_hierarchical_he_only_20k.yaml",
    "configs/recovery_suite/157_hierarchical_global_he_only_20k.yaml",
    "configs/recovery_suite/158_hierarchical_local_he_only_20k.yaml",
    "configs/recovery_suite/159_hierarchical_raw_gex_only_20k.yaml",
    "configs/recovery_suite/160_hierarchical_gex_novae_only_20k.yaml",
    "configs/recovery_suite/156_hierarchical_harmonic_control.yaml",
)
EXPECTED_SPLIT = (
    ["INT1", "INT2", "INT3", "INT4", "INT5", "INT6"],
    ["INT7"],
    ["INT8"],
)
EXPECTED_PANELS = [50, 100, 250]
EXPECTED_FIXED_PANEL = "resources/evaluation/hest_bench/CCRCC_var_50genes.json"


def main() -> None:
    configs = []
    for path_string in CONFIGS:
        path = Path(path_string)
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg = OmegaConf.load(path)
        _validate_task_contract(cfg)
        assert _validated_sample_groups(cfg) == EXPECTED_SPLIT, path
        assert cfg.data.task_contract == "missing_tissue", path
        assert cfg.data.holdout_unit == "sample", path
        assert bool(cfg.data.strict_broken_region), path
        assert bool(cfg.data.require_image_coverage), path
        assert bool(cfg.data.use_images), path
        assert list(cfg.evaluation.train_variance_gene_panel_sizes) == EXPECTED_PANELS, path
        assert (
            cfg.evaluation.fixed_gene_panel_paths.stpath_hest_bench_ccrcc_50
            == EXPECTED_FIXED_PANEL
        ), path
        assert Path(EXPECTED_FIXED_PANEL).is_file(), EXPECTED_FIXED_PANEL
        assert list(cfg.masking.params.radius_range) == [3.0, 6.0], path
        assert int(cfg.masking.max_context_points) == 3000, path
        assert int(cfg.evaluation.n_test_masks) == 16, path
        configs.append((path, cfg))

    learned = configs[:-1]
    harmonic_path, harmonic = configs[-1]
    assert len({cfg.experiment_name for _, cfg in configs}) == len(configs)
    training_bank_paths = {
        str(cfg.evaluation.training_mask_bank_path) for _, cfg in configs
    }
    assert len(training_bank_paths) == len(configs)
    for path, cfg in learned:
        assert cfg.model.name == "hierarchical_missing_tissue_regressor", path
        assert int(cfg.training.epochs) == 20000, path
        assert int(cfg.training.unique_mask_count) == 256, path
        assert int(cfg.model.params.n_heads) == 4, path
        assert int(cfg.model.params.local_k) == 64, path
        # Gene panels must remain evaluation-only. A target-gene subset here
        # would silently turn this into HVG-only training.
        assert "target_gene_names" not in cfg.model.params, path
        assert "n_genes" not in cfg.model.params, path

    expected_modalities = {
        "hierarchical_slide_full_seed10": (True, True, True),
        "hierarchical_no_slide_seed10": (False, True, True),
        "hierarchical_no_novae_seed10": (True, True, False),
        "hierarchical_he_only_seed10": (True, True, False),
        "hierarchical_global_he_only_seed10": (True, False, False),
        "hierarchical_local_he_only_seed10": (False, True, False),
        "hierarchical_raw_gex_only_seed10": (False, False, False),
        "hierarchical_gex_novae_only_seed10": (False, False, True),
    }
    for path, cfg in learned:
        actual = (
            bool(cfg.model.params.use_slide_context),
            bool(cfg.model.params.use_local_images),
            bool(cfg.model.params.use_novae),
        )
        assert actual == expected_modalities[str(cfg.experiment_name)], path

    assert harmonic.model.name == "spatial_baseline", harmonic_path
    assert harmonic.model.params.mode == "harmonic", harmonic_path
    assert int(harmonic.model.params.k) == 128, harmonic_path
    assert int(harmonic.training.epochs) == 1, harmonic_path
    print("Hierarchical suite config audit: PASS (8 full-panel learned runs + harmonic)")


if __name__ == "__main__":
    main()
