#!/usr/bin/env python3
"""Fail-closed static audit for the four same-mask parallel controls."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups


PATHS = (
    Path("configs/recovery_suite/161_hierarchical_coordinate_only_20k.yaml"),
    Path("configs/recovery_suite/162_hierarchical_local_he_raw_gex_20k.yaml"),
    Path("configs/recovery_suite/163_hierarchical_global_he_raw_gex_20k.yaml"),
    Path("configs/recovery_suite/164_hierarchical_official_stpath_control.yaml"),
)
EXPECTED_SPLIT = (
    ["INT1", "INT2", "INT3", "INT4", "INT5", "INT6"],
    ["INT7"],
    ["INT8"],
)
EXPECTED_FIXED_PANEL = "resources/evaluation/hest_bench/CCRCC_var_50genes.json"


def main() -> None:
    configs = {}
    for path in PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg = OmegaConf.load(path)
        _validate_task_contract(cfg)
        assert _validated_sample_groups(cfg) == EXPECTED_SPLIT, path
        assert bool(cfg.data.strict_broken_region), path
        assert bool(cfg.data.require_image_coverage), path
        assert bool(cfg.data.use_images), path
        assert list(cfg.masking.params.radius_range) == [3.0, 6.0], path
        assert cfg.masking.params.radius_unit == "spot_spacing", path
        assert int(cfg.masking.max_context_points) == 3000, path
        assert cfg.evaluation.mask_bank_dir == "results/mask_banks/recovery_suite/hierarchical_slide_v1", path
        assert int(cfg.evaluation.n_test_masks) == 16, path
        assert list(cfg.evaluation.train_variance_gene_panel_sizes) == [50, 100, 250], path
        assert cfg.evaluation.fixed_gene_panel_paths.stpath_hest_bench_ccrcc_50 == EXPECTED_FIXED_PANEL, path
        configs[str(cfg.experiment_name)] = (path, cfg)

    assert Path(EXPECTED_FIXED_PANEL).is_file()
    assert len(configs) == 4
    assert len({str(cfg.evaluation.training_mask_bank_path) for _, cfg in configs.values()}) == 4

    coordinate = configs["hierarchical_control_coordinate_only_seed10"][1]
    assert coordinate.data.modality_ablation == "neither"
    assert (
        bool(coordinate.model.params.use_slide_context),
        bool(coordinate.model.params.use_local_images),
        bool(coordinate.model.params.use_novae),
    ) == (False, False, False)

    local = configs["hierarchical_control_local_he_raw_gex_seed10"][1]
    assert local.data.modality_ablation == "both"
    assert (
        bool(local.model.params.use_slide_context),
        bool(local.model.params.use_local_images),
        bool(local.model.params.use_novae),
    ) == (False, True, False)

    global_gex = configs["hierarchical_control_global_he_raw_gex_seed10"][1]
    assert global_gex.data.modality_ablation == "both"
    assert (
        bool(global_gex.model.params.use_slide_context),
        bool(global_gex.model.params.use_local_images),
        bool(global_gex.model.params.use_novae),
    ) == (True, False, False)

    for name in (
        "hierarchical_control_coordinate_only_seed10",
        "hierarchical_control_local_he_raw_gex_seed10",
        "hierarchical_control_global_he_raw_gex_seed10",
    ):
        path, cfg = configs[name]
        assert cfg.model.name == "hierarchical_missing_tissue_regressor", path
        assert int(cfg.training.epochs) == 20000, path
        assert int(cfg.training.unique_mask_count) == 256, path
        assert "target_gene_names" not in cfg.model.params, path
        assert "n_genes" not in cfg.model.params, path

    stpath = configs["hierarchical_control_official_stpath"][1]
    assert stpath.model.name == "stpath_official"
    assert stpath.model.params.context_encoder_type == "stpath"
    assert int(stpath.training.epochs) == 1
    assert not bool(stpath.validation.enabled)
    print("Hierarchical parallel-control config audit: PASS (3 learned + official STPath)")


if __name__ == "__main__":
    main()
