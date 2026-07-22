#!/usr/bin/env python3
"""Fail-closed static audit for the hierarchical_gene_transport_regressor
20-run suite (configs/recovery_suite/165-184.yaml) -- see the 2026-07-22
handoff's "Twenty-run suite" section. Mirrors
check_hierarchical_slide_configs.py's own pattern for the sibling dense-
decoder suite.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from src.training.train import _validate_task_contract, _validated_sample_groups

CONFIG_DIR = REPO_ROOT / "configs" / "recovery_suite"
MODEL_NAME = "hierarchical_gene_transport_regressor"
CAPACITY_SPLIT_SAMPLE = "INT7"
HELDOUT_SPLIT = (
    ["INT1", "INT2", "INT3", "INT4", "INT5", "INT6"],
    ["INT7"],
    ["INT8"],
)

# number -> (short_id, is_capacity, expected model.params overrides vs. the
# shared baseline -- see gen_transport_suite.py's own BASELINE/RUNS tables).
RUNS = {
    165: ("o01_full_concat_no_residual", True, {}),
    166: ("o02_full_concat_residual64", True, {"use_residual": True, "residual_rank": 64}),
    167: ("o03_gex_novae_no_he", True, {"use_local_images": False, "use_slide_context": False}),
    168: ("o04_gated_experts_residual64", True, {
        "fusion_mode": "gated_experts", "use_residual": True, "residual_rank": 64,
    }),
    169: ("c01_raw_gex_only", False, {
        "use_novae": False, "use_local_images": False, "use_slide_context": False,
    }),
    170: ("c02_gex_novae", False, {"use_local_images": False, "use_slide_context": False}),
    171: ("c03_gex_local_he", False, {"use_novae": False, "use_slide_context": False}),
    172: ("c04_gex_global_he", False, {"use_novae": False, "use_local_images": False}),
    173: ("c05_all_modalities_concat_k128", False, {}),
    174: ("c06_all_modalities_gated_experts", False, {"fusion_mode": "gated_experts"}),
    175: ("c07_geometry_only_scoring", False, {"conditioning_mode": "geometry"}),
    176: ("c08_shared_gene_gate", False, {"gene_gate_mode": "shared"}),
    177: ("c09_per_gene_gate_no_query", False, {"use_query_gate": False}),
    178: ("c10_one_transport_head", False, {"transport_heads": 1}),
    179: ("c11_four_transport_heads", False, {"transport_heads": 4}),
    180: ("c12_sixteen_transport_heads", False, {"transport_heads": 16}),
    181: ("c13_k32", False, {"local_k": 32}),
    182: ("c14_k64", False, {"local_k": 64}),
    183: ("c15_residual_rank32", False, {"use_residual": True, "residual_rank": 32}),
    184: ("c16_residual_rank64", False, {"use_residual": True, "residual_rank": 64}),
}

BASELINE_DEFAULTS = {
    "use_novae": True, "use_local_images": True, "use_slide_context": True,
    "fusion_mode": "concat", "transport_heads": 8, "local_k": 128,
    "conditioning_mode": "hierarchical", "gene_gate_mode": "per_gene",
    "use_query_gate": True, "use_residual": False,
}


def main() -> None:
    configs = []
    for number, (short_id, is_capacity, overrides) in RUNS.items():
        filename = f"{number}_transport_{short_id}.yaml"
        path = CONFIG_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        cfg = OmegaConf.load(path)
        _validate_task_contract(cfg)
        assert cfg.model.name == MODEL_NAME, path
        assert cfg.data.novae_mode == "context_only", path
        assert bool(cfg.data.strict_broken_region), path
        assert bool(cfg.data.require_image_coverage), path
        assert bool(cfg.data.use_images), path
        assert cfg.training.image_mode == "target_zero", path
        assert "target_gene_scale" not in cfg.model.params, (
            f"{path}: target_gene_scale must stay train-only-injected, not hardcoded"
        )
        assert "n_genes" not in cfg.model.params, path

        if is_capacity:
            assert cfg.data.get("sample_ids") is None, path
            assert str(cfg.data.sample_id) == CAPACITY_SPLIT_SAMPLE, path
            assert int(cfg.training.epochs) == 3000, path
            assert int(cfg.training.unique_mask_count) == 1, path
            assert bool(cfg.training.exclude_evaluation_query_spots), path
            assert str(cfg.validation.mask_source) == "training_seed", path
            assert not bool(cfg.evaluation.enabled), path
            assert abs(float(cfg.validation.anchor_min_delta) - 0.002) < 1e-9, path
            assert abs(float(cfg.validation.min_correction_rms) - 0.005) < 1e-9, path
        else:
            assert _validated_sample_groups(cfg) == HELDOUT_SPLIT, path
            assert int(cfg.training.epochs) == 20000, path
            assert int(cfg.training.unique_mask_count) == 256, path
            assert list(cfg.evaluation.train_variance_gene_panel_sizes) == [50, 100, 250], path

        expected = dict(BASELINE_DEFAULTS)
        expected.update(overrides)
        for key, value in expected.items():
            actual = cfg.model.params.get(key)
            assert actual == value, (
                f"{path}: model.params.{key}={actual!r}, expected {value!r} "
                f"(short_id={short_id})"
            )
        configs.append((path, cfg))

    experiment_names = [str(cfg.experiment_name) for _, cfg in configs]
    assert len(set(experiment_names)) == len(experiment_names), "duplicate experiment_name"
    checkpoint_dirs = [str(cfg.training.checkpoint_dir) for _, cfg in configs]
    assert len(set(checkpoint_dirs)) == len(checkpoint_dirs), "duplicate checkpoint_dir"

    print(f"Transport suite config audit: PASS ({len(configs)} runs -- 4 capacity gates + 16 held-out)")


if __name__ == "__main__":
    main()
