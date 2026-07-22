#!/usr/bin/env python3
"""Fail-closed static audit for the hierarchical_gene_transport_regressor
20-run suite, v2 (configs/recovery_suite/186-206.yaml). Identical ablation
grid to the first suite (165-185) with exactly one change:
transport_reg_weight=0.0 (was 0.001) -- the first suite's own logs showed
transport_head_entropy pinned near ln(128)=4.852 (the true maximum) for the
entire 20k-step run on every config, meaning the entropy-toward-uniformity
term likely prevented the transport heads from ever specializing. See
src/models/registry.py's HierarchicalGeneTransportRegressor.training_step
docstring/comment for the full reasoning. This script asserts that fix is
actually present in every v2 config, not just that the ablation grid matches.
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
# shared baseline -- see gen_transport_suite_v2.py's own BASELINE/RUNS tables).
RUNS = {
    186: ("o01_full_concat_no_residual_v2", True, {}),
    187: ("o02_full_concat_residual64_v2", True, {"use_residual": True, "residual_rank": 64}),
    188: ("o03_gex_novae_no_he_v2", True, {"use_local_images": False, "use_slide_context": False}),
    189: ("o04_gated_experts_residual64_v2", True, {
        "fusion_mode": "gated_experts", "use_residual": True, "residual_rank": 64,
    }),
    190: ("c01_raw_gex_only_v2", False, {
        "use_novae": False, "use_local_images": False, "use_slide_context": False,
    }),
    191: ("c02_gex_novae_v2", False, {"use_local_images": False, "use_slide_context": False}),
    192: ("c03_gex_local_he_v2", False, {"use_novae": False, "use_slide_context": False}),
    193: ("c04_gex_global_he_v2", False, {"use_novae": False, "use_local_images": False}),
    194: ("c05_all_modalities_concat_k128_v2", False, {}),
    195: ("c06_all_modalities_gated_experts_v2", False, {"fusion_mode": "gated_experts"}),
    196: ("c07_geometry_only_scoring_v2", False, {"conditioning_mode": "geometry"}),
    197: ("c08_shared_gene_gate_v2", False, {"gene_gate_mode": "shared"}),
    198: ("c09_per_gene_gate_no_query_v2", False, {"use_query_gate": False}),
    199: ("c10_one_transport_head_v2", False, {"transport_heads": 1}),
    200: ("c11_four_transport_heads_v2", False, {"transport_heads": 4}),
    201: ("c12_sixteen_transport_heads_v2", False, {"transport_heads": 16}),
    202: ("c13_k32_v2", False, {"local_k": 32}),
    203: ("c14_k64_v2", False, {"local_k": 64}),
    204: ("c15_residual_rank32_v2", False, {"use_residual": True, "residual_rank": 32}),
    205: ("c16_residual_rank64_v2", False, {"use_residual": True, "residual_rank": 64}),
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
        # The actual fix under test: entropy regularization must be off.
        actual_reg_weight = float(cfg.model.params.get("transport_reg_weight", -1))
        assert actual_reg_weight == 0.0, (
            f"{path}: transport_reg_weight={actual_reg_weight}, expected 0.0 for the v2 fix"
        )

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

    # Also verify no v2 config accidentally collides with a v1 checkpoint_dir
    # (would cause the runner's resumable-artifact detection to silently
    # reuse v1's stale, entropy-regularized results instead of rerunning).
    v1_names = {
        "transport_capacity_o01_full_concat_no_residual",
        "transport_capacity_o02_full_concat_residual64",
        "transport_capacity_o03_gex_novae_no_he",
        "transport_capacity_o04_gated_experts_residual64",
        "transport_c01_raw_gex_only_20k", "transport_c02_gex_novae_20k",
        "transport_c03_gex_local_he_20k", "transport_c04_gex_global_he_20k",
        "transport_c05_all_modalities_concat_k128_20k",
        "transport_c06_all_modalities_gated_experts_20k",
        "transport_c07_geometry_only_scoring_20k", "transport_c08_shared_gene_gate_20k",
        "transport_c09_per_gene_gate_no_query_20k", "transport_c10_one_transport_head_20k",
        "transport_c11_four_transport_heads_20k", "transport_c12_sixteen_transport_heads_20k",
        "transport_c13_k32_20k", "transport_c14_k64_20k",
        "transport_c15_residual_rank32_20k", "transport_c16_residual_rank64_20k",
    }
    collisions = v1_names & set(experiment_names)
    assert not collisions, f"v2 configs collide with v1 experiment_names: {collisions}"

    print(f"Transport suite v2 config audit: PASS ({len(configs)} runs -- "
          f"4 capacity gates + 16 held-out, transport_reg_weight=0.0 confirmed)")


if __name__ == "__main__":
    main()
