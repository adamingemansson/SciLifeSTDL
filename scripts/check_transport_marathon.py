#!/usr/bin/env python3
"""Fail-closed static audit and queue query for the 40-run marathon."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from scripts.transport_marathon_lib import DEFAULT_MATRIX, load_matrix, resolve_run
from src.data.context_features import model_uses_novae
from src.training.train import _validate_task_contract, _validated_sample_groups


EXPECTED_SCOPE_COUNTS = {"cross": 24, "single_int7": 8, "single_int8": 8}
EXPECTED_CROSS_FAMILIES = {
    "geometry": 4, "gex_both": 4, "full_both": 4,
    "conditioner_encoder": 4, "architecture": 6, "uniform": 1, "seed": 1,
}
EXPECTED_SINGLE_FAMILIES = {
    "uniform": 1, "geometry": 3, "gex_both": 2, "full_both": 2,
}
SMOKE_IDS = {"C01", "C07", "C11", "C17", "S701", "S708", "S801", "S808"}


def validate(matrix) -> None:
    runs = list(matrix.runs)
    assert len(runs) == 40, f"expected 40 runs, got {len(runs)}"
    ids = [str(run.id) for run in runs]
    names = [str(run.name) for run in runs]
    assert len(set(ids)) == 40, "matrix run IDs are not unique"
    assert len(set(names)) == 40, "matrix experiment names are not unique"
    assert Counter(int(run.slot) for run in runs) == {0: 10, 1: 10, 2: 10, 3: 10}
    assert Counter(str(run.scope) for run in runs) == EXPECTED_SCOPE_COUNTS
    cross = Counter(str(run.family) for run in runs if str(run.scope) == "cross")
    assert cross == EXPECTED_CROSS_FAMILIES, cross
    for scope in ("single_int7", "single_int8"):
        families = Counter(str(run.family) for run in runs if str(run.scope) == scope)
        assert families == EXPECTED_SINGLE_FAMILIES, (scope, families)
    assert SMOKE_IDS <= set(ids)

    by_scope = {}
    for index, run in enumerate(runs):
        entry, cfg = resolve_run(matrix, index)
        _validate_task_contract(cfg)
        assert cfg.model.name == "context_transport_regressor", entry.id
        assert int(cfg.training.epochs) == 20000, entry.id
        assert int(cfg.training.unique_mask_count) == 256, entry.id
        assert int(cfg.training.mask_seed) == 730000, entry.id
        assert cfg.training.augment_coords is False, entry.id
        assert cfg.training.context_gex_mode == "full", entry.id
        assert cfg.evaluation.context_gex_mode == "full", entry.id
        assert str(cfg.training.image_mode) in {"target_zero", "all_zero"}, entry.id
        assert str(cfg.evaluation.primary_image_mode) == str(cfg.training.image_mode), entry.id
        assert str(cfg.evaluation.validation_image_mode) == str(cfg.training.image_mode), entry.id
        assert "full" not in list(cfg.evaluation.image_modes), entry.id
        assert float(cfg.training.query_image_dropout_p) == 0.0, entry.id
        assert float(cfg.training.context_gex_dropout_p) == 0.0, entry.id
        assert int(cfg.model.params.transport_k) in {8, 16, 32, 64}, entry.id
        assert cfg.data.task_contract == "missing_tissue", entry.id
        assert cfg.data.require_image_coverage is True, entry.id
        assert cfg.data.use_images is True, entry.id
        if cfg.model.params.conditioning_mode == "storm_lite":
            assert cfg.model.params.storm_lite_use_absolute_coords is False, entry.id
            assert int(cfg.model.params.storm_lite_knn_k) == 32, entry.id
            requests_novae = model_uses_novae(
                OmegaConf.to_container(cfg.model.params, resolve=True)
            )
            assert requests_novae == (cfg.data.novae_mode == "context_only"), entry.id
        else:
            assert cfg.data.novae_mode == "disabled", entry.id

        if str(entry.scope) == "cross":
            assert _validated_sample_groups(cfg) == (
                ["INT1", "INT2", "INT3", "INT4", "INT5", "INT6"],
                ["INT7"], ["INT8"],
            ), entry.id
            assert cfg.data.holdout_unit == "sample", entry.id
            assert not bool(cfg.training.get("exclude_evaluation_query_spots", False)), entry.id
            mask_identity = (
                str(cfg.evaluation.mask_bank_dir), int(cfg.evaluation.n_validation_masks),
                int(cfg.evaluation.n_test_masks), int(cfg.evaluation.validation_seed),
                int(cfg.evaluation.test_seed), str(cfg.evaluation.training_mask_bank_path),
            )
        else:
            expected_sample = "INT7" if str(entry.scope) == "single_int7" else "INT8"
            assert str(cfg.data.sample_id) == expected_sample, entry.id
            assert cfg.data.get("sample_ids") is None, entry.id
            assert cfg.data.holdout_unit == "spot", entry.id
            assert cfg.training.exclude_evaluation_query_spots is True, entry.id
            assert int(cfg.evaluation.n_validation_masks) == 4, entry.id
            assert int(cfg.evaluation.n_test_masks) == 8, entry.id
            mask_identity = (
                str(cfg.evaluation.mask_bank_path), int(cfg.evaluation.n_validation_masks),
                int(cfg.evaluation.n_test_masks), int(cfg.evaluation.validation_seed),
                int(cfg.evaluation.test_seed), str(cfg.evaluation.training_mask_bank_path),
            )
        if str(cfg.data.modality_ablation) == "gex_only":
            assert str(cfg.training.image_mode) == "all_zero", entry.id
            assert list(cfg.evaluation.image_modes) == ["all_zero"], entry.id
        else:
            assert str(cfg.data.modality_ablation) == "both", entry.id
            assert str(cfg.training.image_mode) == "target_zero", entry.id
            assert list(cfg.evaluation.image_modes) == ["target_zero", "all_zero"], entry.id
        previous = by_scope.setdefault(str(entry.scope), mask_identity)
        assert previous == mask_identity, f"scope {entry.scope} does not share exact masks"

        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "highly_variable" not in serialized and "hvg" not in serialized, entry.id
        assert "harmonic_k" not in serialized and "harmonic_ridge" not in serialized, entry.id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--indices-for-slot", type=int)
    parser.add_argument("--smoke-indices", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    validate(matrix)
    if args.indices_for_slot is not None:
        if args.indices_for_slot not in {0, 1, 2, 3}:
            raise ValueError("slot must be 0, 1, 2, or 3")
        for index, run in enumerate(matrix.runs):
            if int(run.slot) == args.indices_for_slot:
                print(index)
        return
    if args.smoke_indices:
        for index, run in enumerate(matrix.runs):
            if str(run.id) in SMOKE_IDS:
                print(index)
        return
    print("Transport marathon config audit: PASS")
    print("40 runs = 24 cross-sample + 8 INT7 single-sample + 8 INT8 single-sample")
    print("GPU slots 0/1/2/3 each contain exactly 10 sequential jobs")


if __name__ == "__main__":
    main()
