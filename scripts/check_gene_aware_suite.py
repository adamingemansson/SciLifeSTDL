#!/usr/bin/env python3
"""Fail-closed static audit and queue query for the 16 gene-aware runs."""
from __future__ import annotations

import argparse
from collections import Counter

from omegaconf import OmegaConf

from scripts.gene_aware_suite_lib import DEFAULT_MATRIX, load_matrix, resolve_run
from src.data.context_features import model_uses_novae
from src.training.train import _validate_task_contract, _validated_sample_groups


SMOKE_IDS = {"O03", "C01", "C04", "S802"}


def validate(matrix) -> None:
    runs = list(matrix.runs)
    ids = [str(entry.id) for entry in runs]
    names = [str(entry.name) for entry in runs]
    assert len(runs) == len(set(ids)) == len(set(names)) == 16
    assert Counter(int(entry.slot) for entry in runs) == {0: 4, 1: 4, 2: 4, 3: 4}
    assert Counter(str(entry.stage) for entry in runs) == {"overfit": 4, "heldout": 12}
    assert Counter(str(entry.scope) for entry in runs) == {
        "overfit_int7": 4, "cross": 8, "single_int7": 2, "single_int8": 2,
    }
    for slot in range(4):
        slot_runs = [entry for entry in runs if int(entry.slot) == slot]
        assert Counter(str(entry.stage) for entry in slot_runs) == {
            "overfit": 1, "heldout": 3,
        }
    assert SMOKE_IDS <= set(ids)

    mask_identity = {}
    for index in range(16):
        entry, cfg = resolve_run(matrix, index)
        _validate_task_contract(cfg)
        assert str(cfg.model.name) == "gene_aware_transport_regressor", entry.id
        assert int(cfg.model.params.transport_k) in {32, 64}, entry.id
        assert int(cfg.model.params.transport_heads) in {8, 16}, entry.id
        assert float(cfg.model.params.lr) == 0.001, entry.id
        assert float(cfg.model.params.conditioner_lr) == 0.0003, entry.id
        assert cfg.training.augment_coords is False, entry.id
        assert cfg.training.context_gex_mode == "full", entry.id
        assert cfg.evaluation.context_gex_mode == "full", entry.id
        assert cfg.data.task_contract == "missing_tissue", entry.id
        assert cfg.data.require_image_coverage is True, entry.id
        assert str(cfg.training.image_mode) in {"target_zero", "all_zero"}, entry.id
        assert str(cfg.evaluation.primary_image_mode) == str(cfg.training.image_mode), entry.id
        assert str(cfg.evaluation.validation_image_mode) == str(cfg.training.image_mode), entry.id
        assert "full" not in list(cfg.evaluation.image_modes), entry.id
        assert float(cfg.training.query_image_dropout_p) == 0.0, entry.id
        assert float(cfg.training.context_gex_dropout_p) == 0.0, entry.id

        if str(cfg.model.params.conditioning_mode) == "storm_lite":
            assert cfg.model.params.storm_lite_use_absolute_coords is False, entry.id
            assert int(cfg.model.params.storm_lite_knn_k) == 32, entry.id
            requested = model_uses_novae(
                OmegaConf.to_container(cfg.model.params, resolve=True)
            )
            assert requested == (str(cfg.data.novae_mode) == "context_only"), entry.id
        else:
            assert str(cfg.model.params.conditioning_mode) == "geometry", entry.id
            assert str(cfg.data.novae_mode) == "disabled", entry.id

        if str(cfg.data.modality_ablation) == "gex_only":
            assert cfg.data.use_images is False, entry.id
            assert str(cfg.training.image_mode) == "all_zero", entry.id
            assert list(cfg.evaluation.image_modes) == ["all_zero"], entry.id
        else:
            assert str(cfg.data.modality_ablation) == "both", entry.id
            assert cfg.data.use_images is True, entry.id
            assert str(cfg.training.image_mode) == "target_zero", entry.id
            assert list(cfg.evaluation.image_modes) == ["target_zero", "all_zero"], entry.id

        if str(entry.stage) == "overfit":
            assert int(cfg.training.epochs) == 3000, entry.id
            assert int(cfg.training.unique_mask_count) == 1, entry.id
            assert str(cfg.validation.mask_source) == "training_seed", entry.id
            assert cfg.evaluation.enabled is False, entry.id
            assert str(cfg.data.sample_id) == "INT7", entry.id
            assert cfg.training.exclude_evaluation_query_spots is True, entry.id
        else:
            assert int(cfg.training.epochs) == 10000, entry.id
            assert int(cfg.training.unique_mask_count) == 256, entry.id
            assert str(cfg.validation.mask_source) == "evaluation", entry.id
            assert cfg.evaluation.enabled is True, entry.id
            if str(entry.scope) == "cross":
                assert _validated_sample_groups(cfg) == (
                    ["INT1", "INT2", "INT3", "INT4", "INT5", "INT6"],
                    ["INT7"], ["INT8"],
                ), entry.id
                assert cfg.data.holdout_unit == "sample", entry.id
                assert not bool(cfg.training.get("exclude_evaluation_query_spots", False)), entry.id
            else:
                sample = "INT7" if str(entry.scope) == "single_int7" else "INT8"
                assert str(cfg.data.sample_id) == sample, entry.id
                assert cfg.data.holdout_unit == "spot", entry.id
                assert cfg.training.exclude_evaluation_query_spots is True, entry.id

        if cfg.data.get("sample_ids") is not None:
            identity = (
                str(cfg.evaluation.mask_bank_dir), int(cfg.evaluation.n_validation_masks),
                int(cfg.evaluation.n_test_masks), int(cfg.evaluation.validation_seed),
                int(cfg.evaluation.test_seed),
            )
        else:
            identity = (
                str(cfg.evaluation.mask_bank_path), int(cfg.evaluation.n_validation_masks),
                int(cfg.evaluation.n_test_masks), int(cfg.evaluation.validation_seed),
                int(cfg.evaluation.test_seed),
            )
        prior = mask_identity.setdefault(str(entry.scope), identity)
        assert prior == identity, f"scope {entry.scope} does not share exact masks"

        serialized = OmegaConf.to_yaml(cfg).lower()
        assert "highly_variable" not in serialized and "hvg" not in serialized, entry.id
        assert "harmonic_k" not in serialized and "harmonic_ridge" not in serialized, entry.id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--stage", choices=("overfit", "heldout"))
    parser.add_argument("--slot", type=int)
    parser.add_argument("--smoke-indices", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    validate(matrix)
    if args.smoke_indices:
        for index, entry in enumerate(matrix.runs):
            if str(entry.id) in SMOKE_IDS:
                print(index)
        return
    if args.stage is not None or args.slot is not None:
        if args.slot is not None and args.slot not in {0, 1, 2, 3}:
            raise ValueError("slot must be 0, 1, 2, or 3")
        for index, entry in enumerate(matrix.runs):
            if args.stage is not None and str(entry.stage) != args.stage:
                continue
            if args.slot is not None and int(entry.slot) != args.slot:
                continue
            print(index)
        return
    print("Gene-aware suite config audit: PASS")
    print("16 runs: four capacity gates plus twelve held-out diagnostics")
    print("Each GPU slot contains one overfit job followed by three held-out jobs")


if __name__ == "__main__":
    main()
