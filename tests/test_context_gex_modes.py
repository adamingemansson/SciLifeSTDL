"""Regression tests for missing-tissue context-GEX interventions."""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.training.train import (
    _build_masked_item,
    _validate_task_contract,
    _validated_sample_groups,
)


class _MaskingCfg:
    strategy = "random_dropout_patches"
    params = {"n_patches": 1, "radius_range": (1, 2)}


def _item(mode: str, dropout_p: float = 0.0):
    n, g, d = 8, 3, 2
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    expr = np.arange(1, n * g + 1, dtype=np.float32).reshape(n, g)
    novae = np.arange(101, 101 + n * d, dtype=np.float32).reshape(n, d)
    context_mask = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    return _build_masked_item(
        coords,
        expr,
        np.zeros(n, dtype=np.int64),
        _MaskingCfg(),
        images=None,
        seed=7,
        context_novae_features=novae,
        context_gex_mode=mode,
        context_gex_dropout_p=dropout_p,
        fixed_context_mask=context_mask,
        fixed_query_mask=~context_mask,
    )


def test_zero_removes_raw_expression_and_novae_together():
    item = _item("zero")
    assert torch.count_nonzero(item["context"]["expression"]) == 0
    assert torch.count_nonzero(item["context"]["novae_features"]) == 0
    # The held-out target must never be altered by an input intervention.
    assert torch.count_nonzero(item["target_expression"]) > 0


def test_shared_shuffle_is_deterministic_and_keeps_gex_channels_paired():
    original = _item("full")
    shuffled_a = _item("shuffled")
    shuffled_b = _item("shuffled")
    assert torch.equal(shuffled_a["context"]["expression"], shuffled_b["context"]["expression"])
    assert torch.equal(shuffled_a["context"]["novae_features"], shuffled_b["context"]["novae_features"])
    assert not torch.equal(shuffled_a["context"]["expression"], original["context"]["expression"])

    original_pairs = {
        (tuple(expr.tolist()), tuple(novae.tolist()))
        for expr, novae in zip(
            original["context"]["expression"], original["context"]["novae_features"]
        )
    }
    shuffled_pairs = {
        (tuple(expr.tolist()), tuple(novae.tolist()))
        for expr, novae in zip(
            shuffled_a["context"]["expression"], shuffled_a["context"]["novae_features"]
        )
    }
    assert shuffled_pairs == original_pairs


def test_context_gex_dropout_can_remove_the_complete_modality():
    item = _item("full", dropout_p=1.0)
    assert torch.count_nonzero(item["context"]["expression"]) == 0
    assert torch.count_nonzero(item["context"]["novae_features"]) == 0


@pytest.mark.parametrize(
    ("ablation", "image_mode", "gex_mode"),
    [
        ("both", "target_zero", "full"),
        ("gex_only", "all_zero", "full"),
        ("he_only", "target_zero", "zero"),
        ("neither", "all_zero", "zero"),
    ],
)
def test_missing_tissue_modality_contract_accepts_only_matching_inputs(
    ablation, image_mode, gex_mode
):
    cfg = OmegaConf.create(
        {
            "data": {"task_contract": "missing_tissue", "modality_ablation": ablation},
            "training": {
                "image_mode": image_mode,
                "context_gex_mode": gex_mode,
                "context_gex_dropout_p": 0.0,
                "all_image_dropout_p": 0.0,
            },
            "evaluation": {
                "validation_image_mode": image_mode,
                "primary_image_mode": image_mode,
                "image_modes": [image_mode],
                "context_gex_mode": gex_mode,
            },
        }
    )
    _validate_task_contract(cfg)
    cfg.training.context_gex_mode = "zero" if gex_mode == "full" else "full"
    with pytest.raises(ValueError, match="context_gex_mode"):
        _validate_task_contract(cfg)


def test_sample_holdout_contract_is_disjoint_and_complete():
    cfg = OmegaConf.create(
        {
            "data": {
                "holdout_unit": "sample",
                "sample_ids": ["A", "B", "C", "D"],
                "train_sample_ids": ["A", "B"],
                "validation_sample_ids": ["C"],
                "test_sample_ids": ["D"],
            }
        }
    )
    assert _validated_sample_groups(cfg) == (["A", "B"], ["C"], ["D"])

    cfg.data.test_sample_ids = ["B"]
    with pytest.raises(ValueError, match="overlap"):
        _validated_sample_groups(cfg)

    cfg.data.test_sample_ids = []
    with pytest.raises(ValueError, match="non-empty"):
        _validated_sample_groups(cfg)


def test_training_seed_validation_is_single_sample_only():
    cfg = OmegaConf.create({
        "data": {"task_contract": "missing_tissue", "modality_ablation": "both"},
        "training": {
            "image_mode": "target_zero", "context_gex_mode": "full",
            "context_gex_dropout_p": 0.0, "all_image_dropout_p": 0.0,
            "augment_coords": False,
        },
        "validation": {"mask_source": "training_seed"},
        "evaluation": {
            "validation_image_mode": "target_zero",
            "primary_image_mode": "target_zero",
            "image_modes": ["target_zero"], "context_gex_mode": "full",
        },
    })
    _validate_task_contract(cfg)
    cfg.data.sample_ids = ["A", "B", "C"]
    cfg.data.train_sample_ids = ["A"]
    cfg.data.validation_sample_ids = ["B"]
    cfg.data.test_sample_ids = ["C"]
    cfg.data.holdout_unit = "sample"
    with pytest.raises(ValueError, match="single-sample overfit"):
        _validate_task_contract(cfg)
