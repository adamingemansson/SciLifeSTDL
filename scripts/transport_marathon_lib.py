#!/usr/bin/env python3
"""Shared deterministic config resolution for the 40-run transport marathon."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_MATRIX = Path("configs/recovery_suite/transport_marathon_40.yaml")


def load_matrix(path: str | Path = DEFAULT_MATRIX):
    matrix = OmegaConf.load(path)
    if not matrix.get("runs"):
        raise ValueError(f"transport matrix {path} has no runs")
    return matrix


def resolve_run(matrix, index: int):
    entry = matrix.runs[int(index)]
    profile_name = str(entry.profile)
    if profile_name not in matrix.profiles:
        raise ValueError(f"run {entry.id} references unknown profile {profile_name!r}")
    base_path = Path(str(entry.base_config))
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    cfg = OmegaConf.merge(
        OmegaConf.load(base_path),
        matrix.profiles[profile_name],
        entry.get("overrides", {}),
    )
    cfg.experiment_name = str(entry.name)
    cfg.training.seed = int(entry.seed)
    cfg.training.checkpoint_dir = (
        f"results/checkpoints/recovery_suite/{entry.name}"
    )
    return entry, cfg


def final_metrics_path(cfg) -> Path:
    checkpoint = Path(str(cfg.training.checkpoint_dir))
    if cfg.data.get("sample_ids") is not None:
        return checkpoint / "heldout_sample_summary.json"
    return checkpoint / "audit_test_metrics.json"


def smoke_config(cfg, run_id: str):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    name = str(cfg.experiment_name)
    cfg.training.epochs = 1
    cfg.training.unique_mask_count = 1
    cfg.training.checkpoint_every_n_steps = 0
    cfg.training.log_print_every_n_steps = 1
    cfg.training.checkpoint_dir = (
        f"results/checkpoints/recovery_suite/smoke/{run_id}/{name}"
    )
    cfg.validation.every_n_steps = 1
    cfg.validation.early_stopping_min_steps = 1
    cfg.validation.patience_checks = 1000
    cfg.validation.require_anchor_improvement = False
    cfg.validation.n_samples = 1
    cfg.evaluation.n_validation_masks = 1
    cfg.evaluation.n_test_masks = 1
    cfg.evaluation.n_samples = 1
    smoke_root = f"results/mask_banks/recovery_suite/smoke_transport_marathon/{run_id}"
    if cfg.data.get("sample_ids") is not None:
        cfg.evaluation.mask_bank_dir = f"{smoke_root}/{name}"
    else:
        cfg.evaluation.mask_bank_path = f"{smoke_root}/{name}.json"
    cfg.evaluation.training_mask_bank_path = (
        f"results/mask_banks/training/recovery_suite/"
        f"smoke_transport_marathon_{run_id}_{name}.json"
    )
    return cfg
