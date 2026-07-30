"""Tests for the Step 6 deliverable scripts: the tiny single-sample
overfit/capacity test (step6_overfit_test.py), the four-GPU SHORT
DIAGNOSTIC launcher (step6_four_gpu_diagnostic.py, smoke_only=True
ALWAYS -- never the auto-promoting run_suite_with_smoke_gate), and the
progress-check/result-summary commands (step6_progress.py/
step6_summary.py). Real, small, end-to-end synthetic data throughout."""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.models import model_factory as mf
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.scripts import step6_four_gpu_diagnostic, step6_overfit_test, step6_progress, step6_summary
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment, step6_model_params, write_step6_train_config
from gen3_multiscale.training import train as train_module


def test_build_single_sample_config_restricts_to_one_train_sample_and_zero_held_out(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(
        tmp_path, monkeypatch, samples_per_split={"train": 3, "validation": 1, "test": 1},
    )
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_source"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)

    sample_id = manifest["train_sample_ids"][1]
    overfit_checkpoint_dir = tmp_path / "ckpt_overfit"
    overfit_config_path = step6_overfit_test.build_single_sample_config(
        config_path, sample_id, n_steps=5, checkpoint_dir=overfit_checkpoint_dir,
    )
    overfit_config = yaml.safe_load(overfit_config_path.read_text())
    assert overfit_config["training"]["total_steps"] == 5
    assert overfit_config["training"]["checkpoint_dir"] == str(overfit_checkpoint_dir)

    assert overfit_config["data"]["gen3_manifest_path"] == str(manifest_path)
    assert overfit_config["data"]["train_sample_ids_override"] == [sample_id]
    assert overfit_config["data"]["validation_sample_ids_override"] == []
    # The immutable full manifest stays untouched so staged conditioner
    # and autoencoder identities remain valid during flow capacity gates.
    assert load_dataset_manifest(manifest_path)["train_sample_ids"] == manifest["train_sample_ids"]


def test_build_single_sample_config_rejects_an_unknown_sample_id(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=tmp_path / "ckpt")
    with pytest.raises(ValueError, match="not a sample"):
        step6_overfit_test.build_single_sample_config(config_path, "NOT_A_REAL_SAMPLE", n_steps=5, checkpoint_dir=tmp_path / "ckpt2")


def test_overfit_test_runs_a_real_short_run_on_one_sample(tmp_path, monkeypatch):
    """End-to-end: the derived config actually runs through the real
    trainer and produces real checkpointed weights."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=tmp_path / "ckpt_unused")

    # A real synchronized-init dir for a single architecture -- run_training
    # only needs its own architecture's subdirectory to exist.
    common = dict(n_genes=len(manifest["gene_panel"]), gex_feature_dim=8, seed=0)
    gene_names = list(manifest["gene_panel"])
    residuals = np.random.default_rng(2).normal(size=(10, len(gene_names))).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    models = {
        "architecture1": mf.build_architecture({"model": {"architecture": "1", "params": step6_model_params("1")}}, **common),
        "architecture2": mf.build_architecture({"model": {"architecture": "2", "params": step6_model_params("2", use_anchor_blend=True)}}, **common),
        "architecture3": mf.build_architecture({"model": {"architecture": "3", "params": step6_model_params("3", use_regional_he=True, use_global_gex=True)}}, **common),
        "architecture4": mf.build_architecture(
            {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
            **common, gene_basis=basis, gene_names=gene_names,
        ),
    }
    sync_dir = tmp_path / "sync"
    mf.persist_four_architecture_initializations(models, sync_dir)

    config = yaml.safe_load(config_path.read_text())
    config["training"]["synchronized_init_dir"] = str(sync_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    sample_id = manifest["train_sample_ids"][0]
    checkpoint_dir = tmp_path / "ckpt_overfit_real"
    overfit_config_path = step6_overfit_test.build_single_sample_config(config_path, sample_id, n_steps=2, checkpoint_dir=checkpoint_dir)
    summary = train_module.run_training(str(overfit_config_path), smoke=False)
    assert summary["ok"] is True
    assert summary["final_step"] == 2
    assert (checkpoint_dir / "trainable_weights.pt").is_file()


def _sync_dir_for_overfit(tmp_path, manifest):
    common = dict(n_genes=len(manifest["gene_panel"]), gex_feature_dim=8, seed=0)
    gene_names = list(manifest["gene_panel"])
    residuals = np.random.default_rng(6).normal(size=(10, len(gene_names))).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    models = {
        "architecture1": mf.build_architecture({"model": {"architecture": "1", "params": step6_model_params("1")}}, **common),
        "architecture2": mf.build_architecture({"model": {"architecture": "2", "params": step6_model_params("2", use_anchor_blend=True)}}, **common),
        "architecture3": mf.build_architecture({"model": {"architecture": "3", "params": step6_model_params("3", use_regional_he=True, use_global_gex=True)}}, **common),
        "architecture4": mf.build_architecture(
            {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
            **common, gene_basis=basis, gene_names=gene_names,
        ),
    }
    sync_dir = tmp_path / "sync_overfit"
    mf.persist_four_architecture_initializations(models, sync_dir)
    return sync_dir


def test_run_overfit_gate_passes_when_the_model_genuinely_learns(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): "Merely completing 200 steps is not a capacity test." A
    high learning rate on a single tiny sample should genuinely drive
    RMSE down on the fixed eval mask within a small number of steps."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _sync_dir_for_overfit(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=tmp_path / "unused")
    config = yaml.safe_load(config_path.read_text())
    config["training"]["synchronized_init_dir"] = str(sync_dir)
    config["training"]["lr"] = 0.05
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    sample_id = manifest["train_sample_ids"][0]
    checkpoint_dir = tmp_path / "ckpt_overfit_gate"
    gate_report = step6_overfit_test.run_overfit_gate(
        str(config_path), sample_id, n_steps=60, checkpoint_dir=str(checkpoint_dir),
        min_rmse_improvement_fraction=0.05,
    )
    assert gate_report["passed"] is True
    assert gate_report["after"]["rmse"] < gate_report["before"]["rmse"]


def test_run_overfit_gate_fails_when_the_model_does_not_learn(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _sync_dir_for_overfit(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=tmp_path / "unused2")
    config = yaml.safe_load(config_path.read_text())
    config["training"]["synchronized_init_dir"] = str(sync_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    import torch
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **kw: None)

    sample_id = manifest["train_sample_ids"][0]
    checkpoint_dir = tmp_path / "ckpt_overfit_gate_fail"
    with pytest.raises(RuntimeError, match="overfit gate FAILED"):
        step6_overfit_test.run_overfit_gate(
            str(config_path), sample_id, n_steps=3, checkpoint_dir=str(checkpoint_dir),
        )


def test_run_four_gpu_smoke_diagnostic_calls_launch_suite_with_smoke_only_true(tmp_path, monkeypatch):
    """Wiring test: the diagnostic launcher must call launch_suite with
    smoke_only=True and must NEVER call run_suite_with_smoke_gate (which
    would auto-promote to a full run after a passing smoke gate --
    exactly what Adam's Step 6 instructions forbid)."""
    calls = []

    def _fake_launch_suite(named_configs, config_paths, gpu_list, log_root, threads_per_job=4,
                            smoke_only=False, command_builder=None, skip_fingerprint_check=False):
        calls.append({"smoke_only": smoke_only, "gpu_list": list(gpu_list)})

        class _Result:
            ok = True
            summary_path = str(log_root / "summary.json")

        return _Result()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("run_suite_with_smoke_gate must never be called by the diagnostic launcher")

    monkeypatch.setattr(step6_four_gpu_diagnostic, "launch_suite", _fake_launch_suite)
    monkeypatch.setattr(
        "gen3_multiscale.training.launch_four_gpu_suite.run_suite_with_smoke_gate", _fail_if_called,
    )

    config_paths = {}
    for i in range(1, 5):
        path = tmp_path / f"architecture{i}.yaml"
        path.write_text(yaml.safe_dump({"model": {"architecture": str(i)}, "documented_divergences": ["model.architecture"]}))
        config_paths[f"architecture{i}"] = path

    result = step6_four_gpu_diagnostic.run_four_gpu_smoke_diagnostic(
        config_paths, ["0", "1", "2", "3"], tmp_path / "logs",
    )
    assert result.ok is True
    assert len(calls) == 1
    assert calls[0]["smoke_only"] is True
    assert calls[0]["gpu_list"] == ["0", "1", "2", "3"]


def test_gather_progress_on_a_fresh_checkpoint_dir_reports_step_zero(tmp_path):
    progress = step6_progress.gather_progress(tmp_path / "does_not_exist_yet")
    assert progress["step"] == 0
    assert progress["checkpoint_history_steps"] == []
    assert progress["has_run_manifest"] is False


def test_gather_progress_and_build_summary_reflect_a_real_smoke_run(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True

    progress = step6_progress.gather_progress(checkpoint_dir)
    assert progress["has_run_manifest"] is True
    assert progress["architecture"] == "1"
    assert progress["preflight_passed"] is True
    assert progress["n_train_samples"] == len(manifest["train_sample_ids"])

    result_summary = step6_summary.build_summary(checkpoint_dir)
    assert result_summary["architecture"] == "1"
    assert result_summary["cache_preflight_passed"] is True
    assert result_summary["split"]["n_train_samples"] == len(manifest["train_sample_ids"])
    assert result_summary["train_mask_schedule"]["n_failed"] == 0
    assert result_summary["train_mask_schedule"]["n_samples"] == len(manifest["train_sample_ids"])
    # smoke mode never writes real weights.
    assert result_summary["has_trainable_weights"] is False
