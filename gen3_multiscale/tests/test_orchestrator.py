"""Tests for gen3_multiscale/training/orchestrator.py -- the real staged
deployment orchestrator (Stages A-D), built per the Codex re-audit of
commit 57f0e3c's explicit final instruction. Exercises the real
production code path (real `run_training`/`evaluate_gen3_checkpoint`/
`fit_and_save_architecture4_basis` calls) against the same tiny,
synthetic, CPU-only fixtures every other Step 6 test in this package
already uses -- never real GPU hardware or real HEST-1k data, and no
24-hour run is started by anything here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from gen3_multiscale.config_identity import config_fingerprint, config_identity_fingerprint
from gen3_multiscale.models import model_factory as mf
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment, step6_model_params, write_step6_train_config
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training import orchestrator


def _wrap_as_verified_bundle(config: dict, output_dir: Path) -> Path:
    """Mirrors `resolve_experiment_config.py::resolve_and_save_experiment_
    config`'s own bundle-writing internals (config.yaml + identity.json,
    with config_yaml_sha256/config_fingerprint/config_identity_
    fingerprint), for an already-fully-formed config dict -- the tiny
    synthetic Step 6 configs this test suite builds are not derived from
    the real committed configs/architectureN.yaml templates, so
    `resolve_experiment_config()` itself (which validates against that
    specific real schema) does not apply here; this produces exactly the
    bundle SHAPE `load_verified_resolved_config` consumes."""
    output_dir.mkdir(parents=True)
    config_yaml_bytes = yaml.safe_dump(config, sort_keys=False).encode("utf-8")
    (output_dir / "config.yaml").write_bytes(config_yaml_bytes)
    identity = {
        "version": 2,
        "kind": "gen3_resolved_experiment_config_identity",
        "base_config_path": "synthetic-test-fixture",
        "base_template_sha256": hashlib.sha256(config_yaml_bytes).hexdigest(),
        "config_yaml_sha256": hashlib.sha256(config_yaml_bytes).hexdigest(),
        "overrides": {},
        "config_fingerprint": config_fingerprint(config),
        "config_identity_fingerprint": config_identity_fingerprint(config),
    }
    (output_dir / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
    return output_dir


def _build_sync_dir(tmp_path: Path, manifest: dict) -> Path:
    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    residuals = torch.randn(10, n_genes).numpy()
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    common = dict(n_genes=n_genes, gex_feature_dim=8, seed=0)
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
    return sync_dir


def _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir):
    arch3_config_path = tmp_path / "arch3_raw_config.yaml"
    arch3_checkpoint_dir = tmp_path / "arch3_ckpt"
    config = write_step6_train_config(
        cfg, manifest_path, arch3_config_path, architecture="3", checkpoint_dir=arch3_checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True},
        synchronized_init_dir=str(sync_dir),
    )
    config["training"]["total_steps"] = 2
    bundle_dir = tmp_path / "bundle_architecture3"
    _wrap_as_verified_bundle(config, bundle_dir)
    return bundle_dir


def _run_stages_a_through_c(tmp_path, cfg, manifest, manifest_path, sync_dir, state_path):
    arch3_bundle_dir = _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir)
    orchestrator.run_stage_train_architecture(state_path, "train_architecture3", arch3_bundle_dir, smoke=False)
    orchestrator.run_stage_select_architecture3(state_path, n_masks_per_sample=2)
    basis_path = tmp_path / "gene_residual_basis.pt"
    orchestrator.run_stage_fit_basis(state_path, arch3_bundle_dir, basis_path, n_masks_per_sample=2, rank=4)
    return arch3_bundle_dir, basis_path


def _build_architecture4_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir, *, conditioner_checkpoint_dir, basis_path, name="bundle_architecture4"):
    arch4_config_path = tmp_path / f"{name}_raw_config.yaml"
    arch4_checkpoint_dir = tmp_path / f"{name}_ckpt"
    config = write_step6_train_config(
        cfg, manifest_path, arch4_config_path, architecture="4", checkpoint_dir=arch4_checkpoint_dir,
        model_param_overrides={"use_regional_he": True},
        synchronized_init_dir=str(sync_dir), gene_residual_basis_path=str(basis_path),
    )
    config["training"]["total_steps"] = 1
    config["required_fingerprints"]["architecture3_conditioner_checkpoint"] = str(conditioner_checkpoint_dir)
    bundle_dir = tmp_path / name
    _wrap_as_verified_bundle(config, bundle_dir)
    return bundle_dir


def test_orchestrator_stages_a_through_d_succeed_end_to_end_and_architecture4_consumes_the_selected_bundle(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _build_sync_dir(tmp_path, manifest)
    state_path = tmp_path / "orchestrator_state.json"

    arch3_bundle_dir, basis_path = _run_stages_a_through_c(tmp_path, cfg, manifest, manifest_path, sync_dir, state_path)

    state = orchestrator.load_orchestrator_state(state_path)
    selected_dir = state["stages"]["select_architecture3"]["output"]["selected_checkpoint_identity"]["resolved_dir"]

    arch4_bundle_dir = _build_architecture4_bundle(
        tmp_path, cfg, manifest, manifest_path, sync_dir,
        conditioner_checkpoint_dir=selected_dir, basis_path=basis_path,
    )
    result = orchestrator.run_stage_train_architecture4(state_path, arch4_bundle_dir, smoke=False)
    assert result["status"] == "completed"
    assert result["output"]["architecture3_conditioner_checkpoint"] == selected_dir
    assert result["output"]["gene_residual_basis"] == str(Path(basis_path).resolve())

    final_state = orchestrator.load_orchestrator_state(state_path)
    for stage_name in ("train_architecture3", "select_architecture3", "fit_basis", "train_architecture4"):
        assert final_state["stages"][stage_name]["status"] == "completed"


def test_architecture4_stage_refuses_to_run_before_architecture3_selection_and_basis_fitting(tmp_path, monkeypatch):
    """The audit's explicit ask: 'Architecture 4 cannot start before
    Architecture 3 selection and basis fitting.'"""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _build_sync_dir(tmp_path, manifest)
    state_path = tmp_path / "orchestrator_state.json"

    # No stages have run at all yet -- Stage D must refuse immediately.
    arch4_bundle_dir = _build_architecture4_bundle(
        tmp_path, cfg, manifest, manifest_path, sync_dir,
        conditioner_checkpoint_dir="/nonexistent/never/resolved", basis_path=tmp_path / "no_basis.pt",
    )
    with pytest.raises(RuntimeError, match="select_architecture3"):
        orchestrator.run_stage_train_architecture4(state_path, arch4_bundle_dir, smoke=False)

    # Train + select Architecture 3, but do NOT fit the basis -- Stage D
    # must still refuse (fit_basis precondition specifically).
    arch3_bundle_dir = _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir)
    orchestrator.run_stage_train_architecture(state_path, "train_architecture3", arch3_bundle_dir, smoke=False)
    orchestrator.run_stage_select_architecture3(state_path, n_masks_per_sample=2)
    with pytest.raises(RuntimeError, match="fit_basis"):
        orchestrator.run_stage_train_architecture4(state_path, arch4_bundle_dir, smoke=False)


def test_architecture4_stage_refuses_a_config_pointing_at_a_different_architecture3_bundle(tmp_path, monkeypatch):
    """The audit's explicit ask: Architecture 4 'must consume the precise
    selected bundle identity' -- a config bundle whose OWN configured
    architecture3_conditioner_checkpoint points at a DIFFERENT (but
    otherwise real, valid) Architecture 3 checkpoint must be refused,
    never silently trained against."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _build_sync_dir(tmp_path, manifest)
    state_path = tmp_path / "orchestrator_state.json"
    _run_stages_a_through_c(tmp_path, cfg, manifest, manifest_path, sync_dir, state_path)

    # A second, independently-trained Architecture 3 checkpoint -- real
    # and valid, but NOT the one this orchestration actually selected.
    other_arch3_config_path = tmp_path / "other_arch3_config.yaml"
    other_arch3_checkpoint_dir = tmp_path / "other_arch3_ckpt"
    other_config = write_step6_train_config(
        cfg, manifest_path, other_arch3_config_path, architecture="3", checkpoint_dir=other_arch3_checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True},
        synchronized_init_dir=str(sync_dir),
    )
    other_config["training"]["total_steps"] = 2
    other_arch3_config_path.write_text(yaml.safe_dump(other_config, sort_keys=False))
    from gen3_multiscale.training.train import run_training

    summary = run_training(str(other_arch3_config_path), smoke=False)
    assert summary["ok"] is True

    basis_path = tmp_path / "gene_residual_basis.pt"
    arch4_bundle_dir = _build_architecture4_bundle(
        tmp_path, cfg, manifest, manifest_path, sync_dir,
        conditioner_checkpoint_dir=other_arch3_checkpoint_dir, basis_path=basis_path,
    )
    with pytest.raises(ValueError, match="different, possibly older/pre-existing"):
        orchestrator.run_stage_train_architecture4(state_path, arch4_bundle_dir, smoke=False)


def test_a_completed_stage_resumes_idempotently_without_redoing_work(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _build_sync_dir(tmp_path, manifest)
    state_path = tmp_path / "orchestrator_state.json"
    arch3_bundle_dir = _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir)

    first = orchestrator.run_stage_train_architecture(state_path, "train_architecture3", arch3_bundle_dir, smoke=False)
    second = orchestrator.run_stage_train_architecture(state_path, "train_architecture3", arch3_bundle_dir, smoke=False)
    assert first["output"] == second["output"]
    assert first["completed_at"] == second["completed_at"]  # proves the second call did not re-run training


def test_a_completed_stage_refuses_different_inputs_rather_than_silently_overwriting(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    sync_dir = _build_sync_dir(tmp_path, manifest)
    state_path = tmp_path / "orchestrator_state.json"
    arch3_bundle_dir = _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir)
    orchestrator.run_stage_train_architecture(state_path, "train_architecture3", arch3_bundle_dir, smoke=False)

    other_bundle_dir = tmp_path / "a_different_bundle_dir"
    (other_bundle_dir).mkdir()
    (other_bundle_dir / "config.yaml").write_text("not consumed -- resume_check mismatch fails first\n")
    with pytest.raises(ValueError, match="immutable completed stage"):
        orchestrator.run_stage_train_architecture(state_path, "train_architecture3", other_bundle_dir, smoke=False)


def test_a_failed_stage_is_recorded_and_can_be_retried(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    state_path = tmp_path / "orchestrator_state.json"

    # No synchronized_init_dir configured -- run_training raises for a
    # non-smoke run (a genuine, real failure mode).
    arch3_config_path = tmp_path / "arch3_broken_config.yaml"
    arch3_checkpoint_dir = tmp_path / "arch3_ckpt"
    config = write_step6_train_config(
        cfg, manifest_path, arch3_config_path, architecture="3", checkpoint_dir=arch3_checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True}, synchronized_init_dir=None,
    )
    bundle_dir = tmp_path / "broken_bundle"
    _wrap_as_verified_bundle(config, bundle_dir)

    with pytest.raises(ValueError, match="synchronized_init_dir"):
        orchestrator.run_stage_train_architecture(state_path, "train_architecture3", bundle_dir, smoke=False)

    state = orchestrator.load_orchestrator_state(state_path)
    assert state["stages"]["train_architecture3"]["status"] == "failed"
    assert "synchronized_init_dir" in state["stages"]["train_architecture3"]["error"]

    # A failed stage (unlike a completed one) is not immutable -- it can
    # be retried, even with genuinely different/corrected inputs.
    sync_dir = _build_sync_dir(tmp_path, manifest)
    fixed_bundle_dir = _train_architecture3_bundle(tmp_path, cfg, manifest, manifest_path, sync_dir)
    retried = orchestrator.run_stage_train_architecture(state_path, "train_architecture3", fixed_bundle_dir, smoke=False)
    assert retried["status"] == "completed"
