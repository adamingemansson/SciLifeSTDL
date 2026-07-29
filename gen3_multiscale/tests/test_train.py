"""Tests for gen3_multiscale/training/train.py -- Step 6's real training
entrypoint. Real, small, end-to-end synthetic data throughout
(gen3_multiscale/tests/_step6_fixtures.py, shared with
test_gen3_dataset.py/test_gen3_preflight.py); only the GigaPath tile
encoder is ever monkeypatched.

Architecture 3/4 here use use_regional_he=True but use_global_slide=False
-- a real, legitimate partial-feature configuration (regional H&E pooling
needs only the already-cached tile FEATURES, no live FrozenGigaPathSlideEncoder;
only use_global_slide needs the real `gigapath` package and a real LongNet
checkpoint, neither available in this sandbox). This is not a test-only
shortcut for use_regional_he/use_global_gex -- those run for real. The
use_global_slide=True path itself was validated separately on real
hardware (Adam's A100 LongNet/Architecture 3/4 smoke, reported before this
Step 6 work began) and is not re-exercised here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from gen3_multiscale.data.dataset_manifest import save_dataset_manifest
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, save_gene_residual_basis
from gen3_multiscale.tests._step6_fixtures import (
    prepare_step6_experiment, step6_model_params, write_step6_train_config,
)
from gen3_multiscale.training import train as train_module

_prepare = prepare_step6_experiment
_base_model_params = step6_model_params
_write_config = write_step6_train_config


@pytest.mark.parametrize("architecture", ["1", "2"])
def test_run_training_smoke_runs_one_step_for_architecture1_and_2(tmp_path, monkeypatch, architecture):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / f"ckpt_{architecture}"
    overrides = {"use_anchor_blend": True} if architecture == "2" else {}
    _write_config(
        cfg, manifest_path, config_path, architecture=architecture, checkpoint_dir=checkpoint_dir,
        model_param_overrides=overrides,
    )
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["smoke"] is True
    assert summary["architecture"] == architecture
    assert summary["final_step"] == 1
    # smoke mode never writes a real checkpoint -- only the preflight and run manifests.
    assert (checkpoint_dir / "preflight_report.json").is_file()
    assert (checkpoint_dir / "run_manifest.json").is_file()
    assert not (checkpoint_dir / "trainable_weights.pt").is_file()


def test_run_training_smoke_runs_architecture3_with_regional_he(tmp_path, monkeypatch):
    """use_regional_he=True, use_global_gex=True, use_global_slide=False --
    the real, sandbox-testable subset of Architecture 3 (see module
    docstring for why use_global_slide itself is not exercised here)."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_3"
    _write_config(
        cfg, manifest_path, config_path, architecture="3", checkpoint_dir=checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True},
    )
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["architecture"] == "3"


def test_run_training_smoke_runs_architecture4_with_a_fitted_gene_basis(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    gene_names = list(manifest["gene_panel"])
    import numpy as np
    residuals = np.random.default_rng(0).normal(size=(10, len(gene_names))).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    basis_path = tmp_path / "gene_residual_basis.pt"
    save_gene_residual_basis(basis, basis_path)

    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_4"
    _write_config(
        cfg, manifest_path, config_path, architecture="4", checkpoint_dir=checkpoint_dir,
        model_param_overrides={"use_regional_he": True},
        gene_residual_basis_path=str(basis_path),
    )
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["architecture"] == "4"


def test_run_training_requires_gene_residual_basis_for_architecture4(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_4_missing_basis"
    _write_config(cfg, manifest_path, config_path, architecture="4", checkpoint_dir=checkpoint_dir)
    with pytest.raises(ValueError, match="gene_residual_basis"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_non_smoke_requires_synchronized_init_dir(tmp_path, monkeypatch):
    """Mandatory requirement #7: fail closed if a verified synchronized
    initialization cannot be loaded -- a real (non-smoke) run must never
    silently fall back to four independently-random starting points."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_1_no_sync"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    with pytest.raises(ValueError, match="synchronized_init_dir"):
        train_module.run_training(str(config_path), smoke=False)


def test_run_training_non_smoke_loads_verified_synchronized_initialization_and_checkpoints_resume(tmp_path, monkeypatch):
    """Requirement #7 (mandatory synchronized init) + requirement #9
    (checkpointing/resume) exercised together through the real trainer."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)

    # Build a real synchronized-initialization manifest for all four
    # architectures (model_factory.persist_four_architecture_initializations
    # requires exactly the four) -- tiny, CPU-only, matching this test's
    # own small model_param_overrides.
    from gen3_multiscale.models import model_factory as mf
    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    import numpy as np
    residuals = np.random.default_rng(1).normal(size=(10, n_genes)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)

    common = dict(n_genes=n_genes, gex_feature_dim=8, seed=0)
    models = {
        "architecture1": mf.build_architecture({"model": {"architecture": "1", "params": _base_model_params("1")}}, **common),
        "architecture2": mf.build_architecture({"model": {"architecture": "2", "params": _base_model_params("2", use_anchor_blend=True)}}, **common),
        "architecture3": mf.build_architecture({"model": {"architecture": "3", "params": _base_model_params("3", use_regional_he=True, use_global_gex=True)}}, **common),
        "architecture4": mf.build_architecture(
            {"model": {"architecture": "4", "params": _base_model_params("4", use_regional_he=True)}},
            **common, gene_basis=basis, gene_names=gene_names,
        ),
    }
    sync_dir = tmp_path / "synchronized_init"
    mf.persist_four_architecture_initializations(models, sync_dir)

    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_1_resume"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir), checkpoint_every_n_steps=1,
    )
    # A real, tiny non-smoke run: total_steps is read from training_cfg
    # normally, but run_training's smoke branch is the only place that
    # caps it -- so drive a real short run by setting total_steps small
    # in the config directly.
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    summary_first = train_module.run_training(str(config_path), smoke=False)
    assert summary_first["ok"] is True
    assert summary_first["final_step"] == 2
    assert (checkpoint_dir / "trainable_weights.pt").is_file()
    assert (checkpoint_dir / "optimizer_rng_state.pt").is_file()

    config["training"]["total_steps"] = 1  # resume for one more step
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary_second = train_module.run_training(str(config_path), smoke=False)
    assert summary_second["final_step"] == 3


def test_run_training_rejects_incomplete_cache_coverage(tmp_path, monkeypatch):
    """Regression test: a manifest sample whose cache was never built must
    be caught by preflight before any model/optimizer/DataLoader is
    constructed."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch, samples_per_split={"train": 2, "validation": 1, "test": 1})
    # Remove one training sample's spot-feature cache entirely (real cache
    # location: spot_feature_cache.py's _cache_path -- hest_cache_dir/gigapath_gen3_spot_cache/<sample_id>.npz).
    victim = manifest["train_sample_ids"][0]
    victim_cache = Path(str(cfg.data.hest_cache_dir)) / "gigapath_gen3_spot_cache" / f"{victim}.npz"
    assert victim_cache.is_file()
    victim_cache.unlink()

    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_missing_cache"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    with pytest.raises((ValueError, FileNotFoundError)):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_rejects_provenance_disagreement_with_declared_tile_encoder_revision(tmp_path, monkeypatch):
    """Regression test: data.tile_encoder_revision must actually match what
    the caches were built from -- a stale/typo'd config value must not
    silently train against a different tile encoder than declared."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_wrong_revision"
    config = _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    config["data"]["tile_encoder_revision"] = "9" * 40
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="hf_revision"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_never_calls_the_tile_encoder_during_a_smoke_run(tmp_path, monkeypatch):
    """Mandatory requirement #5, exercised at the full trainer level (not
    just gen3_dataset.py's own unit test): the real GigaPath tile encoder
    must be invoked ONLY during the fixture's own cache-building step,
    never again once run_training starts (preflight, dataset construction,
    and the training step must all slice the precomputed feature cache)."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_no_live_encoding"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)

    def _fail(*args, **kwargs):
        raise AssertionError("the tile encoder must never be called by run_training itself")

    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _fail)
    monkeypatch.setattr("src.models.conditioning._gigapath_preprocess_and_encode", _fail)
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True


def test_run_training_rejects_a_manifest_with_zero_train_samples(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    manifest["train_sample_ids"] = []
    save_dataset_manifest(manifest, manifest_path)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_no_train_samples"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    with pytest.raises(ValueError, match="train_sample_ids"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_skips_a_nonfinite_loss_step_instead_of_corrupting_the_model(tmp_path, monkeypatch):
    """Requirement #9's finite-loss safety check, exercised for real: force
    the loss to be NaN on the one smoke step and confirm the optimizer step
    is skipped (no exception, and the summary records the skip) rather than
    silently stepping on a corrupted gradient."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_nonfinite"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)

    real_compute_step_losses = train_module.compute_step_losses

    def _nan_losses(*args, **kwargs):
        losses = real_compute_step_losses(*args, **kwargs)
        losses = dict(losses)
        losses["total"] = losses["total"] * float("nan")
        return losses

    monkeypatch.setattr(train_module, "compute_step_losses", _nan_losses)
    summary = train_module.run_training(str(config_path), smoke=True)
    assert summary["ok"] is True
    assert summary["n_skipped_nonfinite"] == 1
