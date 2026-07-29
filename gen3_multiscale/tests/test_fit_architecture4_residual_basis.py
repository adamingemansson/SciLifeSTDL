"""Tests for gen3_multiscale/scripts/fit_architecture4_residual_basis.py
-- Adam's Step 6 audit #11 deliverable: the real pipeline that trains
Architecture 3, loads its checkpoint, generates TRAINING-only residuals,
and fits+persists a real gene-residual basis with full provenance.
Also covers train.py::maybe_load_pretrained_conditioner_for_architecture4,
the other half of audit #11 (loading + freezing that exact conditioner
inside a real Architecture 4 instance)."""
from __future__ import annotations

import json

import pytest
import torch
import yaml

from gen3_multiscale.models import model_factory as mf
from gen3_multiscale.scripts.fit_architecture4_residual_basis import (
    compute_training_residuals, fit_and_save_architecture4_basis,
)
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment, step6_model_params, write_step6_train_config
from gen3_multiscale.training import train as train_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule


def _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path) -> "Path":
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    import numpy as np

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    residuals = np.random.default_rng(7).normal(size=(10, n_genes)).astype(np.float32)
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

    arch3_config_path = tmp_path / "architecture3_config.yaml"
    arch3_checkpoint_dir = tmp_path / "arch3_ckpt"
    write_step6_train_config(
        cfg, manifest_path, arch3_config_path, architecture="3", checkpoint_dir=arch3_checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True},
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(arch3_config_path.read_text())
    config["training"]["total_steps"] = 2
    arch3_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = train_module.run_training(str(arch3_config_path), smoke=False)
    assert summary["ok"] is True
    return arch3_config_path, arch3_checkpoint_dir


def test_fit_and_save_architecture4_basis_produces_a_real_basis_with_provenance(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    output_basis_path = tmp_path / "gene_residual_basis.pt"
    provenance = fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(output_basis_path),
        n_masks_per_sample=2, rank=4,
    )
    assert output_basis_path.is_file()
    provenance_path = tmp_path / "gene_residual_basis.pt.provenance.json"
    assert provenance_path.is_file()
    loaded_provenance = json.loads(provenance_path.read_text())
    assert loaded_provenance["kind"] == "gen3_architecture4_residual_basis_provenance"
    assert loaded_provenance["gene_panel_hash"] == provenance["gene_panel_hash"]
    assert loaded_provenance["n_residual_rows"] > 0
    assert loaded_provenance["train_sample_ids"] == sorted(manifest["train_sample_ids"])
    assert loaded_provenance["architecture3_checkpoint_trainable_weights_sha256"] is not None

    from gen3_multiscale.models.gene_basis import load_gene_residual_basis
    basis = load_gene_residual_basis(output_basis_path)
    assert basis.n_genes == len(manifest["gene_panel"])


def test_fit_and_save_architecture4_basis_leaves_no_residual_memmap_file_behind(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #11: residuals
    are now written to a real, disk-backed `numpy.memmap` file rather
    than accumulated in a Python list -- that temp file must be cleaned
    up after fitting completes, not left behind next to the real basis
    output."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    output_basis_path = tmp_path / "gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(output_basis_path), n_masks_per_sample=2, rank=4,
    )
    leftover = list(tmp_path.glob("*.residuals.tmp.*"))
    assert leftover == []


def test_compute_training_residuals_writes_a_real_disk_backed_memmap(tmp_path, monkeypatch):
    import numpy as np

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    from omegaconf import OmegaConf

    from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
    from gen3_multiscale.training.train import build_model_for_inference, expected_tile_encoder_provenance, resolved_config

    config = resolved_config(str(arch3_config_path))
    dataset_manifest = manifest
    train_ids = list(dataset_manifest["train_sample_ids"])
    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, _report = load_and_preflight_samples(cfg_om, dataset_manifest, train_ids, expected_provenance)
    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(dataset_manifest, samples, strata, role="train", n_training_masks_per_sample=2)
    train_dataset = Gen3SpatialFieldDataset(dataset_manifest, samples, schedule, strata)
    gene_names = list(dataset_manifest["gene_panel"])
    model, _info = build_model_for_inference(
        config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=arch3_checkpoint_dir, smoke=False,
        dataset_manifest=dataset_manifest,
    )

    memmap_path = tmp_path / "residuals_test.npy"
    residuals = compute_training_residuals(model, train_dataset, torch.device("cpu"), memmap_path=memmap_path)
    assert isinstance(residuals, np.memmap)
    assert memmap_path.is_file()
    assert residuals.shape[0] > 0
    assert residuals.shape[1] == len(gene_names)
    # The file on disk really holds the same data the in-memory memmap
    # view reports -- proof this is genuinely disk-backed, not merely an
    # in-RAM array wearing a memmap's type.
    reloaded = np.load(memmap_path, mmap_mode="r")
    assert np.array_equal(np.asarray(reloaded), np.asarray(residuals))


def test_fit_and_save_architecture4_basis_rejects_a_non_architecture3_config(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    write_step6_train_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=tmp_path / "ckpt")
    with pytest.raises(ValueError, match="Architecture 3 config"):
        fit_and_save_architecture4_basis(str(config_path), str(tmp_path / "ckpt"), str(tmp_path / "basis.pt"))


def test_maybe_load_pretrained_conditioner_requires_a_real_architecture3_checkpoint_for_non_smoke(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    import numpy as np
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    residuals = np.random.default_rng(2).normal(size=(10, n_genes)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    model = mf.build_architecture(
        {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
        n_genes=n_genes, gex_feature_dim=8, gene_basis=basis, gene_names=gene_names, seed=0,
    )
    with pytest.raises(ValueError, match="architecture3_conditioner_checkpoint"):
        train_module.maybe_load_pretrained_conditioner_for_architecture4(
            model, {"required_fingerprints": {}}, "4", gene_names, smoke=False,
        )
    # smoke is exempt (construction-only smoke, audit #8 of commit a32051b).
    assert train_module.maybe_load_pretrained_conditioner_for_architecture4(
        model, {"required_fingerprints": {}}, "4", gene_names, smoke=True,
    )["loaded"] is False
    # staged smoke (require_for_smoke=True) is NOT exempt -- it must
    # raise exactly like a non-smoke run.
    with pytest.raises(ValueError, match="architecture3_conditioner_checkpoint"):
        train_module.maybe_load_pretrained_conditioner_for_architecture4(
            model, {"required_fingerprints": {}}, "4", gene_names, smoke=True, require_for_smoke=True,
        )


def test_maybe_load_pretrained_conditioner_loads_and_freezes_the_real_conditioner(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    import numpy as np
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    residuals = np.random.default_rng(4).normal(size=(10, n_genes)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    model = mf.build_architecture(
        {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
        n_genes=n_genes, gex_feature_dim=8, gene_basis=basis, gene_names=gene_names, seed=1,
    )
    config = {
        "model": {"architecture": "4", "params": {}},
        "required_fingerprints": {"architecture3_conditioner_checkpoint": str(arch3_checkpoint_dir)},
    }
    conditioner_info = train_module.maybe_load_pretrained_conditioner_for_architecture4(
        model, config, "4", gene_names, smoke=False,
    )
    assert conditioner_info["loaded"] is True
    # Audit #4 of commit a32051b: the exact conditioner checkpoint's
    # identity (sha256 + step) is now returned so callers can bind
    # resume/evaluation to it.
    assert conditioner_info["checkpoint_dir"] == str(arch3_checkpoint_dir)
    assert conditioner_info["checkpoint_sha256"] is not None
    assert conditioner_info["checkpoint_step"] is not None
    assert all(not p.requires_grad for p in model.conditioner.parameters())
    assert any(p.requires_grad for p in model.velocity_network.parameters())


def test_maybe_load_pretrained_conditioner_does_not_freeze_when_configured_off(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    import numpy as np
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    residuals = np.random.default_rng(5).normal(size=(10, n_genes)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    model = mf.build_architecture(
        {"model": {"architecture": "4", "params": step6_model_params("4", use_regional_he=True)}},
        n_genes=n_genes, gex_feature_dim=8, gene_basis=basis, gene_names=gene_names, seed=1,
    )
    config = {
        "model": {"architecture": "4", "params": {"freeze_conditioner_initially": False}},
        "required_fingerprints": {"architecture3_conditioner_checkpoint": str(arch3_checkpoint_dir)},
    }
    train_module.maybe_load_pretrained_conditioner_for_architecture4(model, config, "4", gene_names, smoke=False)
    assert all(p.requires_grad for p in model.conditioner.parameters())
