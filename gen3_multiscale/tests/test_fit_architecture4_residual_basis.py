"""Tests for gen3_multiscale/scripts/fit_architecture4_residual_basis.py
-- Adam's Step 6 audit #11 deliverable: the real pipeline that trains
Architecture 3, loads its checkpoint, generates TRAINING-only residuals,
and fits+persists a real gene-residual basis with full provenance.
Also covers train.py::maybe_load_pretrained_conditioner_for_architecture4,
the other half of audit #11 (loading + freezing that exact conditioner
inside a real Architecture 4 instance)."""
from __future__ import annotations

import json
from pathlib import Path

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
    # Codex re-audit of commit f7bb8a1, launch blocker #5: "Bind the basis
    # sidecar to canonical bundle identity/step, config identity, ...
    # cache content and the complete realized training-mask schedule."
    assert loaded_provenance["architecture3_checkpoint_step"] is not None
    assert loaded_provenance["architecture3_config_identity_fingerprint"]
    assert loaded_provenance["training_mask_schedule_fingerprint"]
    assert loaded_provenance["cache_content_by_sample"]
    assert set(loaded_provenance["cache_content_by_sample"]) == set(manifest["train_sample_ids"])

    from gen3_multiscale.models.gene_basis import load_gene_residual_basis
    basis = load_gene_residual_basis(output_basis_path)
    assert basis.n_genes == len(manifest["gene_panel"])


# ---------------------------------------------------------------------------
# Codex re-audit of commit f7bb8a1, launch blocker #5: "Before residual
# fitting, verify Architecture 3 through the same complete best/latest
# identity pipeline used by evaluation."
# ---------------------------------------------------------------------------

def test_fit_and_save_architecture4_basis_refuses_a_checkpoint_trained_under_a_different_config(tmp_path, monkeypatch):
    """`verify_full_checkpoint_identity` must run BEFORE any residual
    computation -- a checkpoint whose weights hash is internally
    self-consistent but was trained under a DIFFERENT config (here: a
    different masking stratum set than the one this basis-fitting call
    is using) must be refused, not silently accepted."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    config = yaml.safe_load(arch3_config_path.read_text())
    config["masking"]["strata"] = [
        {**stratum, "hole_fraction": min(0.95, float(stratum.get("hole_fraction", 0.3)) + 0.1)}
        for stratum in config["masking"]["strata"]
    ]
    mismatched_config_path = tmp_path / "architecture3_config_mismatched.yaml"
    mismatched_config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    output_basis_path = tmp_path / "gene_residual_basis_mismatched.pt"
    with pytest.raises(ValueError, match="config_identity_fingerprint"):
        fit_and_save_architecture4_basis(
            str(mismatched_config_path), str(arch3_checkpoint_dir), str(output_basis_path),
            n_masks_per_sample=2, rank=4,
        )
    # The fail-closed check runs BEFORE the (expensive) residual pass --
    # no basis or memmap should have been produced at all.
    assert not output_basis_path.is_file()
    assert list(tmp_path.glob("*.residuals.tmp.*")) == []


def test_fit_and_save_architecture4_basis_refuses_a_checkpoint_trained_under_a_different_dataset(tmp_path, monkeypatch):
    """Overwrites the SAME manifest path (config, and therefore
    `config_identity_fingerprint`, is left untouched) with mutated
    content, isolating the `dataset_manifest_fingerprint` check from the
    `config_identity_fingerprint` one exercised above."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)

    from gen3_multiscale.data.dataset_manifest import load_dataset_manifest, save_dataset_manifest

    mutated_manifest = dict(load_dataset_manifest(manifest_path))
    mutated_manifest["gene_panel"] = list(mutated_manifest["gene_panel"])[::-1]
    save_dataset_manifest(mutated_manifest, manifest_path)

    output_basis_path = tmp_path / "gene_residual_basis_mutated_manifest.pt"
    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        fit_and_save_architecture4_basis(
            str(arch3_config_path), str(arch3_checkpoint_dir), str(output_basis_path),
            n_masks_per_sample=2, rank=4,
        )


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


# ---------------------------------------------------------------------------
# Codex re-audit of commit 2162ff4, finding #5: close basis-sidecar
# provenance fail-open gaps -- train_sample_ids mismatch, self-inconsistent
# mask-schedule fingerprint, conditioner bundle_id/manifest_sha256
# mismatch, architecture3 config identity vs the conditioner's own bound
# run_manifest, and numerical basis-content sha256 mismatch.
# ---------------------------------------------------------------------------

def _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path):
    """A real, trained Architecture 4 checkpoint plus the real basis it
    was loaded with -- shared setup for several of the maybe_load_gene_basis
    adversarial tests below."""
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path)
    basis_path = tmp_path / "arch4_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    arch4_config_path = tmp_path / "arch4_config.yaml"
    arch4_checkpoint_dir = tmp_path / "arch4_ckpt"
    write_step6_train_config(
        cfg, manifest_path, arch4_config_path, architecture="4", checkpoint_dir=arch4_checkpoint_dir,
        model_param_overrides={"use_regional_he": True},
        synchronized_init_dir=str((tmp_path / "sync")), gene_residual_basis_path=str(basis_path),
    )
    config = yaml.safe_load(arch4_config_path.read_text())
    config["training"]["total_steps"] = 1
    config["required_fingerprints"]["architecture3_conditioner_checkpoint"] = str(arch3_checkpoint_dir)
    arch4_config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return arch3_config_path, arch3_checkpoint_dir, basis_path, arch4_config_path, arch4_checkpoint_dir


def test_fit_and_save_architecture4_basis_records_train_sample_ids_and_maybe_load_gene_basis_rejects_a_mismatch(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    _arch3_cfg, _arch3_ckpt, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    provenance_path = f"{basis_path}.provenance.json"
    with open(provenance_path) as f:
        provenance = json.load(f)
    assert provenance["train_sample_ids"] == sorted(manifest["train_sample_ids"])

    provenance["train_sample_ids"] = sorted(manifest["train_sample_ids"])[:-1]  # drop one -- a genuine mismatch
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True, default=str)

    with pytest.raises(ValueError, match="different training sample set"):
        train_module.run_training(str(arch4_config_path), smoke=False)


def test_maybe_load_gene_basis_rejects_a_sidecar_whose_recomputed_mask_schedule_fingerprint_disagrees(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #5: 'recompute
    training_mask_schedule_fingerprint from the recorded reports.' A
    sidecar whose two identity records (the fingerprint field and the
    reports it is supposed to describe) disagree with each other must be
    refused, even though there is still no external 'current' value for
    either to be checked against."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    _arch3_cfg, _arch3_ckpt, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    provenance_path = f"{basis_path}.provenance.json"
    with open(provenance_path) as f:
        provenance = json.load(f)
    provenance["training_mask_schedule_fingerprint"] = "0" * 64  # self-inconsistent with mask_schedule_reports
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True, default=str)

    with pytest.raises(ValueError, match="disagree with each other"):
        train_module.run_training(str(arch4_config_path), smoke=False)


def test_maybe_load_gene_basis_rejects_a_basis_file_whose_numerical_content_does_not_match_its_sidecar(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #5: 'record the
    numerical basis SHA256 and verify it when loading.' A basis file
    whose ACTUAL numeric content has been changed (e.g. re-fit differently
    or hand-edited) while its provenance sidecar's recorded
    gene_residual_basis_sha256 was left untouched must be refused."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    _arch3_cfg, _arch3_ckpt, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    from gen3_multiscale.models.gene_basis import load_gene_residual_basis, save_gene_residual_basis

    basis = load_gene_residual_basis(basis_path)
    with torch.no_grad():
        basis.basis.add_(1.0)  # change the numeric content without touching the sidecar
    save_gene_residual_basis(basis, basis_path)

    with pytest.raises(ValueError, match="does not match its provenance sidecar"):
        train_module.run_training(str(arch4_config_path), smoke=False)


def test_maybe_load_gene_basis_rejects_a_conditioner_rolled_to_a_different_bundle_at_the_same_step_and_weights(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #5: 'record and validate
    canonical conditioner bundle_id/manifest SHA/step.' weights_sha256 +
    step alone cannot distinguish two DIFFERENT bundles that happen to
    save byte-identical weights at the same step (e.g. a duplicate save
    with unchanged content) -- re-saves the SAME Architecture 3 weights a
    second time at the SAME step, producing a new, distinct bundle_id/
    manifest_sha256 with an IDENTICAL weights_sha256 and step, then
    checks that resuming Architecture 4 from its recorded basis sidecar
    (bound to the ORIGINAL bundle) refuses to proceed against the new one."""
    from gen3_multiscale.training import checkpoint as checkpoint_module

    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    identity_before = checkpoint_module.resolve_checkpoint_identity(arch3_checkpoint_dir)

    from gen3_multiscale.training.train import build_model_for_inference, resolved_config

    arch3_config = resolved_config(str(arch3_config_path))
    gene_names = list(manifest["gene_panel"])
    model, _info = build_model_for_inference(
        arch3_config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=arch3_checkpoint_dir,
        smoke=False, dataset_manifest=manifest,
    )
    model_config_path = identity_before.resolved_dir / "model_config.json"
    model_config = json.loads(model_config_path.read_text())
    checkpoint_module.save_checkpoint(model, model_config, gene_names, arch3_checkpoint_dir, step=identity_before.step)

    identity_after = checkpoint_module.resolve_checkpoint_identity(arch3_checkpoint_dir)
    assert identity_after.step == identity_before.step
    assert identity_after.weights_sha256 == identity_before.weights_sha256  # unchanged content
    assert identity_after.bundle_dir != identity_before.bundle_dir  # but a genuinely DIFFERENT bundle

    with pytest.raises(ValueError, match="different bundle"):
        train_module.run_training(str(arch4_config_path), smoke=False)


def test_build_model_for_inference_pins_the_conditioner_identity_once_across_basis_validation_and_loading(tmp_path, monkeypatch):
    """Codex re-audit of commit 7a2d819, finding #3: 'Apply the same
    resolve-once principle to Architecture 4's conditioner/basis
    orchestration.' Replaces the Architecture 3 conditioner checkpoint
    with a genuinely DIFFERENT bundle (freshly re-initialized weights,
    different bundle_id) IMMEDIATELY AFTER `build_model_for_inference`'s
    own identity-pinning call resolves it -- i.e. inside construction
    itself, before `maybe_load_gene_basis`'s validation and
    `maybe_load_pretrained_conditioner_for_architecture4`'s weight-load
    run -- by monkeypatching `checkpoint_module.resolve_checkpoint_identity`
    to perform the replacement as a side effect the first (and, since the
    fix resolves exactly once, ONLY) time it is called with the
    conditioner's mutable path. Proves the info dict returned describes
    the ORIGINAL (pre-replacement) bundle throughout -- never mixing in
    the bundle that appeared mid-construction -- and that basis
    validation (which would fail closed on any mismatch) still passed."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    from gen3_multiscale.training import checkpoint as checkpoint_module
    from gen3_multiscale.training.train import build_model_for_inference, resolved_config

    original_identity = checkpoint_module.resolve_checkpoint_identity(arch3_checkpoint_dir)
    gene_names = list(manifest["gene_panel"])

    real_resolve = checkpoint_module.resolve_checkpoint_identity
    replaced = {"done": False}

    def _resolve_and_then_replace(path):
        result = real_resolve(path)
        if not replaced["done"] and Path(path) == Path(arch3_checkpoint_dir):
            replaced["done"] = True
            arch3_config = resolved_config(str(arch3_config_path))
            fresh_model, _info = build_model_for_inference(
                arch3_config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=None,
                smoke=False, dataset_manifest=manifest,
            )
            model_config_path = original_identity.resolved_dir / "model_config.json"
            model_config = json.loads(model_config_path.read_text())
            checkpoint_module.save_checkpoint(
                fresh_model, model_config, gene_names, arch3_checkpoint_dir, step=original_identity.step,
            )
        return result

    monkeypatch.setattr(train_module.checkpoint_module, "resolve_checkpoint_identity", _resolve_and_then_replace)

    arch4_config = resolved_config(str(arch4_config_path))
    model, info = build_model_for_inference(
        arch4_config, gene_names=gene_names, device=torch.device("cpu"), checkpoint_dir=None, smoke=False,
        dataset_manifest=manifest,
    )

    assert replaced["done"] is True  # confirm the mid-construction replacement actually ran
    new_identity = checkpoint_module.resolve_checkpoint_identity(arch3_checkpoint_dir)
    assert new_identity.bundle_dir != original_identity.bundle_dir
    assert new_identity.weights_sha256 != original_identity.weights_sha256

    assert info["architecture3_conditioner"]["checkpoint_bundle_id"] == original_identity.bundle_dir
    assert info["architecture3_conditioner"]["checkpoint_sha256"] == original_identity.weights_sha256


def test_maybe_load_gene_basis_rejects_a_basis_sidecar_whose_cache_content_disagrees_with_the_conditioners_own_canonical_run_manifest(tmp_path, monkeypatch):
    """Codex re-audit of commit 66d65f2, finding #4: 'the basis sidecar's
    training-sample cache identities are required, but they are not
    compared directly against the canonical Architecture 3 run manifest.
    That comparison is available and should be exact.' Tampers with the
    BASIS SIDECAR's own recorded cache_content_by_sample directly (the
    conditioner checkpoint's own canonical run_manifest.json -- and
    therefore its bundle_id/manifest_sha256/weights identity, all
    independently checked elsewhere -- is left completely untouched and
    correct) -- proving the new check catches a sidecar whose recorded
    cache-content provenance simply disagrees with the conditioner's own
    ground truth, a disagreement the caller's-own-preflight-intersection
    comparison alone cannot reliably catch (a caller that never supplies
    cache_content_by_sample, or one whose own preflight coincidentally
    doesn't cover the tampered sample, would previously see nothing wrong)."""
    cfg, manifest, manifest_path = prepare_step6_experiment(tmp_path, monkeypatch)
    arch3_config_path, arch3_checkpoint_dir, basis_path, arch4_config_path, arch4_checkpoint_dir = (
        _train_a_real_architecture4_checkpoint_with_basis(tmp_path, cfg, manifest, manifest_path)
    )
    provenance_path = f"{basis_path}.provenance.json"
    with open(provenance_path) as f:
        provenance = json.load(f)
    a_train_sample_id = manifest["train_sample_ids"][0]
    tampered_content = dict(provenance["cache_content_by_sample"][a_train_sample_id])
    tampered_content["spot_features_content_sha256"] = "0" * 64
    provenance["cache_content_by_sample"][a_train_sample_id] = tampered_content
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True, default=str)

    with pytest.raises(ValueError, match="does not match the configured Architecture 3 conditioner"):
        train_module.run_training(str(arch4_config_path), smoke=False)


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
