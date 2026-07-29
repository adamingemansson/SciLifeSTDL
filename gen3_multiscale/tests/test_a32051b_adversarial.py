"""Adversarial integration tests for commit a32051b's audit round.

Adam's explicit instruction: "Add adversarial integration tests for
A3/A4 using use_global_slide=True, best-checkpoint evaluation, changed
same-gene-order basis contents, swapped conditioner weights, altered
cache contents, interrupted mixed-step checkpoints, and proof that
Architecture 4's reported prediction changes when only its trained flow
weights change."

Real, small, end-to-end synthetic data throughout (the established
session-wide discipline); only the GigaPath TILE encoder is ever
monkeypatched (`_step6_fixtures.py::stub_gigapath`).

`use_global_slide=True` is the one scenario NOT exercised here: it
requires the real, external `gigapath` package (LongNet) plus a real
Prov-GigaPath slide-encoder checkpoint
(`models/slide_encoder.py::FrozenGigaPathSlideEncoder.__init__` does
`import gigapath.slide_encoder` and raises ImportError without it) --
neither is available in this sandbox. This is the SAME documented
limitation `test_train.py`'s own module docstring already states
("validated separately on real hardware... not re-exercised here"); it
has not changed this round. Every other adversarial scenario the audit
asked for is exercised for real below, including the closely related
`use_regional_he=True` dense-WSI path, which needs no external package.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gen3_multiscale.evaluation.gen3_evaluator import evaluate_gen3_checkpoint
from gen3_multiscale.models import model_factory as mf
from gen3_multiscale.models.architectures import Architecture4
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, save_gene_residual_basis
from gen3_multiscale.scripts.fit_architecture4_residual_basis import fit_and_save_architecture4_basis
from gen3_multiscale.tests._step6_fixtures import (
    prepare_step6_experiment, step6_model_params, write_step6_train_config,
)
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training import train as train_module
from gen3_multiscale.tests.test_architectures import _MODEL_KWARGS, _gene_basis_for, _synthetic_inputs

_prepare = prepare_step6_experiment
_write_config = write_step6_train_config


def _build_synchronized_init_dir(tmp_path, manifest):
    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
    residuals = np.random.default_rng(1).normal(size=(10, n_genes)).astype(np.float32)
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


def _train_a_real_architecture3_checkpoint(tmp_path, cfg, manifest, manifest_path, sync_dir, *, name: str, seed: int):
    """A real, trained Architecture 3 checkpoint -- `seed` changes the
    real per-step training-data draw order (`deterministic_train_index_for_step`),
    so two calls with different seeds produce genuinely DIFFERENT
    trainable_weights.pt content even though both start from the
    IDENTICAL synchronized initialization."""
    config_path = tmp_path / f"{name}_config.yaml"
    checkpoint_dir = tmp_path / name
    write_step6_train_config(
        cfg, manifest_path, config_path, architecture="3", checkpoint_dir=checkpoint_dir,
        model_param_overrides={"use_regional_he": True, "use_global_gex": True},
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 3
    config["training"]["seed"] = seed
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True
    return config_path, checkpoint_dir


def _write_architecture4_config(tmp_path, cfg, manifest_path, *, name, checkpoint_dir, sync_dir, basis_path, arch3_checkpoint_dir):
    config_path = tmp_path / f"{name}_config.yaml"
    write_step6_train_config(
        cfg, manifest_path, config_path, architecture="4", checkpoint_dir=checkpoint_dir,
        model_param_overrides={"use_regional_he": True},
        synchronized_init_dir=str(sync_dir), gene_residual_basis_path=str(basis_path),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config["required_fingerprints"]["architecture3_conditioner_checkpoint"] = str(arch3_checkpoint_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


# ---------------------------------------------------------------------------
# (g) Architecture 4's reported prediction genuinely depends on the trained
# flow weights, not merely the frozen conditioner -- direct behavioral proof
# for audit #1.
# ---------------------------------------------------------------------------

def test_architecture4_predictive_mean_changes_when_only_flow_weights_change():
    """Two Architecture4 instances share the EXACT same conditioner (and
    start with identical velocity_network weights too, from the same
    construction seed); only model_b's velocity_network is then
    perturbed. With the SAME generator seed on the SAME inputs, the
    frozen deterministic_mean must be IDENTICAL between the two, but the
    REPORTED prediction (predictive_mean, what audit #1 requires
    evaluation/validation/overfit to actually use) must differ -- proof
    the sampled prediction genuinely flows through the trained flow
    apparatus, not just the conditioner. Also proves the reproducibility
    half of audit #1: the SAME model with the SAME generator seed gives
    the EXACT SAME predictive mean every time (required for exact
    resume/evaluation consistency)."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)

    torch.manual_seed(0)
    model_a = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, n_flow_samples=4, n_ode_steps=4, **_MODEL_KWARGS,
    )
    torch.manual_seed(0)
    model_b = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, n_flow_samples=4, n_ode_steps=4, **_MODEL_KWARGS,
    )
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)

    with torch.no_grad():
        for p in model_b.velocity_network.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    model_a.eval()
    model_b.eval()
    out_a = model_a.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(42))
    out_b = model_b.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(42))

    assert torch.allclose(out_a["deterministic_mean"], out_b["deterministic_mean"])
    assert not torch.allclose(out_a["predictive_mean"], out_b["predictive_mean"])

    out_a_repeat = model_a.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(42))
    assert torch.equal(out_a["predictive_mean"], out_a_repeat["predictive_mean"])


# ---------------------------------------------------------------------------
# (c) A gene-residual basis re-fit to DIFFERENT numeric content but the SAME
# gene ordering (same gene_names_hash) must be caught -- audit #4's "exact
# numeric basis SHA256" binding.
# ---------------------------------------------------------------------------

def test_resume_refuses_a_basis_with_the_same_gene_order_but_different_numeric_content(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    gene_names = list(manifest["gene_panel"])

    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_for_basis_swap", seed=0,
    )

    # A REAL basis, fit via the real pipeline (writes both the .pt AND
    # its .provenance.json sidecar -- mandatory for a non-smoke run since
    # Codex's re-audit of commit 90f853e, launch blocker #3).
    basis_path = tmp_path / "gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    from gen3_multiscale.models.gene_basis import load_gene_residual_basis
    basis_a = load_gene_residual_basis(basis_path)

    checkpoint_dir = tmp_path / "arch4_basis_swap"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_basis_swap", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True

    # A DIFFERENT basis, fit from different residuals -- same gene_names
    # (hence same gene_names_hash), near-certainly different numeric
    # content (different random residuals -> different singular vectors).
    # Only the .pt is swapped -- its ORIGINAL (still structurally valid,
    # still correctly identifying this same dataset/gene-panel/conditioner)
    # provenance sidecar is left untouched, exactly the scenario where a
    # basis file was overwritten out-of-band without re-fitting it
    # through the real pipeline.
    basis_b = fit_gene_residual_basis(
        np.random.default_rng(99).normal(size=(20, len(gene_names))).astype(np.float32), gene_names, rank=4,
    )
    assert basis_b.gene_names_hash == basis_a.gene_names_hash
    assert not torch.allclose(basis_a.basis, basis_b.basis)
    save_gene_residual_basis(basis_b, basis_path)  # overwrite the SAME path

    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="gene_residual_basis_sha256"):
        train_module.run_training(str(config_path), smoke=False)


# ---------------------------------------------------------------------------
# (d) Architecture 4's conditioner checkpoint swapped for a DIFFERENT,
# validly-structured Architecture 3 checkpoint must be caught -- audit #4's
# "exact Architecture 3 conditioner weights/step" binding.
# ---------------------------------------------------------------------------

def test_resume_refuses_a_swapped_architecture3_conditioner_checkpoint(tmp_path, monkeypatch):
    """The realistic "swapped conditioner weights" scenario: the SAME
    `required_fingerprints.architecture3_conditioner_checkpoint` PATH now
    holds DIFFERENT, genuinely-trained weights (e.g. someone re-ran
    Architecture 3 training into that directory) -- config identity is
    UNCHANGED (same path string), so this specifically exercises the
    `architecture3_conditioner_checkpoint_sha256` binding, not merely the
    coarser `config_identity_fingerprint` check a changed PATH would also
    trip."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    gene_names = list(manifest["gene_panel"])

    _cfg_a, arch3_checkpoint_dir_a = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_a", seed=0,
    )
    _cfg_b, arch3_checkpoint_dir_b = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_b", seed=12345,
    )
    weights_a = (arch3_checkpoint_dir_a / "trainable_weights.pt").read_bytes()
    weights_b = (arch3_checkpoint_dir_b / "trainable_weights.pt").read_bytes()
    assert weights_a != weights_b  # genuinely different trained checkpoints

    # A REAL basis, fit against arch3_checkpoint_dir_a via the real
    # pipeline (writes both the .pt AND its .provenance.json sidecar --
    # mandatory for a non-smoke run since Codex's re-audit of commit
    # 90f853e, launch blocker #3).
    basis_path = tmp_path / "gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(_cfg_a), str(arch3_checkpoint_dir_a), str(basis_path), n_masks_per_sample=2, rank=4,
    )

    checkpoint_dir = tmp_path / "arch4_conditioner_swap"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_conditioner_swap", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir_a,
    )
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True

    # Overwrite the SAME path (arch3_checkpoint_dir_a) with checkpoint
    # B's real, differently-trained weights -- both the root mirror AND
    # the underlying transactional bundle are updated together via a
    # real save_checkpoint call, exactly like a real re-run would.
    replacement_model = mf.build_architecture(
        {"model": {"architecture": "3", "params": step6_model_params("3", use_regional_he=True, use_global_gex=True)}},
        n_genes=len(gene_names), gex_feature_dim=8, seed=0,
    )
    checkpoint_module.verify_gene_names(arch3_checkpoint_dir_b, gene_names)
    checkpoint_module.load_trainable_state(replacement_model, arch3_checkpoint_dir_b)
    checkpoint_module.save_checkpoint(
        replacement_model, {"name": "arch3_swap"}, gene_names, arch3_checkpoint_dir_a, step=999,
    )
    assert (arch3_checkpoint_dir_a / "trainable_weights.pt").read_bytes() != weights_a

    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # Two independent layers now catch this, and the newer one (Codex's
    # re-audit of commit 90f853e, launch blocker #3: the basis provenance
    # sidecar's OWN recorded conditioner-checkpoint identity, checked
    # inside maybe_load_gene_basis) fires FIRST, before run_training even
    # reaches build_run_manifest/verify_resume_consistency's
    # architecture3_conditioner_checkpoint_sha256 comparison -- both are
    # real, correct, fail-closed catches of the exact same swap.
    with pytest.raises(ValueError, match="architecture3_conditioner_checkpoint_sha256|different Architecture 3 checkpoint"):
        train_module.run_training(str(config_path), smoke=False)


# ---------------------------------------------------------------------------
# (f) Interrupted mixed-step checkpoint: corrupting a file's CONTENT inside
# an immutable history bundle must be caught by fail-closed transactional
# loading -- audit #5.
# ---------------------------------------------------------------------------

def test_checkpoint_load_refuses_a_bundle_file_corrupted_after_writing(tmp_path):
    """Simulates a crash/corruption that leaves a checkpoint step bundle
    with a file that no longer matches its own recorded manifest hash --
    e.g. a partial write, bit rot, or a mid-step interruption that wrote
    SOME bytes of one file before the process died. `save_checkpoint`
    itself is atomic (a single os.replace materializes the whole bundle
    at once), so this test corrupts the bundle DIRECTLY, after the fact,
    to prove the LOADER -- not just the writer -- is fail-closed."""
    import torch.nn as nn

    model = nn.Linear(4, 4)
    checkpoint_dir = tmp_path / "ckpt_corrupted"
    checkpoint_module.save_checkpoint(model, {"name": "tiny"}, ["g1", "g2", "g3", "g4"], checkpoint_dir, step=1)

    step_dir = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir).resolved_dir
    assert step_dir.is_dir()
    # Corrupt trainable_weights.pt's CONTENT in place -- the file still
    # exists (so this is NOT the "missing file" case already covered by
    # test_train.py's own optimizer-state regression test), but its bytes
    # no longer match the manifest's recorded sha256.
    (step_dir / "trainable_weights.pt").write_bytes(b"not a real torch checkpoint, corrupted")

    with pytest.raises(RuntimeError, match="does not match the sha256"):
        checkpoint_module.load_trainable_state(model, checkpoint_dir)
    with pytest.raises(RuntimeError, match="does not match the sha256"):
        checkpoint_module.verify_gene_names(checkpoint_dir, ["g1", "g2", "g3", "g4"])


# ---------------------------------------------------------------------------
# (b)/(e) best-checkpoint evaluation over an altered/tampered bundle must be
# refused before any weights are loaded -- audit #3's "complete, verifiable
# inference bundle" + audit #4/#7's pre-load verification.
# ---------------------------------------------------------------------------

def _train_a_real_checkpoint_with_best(tmp_path, cfg, manifest, manifest_path, sync_dir):
    config_path = tmp_path / "arch1_eval_config.yaml"
    checkpoint_dir = tmp_path / "arch1_eval_ckpt"
    write_step6_train_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    # total_steps=2, not 1: run_training's validation gate is `step >
    # resume_step` (strictly greater), checked BEFORE `step` is
    # incremented -- the very first step (step=0=resume_step) never
    # validates, so total_steps=1 would never produce a best/ bundle at
    # all. Two steps guarantees the second one (step=1 > resume_step=0)
    # runs a real validation and writes best/.
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True
    # best/ is a real checkpoint.py transactional bundle now (Codex
    # re-audit of commit f7bb8a1, launch blocker #3).
    assert (checkpoint_dir / "best" / "latest_bundle.json").is_file()
    assert (checkpoint_dir / "best" / "run_manifest.json").is_file()
    return config_path, checkpoint_dir


def _resign_bundle_file(bundle_dir, filename: str, new_content: bytes) -> None:
    """Test helper: overwrite one file INSIDE a resolved checkpoint
    bundle with new content and re-sign the bundle's own manifest.json
    (and its pointer's manifest_sha256) to match -- simulates "this
    field's value was WRONG when the checkpoint was originally saved"
    (e.g. a hypothetical bug in build_run_manifest), which is a
    DIFFERENT adversarial scenario than bytes altered after writing
    (already covered by test_checkpoint_load_refuses_a_bundle_file_
    corrupted_after_writing) -- that scenario is deliberately NOT
    re-signed and is correctly refused by the hash check alone."""
    import hashlib as _hashlib
    import json as _json

    bundle_dir = Path(bundle_dir)
    (bundle_dir / filename).write_bytes(new_content)
    manifest_path = bundle_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["files"][filename] = _hashlib.sha256(new_content).hexdigest()
    manifest_path.write_text(_json.dumps(manifest, indent=2, sort_keys=True))
    new_manifest_sha256 = _hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checkpoint_dir = bundle_dir.parent.parent  # bundle_dir == checkpoint_dir/history/<bundle_name>
    pointer_path = checkpoint_dir / "latest_bundle.json"
    pointer = _json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = new_manifest_sha256
    pointer_path.write_text(_json.dumps(pointer, indent=2))


def test_evaluate_gen3_checkpoint_refuses_a_best_bundle_whose_weights_were_altered_after_writing(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path, checkpoint_dir = _train_a_real_checkpoint_with_best(tmp_path, cfg, manifest, manifest_path, sync_dir)

    # A correctly-selected best/ bundle evaluates cleanly first (proves
    # this ISN'T just refusing everything).
    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    assert report["n_items"] > 0

    # Now mutate best/'s weights file AFTER it was written -- exactly
    # audit #4's "altered cache/checkpoint contents" adversarial scenario,
    # applied to the checkpoint bundle itself. Must corrupt the RESOLVED
    # bundle's own copy, not checkpoint_dir/best's root mirror -- the
    # root mirror is a disposable convenience copy (real, independent
    # bytes since the hardlink fix) that no real loader ever trusts.
    bundle_dir = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best").resolved_dir
    weights_path = bundle_dir / "trainable_weights.pt"
    original = weights_path.read_bytes()
    weights_path.write_bytes(original + b"\x00")

    # RuntimeError, not ValueError: this now fails inside checkpoint.py's
    # own resolve_checkpoint_identity (exact-weights verification),
    # BEFORE verify_full_checkpoint_identity ever reaches its own
    # identity-field comparisons -- an earlier, more fundamental check.
    with pytest.raises(RuntimeError, match="does not match the sha256"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)


def test_evaluate_gen3_checkpoint_refuses_a_best_bundle_selected_under_a_different_dataset(tmp_path, monkeypatch):
    """Audit #7: "Verify the checkpoint run manifest against evaluation
    inputs before loading." A best/ bundle whose recorded
    dataset_manifest_fingerprint does not match the dataset actually
    being evaluated against must be refused, even though every per-file
    hash inside the bundle is internally consistent."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path, checkpoint_dir = _train_a_real_checkpoint_with_best(tmp_path, cfg, manifest, manifest_path, sync_dir)

    import json

    bundle_dir = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best").resolved_dir
    run_manifest = json.loads((bundle_dir / "run_manifest.json").read_text())
    run_manifest["dataset_manifest_fingerprint"] = "deadbeef" * 8
    _resign_bundle_file(bundle_dir, "run_manifest.json", json.dumps(run_manifest, indent=2, sort_keys=True).encode())

    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)


# ---------------------------------------------------------------------------
# Codex re-audit of commit 90f853e, launch blocker #9: "Architecture 4
# interval coverage/calibration" -- proof the evaluator computes a REAL
# calibration summary from a real, trained Architecture 4 checkpoint's
# actual predictive_std, not merely `{"n_values": 0}` (already covered
# for Architectures 1-3 in test_gen3_evaluator.py).
# ---------------------------------------------------------------------------

def test_evaluate_gen3_checkpoint_reports_real_architecture4_calibration(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)

    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_for_calibration", seed=0,
    )
    basis_path = tmp_path / "calibration_gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    checkpoint_dir = tmp_path / "arch4_for_calibration"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_for_calibration", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True

    report = evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2, use_best=False)
    calibration = report["architecture4_calibration"]
    assert calibration["n_values"] > 0
    for key in ("z_mean", "z_std", "coverage_68", "coverage_90", "coverage_95"):
        assert key in calibration and np.isfinite(calibration[key])
    # Coverage must be monotonically non-decreasing with the nominal
    # interval width -- a REAL property of any well-formed calibration
    # summary, regardless of whether THIS particular undertrained model
    # happens to be well-calibrated.
    assert calibration["coverage_68"] <= calibration["coverage_90"] <= calibration["coverage_95"]
    assert "predictive_std_mean" in report["per_arm_patient_aggregated_metrics"]["model"]
    # Codex re-audit of commit f7bb8a1, secondary fix #4: the Gaussian
    # summary is now explicitly labeled an approximation, and no
    # empirical-quantile summary is computed unless calibration_n_samples
    # was actually requested.
    assert calibration["method"] == "gaussian_std_approximation"
    assert report["architecture4_empirical_calibration"] == {"n_values": 0}


# ---------------------------------------------------------------------------
# Codex re-audit of commit f7bb8a1, launch blocker #7 + secondary fix #4:
# stable-identity evaluation seeding (never raw item index) and a
# higher-fidelity empirical-quantile calibration alternative to the
# Gaussian-std approximation, with a caller-configurable sample count.
# ---------------------------------------------------------------------------

def test_evaluate_gen3_checkpoint_is_reproducible_and_seed_dependent(tmp_path, monkeypatch):
    """Two evaluations of the SAME checkpoint with the SAME evaluation_seed
    must reproduce bit-identical per-item Architecture 4 predictions
    (stable-key seeding, not wall-clock/process-order-dependent); a
    DIFFERENT evaluation_seed must change at least one item's prediction
    (proof the seed is actually threaded through, not silently ignored)."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)

    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_for_seed_stability", seed=0,
    )
    basis_path = tmp_path / "seed_stability_gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    checkpoint_dir = tmp_path / "arch4_for_seed_stability"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_for_seed_stability", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True

    def _predictions(evaluation_seed: int) -> list[float]:
        report = evaluate_gen3_checkpoint(
            str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2, use_best=False,
            evaluation_seed=evaluation_seed,
        )
        return [record["model"]["rmse"] for record in report["per_item_records"]]

    seed0_first = _predictions(0)
    seed0_second = _predictions(0)
    assert seed0_first == seed0_second

    seed1 = _predictions(1)
    assert seed1 != seed0_first


def test_evaluate_gen3_checkpoint_empirical_calibration_respects_configured_sample_count(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)

    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_for_empirical_calibration", seed=0,
    )
    basis_path = tmp_path / "empirical_calibration_gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    checkpoint_dir = tmp_path / "arch4_for_empirical_calibration"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_for_empirical_calibration", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True

    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2, use_best=False,
        calibration_n_samples=50,
    )
    empirical = report["architecture4_empirical_calibration"]
    assert empirical["method"] == "empirical_quantiles"
    assert empirical["n_samples_per_item"] == 50
    assert empirical["n_values"] > 0
    assert empirical["coverage_68"] <= empirical["coverage_90"] <= empirical["coverage_95"]
    # A real (never Gaussian-assumed) empirical fraction must lie in [0, 1].
    for key in ("coverage_68", "coverage_90", "coverage_95"):
        assert 0.0 <= empirical[key] <= 1.0
    # The Gaussian approximation is still reported alongside it, labeled.
    assert report["architecture4_calibration"]["method"] == "gaussian_std_approximation"


# ---------------------------------------------------------------------------
# Codex re-audit of commit 90f853e, remaining requested adversarial coverage:
# missing basis provenance fields, use_best=True with best/ absent, a
# validly-regenerated cache with different content but identical encoder
# provenance, common random numbers across checkpoint steps, and a REAL
# staged Architecture 4 smoke loading the actual canonical Architecture 3
# bundle.
# ---------------------------------------------------------------------------

def test_maybe_load_gene_basis_refuses_a_provenance_sidecar_missing_a_required_field(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_missing_field", seed=0,
    )
    basis_path = tmp_path / "gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    provenance_path = tmp_path / "gene_residual_basis.pt.provenance.json"
    import json
    provenance = json.loads(provenance_path.read_text())
    assert provenance["architecture3_config_fingerprint"]
    del provenance["architecture3_config_fingerprint"]  # OMIT a required field
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))

    checkpoint_dir = tmp_path / "arch4_missing_field"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_missing_field", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="missing required field 'architecture3_config_fingerprint'"):
        train_module.run_training(str(config_path), smoke=False)


def test_maybe_load_gene_basis_refuses_a_basis_whose_recorded_checkpoint_step_does_not_match(tmp_path, monkeypatch):
    """Codex re-audit of commit f7bb8a1, launch blocker #5: "Bind the
    basis sidecar to canonical bundle identity/step... Validate all of
    those when Architecture 4 loads the basis." A basis whose recorded
    `architecture3_checkpoint_trainable_weights_sha256` happens to still
    match (a real, if contrived, scenario: an operator points
    `required_fingerprints.architecture3_conditioner_checkpoint` at a
    checkpoint_dir that has since moved on to a LATER step with the SAME
    weights hash recorded under a stale/rolled-back `latest_bundle.json`)
    must still be refused once its recorded STEP disagrees."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_step_mismatch", seed=0,
    )
    basis_path = tmp_path / "gene_residual_basis_step_mismatch.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    provenance_path = tmp_path / "gene_residual_basis_step_mismatch.pt.provenance.json"
    import json
    provenance = json.loads(provenance_path.read_text())
    real_step = provenance["architecture3_checkpoint_step"]
    assert real_step is not None
    provenance["architecture3_checkpoint_step"] = int(real_step) + 1000  # a step that never happened
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))

    checkpoint_dir = tmp_path / "arch4_step_mismatch"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_step_mismatch", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="different training step of the conditioner"):
        train_module.run_training(str(config_path), smoke=False)


def test_maybe_load_gene_basis_refuses_a_basis_fit_against_different_cache_content(tmp_path, monkeypatch):
    """Launch blocker #5's other new binding: per-sample cache content.
    A basis whose recorded `cache_content_by_sample` disagrees with the
    CURRENT run's own cache content for a sample both runs share must be
    refused -- proof the basis is not silently trusted against a
    regenerated/stale cache."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_cache_mismatch", seed=0,
    )
    basis_path = tmp_path / "gene_residual_basis_cache_mismatch.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    provenance_path = tmp_path / "gene_residual_basis_cache_mismatch.pt.provenance.json"
    import json
    provenance = json.loads(provenance_path.read_text())
    recorded_cache = provenance["cache_content_by_sample"]
    assert recorded_cache
    a_sample_id = next(iter(recorded_cache))
    recorded_cache[a_sample_id] = {**recorded_cache[a_sample_id], "spot_features_content_sha256": "deadbeef" * 8}
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True))

    checkpoint_dir = tmp_path / "arch4_cache_mismatch"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_cache_mismatch", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="cache content"):
        train_module.run_training(str(config_path), smoke=False)


def test_evaluate_gen3_checkpoint_use_best_true_raises_when_best_bundle_is_absent(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #4: "use_best=True
    must fail when best is absent." Trains a real checkpoint (which does
    produce a real best/ bundle -- validation now fires from the very
    first completed step), then removes best/ to construct the "genuinely
    absent" case directly, rather than relying on best/ never having
    been created at all."""
    import shutil

    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "arch1_no_best_config.yaml"
    checkpoint_dir = tmp_path / "arch1_no_best_ckpt"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True
    assert (checkpoint_dir / "best").exists()
    shutil.rmtree(checkpoint_dir / "best")

    with pytest.raises(ValueError, match="use_best=True but no best/ bundle exists"):
        evaluate_gen3_checkpoint(str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2)
    # use_best=False (the latest checkpoint) still evaluates cleanly.
    report = evaluate_gen3_checkpoint(
        str(config_path), checkpoint_dir, split="validation", n_masks_per_sample=2, use_best=False,
    )
    assert report["n_items"] > 0


def test_resume_refuses_a_regenerated_spot_feature_cache_with_different_content_but_identical_provenance(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #5: "Bind every
    per-sample spot-feature ... cache content digest ... into the run
    manifest, and compare them on resume. Encoder provenance alone is
    insufficient." Simulates a cache validly REGENERATED by the exact
    same pinned tile encoder (so `tile_encoder_provenance` -- repo/
    revision/weights sha256 -- is byte-identical) but producing different
    real feature VALUES for at least one spot (e.g. a non-deterministic
    encoding bug, or an accidental wrong-sample overwrite) -- the prior
    resume-consistency check (encoder provenance only) would have missed
    this entirely; `cache_content_fingerprint` must not."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "arch1_cache_drift_config.yaml"
    checkpoint_dir = tmp_path / "arch1_cache_drift_ckpt"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    # Regenerate ONE sample's spot-feature cache with different feature
    # VALUES but the identical tile-encoder provenance fields -- exactly
    # the "validly regenerated cache, different content, same encoder
    # identity" adversarial scenario.
    cache_dir = cfg.data.get("gen3_spot_feature_cache_dir") or (Path(cfg.data.hest_cache_dir) / "gigapath_gen3_spot_cache")
    sample_id = manifest["train_sample_ids"][0]
    cache_path = Path(str(cache_dir)) / f"{sample_id}.npz"
    assert cache_path.is_file()
    cached = dict(np.load(cache_path, allow_pickle=False))
    cached["features"] = cached["features"] + 1.0  # different real values, same shape/provenance
    np.savez(cache_path, **cached)

    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="cache_content_fingerprint"):
        train_module.run_training(str(config_path), smoke=False)


def test_common_random_validation_seed_is_independent_of_training_step():
    """Codex re-audit of commit 90f853e, launch blocker #6, refined by
    the re-audit of commit f7bb8a1's secondary fix #1: the SAME held-out
    item must get the SAME sampled noise regardless of which training
    step is currently being validated, OR of the item's storage/
    enumeration position -- direct unit proof for the extracted
    `common_random_validation_seed` helper (previously inline, and
    previously keyed on item_index rather than a stable content key)."""
    key_a = "S0:small:deadbeef"
    seed_at_early_step_context = train_module.common_random_validation_seed(seed=42, stable_key=key_a)
    seed_at_late_step_context = train_module.common_random_validation_seed(seed=42, stable_key=key_a)
    assert seed_at_early_step_context == seed_at_late_step_context  # no step argument at all -- structurally step-independent
    # Different items get different seeds (not a constant function).
    key_b = "S0:small:cafef00d"
    assert train_module.common_random_validation_seed(42, key_a) != train_module.common_random_validation_seed(42, key_b)
    # Different runs (different training.seed) get different seeds for
    # the SAME item too.
    assert train_module.common_random_validation_seed(42, key_a) != train_module.common_random_validation_seed(43, key_a)
    # The key is the item's CONTENT identity, not its storage/enumeration
    # position -- reordering two items (swapping which index each lives
    # at) must not reassign their noise draws to each other.
    assert (
        train_module.common_random_validation_seed(42, "S0:small:aaa")
        != train_module.common_random_validation_seed(42, "S1:small:bbb")
    )


def test_staged_architecture4_smoke_loads_the_real_canonical_architecture3_bundle(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #8: "staged
    Architecture 4 smoke loading the actual canonical Architecture 3
    bundle." `train.py --smoke --staged-smoke` (`run_training(smoke=True,
    staged_smoke=True)`) must behave like a non-smoke run for the
    conditioner -- it must load the REAL, already-trained Architecture 3
    checkpoint, never skip it the way a plain construction-only `--smoke`
    does."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    arch3_config_path, arch3_checkpoint_dir = _train_a_real_architecture3_checkpoint(
        tmp_path, cfg, manifest, manifest_path, sync_dir, name="arch3_for_staged_smoke", seed=0,
    )
    basis_path = tmp_path / "staged_smoke_gene_residual_basis.pt"
    fit_and_save_architecture4_basis(
        str(arch3_config_path), str(arch3_checkpoint_dir), str(basis_path), n_masks_per_sample=2, rank=4,
    )
    checkpoint_dir = tmp_path / "arch4_staged_smoke"
    config_path = _write_architecture4_config(
        tmp_path, cfg, manifest_path, name="arch4_staged_smoke", checkpoint_dir=checkpoint_dir,
        sync_dir=sync_dir, basis_path=basis_path, arch3_checkpoint_dir=arch3_checkpoint_dir,
    )

    # A plain smoke never even requires the conditioner checkpoint to be
    # configured -- prove staged_smoke is DIFFERENT by first removing it
    # and confirming a plain smoke still runs fine. A DIFFERENT
    # checkpoint_dir than config_path's own -- these two configs
    # deliberately differ (no conditioner configured at all), and
    # run_training writes run_manifest.json even for a smoke run, so
    # reusing the same checkpoint_dir would trip the (correct, and
    # already covered elsewhere) resume-consistency check instead of
    # exercising what THIS test is actually about.
    config = yaml.safe_load(config_path.read_text())
    plain_smoke_config_path = tmp_path / "arch4_plain_smoke_config.yaml"
    plain_smoke_checkpoint_dir = tmp_path / "arch4_plain_smoke_ckpt"
    plain_smoke_config = dict(config)
    plain_smoke_config["training"] = dict(config["training"])
    plain_smoke_config["training"]["checkpoint_dir"] = str(plain_smoke_checkpoint_dir)
    plain_smoke_config["required_fingerprints"] = dict(config["required_fingerprints"])
    plain_smoke_config["required_fingerprints"]["architecture3_conditioner_checkpoint"] = None
    plain_smoke_config_path.write_text(yaml.safe_dump(plain_smoke_config, sort_keys=False))
    plain_summary = train_module.run_training(str(plain_smoke_config_path), smoke=True, staged_smoke=False)
    assert plain_summary["ok"] is True

    # staged_smoke=True with NO conditioner configured must raise, exactly
    # like a non-smoke run -- proving it is not merely a relabeled plain
    # smoke.
    with pytest.raises(ValueError, match="architecture3_conditioner_checkpoint"):
        train_module.run_training(str(plain_smoke_config_path), smoke=True, staged_smoke=True)

    # And with the REAL conditioner configured, staged_smoke=True succeeds
    # and genuinely loads it (never skips loading the way plain --smoke does).
    staged_summary = train_module.run_training(str(config_path), smoke=True, staged_smoke=True)
    assert staged_summary["ok"] is True
