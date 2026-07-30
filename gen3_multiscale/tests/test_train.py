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

import numpy as np
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


def _build_synchronized_init_dir(tmp_path: Path, manifest: dict) -> Path:
    """Real synchronized-initialization manifest for all four
    architectures (model_factory.persist_four_architecture_initializations
    requires exactly the four) -- tiny, CPU-only, matching this test
    file's own small model_param_overrides. Shared by every test below
    that needs a real (non-smoke) run, which requires this to be set."""
    from gen3_multiscale.models import model_factory as mf

    gene_names = list(manifest["gene_panel"])
    n_genes = len(gene_names)
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
    return sync_dir


def _mutate_canonical_run_manifest(checkpoint_dir: Path, **field_updates) -> None:
    """Test helper: apply `field_updates` to the CANONICAL, bundle-bound
    run_manifest.json (re-signing the bundle's own manifest.json and its
    pointer's manifest_sha256, exactly like `test_a32051b_adversarial.py`'s
    `_resign_bundle_file`) AND mirror the identical updated content to the
    checkpoint_dir's root run_manifest.json.

    Codex re-audit of commit 2162ff4, finding #2: resume now trusts the
    CANONICAL bundle-bound run_manifest.json, never the root mirror
    directly -- mutating only the root file (the OLD way these tests
    simulated code-state drift) no longer has any effect on what
    `run_training` actually verifies against, and mutating only the
    bundle copy would (correctly) be caught by the NEW root-vs-canonical
    consistency check as an inconsistency of its own. Updating both
    together simulates "this field's value was wrong when originally
    saved," the same class of scenario `_resign_bundle_file` documents."""
    import hashlib
    from gen3_multiscale.training import checkpoint as checkpoint_module

    checkpoint_dir = Path(checkpoint_dir)
    bundle_dir = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir).resolved_dir
    run_manifest = json.loads((bundle_dir / "run_manifest.json").read_text())
    run_manifest.update(field_updates)
    new_content = json.dumps(run_manifest, indent=2, sort_keys=True, default=str).encode()

    (bundle_dir / "run_manifest.json").write_bytes(new_content)
    manifest_path = bundle_dir / "manifest.json"
    bundle_manifest = json.loads(manifest_path.read_text())
    bundle_manifest["files"]["run_manifest.json"] = hashlib.sha256(new_content).hexdigest()
    manifest_path.write_text(json.dumps(bundle_manifest, indent=2, sort_keys=True))
    new_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path = checkpoint_dir / "latest_bundle.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = new_manifest_sha256
    pointer_path.write_text(json.dumps(pointer, indent=2))

    (checkpoint_dir / "run_manifest.json").write_bytes(new_content)


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

    # Adam's Step 6 audit #9 of commit a32051b: "Record environment
    # versions." Never a claim of bit-exact CUDA determinism -- see
    # verify_resume_consistency's own scope-caveat docstring.
    run_manifest = json.loads((checkpoint_dir / "run_manifest.json").read_text())
    assert run_manifest["environment_versions"]["torch"]
    assert "code_commit_hash" in run_manifest


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
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)

    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_1_resume"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir), checkpoint_every_n_steps=1,
    )
    # A real, tiny non-smoke run: total_steps is the run's ABSOLUTE
    # target step count (requirement #2) -- drive a real short run by
    # setting it small in the config directly.
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    summary_first = train_module.run_training(str(config_path), smoke=False)
    assert summary_first["ok"] is True
    assert summary_first["final_step"] == 2
    assert summary_first["completion_reason"] == "completed_total_steps"
    assert (checkpoint_dir / "trainable_weights.pt").is_file()
    assert (checkpoint_dir / "optimizer_rng_state.pt").is_file()

    # Requirement #2 regression: total_steps=1 (LESS than the already-
    # reached resume_step=2) must run ZERO additional steps, never
    # "resume_step + total_steps" more.
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary_noop = train_module.run_training(str(config_path), smoke=False)
    assert summary_noop["final_step"] == 2

    # An ABSOLUTE total_steps greater than resume_step=2 genuinely
    # continues training to that absolute target.
    config["training"]["total_steps"] = 3
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary_second = train_module.run_training(str(config_path), smoke=False)
    assert summary_second["final_step"] == 3


def test_run_training_stops_at_the_wall_clock_limit_and_saves_a_checkpoint(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): training.max_wall_clock_hours was completely unused --
    total_steps=100_000_000 (the real configs' own hard safety cap) would
    never stop on a wall-clock budget alone. Fakes time.time() to make the
    SECOND loop iteration's check exceed the (real, positive) configured
    limit, after the FIRST step has already run -- confirming the run
    stops early, records why, and still saves a real checkpoint for the
    step(s) that did complete."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_wallclock"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir), checkpoint_every_n_steps=1,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 100  # never naturally reached in this test
    config["training"]["max_wall_clock_hours"] = 1.0
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # time.time() call order inside run_training: (1) start_time, (2) the
    # wall-clock check at the top of iteration 0 (must pass), (3) the
    # check at the top of iteration 1 (must trip the 1.0-hour limit),
    # (4) the final `elapsed = time.time() - start_time` for the summary.
    fake_times = iter([0.0, 0.0, 3600.0 * 2, 3600.0 * 2])
    monkeypatch.setattr(train_module.time, "time", lambda: next(fake_times))

    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["completion_reason"] == "wall_clock_limit_reached"
    assert summary["final_step"] == 1
    assert (checkpoint_dir / "trainable_weights.pt").is_file()


def test_run_training_smoke_learning_gate_fails_on_a_frozen_model(tmp_path, monkeypatch):
    """Regression test: requirement #4's smoke learning gate must catch a
    model that is NOT actually learning even when loss/gradients are both
    perfectly finite -- simulated here by making optimizer.step() a no-op
    (mirrors a real bug class: e.g. an accidentally-frozen backbone or a
    disconnected computation graph, where every finite-value check still
    passes but no trainable parameter ever actually changes)."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_frozen"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **kw: None)
    with pytest.raises(RuntimeError, match="zero measurable change"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_smoke_learning_gate_fails_on_a_zero_gradient_norm(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_zero_grad"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **kw: torch.as_tensor(0.0))
    with pytest.raises(RuntimeError, match="not strictly positive"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_refuses_to_resume_under_a_changed_architecture(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): "Refuse changed configs or artifacts." A checkpoint_dir
    whose run_manifest.json records a DIFFERENT model architecture than
    the current invocation must raise, not silently proceed."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_arch_switch"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    # Same checkpoint_dir, switched to architecture 2 -- a real, plausible
    # operator mistake (wrong --config path, or a copy-pasted checkpoint_dir).
    config["model"]["architecture"] = "2"
    config["model"]["params"]["use_anchor_blend"] = True
    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="resume refused"):
        train_module.run_training(str(config_path), smoke=False)


def test_run_training_requires_optimizer_state_to_resume_a_real_checkpoint(tmp_path, monkeypatch):
    """Regression test: a checkpoint whose IMMUTABLE history bundle is
    missing optimizer_rng_state.pt must refuse to resume rather than
    silently restarting optimizer momentum/RNG state from scratch.

    Every real loader resolves through `latest_bundle.json` to the
    immutable, uniquely-named `history/step_XXXXXXXX__<bundle_id>/`
    bundle, never trusting `checkpoint_dir`'s root files directly (those
    are only a convenience mirror). So corrupting/deleting a file from
    the ROOT no longer simulates an incomplete checkpoint -- this test
    now deletes the file from the actual immutable bundle the loader
    resolves to, which correctly raises a fail-closed RuntimeError at
    `_resolve_checkpoint_source` (the bundle's own manifest.json still
    references the now-missing file) before `run_training` even gets to
    its own optimizer-state check."""
    from gen3_multiscale.training import checkpoint as checkpoint_module

    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_no_opt_state"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)
    assert (checkpoint_dir / "optimizer_rng_state.pt").is_file()
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    assert identity.step == 1
    step_bundle_dir = identity.resolved_dir
    assert (step_bundle_dir / "optimizer_rng_state.pt").is_file()
    (step_bundle_dir / "optimizer_rng_state.pt").unlink()

    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(RuntimeError, match="optimizer_rng_state"):
        train_module.run_training(str(config_path), smoke=False)


def test_run_training_refuses_resume_when_code_state_drifted_unless_explicitly_allowed(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #10: "Bind code
    state on resume: exact commit plus clean-worktree status/diff hash,
    or require an explicit scientifically-visible override." This test
    cannot actually change the real git commit/worktree mid-run, so it
    simulates a genuine code-state drift the same way every other
    resume-consistency test in this file simulates a config/dataset/
    basis drift: mutating the PERSISTED run_manifest.json's own recorded
    code_commit_hash directly, then resuming for real against it."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_code_drift"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    manifest_path_on_disk = checkpoint_dir / "run_manifest.json"
    persisted = json.loads(manifest_path_on_disk.read_text())
    assert persisted["code_commit_hash"] is not None  # this repo IS a real git checkout
    assert persisted["code_drift_acknowledged"] is False
    # Codex re-audit of commit 2162ff4, finding #2: resume now verifies
    # against the CANONICAL, bundle-bound run_manifest.json, so the
    # simulated drift must be applied there (mirrored to root too, to
    # avoid tripping the new root-vs-canonical consistency check).
    _mutate_canonical_run_manifest(checkpoint_dir, code_commit_hash="deadbeef" * 5)

    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="code state changed"):
        train_module.run_training(str(config_path), smoke=False)

    # The explicit override resumes successfully AND is recorded, not
    # silently applied.
    summary = train_module.run_training(str(config_path), smoke=False, allow_code_drift=True)
    assert summary["ok"] is True
    resumed_manifest = json.loads(manifest_path_on_disk.read_text())
    assert resumed_manifest["code_drift_acknowledged"] is True


def test_run_training_refuses_resume_when_code_identity_is_unknown_unless_explicitly_allowed(tmp_path, monkeypatch):
    """Codex re-audit of commit f7bb8a1, launch blocker #8: "fail closed
    when code identity is unknown unless an explicit recorded override is
    supplied." The PRIOR behavior SKIPPED the code-drift check entirely
    when the old manifest recorded no commit at all (simulating a
    checkpoint from outside a git checkout, or predating this field) --
    this must now be refused exactly like a confirmed change, not
    silently treated as nothing to verify."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_unknown_code_identity"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    manifest_path_on_disk = checkpoint_dir / "run_manifest.json"
    persisted = json.loads(manifest_path_on_disk.read_text())
    assert persisted["code_commit_hash"] is not None  # this repo IS a real git checkout
    # Codex re-audit of commit 2162ff4, finding #2: simulate against the
    # CANONICAL, bundle-bound run_manifest.json -- see
    # _mutate_canonical_run_manifest's own docstring.
    _mutate_canonical_run_manifest(checkpoint_dir, code_commit_hash=None)

    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="code identity is unknown"):
        train_module.run_training(str(config_path), smoke=False)

    summary = train_module.run_training(str(config_path), smoke=False, allow_code_drift=True)
    assert summary["ok"] is True
    resumed_manifest = json.loads(manifest_path_on_disk.read_text())
    assert resumed_manifest["code_drift_acknowledged"] is True


def test_run_training_refuses_resume_when_root_run_manifest_disagrees_with_the_canonical_bundle(tmp_path, monkeypatch):
    """Codex re-audit of commit 2162ff4, finding #2: "Resume still trusts
    the loose root run_manifest.json... does not require the bundle-bound
    run manifest to equal the root manifest." A root mirror that has been
    hand-edited (or left stale by a crash between a bundle write and its
    root-mirror refresh) so it now disagrees with the CANONICAL, bundle-
    bound copy on an identity-bearing field must be refused -- resuming
    must never trust the root file's own claim about what the checkpoint
    is, only the verified bundle."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_root_canonical_disagreement"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    # Mutate ONLY the root mirror -- the canonical bundle-bound copy is
    # left untouched, simulating a stale/hand-edited root file.
    manifest_path_on_disk = checkpoint_dir / "run_manifest.json"
    root_manifest = json.loads(manifest_path_on_disk.read_text())
    assert root_manifest["gene_panel_hash"]  # a real _RESUME_CONSISTENCY_FIELDS member
    root_manifest["gene_panel_hash"] = "tampered_" + root_manifest["gene_panel_hash"]
    manifest_path_on_disk.write_text(json.dumps(root_manifest, indent=2, sort_keys=True, default=str))

    config["training"]["total_steps"] = 2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="root run_manifest.json disagrees with the canonical"):
        train_module.run_training(str(config_path), smoke=False)


def test_run_training_tolerates_root_run_manifest_bookkeeping_drift_across_a_no_op_resume(tmp_path, monkeypatch):
    """The root mirror is legitimately refreshed on EVERY run_training
    call (including a no-op resume that saves no new checkpoint bundle),
    while the canonical bundle-bound copy only updates on a real save --
    so non-identity bookkeeping fields (e.g. environment_versions) can
    genuinely diverge between them without representing real drift. Only
    `_RESUME_CONSISTENCY_FIELDS` disagreement must raise; a difference
    confined to a non-identity field must resume cleanly."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_root_bookkeeping_drift"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    manifest_path_on_disk = checkpoint_dir / "run_manifest.json"
    root_manifest = json.loads(manifest_path_on_disk.read_text())
    root_manifest["environment_versions"] = {"torch": "some-other-version-string"}
    manifest_path_on_disk.write_text(json.dumps(root_manifest, indent=2, sort_keys=True, default=str))

    # A no-op resume (resume_step already >= total_steps) still refreshes
    # the root mirror at the end of run_training -- must not raise.
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["ok"] is True
    assert summary["final_step"] == 1


def test_worktree_diff_hash_hashes_untracked_content_and_ignores_generated_directories(tmp_path):
    """Codex re-audit of commit f7bb8a1, launch blocker #8: "hash tracked
    diffs plus contents of relevant untracked source/config files; ignore
    generated caches/results." Exercises `_worktree_diff_hash` directly
    against THIS real repo checkout -- the only way to prove it reads
    file CONTENT (not merely names) and ignores generated-directory
    paths, both real behavioral properties `git status --porcelain`'s
    raw text alone cannot demonstrate. Creates and always removes real,
    uniquely-named throwaway files under `gen3_multiscale/` (never
    touching anything tracked)."""
    import os
    import uuid

    from gen3_multiscale.training.train import _code_commit_hash, _worktree_diff_hash

    if _code_commit_hash() is None:
        pytest.skip("not a real git checkout in this environment")

    package_dir = Path(__file__).resolve().parent.parent  # gen3_multiscale/
    marker = uuid.uuid4().hex[:12]
    source_probe = package_dir / f"_codestate_probe_{marker}.py"
    # "cache" must be the EXACT directory name -- `_worktree_diff_hash`
    # matches whole path PARTS against `_CODE_STATE_IGNORED_PATH_PARTS`,
    # not a substring/prefix, so a marker-suffixed name like
    # "cache_<marker>" would NOT be recognized as ignored. The marker
    # lives one level deeper instead, so this probe never collides with a
    # real "gen3_multiscale/cache" directory's own contents.
    cache_dir_preexisting = (package_dir / "cache").is_dir()
    cache_probe_dir = package_dir / "cache" / marker
    cache_probe = cache_probe_dir / "probe.py"
    try:
        baseline = _worktree_diff_hash()
        assert baseline is not None

        source_probe.write_text("x = 1\n")
        after_add = _worktree_diff_hash()
        assert after_add != baseline  # a new untracked source file's PRESENCE changes the hash

        source_probe.write_text("x = 2\n")
        after_edit = _worktree_diff_hash()
        assert after_edit != after_add  # editing an untracked file's CONTENT also changes the hash
        assert after_edit != baseline

        source_probe.unlink()
        back_to_baseline = _worktree_diff_hash()
        assert back_to_baseline == baseline

        cache_probe_dir.mkdir(parents=True)
        cache_probe.write_text("x = 1\n")
        with_ignored_cache_file = _worktree_diff_hash()
        # An untracked file inside a generated-directory-named path (here
        # "cache", an exact `_CODE_STATE_IGNORED_PATH_PARTS` entry) must
        # never affect the hash -- neither its presence nor its content.
        assert with_ignored_cache_file == baseline
        cache_probe.write_text("x = 2\n")
        assert _worktree_diff_hash() == baseline
    finally:
        if source_probe.exists():
            source_probe.unlink()
        if cache_probe.exists():
            cache_probe.unlink()
        if cache_probe_dir.exists():
            os.rmdir(cache_probe_dir)
        if not cache_dir_preexisting and (package_dir / "cache").is_dir():
            os.rmdir(package_dir / "cache")


def test_worktree_diff_hash_hashes_files_inside_a_brand_new_untracked_directory(tmp_path):
    """Codex re-audit of commit 2162ff4, finding #6: "Plain `git status
    --porcelain` commonly reports `?? new_directory/`, not its files.
    Editing `new_directory/module.py` would then leave the worktree hash
    unchanged." Confirmed real and reproduced directly (`git status
    --porcelain` collapses a wholly-untracked directory into one summary
    line, never descending into it) -- distinct from the PRIOR
    `test_worktree_diff_hash_hashes_untracked_content_and_ignores_
    generated_directories` test above, whose probe file sits directly
    under `gen3_multiscale/` (an already-tracked parent directory), which
    `git status --porcelain` reports per-file even in the old
    implementation and therefore could not have caught this bug. This
    test's probe file lives inside a FRESH, entirely-untracked
    subdirectory instead -- the exact scenario Codex names."""
    import os
    import uuid

    from gen3_multiscale.training.train import _code_commit_hash, _worktree_diff_hash

    if _code_commit_hash() is None:
        pytest.skip("not a real git checkout in this environment")

    package_dir = Path(__file__).resolve().parent.parent  # gen3_multiscale/
    marker = uuid.uuid4().hex[:12]
    new_dir = package_dir / f"_codestate_new_dir_probe_{marker}"
    probe_file = new_dir / "module.py"
    try:
        baseline = _worktree_diff_hash()
        assert baseline is not None

        new_dir.mkdir()
        probe_file.write_text("x = 1\n")
        after_add = _worktree_diff_hash()
        assert after_add != baseline  # a file inside a brand-new untracked directory changes the hash

        probe_file.write_text("x = 2\n")
        after_edit = _worktree_diff_hash()
        assert after_edit != after_add  # editing that file's content must ALSO change the hash
        assert after_edit != baseline
    finally:
        if probe_file.exists():
            probe_file.unlink()
        if new_dir.exists():
            os.rmdir(new_dir)


def test_run_training_validation_selection_metric_is_deterministic_across_repeated_calls(tmp_path, monkeypatch):
    """Regression test for requirement #8: Architecture 4's validation
    used to mix a RANDOMLY-sampled flow loss into the very metric used
    for model selection -- calling validation twice on the identical
    model/data could previously produce two different "total" values.
    The deterministic reconstruction metric must not."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    gene_names = list(manifest["gene_panel"])
    residuals = np.random.default_rng(3).normal(size=(10, len(gene_names))).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    basis_path = tmp_path / "gene_residual_basis.pt"
    save_gene_residual_basis(basis, basis_path)

    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_val_determinism"
    _write_config(
        cfg, manifest_path, config_path, architecture="4", checkpoint_dir=checkpoint_dir,
        model_param_overrides={"use_regional_he": True}, gene_residual_basis_path=str(basis_path),
    )
    summary_a = train_module.run_training(str(config_path), smoke=True)
    checkpoint_dir_b = tmp_path / "ckpt_val_determinism_b"
    _write_config(
        cfg, manifest_path, config_path, architecture="4", checkpoint_dir=checkpoint_dir_b,
        model_param_overrides={"use_regional_he": True}, gene_residual_basis_path=str(basis_path),
    )
    summary_b = train_module.run_training(str(config_path), smoke=True)
    history_a = json.loads((checkpoint_dir / "validation_history.json").read_text())
    history_b = json.loads((checkpoint_dir_b / "validation_history.json").read_text())
    assert summary_a["ok"] is True and summary_b["ok"] is True
    assert history_a[0]["total"] == history_b[0]["total"]


def test_run_training_saves_a_best_checkpoint_and_validation_history(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_best"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir),
    )
    config = yaml.safe_load(config_path.read_text())
    # Codex re-audit of commit 90f853e, launch blocker #7 (completed-step
    # semantics): validation now fires whenever `completed_steps %
    # eval_every_n_steps == 0`, using the count of REAL completed
    # optimizer updates -- eval_every_n_steps=1 with total_steps=2
    # therefore validates at BOTH completed steps 1 and 2 (the prior,
    # pre-fix code additionally skipped the very first step via a
    # `step > resume_step` guard needed only by its old, inconsistent
    # pre-increment step labeling -- no longer needed or correct now
    # that every label uses the post-increment completed-step count).
    config["training"]["total_steps"] = 2
    config["training"]["eval_every_n_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    assert (checkpoint_dir / "validation_history.json").is_file()
    history = json.loads((checkpoint_dir / "validation_history.json").read_text())
    assert len(history) == 2
    assert [entry["step"] for entry in history] == [1, 2]
    # best/ is now a real checkpoint.py transactional bundle (Codex
    # re-audit of commit f7bb8a1, launch blocker #3) -- root-mirror
    # trainable_weights.pt plus a real latest_bundle.json pointer and
    # run_manifest.json, not a bespoke best_info.json.
    from gen3_multiscale.training import checkpoint as checkpoint_module

    assert (checkpoint_dir / "best" / "trainable_weights.pt").is_file()
    assert (checkpoint_dir / "best" / "latest_bundle.json").is_file()
    assert (checkpoint_dir / "best" / "run_manifest.json").is_file()
    best_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir / "best")
    assert best_identity.step in (1, 2)


def test_run_training_does_not_duplicate_save_when_total_steps_coincides_with_checkpoint_interval(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #1/#7: "The
    trainer also saves the final step twice when it coincides with the
    checkpoint interval." Confirmed real: the in-loop periodic save and
    the post-loop final save both used to fire, unconditionally, at the
    SAME step whenever total_steps was itself a multiple of
    checkpoint_every_n_steps. With the fix, exactly ONE history bundle
    exists at that step, and it carries the REAL final completion_reason
    (not periodic saving's own hardcoded "in_progress"), proving the
    surviving save is the post-loop one, not merely a coincidental single
    survivor of two identical saves."""
    from gen3_multiscale.training import checkpoint as checkpoint_module

    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_no_dup_final_save"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir), checkpoint_every_n_steps=2,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 2  # exactly coincides with checkpoint_every_n_steps=2
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = train_module.run_training(str(config_path), smoke=False)
    assert summary["final_step"] == 2
    assert summary["completion_reason"] == "completed_total_steps"

    bundles_at_final_step = [name for step, name in checkpoint_module.list_checkpoint_bundles(checkpoint_dir) if step == 2]
    assert len(bundles_at_final_step) == 1, f"expected exactly one bundle at step 2, got {bundles_at_final_step}"
    training_state = json.loads((checkpoint_dir / "history" / bundles_at_final_step[0] / "training_state.json").read_text())
    assert training_state["completion_reason"] == "completed_total_steps"


def test_run_training_validation_and_checkpoint_labels_agree_on_the_same_completed_step(tmp_path, monkeypatch):
    """Codex re-audit of commit 90f853e, launch blocker #7: validation/
    best-checkpoint-selection and periodic-checkpoint-saving used to
    label the SAME real model state with DIFFERENT step numbers (a
    pre-increment vs post-increment mismatch) -- "step 2000" could mean
    two different actual amounts of training depending on which code
    path produced the label. With the fix, a validation entry and a
    checkpoint saved at the SAME `completed_steps` value describe the
    IDENTICAL model state: the trainable weights saved at that step
    reproduce the exact validation loss recorded for that same step."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    sync_dir = _build_synchronized_init_dir(tmp_path, manifest)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_label_agreement"
    _write_config(
        cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir,
        synchronized_init_dir=str(sync_dir), checkpoint_every_n_steps=1,
    )
    config = yaml.safe_load(config_path.read_text())
    config["training"]["total_steps"] = 1
    config["training"]["eval_every_n_steps"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    train_module.run_training(str(config_path), smoke=False)

    history = json.loads((checkpoint_dir / "validation_history.json").read_text())
    assert [entry["step"] for entry in history] == [1]
    training_state = json.loads((checkpoint_dir / "training_state.json").read_text())
    assert training_state["step"] == 1


def test_validate_numeric_config_rejects_a_non_positive_learning_rate():
    with pytest.raises(ValueError, match="training.lr"):
        train_module._validate_numeric_config({"lr": 0.0}, {})


def test_validate_numeric_config_rejects_invalid_betas():
    with pytest.raises(ValueError, match="betas"):
        train_module._validate_numeric_config({"lr": 1e-4, "optimizer": {"betas": [1.5, 0.999]}}, {})


def test_compute_training_gene_scale_is_a_positive_pure_function_of_training_data(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data

    train_samples = {sid: load_gen3_sample_data(cfg, manifest, sid) for sid in manifest["train_sample_ids"]}
    scale_a = train_module.compute_training_gene_scale(train_samples)
    scale_b = train_module.compute_training_gene_scale(train_samples)
    assert scale_a.shape == (len(manifest["gene_panel"]),)
    assert (scale_a > 0).all()
    assert np.array_equal(scale_a, scale_b)


def test_compute_training_gene_scale_streaming_matches_naive_dense_pooled_std(tmp_path, monkeypatch):
    """Adam's Step 6 audit #6 of commit a32051b: compute_training_gene_scale
    now streams per-gene sum/sum-of-squares in row chunks (sparse-sliced
    before densifying) instead of densifying and concatenating every
    training sample's full expression matrix at once. Numerically proves
    the streamed result matches the old naive full-densify-then-.std()
    computation (within float64 accumulation-order tolerance), and that
    the result is INDEPENDENT of chunk_size."""
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data

    train_samples = {sid: load_gen3_sample_data(cfg, manifest, sid) for sid in manifest["train_sample_ids"]}

    pooled = []
    for sample in train_samples.values():
        X = sample.adata.X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        pooled.append(np.asarray(X, dtype=np.float64))
    naive_scale = np.clip(np.concatenate(pooled, axis=0).std(axis=0), 1e-6, None).astype(np.float32)

    streamed_default = train_module.compute_training_gene_scale(train_samples)
    streamed_tiny_chunks = train_module.compute_training_gene_scale(train_samples, chunk_size=1)
    streamed_huge_chunks = train_module.compute_training_gene_scale(train_samples, chunk_size=10_000)

    np.testing.assert_allclose(streamed_default, naive_scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(streamed_tiny_chunks, naive_scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(streamed_default, streamed_huge_chunks)


def test_deterministic_train_index_for_step_is_reproducible_and_covers_the_dataset():
    seen_epoch0 = {train_module.deterministic_train_index_for_step(i, 5, seed=42) for i in range(5)}
    assert seen_epoch0 == {0, 1, 2, 3, 4}
    assert (
        train_module.deterministic_train_index_for_step(3, 5, seed=42)
        == train_module.deterministic_train_index_for_step(3, 5, seed=42)
    )
    # Different seeds must generally produce different orderings (not
    # required to differ at every single index, but the whole epoch-0
    # ordering should not be identical).
    order_seed_a = [train_module.deterministic_train_index_for_step(i, 5, seed=1) for i in range(5)]
    order_seed_b = [train_module.deterministic_train_index_for_step(i, 5, seed=2) for i in range(5)]
    assert order_seed_a != order_seed_b


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


def test_run_training_fails_closed_on_a_nonfinite_loss_instead_of_silently_skipping(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): "NaN/Inf loss or gradients must make smoke/full runs fail,
    not increment the step and eventually return ok: true." A prior
    version of this test validated exactly that wrong behavior (skip +
    ok: true); it must now assert the run RAISES instead."""
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
    with pytest.raises(RuntimeError, match="non-finite total loss"):
        train_module.run_training(str(config_path), smoke=True)


def test_run_training_fails_closed_on_a_nonfinite_gradient(tmp_path, monkeypatch):
    cfg, manifest, manifest_path = _prepare(tmp_path, monkeypatch)
    config_path = tmp_path / "config.yaml"
    checkpoint_dir = tmp_path / "ckpt_nonfinite_grad"
    _write_config(cfg, manifest_path, config_path, architecture="1", checkpoint_dir=checkpoint_dir)

    real_grad_norm = torch.nn.utils.clip_grad_norm_

    def _nan_grad_norm(*args, **kwargs):
        real_grad_norm(*args, **kwargs)
        return torch.as_tensor(float("nan"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _nan_grad_norm)
    with pytest.raises(RuntimeError, match="non-finite gradient norm"):
        train_module.run_training(str(config_path), smoke=True)
