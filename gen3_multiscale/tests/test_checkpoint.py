import json
import os
import random
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gen3_multiscale.training.checkpoint import (
    save_checkpoint, load_trainable_state, load_training_state,
    list_checkpoint_history, list_checkpoint_bundles, rollback_checkpoint,
    load_optimizer_and_rng_state, verify_gene_names, resolve_checkpoint_identity,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        frozen = nn.Linear(4, 4)
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.frozen = frozen


class _AllFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad_(False)


class _FrozenWithBuffer(nn.Module):
    """Zero trainable parameters (like `_AllFrozen`), but a real
    non-frozen buffer registered directly on the top-level module (never
    under `self.frozen`, so `_is_frozen_backbone_module` never excludes
    it) -- `_expected_trainable_state_names` is therefore non-empty
    (`{"some_buffer"}`) even though `trainable_names` is empty."""
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.register_buffer("some_buffer", torch.zeros(3))


def test_checkpoint_round_trip_preserves_trainable_weights_only():
    m = _Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(3.14)
        m.frozen.weight.fill_(2.71)
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2"], tmp, step=100)
        m2 = _Tiny()
        assert not torch.allclose(m2.lin.weight, m.lin.weight)
        load_trainable_state(m2, tmp)
        assert torch.allclose(m2.lin.weight, m.lin.weight)
        # frozen weights must NOT be saved/restored -- a fresh instance's
        # frozen submodule stays at its own fresh random init
        assert not torch.allclose(m2.frozen.weight, m.frozen.weight)
        assert load_training_state(tmp)["step"] == 100


def test_checkpoint_zero_trainable_params_still_saves_metadata():
    """Real bug this session (original codebase): skipping ALL checkpoint
    files whenever there was nothing trainable made a fully-frozen model
    impossible to reconstruct later. Metadata must survive even then."""
    import os

    m = _AllFrozen()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "frozen"}, [], tmp, step=0)
        assert not os.path.exists(os.path.join(tmp, "trainable_weights.pt"))
        assert os.path.exists(os.path.join(tmp, "model_config.json"))
        assert os.path.exists(os.path.join(tmp, "gene_names.json"))
        m2 = _AllFrozen()
        load_trainable_state(m2, tmp)  # must be a clean no-op, not crash


def test_load_trainable_state_raises_on_genuine_mismatch():
    """If a model architecture DOES have trainable params but the weights
    file is missing, that's a real incomplete-checkpoint bug, not a
    frozen-by-design case -- must raise loudly, not silently proceed with
    random weights.

    Uses pytest.raises(RuntimeError), not a bare try/except AssertionError
    (2026-07-28, real bug fixed -- 6th Codex re-audit of commit 06f5cce):
    load_trainable_state used to raise a Python `assert`, which
    `python -O` strips entirely, silently turning this fail-closed check
    into a no-op. It now raises RuntimeError explicitly. The OLD version
    of this test was also itself broken independently of that: its
    `assert False, "..."` fallback for the "did not raise" case raised
    the SAME AssertionError type the except block caught, so the test
    could never actually fail even if load_trainable_state stopped
    raising anything at all."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        # write only the metadata a save_checkpoint call for an ALL-FROZEN
        # model would produce, but load it onto a model that DOES have
        # trainable params
        import json
        import os

        with open(os.path.join(tmp, "model_config.json"), "w") as f:
            json.dump({"name": "tiny"}, f)
        with open(os.path.join(tmp, "gene_names.json"), "w") as f:
            json.dump([], f)
        with pytest.raises(RuntimeError, match="looks incomplete"):
            load_trainable_state(m, tmp)


def test_checkpoint_history_is_pruned_to_keep_last():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        for step in (0, 100, 200, 300):
            with torch.no_grad():
                m.lin.weight.fill_(float(step))
            save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=step, keep_last=2)
        # only the two most recent snapshots survive
        assert list_checkpoint_history(tmp) == [200, 300]
        # root ("latest") always reflects the most recent save regardless of pruning
        assert load_training_state(tmp)["step"] == 300


def test_checkpoint_history_pruning_disabled_when_keep_last_zero():
    """Adam's Step 6 audit #5 of commit a32051b made checkpoints
    transactional: a step's history bundle (history/step_XXXXXXXX/) is
    now ALWAYS created, since every real loader resolves through it via
    latest_bundle.json -- keep_last<=0 now means "prune nothing" (matching
    _prune_history's own contract), not "no history/transactionality at
    all" (the old, pre-audit-#5 meaning)."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=0, keep_last=0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1, keep_last=0)
        assert list_checkpoint_history(tmp) == [0, 1]
        assert os.path.exists(os.path.join(tmp, "model_config.json"))
        assert os.path.exists(os.path.join(tmp, "latest_bundle.json"))


def test_rollback_checkpoint_restores_an_earlier_known_good_snapshot():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=3)
        with torch.no_grad():
            m.lin.weight.fill_(999.0)  # simulate a later step that diverged/corrupted
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=200, keep_last=3)

        rollback_checkpoint(tmp, step=100)
        assert load_training_state(tmp)["step"] == 100
        restored = _Tiny()
        load_trainable_state(restored, tmp)
        assert torch.allclose(restored.lin.weight, torch.full((4, 4), 1.0))


def test_rollback_checkpoint_raises_on_unknown_step():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=3)
        try:
            rollback_checkpoint(tmp, step=999)
            assert False, "expected a ValueError listing the real available steps"
        except ValueError as e:
            assert "999" in str(e) and "100" in str(e)


def test_history_snapshots_survive_root_being_overwritten():
    """Hard-linking means the root file at a given step gets a NEW inode on
    the next save (via os.replace) -- the history snapshot's hard link must
    still point at the OLD data, not silently become the new step's data."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=5)
        with torch.no_grad():
            m.lin.weight.fill_(2.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=200, keep_last=5)

        bundle_name_100 = next(name for step, name in list_checkpoint_bundles(tmp) if step == 100)
        snapshot_100 = _Tiny()
        load_trainable_state(snapshot_100, os.path.join(tmp, "history", bundle_name_100))
        assert torch.allclose(snapshot_100.lin.weight, torch.full((4, 4), 1.0))

        latest = _Tiny()
        load_trainable_state(latest, tmp)
        assert torch.allclose(latest.lin.weight, torch.full((4, 4), 2.0))


def test_optimizer_and_rng_state_round_trips_when_saved():
    """GPT-audit-flagged bug (2026-07-27, confirmed and fixed): checkpoints
    used to save ONLY model weights -- no AdamW momentum, no RNG state --
    so a "resumed" run was a warm restart with reset optimization
    dynamics, not a real continuation."""
    m = _Tiny()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # take a real step so the optimizer actually has non-zero momentum/
    # variance state to round-trip (a freshly-constructed optimizer's
    # state dict is empty and would pass trivially even with a broken
    # save/load path)
    loss = m.lin(torch.randn(2, 4)).sum()
    loss.backward()
    optimizer.step()
    saved_momentum = optimizer.state_dict()["state"][0]["exp_avg"].clone()

    loop_rng = random.Random(0)
    loop_rng.random()  # advance state away from the fresh seed=0 state
    draw_before_save = loop_rng.random()

    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1, optimizer=optimizer, rng=loop_rng)

        m2 = _Tiny()
        optimizer2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
        loop_rng2 = random.Random(999)  # deliberately different seed pre-load
        loaded = load_optimizer_and_rng_state(optimizer2, tmp, rng=loop_rng2)

        assert loaded is True
        assert torch.allclose(optimizer2.state_dict()["state"][0]["exp_avg"], saved_momentum)
        # loop_rng2 must now continue exactly where loop_rng left off, not
        # where its own (different) seed would have taken it
        assert loop_rng2.random() == loop_rng.random()


def test_load_optimizer_and_rng_state_is_a_no_op_for_a_checkpoint_saved_without_one():
    m = _Tiny()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)  # no optimizer= passed

        assert load_optimizer_and_rng_state(optimizer, tmp) is False


def test_verify_gene_names_passes_silently_when_panels_match():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2"], tmp, step=1)
        verify_gene_names(tmp, ["g1", "g2"])  # must not raise


def test_verify_gene_names_raises_on_a_reordered_panel():
    """GPT-audit-flagged bug (2026-07-27, second-pass re-audit): a resumed
    run rebuilt the model from a freshly-derived gene panel and loaded
    weights onto it with no check that this panel matched the checkpoint's
    -- trainable_weights.pt is positional, not gene-name-keyed, so a same-
    width but reordered panel would load without error and silently
    corrupt the run."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2", "g3"], tmp, step=1)
        with pytest.raises(ValueError, match="different gene panel"):
            verify_gene_names(tmp, ["g1", "g3", "g2"])


def test_verify_gene_names_raises_when_gene_names_json_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="gene_names.json"):
            verify_gene_names(tmp, ["g1"])


# ---------------------------------------------------------------------------
# Codex re-audit of commit 90f853e, requested adversarial coverage for
# launch blockers #1/#2: crash-safety of the transactional bundle scheme.
# ---------------------------------------------------------------------------

def test_crash_before_pointer_write_leaves_the_prior_checkpoint_fully_loadable():
    """Simulates a crash between a NEW bundle finishing its atomic
    `os.replace` into history/ and `latest_bundle.json` being updated to
    point at it (save_checkpoint's own pointer-update line never runs).
    The OLD pointer -- and therefore the OLD checkpoint -- must remain
    exactly as loadable as before; a crash here must never corrupt or
    lose the PRIOR good state, only fail to advance to the new one."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2"], tmp, step=1, keep_last=1)
        pointer_before = json.loads((Path(tmp) / "latest_bundle.json").read_text())

        # Simulate the crash: write a SECOND bundle directly (mirroring
        # what save_checkpoint's own staging+os.replace does), but never
        # touch latest_bundle.json -- exactly "crash before pointer
        # replacement".
        with torch.no_grad():
            m.lin.weight.fill_(2.0)
        history_dir = Path(tmp) / "history"
        crashed_bundle = history_dir / "step_00000002__crashed_bundle"
        crashed_bundle.mkdir(parents=True)
        torch.save(m.state_dict(), crashed_bundle / "trainable_weights.pt")
        (crashed_bundle / "model_config.json").write_text(json.dumps({"name": "tiny"}))
        (crashed_bundle / "gene_names.json").write_text(json.dumps(["g1", "g2"]))
        (crashed_bundle / "training_state.json").write_text(json.dumps({"step": 2}))

        pointer_after = json.loads((Path(tmp) / "latest_bundle.json").read_text())
        assert pointer_after == pointer_before  # untouched by the "crashed" write

        m2 = _Tiny()
        load_trainable_state(m2, tmp)  # resolves through the UNCHANGED pointer
        assert torch.allclose(m2.lin.weight, torch.full_like(m2.lin.weight, 1.0))
        assert load_training_state(tmp)["step"] == 1
        identity = resolve_checkpoint_identity(tmp)
        assert identity.step == 1


def test_crash_after_pointer_write_before_pruning_leaves_checkpoint_intact_with_keep_last_one():
    """Simulates a crash AFTER `latest_bundle.json` is durably updated to
    the new bundle but BEFORE `_prune_history` runs (pruning is the LAST
    step of save_checkpoint) -- the checkpoint itself must be fully
    valid and loadable even though the stale bundle from the previous
    step was never cleaned up."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        import gen3_multiscale.training.checkpoint as checkpoint_module

        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1, keep_last=1)

        original_prune = checkpoint_module._prune_history

        def _crash_during_prune(*args, **kwargs):
            raise RuntimeError("simulated crash during pruning")

        checkpoint_module._prune_history = _crash_during_prune
        try:
            with torch.no_grad():
                m.lin.weight.fill_(2.0)
            with pytest.raises(RuntimeError, match="simulated crash during pruning"):
                save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=2, keep_last=1)
        finally:
            checkpoint_module._prune_history = original_prune

        # The pointer update (and the new bundle it points at) completed
        # BEFORE pruning was ever attempted -- the checkpoint is fully
        # valid and resolves to step 2's real weights, despite pruning
        # having "crashed."
        m2 = _Tiny()
        load_trainable_state(m2, tmp)
        assert torch.allclose(m2.lin.weight, torch.full_like(m2.lin.weight, 2.0))
        assert load_training_state(tmp)["step"] == 2
        # Both bundles still exist -- pruning never got to run.
        assert list_checkpoint_history(tmp) == [1, 2]


def test_repeated_save_at_the_same_step_is_handled_safely():
    """"Handle repeated saves at the same step safely" (launch blocker
    #1) -- e.g. a training loop that retries a step after a transient
    failure. Two saves at step=5 with DIFFERENT weight content must both
    succeed (never collide/overwrite each other destructively), and the
    checkpoint must end up reflecting the SECOND (most recent) save."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=5, keep_last=2)
        with torch.no_grad():
            m.lin.weight.fill_(2.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=5, keep_last=2)

        bundles_at_step_5 = [name for step, name in list_checkpoint_bundles(tmp) if step == 5]
        assert len(bundles_at_step_5) == 2  # two DISTINCT bundles, neither overwritten
        assert len(set(bundles_at_step_5)) == 2  # genuinely different bundle_ids

        m2 = _Tiny()
        load_trainable_state(m2, tmp)
        assert torch.allclose(m2.lin.weight, torch.full_like(m2.lin.weight, 2.0))
        identity = resolve_checkpoint_identity(tmp)
        assert identity.step == 5
        assert identity.bundle_dir == bundles_at_step_5[-1]  # the most RECENT of the two


# ---------------------------------------------------------------------------
# Codex re-audit of commit 90f853e, requested adversarial coverage: manifest
# omitting a required file, and the root mirror disagreeing with the
# canonical bundle.
# ---------------------------------------------------------------------------

def test_bundle_with_a_file_omitted_from_its_own_manifest_is_refused():
    """A file that physically exists inside a bundle but is NOT listed in
    that bundle's own manifest.json's "files" dict would otherwise never
    be hash-checked at all -- a tampered-but-structurally-valid
    replacement for it would load silently. `_resolve_checkpoint_source`
    must cross-check the manifest against the bundle's REAL contents,
    not merely validate whatever the manifest happens to list."""
    import hashlib

    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        bundle_dir = identity.resolved_dir
        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "gene_names.json" in manifest["files"]
        del manifest["files"]["gene_names.json"]  # OMIT it, without touching the real file
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        # Simulates the realistic version of this bug -- save_checkpoint
        # itself failing to record one file in "files" from the start
        # (rather than a hostile post-hoc edit) -- by keeping the OUTER
        # pointer's manifest_sha256 self-consistent with the (buggy)
        # manifest, so the test actually exercises the NEW per-bundle
        # "does the manifest account for every real file" cross-check,
        # not the separate, already-covered outer-pointer-hash check.
        pointer_path = Path(tmp) / "latest_bundle.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2))

        m2 = _Tiny()
        with pytest.raises(RuntimeError, match="not listed in its own manifest"):
            load_trainable_state(m2, tmp)


def test_root_mirror_disagreeing_with_the_canonical_bundle_is_ignored_by_every_real_loader():
    """checkpoint_dir's root-level files are ONLY EVER a convenience
    mirror -- every real loader resolves through latest_bundle.json to
    the immutable history bundle instead. Corrupting the ROOT mirror
    directly (simulating a crash mid-refresh, or direct tampering) must
    have NO effect on what actually gets loaded."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)

        # Corrupt the root mirror's weights file directly -- the bundle
        # underneath (and the pointer referencing it) are untouched.
        (Path(tmp) / "trainable_weights.pt").write_bytes(b"not a real torch checkpoint at all")

        m2 = _Tiny()
        load_trainable_state(m2, tmp)  # must succeed, reading through the bundle, not the corrupted root
        assert torch.allclose(m2.lin.weight, torch.full_like(m2.lin.weight, 1.0))
        identity = resolve_checkpoint_identity(tmp)
        assert identity.weights_sha256 is not None
        # The identity's own weights_sha256 must be computed from the
        # BUNDLE's file, never the corrupted root mirror.
        import hashlib
        assert identity.weights_sha256 == hashlib.sha256((identity.resolved_dir / "trainable_weights.pt").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Codex re-audit of commit 2162ff4, finding #3: "checkpoint schema
# validation remains partial" -- pointer schema/version/keys, bundle_dir
# constrained to the canonical bundle-name regex, bundle manifest
# version/kind/bundle_id, unexpected model-state keys, and verify-before-
# pointer-change on rollback. Each of these was a real gap: the fields
# existed on disk (written since the transactional scheme was first
# built) but were never independently READ/validated by any loader.
# ---------------------------------------------------------------------------

def _pointer_path(tmp: str) -> Path:
    return Path(tmp) / "latest_bundle.json"


def _read_pointer(tmp: str) -> dict:
    return json.loads(_pointer_path(tmp).read_text())


def _write_pointer(tmp: str, pointer: dict) -> None:
    _pointer_path(tmp).write_text(json.dumps(pointer, indent=2))


def test_pointer_with_an_extra_key_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        pointer = _read_pointer(tmp)
        pointer["unexpected_extra_field"] = "surprise"
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="unrecognized schema"):
            load_trainable_state(_Tiny(), tmp)


def test_pointer_with_a_missing_key_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        pointer = _read_pointer(tmp)
        del pointer["manifest_sha256"]
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="unrecognized schema"):
            load_trainable_state(_Tiny(), tmp)


def test_pointer_with_an_unsupported_schema_version_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        pointer = _read_pointer(tmp)
        pointer["version"] = 999
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="unsupported schema version"):
            load_trainable_state(_Tiny(), tmp)


def test_pointer_with_a_malformed_manifest_sha256_is_rejected():
    """Not merely "truthy" -- a garbage string that happens to be
    non-empty must still be refused (the old check only tested for
    truthiness, accepting any non-empty string as a "valid" hash)."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        pointer = _read_pointer(tmp)
        pointer["manifest_sha256"] = "not-a-real-sha256"
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="not a 64-hex-character"):
            load_trainable_state(_Tiny(), tmp)


def test_pointer_bundle_dir_naming_a_path_traversal_shaped_target_is_rejected():
    """A tampered pointer's `bundle_dir` must be validated against the
    canonical bundle-name schema BEFORE it is ever joined onto the
    history directory to build a filesystem path -- real path-traversal-
    shaped input like `../../etc` must never reach `Path.__truediv__`
    unchecked."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        pointer = _read_pointer(tmp)
        pointer["bundle_dir"] = "../../etc"
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="canonical bundle-name schema"):
            load_trainable_state(_Tiny(), tmp)


def test_bundle_manifest_with_wrong_version_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        import hashlib

        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = 999
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer = _read_pointer(tmp)
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="unsupported manifest schema"):
            load_trainable_state(_Tiny(), tmp)


def test_bundle_manifest_with_wrong_kind_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        import hashlib

        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["kind"] = "some_other_kind"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer = _read_pointer(tmp)
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="unrecognized kind"):
            load_trainable_state(_Tiny(), tmp)


def test_bundle_manifest_with_a_bundle_id_disagreeing_with_its_own_directory_name_is_rejected():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        import hashlib

        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["bundle_id"] = "step_00000001__99999999999999999999_1_000000"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer = _read_pointer(tmp)
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        _write_pointer(tmp, pointer)
        with pytest.raises(RuntimeError, match="disagrees about its own identity"):
            load_trainable_state(_Tiny(), tmp)


def test_load_trainable_state_rejects_a_weights_blob_with_an_unexpected_extra_key():
    """`strict=False` in `model.load_state_dict` silently DROPS any key
    the current model has no matching parameter/buffer for -- a saved
    blob carrying an extra, unrecognized key (corrupted save, or a
    checkpoint from a since-changed architecture) must now be refused
    explicitly instead of loading with the extra key silently ignored."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        weights_path = identity.resolved_dir / "trainable_weights.pt"
        state = torch.load(weights_path, map_location="cpu")
        state["lin.some_key_this_architecture_never_had"] = torch.zeros(2, 2)
        torch.save(state, weights_path)  # tamper the bundle file directly (bypasses the hash check on purpose)
        # Re-sign the manifest/pointer so the tampered CONTENT is the
        # thing under test, not merely the (already-covered) hash-mismatch
        # check -- mirrors test_a32051b_adversarial.py's _resign_bundle_file pattern.
        import hashlib

        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["trainable_weights.pt"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer = _read_pointer(tmp)
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        _write_pointer(tmp, pointer)

        with pytest.raises(RuntimeError, match="unexpected key"):
            load_trainable_state(_Tiny(), tmp)


def test_rollback_verifies_the_target_bundle_before_touching_the_pointer_or_root_files():
    """Codex re-audit of commit 2162ff4, finding #3: "verify rollback
    target completely before changing the pointer." A corrupted rollback
    target must raise BEFORE root files are overwritten and BEFORE
    latest_bundle.json is repointed -- the currently-active checkpoint
    must remain exactly as loadable as before the failed rollback
    attempt."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        import hashlib

        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=3)
        with torch.no_grad():
            m.lin.weight.fill_(2.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=200, keep_last=3)
        pointer_before = _read_pointer(tmp)

        bundle_name_100 = next(name for step, name in list_checkpoint_bundles(tmp) if step == 100)
        target_weights = Path(tmp) / "history" / bundle_name_100 / "trainable_weights.pt"
        target_weights.write_bytes(b"corrupted, not a real torch checkpoint")

        with pytest.raises(RuntimeError, match="does not match the sha256 recorded"):
            rollback_checkpoint(tmp, step=100)

        # Untouched: still points at step 200, still loads step 200's weights.
        assert _read_pointer(tmp) == pointer_before
        assert load_training_state(tmp)["step"] == 200
        restored = _Tiny()
        load_trainable_state(restored, tmp)
        assert torch.allclose(restored.lin.weight, torch.full((4, 4), 2.0))


# ---------------------------------------------------------------------------
# Codex re-audit of commit 66d65f2, finding #2: "load_trainable_state()
# rejects missing trainable parameters and unexpected keys, but not
# missing expected buffers such as target_gene_scale, gene-basis buffers,
# or coordinate frequencies. A damaged checkpoint could therefore
# silently retain freshly initialized buffers." Exercises REAL
# architectures (not the toy _Tiny fixture) so the deleted keys are real,
# registered buffers (`GeneValueTransportHead.target_gene_scale`,
# `Architecture4._gene_basis_matrix`), not synthetic stand-ins.
# ---------------------------------------------------------------------------

def _small_architecture1():
    from gen3_multiscale.models import model_factory as mf

    params = {
        "image_feature_dim": 1536, "hidden_dim": 16, "n_heads": 2, "n_blocks": 1,
        "dense_threshold": 256, "sparse_k": 10, "chunk_size": 1024, "max_boundary_size": None,
        "transport_heads": 2, "transport_temperature": 1.0, "gene_gate_mode": "per_gene",
        "use_query_gate": True, "use_residual": False, "residual_rank": 4,
        "use_anchor_blend": False, "use_regional_he": False, "use_global_gex": False,
        "use_global_slide": False, "global_slide_dim": 16, "n_gex_inducing": 4,
        "harmonic_k_neighbors": 4, "gene_encoder_type": "weighted_linear", "init_seed": 0,
    }
    return mf.build_architecture(
        {"model": {"architecture": "1", "params": params}}, n_genes=6, gex_feature_dim=8, seed=0,
    )


def test_load_trainable_state_rejects_a_checkpoint_missing_the_target_gene_scale_buffer():
    """`target_gene_scale` is a real, non-frozen, non-trainable buffer
    (`GeneValueTransportHead.register_buffer`) -- present in
    `_expected_trainable_state_names` (buffers are included, not just
    trainable parameters) but NOT in `named_parameters(requires_grad=True)`,
    so the OLD `missing_trainable` check (trainable-parameters-only) could
    never have caught its absence."""
    model = _small_architecture1()
    assert "transport_head.target_gene_scale" in dict(model.named_buffers())
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(model, {"name": "arch1"}, [f"g{i}" for i in range(6)], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        weights_path = identity.resolved_dir / "trainable_weights.pt"
        state = torch.load(weights_path, map_location="cpu")
        assert "transport_head.target_gene_scale" in state
        del state["transport_head.target_gene_scale"]
        torch.save(state, weights_path)
        import hashlib
        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["trainable_weights.pt"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer_path = Path(tmp) / "latest_bundle.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2))

        reloaded = _small_architecture1()
        with pytest.raises(RuntimeError, match="missing expected trainable parameter"):
            load_trainable_state(reloaded, tmp)


def test_load_trainable_state_rejects_a_checkpoint_missing_the_gene_basis_matrix_buffer():
    """`_gene_basis_matrix` (Architecture4's own fixed low-rank gene-
    residual-basis buffer) -- a second, structurally different real
    buffer than `target_gene_scale`, confirming the fix generalizes."""
    from gen3_multiscale.models import model_factory as mf
    from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
    import numpy as np

    gene_names = [f"g{i}" for i in range(6)]
    residuals = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    basis = fit_gene_residual_basis(residuals, gene_names, rank=4)
    params = {
        "image_feature_dim": 1536, "hidden_dim": 16, "n_heads": 2, "n_blocks": 1,
        "dense_threshold": 256, "sparse_k": 10, "chunk_size": 1024, "max_boundary_size": None,
        "transport_heads": 2, "transport_temperature": 1.0, "gene_gate_mode": "per_gene",
        "use_query_gate": True, "use_residual": False, "residual_rank": 4,
        "use_regional_he": False, "use_global_slide": False, "global_slide_dim": 16,
        "n_gex_inducing": 4, "harmonic_k_neighbors": 4, "gene_encoder_type": "weighted_linear",
        "init_seed": 0, "n_flow_blocks": 1, "n_flow_samples": 2, "n_ode_steps": 2, "gene_basis_rank": 4,
    }
    model = mf.build_architecture(
        {"model": {"architecture": "4", "params": params}}, n_genes=6, gex_feature_dim=8, seed=0,
        gene_basis=basis, gene_names=gene_names,
    )
    assert "_gene_basis_matrix" in dict(model.named_buffers())
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(model, {"name": "arch4"}, gene_names, tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        weights_path = identity.resolved_dir / "trainable_weights.pt"
        state = torch.load(weights_path, map_location="cpu")
        assert "_gene_basis_matrix" in state
        del state["_gene_basis_matrix"]
        torch.save(state, weights_path)
        import hashlib
        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["trainable_weights.pt"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer_path = Path(tmp) / "latest_bundle.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2))

        reloaded = mf.build_architecture(
            {"model": {"architecture": "4", "params": params}}, n_genes=6, gex_feature_dim=8, seed=1,
            gene_basis=basis, gene_names=gene_names,
        )
        with pytest.raises(RuntimeError, match="missing expected trainable parameter"):
            load_trainable_state(reloaded, tmp)


# ---------------------------------------------------------------------------
# Codex re-audit of commit 66d65f2, minor finding: the bundle-name regex
# only accepted exactly eight step digits, while the configured safety
# ceiling (training.total_steps: 100000000 in every committed config)
# is itself nine digits; negative steps were never explicitly rejected
# at save time either.
# ---------------------------------------------------------------------------

def test_save_checkpoint_accepts_a_nine_digit_step():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100_000_000, keep_last=1)
        identity = resolve_checkpoint_identity(tmp)
        assert identity.step == 100_000_000
        m2 = _Tiny()
        load_trainable_state(m2, tmp)  # must resolve/load without a bundle-name-regex rejection


def test_save_checkpoint_rejects_a_negative_step_explicitly():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="non-negative"):
            save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=-1)
        # Nothing was written -- a failed save must leave no partial state.
        assert list_checkpoint_history(tmp) == []


def test_resolve_checkpoint_identity_recovers_step_and_bundle_id_from_an_already_resolved_bundle_dir():
    """Codex re-audit of commit 57f0e3c (surfaced while wiring the
    orchestrator's resolve-once-and-pin discipline across stage
    boundaries): resolving an ALREADY-RESOLVED bundle directory directly
    (no `latest_bundle.json` pointer AT that level -- `checkpoint.py`'s
    own documented "caller directly resolves a HISTORY BUNDLE'S OWN
    path" exception) previously always returned `step`/`bundle_dir`/
    `manifest_sha256` as None, even though the bundle's own `manifest
    .json` already records `step`/`bundle_id` directly and `manifest_
    sha256` is trivially that file's own hash. Confirmed real, concrete
    consequence: `fit_architecture4_residual_basis.py`'s resolve-once fix
    passes an already-resolved bundle dir downstream, and a provenance
    sidecar recording a `null` step then failed `maybe_load_gene_basis`'s
    own 'missing fields must fail' check on the very next load."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=5)
        identity_via_pointer = resolve_checkpoint_identity(tmp)
        assert identity_via_pointer.step == 5
        assert identity_via_pointer.bundle_dir is not None
        assert identity_via_pointer.manifest_sha256 is not None

        # Resolving the ALREADY-RESOLVED bundle directory directly (no
        # pointer at that level) must recover the SAME identity, not None.
        identity_direct = resolve_checkpoint_identity(identity_via_pointer.resolved_dir)
        assert identity_direct.resolved_dir == identity_via_pointer.resolved_dir
        assert identity_direct.step == identity_via_pointer.step
        assert identity_direct.bundle_dir == identity_via_pointer.bundle_dir
        assert identity_direct.manifest_sha256 == identity_via_pointer.manifest_sha256
        assert identity_direct.weights_sha256 == identity_via_pointer.weights_sha256


def test_load_trainable_state_rejects_a_fully_frozen_model_missing_its_entire_weights_file():
    """Codex re-audit of commit 7a2d819, finding #1: 'if trainable_
    weights.pt is absent, load_trainable_state() only checks whether
    trainable parameters exist -- not expected buffers. A fully frozen
    model with saved buffers can still silently continue with freshly
    initialized buffers.' `_FrozenWithBuffer` has zero trainable
    parameters (so the OLD `trainable_names`-only check would have
    treated a missing weights file as a legitimate no-op) but a real
    non-frozen buffer -- `save_checkpoint` DOES write trainable_weights.pt
    for it (since `expected_names` is non-empty); this test then deletes
    that ENTIRE file plus its manifest entry (not merely a key inside
    it), simulating a damaged/incomplete checkpoint, and proves loading
    now fails instead of silently keeping the fresh buffer value."""
    import hashlib

    m = _FrozenWithBuffer()
    with torch.no_grad():
        m.some_buffer.fill_(9.0)
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "frozen_with_buffer"}, ["g1"], tmp, step=1)
        identity = resolve_checkpoint_identity(tmp)
        weights_path = identity.resolved_dir / "trainable_weights.pt"
        assert weights_path.is_file()  # a real buffer means a real weights file is written

        manifest_path = identity.resolved_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "trainable_weights.pt" in manifest["files"]
        del manifest["files"]["trainable_weights.pt"]
        weights_path.unlink()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        pointer_path = Path(tmp) / "latest_bundle.json"
        pointer = json.loads(pointer_path.read_text())
        pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        pointer_path.write_text(json.dumps(pointer, indent=2))

        reloaded = _FrozenWithBuffer()
        with pytest.raises(RuntimeError, match="missing but this model architecture has expected"):
            load_trainable_state(reloaded, tmp)
