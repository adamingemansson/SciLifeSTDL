"""Phase 8: the four-GPU launcher's own mechanics (static config audit,
fail-closed fingerprint checks, GPU/thread wiring, smoke gate, logging,
exit-code propagation) -- exercised with a stub `command_builder` that
runs a tiny in-process Python one-liner, never a real training job or a
real GPU. See launch_four_gpu_suite.py's module docstring for why
`default_command_builder` itself (pointing at a training entrypoint that
does not exist yet) is deliberately NOT exercised end-to-end here."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import gen3_multiscale.training.launch_four_gpu_suite as launch_four_gpu_suite_module
from gen3_multiscale.training.launch_four_gpu_suite import (
    check_required_fingerprints, default_command_builder, default_staged_smoke_command_builder,
    launch_staged_suite, launch_suite, main, run_suite_with_smoke_gate, static_config_audit,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_REAL_CONFIG_NAMES = ["architecture1", "architecture2", "architecture3", "architecture4"]


def _load_real_configs() -> dict[str, dict]:
    return {
        name: OmegaConf.to_container(OmegaConf.load(_CONFIG_DIR / f"{name}.yaml"), resolve=True)
        for name in _REAL_CONFIG_NAMES
    }


def _stub_command_builder(fail_for: frozenset[str] = frozenset()):
    def builder(config, config_path, smoke):
        name = Path(config_path).stem
        code = (
            "import os,sys;"
            "print('CUDA_VISIBLE_DEVICES=' + os.environ.get('CUDA_VISIBLE_DEVICES',''));"
            "print('OMP_NUM_THREADS=' + os.environ.get('OMP_NUM_THREADS',''));"
            f"print('SMOKE={smoke}');"
            f"print('CONFIG_PATH={config_path}');"
            f"sys.exit(1 if {str(name) in fail_for} else 0)"
        )
        return [sys.executable, "-c", code]
    return builder


def _minimal_named_configs(n: int = 2) -> tuple[dict, dict]:
    named_configs, config_paths = {}, {}
    for i in range(n):
        name = f"cfg{i}"
        named_configs[name] = {"experiment_name": name, "shared_field": 1, "training": {"checkpoint_dir": f"/tmp/{name}"}}
        config_paths[name] = Path(f"/tmp/{name}.yaml")
    return named_configs, config_paths


def _named_configs_with_architecture4(tmp_path, n_others: int = 2) -> tuple[dict, dict]:
    """Like `_minimal_named_configs`, plus one config whose
    `model.architecture == "4"` -- the shape `launch_staged_suite` looks
    for to decide what must be held out of the concurrent group."""
    named_configs, config_paths = {}, {}
    for i in range(n_others):
        name = f"cfg{i}"
        named_configs[name] = {"experiment_name": name, "shared_field": 1, "training": {"checkpoint_dir": str(tmp_path / name)}}
        config_paths[name] = Path(f"/tmp/{name}.yaml")
    named_configs["arch4"] = {
        "experiment_name": "arch4", "shared_field": 1, "model": {"architecture": "4"},
        "training": {"checkpoint_dir": str(tmp_path / "arch4")},
        "required_fingerprints": {},
    }
    config_paths["arch4"] = Path("/tmp/arch4.yaml")
    return named_configs, config_paths


# ---------------------------------------------------------------------------
# static_config_audit
# ---------------------------------------------------------------------------
def test_static_config_audit_passes_for_the_real_four_configs():
    result = static_config_audit(_load_real_configs())
    assert result["ok"] is True
    assert result["violations"] == []
    assert result["n_configs"] == 4


def test_static_config_audit_flags_an_undocumented_divergence():
    named_configs, _ = _minimal_named_configs()
    named_configs["cfg1"]["shared_field"] = 2  # not in documented_divergences -- must be flagged
    result = static_config_audit(named_configs)
    assert result["ok"] is False
    assert result["violations"][0]["key"] == "shared_field"


def test_static_config_audit_allows_a_documented_divergence():
    named_configs, _ = _minimal_named_configs()
    for cfg in named_configs.values():
        cfg["documented_divergences"] = ["shared_field"]
    named_configs["cfg1"]["shared_field"] = 2
    result = static_config_audit(named_configs)
    assert result["ok"] is True


def test_static_config_audit_ignores_keys_absent_from_some_configs():
    """Architecture 4's flow-only params (n_flow_blocks, etc.) have no
    equivalent in architecture{1,2,3}.yaml -- a key missing from some
    configs must never itself be flagged as a violation."""
    named_configs, _ = _minimal_named_configs()
    named_configs["cfg1"]["arch4_only_field"] = 99
    result = static_config_audit(named_configs)
    assert result["ok"] is True


def test_static_config_audit_requires_at_least_two_configs():
    with pytest.raises(ValueError, match="at least two"):
        static_config_audit({"only_one": {"a": 1}})


def test_static_config_audit_still_catches_a_divergence_among_three_configs_when_a_fourth_lacks_the_key():
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): the previous intersection-across-ALL-configs
    implementation meant that a key missing from just ONE config (e.g.
    architecture4.yaml's real, deliberate absence of
    model.params.use_regional_he) silently exempted that key from being
    checked among the OTHER configs too -- so an undocumented divergence
    between cfg0/cfg1/cfg2 on a field all three genuinely share would go
    completely unflagged purely because cfg3 doesn't have it at all."""
    named_configs, _ = _minimal_named_configs(n=4)
    named_configs["cfg0"]["shared_field"] = 1
    named_configs["cfg1"]["shared_field"] = 1
    named_configs["cfg2"]["shared_field"] = 2  # undocumented divergence from cfg0/cfg1
    del named_configs["cfg3"]["shared_field"]  # cfg3 structurally lacks this field entirely
    result = static_config_audit(named_configs)
    assert result["ok"] is False
    assert result["violations"][0]["key"] == "shared_field"
    assert result["violations"][0]["values"] == {"cfg0": 1, "cfg1": 1, "cfg2": 2}  # cfg3 correctly excluded, not compared


# ---------------------------------------------------------------------------
# check_required_fingerprints
# ---------------------------------------------------------------------------
def test_check_required_fingerprints_ignores_unused_entries_regardless_of_missing_or_null(tmp_path):
    """Regression test for a real, confirmed gap (Codex audit of commit
    27e1232): gene_vocabulary/mask-bank paths are not consumed by the
    real trainer for ANY architecture today -- a config declaring them
    missing or null must not be blocked on their account."""
    config = {
        "model": {"architecture": "1", "params": {"use_global_slide": False}},
        "required_fingerprints": {
            "gene_vocabulary": None, "train_mask_bank": str(tmp_path / "nope.json"),
            "validation_mask_bank": None, "test_mask_bank": None,
        },
    }
    assert check_required_fingerprints(config) == []


def test_check_required_fingerprints_requires_gigapath_checkpoint_only_when_use_global_slide(tmp_path):
    base_config = {"required_fingerprints": {"gigapath_checkpoint": None}}
    assert check_required_fingerprints({
        **base_config, "model": {"architecture": "1", "params": {"use_global_slide": False}},
    }) == []
    missing = check_required_fingerprints({
        **base_config, "model": {"architecture": "3", "params": {"use_global_slide": True}},
    })
    assert any("gigapath_checkpoint" in m for m in missing)

    present = tmp_path / "checkpoint.pth"
    present.write_text("fake checkpoint bytes")
    assert check_required_fingerprints({
        "model": {"architecture": "3", "params": {"use_global_slide": True}},
        "required_fingerprints": {"gigapath_checkpoint": str(present)},
    }) == []


def test_check_required_fingerprints_requires_gene_residual_basis_only_for_architecture_4(tmp_path):
    assert check_required_fingerprints({
        "model": {"architecture": "1", "params": {}}, "required_fingerprints": {"gene_residual_basis": None},
    }) == []
    missing = check_required_fingerprints({
        "model": {"architecture": "4", "params": {}}, "required_fingerprints": {"gene_residual_basis": None},
    })
    assert any("gene_residual_basis" in m for m in missing)

    present = tmp_path / "basis.pt"
    present.write_text("fake basis bytes")
    checkpoint_present = tmp_path / "arch3_ckpt" / "trainable_weights.pt"
    checkpoint_present.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_present.write_text("fake checkpoint bytes")
    assert check_required_fingerprints({
        "model": {"architecture": "4", "params": {}},
        "required_fingerprints": {
            "gene_residual_basis": str(present),
            "architecture3_conditioner_checkpoint": str(checkpoint_present.parent),
        },
    }) == []


def test_check_required_fingerprints_requires_architecture3_conditioner_checkpoint_for_architecture_4_non_smoke(tmp_path):
    """Adam's Step 6 audit #8 of commit a32051b: "Require Architecture
    4's conditioner checkpoint in launcher preflight." A non-smoke
    (smoke_only=False, the default) Architecture 4 launch must refuse to
    start without a real, on-disk architecture3_conditioner_checkpoint --
    matching train.py's own runtime requirement exactly, so the launcher
    catches this BEFORE spawning a subprocess rather than the subprocess
    failing deep inside training. A smoke_only=True launch (construction-
    only, per train.py's own --smoke default) is exempt."""
    config = {
        "model": {"architecture": "4", "params": {}},
        "required_fingerprints": {"gene_residual_basis": None, "architecture3_conditioner_checkpoint": None},
    }
    missing = check_required_fingerprints(config)
    assert any("architecture3_conditioner_checkpoint" in m for m in missing)
    # smoke_only exempts it (construction-only smoke never touches it).
    smoke_missing = check_required_fingerprints(config, smoke_only=True)
    assert not any("architecture3_conditioner_checkpoint" in m for m in smoke_missing)

    present_dir = tmp_path / "arch3_ckpt"
    present_dir.mkdir()
    config["required_fingerprints"]["architecture3_conditioner_checkpoint"] = str(present_dir)
    config["required_fingerprints"]["gene_residual_basis"] = str(tmp_path / "basis.pt")
    (tmp_path / "basis.pt").write_text("fake basis bytes")
    assert check_required_fingerprints(config) == []


def test_check_required_fingerprints_empty_when_no_fingerprints_declared():
    assert check_required_fingerprints({}) == []
    assert check_required_fingerprints({"model": {"architecture": "4", "params": {}}}) != []


# ---------------------------------------------------------------------------
# launch_suite
# ---------------------------------------------------------------------------
def test_launch_suite_requires_exactly_one_gpu_per_config(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    with pytest.raises(ValueError, match="one GPU per config"):
        launch_suite(named_configs, config_paths, gpu_list=["0"], log_root=tmp_path, command_builder=_stub_command_builder())


def test_launch_suite_rejects_duplicate_gpu_ids(tmp_path):
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): duplicate-GPU validation used to live only in
    main(), so a caller invoking launch_suite() directly (as every test
    in this file, and any real Python caller, does) bypassed it
    entirely."""
    named_configs, config_paths = _minimal_named_configs(n=2)
    with pytest.raises(ValueError, match="DIFFERENT GPU ids"):
        launch_suite(
            named_configs, config_paths, gpu_list=["0", "0"], log_root=tmp_path,
            command_builder=_stub_command_builder(),
        )
    assert list(tmp_path.iterdir()) == []


def test_launch_suite_rejects_non_positive_threads_per_job(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    with pytest.raises(ValueError, match="threads_per_job must be positive"):
        launch_suite(
            named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path, threads_per_job=0,
            command_builder=_stub_command_builder(),
        )
    assert list(tmp_path.iterdir()) == []


def test_launch_suite_cleans_up_already_spawned_jobs_when_a_later_spawn_fails(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): "add cleanup if Popen succeeds for some arms and
    then fails while spawning another: terminate and wait for
    already-started children and close all log handles." Before this
    fix, an exception partway through the spawn loop (e.g. a
    command_builder producing a command naming a nonexistent executable)
    left every earlier-started subprocess running unmonitored, and every
    already-opened log file handle leaked. subprocess.Popen itself is
    monkeypatched here (rather than relying on real process-table timing,
    which would make this test racy) so the fake processes' terminate()/
    wait() calls can be asserted directly."""
    named_configs, config_paths = _minimal_named_configs(n=3)

    class _FakeProc:
        # A fake, out-of-range pid (no real process ever has this id) so
        # _terminate_process_group's os.getpgid(pid) call reliably raises
        # ProcessLookupError and falls back to the plain terminate()
        # path -- exercising the SAME fallback branch a real environment
        # without process-group support would take, without needing an
        # actual OS process.
        _next_pid = 999_999_001

        def __init__(self):
            self.pid = _FakeProc._next_pid
            _FakeProc._next_pid += 1
            self.terminated = False
            self.waited = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

    created = []

    def fake_popen(command, env=None, stdout=None, stderr=None, start_new_session=None):
        if len(created) == 2:
            raise FileNotFoundError("simulated: no such file or directory")
        proc = _FakeProc()
        created.append(proc)
        return proc

    monkeypatch.setattr(launch_four_gpu_suite_module.subprocess, "Popen", fake_popen)

    with pytest.raises(FileNotFoundError):
        launch_suite(
            named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
            command_builder=_stub_command_builder(),
        )

    assert len(created) == 2  # cfg0 and cfg1 spawned before cfg2's Popen raised
    assert all(proc.terminated and proc.waited for proc in created)
    # All three log files were opened (cfg0, cfg1, cfg2) before the failure.
    assert {p.name for p in tmp_path.iterdir()} == {"cfg0.log", "cfg1.log", "cfg2.log"}


def test_launch_suite_cleans_up_every_job_on_keyboard_interrupt_during_the_wait_phase(tmp_path, monkeypatch):
    """Regression test for a real, confirmed gap (7th Codex re-audit of
    commit 2782ff0): cleanup previously only wrapped the SPAWN loop, not
    the WAIT loop -- exactly the phase a real, hours-long training run
    spends nearly all of its time in. `except Exception` also never
    caught KeyboardInterrupt/SystemExit at all (they inherit from
    BaseException directly), so a real Ctrl+C during either phase
    previously left every started job running unmonitored. Here, one
    job's own wait() call raises KeyboardInterrupt (standing in for a
    real Ctrl+C arriving mid-run) -- every already-spawned job, not just
    the ones before the interruption, must still be terminated."""
    named_configs, config_paths = _minimal_named_configs(n=3)

    class _FakeProc:
        _next_pid = 999_999_101

        def __init__(self, raise_on_wait: bool):
            self.pid = _FakeProc._next_pid
            _FakeProc._next_pid += 1
            self.terminated = False
            self.waited = False
            self._raise_on_wait = raise_on_wait
            self._first_wait = True

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            # Simulate the interrupt arriving during the MAIN wait loop's
            # plain proc.wait() call (timeout=None) -- not during
            # _terminate_process_group's own cleanup wait(timeout=...).
            if self._raise_on_wait and self._first_wait and timeout is None:
                self._first_wait = False
                raise KeyboardInterrupt()
            self.waited = True
            return 0

    created = []

    def fake_popen(command, env=None, stdout=None, stderr=None, start_new_session=None):
        proc = _FakeProc(raise_on_wait=(len(created) == 1))  # the SECOND job's wait() raises
        created.append(proc)
        return proc

    monkeypatch.setattr(launch_four_gpu_suite_module.subprocess, "Popen", fake_popen)

    with pytest.raises(KeyboardInterrupt):
        launch_suite(
            named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
            command_builder=_stub_command_builder(),
        )

    assert len(created) == 3  # all three jobs were spawned before the interrupt
    assert all(proc.terminated for proc in created)


def test_terminate_process_group_escalates_to_sigkill_when_sigterm_is_ignored():
    """Real (not monkeypatched) subprocess test: a child that installs a
    SIGTERM-ignoring handler must still be reaped -- real, confirmed gap
    (7th Codex re-audit of commit 2782ff0: "children that ignore
    SIGTERM"). _terminate_process_group escalates to SIGKILL after a
    short timeout for exactly this case."""
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        launch_four_gpu_suite_module._terminate_process_group(proc, timeout=1.0)
        assert proc.poll() is not None  # reaped via SIGKILL, not left running
    finally:
        if proc.poll() is None:  # pragma: no cover -- safety net only
            proc.kill()
            proc.wait()


def test_terminate_process_group_kills_a_descendant_that_survives_after_the_parent_exits():
    """Real (not monkeypatched) subprocess test for a real, confirmed gap
    (8th Codex re-audit of commit 7b5c267): the previous implementation
    only waited for the PARENT process (via proc.wait()) -- if the parent
    exits promptly on SIGTERM (the DEFAULT disposition; it installs no
    custom handler here) while a descendant in the SAME process group
    explicitly ignores SIGTERM and keeps running, the old code's
    proc.wait() succeeded and it returned WITHOUT ever checking whether
    the group still had a live member, so SIGKILL was never sent. The
    descendant here is spawned as a plain child (not its own new
    session), so it inherits the parent's process group -- exactly the
    real DataLoader-worker-child scenario this whole mechanism exists
    for."""
    parent_code = (
        "import subprocess, sys, time;"
        "subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']);"
        "time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    pgid = os.getpgid(proc.pid)
    try:
        time.sleep(0.5)  # let the parent actually spawn its child before we terminate the group
        launch_four_gpu_suite_module._terminate_process_group(proc, timeout=2.0)
        assert proc.poll() is not None  # the parent itself was reaped
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)  # the WHOLE group, including the surviving descendant, is gone
    finally:
        if proc.poll() is None:  # pragma: no cover -- safety net only
            proc.kill()
            proc.wait()


def test_terminate_process_group_cleans_up_a_descendant_even_when_the_parent_already_exited_before_cleanup_ran():
    """Real (not monkeypatched) subprocess test for a real, confirmed gap
    (9th Codex re-audit of commit a29be53): the previous implementation
    checked `if proc.poll() is not None: return` FIRST, before doing
    anything else -- if the parent had ALREADY exited (naturally, or
    simply because cleanup runs some time after whatever triggered it)
    by the time _terminate_process_group is called, it returned
    immediately without ever attempting group cleanup, even though a
    surviving descendant could still be running. The parent here spawns
    a SIGTERM-ignoring child and then exits immediately on its own --
    cleanup is only invoked once the parent is CONFIRMED already dead,
    exactly the scenario the audit describes. pgid is captured
    immediately after Popen (mirroring launch_suite's own real fix: look
    it up while the process is definitely still alive, not lazily inside
    cleanup, since a lazy os.getpgid(proc.pid) call also fails once the
    leader pid itself no longer exists)."""
    parent_code = (
        "import subprocess, sys;"
        "subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])"
        # the parent exits immediately after spawning its child -- no sleep of its own
    )
    proc = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    pgid = os.getpgid(proc.pid)  # captured immediately, while proc is still alive
    try:
        proc.wait(timeout=5.0)  # confirm the parent has ALREADY exited before cleanup runs
        assert proc.poll() is not None

        launch_four_gpu_suite_module._terminate_process_group(proc, pgid=pgid, timeout=2.0)

        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)  # the surviving descendant, orphaned when its parent exited, is now gone too
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)  # pragma: no cover -- safety net only
        except (ProcessLookupError, PermissionError):
            pass


def test_launch_suite_refuses_to_start_any_job_when_the_audit_fails(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    named_configs["cfg1"]["shared_field"] = 2  # undocumented divergence
    with pytest.raises(ValueError, match="static config audit failed"):
        launch_suite(named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path, command_builder=_stub_command_builder())
    assert list(tmp_path.iterdir()) == []  # nothing was ever spawned or logged


def test_launch_suite_refuses_to_start_when_a_required_fingerprint_is_missing(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    for cfg in named_configs.values():
        cfg["model"] = {"architecture": "3", "params": {"use_global_slide": True}}
        cfg["required_fingerprints"] = {"gigapath_checkpoint": None}
    with pytest.raises(ValueError, match="missing required fingerprints"):
        launch_suite(named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path, command_builder=_stub_command_builder())
    assert list(tmp_path.iterdir()) == []


def test_launch_suite_pins_gpus_caps_threads_writes_logs_and_a_summary(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    result = launch_suite(
        named_configs, config_paths, gpu_list=["3", "5"], log_root=tmp_path,
        threads_per_job=2, command_builder=_stub_command_builder(),
    )
    assert result.ok is True
    assert {job.name: job.gpu for job in result.jobs} == {"cfg0": "3", "cfg1": "5"}

    log0 = (tmp_path / "cfg0.log").read_text()
    assert "CUDA_VISIBLE_DEVICES=3" in log0
    assert "OMP_NUM_THREADS=2" in log0

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["ok"] is True
    assert len(summary["jobs"]) == 2


def test_launch_suite_propagates_failure_and_stops_promotion_when_one_arm_fails(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    result = launch_suite(
        named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path,
        command_builder=_stub_command_builder(fail_for=frozenset({"cfg1"})),
    )
    assert result.ok is False  # "stop promotion if any arm fails"
    by_name = {job.name: job for job in result.jobs}
    assert by_name["cfg0"].succeeded is True
    assert by_name["cfg1"].succeeded is False
    assert by_name["cfg1"].returncode != 0
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["ok"] is False


def test_launch_suite_passes_the_smoke_flag_through_to_the_command_builder(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    launch_suite(
        named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path,
        smoke_only=True, command_builder=_stub_command_builder(),
    )
    assert "SMOKE=True" in (tmp_path / "cfg0.log").read_text()


# ---------------------------------------------------------------------------
# run_suite_with_smoke_gate
# ---------------------------------------------------------------------------
def test_smoke_gate_blocks_the_full_run_when_smoke_fails(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    smoke_result, full_result = run_suite_with_smoke_gate(
        named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path,
        command_builder=_stub_command_builder(fail_for=frozenset({"cfg1"})),
    )
    assert smoke_result.ok is False
    assert full_result is None
    assert not (tmp_path / "full").exists()  # the full run was never even started


def test_smoke_gate_runs_the_full_suite_after_smoke_succeeds(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    smoke_result, full_result = run_suite_with_smoke_gate(
        named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path,
        command_builder=_stub_command_builder(),
    )
    assert smoke_result.ok is True
    assert full_result is not None
    assert full_result.ok is True
    assert "SMOKE=True" in (tmp_path / "smoke" / "cfg0.log").read_text()
    assert "SMOKE=False" in (tmp_path / "full" / "cfg0.log").read_text()


# ---------------------------------------------------------------------------
# launch_staged_suite -- Codex re-audit of commit 90f853e, launch blocker
# #8: "Architecture 4 must run only after the selected Architecture 3
# checkpoint and residual basis exist; it must not be launched
# concurrently with the Architecture 3 run it depends on."
# ---------------------------------------------------------------------------
def test_launch_staged_suite_launches_architecture4_only_after_the_other_configs_finish(tmp_path):
    order: list[str] = []

    def _order_tracking_builder(config, config_path, smoke):
        name = Path(config_path).stem
        code = f"import time,sys; time.sleep(0.05); print({name!r}); sys.exit(0)"
        return [sys.executable, "-c", code]

    named_configs, config_paths = _named_configs_with_architecture4(tmp_path)
    stage1_result, stage2_result = launch_staged_suite(
        named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
        command_builder=_order_tracking_builder, skip_fingerprint_check=True,
    )
    assert stage1_result.ok is True
    assert {job.name for job in stage1_result.jobs} == {"cfg0", "cfg1"}
    assert stage2_result is not None
    assert stage2_result.ok is True
    assert [job.name for job in stage2_result.jobs] == ["arch4"]
    # Architecture 4's own log only exists in stage 2's directory,
    # written strictly after stage 1's jobs (started together, in
    # parallel, in stage 1) have both already exited.
    assert (tmp_path / "architecture4" / "arch4.log").is_file()
    assert not (tmp_path / "pre_architecture4" / "arch4.log").exists()


def test_launch_staged_suite_never_launches_architecture4_when_its_prerequisites_are_missing_after_stage1(tmp_path):
    named_configs, config_paths = _named_configs_with_architecture4(tmp_path)
    named_configs["arch4"]["required_fingerprints"] = {"gene_residual_basis": str(tmp_path / "does_not_exist.pt")}
    with pytest.raises(ValueError, match="missing required fingerprints"):
        launch_staged_suite(
            named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
            command_builder=_stub_command_builder(), skip_fingerprint_check=False,
        )


def test_launch_staged_suite_stops_promotion_when_stage1_fails(tmp_path):
    named_configs, config_paths = _named_configs_with_architecture4(tmp_path)
    stage1_result, stage2_result = launch_staged_suite(
        named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
        command_builder=_stub_command_builder(fail_for=frozenset({"cfg1"})), skip_fingerprint_check=True,
    )
    assert stage1_result.ok is False
    assert stage2_result is None
    assert not (tmp_path / "architecture4").exists()  # Architecture 4 never even started


def test_launch_staged_suite_degenerates_to_a_single_concurrent_stage_with_no_architecture4_present(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    stage1_result, stage2_result = launch_staged_suite(
        named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path,
        command_builder=_stub_command_builder(),
    )
    assert stage2_result is None
    assert {job.name for job in stage1_result.jobs} == {"cfg0", "cfg1"}
    # No staging subdirectories -- log paths are IDENTICAL to plain launch_suite's.
    assert (tmp_path / "cfg0.log").is_file()


def test_default_staged_smoke_command_builder_adds_the_staged_smoke_flag_only_when_smoke():
    command_smoke = default_staged_smoke_command_builder({}, Path("/tmp/architecture4.yaml"), smoke=True)
    assert "--smoke" in command_smoke
    assert "--staged-smoke" in command_smoke
    command_full = default_staged_smoke_command_builder({}, Path("/tmp/architecture4.yaml"), smoke=False)
    assert "--staged-smoke" not in command_full


def test_check_required_fingerprints_staged_smoke_requires_the_conditioner_even_during_smoke_only(tmp_path):
    config = {"model": {"architecture": "4"}, "required_fingerprints": {"gene_residual_basis": str(tmp_path / "b.pt")}}
    (tmp_path / "b.pt").write_text("x")
    # Plain smoke_only=True: conditioner NOT required (construction-only).
    assert check_required_fingerprints(config, smoke_only=True) == []
    # staged_smoke=True during a smoke_only phase: conditioner IS required.
    missing = check_required_fingerprints(config, smoke_only=True, staged_smoke=True)
    assert any("architecture3_conditioner_checkpoint" in entry for entry in missing)


def test_run_suite_with_smoke_gate_stages_architecture4_in_both_phases(tmp_path):
    named_configs, config_paths = _named_configs_with_architecture4(tmp_path)
    smoke_result, full_result = run_suite_with_smoke_gate(
        named_configs, config_paths, gpu_list=["0", "1", "2"], log_root=tmp_path,
        command_builder=_stub_command_builder(), skip_fingerprint_check=True,
    )
    assert smoke_result.ok is True
    assert {job.name for job in smoke_result.jobs} == {"cfg0", "cfg1", "arch4"}
    assert full_result is not None
    assert full_result.ok is True
    assert {job.name for job in full_result.jobs} == {"cfg0", "cfg1", "arch4"}
    # Architecture 4's full-phase log lives under its own staged
    # subdirectory, proving it really ran as stage 2, not concurrently.
    assert (tmp_path / "full" / "architecture4" / "arch4.log").is_file()


# ---------------------------------------------------------------------------
# default_command_builder -- locks in the documented "not implemented yet" gap.
# ---------------------------------------------------------------------------
def test_default_command_builder_points_at_the_not_yet_implemented_entrypoint():
    command = default_command_builder({}, Path("/tmp/architecture1.yaml"), smoke=True)
    assert "gen3_multiscale.training.train" in command
    assert "--smoke" in command


# ---------------------------------------------------------------------------
# main() -- CLI argument validation. Regression tests for a real, confirmed
# gap (Codex audit finding against commit c02a5d1): main() previously
# accepted repeated GPU ids (which would launch two jobs on the same
# device, silently corrupting the "one job per GPU" isolation the whole
# launcher is built around) and non-positive --threads-per-job values
# (which subprocess.Popen would pass straight through as an env var, only
# failing much later inside the training process itself, if at all).
# Both must be rejected before any config is loaded or any subprocess is
# spawned, so real config paths are used but nothing ever actually runs.
# ---------------------------------------------------------------------------
def _real_config_paths() -> list[str]:
    return [str(_CONFIG_DIR / f"{name}.yaml") for name in _REAL_CONFIG_NAMES]


def test_main_rejects_duplicate_gpu_ids(tmp_path, monkeypatch):
    argv = [
        "launch_four_gpu_suite.py", "--configs", *_real_config_paths(),
        "--gpus", "0", "1", "1", "3", "--log-root", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="four DIFFERENT GPU ids"):
        main()


def test_main_rejects_non_positive_threads_per_job(tmp_path, monkeypatch):
    argv = [
        "launch_four_gpu_suite.py", "--configs", *_real_config_paths(),
        "--gpus", "0", "1", "2", "3", "--threads-per-job", "0", "--log-root", str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="threads-per-job must be positive"):
        main()
