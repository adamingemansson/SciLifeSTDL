"""Phase 8: the four-GPU launcher's own mechanics (static config audit,
fail-closed fingerprint checks, GPU/thread wiring, smoke gate, logging,
exit-code propagation) -- exercised with a stub `command_builder` that
runs a tiny in-process Python one-liner, never a real training job or a
real GPU. See launch_four_gpu_suite.py's module docstring for why
`default_command_builder` itself (pointing at a training entrypoint that
does not exist yet) is deliberately NOT exercised end-to-end here."""
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import gen3_multiscale.training.launch_four_gpu_suite as launch_four_gpu_suite_module
from gen3_multiscale.training.launch_four_gpu_suite import (
    check_required_fingerprints, default_command_builder, launch_suite, main, run_suite_with_smoke_gate,
    static_config_audit,
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
def test_check_required_fingerprints_reports_missing_and_null_paths(tmp_path):
    present = tmp_path / "exists.json"
    present.write_text("{}")
    config = {"required_fingerprints": {"present": str(present), "missing": str(tmp_path / "nope.json"), "unset": None}}
    missing = check_required_fingerprints(config)
    assert any("missing" in m for m in missing)
    assert any("unset" in m for m in missing)
    assert not any(m.startswith("present:") for m in missing)


def test_check_required_fingerprints_empty_when_all_present(tmp_path):
    present = tmp_path / "exists.json"
    present.write_text("{}")
    config = {"required_fingerprints": {"present": str(present)}}
    assert check_required_fingerprints(config) == []


def test_check_required_fingerprints_empty_when_no_fingerprints_declared():
    assert check_required_fingerprints({}) == []


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
        def __init__(self):
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return 0

    created = []

    def fake_popen(command, env=None, stdout=None, stderr=None):
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


def test_launch_suite_refuses_to_start_any_job_when_the_audit_fails(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    named_configs["cfg1"]["shared_field"] = 2  # undocumented divergence
    with pytest.raises(ValueError, match="static config audit failed"):
        launch_suite(named_configs, config_paths, gpu_list=["0", "1"], log_root=tmp_path, command_builder=_stub_command_builder())
    assert list(tmp_path.iterdir()) == []  # nothing was ever spawned or logged


def test_launch_suite_refuses_to_start_when_a_required_fingerprint_is_missing(tmp_path):
    named_configs, config_paths = _minimal_named_configs(n=2)
    for cfg in named_configs.values():
        cfg["required_fingerprints"] = {"gene_vocabulary": None}
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
