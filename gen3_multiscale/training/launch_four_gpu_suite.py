"""Phase 8 of the multiscale spatial-field handoff: "Create four configs
and one launcher." This module implements the launcher's mechanics --
static config audit, fail-closed prerequisite checks, GPU/CPU-thread
wiring, a fail-closed smoke-test gate before any full run, one log per
job, a machine-readable summary, and nonzero-exit propagation -- exactly
as specified:

    "The launcher should: accept an explicit four-GPU list; use one job
    per GPU; cap CPU threads per job; run a static config audit first;
    run fail-closed smoke tests before full training; refuse to start if
    any required checkpoint, vocabulary, cache, split, or mask-bank
    fingerprint is absent; write one log per job plus a machine-readable
    summary; preserve nonzero exit status and stop promotion if any arm
    fails. Do not automatically launch the 24-hour jobs as part of
    implementation."

That last sentence is satisfied structurally, not just by convention:
nothing in this module runs on import, and `main()` -- the only code path
that can start a real subprocess against a real GPU -- only executes
under `if __name__ == "__main__":`. Every test in
tests/test_launch_four_gpu_suite.py calls the library functions directly
with a `command_builder` that launches a tiny in-process Python stub
(never a real training job), so importing or testing this module never
touches a GPU.

KNOWN GAP, documented here and in CONTRACT.md's Phase 8 section rather
than silently assumed away: `default_command_builder` below points at
`gen3_multiscale.training.train`, a module that DOES NOT EXIST YET. No
phase of this project (0 through 7) built a real training entrypoint for
gen3_multiscale -- Phases 0-6 built the data schema, model, and
architectures; Phase 7 built losses/metrics/diagnostics as standalone,
tested functions; nothing yet wires an optimizer loop, checkpointing, or
a real per-sample HEST-1k data loader together into a runnable `train.py`
the way gen2_architectures has for its own four architectures. This
launcher's OWN mechanics (config audit, fingerprint checks, GPU/thread
wiring, logging, exit-code propagation) are fully implemented and tested
against a stub command; the actual four 24-hour jobs cannot be started
today, with or without this launcher, until that entrypoint and a real
data builder exist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

# Fields the fairness matrix and Phase 4 item 8 explicitly allow to
# differ between the four architecture configs. Every OTHER key that is
# present in ALL of the configs being compared must be identical --
# "All fields not shown in [the fairness matrix] table must remain
# identical unless a difference is structurally required and
# documented." Anything beyond this fixed list must be declared per-file
# in that config's own `documented_divergences` list (see
# architectureN.yaml's headers).
_ALWAYS_ALLOWED_TO_DIFFER = frozenset({
    "experiment_name",
    "documented_divergences",
    "model.architecture",
    "model.params.use_anchor_blend",
    "model.params.use_regional_he",
    "model.params.use_global_gex",
    "model.params.use_global_slide",
    "training.checkpoint_dir",
})


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = value
    return out


def static_config_audit(named_configs: dict[str, dict]) -> dict:
    """Compare N resolved configs (as plain nested dicts); every key
    present in ALL of them must have an identical value UNLESS it's in
    `_ALWAYS_ALLOWED_TO_DIFFER` or listed in any config's own
    `documented_divergences`. Keys absent from some configs (e.g.
    Architecture 4's flow-only params) are never compared -- a config
    with a structurally different parameter set is not itself a
    violation; only an UNDOCUMENTED disagreement on a key every config
    claims to share is."""
    if len(named_configs) < 2:
        raise ValueError("static_config_audit needs at least two configs to compare")

    documented: set[str] = set()
    for cfg in named_configs.values():
        documented.update(cfg.get("documented_divergences") or [])
    allowed = _ALWAYS_ALLOWED_TO_DIFFER | documented

    flattened = {name: _flatten(cfg) for name, cfg in named_configs.items()}
    shared_keys = set.intersection(*(set(f.keys()) for f in flattened.values()))

    violations = []
    for key in sorted(shared_keys):
        if key in allowed:
            continue
        values = {name: flattened[name][key] for name in flattened}
        if len({repr(v) for v in values.values()}) > 1:
            violations.append({"key": key, "values": values})

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "n_configs": len(named_configs),
        "n_shared_keys_checked": len(shared_keys - allowed),
    }


def check_required_fingerprints(config: dict) -> list[str]:
    """"Refuse to start if any required checkpoint, vocabulary, cache,
    split, or mask-bank fingerprint is absent." A config declares its
    required paths under `required_fingerprints: {name: path}`; this
    returns a human-readable description per MISSING one (empty list =
    every declared fingerprint is present on disk). Fails closed: a path
    set to null/None is treated as missing, never skipped."""
    missing = []
    for name, path in (config.get("required_fingerprints") or {}).items():
        if path is None:
            missing.append(f"{name}: not set in config")
        elif not Path(str(path)).exists():
            missing.append(f"{name}: {path!r} does not exist")
    return missing


def default_command_builder(config: dict, config_path: Path, smoke: bool) -> list[str]:
    """The intended (NOT YET IMPLEMENTED -- see module docstring) real
    training entrypoint. Callers that only want to exercise this
    launcher's own mechanics must pass a different `command_builder`."""
    command = [sys.executable, "-m", "gen3_multiscale.training.train", "--config", str(config_path)]
    if smoke:
        command.append("--smoke")
    return command


@dataclass(frozen=True)
class JobResult:
    name: str
    gpu: str
    command: list[str]
    log_path: str
    returncode: int
    succeeded: bool


@dataclass(frozen=True)
class SuiteResult:
    ok: bool
    smoke_only: bool
    jobs: list[JobResult] = field(default_factory=list)
    summary_path: str = ""


def launch_suite(
    named_configs: dict[str, dict],
    config_paths: dict[str, Path],
    gpu_list: list[str],
    log_root: Path,
    threads_per_job: int = 4,
    smoke_only: bool = False,
    command_builder: Callable[[dict, Path, bool], list[str]] = default_command_builder,
    skip_fingerprint_check: bool = False,
) -> SuiteResult:
    """Launch one subprocess per config, each pinned to its own GPU via
    `CUDA_VISIBLE_DEVICES` and with CPU-thread env vars capped to
    `threads_per_job` (mirrors gen2_architectures/scripts/run_gene_aware_job.py's
    own OMP/MKL/OPENBLAS/NUMEXPR thread-capping discipline). Every job is
    started (Popen, not run) before any is waited on, so all four run
    CONCURRENTLY -- "one job per GPU" means in parallel, not
    sequentially. Runs the static config audit and (unless explicitly
    skipped) the fail-closed fingerprint check BEFORE spawning anything;
    either failing means no subprocess is ever started."""
    if len(named_configs) != len(gpu_list):
        raise ValueError(
            f"launch_suite requires exactly one GPU per config: got {len(named_configs)} "
            f"configs and {len(gpu_list)} GPUs"
        )
    if set(named_configs.keys()) != set(config_paths.keys()):
        raise ValueError("named_configs and config_paths must have the same keys")

    audit = static_config_audit(named_configs)
    if not audit["ok"]:
        raise ValueError(f"static config audit failed -- undocumented divergence in shared fields: {audit['violations']}")

    if not skip_fingerprint_check:
        for name, cfg in named_configs.items():
            missing = check_required_fingerprints(cfg)
            if missing:
                raise ValueError(f"{name}: refusing to start -- missing required fingerprints: {missing}")

    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    pending = []
    log_handles = []
    for (name, cfg), gpu in zip(named_configs.items(), gpu_list):
        command = command_builder(cfg, config_paths[name], smoke_only)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(threads_per_job)
        env["PYTHONUNBUFFERED"] = "1"
        log_path = log_root / f"{name}.log"
        log_file = log_path.open("w")
        log_handles.append(log_file)
        proc = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        pending.append((name, str(gpu), command, log_path, proc))

    jobs = []
    for name, gpu, command, log_path, proc in pending:
        returncode = proc.wait()
        jobs.append(JobResult(
            name=name, gpu=gpu, command=command, log_path=str(log_path),
            returncode=returncode, succeeded=returncode == 0,
        ))
    for handle in log_handles:
        handle.close()

    ok = all(job.succeeded for job in jobs)  # "preserve nonzero exit status and stop promotion if any arm fails"
    summary = {"ok": ok, "smoke_only": smoke_only, "jobs": [asdict(job) for job in jobs]}
    summary_path = log_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return SuiteResult(ok=ok, smoke_only=smoke_only, jobs=jobs, summary_path=str(summary_path))


def run_suite_with_smoke_gate(
    named_configs: dict[str, dict],
    config_paths: dict[str, Path],
    gpu_list: list[str],
    log_root: Path,
    threads_per_job: int = 4,
    command_builder: Callable[[dict, Path, bool], list[str]] = default_command_builder,
    skip_fingerprint_check: bool = False,
) -> tuple[SuiteResult, SuiteResult | None]:
    """"Run fail-closed smoke tests before full training ... stop
    promotion if any arm fails." Runs the smoke variant of every config
    first; the full run is only started if every smoke job succeeded.
    Returns (smoke_result, full_result) -- full_result is None when the
    smoke gate itself failed, so a caller can tell "never started" apart
    from "started and failed"."""
    log_root = Path(log_root)
    smoke_result = launch_suite(
        named_configs, config_paths, gpu_list, log_root / "smoke", threads_per_job,
        smoke_only=True, command_builder=command_builder, skip_fingerprint_check=skip_fingerprint_check,
    )
    if not smoke_result.ok:
        return smoke_result, None
    full_result = launch_suite(
        named_configs, config_paths, gpu_list, log_root / "full", threads_per_job,
        smoke_only=False, command_builder=command_builder, skip_fingerprint_check=skip_fingerprint_check,
    )
    return smoke_result, full_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs=4, required=True, metavar="PATH",
                         help="Exactly four config YAML paths (architecture1.yaml .. architecture4.yaml).")
    parser.add_argument("--gpus", nargs=4, required=True, metavar="GPU_ID",
                         help="Exactly four GPU ids, one per --configs entry in order.")
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--skip-fingerprint-check", action="store_true",
                         help="Only for local dry runs against a stub entrypoint -- never for a real launch.")
    args = parser.parse_args()

    if len(set(args.gpus)) != 4:
        raise ValueError(f"--gpus must name four DIFFERENT GPU ids, got {args.gpus!r}")
    if args.threads_per_job <= 0:
        raise ValueError(f"--threads-per-job must be positive, got {args.threads_per_job}")

    config_paths = {Path(p).stem: Path(p) for p in args.configs}
    if len(config_paths) != 4:
        raise ValueError("--configs must name four DIFFERENT files (by stem)")
    named_configs = {name: OmegaConf.to_container(OmegaConf.load(path), resolve=True) for name, path in config_paths.items()}

    smoke_result, full_result = run_suite_with_smoke_gate(
        named_configs, config_paths, list(args.gpus), Path(args.log_root),
        threads_per_job=args.threads_per_job, skip_fingerprint_check=args.skip_fingerprint_check,
    )
    if not smoke_result.ok:
        print(f"SMOKE TEST FAILED -- refusing to start full training. Summary: {smoke_result.summary_path}", file=sys.stderr)
        sys.exit(1)
    if not full_result.ok:
        print(f"ONE OR MORE ARMS FAILED. Summary: {full_result.summary_path}", file=sys.stderr)
        sys.exit(1)
    print(f"suite ok. Summary: {full_result.summary_path}")


if __name__ == "__main__":
    main()
