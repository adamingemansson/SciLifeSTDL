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
import signal
import subprocess
import sys
import time
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
    shared by two or more of them must have an identical value across
    those configs that declare it, UNLESS it's in
    `_ALWAYS_ALLOWED_TO_DIFFER` or listed in any config's own
    `documented_divergences`. A key present in only ONE config (e.g.
    Architecture 4's flow-only params) is never compared -- a config
    with a structurally different parameter set is not itself a
    violation; only an UNDOCUMENTED disagreement between configs that
    BOTH claim to share a key is.

    Compares over the key UNION with a per-key "which configs actually
    have it" set, not a single intersection across ALL configs (real,
    confirmed gap -- 6th Codex re-audit of commit 06f5cce): the previous
    intersection-only version meant that ANY one config lacking a key
    (e.g. architecture4.yaml genuinely and deliberately has no
    `model.params.use_regional_he`, since Architecture 4 wraps a full
    Architecture 3 conditioner rather than accepting that kwarg directly)
    silently exempted that key from being checked among the OTHER
    configs too -- so an undocumented divergence between
    architecture1/2/3.yaml on a field all three genuinely share would
    have gone completely unflagged, purely because a fourth, structurally
    different config didn't have the field at all. Now the comparison
    happens per-key among whichever configs actually declare it, so that
    3-way (or more) comparison is no longer silently skipped just because
    a structurally different config elsewhere doesn't have the key.

    NOT built here (a substantially larger redesign, out of scope for
    this pass): a canonical "effective experiment schema" that explicitly
    maps Architecture 4's nested conditioner fields onto Architecture 3's
    flat ones so the "should share" set is asserted positively rather
    than inferred from which keys happen to co-occur."""
    if len(named_configs) < 2:
        raise ValueError("static_config_audit needs at least two configs to compare")

    documented: set[str] = set()
    for cfg in named_configs.values():
        documented.update(cfg.get("documented_divergences") or [])
    allowed = _ALWAYS_ALLOWED_TO_DIFFER | documented

    flattened = {name: _flatten(cfg) for name, cfg in named_configs.items()}
    all_keys = set.union(*(set(f.keys()) for f in flattened.values()))
    comparable_keys = {
        key for key in all_keys
        if sum(1 for f in flattened.values() if key in f) >= 2
    }

    violations = []
    for key in sorted(comparable_keys):
        if key in allowed:
            continue
        values = {name: f[key] for name, f in flattened.items() if key in f}
        if len({repr(v) for v in values.values()}) > 1:
            violations.append({"key": key, "values": values})

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "n_configs": len(named_configs),
        "n_shared_keys_checked": len(comparable_keys - allowed),
    }


def check_required_fingerprints(config: dict, *, smoke_only: bool = False, staged_smoke: bool = False) -> list[str]:
    """"Refuse to start if any required checkpoint, vocabulary, cache,
    split, or mask-bank fingerprint is absent." A config declares its
    candidate paths under `required_fingerprints: {name: path}`; this
    returns a human-readable description per MISSING one that this
    SPECIFIC config actually needs (empty list = every fingerprint this
    config needs is present on disk). Fails closed: a path set to
    null/None for a NEEDED fingerprint is treated as missing, never
    skipped.

    Real, confirmed gap (Codex audit of commit 27e1232): a prior version
    treated EVERY key under `required_fingerprints` as unconditionally
    required, regardless of whether the real trainer (training/train.py)
    actually consumes it for this architecture. `gene_vocabulary` and the
    three mask-bank paths are not read by train.py at all today (mask
    banks are generated on the fly by
    `gen3_dataset.build_gen3_mask_schedule`, keyed off `training.
    checkpoint_dir`, not off `required_fingerprints`) -- requiring them
    meant NO config, real or synthetic, could ever launch a real smoke
    without `--skip-fingerprint-check`, which the launcher's own docs
    call "only for local dry runs... never for a real launch."
    `gigapath_checkpoint` is only consumed when `model.params.
    use_global_slide` is true (`train.py::maybe_build_slide_encoder`);
    `gene_residual_basis` only for Architecture 4 (`train.py::
    maybe_load_gene_basis`) -- required even for a smoke launch, since
    Architecture4's constructor needs a real gene basis just to build the
    model at all. Every OTHER declared `required_fingerprints` entry this
    config doesn't actually need is intentionally never checked here -- a
    config listing extra, unused paths (e.g. for a future Step 7/8
    consumer) must not block launch on their account.

    Real, confirmed gap (Adam's Step 6 audit #8 of commit a32051b):
    "Require Architecture 4's conditioner checkpoint in launcher
    preflight." `architecture3_conditioner_checkpoint` was previously
    never checked here at all -- `train.py::
    maybe_load_pretrained_conditioner_for_architecture4` DOES require it
    for any non-smoke Architecture 4 run (and raises if missing), meaning
    the launcher's own preflight gate was weaker than the trainer's own
    runtime check it is supposed to front-run: a real, non-smoke
    Architecture 4 launch could pass this gate and still fail deep inside
    the trainer. `smoke_only`, mirroring `train.py`'s own `--smoke` vs
    `--smoke --staged-smoke` distinction (a plain smoke is CONSTRUCTION-
    ONLY and exempt; `default_command_builder`'s `--smoke` produces a
    construction-only smoke, so this launcher's own smoke phase is
    correctly not blocked on a checkpoint that phase never touches)."""
    required_fingerprints = config.get("required_fingerprints") or {}
    model_params = (config.get("model") or {}).get("params") or {}
    architecture_id = str((config.get("model") or {}).get("architecture", ""))

    needed = set()
    if model_params.get("use_global_slide"):
        needed.add("gigapath_checkpoint")
    if architecture_id == "4":
        needed.add("gene_residual_basis")
        # Codex re-audit of commit 90f853e, launch blocker #8: a REAL
        # staged smoke (`train.py --smoke --staged-smoke`) requires
        # Architecture 4's real conditioner checkpoint exactly like a
        # non-smoke run (`require_for_smoke=staged_smoke` there) -- a
        # plain construction-only `--smoke` is the only case exempt.
        if not smoke_only or staged_smoke:
            needed.add("architecture3_conditioner_checkpoint")

    missing = []
    for name in sorted(needed):
        path = required_fingerprints.get(name)
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


def default_staged_smoke_command_builder(config: dict, config_path: Path, smoke: bool) -> list[str]:
    """Same contract as `default_command_builder`, plus `--staged-smoke`
    whenever `smoke` is True. Codex re-audit of commit 90f853e, launch
    blocker #8: a plain `--smoke` (what `default_command_builder` alone
    produces) is construction-only and never loads Architecture 4's real
    conditioner checkpoint -- exercising the REAL staged smoke this audit
    asked for requires `--staged-smoke` too, so `launch_staged_suite`'s
    Architecture 4 stage uses THIS builder (never the plain one) whenever
    its caller asks for `staged_smoke=True`."""
    command = default_command_builder(config, config_path, smoke)
    if smoke:
        command.append("--staged-smoke")
    return command


def _process_group_alive(pgid: int) -> bool:
    """`os.killpg(pgid, 0)` sends no actual signal -- it only probes
    whether the group still exists (raises ProcessLookupError once every
    member has exited). A PermissionError means the group DOES still
    exist (we just can't signal it), so that counts as alive too."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _terminate_process_group(proc: subprocess.Popen, pgid: int | None = None, timeout: float = 10.0) -> None:
    """Terminate `proc` and every process in its process group (real,
    confirmed gap -- 7th Codex re-audit of commit 2782ff0: "children that
    ignore SIGTERM... subprocess-owned dataloader workers"). `proc` must
    have been started with `start_new_session=True` so it (and anything
    IT spawns, e.g. PyTorch DataLoader worker processes) shares one
    process group distinct from this launcher's own -- signaling that
    whole group, not just `proc`'s own PID, is what actually reaches
    those worker children.

    Waits for the ENTIRE GROUP to disappear, not just `proc` itself (real,
    confirmed gap -- 8th Codex re-audit of commit 7b5c267): the previous
    version's success condition was `proc.wait(timeout=...)` returning --
    if the PARENT process exits promptly on SIGTERM (the common case: no
    custom handler means the default disposition terminates it) while a
    DESCENDANT in the same group ignores SIGTERM and keeps running, that
    wait() call succeeds and the function returned WITHOUT ever checking
    whether the group had any surviving members, let alone escalating to
    SIGKILL. `_group_gone` below requires BOTH `proc` itself to have
    exited AND `os.killpg(pgid, 0)` to confirm no process remains in its
    group before declaring success; only then does this function return
    without escalating.

    `pgid` should be the group id CAPTURED IMMEDIATELY AFTER `Popen`,
    while the process is definitely still alive -- real, confirmed gap
    (9th Codex re-audit of commit a29be53): the previous version both (1)
    returned immediately on `proc.poll() is not None` (the parent already
    exited by the time cleanup runs -- entirely plausible if this is
    called after the wait loop has already reaped some jobs, or simply
    because cleanup runs some time after the failure that triggered it)
    WITHOUT EVER attempting group cleanup, even though a surviving
    descendant could still be running; and (2) looked up the pgid via
    `os.getpgid(proc.pid)` lazily, which raises `ProcessLookupError` once
    the LEADER pid no longer exists as a process -- even though the
    process GROUP itself (identified by that same numeric id) can still
    have live members and still be validly signalable via `os.killpg`.
    Passing the pgid in (captured before any of that can happen) makes
    cleanup work correctly regardless of whether the parent has already
    exited by the time this function runs. The `pgid=None` fallback
    (lazy `os.getpgid` lookup) is kept only for callers that never
    captured one up front."""
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, AttributeError):
            pgid = None

    def _group_gone() -> bool:
        if proc.poll() is None:
            return False
        return pgid is None or not _process_group_alive(pgid)

    if _group_gone():
        proc.wait()
        return

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        proc.terminate()

    if not _wait_until(_group_gone, timeout):
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        else:
            proc.kill()
        _wait_until(_group_gone, timeout)

    # Unconditional, not `if proc.poll() is None:` -- on a real Popen this
    # is cheap and non-blocking once poll() has already observed exit
    # (the returncode is cached), and it guarantees the process is
    # actually reaped rather than relying on poll()'s internal reaping as
    # an implementation detail.
    proc.wait()


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
    skip_config_audit: bool = False,
) -> SuiteResult:
    """Launch one subprocess per config, each pinned to its own GPU via
    `CUDA_VISIBLE_DEVICES` and with CPU-thread env vars capped to
    `threads_per_job` (mirrors gen2_architectures/scripts/run_gene_aware_job.py's
    own OMP/MKL/OPENBLAS/NUMEXPR thread-capping discipline). Every job is
    started (Popen, not run) before any is waited on, so all four run
    CONCURRENTLY -- "one job per GPU" means in parallel, not
    sequentially. Runs the static config audit and (unless explicitly
    skipped) the fail-closed fingerprint check BEFORE spawning anything;
    either failing means no subprocess is ever started.

    Duplicate-GPU and positive-thread-count validation live HERE, not
    only in `main()` (real, confirmed gap -- 6th Codex re-audit of
    commit 06f5cce: "duplicate-GPU and thread validation happens in the
    CLI, but direct calls to launch_suite() bypass it") -- `main()`
    still validates early too (before even loading configs), but this is
    the actual invariant-enforcing location every caller, CLI or direct,
    goes through.
    """
    if len(named_configs) != len(gpu_list):
        raise ValueError(
            f"launch_suite requires exactly one GPU per config: got {len(named_configs)} "
            f"configs and {len(gpu_list)} GPUs"
        )
    if set(named_configs.keys()) != set(config_paths.keys()):
        raise ValueError("named_configs and config_paths must have the same keys")
    if len(set(gpu_list)) != len(gpu_list):
        raise ValueError(f"gpu_list must name DIFFERENT GPU ids, got {gpu_list!r}")
    if threads_per_job <= 0:
        raise ValueError(f"threads_per_job must be positive, got {threads_per_job}")

    if not skip_config_audit:
        audit = static_config_audit(named_configs)
        if not audit["ok"]:
            raise ValueError(f"static config audit failed -- undocumented divergence in shared fields: {audit['violations']}")

    if not skip_fingerprint_check:
        for name, cfg in named_configs.items():
            missing = check_required_fingerprints(cfg, smoke_only=smoke_only)
            if missing:
                raise ValueError(f"{name}: refusing to start -- missing required fingerprints: {missing}")

    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    # Both spawning AND waiting are wrapped together so a failure or
    # interruption at EITHER stage cleans up every already-started job,
    # not just the ones from a failed spawn loop (real, confirmed gap --
    # 7th Codex re-audit of commit 2782ff0: cleanup previously only
    # covered "interruption during spawning", never "interruption during
    # the waiting phase" -- exactly the phase a real, hours-long training
    # run spends nearly all its time in). `except BaseException`, not
    # `except Exception` (same audit: KeyboardInterrupt/SystemExit
    # inherit from BaseException directly, so a Ctrl+C during either
    # phase previously bypassed cleanup entirely, leaving every started
    # job running unmonitored). Each job is started with
    # `start_new_session=True` so `_terminate_process_group` can signal
    # its entire process group -- reaching DataLoader worker children,
    # not just the immediate training process -- with a SIGTERM-then-
    # SIGKILL escalation for anything that ignores SIGTERM.
    pending = []
    log_handles = []
    jobs = []
    try:
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
            proc = subprocess.Popen(
                command, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
            )
            # Captured immediately, while the process is definitely still
            # alive (real, confirmed gap -- 9th Codex re-audit of commit
            # a29be53): looking this up LAZILY inside the cleanup handler
            # (the previous approach) fails once the leader has already
            # exited by the time cleanup runs, even though the process
            # GROUP itself can still have live descendants.
            try:
                pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, AttributeError):
                pgid = None
            pending.append((name, str(gpu), command, log_path, proc, pgid))

        for name, gpu, command, log_path, proc, pgid in pending:
            returncode = proc.wait()
            jobs.append(JobResult(
                name=name, gpu=gpu, command=command, log_path=str(log_path),
                returncode=returncode, succeeded=returncode == 0,
            ))
    except BaseException:
        for _, _, _, _, proc, pgid in pending:
            _terminate_process_group(proc, pgid=pgid)
        raise
    finally:
        for handle in log_handles:
            handle.close()

    ok = all(job.succeeded for job in jobs)  # "preserve nonzero exit status and stop promotion if any arm fails"
    summary = {"ok": ok, "smoke_only": smoke_only, "jobs": [asdict(job) for job in jobs]}
    summary_path = log_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return SuiteResult(ok=ok, smoke_only=smoke_only, jobs=jobs, summary_path=str(summary_path))


def _combine_suite_results(smoke_only: bool, log_root: Path, results: list[SuiteResult | None]) -> SuiteResult:
    """Merge one or two sequential-phase `SuiteResult`s (see
    `launch_staged_suite`) into a single result with the combined job
    list -- so `run_suite_with_smoke_gate`'s external `(SuiteResult,
    SuiteResult | None)` contract stays exactly as it was before staging
    existed, regardless of whether Architecture 4 caused an extra
    internal phase."""
    present = [r for r in results if r is not None]
    jobs = [job for r in present for job in r.jobs]
    ok = all(r.ok for r in present)
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    summary = {"ok": ok, "smoke_only": smoke_only, "jobs": [asdict(job) for job in jobs], "staged": len(present) > 1}
    summary_path = log_root / "suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return SuiteResult(ok=ok, smoke_only=smoke_only, jobs=jobs, summary_path=str(summary_path))


def launch_staged_suite(
    named_configs: dict[str, dict],
    config_paths: dict[str, Path],
    gpu_list: list[str],
    log_root: Path,
    threads_per_job: int = 4,
    smoke_only: bool = False,
    staged_smoke: bool = False,
    command_builder: Callable[[dict, Path, bool], list[str]] = default_command_builder,
    architecture4_command_builder: Callable[[dict, Path, bool], list[str]] | None = None,
    skip_fingerprint_check: bool = False,
) -> tuple[SuiteResult, SuiteResult | None]:
    """Codex re-audit of commit 90f853e, launch blocker #8: "Wire real
    staged smoke into the launcher. Architecture 4 must run only after
    the selected Architecture 3 checkpoint and residual basis exist; it
    must not be launched concurrently with the Architecture 3 run it
    depends on." `launch_suite` (above) launches every config it is
    given CONCURRENTLY -- correct for Architectures 1-3, which train
    from scratch independently, but structurally wrong for Architecture
    4: its `required_fingerprints.architecture3_conditioner_checkpoint`/
    `gene_residual_basis` name a checkpoint/basis Architecture 4 must
    LOAD, never one it can race a freshly (re)started Architecture 3 job
    to produce in the same launch.

    Any config in `named_configs` with `model.architecture == "4"`
    (matched by content, never by dict key) is pulled OUT of the
    concurrent stage-1 group and launched alone, on its own originally-
    assigned GPU, as stage 2 -- started only after stage 1 has fully
    finished (`launch_suite` already blocks until every stage-1 job
    exits), and only after its required fingerprints are RE-CHECKED
    (the checkpoint/basis it needs may not have existed yet when THIS
    function was first called, if stage 1's own Architecture 3 job is
    what produces them).

    `staged_smoke=True` additionally requires Architecture 4's real
    conditioner checkpoint even during a `smoke_only` phase (mirrors
    `train.py --smoke --staged-smoke`'s own `require_for_smoke`
    contract) and, unless `architecture4_command_builder` overrides it,
    runs Architecture 4's stage-2 job through
    `default_staged_smoke_command_builder` so it actually passes
    `--staged-smoke` -- routing a staged-smoke phase through the plain
    `command_builder` would otherwise silently stay construction-only
    and never touch the real checkpoint, defeating the entire point of
    staging it.

    Returns `(stage1_result, stage2_result)`: `stage2_result` is `None`
    when no Architecture 4 config is present (nothing to stage) or when
    stage 1 failed (promotion stops, exactly like
    `run_suite_with_smoke_gate`'s existing `smoke_result, None`
    contract). Raises (fail-closed) if stage 1 succeeded but Architecture
    4's fingerprints are still missing -- a caller must not have
    Architecture 4 silently skipped when its prerequisites genuinely
    never materialized."""
    if len(named_configs) != len(gpu_list):
        raise ValueError(
            f"launch_staged_suite requires exactly one GPU per config: got {len(named_configs)} "
            f"configs and {len(gpu_list)} GPUs"
        )
    # The fairness-matrix audit compares ACROSS all four configs, including
    # Architecture 4 -- run it ONCE here, over the full group, before
    # splitting into stages; each stage's own `launch_suite` call below
    # passes `skip_config_audit=True` so a single-config Architecture 4
    # stage never hits `static_config_audit`'s own "needs at least two
    # configs" precondition, and stage 1 never redundantly re-audits.
    if len(named_configs) >= 2:
        audit = static_config_audit(named_configs)
        if not audit["ok"]:
            raise ValueError(f"static config audit failed -- undocumented divergence in shared fields: {audit['violations']}")
    gpu_by_name = dict(zip(named_configs.keys(), gpu_list))
    architecture4_names = [
        name for name, cfg in named_configs.items()
        if str((cfg.get("model") or {}).get("architecture", "")) == "4"
    ]
    if len(architecture4_names) > 1:
        raise ValueError(
            f"launch_staged_suite supports at most one Architecture 4 config per call, got "
            f"{architecture4_names}"
        )
    arch4_name = architecture4_names[0] if architecture4_names else None

    log_root = Path(log_root)
    stage1_names = [name for name in named_configs if name != arch4_name]
    # Log paths stay IDENTICAL to the pre-staging layout (`log_root/
    # f"{name}.log"`, `log_root/"suite_summary.json"`) whenever there is
    # no Architecture 4 config to stage -- nesting only kicks in once
    # there are genuinely two sequential phases to keep separate.
    stage1_log_root = log_root if arch4_name is None else log_root / "pre_architecture4"
    stage1_result = launch_suite(
        {name: named_configs[name] for name in stage1_names},
        {name: config_paths[name] for name in stage1_names},
        [gpu_by_name[name] for name in stage1_names],
        stage1_log_root, threads_per_job,
        smoke_only=smoke_only, command_builder=command_builder, skip_fingerprint_check=skip_fingerprint_check,
        skip_config_audit=True,
    )

    if arch4_name is None:
        return stage1_result, None
    if not stage1_result.ok:
        return stage1_result, None

    arch4_config = named_configs[arch4_name]
    if not skip_fingerprint_check:
        missing = check_required_fingerprints(arch4_config, smoke_only=smoke_only, staged_smoke=staged_smoke)
        if missing:
            raise ValueError(
                f"{arch4_name}: refusing to start stage 2 after stage 1 finished -- missing required "
                f"fingerprints: {missing}"
            )

    arch4_builder = architecture4_command_builder
    if arch4_builder is None:
        arch4_builder = default_staged_smoke_command_builder if staged_smoke else command_builder
    stage2_result = launch_suite(
        {arch4_name: arch4_config}, {arch4_name: config_paths[arch4_name]}, [gpu_by_name[arch4_name]],
        log_root / "architecture4", threads_per_job, smoke_only=smoke_only, command_builder=arch4_builder,
        skip_fingerprint_check=skip_fingerprint_check, skip_config_audit=True,
    )
    return stage1_result, stage2_result


def run_suite_with_smoke_gate(
    named_configs: dict[str, dict],
    config_paths: dict[str, Path],
    gpu_list: list[str],
    log_root: Path,
    threads_per_job: int = 4,
    command_builder: Callable[[dict, Path, bool], list[str]] = default_command_builder,
    skip_fingerprint_check: bool = False,
    staged_smoke: bool = False,
) -> tuple[SuiteResult, SuiteResult | None]:
    """"Run fail-closed smoke tests before full training ... stop
    promotion if any arm fails." Runs the smoke variant of every config
    first; the full run is only started if every smoke job succeeded.
    Returns (smoke_result, full_result) -- full_result is None when the
    smoke gate itself failed, so a caller can tell "never started" apart
    from "started and failed".

    Both phases now go through `launch_staged_suite` (launch blocker #8):
    whenever an Architecture 4 config is present, it is held out of its
    phase's concurrent group and launched only after that phase's other
    configs finish and its fingerprints are re-verified -- see
    `launch_staged_suite`'s docstring. The external return contract is
    unchanged (`_combine_suite_results` merges Architecture 4's separate
    internal stage back into one `SuiteResult` per phase), so this stays
    a drop-in replacement for the previous always-concurrent behavior."""
    log_root = Path(log_root)
    smoke_stage1, smoke_stage2 = launch_staged_suite(
        named_configs, config_paths, gpu_list, log_root / "smoke", threads_per_job,
        smoke_only=True, staged_smoke=staged_smoke, command_builder=command_builder,
        skip_fingerprint_check=skip_fingerprint_check,
    )
    smoke_result = _combine_suite_results(True, log_root / "smoke", [smoke_stage1, smoke_stage2])
    if not smoke_result.ok:
        return smoke_result, None
    full_stage1, full_stage2 = launch_staged_suite(
        named_configs, config_paths, gpu_list, log_root / "full", threads_per_job,
        smoke_only=False, staged_smoke=False, command_builder=command_builder,
        skip_fingerprint_check=skip_fingerprint_check,
    )
    full_result = _combine_suite_results(False, log_root / "full", [full_stage1, full_stage2])
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
    parser.add_argument(
        "--staged-smoke", action="store_true",
        help="Run Architecture 4's smoke phase as a REAL staged smoke (train.py --smoke --staged-smoke) "
             "against its already-selected Architecture 3 conditioner checkpoint, instead of a plain "
             "construction-only smoke. The full (non-smoke) phase is always staged regardless of this flag.",
    )
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
        staged_smoke=args.staged_smoke,
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
