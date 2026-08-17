#!/usr/bin/env python3
"""Wait for MK finalist replications, evaluate them, and summarize all seeds.

This controller is safe to start while training is active.  It verifies every
training completion and best checkpoint before occupying the same four GPUs.
Evaluation then runs in parallel and is followed by exact-contract, seed-level,
patient-bootstrap and per-gene robustness analyses.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gen3_multiscale.scripts.analyze_mk_seed_gene_robustness import analyze as analyze_genes
from gen3_multiscale.scripts.summarize_mk_seed_replications import (
    ARCHITECTURES,
    discover_records,
    summarize,
)
from gen3_multiscale.training import checkpoint as checkpoint_module


EXPECTED_RUNS = {
    "mk_wb_parallel_gated_seed1": ("mk_wb_parallel_gated", 1),
    "mk_wb_parallel_gated_seed2": ("mk_wb_parallel_gated", 2),
    "mk_wbw_sandwich_seed1": ("mk_wbw_sandwich", 1),
    "mk_wbw_sandwich_seed2": ("mk_wbw_sandwich", 2),
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _pid_is_expected(pid: int, run_name: str) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.is_file():
        return True
    try:
        return run_name in cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return True


def _tail_text(path: Path, max_bytes: int = 1_000_000) -> str:
    """Read only the end of potentially multi-gigabyte training logs."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode(errors="replace")


def _validate_plan(root: Path) -> dict[str, Any]:
    path = root / "replication_plan.json"
    plan = json.loads(path.read_text())
    runs = plan.get("runs") or {}
    if set(runs) != set(EXPECTED_RUNS):
        raise ValueError(
            f"replication plan must contain exactly {sorted(EXPECTED_RUNS)}; got={sorted(runs)}"
        )
    gpu_ids = []
    for run_name, (architecture, seed) in EXPECTED_RUNS.items():
        run = runs[run_name]
        if str(run.get("source_arm")) != architecture or int(run.get("seed", -1)) != seed:
            raise ValueError(f"replication identity mismatch: {run_name}")
        config = Path(str(run.get("config") or ""))
        checkpoint = Path(str(run.get("checkpoint_dir") or ""))
        if not config.is_file():
            raise FileNotFoundError(config)
        if checkpoint != root / "checkpoints" / run_name:
            raise ValueError(f"checkpoint escapes replication root: {run_name}: {checkpoint}")
        gpu_ids.append(str(run["gpu"]))
    if len(set(gpu_ids)) != 4:
        raise ValueError("replication plan must assign four distinct GPUs")
    return plan


def _claim_controller(root: Path) -> None:
    """Refuse a second live postrun controller for the same replication root."""
    control = root / "control"
    control.mkdir(exist_ok=True)
    path = control / "postrun_controller.pid"
    if path.is_file():
        try:
            prior_pid = int(path.read_text().strip())
        except ValueError:
            prior_pid = None
        if prior_pid and _pid_is_expected(prior_pid, "run_mk_final_replication_postrun"):
            raise RuntimeError(
                f"postrun controller already active for this suite: PID={prior_pid}"
            )
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(str(os.getpid()) + "\n")
    os.replace(temporary, path)


def _training_complete(root: Path, run_name: str, run: dict[str, Any]) -> tuple[bool, str]:
    pid_path = root / "control" / f"{run_name}.pid"
    if pid_path.is_file():
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            return False, "invalid PID file"
        if _pid_is_expected(pid, run_name):
            return False, f"training PID {pid} is active"
    log = root / "logs" / f"{run_name}.log"
    if not log.is_file():
        return False, "training log is missing"
    tail = _tail_text(log)
    if "training finished:" not in tail:
        if "Traceback" in tail or "Error" in tail:
            raise RuntimeError(f"{run_name}: training failed; inspect {log}")
        return False, "completion marker is absent"
    checkpoint_dir = Path(run["checkpoint_dir"])
    best_state = checkpoint_module.load_training_state(checkpoint_dir / "best")
    if int(best_state.get("step", 0)) < 1:
        raise RuntimeError(f"{run_name}: no valid best checkpoint in {checkpoint_dir / 'best'}")
    return True, f"best_step={best_state['step']}"


def wait_for_training(root: Path, plan: dict[str, Any], poll_seconds: int) -> None:
    while True:
        pending = []
        snapshot = []
        for run_name in EXPECTED_RUNS:
            complete, detail = _training_complete(root, run_name, plan["runs"][run_name])
            snapshot.append(f"{run_name}: {'complete' if complete else 'waiting'} ({detail})")
            if not complete:
                pending.append(run_name)
        print("\n".join(snapshot), flush=True)
        if not pending:
            return
        print(f"Waiting {poll_seconds}s for {len(pending)} training run(s)...", flush=True)
        time.sleep(poll_seconds)


def _evaluation_command(
    root: Path, output: Path, run_name: str, run: dict[str, Any], *,
    noise_ceiling: Path, allow_code_drift: bool,
) -> list[str]:
    command = [
        sys.executable, "-u", "-m",
        "gen3_multiscale.evaluation.conditional_wae_evaluator",
        "--config", str(run["config"]),
        "--checkpoint-dir", str(run["checkpoint_dir"]),
        "--output", str(output / f"{run_name}_validation.json"),
        "--split", "validation",
        "--n-masks-per-stratum-per-sample", "8",
        "--device", "cuda",
        "--noise-ceiling", str(noise_ceiling),
        "--per-gene-diagnostics-output", str(output / f"{run_name}.per_gene.npz"),
    ]
    if allow_code_drift:
        command.append("--allow-code-drift")
    return command


def run_postrun(
    *, replication_root: str, seed0_evaluation_root: str, noise_ceiling: str,
    output_root: str | None, poll_seconds: int, cpu_threads: int,
    allow_code_drift: bool, dry_run: bool,
) -> dict[str, Any]:
    root = Path(replication_root).expanduser().resolve()
    seed0 = Path(seed0_evaluation_root).expanduser().resolve()
    ceiling = Path(noise_ceiling).expanduser().resolve()
    if not seed0.is_dir() or not ceiling.is_file():
        raise FileNotFoundError(f"seed0 root or noise ceiling is missing: {seed0}, {ceiling}")
    plan = _validate_plan(root)
    _claim_controller(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(output_root).expanduser().resolve() if output_root
        else root / f"evaluation_{stamp}"
    )
    if output.exists():
        raise FileExistsError(f"postrun output already exists: {output}")
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    (output / "analysis").mkdir()
    _atomic_json(output / "postrun_contract.json", {
        "kind": "mk_final_replication_postrun", "version": 1,
        "replication_root": str(root), "seed0_evaluation_root": str(seed0),
        "noise_ceiling": str(ceiling), "expected_fixed_items": 448,
        "expected_whole_slides": 14, "seeds": [0, 1, 2],
        "architectures": list(ARCHITECTURES),
    })
    pointer = root.parent / "LATEST_MK_FINAL_REPLICATION_EVALUATION_ROOT.txt"
    pointer.write_text(str(output) + "\n")

    commands = {
        run_name: _evaluation_command(
            root, output, run_name, plan["runs"][run_name],
            noise_ceiling=ceiling, allow_code_drift=allow_code_drift,
        )
        for run_name in EXPECTED_RUNS
    }
    if dry_run:
        result = {name: {"gpu": plan["runs"][name]["gpu"], "command": command}
                  for name, command in commands.items()}
        _atomic_json(output / "dry_run.json", result)
        return {"dry_run": True, "output_root": str(output), "runs": result}

    wait_for_training(root, plan, poll_seconds)
    status_path = output / "evaluation_status.json"
    status = {
        name: {"gpu": plan["runs"][name]["gpu"], "status": "queued"}
        for name in EXPECTED_RUNS
    }
    _atomic_json(status_path, status)
    processes: dict[str, subprocess.Popen] = {}
    handles = {}
    try:
        for run_name, command in commands.items():
            log = output / "logs" / f"{run_name}_validation.log"
            handle = log.open("a", buffering=1)
            handles[run_name] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(plan["runs"][run_name]["gpu"])
            environment["SCILIFESTDL_CPU_THREADS"] = str(cpu_threads)
            for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            ):
                environment[variable] = str(cpu_threads)
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[2], env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            processes[run_name] = process
            status[run_name].update({
                "status": "running", "pid": process.pid, "log": str(log),
                "started": time.time(), "command": command,
            })
            _atomic_json(status_path, status)
        failures = []
        for run_name, process in processes.items():
            returncode = process.wait()
            status[run_name].update({
                "status": "finished" if returncode == 0 else "failed",
                "returncode": returncode, "finished": time.time(),
            })
            _atomic_json(status_path, status)
            if returncode:
                failures.append(run_name)
        if failures:
            raise RuntimeError(f"replication evaluations failed: {failures}")
    finally:
        for handle in handles.values():
            handle.close()

    records = discover_records(
        seed0_root=seed0, replication_root=root,
        replication_evaluation_root=output,
    )
    summary_paths = summarize(records, output_dir=output / "analysis" / "seed_summary")
    gene_paths = analyze_genes(records, output_dir=output / "analysis" / "gene_robustness")
    result = {
        "ok": True, "output_root": str(output), "evaluation_status": str(status_path),
        "seed_summary": {name: str(path) for name, path in summary_paths.items()},
        "gene_robustness": {name: str(path) for name, path in gene_paths.items()},
    }
    _atomic_json(output / "postrun_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", required=True)
    parser.add_argument("--seed0-evaluation-root", required=True)
    parser.add_argument("--noise-ceiling", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.cpu_threads < 1:
        raise ValueError("poll-seconds and cpu-threads must be positive")
    run_postrun(
        replication_root=args.replication_root,
        seed0_evaluation_root=args.seed0_evaluation_root,
        noise_ceiling=args.noise_ceiling,
        output_root=args.output_root,
        poll_seconds=args.poll_seconds,
        cpu_threads=args.cpu_threads,
        allow_code_drift=args.allow_code_drift,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
