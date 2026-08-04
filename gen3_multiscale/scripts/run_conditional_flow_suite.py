#!/usr/bin/env python3
"""Launch the four prepared MK conditional-flow arms on distinct GPUs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gen3_multiscale.conditional_flow.contract import ARM_SPECS


def _atomic_status(status: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(temporary, path)


def run_suite(
    suite_root: str,
    gpus: list[str],
    *,
    smoke: bool,
    dry_run: bool,
    cpu_threads: int = 8,
    allow_code_drift: bool = False,
) -> dict:
    root = Path(suite_root).resolve()
    arms = list(ARM_SPECS)
    if len(gpus) != len(arms) or len(set(gpus)) != len(gpus):
        raise ValueError(f"provide exactly {len(arms)} distinct GPU indices")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    commands = {}
    for arm, gpu in zip(arms, gpus):
        config = root / "configs" / f"{arm}.yaml"
        if not config.is_file():
            raise FileNotFoundError(config)
        command = [
            sys.executable, "-u", "-m",
            "gen3_multiscale.training.train_conditional_flow",
            "--config", str(config),
        ]
        if smoke:
            command.append("--smoke")
        if allow_code_drift:
            command.append("--allow-code-drift")
        commands[arm] = {"gpu": gpu, "command": command}
    if dry_run:
        return {"dry_run": True, "suite_root": str(root), "arms": commands}
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = root / ("smoke_status.json" if smoke else "training_status.json")
    status = {arm: {"gpu": row["gpu"], "status": "queued"} for arm, row in commands.items()}
    _atomic_status(status, status_path)
    processes, handles = {}, {}
    try:
        for arm, row in commands.items():
            log_path = logs / f"{arm}{'_smoke' if smoke else ''}.log"
            handle = open(log_path, "a", buffering=1)
            handles[arm] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = row["gpu"]
            environment["SCILIFESTDL_CPU_THREADS"] = str(cpu_threads)
            for variable in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            ):
                environment[variable] = str(cpu_threads)
            process = subprocess.Popen(
                row["command"], cwd=Path(__file__).resolve().parents[2],
                env=environment, stdout=handle, stderr=subprocess.STDOUT,
            )
            processes[arm] = process
            status[arm].update({
                "status": "running", "pid": process.pid, "log": str(log_path),
                "started": time.time(),
            })
            _atomic_status(status, status_path)
        failures = []
        for arm, process in processes.items():
            returncode = process.wait()
            status[arm].update({
                "status": "finished" if returncode == 0 else "failed",
                "returncode": returncode, "finished": time.time(),
            })
            if returncode:
                failures.append(arm)
            _atomic_status(status, status_path)
        if failures:
            raise RuntimeError(f"conditional-flow arms failed: {failures}")
        return {"ok": True, "suite_root": str(root), "status": status}
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--gpus", required=True, help="Four comma-separated GPU indices")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_suite(
        args.suite_root,
        [part.strip() for part in args.gpus.split(",") if part.strip()],
        smoke=args.smoke, dry_run=args.dry_run,
        cpu_threads=args.cpu_threads,
        allow_code_drift=args.allow_code_drift,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
