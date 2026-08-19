#!/usr/bin/env python3
"""Run a prepared MK specialist-WAE suite, one arm per GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def run(root: Path, *, smoke: bool = False, dry_run: bool = False) -> dict:
    root = root.expanduser().resolve()
    plan = json.loads((root / "suite_plan.json").read_text())
    if plan.get("kind") != "mk_specialist_wae_suite":
        raise ValueError("suite_plan.json is not an MK specialist-WAE suite")
    commands = []
    for arm in plan["arm_order"]:
        record = plan["arms"][arm]
        command = [sys.executable, "-u", "-m",
                   "gen3_multiscale.training.train_conditional_wae",
                   "--config", record["config"]]
        if smoke:
            command.append("--smoke")
        commands.append((arm, str(record["gpu"]), command))
    if dry_run:
        return {"dry_run": True, "root": str(root), "commands": commands}

    status_path = root / ("smoke_status.json" if smoke else "training_status.json")
    status = {arm: {"gpu": gpu, "status": "queued"} for arm, gpu, _ in commands}
    _write(status, status_path)
    processes, handles = {}, {}
    try:
        for arm, gpu, command in commands:
            log = root / "logs" / f"{arm}{'_smoke' if smoke else ''}.log"
            handle = open(log, "a", buffering=1)
            handles[arm] = handle
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            threads = str(plan["cpu_threads_per_arm"])
            environment["SCILIFESTDL_CPU_THREADS"] = threads
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                environment[name] = threads
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[2], env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            processes[arm] = process
            status[arm].update(status="running", pid=process.pid, log=str(log),
                               started=time.time())
            _write(status, status_path)
        failures = []
        for arm, process in processes.items():
            returncode = process.wait()
            status[arm].update(
                status="finished" if returncode == 0 else "failed",
                returncode=returncode, finished=time.time(),
            )
            if returncode:
                failures.append(arm)
            _write(status, status_path)
        if failures:
            raise RuntimeError(f"specialist-WAE arms failed: {failures}")
        return {"ok": True, "root": str(root), "status": status}
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.suite_root), smoke=args.smoke,
                         dry_run=args.dry_run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
