#!/usr/bin/env python3
"""Run an immutable four-arm MK architecture suite."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _write_status(path: Path, status: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True))
    os.replace(temporary, path)


def run_suite(suite_root: str, *, smoke: bool = False, dry_run: bool = False,
              allow_code_drift: bool = False) -> dict:
    root = Path(suite_root).resolve()
    plan = json.loads((root / "suite_plan.json").read_text())
    if plan.get("kind") not in {"mk_four_architecture_suite", "mk_structured_field_suite"}:
        raise ValueError("suite_plan.json is not a supported MK four-architecture suite")
    commands = {}
    for arm in plan["arm_order"]:
        record = plan["arms"][arm]
        config = Path(record["config"])
        if not config.is_file():
            raise FileNotFoundError(config)
        command = [
            sys.executable, "-u", "-m",
            "gen3_multiscale.training.train_conditional_wae",
            "--config", str(config),
        ]
        if smoke:
            command.append("--smoke")
        if allow_code_drift:
            command.append("--allow-code-drift")
        commands[arm] = {"gpu": str(record["gpu"]), "command": command}
    if dry_run:
        return {"dry_run": True, "suite_root": str(root), "arms": commands}

    status_path = root / ("smoke_status.json" if smoke else "training_status.json")
    status = {arm: {"gpu": item["gpu"], "status": "queued"} for arm, item in commands.items()}
    _write_status(status_path, status)
    processes, handles = {}, {}
    try:
        for arm, record in commands.items():
            log = root / "logs" / f"{arm}{'_smoke' if smoke else ''}.log"
            handle = open(log, "a", buffering=1)
            handles[arm] = handle
            environment = os.environ.copy()
            threads = str(plan["cpu_threads_per_arm"])
            environment["CUDA_VISIBLE_DEVICES"] = record["gpu"]
            environment["SCILIFESTDL_CPU_THREADS"] = threads
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[name] = threads
            process = subprocess.Popen(
                record["command"], cwd=Path(__file__).resolve().parents[2],
                env=environment, stdout=handle, stderr=subprocess.STDOUT,
            )
            processes[arm] = process
            status[arm].update({
                "status": "running", "pid": process.pid,
                "log": str(log), "started": time.time(),
            })
            _write_status(status_path, status)
        failed = []
        for arm, process in processes.items():
            returncode = process.wait()
            status[arm].update({
                "status": "finished" if returncode == 0 else "failed",
                "returncode": returncode, "finished": time.time(),
            })
            _write_status(status_path, status)
            if returncode:
                failed.append(arm)
        if failed:
            raise RuntimeError(f"MK architecture arms failed: {failed}")
        return {"ok": True, "suite_root": str(root), "status": status}
    finally:
        for handle in handles.values():
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_suite(
        args.suite_root, smoke=args.smoke, dry_run=args.dry_run,
        allow_code_drift=args.allow_code_drift,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
