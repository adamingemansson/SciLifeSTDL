#!/usr/bin/env python3
"""Run the prepared MK 16-arm suite as two consecutive waves of eight."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _command(record: dict, *, smoke: bool, allow_code_drift: bool) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "gen3_multiscale.training.train_conditional_wae",
        "--config", record["config"],
    ]
    if smoke:
        command.append("--smoke")
    if allow_code_drift:
        command.append("--allow-code-drift")
    return command


def _pid_is_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def run_master_suite(suite_root: str, *, smoke: bool = False,
                     dry_run: bool = False, allow_code_drift: bool = False,
                     resume_controller: bool = False) -> dict:
    root = Path(suite_root).expanduser().resolve()
    plan = json.loads((root / "master_plan.json").read_text())
    if plan.get("kind") != "mk_16_arm_two_wave_suite":
        raise ValueError("master_plan.json is not an MK 16-arm two-wave suite")
    rendered_waves = []
    for wave in plan["waves"]:
        rendered = []
        for record in wave["arms"]:
            config = Path(record["config"])
            if not config.is_file():
                raise FileNotFoundError(config)
            rendered.append({**record, "command": _command(
                record, smoke=smoke, allow_code_drift=allow_code_drift,
            )})
        rendered_waves.append({**wave, "arms": rendered})
    if dry_run:
        return {"dry_run": True, "suite_root": str(root), "waves": rendered_waves}

    status_path = root / "control" / ("smoke_status.json" if smoke else "training_status.json")
    if status_path.exists() and not resume_controller:
        raise FileExistsError(
            f"{status_path} already exists; refusing an accidental duplicate launch. "
            "Use --resume-controller only after verifying the recorded processes ended."
        )
    if status_path.exists():
        status = json.loads(status_path.read_text())
        live = {
            arm: record.get("pid") for arm, record in status.items()
            if record.get("status") == "running" and _pid_is_alive(record.get("pid"))
        }
        if live:
            raise RuntimeError(f"cannot resume while recorded training processes are alive: {live}")
    else:
        status = {
            record["arm"]: {"batch": record["batch"], "gpu": str(record["gpu"]),
                            "wave": wave["name"], "status": "queued"}
            for wave in rendered_waves for record in wave["arms"]
        }
    _atomic_json(status_path, status)
    repo_root = Path(__file__).resolve().parents[2]
    for wave_index, wave in enumerate(rendered_waves):
        pending_records = [
            record for record in wave["arms"]
            if not (
                status.get(record["arm"], {}).get("status") == "finished"
                and status.get(record["arm"], {}).get("returncode") == 0
            )
        ]
        if not pending_records:
            continue
        processes: dict[str, subprocess.Popen] = {}
        handles = {}
        failed = []
        try:
            for record in pending_records:
                arm = record["arm"]
                log = Path(record["log"])
                if smoke:
                    log = log.with_name(f"{log.stem}_smoke{log.suffix}")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = open(log, "a", buffering=1)
                handles[arm] = handle
                environment = os.environ.copy()
                threads = str(plan["cpu_threads_per_arm"])
                environment["CUDA_VISIBLE_DEVICES"] = str(record["gpu"])
                environment["SCILIFESTDL_CPU_THREADS"] = threads
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS"):
                    environment[name] = threads
                process = subprocess.Popen(
                    record["command"], cwd=repo_root, env=environment,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
                processes[arm] = process
                status[arm].update({"status": "running", "pid": process.pid,
                                    "log": str(log), "started": time.time()})
                _atomic_json(status_path, status)
            for arm, process in processes.items():
                returncode = process.wait()
                status[arm].update({
                    "status": "finished" if returncode == 0 else "failed",
                    "returncode": returncode, "finished": time.time(),
                })
                _atomic_json(status_path, status)
                if returncode:
                    failed.append(arm)
        finally:
            for handle in handles.values():
                handle.close()
        if failed:
            for future_wave in rendered_waves[wave_index + 1:]:
                for record in future_wave["arms"]:
                    status[record["arm"]]["status"] = "blocked_by_previous_wave_failure"
            _atomic_json(status_path, status)
            raise RuntimeError(f"{wave['name']} failed; later wave not started: {failed}")
    return {"ok": True, "suite_root": str(root), "status": status}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument(
        "--resume-controller", action="store_true",
        help="resume failed/unfinished arms; refuses if recorded trainer PIDs are still alive",
    )
    args = parser.parse_args()
    print(json.dumps(run_master_suite(
        args.suite_root, smoke=args.smoke, dry_run=args.dry_run,
        allow_code_drift=args.allow_code_drift,
        resume_controller=args.resume_controller,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
