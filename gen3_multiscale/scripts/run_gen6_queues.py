#!/usr/bin/env python3
"""Run prepared Gen6 configs on fixed GPU queues.

One worker process owns each GPU and runs its assigned arms sequentially;
different GPUs run concurrently.  The command itself should be launched
with nohup if terminal detachment is desired.  No tmux/watch dependency.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def run_queues(suite_root: str, gpus: list[str], arms: list[str], *, dry_run=False) -> dict:
    root = Path(suite_root).resolve()
    if not gpus or not arms:
        raise ValueError("at least one GPU and arm are required")
    queues = {gpu: [] for gpu in gpus}
    for index, arm in enumerate(arms):
        config = root / "configs" / f"{arm}.yaml"
        if not config.is_file():
            raise FileNotFoundError(f"missing prepared config: {config}")
        queues[gpus[index % len(gpus)]].append(arm)
    plan = {"suite_root": str(root), "queues": queues, "dry_run": bool(dry_run)}
    _atomic_json(root / "queue_plan.json", plan)
    if dry_run:
        return plan
    state_path = root / "queue_status.json"
    lock = threading.Lock()
    state = {arm: {"status": "queued"} for arm in arms}

    def update(arm, **values):
        with lock:
            state[arm].update(values)
            _atomic_json(state_path, state)

    def worker(gpu: str, queue: list[str]):
        for arm in queue:
            config = root / "configs" / f"{arm}.yaml"
            log_path = root / "logs" / f"{arm}.log"
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
            command = [sys.executable, "-u", "-m", "gen3_multiscale.training.train", "--config", str(config)]
            update(arm, status="running", gpu=gpu, started=time.time(), command=command, log=str(log_path))
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                update(arm, pid=process.pid)
                returncode = process.wait()
            update(arm, status="finished" if returncode == 0 else "failed",
                   returncode=returncode, finished=time.time())
            if returncode != 0:
                # Fail this GPU queue closed; other independent GPU queues
                # continue and their states remain observable.
                for remaining in queue[queue.index(arm) + 1:]:
                    update(remaining, status="blocked_by_previous_failure", gpu=gpu)
                break

    threads = [threading.Thread(target=worker, args=(gpu, queue), daemon=False)
               for gpu, queue in queues.items() if queue]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {"queues": queues, "status": state}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--arms", default="gen6a,gen6b,gen6c,gen6d,gen6e,gen6f,gen6g,gen6h,gen6i,gen6j")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_queues(
        args.suite_root, [value.strip() for value in args.gpus.split(",") if value.strip()],
        [value.strip() for value in args.arms.split(",") if value.strip()], dry_run=args.dry_run,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
