#!/usr/bin/env python3
"""Minimal live monitor for the eight-arm MK gene/field screen."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


STEP = re.compile(r"^\[step\s+(\d+)\].*(?:train|validation):")


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _latest(log: Path, limit: int = 3) -> list[str]:
    if not log.is_file():
        return ["log not created"]
    lines = log.read_text(errors="replace").splitlines()
    selected = [
        line for line in lines
        if STEP.search(line) or "training finished:" in line
        or "Traceback" in line or "Error:" in line
    ]
    return selected[-limit:] or lines[-limit:] or ["log empty"]


def snapshot(root: Path) -> str:
    plan = json.loads((root / "suite_plan.json").read_text())
    status_path = root / "training_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows = []
    for arm in plan["arm_order"]:
        item = status.get(arm, {})
        pid = item.get("pid")
        state = item.get("status", "queued")
        live = _alive(pid)
        log = Path(item.get("log") or root / "logs" / f"{arm}.log")
        rows.append(
            f"\n----- {arm} | GPU={plan['arms'][arm]['gpu']} | "
            f"{state} | PID={pid or '-'} | alive={live} -----"
        )
        rows.extend(_latest(log))
    rows.append("\n===== GPUs 0,2,3,5 =====")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], check=False, capture_output=True, text=True,
        ).stdout.splitlines()
        rows.extend(line for line in gpu if line.split(",", 1)[0].strip() in {"0", "2", "3", "5"})
    except FileNotFoundError:
        rows.append("nvidia-smi unavailable")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        print("\033[2J\033[H", end="")
        print(time.strftime("%Y-%m-%d %H:%M:%S %Z"))
        print(snapshot(args.suite_root.expanduser().resolve()), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
