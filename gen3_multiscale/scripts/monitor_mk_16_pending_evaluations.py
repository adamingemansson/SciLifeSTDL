#!/usr/bin/env python3
"""Compact terminal monitor for the unified 16-arm MK evaluation."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _progress(log: Path) -> str:
    if not log.is_file():
        return "initializing"
    fixed = whole = None
    last = "initializing"
    for line in log.read_text(errors="replace").splitlines():
        match = re.search(r"conditional WAE evaluation progress: (\d+)/(\d+)", line)
        if match:
            fixed = f"{match.group(1)}/{match.group(2)}"
            last = line
        match = re.search(r"structured whole-slide evaluation progress: (\d+)/(\d+)", line)
        if match:
            whole = f"{match.group(1)}/{match.group(2)}"
            last = line
        if any(token in line for token in ("report saved", "Traceback", "Error")):
            last = line
    if whole:
        return f"whole={whole}"
    if fixed:
        return f"fixed={fixed}"
    return last[-100:]


def snapshot(root: Path) -> str:
    contract = json.loads((root / "evaluation_contract.json").read_text())
    status_path = root / "evaluation_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    lines = [datetime.now().astimezone().strftime("%a %d %b %Y %H:%M:%S %Z")]
    for job in contract["jobs"]:
        arm = job["arm"]
        row = status.get(arm) or {}
        state = row.get("status", "queued")
        pid = row.get("pid")
        alive = _alive(pid)
        log = Path(row.get("log") or root / "logs" / f"{arm}_validation.log")
        lines.append(
            f"{arm:<38} GPU={row.get('gpu', job.get('gpu', '-'))} "
            f"{state:<12} alive={str(alive):<5} {_progress(log)}"
        )
    counts = {}
    for row in status.values():
        counts[row.get("status", "queued")] = counts.get(row.get("status", "queued"), 0) + 1
    lines.append(f"Summary: {counts}")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True,
        ).stdout.strip()
        if gpu:
            lines += ["", "GPU MiB used/total, utilization:", gpu]
    except FileNotFoundError:
        pass
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args()
    root = Path(args.evaluation_root).expanduser().resolve()
    if args.interval < 0:
        raise ValueError("interval cannot be negative")
    while True:
        if args.interval:
            print("\033[2J\033[H", end="")
        print(snapshot(root), flush=True)
        if not args.interval:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
