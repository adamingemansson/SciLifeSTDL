#!/usr/bin/env python3
"""Minimal terminal monitor for the MK specialist-WAE screen."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def _alive(pid) -> bool:
    if not pid:
        return False
    return subprocess.run(
        ["kill", "-0", str(pid)], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _ram_gib(pid) -> str:
    if not pid or not _alive(pid):
        return "-"
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "rss="], capture_output=True, text=True,
        check=False,
    )
    try:
        return f"{int(result.stdout.strip()) / 1024**2:.1f}GiB"
    except ValueError:
        return "-"


def render(root: Path) -> None:
    plan = json.loads((root / "suite_plan.json").read_text())
    status_path = root / "training_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    for arm in plan["arm_order"]:
        row = status.get(arm, {})
        pid = row.get("pid")
        state = row.get("status", "not_started")
        if state == "running" and not _alive(pid):
            state = "ended"
        log = Path(row.get("log") or root / "logs" / f"{arm}.log")
        lines = log.read_text(errors="replace").splitlines()[-3:] if log.is_file() else []
        print(
            f"\n{arm} | GPU={plan['arms'][arm]['gpu']} | {state} | "
            f"PID={pid or '-'} | RAM={_ram_gib(pid)}"
        )
        for line in lines:
            print(f"  {line}")
    print("\nGPUs 0,2,3,5")
    subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits", "-i", "0,2,3,5",
    ], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--interval", type=float, default=0.0)
    args = parser.parse_args()
    root = Path(args.suite_root).expanduser().resolve()
    while True:
        render(root)
        if args.interval <= 0:
            break
        print(f"\nRefreshing in {args.interval:g}s; Ctrl+C stops only this monitor")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
