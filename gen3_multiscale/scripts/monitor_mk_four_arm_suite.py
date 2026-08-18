#!/usr/bin/env python3
"""Concise terminal monitor for a prepared MK four-arm suite."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import subprocess
import time
from pathlib import Path


_STEP = re.compile(r"^\[step\s+(\d+)\]\s+(train|validation):")


def _alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _latest_step(log: Path) -> str:
    if not log.is_file():
        return "-"
    step = "-"
    with log.open(errors="replace") as handle:
        for line in handle:
            match = _STEP.match(line)
            if match:
                step = match.group(1)
    return step


def _gpu_rows(gpus: set[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit() and int(fields[0]) in gpus:
            rows.append(
                f"GPU {fields[0]}  {fields[1]}/{fields[2]} MiB  util={fields[3]}%"
            )
    return rows


def snapshot(root: Path, *, smoke: bool = False, tail_lines: int = 3) -> str:
    plan = json.loads((root / "suite_plan.json").read_text())
    status_name = "smoke_status.json" if smoke else "training_status.json"
    status_path = root / status_name
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows = [time.strftime("%Y-%m-%d %H:%M:%S"), "===== MK FOUR-ARM SUITE ====="]
    rows.extend(_gpu_rows({int(plan["arms"][arm]["gpu"]) for arm in plan["arm_order"]}))
    for arm in plan["arm_order"]:
        record = plan["arms"][arm]
        state = status.get(arm, {})
        pid = state.get("pid")
        log = root / "logs" / f"{arm}{'_smoke' if smoke else ''}.log"
        recorded = state.get("status", "prepared")
        rows.append(
            f"\n--- {arm} | GPU={record['gpu']} | {recorded} | "
            f"alive={_alive(pid)} | step={_latest_step(log)} ---"
        )
        if log.is_file():
            with log.open(errors="replace") as handle:
                rows.extend(line.rstrip("\n") for line in deque(handle, maxlen=tail_lines))
        else:
            rows.append("log not created")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--tail-lines", type=int, default=3)
    args = parser.parse_args()
    if args.interval <= 0 or args.tail_lines < 1:
        raise SystemExit("--interval and --tail-lines must be positive")
    root = Path(args.suite_root).expanduser().resolve()
    while True:
        if args.follow:
            print("\033[2J\033[H", end="")
        print(snapshot(root, smoke=args.smoke, tail_lines=args.tail_lines), flush=True)
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
