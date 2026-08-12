#!/usr/bin/env python3
"""Print a concise snapshot of a prepared/running MK 16-arm suite."""
from __future__ import annotations

import argparse
from collections import deque
import json
import re
import subprocess
import time
from pathlib import Path


PROGRESS = re.compile(r"^\[step\s+(\d+)\]\s+(train|validation):")


def _latest_progress(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "-", "log not created"
    latest = {"train": None, "validation": None}
    for line in path.read_text(errors="replace").splitlines():
        match = PROGRESS.match(line)
        if match:
            latest[match.group(2)] = match.group(1)
    step = latest["train"] or latest["validation"] or "-"
    detail = f"train={latest['train'] or '-'} val={latest['validation'] or '-'}"
    return step, detail


def monitor_snapshot(suite_root: str, *, smoke: bool = False) -> str:
    root = Path(suite_root).expanduser().resolve()
    plan = json.loads((root / "master_plan.json").read_text())
    status_path = root / "control" / ("smoke_status.json" if smoke else "training_status.json")
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows = []
    for wave in plan["waves"]:
        rows.append(f"===== {wave['name']} =====")
        for record in wave["arms"]:
            arm = record["arm"]
            state = status.get(arm, {}).get("status", "prepared")
            pid = status.get(arm, {}).get("pid", "-")
            log = Path(record["log"])
            if smoke:
                log = log.with_name(f"{log.stem}_smoke{log.suffix}")
            _step, detail = _latest_progress(log)
            rows.append(
                f"{arm:<32} GPU={record['gpu']} PID={pid!s:<8} "
                f"{state:<30} {detail}"
            )
    return "\n".join(rows)


def _process_line(pid: object) -> str:
    try:
        result = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "etime=,%cpu=,%mem=,rss=,stat="],
            capture_output=True, text=True, check=False,
        )
    except (OSError, TypeError, ValueError):
        return "process=not running"
    fields = result.stdout.split()
    if len(fields) < 5:
        return "process=not running"
    elapsed, cpu, memory_percent, rss_kib, state = fields[:5]
    return (
        f"elapsed={elapsed} CPU={cpu}% RAM={float(rss_kib) / 1024 / 1024:.1f}GiB "
        f"MEM={memory_percent}% state={state}"
    )


def _gpu_lines(gpus: set[int]) -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return ["nvidia-smi unavailable"]
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 4 and fields[0].isdigit() and int(fields[0]) in gpus:
            rows.append(
                f"GPU {fields[0]}: {fields[1]}/{fields[2]} MiB, util={fields[3]}%"
            )
    return rows or ["no requested GPUs found"]


def detailed_wave_snapshot(
    suite_root: str, *, wave_number: int, smoke: bool = False, tail_lines: int = 3,
) -> str:
    root = Path(suite_root).expanduser().resolve()
    plan = json.loads((root / "master_plan.json").read_text())
    wave_name = f"wave_{wave_number}"
    try:
        wave = next(item for item in plan["waves"] if item["name"] == wave_name)
    except StopIteration as exc:
        raise ValueError(f"suite has no {wave_name}") from exc
    status_path = root / "control" / ("smoke_status.json" if smoke else "training_status.json")
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    rows = [time.strftime("%Y-%m-%d %H:%M:%S"), f"===== {wave_name.upper()} ====="]
    rows.extend(_gpu_lines({int(record["gpu"]) for record in wave["arms"]}))
    for record in wave["arms"]:
        arm = record["arm"]
        state = status.get(arm, {}).get("status", "prepared")
        pid = status.get(arm, {}).get("pid")
        rows.append(f"\n--- {arm} | GPU={record['gpu']} | {state} ---")
        rows.append(_process_line(pid))
        log = Path(record["log"])
        if smoke:
            log = log.with_name(f"{log.stem}_smoke{log.suffix}")
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
    parser.add_argument("--wave", type=int, choices=(1, 2))
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--tail-lines", type=int, default=3)
    args = parser.parse_args()
    if args.interval <= 0 or args.tail_lines < 1:
        raise SystemExit("--interval and --tail-lines must be positive")
    if args.wave is None:
        print(monitor_snapshot(args.suite_root, smoke=args.smoke))
        return
    while True:
        if args.follow:
            print("\033[2J\033[H", end="")
        print(detailed_wave_snapshot(
            args.suite_root, wave_number=args.wave,
            smoke=args.smoke, tail_lines=args.tail_lines,
        ), flush=True)
        if not args.follow:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
