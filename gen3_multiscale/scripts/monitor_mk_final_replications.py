#!/usr/bin/env python3
"""Compact one-shot status for MK finalist training or postrun evaluation."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


RUNS = (
    "mk_wb_parallel_gated_seed1", "mk_wb_parallel_gated_seed2",
    "mk_wbw_sandwich_seed1", "mk_wbw_sandwich_seed2",
)


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _last_match(path: Path, patterns: tuple[str, ...]) -> str:
    if not path.is_file():
        return "initializing"
    lines = path.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if any(re.search(pattern, line) for pattern in patterns):
            return line[-180:]
    return "initializing"


def snapshot(root: Path) -> list[str]:
    plan = json.loads((root / "replication_plan.json").read_text())
    evaluation_pointer = root.parent / "LATEST_MK_FINAL_REPLICATION_EVALUATION_ROOT.txt"
    evaluation_root = None
    status = {}
    if evaluation_pointer.is_file():
        candidate = Path(evaluation_pointer.read_text().strip())
        if candidate.is_dir() and (candidate / "postrun_contract.json").is_file():
            contract = json.loads((candidate / "postrun_contract.json").read_text())
            if Path(contract.get("replication_root", "")) == root:
                evaluation_root = candidate
                status_path = candidate / "evaluation_status.json"
                if status_path.is_file():
                    status = json.loads(status_path.read_text())
    lines = []
    for run_name in RUNS:
        gpu = plan["runs"][run_name]["gpu"]
        eval_record = status.get(run_name) or {}
        if eval_record:
            pid = eval_record.get("pid")
            state = eval_record.get("status", "queued")
            log = evaluation_root / "logs" / f"{run_name}_validation.log"
            progress = _last_match(log, (
                r"conditional WAE evaluation progress:",
                r"structured whole-slide evaluation progress:",
                r"evaluation report saved", r"Traceback", r"Error",
            ))
            lines.append(f"{run_name:<38} GPU={gpu} eval={state:<8} alive={_alive(pid)} | {progress}")
            continue
        pid_path = root / "control" / f"{run_name}.pid"
        try:
            pid = int(pid_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            pid = None
        log = root / "logs" / f"{run_name}.log"
        progress = _last_match(log, (
            r"^\[step \d+\].*(train|validation):", r"checkpoint saved",
            r"training finished:", r"Traceback", r"Error",
        ))
        state = "running" if _alive(pid) else "ended"
        lines.append(f"{run_name:<38} GPU={gpu} train={state:<7} PID={pid or '-'} | {progress}")
    if evaluation_root:
        lines.append(f"evaluation_root={evaluation_root}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", required=True)
    args = parser.parse_args()
    print("\n".join(snapshot(Path(args.replication_root).expanduser().resolve())))


if __name__ == "__main__":
    main()
