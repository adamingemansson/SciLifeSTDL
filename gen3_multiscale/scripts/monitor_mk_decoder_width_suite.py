#!/usr/bin/env python3
"""Print one compact snapshot for an MK decoder-width suite."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


STEP = re.compile(r"\[step\s+(\d+)\]")


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True)
    parser.add_argument("--lines", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.suite_root).expanduser().resolve()
    plan = json.loads((root / "suite_plan.json").read_text())
    status_path = root / "training_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for arm in plan["arm_order"]:
        item = status.get(arm, {"status": "not_started"})
        log = root / "logs" / f"{arm}.log"
        lines = log.read_text(errors="replace").splitlines() if log.exists() else []
        steps = [int(match.group(1)) for line in lines if (match := STEP.search(line))]
        pid = item.get("pid")
        state = item.get("status", "unknown")
        if state == "running" and not _alive(pid):
            state = "ended_unrecorded"
        print(
            f"{arm:<28} GPU={plan['arms'][arm]['gpu']} {state:<16} "
            f"PID={pid or '-'} step={max(steps) if steps else '-'}"
        )
        for line in lines[-max(0, args.lines):]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
