#!/usr/bin/env python3
"""Backfill and follow Gen3--Gen6 scalar histories with TensorBoard.

This process is deliberately read-only with respect to experiment results.  It
parses the trainer's existing text logs and authoritative
``validation_history.json`` files, then writes TensorBoard event files to a
separate directory.  It can therefore be started while training is running
without changing model code, checkpoints, RNG state, or code-drift identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable


_STEP_RE = re.compile(r"^\[step\s+(\d+)\]\s+([^:]+):\s*(.*)$")
_AUTOENCODER_RE = re.compile(
    r"^\[autoencoder epoch\s+(\d+)/(\d+)\]\s+train_mse=([-+0-9.eE]+)"
)
_VALUE_RE = re.compile(
    r"(?:^|,\s*)([A-Za-z][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_ARM_RE = re.compile(r"(?:^|[_-])(gen[3-6][a-z]|arch(?:itecture)?_?\d+)(?:$|[_-])", re.I)


def parse_training_line(line: str) -> list[tuple[str, float, int]]:
    """Return ``(tag, value, step)`` scalars encoded by one log line."""
    stripped = line.strip()
    autoencoder = _AUTOENCODER_RE.match(stripped)
    if autoencoder:
        epoch = int(autoencoder.group(1))
        return [("train/mse", float(autoencoder.group(3)), epoch)]

    match = _STEP_RE.match(stripped)
    if not match:
        return []
    step = int(match.group(1))
    split = match.group(2).strip().lower().replace(" ", "_")
    if split not in {"train", "validation"}:
        return []
    values = []
    for key, raw_value in _VALUE_RE.findall(match.group(3)):
        value = float(raw_value)
        if math.isfinite(value):
            values.append((f"{split}/{key}", value, step))
    return values


def _normalise_arm(value: str) -> str:
    lowered = value.lower().replace("architecture", "arch").replace("_", "")
    return lowered


def run_name_for_path(path: Path, results_root: Path) -> str:
    """Produce a stable ``experiment/arm`` TensorBoard run name."""
    relative = path.resolve().relative_to(results_root.resolve())
    experiment = relative.parts[0]
    candidates = [path.stem, path.parent.name]
    candidates.extend(relative.parts)
    arm = None
    for candidate in candidates:
        checkpoint_match = re.fullmatch(r"checkpoints?[_-](.+)", candidate, re.I)
        if checkpoint_match:
            arm = _normalise_arm(checkpoint_match.group(1))
            break
        padded = f"_{candidate}_"
        match = _ARM_RE.search(padded)
        if match:
            arm = _normalise_arm(match.group(1))
            break
    if arm is None:
        if "autoencoder" in path.name.lower() or "autoencoder" in str(relative).lower():
            arm = "autoencoder"
        else:
            arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    return f"{experiment}/{arm}"


def _file_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()


class TensorBoardHistoryBridge:
    def __init__(
        self,
        results_root: Path,
        output_root: Path,
        *,
        writer_factory: Callable[..., object] | None = None,
    ):
        self.results_root = results_root.resolve()
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_root / "import_state.json"
        self.state = self._load_state()
        if writer_factory is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard is not installed in this Python environment; run "
                    "`python -m pip install tensorboard`"
                ) from exc
            writer_factory = SummaryWriter
        self.writer_factory = writer_factory
        self.writers: dict[str, object] = {}

    def _load_state(self) -> dict:
        if not self.state_path.is_file():
            return {"version": 1, "logs": {}, "validation": {}}
        payload = json.loads(self.state_path.read_text())
        if payload.get("version") != 1:
            raise ValueError(f"unsupported TensorBoard import state version in {self.state_path}")
        payload.setdefault("logs", {})
        payload.setdefault("validation", {})
        return payload

    def _save_state(self) -> None:
        temporary = self.state_path.with_name(f"{self.state_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        os.replace(temporary, self.state_path)

    def _writer(self, run_name: str):
        writer = self.writers.get(run_name)
        if writer is None:
            writer = self.writer_factory(log_dir=str(self.output_root / run_name))
            self.writers[run_name] = writer
        return writer

    def _inside_output(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.output_root)
            return True
        except ValueError:
            return False

    def _sync_log(self, path: Path, *, skip_validation: bool = False) -> int:
        key = _file_key(path)
        offset = int(self.state["logs"].get(key, {}).get("offset", 0))
        size = path.stat().st_size
        if size < offset:  # log was replaced/truncated
            offset = 0
        writer = self._writer(run_name_for_path(path, self.results_root))
        n_scalars = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                for tag, value, step in parse_training_line(line):
                    # validation_history.json is the atomic, authoritative
                    # source when present.  Avoid writing the same
                    # validation/total point once from stdout and again
                    # from JSON at an identical step.
                    if skip_validation and tag.startswith("validation/"):
                        continue
                    writer.add_scalar(tag, value, step)
                    n_scalars += 1
            new_offset = handle.tell()
        self.state["logs"][key] = {"path": str(path.resolve()), "offset": new_offset}
        return n_scalars

    def _sync_validation(self, path: Path) -> int:
        key = _file_key(path)
        imported_steps = set(self.state["validation"].get(key, {}).get("steps", []))
        try:
            entries = json.loads(path.read_text())
        except json.JSONDecodeError:
            return 0  # atomic rewrite may be between rename/visibility on network storage
        if not isinstance(entries, list):
            return 0
        writer = self._writer(run_name_for_path(path, self.results_root))
        n_scalars = 0
        for entry in entries:
            if not isinstance(entry, dict) or "step" not in entry:
                continue
            step = int(entry["step"])
            if step in imported_steps:
                continue
            for name, raw_value in entry.items():
                if name == "step" or not isinstance(raw_value, (int, float)):
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    writer.add_scalar(f"validation/{name}", value, step)
                    n_scalars += 1
            imported_steps.add(step)
        self.state["validation"][key] = {
            "path": str(path.resolve()), "steps": sorted(imported_steps),
        }
        return n_scalars

    def sync_once(self) -> dict:
        n_logs = n_histories = n_scalars = 0
        validation_paths = [
            path for path in sorted(self.results_root.rglob("validation_history.json"))
            if not self._inside_output(path)
        ]
        validation_runs = {
            run_name_for_path(path, self.results_root) for path in validation_paths
        }
        for path in sorted(self.results_root.rglob("*.log")):
            if self._inside_output(path):
                continue
            n_logs += 1
            n_scalars += self._sync_log(
                path,
                skip_validation=run_name_for_path(path, self.results_root) in validation_runs,
            )
        for path in validation_paths:
            n_histories += 1
            n_scalars += self._sync_validation(path)
        for writer in self.writers.values():
            writer.flush()
        self._save_state()
        return {
            "logs_scanned": n_logs,
            "validation_histories_scanned": n_histories,
            "new_scalars": n_scalars,
            "tensorboard_runs": len(self.writers),
        }

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    bridge = TensorBoardHistoryBridge(Path(args.results_root), Path(args.output_root))
    try:
        while True:
            report = bridge.sync_once()
            print(json.dumps(report, sort_keys=True), flush=True)
            if not args.follow:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("TensorBoard history bridge stopped", flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
