#!/usr/bin/env python3
"""Step 6 deliverable: progress-check command.

A fast, read-only glance at one architecture's checkpoint_dir while a real
run is in progress (or between runs) -- current step, whether the last
loss/gradient was finite, and how many checkpoint history snapshots exist.
Reads only the small JSON artifacts train.py already writes
(training_state.json, run_manifest.json, preflight_report.json); never
loads model weights, never touches GPU.

    python -m gen3_multiscale.scripts.step6_progress --checkpoint-dir gen3_multiscale/results/architecture1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen3_multiscale.training.checkpoint import list_checkpoint_history, load_training_state


def gather_progress(checkpoint_dir: str | Path) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    training_state = load_training_state(checkpoint_dir)
    history_steps = list_checkpoint_history(checkpoint_dir)

    run_manifest = None
    run_manifest_path = checkpoint_dir / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text())

    preflight_report = None
    preflight_path = checkpoint_dir / "preflight_report.json"
    if preflight_path.is_file():
        preflight_report = json.loads(preflight_path.read_text())

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "step": training_state.get("step", 0),
        "n_skipped_nonfinite": training_state.get("n_skipped_nonfinite"),
        "checkpoint_history_steps": history_steps,
        "n_checkpoint_snapshots_kept": len(history_steps),
        "has_run_manifest": run_manifest is not None,
        "architecture": (run_manifest or {}).get("model_architecture"),
        "n_train_samples": len((run_manifest or {}).get("split", {}).get("train_sample_ids", [])),
        "n_validation_samples": len((run_manifest or {}).get("split", {}).get("validation_sample_ids", [])),
        "preflight_passed": (preflight_report or {}).get("passed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    progress = gather_progress(args.checkpoint_dir)
    print(json.dumps(progress, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
