#!/usr/bin/env python3
"""Step 6 deliverable: result-summary command.

A fuller, human-readable report of one completed (or in-progress)
architecture's run than step6_progress.py's quick glance: the run
manifest's fingerprints (config, dataset manifest, gene panel), split
sizes, mask-schedule pass/fail counts, cache-preflight status, checkpoint
history, and final training state -- everything a reader would otherwise
have to reconstruct by hand-reading three separate JSON files.

    python -m gen3_multiscale.scripts.step6_summary --checkpoint-dir gen3_multiscale/results/architecture1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen3_multiscale.training.checkpoint import list_checkpoint_history, load_training_state


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def build_summary(checkpoint_dir: str | Path) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    run_manifest = _load_json(checkpoint_dir / "run_manifest.json")
    preflight_report = _load_json(checkpoint_dir / "preflight_report.json")
    training_state = load_training_state(checkpoint_dir)
    history_steps = list_checkpoint_history(checkpoint_dir)

    mask_reports = (run_manifest or {}).get("mask_schedule_reports", {})

    def _pass_fail_counts(reports_by_sample: dict) -> dict:
        passed = sum(1 for r in reports_by_sample.values() if r.get("passed"))
        return {"n_samples": len(reports_by_sample), "n_passed": passed, "n_failed": len(reports_by_sample) - passed}

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "architecture": (run_manifest or {}).get("model_architecture"),
        "config_path": (run_manifest or {}).get("config_path"),
        "config_fingerprint": (run_manifest or {}).get("config_fingerprint"),
        "dataset_manifest_fingerprint": (run_manifest or {}).get("dataset_manifest_fingerprint"),
        "gene_panel_hash": (run_manifest or {}).get("gene_panel_hash"),
        "n_genes": (run_manifest or {}).get("n_genes"),
        "seed": (run_manifest or {}).get("seed"),
        "split": {
            "n_train_samples": len((run_manifest or {}).get("split", {}).get("train_sample_ids", [])),
            "n_validation_samples": len((run_manifest or {}).get("split", {}).get("validation_sample_ids", [])),
            "n_test_samples": len((run_manifest or {}).get("split", {}).get("test_sample_ids", [])),
        },
        "cache_preflight_passed": (preflight_report or {}).get("passed"),
        "cache_preflight_n_samples": (preflight_report or {}).get("n_samples"),
        "train_mask_schedule": _pass_fail_counts(mask_reports.get("train", {})),
        "validation_mask_schedule": _pass_fail_counts(mask_reports.get("validation", {})),
        "synchronized_init_manifest_path": (run_manifest or {}).get("synchronized_init_manifest_path"),
        "current_step": training_state.get("step", 0),
        "n_skipped_nonfinite": training_state.get("n_skipped_nonfinite"),
        "checkpoint_history_steps": history_steps,
        "n_checkpoint_snapshots_kept": len(history_steps),
        "has_trainable_weights": (checkpoint_dir / "trainable_weights.pt").is_file(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build_summary(args.checkpoint_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
