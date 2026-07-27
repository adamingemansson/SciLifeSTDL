"""Aggregate every architecture's held-out test metrics (written by
evaluate_model_on_mask_bank, one JSON file per test sample per
checkpoint_dir) into a single comparison table -- PCC, RMSE, ST-FID,
ST-MMD, plus any fixed gene panel (e.g. lung_hest_bench_50) PCC that's
present for that architecture.

Reads the per-sample JSON files directly off disk -- does not re-run any
evaluation, purely a summary/reporting step over results that already
exist.

Usage:
    python3 -m gen2_architectures.evaluation.collect_metrics
    python3 -m gen2_architectures.evaluation.collect_metrics --results_dir gen2_architectures/results --csv summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _load_sample_metrics(path: Path) -> dict[str, float] | None:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"skipping unreadable {path}: {type(exc).__name__}: {exc}")
        return None
    primary = payload.get("primary_image_mode")
    image_modes = payload.get("image_modes", {})
    if primary not in image_modes:
        print(f"skipping {path}: no summary for primary_image_mode {primary!r}")
        return None
    summary = image_modes[primary].get("summary", {})
    return {key: value.get("mean", float("nan")) for key, value in summary.items()}


def collect(results_dir: str | Path) -> list[dict]:
    """One row per architecture checkpoint_dir found under results_dir,
    each a mean-of-per-sample-means across every audit_test_metrics_*.json
    file in that directory (a simple, symmetric aggregate -- each held-out
    sample counts equally regardless of how many masks/modes it has)."""
    results_dir = Path(results_dir)
    rows = []
    for checkpoint_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        sample_files = sorted(checkpoint_dir.glob("audit_test_metrics_*.json"))
        if not sample_files:
            continue
        per_sample = [m for m in (_load_sample_metrics(f) for f in sample_files) if m is not None]
        if not per_sample:
            continue
        all_keys = sorted({key for sample in per_sample for key in sample})
        aggregated = {"architecture": checkpoint_dir.name, "n_test_samples": len(per_sample)}
        for key in all_keys:
            values = np.asarray([sample.get(key, np.nan) for sample in per_sample], dtype=float)
            finite = values[np.isfinite(values)]
            aggregated[key] = float(finite.mean()) if finite.size else float("nan")
        rows.append(aggregated)
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("no audit_test_metrics_*.json files found -- nothing to report")
        return
    preferred_order = ["architecture", "n_test_samples", "pcc", "rmse", "st_fid", "st_mmd"]
    all_keys = sorted({key for row in rows for key in row})
    columns = [k for k in preferred_order if k in all_keys] + [
        k for k in all_keys if k not in preferred_order
    ]
    widths = {col: max(len(col), *(len(f"{row.get(col, ''):.4f}" if isinstance(row.get(col), float)
                                     else str(row.get(col, ""))) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            cells.append((f"{value:.4f}" if isinstance(value, float) else str(value)).ljust(widths[col]))
        print("  ".join(cells))


def main(results_dir: str, csv_path: str | None) -> None:
    rows = collect(results_dir)
    _print_table(rows)
    if csv_path and rows:
        all_keys = sorted({key for row in rows for key in row})
        fieldnames = ["architecture", "n_test_samples"] + [
            k for k in all_keys if k not in ("architecture", "n_test_samples")
        ]
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="gen2_architectures/results",
                         help="Directory containing one subdirectory per architecture's checkpoint_dir.")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to also write a CSV.")
    args = parser.parse_args()
    main(args.results_dir, args.csv)
