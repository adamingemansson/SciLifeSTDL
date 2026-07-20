#!/usr/bin/env python3
"""Create a wide comparison table across all H&E availability modes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MODES = ("full", "target_zero", "all_zero", "shuffled")


def metric_mean(mode_payload: dict, name: str):
    value = mode_payload.get("summary", {}).get(name)
    return value.get("mean") if isinstance(value, dict) else value


def collect(root: Path) -> list[dict]:
    rows = []
    for exp_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        metrics_path = exp_dir / "audit_test_metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text())
        row = {"experiment_name": payload.get("experiment_name") or exp_dir.name}
        modes = payload.get("image_modes", {})
        for mode in MODES:
            mode_payload = modes.get(mode, {})
            for metric in ("pcc", "rmse", "st_fid", "predictive_std", "interval90_coverage"):
                row[f"{mode}_{metric}"] = metric_mean(mode_payload, metric)
        if row.get("full_pcc") is not None and row.get("shuffled_pcc") is not None:
            row["matched_image_pcc_gain"] = row["full_pcc"] - row["shuffled_pcc"]
        if row.get("full_rmse") is not None and row.get("target_zero_rmse") is not None:
            row["target_missing_rmse_penalty"] = row["target_zero_rmse"] - row["full_rmse"]
        if row.get("full_rmse") is not None and row.get("all_zero_rmse") is not None:
            row["all_missing_rmse_penalty"] = row["all_zero_rmse"] - row["full_rmse"]
        rows.append(row)
    return sorted(rows, key=lambda row: (row.get("full_rmse") is None, row.get("full_rmse", float("inf"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/checkpoints/complexity_ladder"))
    parser.add_argument("--output", type=Path, default=Path("reports/complexity_ladder/latest_wide_summary.csv"))
    args = parser.parse_args()
    rows = collect(args.checkpoint_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment_name"] + sorted({key for row in rows for key in row if key != "experiment_name"})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} ({len(rows)} completed experiments)")
    print("\nTop completed runs by full-image RMSE:")
    for row in rows[:10]:
        print(
            f"{row['experiment_name']:<48} "
            f"PCC={row.get('full_pcc', float('nan')):.4f} "
            f"RMSE={row.get('full_rmse', float('nan')):.4f} "
            f"target0={row.get('target_zero_rmse', float('nan')):.4f} "
            f"all0={row.get('all_zero_rmse', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
