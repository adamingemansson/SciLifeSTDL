#!/usr/bin/env python3
"""Summarize the four matched hierarchical missing-tissue runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = (
    "hierarchical_slide_full_seed10",
    "hierarchical_no_slide_seed10",
    "hierarchical_no_novae_seed10",
    "hierarchical_he_only_seed10",
    "hierarchical_harmonic_k128",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/checkpoints/recovery_suite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for run in RUNS:
        path = Path(args.checkpoint_root) / run / "heldout_sample_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed held-out report: {path}")
        payload = json.loads(path.read_text())
        mode = str(payload["primary_image_mode"])
        summary = payload["image_modes"][mode]
        rows.append({
            "experiment_name": run,
            "primary_image_mode": mode,
            "context_gex_mode": payload["context_gex_mode"],
            "modality_ablation": payload["modality_ablation"],
            "pcc": summary["pcc_mean"],
            "rmse": summary["rmse_mean"],
            "nonzero_auc": summary["nonzero_auc_mean"],
            "st_fid": summary["st_fid_mean"],
            "st_mmd": summary["st_mmd_mean"],
            "spatial_domain_plausibility": summary["spatial_domain_plausibility_mean"],
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
