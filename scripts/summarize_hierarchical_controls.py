#!/usr/bin/env python3
"""Summarize the four exact-mask hierarchical parallel controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = (
    "hierarchical_control_coordinate_only_seed10",
    "hierarchical_control_local_he_raw_gex_seed10",
    "hierarchical_control_global_he_raw_gex_seed10",
    "hierarchical_control_official_stpath",
)
GENE_PANELS = (
    "stpath_hest_bench_ccrcc_50",
    "train_variance_top50",
    "train_variance_top100",
    "train_variance_top250",
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
        row = {
            "experiment_name": run,
            "primary_image_mode": mode,
            "n_evaluated_genes": payload["n_evaluated_genes"],
            "n_shared_training_genes": payload.get(
                "n_shared_training_genes", payload["n_evaluated_genes"]
            ),
            "pcc": summary["pcc_mean"],
            "rmse": summary["rmse_mean"],
            "nonzero_auc": summary["nonzero_auc_mean"],
            "st_fid": summary["st_fid_mean"],
            "st_mmd": summary["st_mmd_mean"],
        }
        for panel in GENE_PANELS:
            panel_meta = payload["gene_panels"][panel]
            row[f"evaluated_genes_{panel}"] = panel_meta["evaluated_count"]
            row[f"pcc_{panel}"] = summary[f"pcc_{panel}_mean"]
            row[f"rmse_{panel}"] = summary[f"rmse_{panel}_mean"]
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
