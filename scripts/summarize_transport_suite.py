#!/usr/bin/env python3
"""Summarize the hierarchical_gene_transport_regressor 20-run suite.

Two phases share one CSV, distinguished by the ``phase`` column:
  * ``capacity`` -- O01-O04, reads each run's ``quality_gate.json``
    (evaluation is disabled for these, by design -- see
    docs/hierarchical_missing_tissue.md and the 2026-07-22 handoff's
    "Capacity criteria").
  * ``heldout``  -- C01-C16 plus the harmonic k128 control, reads each run's
    ``heldout_sample_summary.json`` (train.py's own final report), mirroring
    scripts/summarize_hierarchical_slide.py's column layout exactly so the
    two suites stay directly comparable.

Run after (or during, for a partial view) scripts/run_transport_suite_4gpu.sh;
rows for runs that haven't finished yet are still written, marked status="not_run".
"""
# (no functional change -- this line exists only to verify GITHUB_PUSH_PAT-based
# push access from a resumed session; safe to remove on the next real edit.)
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CAPACITY_RUNS = (
    "transport_capacity_o01_full_concat_no_residual",
    "transport_capacity_o02_full_concat_residual64",
    "transport_capacity_o03_gex_novae_no_he",
    "transport_capacity_o04_gated_experts_residual64",
)

HELDOUT_RUNS = (
    "transport_c01_raw_gex_only_20k",
    "transport_c02_gex_novae_20k",
    "transport_c03_gex_local_he_20k",
    "transport_c04_gex_global_he_20k",
    "transport_c05_all_modalities_concat_k128_20k",
    "transport_c06_all_modalities_gated_experts_20k",
    "transport_c07_geometry_only_scoring_20k",
    "transport_c08_shared_gene_gate_20k",
    "transport_c09_per_gene_gate_no_query_20k",
    "transport_c10_one_transport_head_20k",
    "transport_c11_four_transport_heads_20k",
    "transport_c12_sixteen_transport_heads_20k",
    "transport_c13_k32_20k",
    "transport_c14_k64_20k",
    "transport_c15_residual_rank32_20k",
    "transport_c16_residual_rank64_20k",
    "transport_harmonic_k128",
)

GENE_PANELS = (
    "stpath_hest_bench_ccrcc_50",
    "train_variance_top50",
    "train_variance_top100",
    "train_variance_top250",
)

FIELDNAMES = [
    "phase", "experiment_name", "status",
    "capacity_passed", "capacity_metric", "capacity_best_score",
    "capacity_anchor_score", "capacity_correction_rms",
    "primary_image_mode", "context_gex_mode", "modality_ablation",
    "pcc", "rmse", "nonzero_auc", "st_fid", "st_mmd",
    "spatial_domain_plausibility",
]
for _panel in GENE_PANELS:
    FIELDNAMES += [f"pcc_{_panel}", f"rmse_{_panel}", f"n_pcc_genes_{_panel}"]


def _capacity_row(checkpoint_root: Path, run: str) -> dict:
    row = {name: "" for name in FIELDNAMES}
    row.update({"phase": "capacity", "experiment_name": run})
    path = checkpoint_root / run / "quality_gate.json"
    if not path.is_file():
        row["status"] = "not_run"
        return row
    payload = json.loads(path.read_text())
    row.update({
        "status": "complete",
        "capacity_passed": payload.get("passed"),
        "capacity_metric": payload.get("metric"),
        "capacity_best_score": payload.get("best_score"),
        "capacity_anchor_score": payload.get("anchor_score"),
        "capacity_correction_rms": payload.get("correction_rms"),
    })
    return row


def _heldout_row(checkpoint_root: Path, run: str) -> dict:
    row = {name: "" for name in FIELDNAMES}
    row.update({"phase": "heldout", "experiment_name": run})
    path = checkpoint_root / run / "heldout_sample_summary.json"
    if not path.is_file():
        row["status"] = "not_run"
        return row
    payload = json.loads(path.read_text())
    mode = str(payload["primary_image_mode"])
    summary = payload["image_modes"][mode]
    row.update({
        "status": "complete",
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
    for panel in GENE_PANELS:
        row[f"pcc_{panel}"] = summary[f"pcc_{panel}_mean"]
        row[f"rmse_{panel}"] = summary[f"rmse_{panel}_mean"]
        row[f"n_pcc_genes_{panel}"] = summary[f"n_pcc_genes_{panel}_mean"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="results/checkpoints/recovery_suite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint_root = Path(args.checkpoint_root)

    rows = [_capacity_row(checkpoint_root, run) for run in CAPACITY_RUNS]
    rows += [_heldout_row(checkpoint_root, run) for run in HELDOUT_RUNS]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    n_missing = sum(1 for row in rows if row["status"] != "complete")
    print(output)
    if n_missing:
        print(f"NOTE: {n_missing}/{len(rows)} runs have no completed artifact yet.")
    harmonic = next((r for r in rows if r["experiment_name"] == "transport_harmonic_k128"), None)
    c05 = next((r for r in rows if r["experiment_name"] == "transport_c05_all_modalities_concat_k128_20k"), None)
    if harmonic and c05 and harmonic["status"] == "complete" and c05["status"] == "complete":
        verdict = "BEATS" if float(c05["pcc"]) > float(harmonic["pcc"]) else "does not beat"
        print(f"C05 (primary) full-panel PCC={c05['pcc']} vs harmonic k128 PCC={harmonic['pcc']} -> C05 {verdict} harmonic.")


if __name__ == "__main__":
    main()
