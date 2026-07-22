#!/usr/bin/env python3
"""Summarize the hierarchical_gene_transport_regressor 20-run suite, v2
(transport_reg_weight=0.0 fix -- see run_transport_suite_v2_4gpu.sh's own
header). Same two-phase CSV layout as summarize_transport_suite.py, plus a
direct v1-vs-v2 comparison printed at the end for every held-out run that
has completed artifacts in both checkpoint_roots, since that comparison is
the entire point of this rerun.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CAPACITY_RUNS = (
    "transport_capacity_o01_full_concat_no_residual_v2",
    "transport_capacity_o02_full_concat_residual64_v2",
    "transport_capacity_o03_gex_novae_no_he_v2",
    "transport_capacity_o04_gated_experts_residual64_v2",
)

HELDOUT_RUNS = (
    "transport_c01_raw_gex_only_v2_20k",
    "transport_c02_gex_novae_v2_20k",
    "transport_c03_gex_local_he_v2_20k",
    "transport_c04_gex_global_he_v2_20k",
    "transport_c05_all_modalities_concat_k128_v2_20k",
    "transport_c06_all_modalities_gated_experts_v2_20k",
    "transport_c07_geometry_only_scoring_v2_20k",
    "transport_c08_shared_gene_gate_v2_20k",
    "transport_c09_per_gene_gate_no_query_v2_20k",
    "transport_c10_one_transport_head_v2_20k",
    "transport_c11_four_transport_heads_v2_20k",
    "transport_c12_sixteen_transport_heads_v2_20k",
    "transport_c13_k32_v2_20k",
    "transport_c14_k64_v2_20k",
    "transport_c15_residual_rank32_v2_20k",
    "transport_c16_residual_rank64_v2_20k",
    "transport_harmonic_k128_v2",
)

# v2 experiment_name -> matching v1 experiment_name, for the comparison table.
V1_MATCH = {
    "transport_capacity_o01_full_concat_no_residual_v2": "transport_capacity_o01_full_concat_no_residual",
    "transport_capacity_o02_full_concat_residual64_v2": "transport_capacity_o02_full_concat_residual64",
    "transport_capacity_o03_gex_novae_no_he_v2": "transport_capacity_o03_gex_novae_no_he",
    "transport_capacity_o04_gated_experts_residual64_v2": "transport_capacity_o04_gated_experts_residual64",
    "transport_c01_raw_gex_only_v2_20k": "transport_c01_raw_gex_only_20k",
    "transport_c02_gex_novae_v2_20k": "transport_c02_gex_novae_20k",
    "transport_c03_gex_local_he_v2_20k": "transport_c03_gex_local_he_20k",
    "transport_c04_gex_global_he_v2_20k": "transport_c04_gex_global_he_20k",
    "transport_c05_all_modalities_concat_k128_v2_20k": "transport_c05_all_modalities_concat_k128_20k",
    "transport_c06_all_modalities_gated_experts_v2_20k": "transport_c06_all_modalities_gated_experts_20k",
    "transport_c07_geometry_only_scoring_v2_20k": "transport_c07_geometry_only_scoring_20k",
    "transport_c08_shared_gene_gate_v2_20k": "transport_c08_shared_gene_gate_20k",
    "transport_c09_per_gene_gate_no_query_v2_20k": "transport_c09_per_gene_gate_no_query_20k",
    "transport_c10_one_transport_head_v2_20k": "transport_c10_one_transport_head_20k",
    "transport_c11_four_transport_heads_v2_20k": "transport_c11_four_transport_heads_20k",
    "transport_c12_sixteen_transport_heads_v2_20k": "transport_c12_sixteen_transport_heads_20k",
    "transport_c13_k32_v2_20k": "transport_c13_k32_20k",
    "transport_c14_k64_v2_20k": "transport_c14_k64_20k",
    "transport_c15_residual_rank32_v2_20k": "transport_c15_residual_rank32_20k",
    "transport_c16_residual_rank64_v2_20k": "transport_c16_residual_rank64_20k",
    "transport_harmonic_k128_v2": "transport_harmonic_k128",
}

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

    harmonic = next((r for r in rows if r["experiment_name"] == "transport_harmonic_k128_v2"), None)
    c05 = next((r for r in rows if r["experiment_name"] == "transport_c05_all_modalities_concat_k128_v2_20k"), None)
    if harmonic and c05 and harmonic["status"] == "complete" and c05["status"] == "complete":
        verdict = "BEATS" if float(c05["pcc"]) > float(harmonic["pcc"]) else "does not beat"
        print(f"C05 v2 (primary) full-panel PCC={c05['pcc']} vs harmonic k128 PCC={harmonic['pcc']} -> C05 v2 {verdict} harmonic.")

    print("\n=== v1 (transport_reg_weight=1e-3) vs v2 (transport_reg_weight=0.0) ===")
    heldout_v2 = {r["experiment_name"]: r for r in rows if r["phase"] == "heldout"}
    for v2_name, v1_name in V1_MATCH.items():
        if v2_name not in heldout_v2 or heldout_v2[v2_name]["status"] != "complete":
            continue
        v1_path = checkpoint_root / v1_name / "heldout_sample_summary.json"
        if not v1_path.is_file():
            continue
        v1_payload = json.loads(v1_path.read_text())
        v1_mode = str(v1_payload["primary_image_mode"])
        v1_pcc = v1_payload["image_modes"][v1_mode]["pcc_mean"]
        v2_pcc = float(heldout_v2[v2_name]["pcc"])
        delta = v2_pcc - v1_pcc
        arrow = "UP" if delta > 0 else ("DOWN" if delta < 0 else "=")
        print(f"{v2_name}: v1 PCC={v1_pcc:.4f} -> v2 PCC={v2_pcc:.4f} ({arrow} {delta:+.4f})")


if __name__ == "__main__":
    main()
