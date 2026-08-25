#!/usr/bin/env python3
"""Print a clean side-by-side comparison table across the 4 UNI2
geneencoder arms -- full-panel, HVG-50, HVG-200 PCC/RMSE plus calibration
coverage. Uses "model" (the real sampled predictive distribution, not
"conditional_mean") and "patient_mean" (the primary reportable number
per aggregate_patient_metrics's own docstring, not "pooled_mean").
Real field names confirmed against gen3_multiscale/evaluation/metrics.py
and gen3_multiscale/evaluation/gen3_evaluator.py.
"""
import json
from pathlib import Path

SUITE_ROOT = Path(
    "/data/adam.ingemansson/SciLifeSTDL-MK/gen3_multiscale/results/"
    "wae_mmd_geneencoder_ablation_uni2_v1"
)
ARMS = [
    "wae_he_mmd_geneencoder_scfoundation_film",
    "wae_he_mmd_geneencoder_scfoundation_nofilm",
    "wae_he_mmd_geneencoder_mlp_film",
    "wae_he_mmd_geneencoder_mlp_nofilm",
]
SHORT = {
    "wae_he_mmd_geneencoder_scfoundation_film": "scF+FiLM",
    "wae_he_mmd_geneencoder_scfoundation_nofilm": "scF+noFiLM",
    "wae_he_mmd_geneencoder_mlp_film": "MLP+FiLM",
    "wae_he_mmd_geneencoder_mlp_nofilm": "MLP+noFiLM",
}
PANELS = ["train_log1p_variance_top50", "train_log1p_variance_top200"]


def fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    reports = {}
    for arm in ARMS:
        path = SUITE_ROOT / "logs" / f"{arm}_eval.json"
        if not path.is_file():
            print(f"{arm}: MISSING ({path})")
            continue
        reports[arm] = json.loads(path.read_text())

    if not reports:
        return

    rows = []
    for arm, report in reports.items():
        full = report["per_arm_patient_aggregated_metrics"]["model"]
        panel_metrics = report["per_panel_patient_aggregated_metrics"]
        cal = report.get("conditional_wae_calibration", {})
        rows.append({
            "arm": SHORT.get(arm, arm),
            "checkpoint_step": report.get("checkpoint_step"),
            "n_items": report.get("n_items"),
            "full_pcc": full["pcc"]["patient_mean"],
            "full_rmse": full["rmse"]["patient_mean"],
            "hvg50_pcc": panel_metrics.get(PANELS[0], {}).get("model", {}).get("pcc", {}).get("patient_mean"),
            "hvg50_rmse": panel_metrics.get(PANELS[0], {}).get("model", {}).get("rmse", {}).get("patient_mean"),
            "hvg200_pcc": panel_metrics.get(PANELS[1], {}).get("model", {}).get("pcc", {}).get("patient_mean"),
            "hvg200_rmse": panel_metrics.get(PANELS[1], {}).get("model", {}).get("rmse", {}).get("patient_mean"),
            "z_std": cal.get("z_std"),
            "coverage_68": cal.get("coverage_68"),
            "coverage_90": cal.get("coverage_90"),
            "coverage_95": cal.get("coverage_95"),
        })

    col_width = max(len(r["arm"]) for r in rows) + 2
    metric_labels = [
        ("checkpoint_step", "checkpoint step"),
        ("n_items", "n eval items"),
        ("full_pcc", "full-panel PCC"),
        ("full_rmse", "full-panel RMSE"),
        ("hvg50_pcc", "HVG-50 PCC"),
        ("hvg50_rmse", "HVG-50 RMSE"),
        ("hvg200_pcc", "HVG-200 PCC"),
        ("hvg200_rmse", "HVG-200 RMSE"),
        ("z_std", "calibration z_std (ideal=1.0)"),
        ("coverage_68", "coverage @ 68%"),
        ("coverage_90", "coverage @ 90%"),
        ("coverage_95", "coverage @ 95%"),
    ]

    header_label_width = max(len(label) for _, label in metric_labels) + 2
    header = " " * header_label_width + "".join(r["arm"].ljust(col_width) for r in rows)
    print(header)
    print("-" * len(header))
    for key, label in metric_labels:
        line = label.ljust(header_label_width)
        for r in rows:
            line += fmt(r[key]).ljust(col_width)
        print(line)


if __name__ == "__main__":
    main()
