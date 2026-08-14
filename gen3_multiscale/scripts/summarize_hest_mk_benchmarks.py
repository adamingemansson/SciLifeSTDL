#!/usr/bin/env python3
"""Extract comparable rows from MK, STPath and frozen-ridge JSON reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "track", "scope", "method", "prediction", "panel", "pcc", "spearman",
    "rmse", "mse", "mae", "r2", "median_gene_r2", "median_gene_pcc",
    "fraction_gene_pcc_gt_0_1", "fraction_gene_pcc_gt_0_2", "auc",
    "n_patients", "source",
)


def _metric_value(metrics: dict, name: str):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("patient_mean")
    return value


def _row(*, track, scope, method, prediction, panel, metrics, source):
    return {
        "track": track,
        "scope": scope,
        "method": method,
        "prediction": prediction,
        "panel": panel,
        **{name: _metric_value(metrics, name) for name in FIELDS[5:-2]},
        "n_patients": (
            metrics.get("pcc", {}).get("n_patients")
            if isinstance(metrics.get("pcc"), dict) else None
        ),
        "source": str(source),
    }


def _method_name(report: dict, path: Path) -> str:
    config = Path(str(report.get("config_path", ""))).stem
    return config or path.stem.replace("_validation", "")


def rows_from_report(path: Path, report: dict) -> list[dict]:
    kind = report.get("kind")
    rows = []
    if kind == "frozen_feature_pca_ridge_whole_slide_benchmark":
        method = f"{report['image_encoder']}_pca{report['pca_components']}_ridge"
        for panel, metrics in report["point_metrics_patient_aggregated"].items():
            rows.append(_row(
                track="expanded_exact_split", scope="whole_slide", method=method,
                prediction="deterministic", panel=panel, metrics=metrics, source=path,
            ))
        return rows

    if kind == "conditional_wae_supervisor_evaluation":
        method = _method_name(report, path)
        prediction = report.get("prediction_roles", {}).get(
            "primary_point_prediction", "conditional_mean",
        )
        fixed_all = report.get("per_arm_patient_aggregated_metrics", {}).get(prediction)
        if fixed_all:
            rows.append(_row(
                track="expanded_exact_split", scope="fixed_mask", method=method,
                prediction=prediction, panel="all_genes", metrics=fixed_all, source=path,
            ))
        for panel, arms in report.get("per_panel_patient_aggregated_metrics", {}).items():
            metrics = arms.get(prediction)
            if metrics:
                rows.append(_row(
                    track="expanded_exact_split", scope="fixed_mask", method=method,
                    prediction=prediction, panel=panel, metrics=metrics, source=path,
                ))
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        for panel, metrics in whole.get("point_metrics_patient_aggregated", {}).items():
            rows.append(_row(
                track="expanded_exact_split", scope="whole_slide", method=method,
                prediction="conditional_mean", panel=panel, metrics=metrics, source=path,
            ))
        return rows

    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        method = f"stpath_{report.get('task', 'unknown')}"
        all_metrics = report.get("normalized_log1p_patient_aggregated_metrics")
        if all_metrics:
            rows.append(_row(
                track="expanded_exact_split", scope="fixed_mask", method=method,
                prediction="pretrained_zero_shot", panel="all_genes",
                metrics=all_metrics, source=path,
            ))
        for panel, metrics in report.get("per_panel_patient_aggregated_metrics", {}).items():
            rows.append(_row(
                track="expanded_exact_split", scope="fixed_mask", method=method,
                prediction="pretrained_zero_shot", panel=panel, metrics=metrics, source=path,
            ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="Report JSON files or directories to scan")
    parser.add_argument("--output", help="Optional CSV output path")
    args = parser.parse_args()
    paths = set()
    for raw in args.roots:
        path = Path(raw).expanduser()
        if path.is_file():
            paths.add(path.resolve())
        elif path.is_dir():
            paths.update(candidate.resolve() for candidate in path.rglob("*.json"))
    rows = []
    for path in sorted(paths):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(report, dict):
            rows.extend(rows_from_report(path, report))
    rows.sort(key=lambda row: (
        row["track"], row["scope"], row["panel"], row["method"], row["prediction"],
    ))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {output}")
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
