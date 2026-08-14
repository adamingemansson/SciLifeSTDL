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
    "gene_pcc_q25", "gene_pcc_q75", "fraction_gene_pcc_gt_0",
    "fraction_gene_pcc_gt_0_1", "fraction_gene_pcc_gt_0_2",
    "fraction_gene_pcc_gt_0_3", "auc",
    "mean_spot_profile_pcc", "median_spot_profile_pcc",
    "coexpression_matrix_pcc", "coexpression_matrix_mae",
    "mean_per_gene_ssim", "median_per_gene_ssim",
    "moran_i_pcc", "moran_i_mae",
    "local_signed_gradient_pcc", "wide_signed_gradient_pcc",
    "local_gradient_energy_ratio", "wide_gradient_energy_ratio",
    "local_gradient_sign_agreement", "wide_gradient_sign_agreement",
    "n_patients", "runtime_seconds", "peak_gpu_memory_mib", "parameter_count",
    "mean_image_coverage_fraction", "source",
)

POINT_FIELDS = (
    "pcc", "spearman", "rmse", "mse", "mae", "r2", "median_gene_r2",
    "median_gene_pcc", "gene_pcc_q25", "gene_pcc_q75",
    "fraction_gene_pcc_gt_0", "fraction_gene_pcc_gt_0_1",
    "fraction_gene_pcc_gt_0_2", "fraction_gene_pcc_gt_0_3", "auc",
)

STRUCTURED_FIELDS = {
    "mean_spot_profile_pcc": "spot_profile.mean_spot_profile_pcc",
    "median_spot_profile_pcc": "spot_profile.median_spot_profile_pcc",
    "coexpression_matrix_pcc": "coexpression.correlation_matrix_pcc",
    "coexpression_matrix_mae": "coexpression.correlation_matrix_mae",
    "mean_per_gene_ssim": "spatial_ssim.mean_per_gene_ssim",
    "median_per_gene_ssim": "spatial_ssim.median_per_gene_ssim",
    "moran_i_pcc": "moran_local.moran_i_pcc",
    "moran_i_mae": "moran_local.moran_i_mae",
    "local_signed_gradient_pcc": "gradient_local.signed_gradient_pcc",
    "wide_signed_gradient_pcc": "gradient_wide.signed_gradient_pcc",
    "local_gradient_energy_ratio": "gradient_local.gradient_energy_ratio",
    "wide_gradient_energy_ratio": "gradient_wide.gradient_energy_ratio",
    "local_gradient_sign_agreement": "gradient_local.sign_agreement_nontrivial",
    "wide_gradient_sign_agreement": "gradient_wide.sign_agreement_nontrivial",
}


def _metric_value(metrics: dict, name: str):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get("patient_mean")
    return value


def _row(
    *, track, scope, method, prediction, panel, metrics, source,
    structured_metrics: dict | None = None, metadata: dict | None = None,
):
    metadata = metadata or {}
    coverage = metadata.get("mean_image_coverage_fraction")
    if coverage is None:
        slide_records = metadata.get("per_slide_records") or []
        values = [
            row.get("image_coverage_fraction") for row in slide_records
            if row.get("image_coverage_fraction") is not None
        ]
        coverage = sum(float(value) for value in values) / len(values) if values else None
    return {
        "track": track,
        "scope": scope,
        "method": method,
        "prediction": prediction,
        "panel": panel,
        **{name: _metric_value(metrics, name) for name in POINT_FIELDS},
        **{
            output_name: _metric_value(structured_metrics or {}, metric_name)
            for output_name, metric_name in STRUCTURED_FIELDS.items()
        },
        "n_patients": (
            metrics.get("pcc", {}).get("n_patients")
            if isinstance(metrics.get("pcc"), dict) else None
        ),
        "runtime_seconds": metadata.get(
            "runtime_seconds", metadata.get("reevaluation_runtime_seconds"),
        ),
        "peak_gpu_memory_mib": metadata.get("peak_gpu_memory_mib"),
        "parameter_count": metadata.get("parameter_count"),
        "mean_image_coverage_fraction": coverage,
        "source": str(source),
    }


def _method_name(report: dict, path: Path) -> str:
    config = Path(str(report.get("config_path", ""))).stem
    return config or path.stem.replace("_validation", "")


def rows_from_report(path: Path, report: dict) -> list[dict]:
    kind = report.get("kind")
    rows = []
    if kind in {
        "frozen_feature_pca_ridge_whole_slide_benchmark",
        "frozen_feature_pca_mlp_whole_slide_benchmark",
    }:
        decoder = "ridge" if "ridge" in kind else "mlp"
        method = f"{report['image_encoder']}_pca{report['pca_components']}_{decoder}"
        structured = report.get("structured_metrics_patient_aggregated") or {}
        for panel, metrics in report["point_metrics_patient_aggregated"].items():
            rows.append(_row(
                track="expanded_exact_split", scope="whole_slide", method=method,
                prediction="deterministic", panel=panel, metrics=metrics, source=path,
                structured_metrics=structured.get(panel), metadata=report,
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
                metadata=report,
            ))
        for panel, arms in report.get("per_panel_patient_aggregated_metrics", {}).items():
            metrics = arms.get(prediction)
            if metrics:
                rows.append(_row(
                    track="expanded_exact_split", scope="fixed_mask", method=method,
                    prediction=prediction, panel=panel, metrics=metrics, source=path,
                    metadata=report,
                ))
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        structured = whole.get("structured_metrics_patient_aggregated") or {}
        for panel, metrics in whole.get("point_metrics_patient_aggregated", {}).items():
            rows.append(_row(
                track="expanded_exact_split", scope="whole_slide", method=method,
                prediction="conditional_mean", panel=panel, metrics=metrics, source=path,
                structured_metrics=structured.get(panel),
                metadata={**report, "per_slide_records": whole.get("per_slide_records", [])},
            ))
        return rows

    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        method = f"stpath_{report.get('task', 'unknown')}"
        all_metrics = report.get("normalized_log1p_patient_aggregated_metrics")
        if all_metrics:
            rows.append(_row(
                track="expanded_exact_split", scope="fixed_mask", method=method,
                prediction="pretrained_zero_shot", panel="all_genes",
                metrics=all_metrics, source=path, metadata=report,
            ))
        for panel, metrics in report.get("per_panel_patient_aggregated_metrics", {}).items():
            rows.append(_row(
                track="expanded_exact_split", scope="fixed_mask", method=method,
                prediction="pretrained_zero_shot", panel=panel, metrics=metrics, source=path,
                metadata=report,
            ))
        whole = report.get("whole_slide_point_evaluation") or {}
        for panel, metrics in whole.get("point_metrics_patient_aggregated", {}).items():
            rows.append(_row(
                track="expanded_exact_split", scope="whole_slide", method=method,
                prediction="pretrained_zero_shot", panel=panel,
                metrics=metrics, source=path,
                metadata={**report, "per_slide_records": whole.get("per_slide_records", [])},
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
