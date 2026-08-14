#!/usr/bin/env python3
"""Compare MK whole-slide reports with an exact frozen-feature ridge control.

The ridge report defines the held-out cohort and evaluated spot count for each
slide. Conditional-WAE reports enter the table only when they evaluate exactly
that cohort, exactly those spots, use the deterministic H&E point prediction,
and explicitly declare that query GEX is hidden. This prevents older 9-slide
and fixed-mask reports from being mixed into the expanded-cohort comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


PANELS = (
    "all_genes",
    "train_log1p_variance_top50",
    "train_log1p_variance_top200",
    "train_within_slide_variance_top50",
    "train_within_slide_variance_top200",
)

FIELDS = (
    "rank", "method", "suite", "panel", "pcc", "pcc_ci95_low",
    "pcc_ci95_high", "pcc_delta_vs_ridge", "pcc_delta_ci95_low",
    "pcc_delta_ci95_high", "rmse", "rmse_ci95_low",
    "rmse_ci95_high", "rmse_improvement_vs_ridge",
    "rmse_improvement_ci95_low", "rmse_improvement_ci95_high",
    "auc", "n_patients",
    "n_slides", "checkpoint_step", "source",
)

STRUCTURED_METRICS = {
    "spot_profile_pcc": "spot_profile.mean_spot_profile_pcc",
    "coexpression_pcc": "coexpression.correlation_matrix_pcc",
    "spatial_ssim": "spatial_ssim.mean_per_gene_ssim",
    "moran_i_pcc": "moran_local.moran_i_pcc",
    "local_gradient_pcc": "gradient_local.signed_gradient_pcc",
    "wide_gradient_pcc": "gradient_wide.signed_gradient_pcc",
}
STRUCTURED_FIELDS = (
    "method", "suite", "panel",
    *tuple(STRUCTURED_METRICS),
    *tuple(f"{name}_delta_vs_ridge" for name in STRUCTURED_METRICS),
    *tuple(f"{name}_delta_ci95_low" for name in STRUCTURED_METRICS),
    *tuple(f"{name}_delta_ci95_high" for name in STRUCTURED_METRICS),
    "n_patients", "n_slides", "checkpoint_step", "source",
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return payload


def _metric(metrics: dict, name: str, field: str = "patient_mean"):
    value = metrics.get(name)
    if isinstance(value, dict):
        return value.get(field)
    return value if field == "patient_mean" else None


def _paired_patient_delta(
    reference_records: list[dict], candidate_records: list[dict], *,
    panel: str, metric: str, section: str = "point_metrics",
    higher_is_better: bool = True, n_bootstrap: int = 10_000, seed: int = 0,
) -> dict[str, float | int]:
    """Matched patient-macro delta and percentile bootstrap confidence interval."""
    reference = {str(row["sample_id"]): row for row in reference_records}
    candidate = {str(row["sample_id"]): row for row in candidate_records}
    if set(reference) != set(candidate):
        raise ValueError("paired delta requires identical held-out slide IDs")
    by_patient: dict[str, list[float]] = {}
    for sample_id in sorted(reference):
        left = reference[sample_id]
        right = candidate[sample_id]
        if str(left["patient_id"]) != str(right["patient_id"]):
            raise ValueError(f"patient mismatch for held-out slide {sample_id}")
        if section == "structured_field":
            left_metrics = _flatten_structured_panel(left[section]["panels"][panel])
            right_metrics = _flatten_structured_panel(right[section]["panels"][panel])
        else:
            left_metrics = left[section][panel]
            right_metrics = right[section][panel]
        left_value = left_metrics.get(metric)
        right_value = right_metrics.get(metric)
        if left_value is None or right_value is None:
            continue
        left_value, right_value = float(left_value), float(right_value)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            continue
        delta = right_value - left_value if higher_is_better else left_value - right_value
        by_patient.setdefault(str(left["patient_id"]), []).append(delta)
    patient_deltas = np.asarray(
        [np.mean(values) for values in by_patient.values()], dtype=np.float64,
    )
    if patient_deltas.size == 0:
        return {
            "patient_mean": float("nan"), "ci95_low": float("nan"),
            "ci95_high": float("nan"), "n_patients": 0,
        }
    mean = float(patient_deltas.mean())
    if patient_deltas.size < 2:
        low = high = float("nan")
    else:
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(
            patient_deltas, size=(int(n_bootstrap), len(patient_deltas)), replace=True,
        ).mean(axis=1)
        low, high = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    return {
        "patient_mean": mean, "ci95_low": low, "ci95_high": high,
        "n_patients": int(patient_deltas.size),
    }


def _flatten_structured_panel(panel: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section, values in panel.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, (int, float)):
                flat[f"{section}.{name}"] = float(value)
    return flat


def _paired_deltas_by_panel(
    reference_records: list[dict], candidate_records: list[dict], *,
    structured: bool = False,
) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    panels = (
        reference_records[0]["structured_field"]["panels"]
        if structured else reference_records[0]["point_metrics"]
    )
    metric_names = tuple(STRUCTURED_METRICS.values()) if structured else ("pcc", "rmse")
    for panel in panels:
        result[panel] = {}
        for offset, metric in enumerate(metric_names):
            result[panel][metric] = _paired_patient_delta(
                reference_records, candidate_records,
                panel=panel, metric=metric,
                section="structured_field" if structured else "point_metrics",
                higher_is_better=(metric != "rmse"),
                seed=17_000 + 101 * offset + sum(ord(char) for char in panel),
            )
    return result


def _method(report: dict, path: Path) -> str:
    config = Path(str(report.get("config_path") or ""))
    return config.stem or path.stem.replace("_validation", "")


def _manifest_contract(manifest_path: Path) -> dict:
    manifest = _load(manifest_path)
    gene_names = [str(value) for value in manifest.get("gene_panel") or []]
    validation_ids = [str(value) for value in manifest.get("validation_sample_ids") or []]
    if not gene_names or not validation_ids:
        raise ValueError(f"manifest has no gene panel or validation split: {manifest_path}")
    build_args = manifest.get("build_args") or {}
    return {
        "gene_panel_sha256": hashlib.sha256(
            json.dumps(gene_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "validation_sample_ids": validation_ids,
        "expression_transform": build_args.get("expression_transform"),
        "expression_target_sum": build_args.get("expression_target_sum"),
    }


def _manifest_contract_for_mk(report: dict) -> dict:
    config_path = Path(str(report.get("config_path") or "")).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"reported config is unavailable: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"reported config is not a mapping: {config_path}")
    manifest_path = Path(str((config.get("data") or {}).get("gen3_manifest_path") or ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"configured manifest is unavailable: {manifest_path}")
    return _manifest_contract(manifest_path)


def _suite(report: dict, path: Path) -> str:
    config = Path(str(report.get("config_path") or ""))
    parts = list(config.parts)
    for marker in ("configs", "eval_configs"):
        if marker in parts:
            index = len(parts) - 1 - parts[::-1].index(marker)
            if index:
                return parts[index - 1]
    return path.parent.name


def _point_rows(
    *, method: str, suite: str, source: Path, metrics_by_panel: dict,
    reference: dict[str, dict], n_slides: int, checkpoint_step=None,
    paired_deltas: dict[str, dict[str, dict]] | None = None,
) -> list[dict]:
    rows = []
    for panel in PANELS:
        metrics = metrics_by_panel.get(panel)
        baseline = reference.get(panel)
        if not metrics or not baseline:
            continue
        pcc = _metric(metrics, "pcc")
        rmse = _metric(metrics, "rmse")
        baseline_pcc = _metric(baseline, "pcc")
        baseline_rmse = _metric(baseline, "rmse")
        panel_deltas = (paired_deltas or {}).get(panel, {})
        pcc_delta = panel_deltas.get("pcc")
        rmse_delta = panel_deltas.get("rmse")
        rows.append({
            "rank": None,
            "method": method,
            "suite": suite,
            "panel": panel,
            "pcc": pcc,
            "pcc_ci95_low": _metric(metrics, "pcc", "patient_ci95_low"),
            "pcc_ci95_high": _metric(metrics, "pcc", "patient_ci95_high"),
            "pcc_delta_vs_ridge": (
                pcc_delta["patient_mean"] if pcc_delta is not None
                else float(pcc) - float(baseline_pcc)
                if pcc is not None and baseline_pcc is not None else None
            ),
            "pcc_delta_ci95_low": pcc_delta.get("ci95_low") if pcc_delta else None,
            "pcc_delta_ci95_high": pcc_delta.get("ci95_high") if pcc_delta else None,
            "rmse": rmse,
            "rmse_ci95_low": _metric(metrics, "rmse", "patient_ci95_low"),
            "rmse_ci95_high": _metric(metrics, "rmse", "patient_ci95_high"),
            "rmse_improvement_vs_ridge": (
                rmse_delta["patient_mean"] if rmse_delta is not None
                else float(baseline_rmse) - float(rmse)
                if rmse is not None and baseline_rmse is not None else None
            ),
            "rmse_improvement_ci95_low": rmse_delta.get("ci95_low") if rmse_delta else None,
            "rmse_improvement_ci95_high": rmse_delta.get("ci95_high") if rmse_delta else None,
            "auc": _metric(metrics, "auc"),
            "n_patients": (
                metrics.get("pcc", {}).get("n_patients")
                if isinstance(metrics.get("pcc"), dict) else None
            ),
            "n_slides": n_slides,
            "checkpoint_step": checkpoint_step,
            "source": str(source),
        })
    return rows


def _structured_rows(
    *, method: str, suite: str, source: Path, metrics_by_panel: dict,
    reference: dict[str, dict], n_slides: int, checkpoint_step=None,
    paired_deltas: dict[str, dict[str, dict]] | None = None,
) -> list[dict]:
    rows = []
    for panel in PANELS:
        metrics = metrics_by_panel.get(panel)
        baseline = reference.get(panel)
        if not metrics or not baseline:
            continue
        row = {
            "method": method,
            "suite": suite,
            "panel": panel,
            "n_patients": None,
            "n_slides": n_slides,
            "checkpoint_step": checkpoint_step,
            "source": str(source),
        }
        for output_name, metric_name in STRUCTURED_METRICS.items():
            value = _metric(metrics, metric_name)
            baseline_value = _metric(baseline, metric_name)
            row[output_name] = value
            delta = (paired_deltas or {}).get(panel, {}).get(metric_name)
            row[f"{output_name}_delta_vs_ridge"] = (
                delta["patient_mean"] if delta is not None
                else float(value) - float(baseline_value)
                if value is not None and baseline_value is not None else None
            )
            row[f"{output_name}_delta_ci95_low"] = delta.get("ci95_low") if delta else None
            row[f"{output_name}_delta_ci95_high"] = delta.get("ci95_high") if delta else None
            metric_payload = metrics.get(metric_name)
            if row["n_patients"] is None and isinstance(metric_payload, dict):
                row["n_patients"] = metric_payload.get("n_patients")
        rows.append(row)
    return rows


def _contract_from_ridge(
    report: dict,
) -> tuple[dict[str, int], dict[str, dict], dict | None]:
    if report.get("kind") != "frozen_feature_pca_ridge_whole_slide_benchmark":
        raise ValueError("--ridge-report is not a frozen-feature ridge report")
    if report.get("split") != "validation":
        raise ValueError("ridge reference must use the validation split")
    if report.get("missing_image_policy") != "zero":
        raise ValueError("primary MK comparison requires missing_image_policy=zero")
    records = report.get("per_slide_records") or []
    counts = {
        str(row["sample_id"]): int(row["n_evaluated_spots"])
        for row in records
    }
    if len(counts) != int(report.get("n_validation_samples", -1)) or len(counts) != 14:
        raise ValueError(
            f"ridge reference must contain exactly 14 distinct slides, found {len(counts)}"
        )
    metrics = report.get("point_metrics_patient_aggregated") or {}
    if "all_genes" not in metrics:
        raise ValueError("ridge reference has no all_genes point metrics")
    manifest_path = Path(str(report.get("manifest_path") or "")).expanduser()
    dataset_contract = _manifest_contract(manifest_path) if manifest_path.is_file() else None
    return counts, metrics, dataset_contract


def _compatible_mk(
    report: dict, expected_counts: dict[str, int],
    expected_dataset_contract: dict | None = None,
) -> tuple[bool, str]:
    if report.get("kind") != "conditional_wae_supervisor_evaluation":
        return False, "not_conditional_wae"
    if report.get("split") != "validation":
        return False, "wrong_split"
    if report.get("query_gex_visible") is not False:
        return False, "query_gex_not_explicitly_hidden"
    if report.get("query_he_visible") is not True:
        return False, "query_he_not_visible"
    whole = report.get("whole_slide_structured_field_evaluation") or {}
    if whole.get("scope") != "all_held_out_slides_every_spot_exactly_once":
        return False, "not_exact_whole_slide"
    if whole.get("target_gex_visible_to_model") is not False:
        return False, "whole_slide_target_gex_not_explicitly_hidden"
    if whole.get("primary_prediction") != "deterministic_h_and_e_point_prediction":
        return False, "wrong_primary_prediction"
    records = whole.get("per_slide_records") or []
    actual = {str(row["sample_id"]): int(row["n_spots"]) for row in records}
    if actual != expected_counts:
        return False, "cohort_or_spot_count_mismatch"
    if not whole.get("point_metrics_patient_aggregated"):
        return False, "missing_point_metrics"
    if expected_dataset_contract is not None:
        try:
            actual_contract = _manifest_contract_for_mk(report)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            return False, "dataset_contract_unverifiable"
        if actual_contract != expected_dataset_contract:
            return False, "dataset_contract_mismatch"
    return True, "compatible"


def _display(rows: list[dict]) -> None:
    print(
        f"{'RANK':>4}  {'METHOD':<39} {'PANEL':<35} "
        f"{'PCC':>9} {'DELTA':>9} {'RMSE':>9} {'RMSE IMP':>10} {'N':>3}"
    )
    print("-" * 128)
    for row in rows:
        def fmt(value, width=9):
            try:
                value = float(value)
                return f"{value:{width}.6f}" if math.isfinite(value) else f"{'NA':>{width}}"
            except (TypeError, ValueError):
                return f"{'NA':>{width}}"
        print(
            f"{str(row['rank'] or '-'):>4}  {row['method']:<39.39} "
            f"{row['panel']:<35.35} {fmt(row['pcc'])} "
            f"{fmt(row['pcc_delta_vs_ridge'])} {fmt(row['rmse'])} "
            f"{fmt(row['rmse_improvement_vs_ridge'], 10)} "
            f"{str(row['n_patients'] or 'NA'):>3}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-report", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ridge_path = Path(args.ridge_report).expanduser().resolve()
    ridge = _load(ridge_path)
    expected_counts, reference, dataset_contract = _contract_from_ridge(ridge)
    if dataset_contract is None:
        raise ValueError("ridge report's dataset manifest is unavailable; cannot compare safely")

    candidates: dict[str, tuple[float, Path, dict]] = {}
    skipped: dict[str, int] = {}
    for raw in args.roots:
        root = Path(raw).expanduser()
        paths = [root] if root.is_file() else root.rglob("*.json") if root.is_dir() else []
        for path in paths:
            try:
                report = _load(path)
            except Exception:
                continue
            compatible, reason = _compatible_mk(
                report, expected_counts, dataset_contract,
            )
            if not compatible:
                if report.get("kind") == "conditional_wae_supervisor_evaluation":
                    skipped[reason] = skipped.get(reason, 0) + 1
                continue
            # One current row per scientific configuration. A repeated or
            # continued evaluation supersedes its older report by mtime.
            key = str(report.get("config_path"))
            current = candidates.get(key)
            item = (path.stat().st_mtime, path.resolve(), report)
            if current is None or item[0] > current[0]:
                candidates[key] = item

    rows = _point_rows(
        method=f"{ridge['image_encoder']}_pca{ridge['pca_components']}_ridge",
        suite=ridge_path.parent.name,
        source=ridge_path,
        metrics_by_panel=reference,
        reference=reference,
        n_slides=len(expected_counts),
    )
    structured_reference = ridge.get("structured_metrics_patient_aggregated") or {}
    structured_rows = _structured_rows(
        method=f"{ridge['image_encoder']}_pca{ridge['pca_components']}_ridge",
        suite=ridge_path.parent.name,
        source=ridge_path,
        metrics_by_panel=structured_reference,
        reference=structured_reference,
        n_slides=len(expected_counts),
    )
    for _, path, report in candidates.values():
        whole = report["whole_slide_structured_field_evaluation"]
        point_deltas = _paired_deltas_by_panel(
            ridge["per_slide_records"], whole["per_slide_records"],
        )
        rows.extend(_point_rows(
            method=_method(report, path),
            suite=_suite(report, path),
            source=path,
            metrics_by_panel=whole["point_metrics_patient_aggregated"],
            reference=reference,
            n_slides=len(expected_counts),
            checkpoint_step=report.get("checkpoint_step"),
            paired_deltas=point_deltas,
        ))
        structured_deltas = _paired_deltas_by_panel(
            ridge["per_slide_records"], whole["per_slide_records"], structured=True,
        )
        structured_rows.extend(_structured_rows(
            method=_method(report, path),
            suite=_suite(report, path),
            source=path,
            metrics_by_panel=whole.get("structured_metrics_patient_aggregated") or {},
            reference=structured_reference,
            n_slides=len(expected_counts),
            checkpoint_step=report.get("checkpoint_step"),
            paired_deltas=structured_deltas,
        ))

    for panel in PANELS:
        panel_rows = [row for row in rows if row["panel"] == panel]
        panel_rows.sort(key=lambda row: float(row["pcc"]), reverse=True)
        for rank, row in enumerate(panel_rows, 1):
            row["rank"] = rank
    rows.sort(key=lambda row: (PANELS.index(row["panel"]), row["rank"]))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    structured_output = output.with_name(f"{output.stem}_structured{output.suffix}")
    with structured_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURED_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(structured_rows)
    _display(rows)
    print(f"\nCompatible MK reports: {len(candidates)}")
    print(f"Rejected MK reports by reason: {dict(sorted(skipped.items()))}")
    print(f"Comparison table: {output}")
    print(f"Structured comparison table: {structured_output}")


if __name__ == "__main__":
    main()
