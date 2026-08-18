#!/usr/bin/env python3
"""Fail-closed summary of replicated MK finalist evaluations.

The report deliberately keeps two uncertainty sources separate:

* ``seed_summary.tsv`` describes variation across independently trained seeds.
* ``architecture_delta_hierarchical_bootstrap.tsv`` resamples both seeds and
  held-out patients for the direct parallel-gated minus sandwich comparison.

All inputs must use the exact expanded validation contract (448 fixed masks,
14 whole slides, deterministic H&E-only prediction and all five panels).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import t as student_t


ARCHITECTURES = ("mk_wb_parallel_gated", "mk_wbw_sandwich")
PANELS = (
    "all_genes",
    "train_log1p_variance_top50",
    "train_log1p_variance_top200",
    "train_within_slide_variance_top50",
    "train_within_slide_variance_top200",
)
POINT_METRICS = ("pcc", "rmse", "auc")
STRUCTURED_METRICS = {
    "spot_profile_pcc": "spot_profile.mean_spot_profile_pcc",
    "coexpression_pcc": "coexpression.correlation_matrix_pcc",
    "spatial_ssim": "spatial_ssim.mean_per_gene_ssim",
    "moran_i_pcc": "moran_local.moran_i_pcc",
    "local_gradient_pcc": "gradient_local.signed_gradient_pcc",
    "wide_gradient_pcc": "gradient_wide.signed_gradient_pcc",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"report is not a JSON object: {path}")
    return payload


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _patient_mean(metric: Any) -> float:
    return _finite(metric.get("patient_mean")) if isinstance(metric, dict) else _finite(metric)


def _primary_role(report: dict[str, Any]) -> str:
    role = (report.get("prediction_roles") or {}).get("primary_point_prediction")
    if role not in {"model", "conditional_mean"}:
        raise ValueError(f"report has no recognized primary prediction role: {role!r}")
    return str(role)


def _config_seed(report: dict[str, Any], path: Path) -> int:
    config_path = Path(str(report.get("config_path") or ""))
    if not config_path.is_file():
        raise FileNotFoundError(f"reported config is unavailable: {config_path} ({path})")
    config = yaml.safe_load(config_path.read_text())
    return int((config.get("training") or {}).get("seed", 0))


def _fixed_panel_metrics(report: dict[str, Any], panel: str) -> dict[str, Any]:
    role = _primary_role(report)
    if panel == "all_genes":
        return (report.get("per_arm_patient_aggregated_metrics") or {}).get(role) or {}
    return (
        ((report.get("per_panel_patient_aggregated_metrics") or {}).get(panel) or {}).get(role)
        or {}
    )


def _whole_panel_metrics(report: dict[str, Any], panel: str) -> dict[str, Any]:
    whole = report.get("whole_slide_structured_field_evaluation") or {}
    return (whole.get("point_metrics_patient_aggregated") or {}).get(panel) or {}


def _flatten_structured(panel: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for section, values in panel.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            number = _finite(value)
            if math.isfinite(number):
                result[f"{section}.{name}"] = number
    return result


def audit_report(
    report: dict[str, Any], path: Path, *, expected_seed: int,
    expected_items: int = 448, expected_slides: int = 14,
) -> dict[str, int]:
    """Validate the complete scientific/evaluation contract for one seed."""
    failures = []
    if report.get("kind") != "conditional_wae_supervisor_evaluation":
        failures.append("wrong report kind")
    if report.get("split") != "validation":
        failures.append("split is not validation")
    if int(report.get("n_items", -1)) != expected_items:
        failures.append(f"fixed items != {expected_items}")
    if int(report.get("n_samples", -1)) != expected_slides:
        failures.append(f"fixed sample count != {expected_slides}")
    if int(report.get("n_mask_strata", -1)) != 4:
        failures.append("mask strata != 4")
    if int(report.get("n_masks_per_stratum_per_sample", -1)) != 8:
        failures.append("masks per stratum per sample != 8")
    fixed_records = report.get("per_item_records") or []
    if len(fixed_records) != expected_items:
        failures.append(f"fixed per-item record count != {expected_items}")
    if report.get("query_gex_visible") is not False:
        failures.append("query GEX is not explicitly hidden")
    if report.get("query_he_visible") is not True:
        failures.append("query H&E is not explicitly visible")
    if report.get("task") != "he_to_st":
        failures.append("task is not he_to_st")
    seed = _config_seed(report, path)
    if seed != int(expected_seed):
        failures.append(f"training seed {seed} != expected {expected_seed}")

    try:
        _primary_role(report)
    except ValueError as exc:
        failures.append(str(exc))

    fixed_panels = {"all_genes"}
    fixed_panels.update((report.get("per_panel_patient_aggregated_metrics") or {}).keys())
    missing_fixed = sorted(set(PANELS).difference(fixed_panels))
    if missing_fixed:
        failures.append(f"fixed-mask panels missing={missing_fixed}")

    whole = report.get("whole_slide_structured_field_evaluation") or {}
    if whole.get("scope") != "all_held_out_slides_every_spot_exactly_once":
        failures.append("whole-slide scope is incomplete")
    if whole.get("primary_prediction") != "deterministic_h_and_e_point_prediction":
        failures.append("whole-slide primary prediction is not deterministic H&E")
    if whole.get("target_gex_visible_to_model") is not False:
        failures.append("whole-slide target GEX is not explicitly hidden")
    records = whole.get("per_slide_records") or []
    if len(records) != expected_slides:
        failures.append(f"whole-slide record count != {expected_slides}")
    sample_ids = [str(row.get("sample_id")) for row in records]
    if len(set(sample_ids)) != len(sample_ids):
        failures.append("whole-slide sample IDs are duplicated")
    if any(int(row.get("n_spots", 0)) < 1 for row in records):
        failures.append("whole-slide record has no spots")
    whole_panels = set((whole.get("point_metrics_patient_aggregated") or {}).keys())
    missing_whole = sorted(set(PANELS).difference(whole_panels))
    if missing_whole:
        failures.append(f"whole-slide panels missing={missing_whole}")
    structured_panels = set((whole.get("structured_metrics_patient_aggregated") or {}).keys())
    missing_structured = sorted(set(PANELS).difference(structured_panels))
    if missing_structured:
        failures.append(f"structured panels missing={missing_structured}")
    sidecar = whole.get("per_gene_diagnostics_path")
    if not sidecar or not Path(str(sidecar)).is_file():
        failures.append("per-gene diagnostics sidecar is missing")
    if failures:
        raise ValueError(f"{path}: incomplete evaluation: {'; '.join(failures)}")
    return {
        "fixed_items": int(report["n_items"]),
        "whole_slides": len(records),
        "checkpoint_step": int(report.get("checkpoint_step") or 0),
    }


def _record_metric(
    report: dict[str, Any], *, scope: str, panel: str, metric: str,
) -> dict[str, float]:
    """Per-slide/per-mask values averaged inside patient for paired analyses."""
    result: dict[str, list[float]] = {}
    if scope == "fixed_mask":
        primary = _primary_role(report)
        records = report["per_item_records"]
        for row in records:
            values = row[primary] if panel == "all_genes" else row["gene_panels"][primary][panel]
            value = _finite(values.get(metric))
            if math.isfinite(value):
                result.setdefault(str(row["patient_id"]), []).append(value)
    elif scope == "whole_slide":
        records = report["whole_slide_structured_field_evaluation"]["per_slide_records"]
        for row in records:
            value = _finite(row["point_metrics"][panel].get(metric))
            if math.isfinite(value):
                result.setdefault(str(row["patient_id"]), []).append(value)
    else:
        raise ValueError(f"unknown point scope {scope!r}")
    return {patient: float(np.mean(values)) for patient, values in result.items()}


def _structured_record_metric(
    report: dict[str, Any], *, panel: str, metric_path: str,
) -> dict[str, float]:
    result: dict[str, list[float]] = {}
    records = report["whole_slide_structured_field_evaluation"]["per_slide_records"]
    for row in records:
        flat = _flatten_structured(row["structured_field"]["panels"][panel])
        value = _finite(flat.get(metric_path))
        if math.isfinite(value):
            result.setdefault(str(row["patient_id"]), []).append(value)
    return {patient: float(np.mean(values)) for patient, values in result.items()}


def _t_interval(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    if len(array) < 2:
        return mean, sd, float("nan"), float("nan")
    half = float(student_t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)) )
    return mean, sd, mean - half, mean + half


def _hierarchical_delta(
    left: dict[int, dict[str, float]], right: dict[int, dict[str, float]], *,
    higher_is_better: bool, seed: int, n_bootstrap: int,
) -> dict[str, float | int]:
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("hierarchical comparison has no shared random seeds")
    patients = sorted(set.intersection(*(
        set(left[value]) & set(right[value]) for value in seeds
    )))
    if not patients:
        raise ValueError("hierarchical comparison has no shared held-out patients")
    matrix = np.asarray([
        [
            (left[run_seed][patient] - right[run_seed][patient])
            if higher_is_better else (right[run_seed][patient] - left[run_seed][patient])
            for patient in patients
        ]
        for run_seed in seeds
    ], dtype=np.float64)
    observed = float(matrix.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(int(n_bootstrap)):
        seed_rows = rng.integers(0, len(seeds), size=len(seeds))
        patient_columns = rng.integers(0, len(patients), size=len(patients))
        draws[index] = matrix[np.ix_(seed_rows, patient_columns)].mean()
    low, high = (float(value) for value in np.quantile(draws, [0.025, 0.975]))
    return {
        "mean_delta": observed, "ci95_low": low, "ci95_high": high,
        "n_seeds": len(seeds), "n_patients": len(patients),
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    records: dict[str, dict[int, tuple[Path, dict[str, Any]]]], *,
    output_dir: str | Path, n_bootstrap: int = 10_000,
) -> dict[str, Path]:
    if set(records) != set(ARCHITECTURES):
        raise ValueError(f"expected architectures={ARCHITECTURES}, got={sorted(records)}")
    seed_sets = {architecture: set(records[architecture]) for architecture in ARCHITECTURES}
    expected_seeds = seed_sets[ARCHITECTURES[0]]
    if len(expected_seeds) != 3:
        raise ValueError(f"exactly three independent seeds are required; got={sorted(expected_seeds)}")
    for architecture in ARCHITECTURES:
        if seed_sets[architecture] != expected_seeds:
            raise ValueError(
                f"{architecture}: expected seeds={sorted(expected_seeds)}, "
                f"got={sorted(records[architecture])}"
            )

    identity = None
    audit_rows = []
    for architecture, by_seed in records.items():
        for run_seed, (path, report) in by_seed.items():
            audit = audit_report(report, path, expected_seed=run_seed)
            whole_records = report["whole_slide_structured_field_evaluation"]["per_slide_records"]
            current = tuple(sorted((str(row["sample_id"]), int(row["n_spots"])) for row in whole_records))
            if identity is None:
                identity = current
            elif current != identity:
                raise ValueError(f"{path}: held-out slide/spot identity differs across seeds")
            audit_rows.append({
                "architecture": architecture, "seed": run_seed,
                **audit, "report": str(path),
            })

    seed_rows: list[dict[str, Any]] = []
    for architecture, by_seed in records.items():
        for run_seed, (path, report) in sorted(by_seed.items()):
            for scope, getter in (
                ("fixed_mask", _fixed_panel_metrics),
                ("whole_slide", _whole_panel_metrics),
            ):
                for panel in PANELS:
                    metrics = getter(report, panel)
                    seed_rows.append({
                        "architecture": architecture, "seed": run_seed,
                        "scope": scope, "panel": panel,
                        **{metric: _patient_mean(metrics.get(metric)) for metric in POINT_METRICS},
                        "checkpoint_step": report.get("checkpoint_step"),
                        "checkpoint_masks_seen": report.get("checkpoint_masks_seen"),
                        "source": str(path),
                    })

    summary_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for scope in ("fixed_mask", "whole_slide"):
            for panel in PANELS:
                selected = [
                    row for row in seed_rows
                    if row["architecture"] == architecture
                    and row["scope"] == scope and row["panel"] == panel
                ]
                row: dict[str, Any] = {
                    "architecture": architecture, "scope": scope,
                    "panel": panel, "n_seeds": len(selected),
                }
                for metric in POINT_METRICS:
                    values = [_finite(value[metric]) for value in selected]
                    finite = [value for value in values if math.isfinite(value)]
                    if not finite:
                        row.update({
                            f"{metric}_mean": float("nan"), f"{metric}_sd": float("nan"),
                            f"{metric}_min": float("nan"), f"{metric}_max": float("nan"),
                        })
                    else:
                        row.update({
                            f"{metric}_mean": float(np.mean(finite)),
                            f"{metric}_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
                            f"{metric}_min": float(np.min(finite)),
                            f"{metric}_max": float(np.max(finite)),
                        })
                summary_rows.append(row)

    seed_delta_rows: list[dict[str, Any]] = []
    hierarchical_rows: list[dict[str, Any]] = []
    left_arch, right_arch = ARCHITECTURES
    for scope in ("fixed_mask", "whole_slide"):
        for panel in PANELS:
            for metric in POINT_METRICS:
                # Panel-level evaluator records intentionally do not define a
                # flattened non-zero AUC; only the all-gene output does.
                if metric == "auc" and panel != "all_genes":
                    continue
                higher = metric != "rmse"
                seed_deltas: dict[int, float] = {}
                left_by_seed, right_by_seed = {}, {}
                for run_seed in sorted(expected_seeds):
                    left_report = records[left_arch][run_seed][1]
                    right_report = records[right_arch][run_seed][1]
                    left_patient = _record_metric(
                        left_report, scope=scope, panel=panel, metric=metric,
                    )
                    right_patient = _record_metric(
                        right_report, scope=scope, panel=panel, metric=metric,
                    )
                    if set(left_patient) != set(right_patient):
                        raise ValueError(f"{scope}/{panel}/{metric}: patient identity differs")
                    left_by_seed[run_seed] = left_patient
                    right_by_seed[run_seed] = right_patient
                    per_patient = [
                        (left_patient[patient] - right_patient[patient])
                        if higher else (right_patient[patient] - left_patient[patient])
                        for patient in sorted(left_patient)
                    ]
                    seed_deltas[run_seed] = float(np.mean(per_patient))
                mean, sd, low, high = _t_interval(list(seed_deltas.values()))
                seed_delta_rows.append({
                    "left": left_arch, "right": right_arch, "scope": scope,
                    "panel": panel, "metric": metric,
                    "positive_means_left_better": True,
                    "mean_delta": mean, "seed_sd": sd,
                    "seed_t_ci95_low": low, "seed_t_ci95_high": high,
                    "n_seeds": len(seed_deltas),
                    **{f"seed{run_seed}_delta": seed_deltas[run_seed] for run_seed in sorted(expected_seeds)},
                })
                hierarchy = _hierarchical_delta(
                    left_by_seed, right_by_seed, higher_is_better=higher,
                    seed=41_000 + len(hierarchical_rows), n_bootstrap=n_bootstrap,
                )
                hierarchical_rows.append({
                    "left": left_arch, "right": right_arch, "scope": scope,
                    "panel": panel, "metric": metric,
                    "positive_means_left_better": True, **hierarchy,
                })

    structured_seed_rows: list[dict[str, Any]] = []
    for architecture, by_seed in records.items():
        for run_seed, (path, report) in sorted(by_seed.items()):
            structured = report["whole_slide_structured_field_evaluation"][
                "structured_metrics_patient_aggregated"
            ]
            for panel in PANELS:
                flat = _flatten_structured(structured[panel])
                structured_seed_rows.append({
                    "architecture": architecture, "seed": run_seed, "panel": panel,
                    **{name: _finite(flat.get(metric_path)) for name, metric_path in STRUCTURED_METRICS.items()},
                    "source": str(path),
                })

    structured_delta_rows: list[dict[str, Any]] = []
    for panel in PANELS:
        for output_name, metric_path in STRUCTURED_METRICS.items():
            left_by_seed, right_by_seed, seed_deltas = {}, {}, {}
            unavailable = False
            for run_seed in sorted(expected_seeds):
                left_report = records[left_arch][run_seed][1]
                right_report = records[right_arch][run_seed][1]
                left_patient = _structured_record_metric(left_report, panel=panel, metric_path=metric_path)
                right_patient = _structured_record_metric(right_report, panel=panel, metric_path=metric_path)
                if not left_patient and not right_patient:
                    unavailable = True
                    break
                if set(left_patient) != set(right_patient):
                    raise ValueError(f"structured/{panel}/{output_name}: patient identity differs")
                left_by_seed[run_seed], right_by_seed[run_seed] = left_patient, right_patient
                seed_deltas[run_seed] = float(np.mean([
                    left_patient[patient] - right_patient[patient]
                    for patient in sorted(left_patient)
                ]))
            if unavailable:
                structured_delta_rows.append({
                    "left": left_arch, "right": right_arch, "panel": panel,
                    "metric": output_name, "positive_means_left_better": True,
                    "mean_delta": float("nan"), "seed_sd": float("nan"),
                    "seed_t_ci95_low": float("nan"), "seed_t_ci95_high": float("nan"),
                    **{f"seed{run_seed}_delta": float("nan") for run_seed in sorted(expected_seeds)},
                    "hierarchical_ci95_low": float("nan"),
                    "hierarchical_ci95_high": float("nan"),
                    "n_seeds": len(expected_seeds), "n_patients": 0,
                })
                continue
            mean, sd, low, high = _t_interval(list(seed_deltas.values()))
            hierarchy = _hierarchical_delta(
                left_by_seed, right_by_seed, higher_is_better=True,
                seed=71_000 + len(structured_delta_rows), n_bootstrap=n_bootstrap,
            )
            structured_delta_rows.append({
                "left": left_arch, "right": right_arch, "panel": panel,
                "metric": output_name, "positive_means_left_better": True,
                "mean_delta": mean, "seed_sd": sd,
                "seed_t_ci95_low": low, "seed_t_ci95_high": high,
                **{f"seed{run_seed}_delta": seed_deltas[run_seed] for run_seed in sorted(expected_seeds)},
                "hierarchical_ci95_low": hierarchy["ci95_low"],
                "hierarchical_ci95_high": hierarchy["ci95_high"],
                "n_seeds": hierarchy["n_seeds"], "n_patients": hierarchy["n_patients"],
            })

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": output / "completeness_audit.tsv",
        "per_seed": output / "per_seed_point_metrics.tsv",
        "seed_summary": output / "seed_summary.tsv",
        "seed_delta": output / "architecture_delta_by_seed.tsv",
        "hierarchical_delta": output / "architecture_delta_hierarchical_bootstrap.tsv",
        "structured_per_seed": output / "per_seed_structured_metrics.tsv",
        "structured_delta": output / "structured_architecture_delta.tsv",
    }
    for path, rows in (
        (paths["audit"], audit_rows), (paths["per_seed"], seed_rows),
        (paths["seed_summary"], summary_rows), (paths["seed_delta"], seed_delta_rows),
        (paths["hierarchical_delta"], hierarchical_rows),
        (paths["structured_per_seed"], structured_seed_rows),
        (paths["structured_delta"], structured_delta_rows),
    ):
        _write(path, rows)
    manifest = {
        "kind": "mk_final_seed_replication_summary", "version": 1,
        "architectures": list(ARCHITECTURES), "seeds": sorted(expected_seeds),
        "panels": list(PANELS), "n_bootstrap": int(n_bootstrap),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    manifest_path = output / "summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    paths["manifest"] = manifest_path
    return paths


def discover_records(
    *, seed0_root: str | Path, replication_root: str | Path,
    replication_evaluation_root: str | Path,
) -> dict[str, dict[int, tuple[Path, dict[str, Any]]]]:
    seed0 = Path(seed0_root).expanduser().resolve()
    replication = Path(replication_root).expanduser().resolve()
    evaluation = Path(replication_evaluation_root).expanduser().resolve()
    plan = _load_json(replication / "replication_plan.json")
    records: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = {
        architecture: {} for architecture in ARCHITECTURES
    }
    for architecture in ARCHITECTURES:
        path = seed0 / f"{architecture}_validation.json"
        report = _load_json(path)
        source_seed = _config_seed(report, path)
        records[architecture][source_seed] = (path, report)
    for run_name, run in (plan.get("runs") or {}).items():
        architecture = str(run["source_arm"])
        run_seed = int(run["seed"])
        if architecture not in records:
            raise ValueError(f"unexpected replication plan entry: {run_name}")
        path = evaluation / f"{run_name}_validation.json"
        if run_seed in records[architecture]:
            raise ValueError(f"duplicate {architecture} seed {run_seed}")
        records[architecture][run_seed] = (path, _load_json(path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-evaluation-root", required=True)
    parser.add_argument("--replication-root", required=True)
    parser.add_argument("--replication-evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    args = parser.parse_args()
    records = discover_records(
        seed0_root=args.seed0_evaluation_root,
        replication_root=args.replication_root,
        replication_evaluation_root=args.replication_evaluation_root,
    )
    outputs = summarize(records, output_dir=args.output_dir, n_bootstrap=args.n_bootstrap)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
