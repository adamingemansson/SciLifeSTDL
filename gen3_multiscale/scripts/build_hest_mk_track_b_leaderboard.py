#!/usr/bin/env python3
"""Build a fail-closed Track-B whole-slide leaderboard.

The frozen-feature ridge report is the cohort reference.  Conditional MK,
frozen-feature MLP/ridge and released STPath reports are admitted only when
they use the identical expanded validation cohort, spot counts, target space,
and H&E-only deterministic inference contract.  Rejected reports and reasons
are written to a machine-readable audit file.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from gen3_multiscale.scripts.compare_mk_whole_slide_to_ridge import (
    FIELDS,
    PANELS,
    STRUCTURED_FIELDS,
    _contract_from_ridge,
    _load,
    _manifest_contract,
    _manifest_contract_for_mk,
    _method,
    _paired_deltas_by_panel,
    _point_rows,
    _structured_rows,
    _suite,
)


SUPPORTED_KINDS = {
    "conditional_wae_supervisor_evaluation",
    "frozen_feature_pca_ridge_whole_slide_benchmark",
    "frozen_feature_pca_mlp_whole_slide_benchmark",
    "stpath_supervisor_zero_shot_evaluation_report",
}
EXTRA_FIELDS = ("method_family", "pretraining_overlap_caveat")


def _manifest_contract_for_report(report: dict) -> dict:
    if report.get("kind") == "conditional_wae_supervisor_evaluation":
        return _manifest_contract_for_mk(report)
    path = Path(str(report.get("manifest_path") or "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"reported manifest is unavailable: {path}")
    return _manifest_contract(path)


def _method_name(report: dict, path: Path) -> str:
    kind = report.get("kind")
    if kind in {
        "frozen_feature_pca_ridge_whole_slide_benchmark",
        "frozen_feature_pca_mlp_whole_slide_benchmark",
    }:
        decoder = "ridge" if "ridge" in kind else "mlp"
        return f"{report.get('image_encoder')}_pca{report.get('pca_components')}_{decoder}"
    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        return f"stpath_{report.get('task', 'unknown')}"
    return _method(report, path)


def _sections(report: dict) -> tuple[list[dict], dict, dict]:
    kind = report.get("kind")
    if kind in {
        "frozen_feature_pca_ridge_whole_slide_benchmark",
        "frozen_feature_pca_mlp_whole_slide_benchmark",
    }:
        return (
            report.get("per_slide_records") or [],
            report.get("point_metrics_patient_aggregated") or {},
            report.get("structured_metrics_patient_aggregated") or {},
        )
    if kind == "conditional_wae_supervisor_evaluation":
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        return (
            whole.get("per_slide_records") or [],
            whole.get("point_metrics_patient_aggregated") or {},
            whole.get("structured_metrics_patient_aggregated") or {},
        )
    whole = report.get("whole_slide_point_evaluation") or {}
    return (
        whole.get("per_slide_records") or [],
        whole.get("point_metrics_patient_aggregated") or {},
        {},
    )


def _count(row: dict) -> int:
    value = row.get("n_evaluated_spots", row.get("n_spots"))
    if value is None:
        raise ValueError("whole-slide record has no evaluated spot count")
    return int(value)


def audit_report(
    report: dict, *, expected_counts: dict[str, int], expected_contract: dict,
) -> tuple[bool, str]:
    kind = report.get("kind")
    if kind not in SUPPORTED_KINDS:
        return False, "unsupported_report_kind"
    if report.get("split") != "validation":
        return False, "wrong_split"
    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        if report.get("task") != "he_to_st":
            return False, "stpath_not_he_to_st"
        stpath = report.get("stpath") or {}
        if stpath.get("query_expression_visible") is not False:
            return False, "query_gex_not_explicitly_hidden"
        if stpath.get("surrounding_expression_visible") is not False:
            return False, "surrounding_gex_not_explicitly_hidden"
        whole = report.get("whole_slide_point_evaluation") or {}
        if whole.get("primary_prediction") != "pretrained_stpath_h_and_e_only":
            return False, "wrong_primary_prediction"
        if whole.get("target_space") != "normalize_total_then_log1p":
            return False, "wrong_target_space"
    elif kind == "conditional_wae_supervisor_evaluation":
        if report.get("query_gex_visible") is not False:
            return False, "query_gex_not_explicitly_hidden"
        whole = report.get("whole_slide_structured_field_evaluation") or {}
        if whole.get("primary_prediction") != "deterministic_h_and_e_point_prediction":
            return False, "wrong_primary_prediction"
        if whole.get("target_gex_visible_to_model") is not False:
            return False, "whole_slide_target_gex_not_explicitly_hidden"
    else:
        notes = report.get("comparability_notes") or {}
        if notes.get("target_gex_visible_to_model") is not False:
            return False, "query_gex_not_explicitly_hidden"
        if report.get("missing_image_policy") != "zero":
            return False, "missing_image_policy_not_zero"

    records, metrics, _structured = _sections(report)
    try:
        actual_counts = {str(row["sample_id"]): _count(row) for row in records}
    except (KeyError, TypeError, ValueError):
        return False, "malformed_whole_slide_records"
    if actual_counts != expected_counts:
        return False, "cohort_or_spot_count_mismatch"
    if "all_genes" not in metrics:
        return False, "missing_all_genes_metrics"
    try:
        actual_contract = _manifest_contract_for_report(report)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False, "dataset_contract_unverifiable"
    if actual_contract != expected_contract:
        return False, "dataset_contract_mismatch"
    return True, "compatible"


def _family_and_caveat(report: dict) -> tuple[str, str]:
    kind = report.get("kind")
    if kind == "stpath_supervisor_zero_shot_evaluation_report":
        return (
            "released_stpath",
            "Released STPath checkpoint was trained on HEST-1k; this is not an unseen-pretraining comparison.",
        )
    if kind == "conditional_wae_supervisor_evaluation":
        return (
            "mk_learned_decoder",
            "Uses frozen pathology-foundation features; pretraining overlap must be reported separately.",
        )
    decoder = "ridge" if "ridge" in str(kind) else "mlp"
    return (
        f"frozen_feature_{decoder}",
        "Uses frozen pathology-foundation features; pretraining overlap must be reported separately.",
    )


def _scan(roots: list[str]) -> list[Path]:
    result: set[Path] = set()
    for raw in roots:
        path = Path(raw).expanduser()
        if path.is_file():
            result.add(path.resolve())
        elif path.is_dir():
            result.update(item.resolve() for item in path.rglob("*.json"))
    return sorted(result)


def build(*, ridge_report: str, roots: list[str], output: str) -> dict[str, Path]:
    ridge_path = Path(ridge_report).expanduser().resolve()
    ridge = _load(ridge_path)
    expected_counts, reference, contract = _contract_from_ridge(ridge)
    if contract is None:
        raise ValueError("ridge manifest is unavailable; exact comparison is impossible")

    accepted: dict[str, tuple[float, Path, dict]] = {}
    rejected: list[dict[str, str]] = []
    for path in _scan(roots):
        try:
            report = _load(path)
        except Exception as error:
            rejected.append({"source": str(path), "reason": f"invalid_json:{type(error).__name__}"})
            continue
        if report.get("kind") not in SUPPORTED_KINDS:
            continue
        compatible, reason = audit_report(
            report, expected_counts=expected_counts, expected_contract=contract,
        )
        if not compatible:
            rejected.append({"source": str(path), "reason": reason})
            continue
        method = _method_name(report, path)
        current = accepted.get(method)
        item = (path.stat().st_mtime, path, report)
        if current is None or item[0] > current[0]:
            accepted[method] = item

    # Ensure the exact reference enters even if roots do not include it.
    accepted[_method_name(ridge, ridge_path)] = (
        ridge_path.stat().st_mtime, ridge_path, ridge,
    )
    rows: list[dict[str, Any]] = []
    structured_rows: list[dict[str, Any]] = []
    reference_records = ridge["per_slide_records"]
    structured_reference = ridge.get("structured_metrics_patient_aggregated") or {}
    for _mtime, path, report in accepted.values():
        method = _method_name(report, path)
        records, point_metrics, structured_metrics = _sections(report)
        point_deltas = _paired_deltas_by_panel(reference_records, records)
        current_rows = _point_rows(
            method=method, suite=_suite(report, path), source=path,
            metrics_by_panel=point_metrics, reference=reference,
            n_slides=len(expected_counts), checkpoint_step=report.get("checkpoint_step"),
            paired_deltas=point_deltas,
        )
        family, caveat = _family_and_caveat(report)
        for row in current_rows:
            row.update(method_family=family, pretraining_overlap_caveat=caveat)
        rows.extend(current_rows)
        if structured_metrics:
            structured_deltas = _paired_deltas_by_panel(
                reference_records, records, structured=True,
            )
            current_structured = _structured_rows(
                method=method, suite=_suite(report, path), source=path,
                metrics_by_panel=structured_metrics, reference=structured_reference,
                n_slides=len(expected_counts), checkpoint_step=report.get("checkpoint_step"),
                paired_deltas=structured_deltas,
            )
            for row in current_structured:
                row.update(method_family=family, pretraining_overlap_caveat=caveat)
            structured_rows.extend(current_structured)

    for panel in PANELS:
        selected = [row for row in rows if row["panel"] == panel]
        selected.sort(key=lambda row: float(row["pcc"]), reverse=True)
        for rank, row in enumerate(selected, 1):
            row["rank"] = rank
    rows.sort(key=lambda row: (PANELS.index(row["panel"]), row["rank"]))
    structured_rows.sort(key=lambda row: (PANELS.index(row["panel"]), row["method"]))

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*FIELDS, *EXTRA_FIELDS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    structured_path = output_path.with_name(f"{output_path.stem}_structured{output_path.suffix}")
    with structured_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=(*STRUCTURED_FIELDS, *EXTRA_FIELDS), delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(structured_rows)
    audit_path = output_path.with_name(f"{output_path.stem}_audit.json")
    audit_path.write_text(json.dumps({
        "version": 1, "kind": "hest_mk_track_b_leaderboard_audit",
        "reference": str(ridge_path), "dataset_contract": contract,
        "expected_spot_counts": expected_counts,
        "accepted": [
            {"method": method, "source": str(item[1])}
            for method, item in sorted(accepted.items())
        ],
        "rejected": rejected,
    }, indent=2, sort_keys=True))
    return {"point": output_path, "structured": structured_path, "audit": audit_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-report", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outputs = build(ridge_report=args.ridge_report, roots=args.roots, output=args.output)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
