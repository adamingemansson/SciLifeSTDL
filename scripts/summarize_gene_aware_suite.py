#!/usr/bin/env python3
"""Summarize capacity gates and honest held-out gene-aware results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.gene_aware_suite_lib import DEFAULT_MATRIX, load_matrix, resolve_run


METRICS = ("pcc", "rmse", "nonzero_auc", "st_fid", "st_mmd")


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _fmt(value, width: int = 9) -> str:
    return f"{value:{width}.5f}" if value is not None else f"{'NA':>{width}s}"


def _heldout_metrics(entry, cfg):
    root = Path(str(cfg.training.checkpoint_dir))
    if str(entry.scope) == "cross":
        payload = json.loads((root / "heldout_sample_summary.json").read_text())
        mode = str(payload["primary_image_mode"])
        source = payload["image_modes"][mode]
        values = {metric: _finite(source.get(f"{metric}_mean")) for metric in METRICS}
        all_zero = payload["image_modes"].get("all_zero")
        all_zero_pcc = _finite(all_zero.get("pcc_mean")) if all_zero else None
        all_zero_rmse = _finite(all_zero.get("rmse_mean")) if all_zero else None
        genes = payload.get("n_evaluated_genes")
        units = len(payload.get("test_sample_ids", []))
    else:
        payload = json.loads((root / "audit_test_metrics.json").read_text())
        mode = str(payload["primary_image_mode"])
        source = payload["image_modes"][mode]["summary"]
        values = {
            metric: _finite(
                source.get(metric, {}).get("mean")
                if isinstance(source.get(metric), dict) else source.get(metric)
            )
            for metric in METRICS
        }
        all_zero = payload["image_modes"].get("all_zero")
        all_zero_summary = all_zero.get("summary", {}) if all_zero else {}
        all_zero_pcc_value = all_zero_summary.get("pcc")
        all_zero_rmse_value = all_zero_summary.get("rmse")
        all_zero_pcc = _finite(
            all_zero_pcc_value.get("mean")
            if isinstance(all_zero_pcc_value, dict) else all_zero_pcc_value
        )
        all_zero_rmse = _finite(
            all_zero_rmse_value.get("mean")
            if isinstance(all_zero_rmse_value, dict) else all_zero_rmse_value
        )
        genes = payload.get("n_evaluated_genes")
        units = payload.get("n_test_masks")
    gate_path = root / "quality_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
    exclusion_path = root / "training_exclusion.json"
    exclusion = json.loads(exclusion_path.read_text()) if exclusion_path.is_file() else {}
    anchor = _finite(gate.get("anchor_score"))
    best = _finite(gate.get("best_score"))
    return {
        "primary_image_mode": mode,
        "n_evaluated_genes": genes,
        "n_test_units": units,
        "n_training_excluded": exclusion.get("n_excluded", 0),
        **values,
        "all_zero_pcc": all_zero_pcc,
        "all_zero_rmse": all_zero_rmse,
        "he_sensitivity_pcc": (
            values["pcc"] - all_zero_pcc
            if mode == "target_zero" and all_zero_pcc is not None else None
        ),
        "he_sensitivity_rmse": (
            values["rmse"] - all_zero_rmse
            if mode == "target_zero" and all_zero_rmse is not None else None
        ),
        "validation_gate_passed": gate.get("passed"),
        "validation_anchor_rmse": anchor,
        "validation_best_rmse": best,
        "validation_improvement": (
            anchor - best if anchor is not None and best is not None else None
        ),
        "prediction_delta_rms": _finite(gate.get("correction_rms")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    heldout_rows, overfit_rows, missing = [], [], []
    for index, entry in enumerate(matrix.runs):
        _entry, cfg = resolve_run(matrix, index)
        root = Path(str(cfg.training.checkpoint_dir))
        common = {
            "id": str(entry.id), "scope": str(entry.scope),
            "family": str(entry.family), "experiment_name": str(entry.name),
            "transport_k": int(cfg.model.params.transport_k),
            "transport_heads": int(cfg.model.params.transport_heads),
            "conditioning_mode": str(cfg.model.params.conditioning_mode),
            "gene_encoder_type": str(cfg.model.params.gene_encoder_type),
            "modality_ablation": str(cfg.data.modality_ablation),
            "correlation_loss_weight": float(cfg.model.params.correlation_loss_weight),
        }
        if str(entry.stage) == "overfit":
            path = root / "quality_gate.json"
            if not path.is_file():
                missing.append(str(entry.name))
                continue
            gate = json.loads(path.read_text())
            anchor, best = float(gate["anchor_score"]), float(gate["best_score"])
            if gate.get("correction_rms") is None:
                raise RuntimeError(f"{entry.name} quality gate has no correction_rms")
            overfit_rows.append({
                **common, "passed": gate.get("passed"),
                "anchor_rmse": anchor, "best_rmse": best,
                "improvement": anchor - best,
                "correction_rms": gate.get("correction_rms"),
            })
        else:
            expected = root / (
                "heldout_sample_summary.json" if str(entry.scope) == "cross"
                else "audit_test_metrics.json"
            )
            if not expected.is_file():
                missing.append(str(entry.name))
                continue
            heldout_rows.append({**common, **_heldout_metrics(entry, cfg)})

    if missing and not args.allow_incomplete:
        raise RuntimeError(f"missing {len(missing)} final artifacts: {missing}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if overfit_rows:
        with (output / "overfit_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(overfit_rows[0]))
            writer.writeheader()
            writer.writerows(overfit_rows)
    if heldout_rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(heldout_rows[0]))
            writer.writeheader()
            writer.writerows(heldout_rows)

    print("CAPACITY GATE")
    print(f"{'ID':5s} {'ANCHOR':>9s} {'BEST':>9s} {'IMPROVE':>9s} {'DELTA':>9s} {'PASS':>6s}")
    for row in overfit_rows:
        print(
            f"{row['id']:5s} {_fmt(row['anchor_rmse'])} {_fmt(row['best_rmse'])} "
            f"{_fmt(row['improvement'])} {_fmt(float(row['correction_rms']))} "
            f"{str(row['passed']):>6s}"
        )
    print("\nHELD-OUT RESULTS")
    print(f"{'ID':5s} {'SCOPE':12s} {'PCC':>9s} {'RMSE':>9s} {'AUC':>9s} {'VAL':>6s}")
    for row in heldout_rows:
        print(
            f"{row['id']:5s} {row['scope']:12s} {_fmt(row['pcc'])} "
            f"{_fmt(row['rmse'])} {_fmt(row['nonzero_auc'])} "
            f"{str(row['validation_gate_passed']):>6s}"
        )
    if missing:
        print(f"INCOMPLETE: {', '.join(missing)}")
        return

    by_id = {row["id"]: row for row in heldout_rows}
    geometry = by_id["C01"]
    conditioned = [by_id[key] for key in ("C02", "C03", "C04", "C05", "C06", "C07", "C08")]
    promoted = []
    if geometry["pcc"] is not None and geometry["rmse"] is not None:
        promoted = [
            row for row in conditioned
            if row["validation_gate_passed"] is True
            and row["pcc"] is not None and row["rmse"] is not None
            and row["pcc"] >= geometry["pcc"] + 0.005
            and row["rmse"] <= geometry["rmse"] - 0.001
        ]
    print("\nDECISION: " + (
        f"PROMOTE {max(promoted, key=lambda row: row['pcc'])['experiment_name']} to folds/seeds"
        if promoted else
        "DO NOT PROMOTE: no conditioned model cleared validation and beat geometry by the declared PCC/RMSE margins"
    ))
    full, gex = by_id["C05"], by_id["C04"]
    if all(row[metric] is not None for row in (full, gex) for metric in ("pcc", "rmse")):
        print(
            f"Matched H&E contribution (C05-C04): PCC {full['pcc'] - gex['pcc']:+.5f}, "
            f"RMSE {full['rmse'] - gex['rmse']:+.5f}"
        )
    else:
        print("Matched H&E contribution (C05-C04): unavailable because a metric is non-finite")
    if full["he_sensitivity_pcc"] is not None and full["he_sensitivity_rmse"] is not None:
        print(
            f"Within-model context-H&E sensitivity (C05 target_zero-all_zero): "
            f"PCC {full['he_sensitivity_pcc']:+.5f}, "
            f"RMSE {full['he_sensitivity_rmse']:+.5f}"
        )
    else:
        print("Within-model context-H&E sensitivity: unavailable because a metric is non-finite")
    print(f"Reports: {output / 'overfit_summary.csv'} and {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
