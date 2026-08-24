#!/usr/bin/env python3
"""Compare every individual WAE draw with its deterministic point prediction.

The input is one output directory from
``analyze_conditional_wae_predictive_diversity``.  Metrics use that audit's
same deterministic subsample of spot-gene values.  The per-slide oracle is
reported explicitly as an optimistic diagnostic, never as deployable model
performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(*, diversity_root: str, output_dir: str, tolerance: float = 1e-8) -> dict[str, Path]:
    source = Path(diversity_root).expanduser().resolve()
    report = json.loads((source / "report.json").read_text())
    draws = _read(source / "per_draw.tsv")
    convergence = _read(source / "ensemble_convergence.tsv")
    slides = sorted({row["sample_id"] for row in draws})
    if len(slides) != int(report["n_slides"]):
        raise ValueError("per-draw rows do not cover every reported slide")

    deterministic: dict[str, dict[str, float]] = {}
    for sample_id in slides:
        candidates = [row for row in convergence if row["sample_id"] == sample_id]
        if not candidates:
            raise ValueError(f"{sample_id}: no ensemble convergence records")
        reference = min(candidates, key=lambda row: float(row["sampled_value_rmse_vs_deterministic"]))
        distance = float(reference["sampled_value_rmse_vs_deterministic"])
        if distance > 1e-6:
            raise ValueError(
                f"{sample_id}: no ensemble entry reproduces the deterministic point prediction; "
                f"closest RMSE={distance}"
            )
        deterministic[sample_id] = {
            "pcc": float(reference["sampled_value_pcc_vs_target"]),
            "rmse": float(reference["sampled_value_rmse_vs_target"]),
            "reference_n_draws": int(reference["n_draws"]),
        }

    draw_ids = sorted({int(row["draw"]) for row in draws})
    macro_rows = []
    for draw in draw_ids:
        selected = [row for row in draws if int(row["draw"]) == draw]
        if {row["sample_id"] for row in selected} != set(slides):
            raise ValueError(f"draw {draw} does not cover every slide")
        macro_rows.append({
            "draw": draw,
            "n_slides": len(selected),
            "macro_pcc": float(np.mean([float(row["draw_vs_target_pcc"]) for row in selected])),
            "macro_rmse": float(np.mean([float(row["draw_vs_target_rmse"]) for row in selected])),
            "slides_beating_deterministic_pcc": sum(
                float(row["draw_vs_target_pcc"]) > deterministic[row["sample_id"]]["pcc"] + tolerance
                for row in selected
            ),
            "slides_beating_deterministic_rmse": sum(
                float(row["draw_vs_target_rmse"]) < deterministic[row["sample_id"]]["rmse"] - tolerance
                for row in selected
            ),
        })

    oracle_rows = []
    for sample_id in slides:
        selected = [row for row in draws if row["sample_id"] == sample_id]
        best_pcc = max(selected, key=lambda row: float(row["draw_vs_target_pcc"]))
        best_rmse = min(selected, key=lambda row: float(row["draw_vs_target_rmse"]))
        reference = deterministic[sample_id]
        oracle_rows.append({
            "sample_id": sample_id,
            "organ": selected[0]["organ"],
            "deterministic_pcc": reference["pcc"],
            "best_draw_by_pcc": int(best_pcc["draw"]),
            "best_draw_pcc": float(best_pcc["draw_vs_target_pcc"]),
            "oracle_pcc_improvement": float(best_pcc["draw_vs_target_pcc"]) - reference["pcc"],
            "deterministic_rmse": reference["rmse"],
            "best_draw_by_rmse": int(best_rmse["draw"]),
            "best_draw_rmse": float(best_rmse["draw_vs_target_rmse"]),
            "oracle_rmse_improvement": reference["rmse"] - float(best_rmse["draw_vs_target_rmse"]),
        })

    deterministic_macro_pcc = float(np.mean([row["pcc"] for row in deterministic.values()]))
    deterministic_macro_rmse = float(np.mean([row["rmse"] for row in deterministic.values()]))
    best_pcc_draw = max(macro_rows, key=lambda row: row["macro_pcc"])
    best_rmse_draw = min(macro_rows, key=lambda row: row["macro_rmse"])
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    macro_path = root / "individual_draw_macro.tsv"
    oracle_path = root / "per_slide_oracle_best_draw.tsv"
    _write(macro_path, macro_rows)
    _write(oracle_path, oracle_rows)
    summary_path = root / "best_individual_draw_summary.json"
    summary_path.write_text(json.dumps({
        "kind": "mk_wae_best_individual_draw_diagnostic",
        "arm": report["arm"],
        "n_slides": len(slides),
        "n_draws": len(draw_ids),
        "metric_scope": "same deterministic subsample of spot-gene values per held-out slide",
        "deterministic_macro_pcc": deterministic_macro_pcc,
        "deterministic_macro_rmse": deterministic_macro_rmse,
        "best_single_global_draw_by_pcc": best_pcc_draw,
        "best_single_global_draw_by_rmse": best_rmse_draw,
        "best_global_draw_pcc_improvement": best_pcc_draw["macro_pcc"] - deterministic_macro_pcc,
        "best_global_draw_rmse_improvement": deterministic_macro_rmse - best_rmse_draw["macro_rmse"],
        "per_slide_oracle_macro_pcc_improvement": float(np.mean([
            row["oracle_pcc_improvement"] for row in oracle_rows
        ])),
        "per_slide_oracle_macro_rmse_improvement": float(np.mean([
            row["oracle_rmse_improvement"] for row in oracle_rows
        ])),
        "warning": (
            "The per-slide oracle selects a draw after observing target GEX and is not a valid "
            "deployable predictor. Only a fixed global draw or predeclared ensemble is comparable."
        ),
    }, indent=2, allow_nan=False) + "\n")
    return {"draw_macro": macro_path, "oracle": oracle_path, "summary": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diversity-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = summarize(diversity_root=args.diversity_root, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
