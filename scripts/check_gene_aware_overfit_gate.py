#!/usr/bin/env python3
"""Promote held-out jobs only after a conditioned overfit arm learns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.gene_aware_suite_lib import DEFAULT_MATRIX, load_matrix, resolve_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matrix = load_matrix(args.matrix)
    rows = []
    for index, entry in enumerate(matrix.runs):
        if str(entry.stage) != "overfit":
            continue
        _entry, cfg = resolve_run(matrix, index)
        path = Path(str(cfg.training.checkpoint_dir)) / "quality_gate.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        gate = json.loads(path.read_text())
        anchor = float(gate["anchor_score"])
        best = float(gate["best_score"])
        row = {
            "id": str(entry.id), "name": str(entry.name),
            "conditioned": str(cfg.model.params.conditioning_mode) == "storm_lite",
            "passed": gate.get("passed") is True,
            "anchor_rmse": anchor, "best_rmse": best,
            "improvement": anchor - best,
            "correction_rms": gate.get("correction_rms"),
        }
        if row["correction_rms"] is None:
            raise RuntimeError(f"{entry.name} quality gate has no correction_rms")
        rows.append(row)
    conditioned_passes = [row for row in rows if row["conditioned"] and row["passed"]]
    decision = {
        "version": 1,
        "passed": bool(conditioned_passes),
        "criterion": (
            "at least one GEX/Novae or full StormLite arm must beat the exact "
            "training-mask local-mean anchor by 0.002 RMSE with correction RMS >= 0.005"
        ),
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2))
    print(f"{'ID':5s} {'ANCHOR':>9s} {'BEST':>9s} {'IMPROVE':>9s} {'DELTA':>9s} {'PASS':>6s}")
    for row in rows:
        print(
            f"{row['id']:5s} {row['anchor_rmse']:9.5f} {row['best_rmse']:9.5f} "
            f"{row['improvement']:9.5f} {float(row['correction_rms']):9.5f} "
            f"{str(row['passed']):>6s}"
        )
    if not conditioned_passes:
        raise SystemExit(
            "GENE-AWARE CAPACITY GATE: FAIL. Held-out jobs were not started."
        )
    print("GENE-AWARE CAPACITY GATE: PASS. Held-out jobs may start.")


if __name__ == "__main__":
    main()
