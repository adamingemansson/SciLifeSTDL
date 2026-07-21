#!/usr/bin/env python3
"""Write one honest, comparable table for the 40-run transport marathon."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.transport_marathon_lib import DEFAULT_MATRIX, load_matrix, resolve_run


METRIC_NAMES = (
    "pcc", "rmse", "nonzero_auc", "st_fid", "st_mmd",
    "spatial_domain_plausibility",
)


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _primary_metrics(cfg, scope: str) -> tuple[str, dict, int | None, int | None]:
    root = Path(str(cfg.training.checkpoint_dir))
    if scope == "cross":
        payload = json.loads((root / "heldout_sample_summary.json").read_text())
        mode = str(payload["primary_image_mode"])
        source = payload["image_modes"][mode]
        metrics = {name: _finite(source.get(f"{name}_mean")) for name in METRIC_NAMES}
        return mode, metrics, payload.get("n_evaluated_genes"), len(payload.get("test_sample_ids", []))

    payload = json.loads((root / "audit_test_metrics.json").read_text())
    mode = str(payload["primary_image_mode"])
    source = payload["image_modes"][mode]["summary"]
    metrics = {}
    for name in METRIC_NAMES:
        value = source.get(name)
        metrics[name] = _finite(value.get("mean") if isinstance(value, dict) else value)
    return mode, metrics, payload.get("n_evaluated_genes"), payload.get("n_test_masks")


def _gate(checkpoint: Path) -> dict:
    path = checkpoint / "quality_gate.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    rows = []
    missing = []
    for index in range(len(matrix.runs)):
        entry, cfg = resolve_run(matrix, index)
        checkpoint = Path(str(cfg.training.checkpoint_dir))
        expected = checkpoint / (
            "heldout_sample_summary.json" if str(entry.scope) == "cross"
            else "audit_test_metrics.json"
        )
        if not expected.is_file():
            missing.append(str(entry.name))
            continue
        mode, metrics, genes, n_test_units = _primary_metrics(cfg, str(entry.scope))
        gate = _gate(checkpoint)
        exclusion_path = checkpoint / "training_exclusion.json"
        exclusion = (
            json.loads(exclusion_path.read_text()) if exclusion_path.is_file() else {}
        )
        anchor = _finite(gate.get("anchor_score"))
        best = _finite(gate.get("best_score"))
        improvement = anchor - best if anchor is not None and best is not None else None
        rows.append({
            "id": str(entry.id),
            "scope": str(entry.scope),
            "family": str(entry.family),
            "profile": str(entry.profile),
            "experiment_name": str(entry.name),
            "seed": int(entry.seed),
            "transport_k": int(cfg.model.params.transport_k),
            "conditioning_mode": str(cfg.model.params.conditioning_mode),
            "gene_encoder_type": str(cfg.model.params.gene_encoder_type),
            "modality_ablation": str(cfg.data.modality_ablation),
            "novae_mode": str(cfg.data.novae_mode),
            "primary_image_mode": mode,
            "n_evaluated_genes": genes,
            "n_test_units": n_test_units,
            "n_training_excluded": exclusion.get("n_excluded", 0),
            **metrics,
            "validation_gate_passed": gate.get("passed"),
            "validation_anchor_rmse": anchor,
            "validation_best_rmse": best,
            "validation_improvement": improvement,
            "prediction_delta_rms": _finite(
                gate.get("prediction_delta_rms", gate.get("correction_rms"))
            ),
        })

    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(missing)}/40 runs lack final metrics; first missing: {missing[:5]}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.csv"
    if not rows:
        raise RuntimeError("no completed marathon metrics found")
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Transport marathon results: {len(rows)}/40 complete")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    for scope in ("cross", "single_int7", "single_int8"):
        subset = [row for row in rows if row["scope"] == scope and row["pcc"] is not None]
        subset.sort(key=lambda row: (-row["pcc"], row["rmse"] if row["rmse"] is not None else math.inf))
        print(f"\n{scope}: top PCC")
        print(f"{'ID':5s} {'PCC':>8s} {'RMSE':>8s} {'K':>4s}  RUN")
        for row in subset[:5]:
            print(
                f"{row['id']:5s} {row['pcc']:8.4f} {row['rmse']:8.4f} "
                f"{row['transport_k']:4d}  {row['experiment_name']}"
            )
    print(f"\nCSV: {output_path}")


if __name__ == "__main__":
    main()
