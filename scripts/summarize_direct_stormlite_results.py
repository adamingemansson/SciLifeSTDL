#!/usr/bin/env python3
"""Summarize direct absolute-GEX models against honest external baselines."""
import json
from pathlib import Path


RUNS = [
    ("harmonic_external", "missing_tissue_simple_local_harmonic_k32", True),
    ("direct_neither", "missing_tissue_direct_transformer_neither_seed10", False),
    ("direct_gex_novae", "missing_tissue_direct_transformer_gex_novae_seed10", False),
    ("direct_full_novae", "missing_tissue_direct_transformer_full_novae_seed10", False),
    ("direct_mome_novae", "missing_tissue_direct_mome_full_novae_seed10", False),
]


def main() -> None:
    root = Path("results/checkpoints/recovery_suite")
    rows = []
    for label, name, optional in RUNS:
        summary_path = root / name / "heldout_sample_summary.json"
        if not summary_path.is_file():
            if optional:
                continue
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text())
        mode = payload["primary_image_mode"]
        metrics = payload["image_modes"][mode]
        gate_path = root / name / "quality_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        rows.append({
            "label": label,
            "mode": mode,
            "pcc": float(metrics["pcc_mean"]),
            "rmse": float(metrics["rmse_mean"]),
            "mean_gate": gate.get("passed"),
            "genes": int(payload["n_evaluated_genes"]),
        })

    if len({row["genes"] for row in rows}) != 1:
        raise RuntimeError(f"runs evaluated different gene panels: {rows}")
    print(f"{'ARCHITECTURE':25s} {'MODE':12s} {'PCC':>9s} {'RMSE':>9s} {'MEAN_GATE':>10s}")
    for row in rows:
        print(
            f"{row['label']:25s} {row['mode']:12s} {row['pcc']:9.4f} "
            f"{row['rmse']:9.4f} {str(row['mean_gate']):>10s}"
        )

    harmonic = next((row for row in rows if row["label"] == "harmonic_external"), None)
    candidates = [
        row for row in rows
        if row["label"] in {"direct_full_novae", "direct_mome_novae"}
    ]
    promoted = bool(harmonic) and any(
        row["mean_gate"] is True
        and row["pcc"] > harmonic["pcc"]
        and row["rmse"] < harmonic["rmse"]
        for row in candidates
    )
    print("DECISION: " + (
        "PROMOTE the passing direct full-context model to folds/seeds"
        if promoted
        else "DO NOT PROMOTE; no direct full-context model beat both the training-mean gate and held-out harmonic"
    ))


if __name__ == "__main__":
    main()
