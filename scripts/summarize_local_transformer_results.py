#!/usr/bin/env python3
"""Summarize matched local Transformer/MoME held-out results."""
import json
from pathlib import Path


RUNS = [
    ("harmonic", "missing_tissue_simple_local_harmonic_k32", True),
    ("local_pool", "missing_tissue_simple_local_full_novae_seed10", True),
    ("transformer_neither", "missing_tissue_local_transformer_neither_seed10", False),
    ("transformer_gex_novae", "missing_tissue_local_transformer_gex_novae_seed10", False),
    ("transformer_full_novae", "missing_tissue_local_transformer_full_novae_seed10", False),
    ("mome_full_novae", "missing_tissue_local_mome_full_novae_seed10", False),
]


def main() -> None:
    root = Path("results/checkpoints/recovery_suite")
    rows = []
    for label, name, reference in RUNS:
        summary_path = root / name / "heldout_sample_summary.json"
        if not summary_path.is_file():
            if reference:
                continue
            raise FileNotFoundError(summary_path)
        payload = json.loads(summary_path.read_text())
        mode = payload["primary_image_mode"]
        metrics = payload["image_modes"][mode]
        gate_path = root / name / "quality_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        rows.append({
            "label": label, "name": name, "mode": mode,
            "pcc": float(metrics["pcc_mean"]),
            "rmse": float(metrics["rmse_mean"]),
            "gate": gate.get("passed"),
            "genes": int(payload["n_evaluated_genes"]),
        })
    if len({row["genes"] for row in rows}) != 1:
        raise RuntimeError(f"runs evaluated different gene panels: {rows}")
    print(f"{'ARCHITECTURE':26s} {'MODE':12s} {'PCC':>9s} {'RMSE':>9s} {'VAL_GATE':>10s}")
    for row in rows:
        print(
            f"{row['label']:26s} {row['mode']:12s} {row['pcc']:9.4f} "
            f"{row['rmse']:9.4f} {str(row['gate']):>10s}"
        )
    anchor = next((row for row in rows if row["label"] == "harmonic"), None)
    candidates = [row for row in rows if row["label"] in {"transformer_full_novae", "mome_full_novae"}]
    promoted = bool(anchor) and any(
        row["gate"] is True and row["pcc"] > anchor["pcc"] and row["rmse"] < anchor["rmse"]
        for row in candidates
    )
    print("DECISION: " + (
        "PROMOTE the passing full-context architecture to folds/seeds"
        if promoted else "DO NOT PROMOTE; no full-context Transformer cleared validation and harmonic"
    ))


if __name__ == "__main__":
    main()
