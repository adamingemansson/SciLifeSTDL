#!/usr/bin/env python3
"""Print the decision table for the simple-local diagnostic."""
from __future__ import annotations

import json
from pathlib import Path


RUNS = [
    "missing_tissue_simple_local_harmonic_k32",
    "missing_tissue_simple_local_gex_novae_seed10",
    "missing_tissue_simple_local_full_novae_seed10",
    "missing_tissue_simple_local_full_mlp_seed10",
]


def main() -> None:
    root = Path("results/checkpoints/recovery_suite")
    rows = []
    for name in RUNS:
        payload = json.loads((root / name / "heldout_sample_summary.json").read_text())
        mode = payload["primary_image_mode"]
        summary = payload["image_modes"][mode]
        gate_path = root / name / "quality_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {}
        rows.append({
            "name": name,
            "mode": mode,
            "pcc": float(summary["pcc_mean"]),
            "rmse": float(summary["rmse_mean"]),
            "genes": int(payload["n_evaluated_genes"]),
            "validation_gate": gate.get("passed"),
        })
    if len({row["genes"] for row in rows}) != 1:
        raise RuntimeError(f"runs evaluated different gene widths: {rows}")
    print(f"{'RUN':58s} {'MODE':12s} {'PCC':>9s} {'RMSE':>9s} {'VAL_GATE':>10s}")
    for row in rows:
        print(
            f"{row['name']:58s} {row['mode']:12s} "
            f"{row['pcc']:9.4f} {row['rmse']:9.4f} {str(row['validation_gate']):>10s}"
        )
    anchor = rows[0]
    full_novae = rows[2]
    promoted = (
        full_novae["validation_gate"] is True
        and full_novae["pcc"] > anchor["pcc"]
        and full_novae["rmse"] < anchor["rmse"]
    )
    print(
        "DECISION: "
        + ("PROMOTE simple local + Novae to replicated seeds/folds"
           if promoted else "DO NOT PROMOTE yet; it did not clear validation plus held-out harmonic")
    )


if __name__ == "__main__":
    main()
