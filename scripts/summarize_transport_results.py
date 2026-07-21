#!/usr/bin/env python3
"""Summarize identity-preserving transport on the held-out sample."""
import json
from pathlib import Path


RUNS = [
    ("harmonic_external", "missing_tissue_simple_local_harmonic_k32", True),
    ("uniform_local_k32", "missing_tissue_transport_uniform_k32", False),
    ("learned_geometry", "missing_tissue_transport_geometry_k32_seed10", False),
    ("transport_gex_novae", "missing_tissue_transport_gex_novae_k32_seed10", False),
    ("transport_full_mome", "missing_tissue_transport_full_mome_k32_seed10", False),
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
            "local_gate": gate.get("passed"),
            "genes": int(payload["n_evaluated_genes"]),
        })

    if len({row["genes"] for row in rows}) != 1:
        raise RuntimeError(f"runs evaluated different gene panels: {rows}")
    print(f"{'ARCHITECTURE':25s} {'MODE':12s} {'PCC':>9s} {'RMSE':>9s} {'LOCAL_GATE':>11s}")
    for row in rows:
        print(
            f"{row['label']:25s} {row['mode']:12s} {row['pcc']:9.4f} "
            f"{row['rmse']:9.4f} {str(row['local_gate']):>11s}"
        )

    harmonic = next((row for row in rows if row["label"] == "harmonic_external"), None)
    uniform = next(row for row in rows if row["label"] == "uniform_local_k32")
    candidates = [row for row in rows if row["label"].startswith("transport_")]
    promoted = bool(harmonic) and any(
        row["local_gate"] is True
        and row["pcc"] > max(harmonic["pcc"], uniform["pcc"])
        and row["rmse"] < min(harmonic["rmse"], uniform["rmse"])
        for row in candidates
    )
    print("DECISION: " + (
        "PROMOTE the passing conditioned transport model to folds/seeds"
        if promoted
        else "DO NOT PROMOTE; no conditioned transport model beat both uniform transport and held-out harmonic"
    ))


if __name__ == "__main__":
    main()
