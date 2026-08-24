#!/usr/bin/env python3
"""Compare WAE latent-draw diversity with deterministic seed stability.

The two quantities are deliberately kept separate: WAE predictive standard
deviation is within-model output dispersion, while deterministic seed SD is
variation in held-out performance across independently trained models.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


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


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _rho(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.std(left[finite]) == 0 or np.std(right[finite]) == 0:
        return float("nan"), int(finite.sum())
    return float(spearmanr(left[finite], right[finite]).statistic), int(finite.sum())


def compare(
    *, deterministic_path: str, wae_inputs: list[str], output_dir: str,
    deterministic_architecture: str = "mk_wb_parallel_gated",
) -> dict[str, Path]:
    deterministic_rows = [
        row for row in _read(Path(deterministic_path).expanduser().resolve())
        if row.get("architecture") == deterministic_architecture
    ]
    deterministic = {row["gene"]: row for row in deterministic_rows}
    if len(deterministic) < 100:
        raise ValueError(
            f"too few deterministic genes for {deterministic_architecture}: {len(deterministic)}"
        )
    gene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    for specification in wae_inputs:
        if "=" not in specification:
            raise ValueError("--wae must be ARM=/path/to/per_gene_per_slide.tsv")
        arm, raw_path = specification.split("=", 1)
        rows = _read(Path(raw_path).expanduser().resolve())
        by_gene: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_gene.setdefault(row["gene"], []).append(row)
        if set(by_gene) != set(deterministic):
            raise ValueError(
                f"{arm}: WAE and deterministic gene identities differ "
                f"({len(by_gene)} vs {len(deterministic)})"
            )
        arm_rows = []
        for gene in sorted(by_gene):
            values = by_gene[gene]
            target_std = _finite_mean([float(row["target_std"]) for row in values])
            predictive_std = _finite_mean([float(row["predictive_std_mean"]) for row in values])
            point_rmse = _finite_mean([float(row["point_rmse"]) for row in values])
            mean_rmse = _finite_mean([float(row["predictive_mean_rmse"]) for row in values])
            row = {
                "wae_arm": arm,
                "deterministic_architecture": deterministic_architecture,
                "gene": gene,
                "wae_predictive_std_mean": predictive_std,
                "wae_predictive_std_to_target_std": (
                    predictive_std / target_std if target_std > 1e-12 else float("nan")
                ),
                "wae_point_rmse": point_rmse,
                "wae_predictive_mean_rmse": mean_rmse,
                "wae_point_minus_mean_rmse": point_rmse - mean_rmse,
                "deterministic_pcc_mean": float(deterministic[gene]["pcc_mean"]),
                "deterministic_pcc_seed_sd": float(deterministic[gene]["pcc_seed_sd"]),
            }
            arm_rows.append(row)
            gene_rows.append(row)
        arrays = {
            name: np.asarray([row[name] for row in arm_rows], dtype=np.float64)
            for name in arm_rows[0] if name not in {"wae_arm", "deterministic_architecture", "gene"}
        }
        summary_rows.append({
            "wae_arm": arm,
            "n_genes": len(arm_rows),
            "median_wae_predictive_std_to_target_std": float(np.nanmedian(
                arrays["wae_predictive_std_to_target_std"]
            )),
            "median_deterministic_pcc_seed_sd": float(np.nanmedian(
                arrays["deterministic_pcc_seed_sd"]
            )),
            "mean_wae_point_minus_mean_rmse": float(np.nanmean(
                arrays["wae_point_minus_mean_rmse"]
            )),
        })
        for outcome in (
            "wae_point_rmse", "wae_predictive_mean_rmse", "deterministic_pcc_mean",
            "deterministic_pcc_seed_sd",
        ):
            rho, n = _rho(arrays["wae_predictive_std_to_target_std"], arrays[outcome])
            association_rows.append({
                "wae_arm": arm,
                "left_quantity": "wae_predictive_std_to_target_std",
                "right_quantity": outcome,
                "spearman_rho": rho,
                "n_genes": n,
                "interpretation_scope": (
                    "within_model_output_dispersion_vs_held_out_error"
                    if outcome.startswith("wae_")
                    else "within_model_dispersion_vs_across_training_seed_performance"
                ),
            })
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    gene_path = root / "wae_vs_seed_variability_per_gene.tsv"
    summary_path = root / "wae_vs_seed_variability_summary.tsv"
    association_path = root / "wae_vs_seed_variability_associations.tsv"
    _write(gene_path, gene_rows)
    _write(summary_path, summary_rows)
    _write(association_path, association_rows)
    manifest_path = root / "variability_comparison_manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "mk_wae_vs_deterministic_seed_variability",
        "deterministic_architecture": deterministic_architecture,
        "warning": (
            "WAE predictive standard deviation and deterministic seed performance SD "
            "are different quantities and are not interchangeable uncertainty estimates."
        ),
        "outputs": [str(gene_path), str(summary_path), str(association_path)],
    }, indent=2) + "\n")
    return {"per_gene": gene_path, "summary": summary_path, "associations": association_path, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-per-gene", required=True)
    parser.add_argument("--deterministic-architecture", default="mk_wb_parallel_gated")
    parser.add_argument("--wae", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = compare(
        deterministic_path=args.deterministic_per_gene,
        wae_inputs=args.wae,
        output_dir=args.output_dir,
        deterministic_architecture=args.deterministic_architecture,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
