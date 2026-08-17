#!/usr/bin/env python3
"""Measure whether per-gene MK performance is reproducible across seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from gen3_multiscale.evaluation.per_gene_diagnostics import load_per_gene_diagnostics
from gen3_multiscale.scripts.summarize_mk_seed_replications import (
    ARCHITECTURES,
    discover_records,
)


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(np.where(finite, values, 0).sum(axis=0), count, out=result, where=count > 0)
    return result


def _patient_macro(values: np.ndarray, patient_ids: list[str]) -> np.ndarray:
    return _nanmean_axis0(np.stack([
        _nanmean_axis0(values[np.asarray([value == patient for value in patient_ids])])
        for patient in sorted(set(patient_ids))
    ]))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan"), int(finite.sum())
    x, y = rankdata(left[finite]), rankdata(right[finite])
    return float(np.corrcoef(x, y)[0, 1]), int(finite.sum())


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    records: dict[str, dict[int, tuple[Path, dict[str, Any]]]], *,
    output_dir: str | Path,
) -> dict[str, Path]:
    payloads: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARCHITECTURES}
    identity = None
    target_reference = None
    for architecture in ARCHITECTURES:
        if set(records.get(architecture, {})) != {0, 1, 2}:
            raise ValueError(f"{architecture}: exactly seeds 0,1,2 are required")
        for seed, (_report_path, report) in records[architecture].items():
            whole = report.get("whole_slide_structured_field_evaluation") or {}
            sidecar_path = Path(str(whole.get("per_gene_diagnostics_path") or ""))
            if not sidecar_path.is_file():
                raise FileNotFoundError(f"missing per-gene sidecar: {sidecar_path}")
            payload = load_per_gene_diagnostics(sidecar_path)
            current_identity = (
                payload["gene_names"], payload["sample_ids"],
                payload["patient_ids"], payload["organs"],
            )
            if identity is None:
                identity = current_identity
                target_reference = payload
            elif current_identity != identity:
                raise ValueError(f"sidecar identity differs: {sidecar_path}")
            for name in (
                "target_mean", "target_std", "target_nonzero_fraction",
                "target_moran_i", "target_local_gradient_energy",
            ):
                if not np.allclose(
                    payload[name], target_reference[name], rtol=1e-5, atol=1e-6,
                    equal_nan=True,
                ):
                    raise ValueError(f"held-out target attribute {name} differs: {sidecar_path}")
            payloads[architecture][seed] = payload

    assert identity is not None
    gene_names, _samples, patient_ids, _organs = identity
    metrics = ("pcc", "spearman", "r2", "rmse", "mae", "local_gradient_pcc")
    macro: dict[str, dict[int, dict[str, np.ndarray]]] = {arm: {} for arm in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for seed, payload in payloads[architecture].items():
            macro[architecture][seed] = {
                metric: _patient_macro(payload[metric], patient_ids) for metric in metrics
            }

    gene_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for left_seed in (0, 1, 2):
            for right_seed in range(left_seed + 1, 3):
                rho, n = _rank_correlation(
                    macro[architecture][left_seed]["pcc"],
                    macro[architecture][right_seed]["pcc"],
                )
                rank_rows.append({
                    "architecture": architecture, "metric": "pcc",
                    "left_seed": left_seed, "right_seed": right_seed,
                    "spearman_across_genes": rho, "n_genes": n,
                })
        pcc_matrix = np.stack([macro[architecture][seed]["pcc"] for seed in (0, 1, 2)])
        finite_sd = np.nanstd(pcc_matrix, axis=0, ddof=1)
        finite_mean = np.nanmean(pcc_matrix, axis=0)
        for index, gene in enumerate(gene_names):
            row: dict[str, Any] = {
                "architecture": architecture, "gene": gene,
                "pcc_mean": float(finite_mean[index]),
                "pcc_seed_sd": float(finite_sd[index]),
                "pcc_min": float(np.nanmin(pcc_matrix[:, index])),
                "pcc_max": float(np.nanmax(pcc_matrix[:, index])),
                "fraction_seeds_pcc_positive": float(np.mean(pcc_matrix[:, index] > 0)),
                "all_seeds_pcc_positive": bool(np.all(pcc_matrix[:, index] > 0)),
            }
            for seed in (0, 1, 2):
                row[f"seed{seed}_pcc"] = float(macro[architecture][seed]["pcc"][index])
            for metric in metrics[1:]:
                values = np.asarray([
                    macro[architecture][seed][metric][index] for seed in (0, 1, 2)
                ], dtype=np.float64)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_seed_sd"] = float(np.nanstd(values, ddof=1))
            gene_rows.append(row)
        finite = np.isfinite(finite_mean) & np.isfinite(finite_sd)
        summary_rows.append({
            "architecture": architecture,
            "mean_gene_pcc": float(np.mean(finite_mean[finite])),
            "median_gene_pcc": float(np.median(finite_mean[finite])),
            "median_gene_pcc_seed_sd": float(np.median(finite_sd[finite])),
            "p90_gene_pcc_seed_sd": float(np.quantile(finite_sd[finite], 0.9)),
            "fraction_genes_all_seeds_positive": float(np.mean(np.all(pcc_matrix[:, finite] > 0, axis=0))),
            "fraction_genes_mean_pcc_gt_0_1": float(np.mean(finite_mean[finite] > 0.1)),
            "fraction_genes_mean_pcc_gt_0_2": float(np.mean(finite_mean[finite] > 0.2)),
            "n_genes": int(finite.sum()),
        })

    delta_rows: list[dict[str, Any]] = []
    left, right = ARCHITECTURES
    for index, gene in enumerate(gene_names):
        deltas = np.asarray([
            macro[left][seed]["pcc"][index] - macro[right][seed]["pcc"][index]
            for seed in (0, 1, 2)
        ], dtype=np.float64)
        delta_rows.append({
            "gene": gene, "left": left, "right": right,
            "mean_pcc_delta": float(np.nanmean(deltas)),
            "pcc_delta_seed_sd": float(np.nanstd(deltas, ddof=1)),
            "fraction_seeds_left_better": float(np.mean(deltas > 0)),
            "all_seeds_left_better": bool(np.all(deltas > 0)),
            **{f"seed{seed}_pcc_delta": float(deltas[seed]) for seed in (0, 1, 2)},
        })

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_gene": output / "per_gene_seed_robustness.tsv",
        "summary": output / "gene_robustness_summary.tsv",
        "rank_stability": output / "seed_pair_gene_rank_stability.tsv",
        "architecture_delta": output / "per_gene_architecture_delta.tsv",
    }
    for path, rows in (
        (paths["per_gene"], gene_rows), (paths["summary"], summary_rows),
        (paths["rank_stability"], rank_rows),
        (paths["architecture_delta"], delta_rows),
    ):
        _write(path, rows)
    manifest_path = output / "gene_robustness_manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "mk_seed_gene_robustness", "version": 1,
        "architectures": list(ARCHITECTURES), "seeds": [0, 1, 2],
        "outputs": {name: str(path) for name, path in paths.items()},
    }, indent=2, sort_keys=True))
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-evaluation-root", required=True)
    parser.add_argument("--replication-root", required=True)
    parser.add_argument("--replication-evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    records = discover_records(
        seed0_root=args.seed0_evaluation_root,
        replication_root=args.replication_root,
        replication_evaluation_root=args.replication_evaluation_root,
    )
    outputs = analyze(records, output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
