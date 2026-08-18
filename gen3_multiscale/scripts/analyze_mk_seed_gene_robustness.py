#!/usr/bin/env python3
"""Measure whether per-gene MK performance is reproducible across seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from gen3_multiscale.evaluation.per_gene_diagnostics import load_per_gene_diagnostics
from gen3_multiscale.scripts.summarize_mk_seed_replications import (
    ARCHITECTURES,
    discover_records,
)


TARGET_ATTRIBUTES = (
    "target_mean", "target_std", "target_nonzero_fraction",
    "target_moran_i", "target_local_gradient_energy", "noise_ceiling",
)


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(np.where(finite, values, 0).sum(axis=0), count, out=result, where=count > 0)
    return result


def _finite_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"),) * 4
    return (
        float(finite.mean()),
        float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        float(finite.min()),
        float(finite.max()),
    )


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


def _noise_ceiling(
    path: str | Path | None, *, sample_ids: list[str], gene_names: list[str],
) -> np.ndarray | None:
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if payload.get("kind") != "gene_noise_ceiling_by_count_splitting":
        raise ValueError(f"not a count-split noise-ceiling artifact: {source}")
    by_sample = {str(row["sample_id"]): row for row in payload.get("per_slide", [])}
    if set(by_sample) != set(sample_ids):
        raise ValueError("noise-ceiling and per-gene diagnostic cohorts differ")
    result = np.full((len(sample_ids), len(gene_names)), np.nan, dtype=np.float64)
    for row, sample_id in enumerate(sample_ids):
        values = by_sample[sample_id].get("ceiling_by_gene") or {}
        result[row] = [float(values.get(gene, np.nan)) for gene in gene_names]
    return result


def analyze(
    records: dict[str, dict[int, tuple[Path, dict[str, Any]]]], *,
    output_dir: str | Path, noise_ceiling_path: str | Path | None = None,
) -> dict[str, Path]:
    payloads: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARCHITECTURES}
    seed_sets = {architecture: set(records.get(architecture, {})) for architecture in ARCHITECTURES}
    seeds = tuple(sorted(seed_sets[ARCHITECTURES[0]]))
    if len(seeds) != 3:
        raise ValueError(f"exactly three independent seeds are required; got={list(seeds)}")
    identity = None
    target_reference = None
    for architecture in ARCHITECTURES:
        if seed_sets[architecture] != set(seeds):
            raise ValueError(
                f"{architecture}: seed set differs; expected={list(seeds)}, "
                f"got={sorted(seed_sets[architecture])}"
            )
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
    gene_names, sample_ids, patient_ids, _organs = identity
    target_macro = {
        name: _patient_macro(target_reference[name], patient_ids)
        for name in TARGET_ATTRIBUTES if name != "noise_ceiling"
    }
    ceiling = _noise_ceiling(
        noise_ceiling_path, sample_ids=sample_ids, gene_names=gene_names,
    )
    target_macro["noise_ceiling"] = (
        _patient_macro(ceiling, patient_ids)
        if ceiling is not None else np.full(len(gene_names), np.nan, dtype=np.float64)
    )
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
        for left_seed, right_seed in combinations(seeds, 2):
                rho, n = _rank_correlation(
                    macro[architecture][left_seed]["pcc"],
                    macro[architecture][right_seed]["pcc"],
                )
                rank_rows.append({
                    "architecture": architecture, "metric": "pcc",
                    "left_seed": left_seed, "right_seed": right_seed,
                    "spearman_across_genes": rho, "n_genes": n,
                })
        pcc_matrix = np.stack([macro[architecture][seed]["pcc"] for seed in seeds])
        finite_mean = _nanmean_axis0(pcc_matrix)
        finite_sd = np.full(pcc_matrix.shape[1], np.nan, dtype=np.float64)
        finite_count = np.isfinite(pcc_matrix).sum(axis=0)
        for index in np.flatnonzero(finite_count > 1):
            finite_sd[index] = np.std(
                pcc_matrix[np.isfinite(pcc_matrix[:, index]), index], ddof=1,
            )
        for index, gene in enumerate(gene_names):
            pcc_mean, pcc_sd, pcc_min, pcc_max = _finite_summary(pcc_matrix[:, index])
            row: dict[str, Any] = {
                "architecture": architecture, "gene": gene,
                "pcc_mean": pcc_mean,
                "pcc_seed_sd": pcc_sd,
                "pcc_min": pcc_min,
                "pcc_max": pcc_max,
                "fraction_seeds_pcc_positive": float(np.mean(pcc_matrix[:, index] > 0)),
                "all_seeds_pcc_positive": bool(np.all(pcc_matrix[:, index] > 0)),
                **{
                    name: float(values[index])
                    for name, values in target_macro.items()
                },
            }
            for seed in seeds:
                row[f"seed{seed}_pcc"] = float(macro[architecture][seed]["pcc"][index])
            for metric in metrics[1:]:
                values = np.asarray([
                    macro[architecture][seed][metric][index] for seed in seeds
                ], dtype=np.float64)
                metric_mean, metric_sd, _metric_min, _metric_max = _finite_summary(values)
                row[f"{metric}_mean"] = metric_mean
                row[f"{metric}_seed_sd"] = metric_sd
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
    association_rows: list[dict[str, Any]] = []
    left, right = ARCHITECTURES
    mean_deltas = np.full(len(gene_names), np.nan, dtype=np.float64)
    for index, gene in enumerate(gene_names):
        delta_by_seed = {
            seed: float(
                macro[left][seed]["pcc"][index] - macro[right][seed]["pcc"][index]
            )
            for seed in seeds
        }
        deltas = np.asarray([delta_by_seed[seed] for seed in seeds], dtype=np.float64)
        delta_mean, delta_sd, _delta_min, _delta_max = _finite_summary(deltas)
        mean_deltas[index] = delta_mean
        delta_rows.append({
            "gene": gene, "left": left, "right": right,
            "mean_pcc_delta": mean_deltas[index],
            "pcc_delta_seed_sd": delta_sd,
            "fraction_seeds_left_better": float(np.mean(deltas > 0)),
            "all_seeds_left_better": bool(np.all(deltas > 0)),
            **{f"seed{seed}_pcc_delta": delta_by_seed[seed] for seed in seeds},
            **{
                name: float(values[index])
                for name, values in target_macro.items()
            },
        })
    for name, values in target_macro.items():
        rho, n = _rank_correlation(mean_deltas, values)
        association_rows.append({
            "left": left, "right": right, "performance_metric": "mean_pcc_delta",
            "gene_attribute": name, "spearman_rho": rho, "n_genes": n,
        })

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_gene": output / "per_gene_seed_robustness.tsv",
        "summary": output / "gene_robustness_summary.tsv",
        "rank_stability": output / "seed_pair_gene_rank_stability.tsv",
        "architecture_delta": output / "per_gene_architecture_delta.tsv",
        "delta_attribute_associations": output / "architecture_delta_attribute_associations.tsv",
    }
    for path, rows in (
        (paths["per_gene"], gene_rows), (paths["summary"], summary_rows),
        (paths["rank_stability"], rank_rows),
        (paths["architecture_delta"], delta_rows),
        (paths["delta_attribute_associations"], association_rows),
    ):
        _write(path, rows)
    manifest_path = output / "gene_robustness_manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "mk_seed_gene_robustness", "version": 1,
        "architectures": list(ARCHITECTURES), "seeds": list(seeds),
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
    parser.add_argument("--noise-ceiling")
    args = parser.parse_args()
    records = discover_records(
        seed0_root=args.seed0_evaluation_root,
        replication_root=args.replication_root,
        replication_evaluation_root=args.replication_evaluation_root,
    )
    outputs = analyze(
        records, output_dir=args.output_dir,
        noise_ceiling_path=args.noise_ceiling,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
