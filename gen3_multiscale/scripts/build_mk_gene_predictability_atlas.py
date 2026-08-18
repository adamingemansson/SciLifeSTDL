#!/usr/bin/env python3
"""Build an auditable gene-level predictability atlas from MK seed results.

This is a post-processing tool: it reads the TSV written by
``analyze_mk_seed_gene_robustness`` and never loads a model or held-out image.
It separates reproducible prediction, measurement ceiling, target biology and
architecture complementarity instead of hiding them in one aggregate PCC.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ATTRIBUTES = (
    "target_mean",
    "target_std",
    "target_nonzero_fraction",
    "target_moran_i",
    "target_local_gradient_energy",
    "noise_ceiling",
)


def _read(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty seed-level gene table: {source}")
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        value = float("nan")
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for finite one-dimensional values, without SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan"), int(finite.sum())
    x = _rankdata(left[finite])
    y = _rankdata(right[finite])
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), int(finite.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(finite.sum())


def _linear_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan"), int(finite.sum())
    x, y = left[finite], right[finite]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), int(finite.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(finite.sum())


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / scale


def _conditional_associations(
    architecture: str,
    *,
    pcc: np.ndarray,
    attributes: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Partial rank associations and standardized multivariable coefficients.

    These are descriptive diagnostics across genes. They separate correlated
    gene properties, but do not establish a biological causal effect.
    """
    matrix = np.column_stack([attributes[name] for name in ATTRIBUTES])
    finite = np.isfinite(pcc) & np.all(np.isfinite(matrix), axis=1)
    y_raw, x_raw = pcc[finite], matrix[finite]
    if y_raw.size < len(ATTRIBUTES) + 2:
        # Small synthetic fixtures can still exercise output production. The
        # resulting underdetermined coefficients are explicitly diagnostic.
        if y_raw.size < 3:
            raise ValueError("too few finite genes for conditional attribute analysis")
    y = _zscore(_rankdata(y_raw))
    x = np.column_stack([_zscore(_rankdata(x_raw[:, index])) for index in range(x_raw.shape[1])])
    design = np.column_stack([np.ones(y.size), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    condition_number = float(np.linalg.cond(x))
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(ATTRIBUTES):
        controls = np.delete(x, index, axis=1)
        controls = np.column_stack([np.ones(y.size), controls])
        y_residual = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
        x_target = x[:, index]
        x_residual = x_target - controls @ np.linalg.lstsq(
            controls, x_target, rcond=None,
        )[0]
        partial, _ = _linear_correlation(y_residual, x_residual)
        residual_ss = float(np.sum(np.square(x_residual)))
        total_ss = float(np.sum(np.square(x_target - np.mean(x_target))))
        r_squared = 1.0 - residual_ss / total_ss if total_ss > 1e-12 else float("nan")
        vif = 1.0 / (1.0 - r_squared) if math.isfinite(r_squared) and r_squared < 1.0 else float("inf")
        marginal, _ = _rank_correlation(y_raw, x_raw[:, index])
        rows.append({
            "architecture": architecture,
            "performance_metric": "pcc_mean",
            "gene_attribute": name,
            "marginal_spearman": marginal,
            "partial_spearman_controlling_other_attributes": partial,
            "standardized_multivariable_rank_beta": float(beta[index]),
            "variance_inflation_factor": vif,
            "design_condition_number": condition_number,
            "n_genes_complete_case": int(y.size),
        })
    return rows


def _seed_columns(rows: list[dict[str, str]]) -> list[str]:
    columns = [
        key for key in rows[0]
        if key.startswith("seed") and key.endswith("_pcc")
    ]
    if len(columns) < 2:
        raise ValueError("at least two explicit seed*_pcc columns are required")
    return sorted(columns, key=lambda value: int(value[4:-4]))


def _by_architecture(
    rows: list[dict[str, str]], architectures: tuple[str, str] | None,
) -> tuple[tuple[str, str], dict[str, dict[str, dict[str, str]]]]:
    available = sorted({str(row.get("architecture", "")) for row in rows})
    if "" in available:
        raise ValueError("every row must name an architecture")
    selected = tuple(available) if architectures is None else architectures
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("exactly two distinct architectures are required")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"requested architecture(s) missing: {missing}")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for architecture in selected:
        indexed: dict[str, dict[str, str]] = {}
        for row in rows:
            if row["architecture"] != architecture:
                continue
            gene = str(row.get("gene", ""))
            if not gene or gene in indexed:
                raise ValueError(f"{architecture}: empty or duplicate gene {gene!r}")
            indexed[gene] = row
        result[architecture] = indexed
    if set(result[selected[0]]) != set(result[selected[1]]):
        raise ValueError("architecture gene vocabularies differ")
    return (selected[0], selected[1]), result


def build_atlas(
    seed_gene_table: str | Path,
    *,
    output_dir: str | Path,
    architectures: tuple[str, str] | None = None,
    ceiling_threshold: float = 0.1,
    meaningful_delta: float = 0.01,
    top_n: int = 100,
) -> dict[str, Path]:
    if ceiling_threshold <= 0 or meaningful_delta <= 0 or top_n < 1:
        raise ValueError("thresholds and top_n must be positive")
    source_rows = _read(seed_gene_table)
    seed_columns = _seed_columns(source_rows)
    (left, right), indexed = _by_architecture(source_rows, architectures)
    genes = sorted(indexed[left])

    atlas_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    conditional_association_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    headroom_rows: list[dict[str, Any]] = []
    architecture_arrays: dict[str, np.ndarray] = {}
    for architecture in (left, right):
        pcc = np.asarray([_number(indexed[architecture][gene], "pcc_mean") for gene in genes])
        architecture_arrays[architecture] = pcc
        for gene in genes:
            row = indexed[architecture][gene]
            ceiling = _number(row, "noise_ceiling")
            pcc_mean = _number(row, "pcc_mean")
            seed_values = np.asarray([_number(row, key) for key in seed_columns])
            finite_seeds = seed_values[np.isfinite(seed_values)]
            ceiling_eligible = math.isfinite(ceiling) and ceiling >= ceiling_threshold
            ratio = pcc_mean / ceiling if ceiling_eligible and math.isfinite(pcc_mean) else float("nan")
            atlas_rows.append({
                "architecture": architecture,
                "gene": gene,
                "pcc_mean": pcc_mean,
                "pcc_seed_sd": _number(row, "pcc_seed_sd"),
                "pcc_worst_seed": float(np.min(finite_seeds)) if finite_seeds.size else float("nan"),
                "pcc_best_seed": float(np.max(finite_seeds)) if finite_seeds.size else float("nan"),
                "all_seeds_positive": bool(finite_seeds.size and np.all(finite_seeds > 0)),
                "all_seeds_pcc_ge_0_1": bool(finite_seeds.size and np.all(finite_seeds >= 0.1)),
                "all_seeds_pcc_ge_0_2": bool(finite_seeds.size and np.all(finite_seeds >= 0.2)),
                "noise_ceiling": ceiling,
                "ceiling_eligible": ceiling_eligible,
                # Descriptive and intentionally not clipped: count-split ceilings
                # are estimates, so ratios can exceed one.
                "pcc_over_noise_ceiling": ratio,
                **{name: _number(row, name) for name in ATTRIBUTES[:-1]},
                "spearman_mean": _number(row, "spearman_mean"),
                "r2_mean": _number(row, "r2_mean"),
                "rmse_mean": _number(row, "rmse_mean"),
                "mae_mean": _number(row, "mae_mean"),
                "local_gradient_pcc_mean": _number(row, "local_gradient_pcc_mean"),
            })
        for attribute in ATTRIBUTES:
            values = np.asarray([_number(indexed[architecture][gene], attribute) for gene in genes])
            rho, n = _rank_correlation(pcc, values)
            association_rows.append({
                "architecture": architecture,
                "performance_metric": "pcc_mean",
                "gene_attribute": attribute,
                "spearman_rho": rho,
                "n_genes": n,
            })
        conditional_association_rows.extend(_conditional_associations(
            architecture,
            pcc=pcc,
            attributes={
                name: np.asarray([
                    _number(indexed[architecture][gene], name) for gene in genes
                ])
                for name in ATTRIBUTES
            },
        ))

        ranked = sorted(
            (row for row in atlas_rows if row["architecture"] == architecture),
            key=lambda row: (
                -float(row["pcc_worst_seed"]) if math.isfinite(float(row["pcc_worst_seed"])) else math.inf,
                -float(row["pcc_mean"]) if math.isfinite(float(row["pcc_mean"])) else math.inf,
                str(row["gene"]),
            ),
        )
        for rank, row in enumerate(ranked[:top_n], 1):
            robust_rows.append({"architecture": architecture, "rank": rank, **row})

        headroom = sorted(
            (
                {
                    "architecture": architecture,
                    "gene": row["gene"],
                    "pcc_mean": row["pcc_mean"],
                    "pcc_worst_seed": row["pcc_worst_seed"],
                    "pcc_seed_sd": row["pcc_seed_sd"],
                    "noise_ceiling": row["noise_ceiling"],
                    "ceiling_minus_pcc": float(row["noise_ceiling"]) - float(row["pcc_mean"]),
                    "pcc_over_noise_ceiling": row["pcc_over_noise_ceiling"],
                    "all_seeds_positive": row["all_seeds_positive"],
                    "target_mean": row["target_mean"],
                    "target_std": row["target_std"],
                    "target_nonzero_fraction": row["target_nonzero_fraction"],
                    "target_moran_i": row["target_moran_i"],
                    "target_local_gradient_energy": row["target_local_gradient_energy"],
                    "local_gradient_pcc_mean": row["local_gradient_pcc_mean"],
                }
                for row in atlas_rows
                if row["architecture"] == architecture
                and bool(row["ceiling_eligible"])
                and math.isfinite(float(row["pcc_mean"]))
            ),
            key=lambda row: (-float(row["ceiling_minus_pcc"]), str(row["gene"])),
        )
        for rank, row in enumerate(headroom[:top_n], 1):
            headroom_rows.append({"rank": rank, **row})

    comparison_rows: list[dict[str, Any]] = []
    left_wins = right_wins = stable_left = stable_right = 0
    for gene in genes:
        lrow, rrow = indexed[left][gene], indexed[right][gene]
        lseed = np.asarray([_number(lrow, key) for key in seed_columns])
        rseed = np.asarray([_number(rrow, key) for key in seed_columns])
        deltas = lseed - rseed
        finite = np.isfinite(deltas)
        delta = _number(lrow, "pcc_mean") - _number(rrow, "pcc_mean")
        if math.isfinite(delta) and delta > 0:
            left_wins += 1
        elif math.isfinite(delta) and delta < 0:
            right_wins += 1
        stable_l = bool(finite.any() and np.all(deltas[finite] >= meaningful_delta))
        stable_r = bool(finite.any() and np.all(deltas[finite] <= -meaningful_delta))
        stable_left += int(stable_l)
        stable_right += int(stable_r)
        lpcc, rpcc = _number(lrow, "pcc_mean"), _number(rrow, "pcc_mean")
        comparison_rows.append({
            "gene": gene,
            "left": left,
            "right": right,
            "left_pcc_mean": lpcc,
            "right_pcc_mean": rpcc,
            "left_minus_right_pcc": delta,
            "oracle_best_pcc": max(lpcc, rpcc),
            "stable_left_advantage": stable_l,
            "stable_right_advantage": stable_r,
            "meaningful_delta_threshold": meaningful_delta,
            **{f"{key}_delta": float(value) for key, value in zip(seed_columns, deltas)},
        })

    left_values = architecture_arrays[left]
    right_values = architecture_arrays[right]
    pearson, n_paired = _linear_correlation(left_values, right_values)
    spearman, _ = _rank_correlation(left_values, right_values)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    oracle = np.maximum(left_values[finite], right_values[finite])
    best_single_mean = max(float(np.mean(left_values[finite])), float(np.mean(right_values[finite])))
    summary_rows: list[dict[str, Any]] = [{
        "left": left,
        "right": right,
        "n_genes": n_paired,
        "pearson_gene_pcc": pearson,
        "spearman_gene_pcc": spearman,
        "fraction_genes_left_better": left_wins / n_paired,
        "fraction_genes_right_better": right_wins / n_paired,
        "fraction_genes_stable_left_advantage": stable_left / n_paired,
        "fraction_genes_stable_right_advantage": stable_right / n_paired,
        "mean_gene_pcc_left": float(np.mean(left_values[finite])),
        "mean_gene_pcc_right": float(np.mean(right_values[finite])),
        "per_gene_oracle_mean_pcc": float(np.mean(oracle)),
        "oracle_gain_over_best_single": float(np.mean(oracle)) - best_single_mean,
        "meaningful_delta_threshold": meaningful_delta,
    }]
    for top_k in (50, 200, 1000):
        k = min(top_k, n_paired)
        left_top = set(np.asarray(genes)[finite][np.argsort(left_values[finite])[-k:]])
        right_top = set(np.asarray(genes)[finite][np.argsort(right_values[finite])[-k:]])
        summary_rows[0][f"top{k}_overlap"] = len(left_top & right_top)
        summary_rows[0][f"top{k}_jaccard"] = len(left_top & right_top) / len(left_top | right_top)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "atlas": output / "gene_predictability_atlas.tsv",
        "robust_genes": output / "top_robust_genes.tsv",
        "associations": output / "performance_attribute_associations.tsv",
        "conditional_associations": output / "conditional_attribute_associations.tsv",
        "high_headroom_genes": output / "high_headroom_genes.tsv",
        "architecture_comparison": output / "per_gene_architecture_comparison.tsv",
        "complementarity": output / "architecture_complementarity_summary.tsv",
    }
    for key, rows in (
        ("atlas", atlas_rows),
        ("robust_genes", robust_rows),
        ("associations", association_rows),
        ("conditional_associations", conditional_association_rows),
        ("high_headroom_genes", headroom_rows),
        ("architecture_comparison", comparison_rows),
        ("complementarity", summary_rows),
    ):
        _write(outputs[key], rows)
    manifest = output / "predictability_atlas_manifest.json"
    manifest.write_text(json.dumps({
        "kind": "mk_gene_predictability_atlas",
        "version": 1,
        "source": str(Path(seed_gene_table).expanduser().resolve()),
        "architectures": [left, right],
        "seed_columns": seed_columns,
        "ceiling_threshold": ceiling_threshold,
        "meaningful_delta": meaningful_delta,
        "top_n": top_n,
        "notes": {
            "pcc_over_noise_ceiling": "descriptive, unbounded, and emitted only when ceiling meets threshold",
            "oracle_gain": "diagnostic upper bound from choosing the better architecture separately for each held-out gene",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }, indent=2, sort_keys=True))
    outputs["manifest"] = manifest
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-gene-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architectures", nargs=2)
    parser.add_argument("--ceiling-threshold", type=float, default=0.1)
    parser.add_argument("--meaningful-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    outputs = build_atlas(
        args.seed_gene_table,
        output_dir=args.output_dir,
        architectures=tuple(args.architectures) if args.architectures else None,
        ceiling_threshold=args.ceiling_threshold,
        meaningful_delta=args.meaningful_delta,
        top_n=args.top_n,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
