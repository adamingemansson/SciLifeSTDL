#!/usr/bin/env python3
"""Explain which genes an H&E-to-ST model predicts and which properties matter.

Consumes compressed sidecars written during exact whole-slide evaluation.  It
produces (1) a long model-by-gene table, (2) correlations between performance
and measurable gene properties, and (3) auditable top/bottom gene lists.  All
aggregation is slide-within-patient followed by a patient macro-average.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from gen3_multiscale.evaluation.per_gene_diagnostics import (
    load_per_gene_diagnostics,
)


PERFORMANCE = (
    "pcc", "spearman", "r2", "rmse", "mae", "local_gradient_pcc",
    "local_gradient_sign_agreement",
)
ATTRIBUTES = (
    "target_mean", "target_std", "target_nonzero_fraction", "target_moran_i",
    "target_local_gradient_energy", "noise_ceiling",
)


def _patient_macro(values: np.ndarray, patient_ids: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(patient_ids):
        raise ValueError("patient-macro values must be [n_slides, n_genes]")
    patients = sorted(set(map(str, patient_ids)))
    per_patient = []
    for patient in patients:
        rows = np.asarray([value == patient for value in patient_ids], dtype=bool)
        per_patient.append(_nanmean_axis0(values[rows]))
    return _nanmean_axis0(np.stack(per_patient, axis=0))


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    """NaN-aware column mean without warnings for entirely missing genes."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(
        np.where(finite, values, 0.0).sum(axis=0), counts,
        out=result, where=counts > 0,
    )
    return result


def _assert_target_attributes_identical(
    reference: dict[str, Any], candidate: dict[str, Any], *, path: Path,
) -> None:
    """Fail closed if two methods were not scored against identical targets."""
    for name in (
        "target_mean", "target_std", "target_nonzero_fraction",
        "target_moran_i", "target_local_gradient_energy",
    ):
        left = np.asarray(reference[name], dtype=np.float64)
        right = np.asarray(candidate[name], dtype=np.float64)
        if not np.allclose(left, right, rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(
                f"sidecar target attribute {name!r} differs: {path}; "
                "reports are not an exact paired comparison"
            )


def _correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan"), int(finite.sum())
    left_rank = rankdata(left[finite], method="average")
    right_rank = rankdata(right[finite], method="average")
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return float("nan"), int(finite.sum())
    return float(np.corrcoef(left_rank, right_rank)[0, 1]), int(finite.sum())


def _method(payload: dict[str, Any], path: Path) -> str:
    provenance = payload["metadata"].get("provenance") or {}
    return str(provenance.get("method") or provenance.get("arm") or path.stem)


def _noise_ceiling(
    path: str | None, *, sample_ids: list[str], gene_names: list[str],
) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text())
    if payload.get("kind") != "gene_noise_ceiling_by_count_splitting":
        raise ValueError("--noise-ceiling is not a count-split noise-ceiling report")
    by_sample = {str(row["sample_id"]): row for row in payload.get("per_slide", [])}
    if set(by_sample) != set(sample_ids):
        raise ValueError("noise-ceiling and diagnostics slide cohorts differ")
    result = np.full((len(sample_ids), len(gene_names)), np.nan, dtype=np.float64)
    for row, sample_id in enumerate(sample_ids):
        values = by_sample[sample_id].get("ceiling_by_gene") or {}
        result[row] = [float(values.get(gene, np.nan)) for gene in gene_names]
    return result


def _write(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    sidecars: list[str | Path], *, output_dir: str | Path,
    noise_ceiling_path: str | None = None, top_n: int = 50,
) -> dict[str, Path]:
    paths = [Path(value).expanduser().resolve() for value in sidecars]
    if not paths or top_n < 1:
        raise ValueError("at least one sidecar and a positive top_n are required")
    loaded = [(path, load_per_gene_diagnostics(path)) for path in paths]
    reference = loaded[0][1]
    identity = (
        reference["gene_names"], reference["sample_ids"], reference["patient_ids"],
        reference["organs"],
    )
    for path, payload in loaded[1:]:
        actual = (
            payload["gene_names"], payload["sample_ids"], payload["patient_ids"],
            payload["organs"],
        )
        if actual != identity:
            raise ValueError(f"sidecar cohort/gene identity differs: {path}")
        _assert_target_attributes_identical(reference, payload, path=path)

    gene_names, sample_ids, patient_ids, _organs = identity
    ceiling_by_slide = _noise_ceiling(
        noise_ceiling_path, sample_ids=sample_ids, gene_names=gene_names,
    )
    macro_ceiling = (
        _patient_macro(ceiling_by_slide, patient_ids) if ceiling_by_slide is not None else None
    )
    long_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    model_summary_rows: list[dict[str, Any]] = []

    for path, payload in loaded:
        method = _method(payload, path)
        macro = {
            name: _patient_macro(payload[name], patient_ids)
            for name in (
                *PERFORMANCE, "target_mean", "target_std", "target_nonzero_fraction",
                "prediction_mean", "prediction_std", "target_moran_i",
                "prediction_moran_i", "target_local_gradient_energy",
                "prediction_local_gradient_energy",
            )
        }
        if macro_ceiling is not None:
            macro["noise_ceiling"] = macro_ceiling
            eligible = np.isfinite(macro["pcc"]) & np.isfinite(macro_ceiling) & (macro_ceiling >= 0.05)
            fraction_ceiling = np.full(len(gene_names), np.nan, dtype=np.float64)
            fraction_ceiling[eligible] = macro["pcc"][eligible] / macro_ceiling[eligible]
            macro["fraction_noise_ceiling"] = fraction_ceiling
        else:
            macro["noise_ceiling"] = np.full(len(gene_names), np.nan)
            macro["fraction_noise_ceiling"] = np.full(len(gene_names), np.nan)

        for index, gene in enumerate(gene_names):
            long_rows.append({
                "method": method, "gene": gene,
                **{name: float(values[index]) for name, values in macro.items()},
            })
        for performance in PERFORMANCE:
            for attribute in ATTRIBUTES:
                rho, n_genes = _correlation(macro[performance], macro[attribute])
                association_rows.append({
                    "method": method, "performance_metric": performance,
                    "gene_attribute": attribute, "spearman_rho": rho,
                    "n_genes": n_genes,
                })

        finite_pcc = np.flatnonzero(np.isfinite(macro["pcc"]))
        order = finite_pcc[np.argsort(macro["pcc"][finite_pcc])]
        selections = {
            "bottom": order[:top_n],
            "top": order[-top_n:][::-1],
        }
        for group, indices in selections.items():
            for rank, index in enumerate(indices, 1):
                extreme_rows.append({
                    "method": method, "group": group, "rank": rank,
                    "gene": gene_names[index], "pcc": float(macro["pcc"][index]),
                    "rmse": float(macro["rmse"][index]),
                    "target_mean": float(macro["target_mean"][index]),
                    "target_std": float(macro["target_std"][index]),
                    "target_nonzero_fraction": float(macro["target_nonzero_fraction"][index]),
                    "target_moran_i": float(macro["target_moran_i"][index]),
                    "target_local_gradient_energy": float(
                        macro["target_local_gradient_energy"][index]
                    ),
                    "noise_ceiling": float(macro["noise_ceiling"][index]),
                })
        pcc = macro["pcc"][np.isfinite(macro["pcc"])]
        model_summary_rows.append({
            "method": method,
            "mean_gene_pcc": float(np.mean(pcc)) if pcc.size else float("nan"),
            "median_gene_pcc": float(np.median(pcc)) if pcc.size else float("nan"),
            "fraction_gene_pcc_gt_0": float(np.mean(pcc > 0)) if pcc.size else float("nan"),
            "fraction_gene_pcc_gt_0_1": float(np.mean(pcc > 0.1)) if pcc.size else float("nan"),
            "fraction_gene_pcc_gt_0_2": float(np.mean(pcc > 0.2)) if pcc.size else float("nan"),
            "fraction_gene_pcc_gt_0_3": float(np.mean(pcc > 0.3)) if pcc.size else float("nan"),
            "n_scored_genes": int(pcc.size),
            "source": str(path),
        })

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    long_path = output / "per_gene_metrics.tsv"
    association_path = output / "performance_attribute_associations.tsv"
    extremes_path = output / "top_bottom_genes.tsv"
    summary_path = output / "model_gene_breadth.tsv"
    _write(long_path, list(long_rows[0]), long_rows)
    _write(association_path, list(association_rows[0]), association_rows)
    _write(extremes_path, list(extreme_rows[0]), extreme_rows)
    _write(summary_path, list(model_summary_rows[0]), model_summary_rows)
    return {
        "per_gene": long_path, "associations": association_path,
        "top_bottom": extremes_path, "summary": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecars", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--noise-ceiling")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    outputs = analyze(
        args.sidecars, output_dir=args.output_dir,
        noise_ceiling_path=args.noise_ceiling, top_n=args.top_n,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
