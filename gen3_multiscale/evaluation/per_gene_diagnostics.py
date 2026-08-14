"""Compact per-gene diagnostics for exact whole-slide H&E-to-ST evaluation.

Aggregate report tables answer which model is best, but not *which genes* it
predicts or why.  This module records one row per held-out slide and one column
per gene in a compressed sidecar.  The sidecar is deliberately separate from
the JSON report: embedding 14 x 17k vectors in every report makes ordinary
summaries unwieldy.

All quantities are descriptive held-out diagnostics.  They must never be used
to choose training genes or feed a model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from gen3_multiscale.evaluation.metrics import (
    pearson_per_gene,
    r2_per_gene,
    spearman_per_gene,
)
from gen3_multiscale.evaluation.structured_field_metrics import (
    _moran_by_gene,
    undirected_knn_edges,
)


METRIC_NAMES = (
    "pcc", "spearman", "r2", "rmse", "mae",
    "target_mean", "target_std", "target_nonzero_fraction",
    "prediction_mean", "prediction_std",
    "target_moran_i", "prediction_moran_i",
    "local_gradient_pcc", "target_local_gradient_energy",
    "prediction_local_gradient_energy", "local_gradient_sign_agreement",
)


def per_gene_whole_slide_diagnostics(
    predicted: np.ndarray,
    target: np.ndarray,
    coords: np.ndarray,
    *,
    local_k: int = 6,
    nontrivial_threshold: float = 0.05,
    chunk_size: int = 32,
) -> dict[str, np.ndarray]:
    """Return per-gene point, spatial and gradient diagnostics for one slide."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != target.shape or predicted.shape[0] < 2:
        raise ValueError("predicted and target must be matching [n_spots >= 2, n_genes] arrays")
    if coords.shape != (predicted.shape[0], 2):
        raise ValueError("coords must align with expression rows")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("per-gene diagnostic inputs must be finite")
    if local_k < 1 or chunk_size < 1 or nontrivial_threshold < 0:
        raise ValueError("invalid per-gene spatial diagnostic settings")

    error = predicted - target
    result = {
        "pcc": pearson_per_gene(predicted, target),
        "spearman": spearman_per_gene(predicted, target),
        "r2": r2_per_gene(predicted, target),
        "rmse": np.sqrt(np.mean(np.square(error), axis=0)),
        "mae": np.mean(np.abs(error), axis=0),
        "target_mean": np.mean(target, axis=0),
        "target_std": np.std(target, axis=0),
        "target_nonzero_fraction": np.mean(target > 0, axis=0),
        "prediction_mean": np.mean(predicted, axis=0),
        "prediction_std": np.std(predicted, axis=0),
    }

    edges = undirected_knn_edges(coords, k_neighbors=local_k)
    result["target_moran_i"] = _moran_by_gene(target, edges)
    result["prediction_moran_i"] = _moran_by_gene(predicted, edges)
    left, right = edges[:, 0], edges[:, 1]
    n_genes = predicted.shape[1]
    gradient_pcc = np.full(n_genes, np.nan, dtype=np.float64)
    target_energy = np.full(n_genes, np.nan, dtype=np.float64)
    prediction_energy = np.full(n_genes, np.nan, dtype=np.float64)
    sign_agreement = np.full(n_genes, np.nan, dtype=np.float64)
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        # Keep these diagnostics in the common normalized-log1p target space.
        # Evaluator-specific training scales are intentionally not used: that
        # would make the same held-out target gene acquire different
        # attributes in different model sidecars.
        pred_delta = predicted[right, start:end] - predicted[left, start:end]
        true_delta = target[right, start:end] - target[left, start:end]
        gradient_pcc[start:end] = pearson_per_gene(pred_delta, true_delta)
        prediction_energy[start:end] = np.mean(np.square(pred_delta), axis=0)
        target_energy[start:end] = np.mean(np.square(true_delta), axis=0)
        nontrivial = np.abs(true_delta) >= nontrivial_threshold
        matches = (np.signbit(pred_delta) == np.signbit(true_delta)) & nontrivial
        counts = nontrivial.sum(axis=0)
        np.divide(
            matches.sum(axis=0), counts,
            out=sign_agreement[start:end], where=counts > 0,
        )
    result.update({
        "local_gradient_pcc": gradient_pcc,
        "target_local_gradient_energy": target_energy,
        "prediction_local_gradient_energy": prediction_energy,
        "local_gradient_sign_agreement": sign_agreement,
    })
    return {name: np.asarray(result[name], dtype=np.float32) for name in METRIC_NAMES}


class PerGeneDiagnosticsAccumulator:
    """Collect fixed-width per-slide arrays and save one validated NPZ sidecar."""

    def __init__(self, gene_names: list[str]) -> None:
        self.gene_names = [str(value) for value in gene_names]
        if not self.gene_names or len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must be non-empty and unique")
        self._metadata: list[dict[str, str]] = []
        self._metrics: dict[str, list[np.ndarray]] = {name: [] for name in METRIC_NAMES}

    def add_slide(
        self, *, sample_id: str, patient_id: str, organ: str,
        predicted: np.ndarray, target: np.ndarray, coords: np.ndarray,
        local_k: int = 6,
    ) -> None:
        if any(row["sample_id"] == str(sample_id) for row in self._metadata):
            raise ValueError(f"duplicate per-gene diagnostic slide: {sample_id}")
        values = per_gene_whole_slide_diagnostics(
            predicted, target, coords, local_k=local_k,
        )
        if any(value.shape != (len(self.gene_names),) for value in values.values()):
            raise ValueError("per-gene diagnostics do not align with gene_names")
        self._metadata.append({
            "sample_id": str(sample_id), "patient_id": str(patient_id), "organ": str(organ),
        })
        for name in METRIC_NAMES:
            self._metrics[name].append(values[name])

    def save(self, path: str | Path, *, provenance: dict[str, Any]) -> Path:
        if not self._metadata:
            raise ValueError("cannot save empty per-gene diagnostics")
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        metadata_json = json.dumps({
            "version": 1,
            "kind": "hest_mk_per_gene_whole_slide_diagnostics",
            "n_slides": len(self._metadata),
            "n_genes": len(self.gene_names),
            "gradient_units": "normalized_log1p_expression_difference_per_knn_edge",
            "provenance": provenance,
        }, sort_keys=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(metadata_json),
                gene_names=np.asarray(self.gene_names),
                sample_ids=np.asarray([row["sample_id"] for row in self._metadata]),
                patient_ids=np.asarray([row["patient_id"] for row in self._metadata]),
                organs=np.asarray([row["organ"] for row in self._metadata]),
                **{
                    name: np.stack(self._metrics[name], axis=0).astype(np.float32)
                    for name in METRIC_NAMES
                },
            )
        os.replace(temporary, path)
        return path


def load_per_gene_diagnostics(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on a malformed diagnostics sidecar."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "metadata_json", "gene_names", "sample_ids", "patient_ids", "organs", *METRIC_NAMES,
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"per-gene diagnostics are incomplete; missing={missing}")
        result: dict[str, Any] = {
            "metadata": json.loads(str(payload["metadata_json"].item())),
            "gene_names": [str(value) for value in payload["gene_names"].tolist()],
            "sample_ids": [str(value) for value in payload["sample_ids"].tolist()],
            "patient_ids": [str(value) for value in payload["patient_ids"].tolist()],
            "organs": [str(value) for value in payload["organs"].tolist()],
        }
        for name in METRIC_NAMES:
            result[name] = np.asarray(payload[name], dtype=np.float32)
    shape = (len(result["sample_ids"]), len(result["gene_names"]))
    if len(set(result["sample_ids"])) != len(result["sample_ids"]):
        raise ValueError("per-gene diagnostics contain duplicate sample IDs")
    if not all(result[name].shape == shape for name in METRIC_NAMES):
        raise ValueError("per-gene diagnostic arrays have inconsistent shapes")
    if result["metadata"].get("kind") != "hest_mk_per_gene_whole_slide_diagnostics":
        raise ValueError("unrecognized per-gene diagnostics kind")
    return result
