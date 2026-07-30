"""Deterministic train-only gene panels for secondary Gen3 evaluation.

The model is still trained on the complete frozen manifest gene panel and
full-panel PCC/RMSE remain the primary metrics.  These panels only add the
common top-variable-gene views used by related work.  Ranking is computed in
the exact normalized/log1p target space consumed by Gen3, using training
samples only; validation and test expression are never inspected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from gen3_multiscale.data.dataset_manifest import gene_panel_hash
from gen3_multiscale.data.example_builder import load_expression_for_model_target_space

_VERSION = 1
_KIND = "gen3_train_derived_gene_panels"
_METHOD = "pooled_training_spot_variance_in_model_target_space"


def dataset_manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _canonical_hash(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _column_sums_and_squares(matrix) -> tuple[np.ndarray, np.ndarray, int]:
    if sparse.issparse(matrix):
        sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        squares = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(np.float64)
        n_rows = int(matrix.shape[0])
    else:
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"training expression must be rank-2, got shape {array.shape}")
        sums = array.sum(axis=0, dtype=np.float64)
        squares = np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n_rows = int(array.shape[0])
    if not np.isfinite(sums).all() or not np.isfinite(squares).all():
        raise ValueError("training expression contains NaN/Inf")
    return sums, squares, n_rows


def build_train_derived_gene_panels(manifest: dict, *, panel_sizes: tuple[int, ...] = (50, 200)) -> dict:
    """Build nested variance-ranked panels from ``manifest.train_sample_ids`` only."""
    gene_names = [str(gene) for gene in manifest["gene_panel"]]
    train_ids = [str(sample_id) for sample_id in manifest.get("train_sample_ids", [])]
    if not gene_names or not train_ids:
        raise ValueError("manifest must contain a non-empty gene_panel and train_sample_ids")
    build_args = manifest.get("build_args") or {}
    transform = str(build_args.get("expression_transform", ""))
    if "log1p" not in transform:
        raise ValueError(
            f"train_log1p_variance panels require a log1p target transform, got {transform!r}"
        )
    sizes = tuple(sorted(set(int(size) for size in panel_sizes)))
    if not sizes or sizes[0] <= 0 or sizes[-1] > len(gene_names):
        raise ValueError(f"panel_sizes must be unique positive values <= {len(gene_names)}, got {panel_sizes}")

    total_sum = np.zeros(len(gene_names), dtype=np.float64)
    total_squares = np.zeros(len(gene_names), dtype=np.float64)
    n_spots = 0
    for sample_id in train_ids:
        adata = load_expression_for_model_target_space(manifest, sample_id)
        if list(map(str, adata.var_names)) != gene_names:
            raise ValueError(f"{sample_id}: loaded gene order differs from the frozen manifest gene panel")
        sums, squares, rows = _column_sums_and_squares(adata.X)
        if sums.shape != total_sum.shape:
            raise ValueError(f"{sample_id}: expression width {sums.size} != gene panel width {len(gene_names)}")
        total_sum += sums
        total_squares += squares
        n_spots += rows
    if n_spots < 2:
        raise ValueError("at least two training spots are required to rank genes by variance")

    variance = np.maximum(total_squares / n_spots - np.square(total_sum / n_spots), 0.0)
    # Explicit gene-name tie break makes the result independent of sort implementation.
    order = sorted(range(len(gene_names)), key=lambda idx: (-float(variance[idx]), gene_names[idx]))
    ranking = [{"gene": gene_names[idx], "variance": float(variance[idx])} for idx in order]
    panels = {
        f"train_log1p_variance_top{size}": [gene_names[idx] for idx in order[:size]]
        for size in sizes
    }
    artifact = {
        "version": _VERSION,
        "kind": _KIND,
        "method": _METHOD,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "train_sample_ids": train_ids,
        "n_training_spots": n_spots,
        "expression_transform": build_args.get("expression_transform"),
        "expression_target_sum": build_args.get("expression_target_sum"),
        "ranking": ranking,
        "panels": panels,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    return artifact


def validate_train_derived_gene_panels(artifact: dict, manifest: dict) -> dict:
    if artifact.get("version") != _VERSION or artifact.get("kind") != _KIND or artifact.get("method") != _METHOD:
        raise ValueError("unsupported train-derived gene-panel artifact schema")
    if artifact.get("artifact_sha256") != _canonical_hash(artifact):
        raise ValueError("train-derived gene-panel artifact SHA256 mismatch")
    expected = {
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(manifest),
        "gene_panel_hash": gene_panel_hash([str(g) for g in manifest["gene_panel"]]),
        "train_sample_ids": [str(s) for s in manifest.get("train_sample_ids", [])],
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f"train-derived gene-panel artifact {key} does not match the live manifest")
    ranking = artifact.get("ranking")
    panels = artifact.get("panels")
    if not isinstance(ranking, list) or len(ranking) != len(manifest["gene_panel"]):
        raise ValueError("train-derived gene-panel ranking is incomplete")
    ranked_genes = [row.get("gene") for row in ranking if isinstance(row, dict)]
    if len(ranked_genes) != len(ranking) or len(set(ranked_genes)) != len(ranked_genes):
        raise ValueError("train-derived gene-panel ranking contains malformed or duplicate genes")
    if set(ranked_genes) != set(map(str, manifest["gene_panel"])):
        raise ValueError("train-derived gene-panel ranking does not cover the frozen gene panel exactly")
    if not all(np.isfinite(float(row.get("variance", np.nan))) and float(row["variance"]) >= 0 for row in ranking):
        raise ValueError("train-derived gene-panel ranking contains invalid variance values")
    if not isinstance(panels, dict) or not panels:
        raise ValueError("train-derived gene-panel artifact has no panels")
    previous: list[str] = []
    for name, genes in sorted(panels.items(), key=lambda item: len(item[1])):
        if not isinstance(name, str) or not isinstance(genes, list) or not genes or len(set(genes)) != len(genes):
            raise ValueError(f"invalid train-derived panel {name!r}")
        if genes != ranked_genes[: len(genes)]:
            raise ValueError(f"train-derived panel {name!r} is not a prefix of the recorded ranking")
        if previous and genes[: len(previous)] != previous:
            raise ValueError("train-derived panels are not nested")
        previous = genes
    return artifact


def save_train_derived_gene_panels(artifact: dict, path: str | Path) -> Path:
    path = Path(path)
    encoded = (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() == encoded:
            return path
        raise FileExistsError(f"{path} already exists with different content; panel artifacts are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    return path


def load_train_derived_gene_panels(path: str | Path, manifest: dict) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"train-derived gene-panel artifact is missing: {path}")
    return validate_train_derived_gene_panels(json.loads(path.read_text()), manifest)
