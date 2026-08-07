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

_VERSION = 2
_KIND = "gen3_train_derived_gene_panels"
_METHOD = "pooled_training_spot_variance_in_model_target_space"
_METHOD_BY_ORGAN = "organ_stratified_dispersion_in_model_target_space"
_MIN_TRAIN_SAMPLES_PER_ORGAN = 2
_DISPERSION_N_BINS = 20


def _binned_dispersion_ranking(
    gene_names: list[str], mean: np.ndarray, variance: np.ndarray, *, n_bins: int = _DISPERSION_N_BINS,
) -> list[dict]:
    """Macosko/Seurat-style mean-variance-trend-corrected dispersion ranking.

    Raw variance favors genes that are near-zero everywhere except a
    handful of extreme spikes (e.g. a liver marker like ALB pooled
    across organs, or any sparse/dropout-dominated gene) -- those spikes
    inflate variance without reflecting a real spatial gradient. Binning
    genes by mean expression and z-scoring dispersion (variance/mean)
    within each bin corrects for the mean-variance relationship so
    ranking reflects excess variance relative to genes of similar
    expression level, not raw magnitude.
    """
    n_genes = len(gene_names)
    dispersion = np.zeros(n_genes, dtype=np.float64)
    positive = mean > 1e-12
    dispersion[positive] = variance[positive] / mean[positive]
    bin_count = max(1, min(n_bins, n_genes))
    bin_edges = np.quantile(mean, np.linspace(0.0, 1.0, bin_count + 1))
    bin_edges[-1] += 1e-9  # right-inclusive last bin
    bin_index = np.clip(np.searchsorted(bin_edges, mean, side="right") - 1, 0, bin_count - 1)
    dispersion_norm = np.zeros(n_genes, dtype=np.float64)
    for bin_id in range(bin_count):
        members = bin_index == bin_id
        if not members.any():
            continue
        bin_values = dispersion[members]
        bin_mean = float(bin_values.mean())
        bin_std = float(bin_values.std())
        dispersion_norm[members] = (
            (bin_values - bin_mean) / bin_std if bin_std > 1e-12 else bin_values - bin_mean
        )
    order = sorted(range(n_genes), key=lambda idx: (-float(dispersion_norm[idx]), gene_names[idx]))
    return [
        {
            "gene": gene_names[idx], "mean": float(mean[idx]), "variance": float(variance[idx]),
            "dispersion_norm": float(dispersion_norm[idx]),
        }
        for idx in order
    ]


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

    samples = manifest.get("samples") or {}
    organ_by_sample = {}
    for sample_id in train_ids:
        record = samples.get(sample_id)
        if not record or "organ" not in record:
            raise ValueError(f"{sample_id}: manifest.samples is missing an 'organ' field")
        organ_by_sample[sample_id] = str(record["organ"])

    total_sum = np.zeros(len(gene_names), dtype=np.float64)
    total_squares = np.zeros(len(gene_names), dtype=np.float64)
    n_spots = 0
    organ_sum: dict[str, np.ndarray] = {}
    organ_squares: dict[str, np.ndarray] = {}
    organ_n_spots: dict[str, int] = {}
    organ_train_ids: dict[str, list[str]] = {}
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
        organ = organ_by_sample[sample_id]
        organ_sum[organ] = organ_sum.get(organ, np.zeros(len(gene_names), dtype=np.float64)) + sums
        organ_squares[organ] = organ_squares.get(organ, np.zeros(len(gene_names), dtype=np.float64)) + squares
        organ_n_spots[organ] = organ_n_spots.get(organ, 0) + rows
        organ_train_ids.setdefault(organ, []).append(sample_id)
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

    ranking_by_organ: dict[str, list[dict]] = {}
    panels_by_organ: dict[str, dict[str, list[str]]] = {}
    for organ in sorted(organ_train_ids):
        n_organ_train_samples = len(organ_train_ids[organ])
        if n_organ_train_samples < _MIN_TRAIN_SAMPLES_PER_ORGAN:
            raise ValueError(
                f"organ {organ!r} has only {n_organ_train_samples} training sample(s); at least "
                f"{_MIN_TRAIN_SAMPLES_PER_ORGAN} are required to rank genes by organ-specific dispersion"
            )
        organ_mean = organ_sum[organ] / organ_n_spots[organ]
        organ_variance = np.maximum(organ_squares[organ] / organ_n_spots[organ] - np.square(organ_mean), 0.0)
        organ_ranking = _binned_dispersion_ranking(gene_names, organ_mean, organ_variance)
        ranking_by_organ[organ] = organ_ranking
        ranked_genes = [row["gene"] for row in organ_ranking]
        panels_by_organ[organ] = {
            f"train_dispersion_top{size}": ranked_genes[:size] for size in sizes
        }

    artifact = {
        "version": _VERSION,
        "kind": _KIND,
        "method": _METHOD,
        "method_by_organ": _METHOD_BY_ORGAN,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "train_sample_ids": train_ids,
        "n_training_spots": n_spots,
        "expression_transform": build_args.get("expression_transform"),
        "expression_target_sum": build_args.get("expression_target_sum"),
        "ranking": ranking,
        "panels": panels,
        "ranking_by_organ": ranking_by_organ,
        "panels_by_organ": panels_by_organ,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    return artifact


def validate_train_derived_gene_panels(artifact: dict, manifest: dict) -> dict:
    if (
        artifact.get("version") != _VERSION or artifact.get("kind") != _KIND
        or artifact.get("method") != _METHOD or artifact.get("method_by_organ") != _METHOD_BY_ORGAN
    ):
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

    samples = manifest.get("samples") or {}
    expected_organs = {
        str(samples[sid]["organ"])
        for sid in expected["train_sample_ids"]
        if sid in samples and "organ" in samples[sid]
    }
    ranking_by_organ = artifact.get("ranking_by_organ")
    panels_by_organ = artifact.get("panels_by_organ")
    if not isinstance(ranking_by_organ, dict) or not isinstance(panels_by_organ, dict):
        raise ValueError("train-derived gene-panel artifact is missing ranking_by_organ/panels_by_organ")
    if set(ranking_by_organ) != expected_organs or set(panels_by_organ) != expected_organs:
        raise ValueError("train-derived gene-panel artifact organs do not match the live manifest's train organs")
    for organ in expected_organs:
        organ_ranking = ranking_by_organ[organ]
        if not isinstance(organ_ranking, list) or len(organ_ranking) != len(manifest["gene_panel"]):
            raise ValueError(f"train-derived gene-panel ranking for organ {organ!r} is incomplete")
        organ_ranked_genes = [row.get("gene") for row in organ_ranking if isinstance(row, dict)]
        if len(organ_ranked_genes) != len(organ_ranking) or len(set(organ_ranked_genes)) != len(organ_ranked_genes):
            raise ValueError(f"train-derived gene-panel ranking for organ {organ!r} has malformed/duplicate genes")
        if set(organ_ranked_genes) != set(ranked_genes):
            raise ValueError(f"train-derived gene-panel ranking for organ {organ!r} does not cover the gene panel")
        if not all(np.isfinite(float(row.get("dispersion_norm", np.nan))) for row in organ_ranking):
            raise ValueError(f"train-derived gene-panel ranking for organ {organ!r} has invalid dispersion values")
        organ_panels = panels_by_organ[organ]
        if not isinstance(organ_panels, dict) or not organ_panels:
            raise ValueError(f"train-derived gene-panel artifact has no panels for organ {organ!r}")
        organ_previous: list[str] = []
        for name, genes in sorted(organ_panels.items(), key=lambda item: len(item[1])):
            if not isinstance(name, str) or not isinstance(genes, list) or not genes or len(set(genes)) != len(genes):
                raise ValueError(f"invalid train-derived panel {name!r} for organ {organ!r}")
            if genes != organ_ranked_genes[: len(genes)]:
                raise ValueError(f"train-derived panel {name!r} for organ {organ!r} is not a prefix of its ranking")
            if organ_previous and genes[: len(organ_previous)] != organ_previous:
                raise ValueError(f"train-derived panels for organ {organ!r} are not nested")
            organ_previous = genes
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
