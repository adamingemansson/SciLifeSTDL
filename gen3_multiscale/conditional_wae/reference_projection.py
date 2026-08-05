"""Shared, frozen output-space GEX projection for MK validation diagnostics.

Fits one PCA basis (and fixed downstream clusters) on TRUE validation GEX
from a fixed reference cohort, once. Both true and predicted GEX are then
projected through this SAME frozen basis, so spatial PC maps and cluster
colors mean the same thing across training steps, WAE dimensionalities and
architectures (plain vs. FiLM-conditioned encoder). This is diagnostic only
-- it never feeds into any loss or checkpoint-selection decision.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

_VERSION = 1
_KIND = "conditional_wae_reference_gex_projection"
_ARRAY_KEYS = (
    "mean", "scale", "components", "explained_variance_ratio", "pc_ranges", "cluster_centroids",
)


def _even_positions(length: int, limit: int) -> np.ndarray:
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, limit, dtype=np.int64))


def build_reference_gex_projection(
    samples: dict, sample_ids: list, gene_names: list[str], *,
    n_components: int = 3, n_clusters: int = 6, seed: int = 0, max_points: int = 20_000,
) -> dict:
    """Deterministic: a fixed cohort (sorted sample_ids, evenly-subsampled
    spots per sample) always produces the same basis and clusters."""
    if n_components < 1 or n_clusters < 1 or max_points < 1:
        raise ValueError("n_components, n_clusters and max_points must be positive")
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")
    gene_names = [str(gene) for gene in gene_names]
    ordered_ids = sorted(str(sid) for sid in sample_ids)
    per_sample_limit = max(1, max_points // len(ordered_ids))
    rows = []
    cohort_ids: list[str] = []
    for sample_id in ordered_ids:
        sample = samples[sample_id]
        expression = np.asarray(sample.adata.X, dtype=np.float32)
        if list(map(str, sample.adata.var_names)) != gene_names:
            raise ValueError(f"{sample_id}: gene order does not match the frozen gene panel")
        remaining = max_points - sum(len(r) for r in rows)
        if remaining <= 0:
            break
        positions = _even_positions(expression.shape[0], min(per_sample_limit, remaining))
        rows.append(expression[positions])
        cohort_ids.extend(f"{sample_id}:{int(position)}" for position in positions)
    if not rows:
        raise ValueError("reference cohort produced zero rows")
    pooled = np.concatenate(rows, axis=0)
    if not np.isfinite(pooled).all():
        raise ValueError("reference cohort expression contains a non-finite value")

    mean = pooled.mean(axis=0)
    centered = pooled - mean
    scale = centered.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = centered / scale

    rank = min(n_components, normalized.shape[0], normalized.shape[1])
    _u, s, vh = np.linalg.svd(normalized, full_matrices=False)
    components = vh[:rank].astype(np.float32)
    denom = max(1, normalized.shape[0] - 1)
    explained_variance = (s[:rank] ** 2) / denom
    total_variance = float((s ** 2).sum() / denom)
    explained_variance_ratio = (
        (explained_variance / total_variance) if total_variance > 0 else np.zeros(rank)
    ).astype(np.float32)
    coords = normalized @ components.T
    pc_ranges = np.stack([coords.min(axis=0), coords.max(axis=0)], axis=1).astype(np.float32)

    from sklearn.cluster import KMeans
    n_clusters_used = min(n_clusters, len(coords))
    kmeans = KMeans(n_clusters=n_clusters_used, random_state=int(seed), n_init=10)
    kmeans.fit(coords)

    metadata = {
        "version": _VERSION,
        "kind": _KIND,
        "gene_names": gene_names,
        "cohort_sample_ids": ordered_ids,
        "n_points": int(len(pooled)),
        "max_points": int(max_points),
        "n_clusters": int(n_clusters_used),
        "seed": int(seed),
    }
    provenance_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return {
        **metadata,
        "provenance_hash": provenance_hash,
        "cohort_ids": cohort_ids,
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "components": components,
        "explained_variance_ratio": explained_variance_ratio,
        "pc_ranges": pc_ranges,
        "cluster_centroids": kmeans.cluster_centers_.astype(np.float32),
    }


def project_onto_reference(gex_matrix, gene_names: list[str], projection: dict) -> np.ndarray:
    """Project ANY [N, n_genes] GEX matrix (true or predicted) through the
    frozen reference basis -- the whole point is that both use this exact
    same function, so their PC coordinates are directly comparable."""
    gex_matrix = np.asarray(gex_matrix, dtype=np.float32)
    if [str(gene) for gene in gene_names] != projection["gene_names"]:
        raise ValueError("gene order does not match the reference projection's frozen gene panel")
    if gex_matrix.ndim != 2 or gex_matrix.shape[1] != len(projection["gene_names"]):
        raise ValueError(f"gex_matrix must be [N, {len(projection['gene_names'])}]")
    normalized = (gex_matrix - projection["mean"]) / projection["scale"]
    return normalized @ projection["components"].T


def assign_clusters(pca_coords: np.ndarray, projection: dict) -> np.ndarray:
    centroids = np.asarray(projection["cluster_centroids"])
    pca_coords = np.asarray(pca_coords, dtype=np.float32)
    if pca_coords.ndim != 2 or pca_coords.shape[1] != centroids.shape[1]:
        raise ValueError(f"pca_coords must be [N, {centroids.shape[1]}]")
    distances = np.linalg.norm(pca_coords[:, None, :] - centroids[None, :, :], axis=-1)
    return distances.argmin(axis=1)


def save_reference_projection(projection: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    npz_path = path.with_suffix(".npz")
    metadata = {key: value for key, value in projection.items() if key not in _ARRAY_KEYS}
    tmp_npz = npz_path.with_name(f"{npz_path.name}.tmp.{os.getpid()}")
    with open(tmp_npz, "wb") as handle:
        np.savez(handle, **{key: np.asarray(projection[key]) for key in _ARRAY_KEYS})
    os.replace(tmp_npz, npz_path)
    tmp_json = json_path.with_name(f"{json_path.name}.tmp.{os.getpid()}")
    tmp_json.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    os.replace(tmp_json, json_path)
    return json_path


def load_reference_projection(path: str | Path) -> dict:
    path = Path(path)
    json_path = path.with_suffix(".json")
    npz_path = path.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(f"reference projection not found at {path}")
    metadata = json.loads(json_path.read_text())
    with np.load(npz_path) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return {**metadata, **arrays}


def ensure_reference_gex_projection(
    path: str | Path, samples: dict, sample_ids: list, gene_names: list[str], **build_kwargs,
) -> dict:
    """Load-or-atomically-build, mirroring dataset_manifest.ensure_dataset_manifest:
    fails closed if an existing on-disk projection's cohort/gene-panel/seed
    identity doesn't match what these exact inputs would produce, rather than
    silently reusing a stale basis."""
    path = Path(path)
    requested_identity = {
        "gene_names": [str(gene) for gene in gene_names],
        "cohort_sample_ids": sorted(str(sample_id) for sample_id in sample_ids),
        "max_points": int(build_kwargs.get("max_points", 20_000)),
        "seed": int(build_kwargs.get("seed", 0)),
    }
    if path.with_suffix(".json").is_file():
        existing = load_reference_projection(path)
        existing_identity = {
            "gene_names": existing["gene_names"],
            "cohort_sample_ids": existing["cohort_sample_ids"],
            "max_points": existing["max_points"],
            "seed": existing["seed"],
        }
        if existing_identity != requested_identity:
            raise ValueError(
                f"{path}: existing reference projection does not match the requested "
                "cohort/gene-panel/seed -- use a new path or remove the stale projection"
            )
        return existing
    projection = build_reference_gex_projection(samples, sample_ids, gene_names, **build_kwargs)
    save_reference_projection(projection, path)
    return projection
