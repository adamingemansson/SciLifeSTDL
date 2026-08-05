"""Gen3-specific, manifest-driven histology-morphology spot-feature
cache -- mirrors `data/spot_feature_cache.py`'s GigaPath discipline
(mandatory content provenance, atomic writes, strict validation on
load, its own on-disk directory) for the deterministic, non-model
features `histology_features.py` computes.

Unlike the GigaPath/UNI2 caches, an unavailable spot's cached row is
NOT all-zero: only its own-tile slice is (matching `compute_spot_tile_
features`'s zero-placeholder contract) -- its neighbor/regional/slide
aggregate slices still carry real values pooled from nearby AVAILABLE
spots, by design (see `histology_features.compute_multiscale_histology_
features`'s own docstring). Load-time validation checks the own-tile
slice specifically, not the whole row.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from gen3_multiscale.conditional_wae.histology_features import (
    FEATURE_DIM, FEATURE_SPEC, _TILE_FEATURE_DIM,
    compute_multiscale_histology_features, compute_spot_tile_features,
)

_REQUIRED_FIELDS = {
    "features", "barcodes", "image_source_available", "content_sha256",
    "feature_spec", "feature_dim", "neighbor_k", "regional_radius_multiplier", "schema_version",
}
_SUPPORTED_SCHEMA_VERSIONS = {1}


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    return Path(cache_root) / "histology_gen3_spot_cache" / f"{sample_id}.npz"


def _content_sha256(
    barcodes: np.ndarray, image_source_available: np.ndarray, patches: np.ndarray, coords: np.ndarray,
) -> str:
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    coords = np.asarray(coords, dtype=np.float64)
    available_idx = np.flatnonzero(image_source_available)
    available_patches = np.ascontiguousarray(patches[available_idx])
    digest = hashlib.sha256()
    digest.update(b"\x1f".join(b.encode("utf-8") for b in barcodes))
    digest.update(image_source_available.tobytes())
    digest.update(str(available_patches.shape).encode("ascii"))
    digest.update(str(available_patches.dtype).encode("ascii"))
    digest.update(available_patches.tobytes())
    digest.update(np.ascontiguousarray(coords).tobytes())
    return digest.hexdigest()


def build_histology_feature_cache(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    coords: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
    *,
    neighbor_k: int = 6,
    regional_radius_multiplier: float = 5.0,
) -> Path:
    """One deterministic compute pass per manifest sample -- no model, no
    checkpoint, purely `histology_features.py`'s fixed formulas over the
    real patches/coordinates this sample already has."""
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    coords = np.asarray(coords, dtype=np.float64)
    n = barcodes.shape[0]
    if patches.shape[0] != n or image_source_available.shape[0] != n or coords.shape[0] != n:
        raise ValueError(f"{sample_id}: barcodes/coords/patches/image_source_available must be row-aligned")
    unique_barcodes, counts = np.unique(barcodes, return_counts=True)
    duplicated = unique_barcodes[counts > 1]
    if duplicated.size:
        raise ValueError(f"{sample_id}: barcodes contain {duplicated.size} duplicate value(s)")

    tile_features = compute_spot_tile_features(patches, image_source_available)
    features = compute_multiscale_histology_features(
        tile_features, coords, image_source_available,
        neighbor_k=neighbor_k, regional_radius_multiplier=regional_radius_multiplier,
    )
    if features.shape != (n, FEATURE_DIM):
        raise RuntimeError(f"{sample_id}: internal error, features has shape {features.shape}, expected ({n}, {FEATURE_DIM})")
    if not np.isfinite(features).all():
        raise ValueError(f"{sample_id}: computed histology features contain non-finite values")

    path = _cache_path(cache_root, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            features=features, barcodes=barcodes, image_source_available=image_source_available,
            content_sha256=np.asarray(_content_sha256(barcodes, image_source_available, patches, coords)),
            feature_spec=np.asarray(FEATURE_SPEC), feature_dim=np.asarray(FEATURE_DIM),
            neighbor_k=np.asarray(int(neighbor_k)), regional_radius_multiplier=np.asarray(float(regional_radius_multiplier)),
            schema_version=np.asarray(1),
        )
    os.replace(tmp, path)
    print(
        f"{sample_id}: wrote histology-feature cache for {int(image_source_available.sum())}/{n} "
        f"available spots to {path}", flush=True,
    )
    return path


def load_histology_features(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    coords: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> dict:
    """Load-and-strictly-validate: required fields present, feature spec/
    dim/schema match this module's current expectations, barcode
    identity/order and availability match the ALREADY-LOADED real data
    the caller passes in, own-tile slice is exactly zero for unavailable
    spots, and the real patch+coordinate content hash matches (a cache
    built from since-changed patches or coordinates is rejected)."""
    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"histology-feature cache missing for {sample_id}: {path}. Build it with "
            "scripts/precompute_gen3_histology_features.py before training."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(f"histology-feature cache {path} is missing fields {missing} -- rebuild it")

    if str(cached["feature_spec"]) != FEATURE_SPEC:
        raise ValueError(
            f"histology-feature cache {path} feature_spec={str(cached['feature_spec'])!r}, expected "
            f"{FEATURE_SPEC!r} -- rebuild it against the current feature definition"
        )
    schema_version = int(np.asarray(cached["schema_version"]).item())
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"histology-feature cache {path} schema_version={schema_version} is not supported")

    real_barcodes = np.asarray([str(b) for b in barcodes])
    real_availability = np.asarray(image_source_available, dtype=bool)
    real_patches = np.asarray(patches)
    real_coords = np.asarray(coords, dtype=np.float64)
    n = real_barcodes.shape[0]
    if real_patches.shape[0] != n or real_availability.shape[0] != n or real_coords.shape[0] != n:
        raise ValueError(f"{sample_id}: barcodes/coords/patches/image_source_available passed to load are not row-aligned")

    cached_barcodes = np.asarray([str(b) for b in cached["barcodes"]])
    if cached_barcodes.shape[0] != n or not np.array_equal(cached_barcodes, real_barcodes):
        raise ValueError(f"histology-feature cache {path} barcode identity/order mismatch -- rebuild it")
    cached_availability = np.asarray(cached["image_source_available"], dtype=bool)
    if cached_availability.shape != real_availability.shape or not np.array_equal(cached_availability, real_availability):
        raise ValueError(f"histology-feature cache {path} image_source_available mismatch -- rebuild it")

    feature_dim = int(np.asarray(cached["feature_dim"]).item())
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"histology-feature cache {path} feature_dim={feature_dim}, expected {FEATURE_DIM}")
    features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (n, FEATURE_DIM):
        raise ValueError(f"histology-feature cache {path} features has shape {features.shape}, expected ({n}, {FEATURE_DIM})")
    if not np.isfinite(features).all():
        raise ValueError(f"histology-feature cache {path} contains non-finite feature values")

    unavailable_idx = np.flatnonzero(~real_availability)
    own_tile_slice = features[unavailable_idx, :_TILE_FEATURE_DIM]
    if unavailable_idx.size and not np.array_equal(
        own_tile_slice, np.zeros((unavailable_idx.size, _TILE_FEATURE_DIM), dtype=np.float32),
    ):
        raise ValueError(
            f"histology-feature cache {path} has a nonzero own-tile slice for a spot marked "
            "image_source_available=False -- corrupted cache, refusing to load"
        )

    real_content_hash = _content_sha256(real_barcodes, real_availability, real_patches, real_coords)
    if real_content_hash != str(cached["content_sha256"]):
        raise ValueError(
            f"histology-feature cache {path} content_sha256 does not match the real, currently-loaded "
            f"patches/coordinates for {sample_id} -- rebuild it"
        )

    return {
        "features": features,
        "barcodes": real_barcodes,
        "image_source_available": real_availability,
        "provenance": {
            "feature_spec": FEATURE_SPEC,
            "feature_dim": feature_dim,
            "neighbor_k": int(np.asarray(cached["neighbor_k"]).item()),
            "regional_radius_multiplier": float(np.asarray(cached["regional_radius_multiplier"]).item()),
            "schema_version": schema_version,
        },
    }


def cfg_cache_root(cfg) -> Path:
    """Mirrors `gen4.uni2_spot_cache.cfg_cache_root`'s resolution exactly
    -- a distinct config key, falling back to the same shared
    `hest_cache_dir`/`hest_data_dir` default every other Gen3 spot-
    feature cache uses."""
    configured = cfg.data.get("gen3_histology_feature_cache_dir")
    if configured:
        return Path(str(configured))
    cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
    return Path(str(cache_root))


def load_gen3_histology_features(
    cfg,
    sample_id: str,
    barcodes: np.ndarray,
    coords: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> np.ndarray:
    """Config-driven wrapper: resolves the cache root from `cfg` and
    returns the verified `[N, FEATURE_DIM]` feature matrix directly."""
    cache_root = cfg_cache_root(cfg)
    loaded = load_histology_features(cache_root, sample_id, barcodes, coords, patches, image_source_available)
    return loaded["features"]
