"""UNI2 per-spot feature cache -- mirrors
`data/spot_feature_cache.py`'s GigaPath cache exactly (GEN4_CONTRACT.md
section 6/9): one encode pass per manifest sample, mandatory pinned
revision, full provenance, atomic writes, strict validation on load, its
own on-disk directory (`uni2_gen3_spot_cache/`) so it can never collide
with or be mistaken for the GigaPath cache.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

_REQUIRED_FIELDS = {
    "features", "barcodes", "image_source_available", "patch_content_sha256",
    "uni2_checkpoint_sha256", "uni2_pinned_revision", "uni2_package_version",
    "uni2_preprocessing_spec", "uni2_output_dim", "uni2_schema_version",
}


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    return Path(cache_root) / "uni2_gen3_spot_cache" / f"{sample_id}.npz"


def _patch_content_sha256(barcodes: np.ndarray, image_source_available: np.ndarray, patches: np.ndarray) -> str:
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    available_idx = np.flatnonzero(image_source_available)
    available_patches = np.ascontiguousarray(patches[available_idx])
    digest = hashlib.sha256()
    digest.update(b"\x1f".join(b.encode("utf-8") for b in barcodes))
    digest.update(image_source_available.tobytes())
    digest.update(str(available_patches.shape).encode("ascii"))
    digest.update(str(available_patches.dtype).encode("ascii"))
    digest.update(available_patches.tobytes())
    return digest.hexdigest()


def build_uni2_spot_feature_cache(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
    encoder,
    batch_size: int = 32,
) -> Path:
    """`encoder` must satisfy `gen4.providers.ImageContextProvider`
    (`encoder.identity`, `encoder.encode_available_patches`). Never call
    the encoder on an unavailable/placeholder patch -- its cached row is
    an explicit zero instead, exactly matching the GigaPath cache's
    contract."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    n = barcodes.shape[0]
    if patches.shape[0] != n or image_source_available.shape[0] != n:
        raise ValueError(f"{sample_id}: barcodes/patches/image_source_available must be row-aligned")
    unique_barcodes, counts = np.unique(barcodes, return_counts=True)
    duplicated = unique_barcodes[counts > 1]
    if duplicated.size:
        raise ValueError(f"{sample_id}: barcodes contain {duplicated.size} duplicate value(s)")

    output_dim = encoder.identity.output_dim
    features = np.zeros((n, output_dim), dtype=np.float32)
    available_idx = np.flatnonzero(image_source_available)
    for start in range(0, available_idx.shape[0], batch_size):
        batch_idx = available_idx[start:start + batch_size]
        out = encoder.encode_available_patches(patches[batch_idx])
        if out.shape != (batch_idx.shape[0], output_dim):
            raise ValueError(f"{sample_id}: encoder returned shape {out.shape}, expected ({batch_idx.shape[0]}, {output_dim})")
        features[batch_idx] = out.astype(np.float32)

    path = _cache_path(cache_root, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            features=features, barcodes=barcodes, image_source_available=image_source_available,
            patch_content_sha256=np.asarray(_patch_content_sha256(barcodes, image_source_available, patches)),
            uni2_checkpoint_sha256=np.asarray(encoder.identity.checkpoint_sha256),
            uni2_pinned_revision=np.asarray(encoder.identity.pinned_revision),
            uni2_package_version=np.asarray(encoder.identity.package_version),
            uni2_preprocessing_spec=np.asarray(encoder.identity.preprocessing_spec),
            uni2_output_dim=np.asarray(output_dim),
            uni2_schema_version=np.asarray(1),
        )
    os.replace(tmp, path)
    return path


def load_uni2_spot_features(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> dict:
    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"UNI2 spot-feature cache missing for {sample_id}: {path}. Build it with "
            "gen4.uni2_spot_cache.build_uni2_spot_feature_cache before training."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(f"UNI2 spot-feature cache {path} is missing fields {missing} -- rebuild it")

    real_barcodes = np.asarray([str(b) for b in barcodes])
    real_availability = np.asarray(image_source_available, dtype=bool)
    real_patches = np.asarray(patches)
    if real_patches.shape[0] != real_barcodes.shape[0] or real_availability.shape[0] != real_barcodes.shape[0]:
        raise ValueError(f"{sample_id}: barcodes/patches/image_source_available passed to load are not row-aligned")

    cached_barcodes = np.asarray([str(b) for b in cached["barcodes"]])
    if not np.array_equal(cached_barcodes, real_barcodes):
        raise ValueError(f"UNI2 spot-feature cache {path} barcode identity/order mismatch -- rebuild it")
    cached_availability = np.asarray(cached["image_source_available"], dtype=bool)
    if not np.array_equal(cached_availability, real_availability):
        raise ValueError(f"UNI2 spot-feature cache {path} image_source_available mismatch -- rebuild it")

    output_dim = int(np.asarray(cached["uni2_output_dim"]).item())
    features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (real_barcodes.shape[0], output_dim):
        raise ValueError(f"UNI2 spot-feature cache {path} features has wrong shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"UNI2 spot-feature cache {path} contains non-finite values")

    unavailable_idx = np.flatnonzero(~real_availability)
    if unavailable_idx.size and not np.array_equal(features[unavailable_idx], np.zeros((unavailable_idx.size, output_dim), dtype=np.float32)):
        raise ValueError(f"UNI2 spot-feature cache {path} has a nonzero row for an unavailable spot -- corrupted")

    real_content_hash = _patch_content_sha256(real_barcodes, real_availability, real_patches)
    if real_content_hash != str(cached["patch_content_sha256"]):
        raise ValueError(f"UNI2 spot-feature cache {path} patch content mismatch -- rebuild it")

    return {
        "features": features,
        "barcodes": real_barcodes,
        "image_source_available": real_availability,
        "provenance": {
            "checkpoint_sha256": str(cached["uni2_checkpoint_sha256"]),
            "pinned_revision": str(cached["uni2_pinned_revision"]),
            "package_version": str(cached["uni2_package_version"]),
            "preprocessing_spec": str(cached["uni2_preprocessing_spec"]),
            "output_dim": output_dim,
            "schema_version": int(np.asarray(cached["uni2_schema_version"]).item()),
        },
    }
