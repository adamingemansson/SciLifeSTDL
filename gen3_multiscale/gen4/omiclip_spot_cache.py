"""OmiCLIP per-spot feature cache -- mirrors `gen4/uni2_spot_cache.py`
exactly: one encode pass per manifest sample, mandatory pinned revision,
full provenance, atomic writes, strict validation on load, its own on-disk
directory (`omiclip_gen3_spot_cache/`) so it can never collide with or be
mistaken for the GigaPath/UNI2 caches.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

_REQUIRED_FIELDS = {
    "features", "barcodes", "image_source_available", "patch_content_sha256",
    "omiclip_checkpoint_sha256", "omiclip_pinned_revision", "omiclip_package_version",
    "omiclip_preprocessing_spec", "omiclip_output_dim", "omiclip_schema_version",
}

# Must match FrozenOmiCLIPTileEncoder's own identity exactly
# (omiclip_encoder.py): OmiCLIP is only ever constructed as coca_ViT-L-14,
# so both the preprocessing spec string and the output dim are fixed, not
# caller-configurable -- a provenance dict claiming anything else did not
# come from a real FrozenOmiCLIPTileEncoder.
_EXPECTED_OMICLIP_PREPROCESSING_SPEC = "omiclip_tile_v1:coca_ViT-L-14:open_clip_eval_transform"
_EXPECTED_OMICLIP_OUTPUT_DIM = 768
_SUPPORTED_OMICLIP_SCHEMA_VERSIONS = {1}
_HF_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# The six fields that identify one OmiCLIP tile-encoder build -- the
# OmiCLIP analogue of uni2_spot_cache.py's own `_PROVENANCE_FIELDS`.
_PROVENANCE_FIELDS = (
    "checkpoint_sha256", "pinned_revision", "package_version",
    "preprocessing_spec", "output_dim", "schema_version",
)


def validate_omiclip_tile_encoder_provenance(source: str, provenance: dict) -> None:
    """Fail-closed validation of a real OmiCLIP tile-encoder provenance
    dict (the same shape `FrozenOmiCLIPTileEncoder.identity`/
    `build_omiclip_spot_feature_cache` produces). Mirrors
    `uni2_spot_cache.validate_uni2_tile_encoder_provenance`'s discipline,
    applied to OmiCLIP's own field set -- `source` is a human-readable
    identifier (e.g. a file path) used only in error messages."""
    if not _HF_COMMIT_SHA_RE.match(str(provenance.get("pinned_revision", ""))):
        raise ValueError(
            f"{source} omiclip_pinned_revision={provenance.get('pinned_revision')!r} is not a full "
            "40-character lowercase hex Hugging Face commit SHA -- rebuild with a pinned, "
            "immutable --omiclip-pinned-revision"
        )
    if not _SHA256_HEX_RE.match(str(provenance.get("checkpoint_sha256", ""))):
        raise ValueError(
            f"{source} omiclip_checkpoint_sha256={provenance.get('checkpoint_sha256')!r} is not a "
            "well-formed 64-character lowercase hex sha256 digest"
        )
    if not str(provenance.get("package_version", "")).strip():
        raise ValueError(f"{source} omiclip_package_version must be nonblank")
    if provenance.get("preprocessing_spec") != _EXPECTED_OMICLIP_PREPROCESSING_SPEC:
        raise ValueError(
            f"{source} omiclip_preprocessing_spec={provenance.get('preprocessing_spec')!r}, expected "
            f"{_EXPECTED_OMICLIP_PREPROCESSING_SPEC!r}"
        )
    if int(provenance.get("output_dim", -1)) != _EXPECTED_OMICLIP_OUTPUT_DIM:
        raise ValueError(
            f"{source} omiclip_output_dim={provenance.get('output_dim')!r}, expected "
            f"{_EXPECTED_OMICLIP_OUTPUT_DIM}"
        )
    if int(provenance.get("schema_version", -1)) not in _SUPPORTED_OMICLIP_SCHEMA_VERSIONS:
        raise ValueError(
            f"{source} omiclip_schema_version={provenance.get('schema_version')!r} is not supported "
            f"(supported: {sorted(_SUPPORTED_OMICLIP_SCHEMA_VERSIONS)})"
        )


def require_consistent_omiclip_tile_encoder_provenance(
    provenance_by_source: dict[str, dict],
    expected_provenance: dict,
) -> None:
    """OmiCLIP analogue of
    `uni2_spot_cache.require_consistent_uni2_tile_encoder_provenance`:
    every provenance dict in `provenance_by_source` is individually
    validated, then required to be identical across every entry, then
    required to exactly match `expected_provenance` field-by-field for
    every field it specifies. `expected_provenance` is mandatory and must
    declare at least `pinned_revision`."""
    if not provenance_by_source:
        raise ValueError("require_consistent_omiclip_tile_encoder_provenance: no provenance entries given")
    if not expected_provenance or "pinned_revision" not in expected_provenance:
        raise ValueError(
            "require_consistent_omiclip_tile_encoder_provenance: expected_provenance must declare at "
            "least pinned_revision -- an omitted or empty expected_provenance would let every "
            "cache in this experiment consistently agree on the WRONG OmiCLIP checkpoint without "
            "this gate ever noticing"
        )
    for source, provenance in provenance_by_source.items():
        missing_fields = [field for field in _PROVENANCE_FIELDS if field not in provenance]
        if missing_fields:
            raise ValueError(
                f"OmiCLIP tile-encoder provenance for {source!r} is missing field(s) {missing_fields} "
                "-- refusing to preflight-check an incomplete provenance record"
            )
        validate_omiclip_tile_encoder_provenance(f"preflight entry {source!r}", provenance)

    reference_source, reference = next(iter(provenance_by_source.items()))
    for source, provenance in provenance_by_source.items():
        for field in _PROVENANCE_FIELDS:
            if provenance.get(field) != reference.get(field):
                raise ValueError(
                    f"OmiCLIP tile-encoder provenance mismatch: {source}.{field}={provenance.get(field)!r} "
                    f"!= {reference_source}.{field}={reference.get(field)!r} -- every spot-feature "
                    "cache used in one experiment must share the exact same OmiCLIP checkpoint identity"
                )
    for field in _PROVENANCE_FIELDS:
        if field not in expected_provenance:
            continue  # expected_provenance may deliberately pin only a subset of fields
        if reference.get(field) != expected_provenance[field]:
            raise ValueError(
                f"OmiCLIP tile-encoder provenance mismatch: every cache has {field}="
                f"{reference.get(field)!r}, but the experiment config declares an expected "
                f"{field}={expected_provenance[field]!r}"
            )


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    return Path(cache_root) / "omiclip_gen3_spot_cache" / f"{sample_id}.npz"


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


def build_omiclip_spot_feature_cache(
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
    an explicit zero instead, exactly matching the GigaPath/UNI2 caches'
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
            omiclip_checkpoint_sha256=np.asarray(encoder.identity.checkpoint_sha256),
            omiclip_pinned_revision=np.asarray(encoder.identity.pinned_revision),
            omiclip_package_version=np.asarray(encoder.identity.package_version),
            omiclip_preprocessing_spec=np.asarray(encoder.identity.preprocessing_spec),
            omiclip_output_dim=np.asarray(output_dim),
            omiclip_schema_version=np.asarray(1),
        )
    os.replace(tmp, path)
    return path


def load_omiclip_spot_features(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> dict:
    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"OmiCLIP spot-feature cache missing for {sample_id}: {path}. Build it with "
            "gen4.omiclip_spot_cache.build_omiclip_spot_feature_cache before training."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(f"OmiCLIP spot-feature cache {path} is missing fields {missing} -- rebuild it")

    real_barcodes = np.asarray([str(b) for b in barcodes])
    real_availability = np.asarray(image_source_available, dtype=bool)
    real_patches = np.asarray(patches)
    if real_patches.shape[0] != real_barcodes.shape[0] or real_availability.shape[0] != real_barcodes.shape[0]:
        raise ValueError(f"{sample_id}: barcodes/patches/image_source_available passed to load are not row-aligned")

    cached_barcodes = np.asarray([str(b) for b in cached["barcodes"]])
    if not np.array_equal(cached_barcodes, real_barcodes):
        raise ValueError(f"OmiCLIP spot-feature cache {path} barcode identity/order mismatch -- rebuild it")
    cached_availability = np.asarray(cached["image_source_available"], dtype=bool)
    if not np.array_equal(cached_availability, real_availability):
        raise ValueError(f"OmiCLIP spot-feature cache {path} image_source_available mismatch -- rebuild it")

    output_dim = int(np.asarray(cached["omiclip_output_dim"]).item())
    features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (real_barcodes.shape[0], output_dim):
        raise ValueError(f"OmiCLIP spot-feature cache {path} features has wrong shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"OmiCLIP spot-feature cache {path} contains non-finite values")

    unavailable_idx = np.flatnonzero(~real_availability)
    if unavailable_idx.size and not np.array_equal(features[unavailable_idx], np.zeros((unavailable_idx.size, output_dim), dtype=np.float32)):
        raise ValueError(f"OmiCLIP spot-feature cache {path} has a nonzero row for an unavailable spot -- corrupted")

    real_content_hash = _patch_content_sha256(real_barcodes, real_availability, real_patches)
    if real_content_hash != str(cached["patch_content_sha256"]):
        raise ValueError(f"OmiCLIP spot-feature cache {path} patch content mismatch -- rebuild it")

    provenance = {
        "checkpoint_sha256": str(cached["omiclip_checkpoint_sha256"]),
        "pinned_revision": str(cached["omiclip_pinned_revision"]),
        "package_version": str(cached["omiclip_package_version"]),
        "preprocessing_spec": str(cached["omiclip_preprocessing_spec"]),
        "output_dim": output_dim,
        "schema_version": int(np.asarray(cached["omiclip_schema_version"]).item()),
    }
    return {
        "features": features,
        "barcodes": real_barcodes,
        "image_source_available": real_availability,
        "provenance": provenance,
    }


def cfg_cache_root(cfg) -> Path:
    """Mirrors `uni2_spot_cache.cfg_cache_root`'s resolution exactly -- a
    distinct config key, falling back to the same shared
    `hest_cache_dir`/`hest_data_dir` default every other Gen3 spot-
    feature cache uses."""
    configured = cfg.data.get("gen3_omiclip_spot_feature_cache_dir")
    if configured:
        return Path(str(configured))
    cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
    return Path(str(cache_root))


def load_gen3_omiclip_spot_features(
    cfg,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> dict:
    """Config-driven wrapper around `load_omiclip_spot_features` -- mirrors
    `uni2_spot_cache.load_gen3_uni2_spot_features` exactly, renaming the
    `"provenance"` key to `"tile_encoder_provenance"` so
    `gen3_dataset.py::load_gen3_sample_data` can treat any encoder's
    loader result uniformly, and additionally requiring the cache's
    provenance to be well-formed real-OmiCLIP-shaped."""
    cache_root = cfg_cache_root(cfg)
    loaded = load_omiclip_spot_features(cache_root, sample_id, barcodes, patches, image_source_available)
    validate_omiclip_tile_encoder_provenance(
        f"Gen3 OmiCLIP spot-feature cache ({sample_id})", loaded["provenance"],
    )
    return {
        "features": loaded["features"],
        "barcodes": loaded["barcodes"],
        "image_source_available": loaded["image_source_available"],
        "tile_encoder_provenance": loaded["provenance"],
    }
