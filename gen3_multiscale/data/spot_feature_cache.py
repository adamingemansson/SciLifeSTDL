"""Gen3-specific, manifest-driven GigaPath spot-feature cache/provider.

20th Codex re-audit (Step 5 Part 2 -> Step 6 boundaries, response to the
19th re-audit's confirmed-closed dense-WSI launch blocker): the legacy
``scripts/precompute_gigapath_samples.py`` spot-feature cache
(``src.training.train.get_gigapath_features``) is NOT a valid Gen3
input -- it calls ``_load_gigapath_tile_encoder()`` with no ``revision``
(the same unpinned default path STPath/Gen2 callers still legitimately
use), and its on-disk cache format carries no tile-encoder provenance at
all, so it cannot be bound into ``load_slide_context``'s now-mandatory
provenance discipline. This module is Gen3's OWN independent spot-
feature cache: one real encode pass per manifest sample, a MANDATORY
immutable tile-encoder revision, the full provenance object plus the
exact aligned barcode/availability contract bound into every cache file,
and strict validation on load -- mirroring
``gen3_multiscale/data/slide_context.py``'s ``dense_wsi_cache``
discipline for the per-spot feature path instead of the dense-WSI-tile
path. Deliberately a DIFFERENT on-disk directory
(``gigapath_gen3_spot_cache/``) than both the legacy cache
(``gigapath_cache/``) and the dense WSI cache (``gigapath_slide_cache/``)
so this module can never accidentally load either.

Never call the tile encoder per training example (Adam's explicit Step 6
requirement, recorded in CONTRACT.md section 44): this module's entire
purpose is precompute-once/load-many. The real trainer's
``image_feature_fn`` should become a slice into
``load_gen3_spot_features(...)["features"]``, never a live GigaPath
forward pass -- re-encoding nearly-identical context patches on every
masking draw would be both slow and pointless, since the frozen
encoder's output for a given patch never changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gen3_multiscale.data.slide_context import validate_tile_encoder_provenance

# Prov-GigaPath's real tile-encoder output dim -- must match
# src.models.conditioning._GIGAPATH_FEAT_DIM. Imported lazily inside the
# functions below (not at module import time) for the same reason every
# other GigaPath-touching function in this codebase does: importing
# src.models.conditioning pulls in torch at import time, which most
# callers of THIS module (e.g. dataset_manifest.py-adjacent tooling) have
# no other reason to need eagerly.
_GIGAPATH_FEAT_DIM = 1536

_REQUIRED_FIELDS = {
    "features", "barcodes", "image_source_available", "patch_content_sha256",
    "tile_encoder_hf_repo_id", "tile_encoder_hf_revision", "tile_encoder_timm_version",
    "tile_encoder_preprocessing_spec", "tile_encoder_state_dict_sha256", "tile_encoder_schema_version",
}


def _cache_path(cfg, sample_id: str) -> Path:
    configured = cfg.data.get("gen3_spot_feature_cache_dir")
    if configured:
        root = Path(str(configured))
    else:
        cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
        # Deliberately distinct from both "gigapath_cache" (the legacy,
        # unpinned src.training.train cache) and "gigapath_slide_cache"
        # (the dense WSI cache) -- see module docstring.
        root = Path(str(cache_root)) / "gigapath_gen3_spot_cache"
    return root / f"{sample_id}.npz"


def _patch_content_sha256(
    barcodes: np.ndarray, image_source_available: np.ndarray, patches: np.ndarray,
) -> str:
    """Real content hash of exactly what a build actually encoded --
    barcode order, the availability mask, and the pixel bytes of every
    AVAILABLE patch (never the zero-placeholder rows, which carry no
    real information and would make two samples with different numbers
    of real patches but identical placeholders collide). Recomputed on
    load from whatever real, already-loaded barcodes/availability/
    patches the caller passes in -- this is how "validate patch content
    on load" is satisfied without a second disk read: the caller (e.g.
    the real trainer's dataset construction) already loaded these via
    ``example_builder.load_sample_for_examples`` for other reasons."""
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    available_idx = np.flatnonzero(image_source_available)
    digest = hashlib.sha256()
    digest.update(b"\x1f".join(b.encode("utf-8") for b in barcodes))
    digest.update(image_source_available.tobytes())
    digest.update(np.ascontiguousarray(patches[available_idx]).tobytes())
    return digest.hexdigest()


def build_gen3_spot_feature_cache(
    cfg,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
    tile_encoder_revision: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> Path:
    """Encode every AVAILABLE H&E patch for one manifest sample exactly
    once, with a MANDATORY immutable tile-encoder revision, and cache the
    result keyed to this sample's real barcode order and patch content.

    ``barcodes``/``patches``/``image_source_available``: the ALIGNED
    triple ``loaders.align_patches_to_adata`` (via
    ``example_builder.load_sample_for_examples``) already produces --
    every manifest-declared spot, in ``adata.obs_names`` order, with a
    zero-placeholder row (and ``image_source_available=False``) for a
    spot with no matching H&E patch. Placeholder rows are NEVER fed to
    the tile encoder (the same discipline Step 5 Part 2 launch blocker
    #1 established: "never pass unavailable patches to GigaPath") --
    their cached feature row is an explicit zero vector instead, exactly
    matching what ``build_spatial_field_example`` already does for a
    context spot with ``observed_image_available=False``.
    """
    from src.models.conditioning import (
        _gigapath_preprocess_and_encode, _load_gigapath_tile_encoder,
        _validate_immutable_hf_revision, gigapath_tile_encoder_provenance,
    )
    import torch

    tile_encoder_revision = _validate_immutable_hf_revision(tile_encoder_revision)
    barcodes = np.asarray([str(b) for b in barcodes])
    image_source_available = np.asarray(image_source_available, dtype=bool)
    patches = np.asarray(patches)
    n = barcodes.shape[0]
    if patches.shape[0] != n or image_source_available.shape[0] != n:
        raise ValueError(
            f"{sample_id}: barcodes ({n}), patches ({patches.shape[0]}), and "
            f"image_source_available ({image_source_available.shape[0]}) must all be the same "
            "length and row-aligned, exactly as load_sample_for_examples returns them together"
        )
    unique_barcodes, counts = np.unique(barcodes, return_counts=True)
    duplicated = unique_barcodes[counts > 1]
    if duplicated.size:
        raise ValueError(
            f"{sample_id}: barcodes contain {duplicated.size} duplicate value(s) (examples: "
            f"{duplicated[:5].tolist()}) -- refusing to build an ambiguous cache"
        )

    encoder = _load_gigapath_tile_encoder(revision=tile_encoder_revision).to(device).eval()
    provenance = gigapath_tile_encoder_provenance(encoder, revision=tile_encoder_revision)

    features = np.zeros((n, _GIGAPATH_FEAT_DIM), dtype=np.float32)
    available_idx = np.flatnonzero(image_source_available)
    with torch.inference_mode():
        for start in range(0, available_idx.shape[0], batch_size):
            batch_idx = available_idx[start:start + batch_size]
            tensor = torch.from_numpy(patches[batch_idx]).permute(0, 3, 1, 2)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            out = _gigapath_preprocess_and_encode(encoder, tensor).cpu().numpy()
            features[batch_idx] = out.astype(np.float32)

    path = _cache_path(cfg, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            features=features,
            barcodes=barcodes,
            image_source_available=image_source_available,
            patch_content_sha256=np.asarray(
                _patch_content_sha256(barcodes, image_source_available, patches)
            ),
            tile_encoder_hf_repo_id=np.asarray(provenance["hf_repo_id"]),
            tile_encoder_hf_revision=np.asarray(provenance["hf_revision"]),
            tile_encoder_timm_version=np.asarray(str(provenance["timm_version"])),
            tile_encoder_preprocessing_spec=np.asarray(provenance["preprocessing_spec"]),
            tile_encoder_state_dict_sha256=np.asarray(provenance["state_dict_sha256"]),
            tile_encoder_schema_version=np.asarray(provenance["schema_version"]),
        )
    tmp.replace(path)
    print(f"{sample_id}: wrote Gen3 spot-feature cache for {available_idx.shape[0]}/{n} "
          f"available spots to {path}", flush=True)
    return path


def load_gen3_spot_features(
    cfg,
    sample_id: str,
    barcodes: np.ndarray,
    patches: np.ndarray,
    image_source_available: np.ndarray,
) -> dict:
    """Load and strictly validate one sample's Gen3 spot-feature cache
    against the ALREADY-LOADED real ``barcodes``/``patches``/
    ``image_source_available`` the caller has in memory (from
    ``example_builder.load_sample_for_examples``) -- never a separate
    disk re-read, so patch-content validation is essentially free.

    Validates: every required field present (fails closed on an old-
    format or legacy cache); the full tile-encoder provenance (via the
    same ``validate_tile_encoder_provenance`` ``load_slide_context``
    uses); barcode identity AND order; ``image_source_available``
    identity; ``features`` shape and finiteness; and real patch content
    (a cache built from since-changed patches on disk is rejected, not
    silently trusted).
    """
    path = _cache_path(cfg, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gen3 spot-feature cache missing for {sample_id}: {path}. Run "
            "scripts/precompute_gen3_spot_features.py before training. The legacy "
            "scripts/precompute_gigapath_samples.py cache is NOT a valid substitute -- it uses "
            "an unpinned tile-encoder revision and carries no provenance."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(
            f"Gen3 spot-feature cache {path} is missing fields {missing} -- rebuild it with the "
            "current scripts/precompute_gen3_spot_features.py"
        )

    provenance = {
        "hf_repo_id": str(cached["tile_encoder_hf_repo_id"]),
        "hf_revision": str(cached["tile_encoder_hf_revision"]),
        "timm_version": str(cached["tile_encoder_timm_version"]),
        "preprocessing_spec": str(cached["tile_encoder_preprocessing_spec"]),
        "state_dict_sha256": str(cached["tile_encoder_state_dict_sha256"]),
        "schema_version": int(np.asarray(cached["tile_encoder_schema_version"]).item()),
    }
    validate_tile_encoder_provenance(f"Gen3 spot-feature cache {path}", provenance)

    real_barcodes = np.asarray([str(b) for b in barcodes])
    real_availability = np.asarray(image_source_available, dtype=bool)
    real_patches = np.asarray(patches)
    if real_patches.shape[0] != real_barcodes.shape[0] or real_availability.shape[0] != real_barcodes.shape[0]:
        raise ValueError(
            f"{sample_id}: barcodes/patches/image_source_available passed to "
            "load_gen3_spot_features are not row-aligned"
        )

    cached_barcodes = np.asarray([str(b) for b in cached["barcodes"]])
    if cached_barcodes.shape[0] != real_barcodes.shape[0] or not np.array_equal(cached_barcodes, real_barcodes):
        raise ValueError(
            f"Gen3 spot-feature cache {path} barcode identity/order does not match the real, "
            f"currently-aligned sample data for {sample_id} -- rebuild it"
        )

    cached_availability = np.asarray(cached["image_source_available"], dtype=bool)
    if cached_availability.shape != real_availability.shape or not np.array_equal(cached_availability, real_availability):
        raise ValueError(
            f"Gen3 spot-feature cache {path} image_source_available does not match the real, "
            f"currently-aligned sample data for {sample_id} -- rebuild it"
        )

    features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (real_barcodes.shape[0], _GIGAPATH_FEAT_DIM):
        raise ValueError(
            f"Gen3 spot-feature cache {path} features has shape {features.shape}, expected "
            f"({real_barcodes.shape[0]}, {_GIGAPATH_FEAT_DIM})"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"Gen3 spot-feature cache {path} contains non-finite feature values")

    real_patch_content_sha256 = _patch_content_sha256(real_barcodes, real_availability, real_patches)
    cached_patch_content_sha256 = str(cached["patch_content_sha256"])
    if real_patch_content_sha256 != cached_patch_content_sha256:
        raise ValueError(
            f"Gen3 spot-feature cache {path} patch_content_sha256 does not match the real, "
            f"currently-loaded H&E patches for {sample_id} -- the patches on disk (or the "
            "barcode/availability alignment) changed since this cache was built; rebuild it"
        )

    return {
        "features": features,
        "barcodes": real_barcodes,
        "image_source_available": real_availability,
        "tile_encoder_provenance": provenance,
    }
