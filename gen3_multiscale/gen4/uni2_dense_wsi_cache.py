"""UNI2 dense-WSI tile-feature cache -- Codex audit finding (Item 1):
"no real UNI2 dense-WSI cache builder exists yet" (see gen4/inputs.py's
`wsi_tile_feature_provenance` docstring). Arms A/C's
`global_context_source="uni2_pool"` need a real, per-sample, tissue-wide
UNI2 tile grid -- the SAME physical tiling GigaPath's own dense_wsi_cache
uses (`scripts/precompute_gigapath_wsi_tiles.py` /
`data/slide_context.py`), only with UNI2's per-tile encoder run instead
of GigaPath's.

Design: this module owns UNI2's own on-disk cache directory and
provenance schema (mirrors `gen4/uni2_spot_cache.py`'s convention --
`uni2_*` fields, never GigaPath's `tile_encoder_hf_*` schema, since UNI2
is loaded from a pinned LOCAL checkpoint file, not a Hugging Face repo
id -- see `gen4/uni2_encoder.py::FrozenUNI2TileEncoder`). It does NOT
reimplement WSI reading/tiling/tissue-detection or the hole-overlap
masking test: `build_uni2_dense_wsi_cache` reuses
`scripts/precompute_gigapath_wsi_tiles.py`'s already-audited
`_resolve_wsi`/`_open_slide`/`_slide_mpp`/`_tile_grid` helpers verbatim
(imported, not copied), and `load_uni2_dense_wsi_context`'s return dict
is shaped EXACTLY like `data.slide_context.load_slide_context`'s own
return value, so it plugs into `example_builder.build_spatial_field_example`'s
existing `slide_context=` parameter unmodified -- `visible_slide_context`/
`tile_centers` (the real, audited hole-overlap and coordinate-frame
logic) run completely unchanged, on a cache with a different encoder's
features. Shape/finiteness and dense-tile-geometry checks are the exact
same real checks GigaPath's own dense_wsi_cache path runs
(`data.slide_context.validate_slide_context_arrays`/
`validate_dense_wsi_tile_geometry`, factored out of `load_slide_context`
for exactly this kind of second real consumer -- never reimplemented
here).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gen3_multiscale.data.slide_context import validate_dense_wsi_tile_geometry, validate_slide_context_arrays

_REQUIRED_FIELDS = {
    "features", "coords", "level0_coords", "tile_size", "level0_tile_size", "coords_are_centers",
    "wsi_dimensions", "uni2_checkpoint_sha256", "uni2_pinned_revision", "uni2_package_version",
    "uni2_preprocessing_spec", "uni2_output_dim", "uni2_schema_version",
}


def _cache_path(cfg, sample_id: str) -> Path:
    configured = cfg.data.get("uni2_slide_context_cache_dir")
    if configured:
        root = Path(str(configured))
    else:
        cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
        root = Path(str(cache_root)) / "uni2_dense_wsi_cache"
    return root / f"{sample_id}.npz"


def build_uni2_dense_wsi_cache(
    cfg, sample_id: str, encoder, batch_size: int = 32, target_mpp: float = 0.5,
    min_tissue_fraction: float = 0.10, device: str = "cpu",
) -> Path:
    """`encoder` must satisfy `gen4.providers.ImageContextProvider`
    (`encoder.identity`, `encoder.encode_available_patches`) -- pass a
    real `FrozenUNI2TileEncoder` (gen4/uni2_encoder.py) in production.
    Tiling/tissue-detection reuses
    `scripts/precompute_gigapath_wsi_tiles.py`'s own helpers verbatim, so
    a UNI2 cache and a GigaPath cache built from the same config always
    tile the identical slide region the identical way -- only the
    per-tile encoder differs."""
    import importlib.util
    import sys

    root = Path(str(cfg.data.hest_data_dir))
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "precompute_gigapath_wsi_tiles.py"
    spec = importlib.util.spec_from_file_location("_precompute_gigapath_wsi_tiles", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)

    wsi_path = module._resolve_wsi(root, sample_id)
    slide, _backend = module._open_slide(wsi_path)
    try:
        mpp_x, mpp_y = module._slide_mpp(slide)
        dimensions = tuple(map(int, slide.dimensions))
        source_span = int(round(256 * target_mpp / ((mpp_x + mpp_y) / 2.0)))
        if source_span < 32:
            raise ValueError(f"invalid source tile span {source_span} for {wsi_path}")
        coords_batch, patches_batch = [], []
        level0_coords_list, features_list = [], []
        for x, y, tile in module._tile_grid(slide, source_span, 256, min_tissue_fraction):
            coords_batch.append((x, y))
            patches_batch.append(tile)
            if len(patches_batch) == batch_size:
                level0_coords_list.append(np.asarray(coords_batch, dtype=np.float32))
                features_list.append(encoder.encode_available_patches(np.stack(patches_batch)))
                coords_batch, patches_batch = [], []
        if patches_batch:
            level0_coords_list.append(np.asarray(coords_batch, dtype=np.float32))
            features_list.append(encoder.encode_available_patches(np.stack(patches_batch)))
    finally:
        slide.close()
    if not features_list:
        raise ValueError(f"no tissue tiles survived WSI background filtering for {sample_id}")

    level0_coords = np.concatenate(level0_coords_list).astype(np.float32)
    features = np.concatenate(features_list).astype(np.float32)
    if features.shape != (level0_coords.shape[0], encoder.identity.output_dim):
        raise ValueError(
            f"{sample_id}: UNI2 encoder returned shape {features.shape}, expected "
            f"({level0_coords.shape[0]}, {encoder.identity.output_dim})"
        )
    coords = level0_coords * np.asarray([mpp_x / target_mpp, mpp_y / target_mpp], dtype=np.float32)

    path = _cache_path(cfg, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            features=features, coords=coords, level0_coords=level0_coords,
            tile_size=np.asarray(256.0, dtype=np.float32),
            level0_tile_size=np.asarray(float(source_span), dtype=np.float32),
            coords_are_centers=np.asarray(False), wsi_dimensions=np.asarray(dimensions, dtype=np.int64),
            uni2_checkpoint_sha256=np.asarray(encoder.identity.checkpoint_sha256),
            uni2_pinned_revision=np.asarray(encoder.identity.pinned_revision),
            uni2_package_version=np.asarray(encoder.identity.package_version),
            uni2_preprocessing_spec=np.asarray(encoder.identity.preprocessing_spec),
            uni2_output_dim=np.asarray(encoder.identity.output_dim),
            uni2_schema_version=np.asarray(1),
        )
    tmp.replace(path)
    return path


def load_uni2_dense_wsi_context(cfg, sample_id: str, spot_coords: np.ndarray) -> dict:
    """Returns a dict shaped identically to
    `data.slide_context.load_slide_context`'s own return value (see
    module docstring) -- pass it straight through as
    `build_gen4_spatial_field_example`'s `slide_context=` argument, and
    set `wsi_tile_feature_provenance="uni2"` on the result (Gen4's own
    field, not part of the base slide_context dict)."""
    path = _cache_path(cfg, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"UNI2 dense-WSI cache missing for {sample_id}: {path}. Build it with "
            "gen4.uni2_dense_wsi_cache.build_uni2_dense_wsi_cache before training."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(f"UNI2 dense-WSI cache {path} is missing fields {missing} -- rebuild it")

    features = np.asarray(cached["features"], dtype=np.float32)
    coords = np.asarray(cached["coords"], dtype=np.float32)
    level0_coords = np.asarray(cached["level0_coords"], dtype=np.float32)
    tile_size = float(np.asarray(cached["tile_size"]).item())
    level0_tile_size = float(np.asarray(cached["level0_tile_size"]).item())
    coords_are_centers = bool(np.asarray(cached["coords_are_centers"]).item())
    wsi_dimensions = np.asarray(cached["wsi_dimensions"], dtype=np.float64)

    tile_encoder_provenance = {
        "uni2_checkpoint_sha256": str(cached["uni2_checkpoint_sha256"]),
        "uni2_pinned_revision": str(cached["uni2_pinned_revision"]),
        "uni2_package_version": str(cached["uni2_package_version"]),
        "uni2_preprocessing_spec": str(cached["uni2_preprocessing_spec"]),
        "uni2_output_dim": int(np.asarray(cached["uni2_output_dim"]).item()),
        "uni2_schema_version": int(np.asarray(cached["uni2_schema_version"]).item()),
    }
    validate_slide_context_arrays(sample_id, features, coords, level0_coords, tile_size, level0_tile_size)
    validate_dense_wsi_tile_geometry(
        sample_id, coords, level0_coords, level0_tile_size, spot_coords,
        wsi_dimensions=wsi_dimensions, cache_label=f"UNI2 dense-WSI cache {path}",
    )
    content_digest = hashlib.sha256()
    content_digest.update(np.ascontiguousarray(features).tobytes())
    content_digest.update(np.ascontiguousarray(coords).tobytes())
    content_digest.update(np.ascontiguousarray(level0_coords).tobytes())
    content_digest.update(str(tile_size).encode())
    content_digest.update(str(level0_tile_size).encode())
    content_digest.update(str(coords_are_centers).encode())
    content_digest.update(json.dumps(tile_encoder_provenance, sort_keys=True).encode())
    identity = f"{sample_id}:uni2dense:{content_digest.hexdigest()}"

    return {
        "features": features,
        "coords": coords,
        "mask_coords": level0_coords,
        "tile_size": tile_size,
        "mask_tile_size": level0_tile_size,
        "coords_are_centers": coords_are_centers,
        "context_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "source": "uni2_dense_wsi_cache",
        "tile_encoder_provenance": tile_encoder_provenance,
    }
