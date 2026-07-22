"""Mask-aware WSI tile-cache loading for hierarchical missing tissue."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def _cache_path(cfg, sample_id: str) -> Path:
    configured = cfg.data.get("slide_context_cache_dir")
    if configured:
        root = Path(str(configured))
    else:
        cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
        root = Path(str(cache_root)) / "gigapath_slide_cache"
    return root / f"{sample_id}.npz"


def load_slide_context(
    cfg,
    sample_id: str,
    spot_features: np.ndarray | None,
    spot_coords: np.ndarray,
) -> dict | None:
    """Load the configured slide tiles without silently changing semantics.

    ``dense_wsi_cache`` is the intended biological path: a tissue-wide,
    non-overlapping WSI tile grid produced by
    ``scripts/precompute_gigapath_wsi_tiles.py``.  ``spot_aligned`` is an
    explicit diagnostic fallback over the ST-covered patch lattice; it is
    never presented as complete WSI coverage.
    """
    source = str(cfg.data.get("slide_context_source", "disabled"))
    if source == "disabled":
        return None
    if source == "spot_aligned":
        if spot_features is None or spot_features.ndim != 2:
            raise ValueError(
                "slide_context_source=spot_aligned requires precomputed GigaPath spot features"
            )
        tile_size = float(cfg.data.get("spot_patch_size_fullres", 224.0))
        features = np.asarray(spot_features, dtype=np.float32)
        # HEST spot coordinates are patch centers; Prov-GigaPath expects
        # level-0 tile coordinates. Convert explicitly instead of silently
        # shifting its 2-D positional bins by half a patch.
        coords = np.asarray(spot_coords[:, :2], dtype=np.float32) - tile_size / 2.0
        mask_coords = coords
        mask_tile_size = tile_size
        coords_are_centers = False
        identity = f"{sample_id}:spot_aligned:{features.shape[0]}"
    elif source == "dense_wsi_cache":
        path = _cache_path(cfg, sample_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dense WSI GigaPath cache missing for {sample_id}: {path}. Run "
                "scripts/precompute_gigapath_wsi_tiles.py before training."
            )
        cached = np.load(path, allow_pickle=False)
        required = {"features", "coords", "tile_size", "coords_are_centers"}
        missing = sorted(required.difference(cached.files))
        if missing:
            raise ValueError(f"slide cache {path} is missing fields {missing}")
        features = np.asarray(cached["features"], dtype=np.float32)
        coords = np.asarray(cached["coords"], dtype=np.float32)
        tile_size = float(np.asarray(cached["tile_size"]).item())
        coords_are_centers = bool(np.asarray(cached["coords_are_centers"]).item())
        # ``coords`` may be expressed in the 0.5-um/px virtual coordinate
        # system expected by GigaPath.  Hole filtering must instead happen
        # in the HEST level-0 coordinate frame used by spot coordinates.
        mask_coords = np.asarray(
            cached["level0_coords"] if "level0_coords" in cached.files else coords,
            dtype=np.float32,
        )
        mask_tile_size = float(
            np.asarray(
                cached["level0_tile_size"]
                if "level0_tile_size" in cached.files else cached["tile_size"]
            ).item()
        )
        wsi_dimensions = (
            np.asarray(cached["wsi_dimensions"], dtype=np.float64)
            if "wsi_dimensions" in cached.files else None
        )
        stat = path.stat()
        identity = f"{sample_id}:dense:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    else:
        raise ValueError(
            "data.slide_context_source must be disabled, spot_aligned, or dense_wsi_cache"
        )

    if features.ndim != 2 or features.shape[1] != 1536:
        raise ValueError(
            f"slide context features must be [N,1536], got {features.shape} for {sample_id}"
        )
    if coords.shape != (features.shape[0], 2):
        raise ValueError(
            f"slide context coords must be [N,2] aligned with features, got {coords.shape}"
        )
    if mask_coords.shape != (features.shape[0], 2):
        raise ValueError(
            f"slide context level0_coords must be [N,2] aligned with features, got "
            f"{mask_coords.shape}"
        )
    if features.shape[0] < 1 or tile_size <= 0:
        raise ValueError(f"slide context for {sample_id} is empty or has invalid tile_size")
    if mask_tile_size <= 0:
        raise ValueError(f"slide context for {sample_id} has invalid level0_tile_size")
    if not np.isfinite(features).all() or not np.isfinite(coords).all() or not np.isfinite(mask_coords).all():
        raise ValueError(f"slide context for {sample_id} contains non-finite values")
    if source == "dense_wsi_cache":
        spot_xy = np.asarray(spot_coords[:, :2], dtype=np.float64)
        if wsi_dimensions is not None:
            if wsi_dimensions.shape != (2,) or np.any(wsi_dimensions <= 0):
                raise ValueError(f"slide cache for {sample_id} has invalid wsi_dimensions")
            inside = (
                (spot_xy[:, 0] >= 0) & (spot_xy[:, 0] < wsi_dimensions[0])
                & (spot_xy[:, 1] >= 0) & (spot_xy[:, 1] < wsi_dimensions[1])
            )
            if not bool(inside.all()):
                raise ValueError(
                    f"{(~inside).sum()}/{len(inside)} ST spots fall outside the cached WSI; "
                    "the H5AD and WSI coordinate frames do not match"
                )
        # Most measured spots must fall in, or immediately beside, a retained
        # tissue tile.  This catches the far more dangerous case where both
        # arrays have plausible positive coordinates but refer to different
        # crops/origins of the slide.
        tile_bins = {
            (int(np.floor(x / mask_tile_size)), int(np.floor(y / mask_tile_size)))
            for x, y in mask_coords
        }
        covered = []
        for x, y in spot_xy:
            bx, by = int(np.floor(x / mask_tile_size)), int(np.floor(y / mask_tile_size))
            covered.append(any(
                (bx + dx, by + dy) in tile_bins
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            ))
        coverage = float(np.mean(covered))
        if coverage < 0.90:
            raise ValueError(
                f"only {coverage:.1%} of {sample_id} ST spots align near retained WSI tissue "
                "tiles; refusing a likely mismatched WSI/H5AD coordinate frame"
            )
    return {
        "features": features,
        "coords": coords,
        "mask_coords": mask_coords,
        "tile_size": tile_size,
        "mask_tile_size": mask_tile_size,
        "coords_are_centers": coords_are_centers,
        "context_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "source": source,
    }


def _overlaps_query_hole(
    tile_centers: np.ndarray,
    tile_half_size: float,
    query_centers: np.ndarray,
    query_half_size: float,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Conservative axis-aligned tile/target-patch intersection test."""
    overlap = np.zeros(tile_centers.shape[0], dtype=bool)
    limit = float(tile_half_size + query_half_size)
    for start in range(0, tile_centers.shape[0], chunk_size):
        block = tile_centers[start : start + chunk_size]
        delta = np.abs(block[:, None, :] - query_centers[None, :, :])
        overlap[start : start + len(block)] = np.any(
            (delta[..., 0] < limit) & (delta[..., 1] < limit), axis=1
        )
    return overlap


def visible_slide_context(
    slide_context: dict | None,
    query_coords: np.ndarray,
    image_mode: str,
    query_patch_size: float,
) -> dict:
    """Return only WSI tiles that could still exist after physical damage."""
    if slide_context is None or image_mode == "all_zero":
        return {"available": False}
    features = slide_context["features"]
    coords = slide_context["coords"]
    mask_coords = slide_context["mask_coords"]
    tile_size = float(slide_context["tile_size"])
    mask_tile_size = float(slide_context["mask_tile_size"])
    if bool(slide_context["coords_are_centers"]):
        mask_centers = mask_coords
    else:
        mask_centers = mask_coords + mask_tile_size / 2.0

    visible = np.ones(features.shape[0], dtype=bool)
    if image_mode == "target_zero":
        query_xy = np.asarray(query_coords[:, :2], dtype=np.float32)
        visible &= ~_overlaps_query_hole(
            mask_centers, mask_tile_size / 2.0, query_xy, float(query_patch_size) / 2.0
        )
    elif image_mode not in {"full", "shuffled"}:
        raise ValueError(f"unsupported slide image mode {image_mode!r}")
    if not visible.any():
        raise ValueError("physical missing-tissue mask removed every WSI context tile")
    return {
        "available": True,
        "features": features[visible],
        # GigaPath expects level-0 tile coordinates. Preserve the cached
        # convention (top-left for dense WSI, centers for spot fallback).
        "coords": coords[visible],
        "context_id": f"{slide_context['context_id']}:{image_mode}",
        "n_total": int(features.shape[0]),
        "n_visible": int(visible.sum()),
    }


def nonoverlapping_context_patch_mask(
    context_coords: np.ndarray,
    query_coords: np.ndarray,
    patch_size: float,
) -> np.ndarray:
    """Context spot patches whose raw pixels do not intersect the hole."""
    return ~_overlaps_query_hole(
        np.asarray(context_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
        np.asarray(query_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
    )
