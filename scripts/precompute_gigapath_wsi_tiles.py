#!/usr/bin/env python3
"""Build dense, tissue-only Prov-GigaPath WSI tile caches.

Unlike the historical spot-patch cache, this cache represents the visible
whole slide.  It stores two coordinate systems deliberately:

* ``coords``: virtual target-MPP coordinates consumed by GigaPath LongNet;
* ``level0_coords``: raw WSI coordinates used to remove every tile whose
  pixels intersect a missing GEX/H&E query region.

The script fails closed when slide resolution is unavailable.  Guessing an
MPP would make image positions and missing-region masks incomparable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch


WSI_SUFFIXES = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")


def _resolve_wsi(root: Path, sample_id: str) -> Path:
    candidates = []
    for suffix in WSI_SUFFIXES:
        candidates.extend(root.glob(f"**/{sample_id}{suffix}"))
        candidates.extend(root.glob(f"**/{sample_id}{suffix.upper()}"))
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one WSI for {sample_id} below {root}, found {files}"
        )
    return files[0]


def _slide_mpp(slide) -> tuple[float, float]:
    properties = slide.properties
    x_keys = ("tiffslide.mpp-x", "openslide.mpp-x")
    y_keys = ("tiffslide.mpp-y", "openslide.mpp-y")
    x = next((properties.get(key) for key in x_keys if properties.get(key)), None)
    y = next((properties.get(key) for key in y_keys if properties.get(key)), None)
    if x is None or y is None:
        raise ValueError(
            "WSI has no explicit microns-per-pixel metadata; refusing to guess slide scale"
        )
    mpp_x, mpp_y = float(x), float(y)
    if not np.isfinite([mpp_x, mpp_y]).all() or min(mpp_x, mpp_y) <= 0:
        raise ValueError(f"invalid WSI MPP ({mpp_x}, {mpp_y})")
    if abs(mpp_x - mpp_y) / max(mpp_x, mpp_y) > 0.02:
        raise ValueError(f"anisotropic WSI pixels are not supported: ({mpp_x}, {mpp_y})")
    return mpp_x, mpp_y


def _is_tissue(tile: np.ndarray, min_tissue_fraction: float) -> bool:
    """Cheap conservative background rejection in RGB space."""
    rgb = tile.astype(np.float32) / 255.0
    channel_range = rgb.max(axis=2) - rgb.min(axis=2)
    not_white = rgb.mean(axis=2) < 0.92
    stained = channel_range > 0.05
    return float(np.mean(not_white & stained)) >= min_tissue_fraction


def _tile_grid(slide, source_span: int, output_size: int, min_tissue_fraction: float):
    width, height = map(int, slide.dimensions)
    for y in range(0, height - source_span + 1, source_span):
        for x in range(0, width - source_span + 1, source_span):
            region = slide.read_region((x, y), 0, (source_span, source_span)).convert("RGB")
            if source_span != output_size:
                region = region.resize((output_size, output_size), Image.Resampling.BILINEAR)
            tile = np.asarray(region, dtype=np.uint8)
            if _is_tissue(tile, min_tissue_fraction):
                yield x, y, tile


def _encode_batches(records, batch_size: int, device: str):
    from src.models.conditioning import (
        _gigapath_preprocess_and_encode,
        _load_gigapath_tile_encoder,
    )

    encoder = _load_gigapath_tile_encoder().to(device).eval()
    coords, features, batch = [], [], []
    with torch.inference_mode():
        for x, y, tile in records:
            coords.append((x, y))
            batch.append(tile)
            if len(batch) == batch_size:
                tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
                tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
                features.append(_gigapath_preprocess_and_encode(encoder, tensor).cpu().numpy())
                batch.clear()
        if batch:
            tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
            tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
            features.append(_gigapath_preprocess_and_encode(encoder, tensor).cpu().numpy())
    if not features:
        raise ValueError("no tissue tiles survived WSI background filtering")
    return np.asarray(coords, dtype=np.float32), np.concatenate(features).astype(np.float32)


def build_cache(cfg, sample_id: str, batch_size: int, target_mpp: float,
                min_tissue_fraction: float, device: str) -> Path:
    try:
        import tiffslide
    except Exception as exc:
        raise ImportError("dense WSI caching requires tiffslide") from exc

    root = Path(str(cfg.data.hest_data_dir))
    wsi_path = _resolve_wsi(root, sample_id)
    slide = tiffslide.TiffSlide(str(wsi_path))
    mpp_x, mpp_y = _slide_mpp(slide)
    source_span = int(round(256 * target_mpp / ((mpp_x + mpp_y) / 2.0)))
    if source_span < 32:
        raise ValueError(f"invalid source tile span {source_span} for {wsi_path}")
    virtual_dimensions = np.asarray(slide.dimensions) * np.asarray(
        [mpp_x / target_mpp, mpp_y / target_mpp]
    )
    if np.any(virtual_dimensions >= 256_000):
        raise ValueError(
            f"{sample_id} virtual dimensions {virtual_dimensions.tolist()} exceed "
            "GigaPath LongNet's 1000x1000 positional grid at 256px per tile"
        )
    print(
        f"{sample_id}: {wsi_path} dimensions={slide.dimensions} "
        f"mpp=({mpp_x:.4f},{mpp_y:.4f}) level0_span={source_span}",
        flush=True,
    )
    level0_coords, features = _encode_batches(
        _tile_grid(slide, source_span, 256, min_tissue_fraction), batch_size, device
    )
    # LongNet bins positions by its 256-pixel tile size.  Express positions
    # in the target-MPP frame even when the source WSI has a different MPP.
    coords = level0_coords * np.asarray([mpp_x / target_mpp, mpp_y / target_mpp])

    cache_dir = Path(str(cfg.data.get(
        "slide_context_cache_dir",
        Path(str(cfg.data.get("hest_cache_dir", root))) / "gigapath_slide_cache",
    )))
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{sample_id}.npz"
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        # Float embeddings are effectively incompressible; compression adds
        # substantial CPU time and load latency for negligible disk savings.
        np.savez(
            handle,
            features=features,
            coords=coords.astype(np.float32),
            level0_coords=level0_coords,
            tile_size=np.asarray(256.0, dtype=np.float32),
            level0_tile_size=np.asarray(float(source_span), dtype=np.float32),
            coords_are_centers=np.asarray(False),
            target_mpp=np.asarray(target_mpp, dtype=np.float32),
            source_mpp=np.asarray([mpp_x, mpp_y], dtype=np.float32),
            wsi_dimensions=np.asarray(slide.dimensions, dtype=np.int64),
            wsi_path=np.asarray(str(wsi_path)),
        )
    temporary.replace(output)
    print(f"{sample_id}: wrote {features.shape[0]} tiles to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.target_mpp <= 0 or not 0 <= args.min_tissue_fraction <= 1:
        raise ValueError("target-mpp must be positive and min-tissue-fraction must be in [0,1]")
    cfg = OmegaConf.load(args.config)
    ids = list(args.sample_ids or cfg.data.get("sample_ids", []))
    if not ids:
        ids = [str(cfg.data.sample_id)]
    for sample_id in ids:
        build_cache(
            cfg, str(sample_id), args.batch_size, args.target_mpp,
            args.min_tissue_fraction, args.device,
        )


if __name__ == "__main__":
    main()
