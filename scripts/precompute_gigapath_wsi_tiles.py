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
from fractions import Fraction
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


def _property_float(value) -> float | None:
    """Parse decimal or TIFF rational property values without guessing."""
    if value is None:
        return None
    try:
        return float(Fraction(str(value).strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _resolution_unit_um(value) -> float | None:
    """Return microns per TIFF resolution unit."""
    normalized = str(value).strip().lower()
    if normalized in {"2", "inch", "inches"}:
        return 25_400.0
    if normalized in {"3", "centimeter", "centimeters", "cm"}:
        return 10_000.0
    return None


def _slide_mpp(slide) -> tuple[float, float]:
    properties = slide.properties
    x_keys = ("tiffslide.mpp-x", "openslide.mpp-x")
    y_keys = ("tiffslide.mpp-y", "openslide.mpp-y")
    x = next((properties.get(key) for key in x_keys if properties.get(key)), None)
    y = next((properties.get(key) for key in y_keys if properties.get(key)), None)
    # OpenSlide's generic-TIFF backend exposes the source TIFF resolution
    # even when it does not synthesize openslide.mpp-x/y.  HEST's TIFFs use
    # pixels/cm (for example INT1 is 21889.4 px/cm), so this is still explicit
    # slide metadata rather than an inferred/default scale.
    if x is None or y is None:
        unit_um = _resolution_unit_um(
            properties.get("tiff.ResolutionUnit")
            or properties.get("tiffslide.resolution-unit")
        )
        x_resolution = _property_float(
            properties.get("tiff.XResolution")
            or properties.get("tiffslide.x-resolution")
        )
        y_resolution = _property_float(
            properties.get("tiff.YResolution")
            or properties.get("tiffslide.y-resolution")
        )
        if unit_um is not None and x_resolution and y_resolution:
            x = unit_um / x_resolution
            y = unit_um / y_resolution
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


def _open_slide(path: Path):
    """Open a WSI with independent backends and validate lazy metadata.

    HEST's plain pyramidal tiled TIFFs can trigger ``incompatible keyframe``
    in tifffile's shaped-series inference through TiffSlide.  OpenSlide has a
    dedicated generic tiled-TIFF backend and does not use that inference.
    Keep TiffSlide as a fallback for formats that it handles successfully.
    """
    errors: list[str] = []
    slide = None
    try:
        import openslide

        slide = openslide.OpenSlide(str(path))
        # Force lazy failures here so a broken backend is never returned.
        _ = slide.dimensions
        _ = slide.properties
        return slide, "openslide"
    except Exception as exc:
        errors.append(f"OpenSlide: {type(exc).__name__}: {exc}")
        if slide is not None:
            slide.close()

    slide = None
    try:
        import tiffslide

        slide = tiffslide.TiffSlide(str(path))
        _ = slide.dimensions
        _ = slide.properties
        return slide, "tiffslide"
    except Exception as exc:
        errors.append(f"TiffSlide: {type(exc).__name__}: {exc}")
        if slide is not None:
            slide.close()

    detail = "\n  - ".join(errors)
    raise RuntimeError(
        f"no WSI backend could open {path}:\n  - {detail}\n"
        "For HEST pyramidal TIFFs install the official OpenSlide backend with "
        "`python3 -m pip install openslide-bin openslide-python`."
    )


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


def _encode_batches(records, batch_size: int, device: str, tile_encoder_revision: str):
    from src.models.conditioning import (
        _gigapath_preprocess_and_encode,
        _load_gigapath_tile_encoder,
        _validate_immutable_hf_revision,
        gigapath_tile_encoder_provenance,
    )

    tile_encoder_revision = _validate_immutable_hf_revision(tile_encoder_revision)

    encoder = _load_gigapath_tile_encoder(revision=tile_encoder_revision).to(device).eval()
    # 18th Codex re-audit (Step 5 Part 2 launch blocker #2): record the
    # REAL tile-encoder identity actually loaded (never assumed from the
    # revision string alone) -- distinguishes this cache from one built
    # with different tile-encoder weights/preprocessing, which would
    # otherwise both appear equally valid.
    provenance = gigapath_tile_encoder_provenance(encoder, revision=tile_encoder_revision)
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
    return np.asarray(coords, dtype=np.float32), np.concatenate(features).astype(np.float32), provenance


def build_cache(cfg, sample_id: str, batch_size: int, target_mpp: float,
                min_tissue_fraction: float, device: str, tile_encoder_revision: str) -> Path:
    root = Path(str(cfg.data.hest_data_dir))
    wsi_path = _resolve_wsi(root, sample_id)
    slide, backend = _open_slide(wsi_path)
    try:
        mpp_x, mpp_y = _slide_mpp(slide)
        dimensions = tuple(map(int, slide.dimensions))
        source_span = int(round(256 * target_mpp / ((mpp_x + mpp_y) / 2.0)))
        if source_span < 32:
            raise ValueError(f"invalid source tile span {source_span} for {wsi_path}")
        virtual_dimensions = np.asarray(dimensions) * np.asarray(
            [mpp_x / target_mpp, mpp_y / target_mpp]
        )
        if np.any(virtual_dimensions >= 256_000):
            raise ValueError(
                f"{sample_id} virtual dimensions {virtual_dimensions.tolist()} exceed "
                "GigaPath LongNet's 1000x1000 positional grid at 256px per tile"
            )
        print(
            f"{sample_id}: {wsi_path} reader={backend} dimensions={dimensions} "
            f"mpp=({mpp_x:.4f},{mpp_y:.4f}) level0_span={source_span}",
            flush=True,
        )
        level0_coords, features, tile_encoder_provenance = _encode_batches(
            _tile_grid(slide, source_span, 256, min_tissue_fraction), batch_size, device,
            tile_encoder_revision=tile_encoder_revision,
        )
    finally:
        slide.close()
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
            wsi_dimensions=np.asarray(dimensions, dtype=np.int64),
            wsi_path=np.asarray(str(wsi_path)),
            # 18th Codex re-audit (Step 5 Part 2 launch blocker #2): the
            # real tile-encoder identity, distinguished from
            # FrozenGigaPathSlideEncoder's separate LongNet checkpoint
            # SHA256 -- see gigapath_tile_encoder_provenance's own
            # docstring (src/models/conditioning.py).
            tile_encoder_hf_repo_id=np.asarray(tile_encoder_provenance["hf_repo_id"]),
            tile_encoder_hf_revision=np.asarray(str(tile_encoder_provenance["hf_revision"])),
            tile_encoder_timm_version=np.asarray(str(tile_encoder_provenance["timm_version"])),
            tile_encoder_preprocessing_spec=np.asarray(tile_encoder_provenance["preprocessing_spec"]),
            tile_encoder_state_dict_sha256=np.asarray(tile_encoder_provenance["state_dict_sha256"]),
            tile_encoder_schema_version=np.asarray(tile_encoder_provenance["schema_version"]),
        )
    temporary.replace(output)
    print(f"{sample_id}: wrote {features.shape[0]} tiles to {output}", flush=True)
    return output


def probe_slide(cfg, sample_id: str) -> None:
    """Cheap reader/metadata/pixel-access check before launching GPU workers."""
    root = Path(str(cfg.data.hest_data_dir))
    wsi_path = _resolve_wsi(root, sample_id)
    slide, backend = _open_slide(wsi_path)
    try:
        dimensions = tuple(map(int, slide.dimensions))
        mpp_x, mpp_y = _slide_mpp(slide)
        probe_size = tuple(min(256, value) for value in dimensions)
        region = slide.read_region((0, 0), 0, probe_size).convert("RGB")
        if region.size != probe_size:
            raise RuntimeError(
                f"WSI reader returned probe size {region.size}, expected {probe_size}"
            )
    finally:
        slide.close()
    print(
        f"WSI probe ready: {sample_id} reader={backend} dimensions={dimensions} "
        f"mpp=({mpp_x:.4f},{mpp_y:.4f}) path={wsi_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--tile-encoder-revision", required=True,
        help="MANDATORY: an already-resolved, immutable Hugging Face commit SHA (40 "
             "lowercase hex characters) for prov-gigapath/prov-gigapath's tile encoder -- "
             "resolve a tag/branch (e.g. 'main') to its real commit SHA yourself first "
             "(e.g. via the HuggingFace web UI or huggingface_hub.HfApi().model_info(...).sha) "
             "and pass that SHA here. A moving ref is refused: two cache builds using "
             "'main' at different times could silently use different weights while "
             "appearing identically pinned. See _load_gigapath_tile_encoder's docstring "
             "(src/models/conditioning.py).",
    )
    args = parser.parse_args()
    if args.target_mpp <= 0 or not 0 <= args.min_tissue_fraction <= 1:
        raise ValueError("target-mpp must be positive and min-tissue-fraction must be in [0,1]")
    from src.models.conditioning import _validate_immutable_hf_revision
    _validate_immutable_hf_revision(args.tile_encoder_revision)
    cfg = OmegaConf.load(args.config)
    ids = list(args.sample_ids or cfg.data.get("sample_ids", []))
    if not ids:
        ids = [str(cfg.data.sample_id)]
    for sample_id in ids:
        if args.probe_only:
            probe_slide(cfg, str(sample_id))
        else:
            build_cache(
                cfg, str(sample_id), args.batch_size, args.target_mpp,
                args.min_tissue_fraction, args.device,
                tile_encoder_revision=args.tile_encoder_revision,
            )


if __name__ == "__main__":
    main()
