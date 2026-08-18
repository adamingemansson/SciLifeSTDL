#!/usr/bin/env python3
"""Export spot-free H&E overview images for an immutable HEST split.

The raw WSI remains untouched.  Each output is a downsampled JPEG suitable
for local inspection and figure planning; no Visium coordinates, spots, gene
values, or model predictions are drawn.  Transparent and truly black scanner
canvas pixels are composited to white so exported overviews do not acquire a
large artificial black background.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from scripts.precompute_gigapath_wsi_tiles import _open_slide, _resolve_wsi


def _white_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    canvas.alpha_composite(rgba)
    rgb = np.asarray(canvas.convert("RGB"), dtype=np.uint8).copy()
    # Some WSI readers return the unused scan canvas as opaque RGB black,
    # rather than transparent.  Only replace essentially exact black; dark
    # purple nuclei and real tissue remain untouched.
    scanner_black = np.max(rgb, axis=2) <= 3
    rgb[scanner_black] = 255
    return Image.fromarray(rgb, mode="RGB")


def _overview(slide, max_dimension: int) -> Image.Image:
    width, height = map(int, slide.dimensions)
    scale = min(1.0, float(max_dimension) / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = slide.get_thumbnail(size)
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return _white_background(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-dimension", type=int, default=6000)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()
    if args.max_dimension < 256:
        parser.error("--max-dimension must be at least 256")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_dataset_manifest(manifest_path)
    hest_root = Path(args.hest_data_dir or manifest["hest_data_dir"]).expanduser().resolve()
    available = list(manifest[f"{args.split}_sample_ids"])
    selected = list(args.sample_id) if args.sample_id else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"samples are not in the {args.split} split: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("selected sample IDs must be unique")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample_id in enumerate(selected):
        path = _resolve_wsi(hest_root, sample_id)
        slide, backend = _open_slide(path)
        try:
            dimensions = tuple(map(int, slide.dimensions))
            overview = _overview(slide, args.max_dimension)
        finally:
            slide.close()
        organ = str(manifest["samples"][sample_id]["organ"])
        destination = output_dir / f"{organ}__{sample_id}__HE.jpg"
        overview.save(
            destination, format="JPEG", quality=args.jpeg_quality,
            subsampling=0, optimize=True,
        )
        records.append({
            "sample_id": sample_id,
            "organ": organ,
            "source_wsi": str(path),
            "backend": backend,
            "level0_width": dimensions[0],
            "level0_height": dimensions[1],
            "overview_width": overview.width,
            "overview_height": overview.height,
            "output": str(destination),
        })
        print(
            f"H&E overview: {index + 1}/{len(selected)} sample={sample_id} "
            f"organ={organ} size={overview.width}x{overview.height}",
            flush=True,
        )

    with open(output_dir / "manifest.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "provenance.json").write_text(json.dumps({
        "kind": "hest_spot_free_he_overviews",
        "source_manifest": str(manifest_path),
        "split": args.split,
        "sample_ids": selected,
        "max_dimension": args.max_dimension,
        "jpeg_quality": args.jpeg_quality,
        "contains_spot_overlay": False,
        "contains_expression_data": False,
    }, indent=2, sort_keys=True))
    print(f"Saved {len(records)} spot-free H&E overviews to {output_dir}")


if __name__ == "__main__":
    main()
