#!/usr/bin/env python3
"""Export a REAL, continuous whole-slide-image crop with the missing-tissue
hole actually masked out at pixel resolution -- not spot patches stitched
together (see export_real_masking_example.py for that), one real photo of
the slide itself with the hole blacked out, plus a second copy with the
real spot centers drawn on top so it's visually obvious which spots are
considered inside the missing region.

Reuses _resolve_wsi / _open_slide from scripts/precompute_gigapath_wsi_tiles.py
(the same real WSI-reading code the hierarchical dense-tile precompute uses)
rather than reimplementing WSI I/O. adata.obsm['spatial'] and the raw WSI's
level-0 pixel coordinates are the SAME coordinate frame in HEST-1k (this is
exactly why scripts/precompute_gigapath_wsi_tiles.py's own dense-tile cache
stores "level0_coords" specifically so it can be compared against query-hole
coordinates directly -- see that script's own module docstring).

Must run on a machine with the real HEST-1k data AND the real WSI file
downloaded (data/raw/hest1k/**/{sample_id}.svs|.tif|.tiff|.ndpi|.mrxs) --
this repo's sandbox has neither, so this cannot be run/tested here; run it
on st-a100. Needs openslide (or tiffslide): see
docs/hierarchical_missing_tissue.md's "WSI runtime" section for install.

Per-pixel context/query/boundary-zeroed assignment uses nearest-spot
(Voronoi-style, via cKDTree) rather than trying to recover the exact
internal random circle/ellipse/blob parameters random_dropout_patches drew
internally (it only returns final spot-level boolean masks, not those
parameters) -- this is the natural pixel-resolution extension of a
spot-level mask and matches what actually determines model input (context
vs. query is a per-spot decision; there is no finer-grained ground truth
than that).

Writes to --output-dir:
  masked_tissue.png            -- real WSI crop, hole pixels blacked out
  masked_tissue_with_spots.png -- same, real spot centers overlaid
                                   (blue=context w/ real image,
                                   orange=context w/ image zeroed by
                                   strict_broken_region, red=query)

Usage:
  python3 -m scripts.export_real_masked_wsi_region --sample-id INT8 \
      --hest-data-dir data/raw/hest1k --seed 960000 \
      --output-dir results/mask_exports/INT8_seed960000_wsi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-data-dir", default="data/raw/hest1k")
    parser.add_argument("--sample-id", default="INT8")
    parser.add_argument("--seed", type=int, default=960000)
    parser.add_argument("--n-patches", type=int, default=1)
    parser.add_argument("--radius-range", type=float, nargs=2, default=(3.0, 6.0))
    parser.add_argument("--radius-unit", default="spot_spacing", choices=["spot_spacing", "coordinate"])
    parser.add_argument("--shape", default="mixed", choices=["circle", "ellipse", "irregular", "mixed"])
    parser.add_argument("--query-patch-size", type=float, default=224.0)
    parser.add_argument("--margin-spacings", type=float, default=8.0,
                         help="How many extra spot-spacings of real tissue to show around "
                              "the hole, beyond the hole's own realized radius.")
    parser.add_argument("--max-window-px", type=int, default=6000,
                         help="Safety cap on the read_region crop size in pixels per side.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from src.data.loaders import (
        load_hest_sample, basic_qc_and_normalize, load_hest_patches, align_patches_to_adata,
    )
    from src.data import masking
    from src.data.slide_context import nonoverlapping_context_patch_mask
    from scripts.precompute_gigapath_wsi_tiles import _resolve_wsi, _open_slide

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading real {args.sample_id} expression + spot coordinates...")
    adata = load_hest_sample(args.hest_data_dir, args.sample_id)
    adata = basic_qc_and_normalize(adata)
    patches, barcodes = load_hest_patches(args.hest_data_dir, args.sample_id)
    adata, _patches = align_patches_to_adata(adata, patches, barcodes)
    coords_xy = np.asarray(adata.obsm["spatial"][:, :2], dtype="float64")
    slice_ids = np.zeros(adata.n_obs, dtype=int)
    print(f"  {adata.n_obs} real spots")

    print(f"Calling the REAL src.data.masking.random_dropout_patches with the exact "
          f"transport-suite params (seed={args.seed})...")
    context_mask, query_mask = masking.random_dropout_patches(
        coords_xy, slice_ids, n_patches=args.n_patches,
        radius_range=tuple(args.radius_range), radius_unit=args.radius_unit,
        shape=args.shape, seed=args.seed,
    )
    context_idx = np.where(context_mask)[0]
    query_idx = np.where(query_mask)[0]
    safe_image = nonoverlapping_context_patch_mask(
        coords_xy[context_idx], coords_xy[query_idx], args.query_patch_size,
    )
    image_available = np.ones(len(coords_xy), dtype=bool)
    image_available[query_idx] = False
    image_available[context_idx] = safe_image
    print(f"  context: {context_mask.sum()}, query: {query_mask.sum()}, "
          f"context-with-image-also-zeroed: {int((~safe_image).sum())}")

    from scipy.spatial import cKDTree
    neighbour_d, _ = cKDTree(coords_xy).query(coords_xy, k=2)
    spacing = float(np.median(neighbour_d[:, 1]))
    hole_center = coords_xy[query_idx].mean(axis=0)
    realized_hole_radius = float(np.linalg.norm(coords_xy[query_idx] - hole_center, axis=1).max())
    window = realized_hole_radius + args.margin_spacings * spacing
    window = min(window, args.max_window_px / 2.0)

    print(f"Resolving and opening the REAL WSI for {args.sample_id}...")
    wsi_path = _resolve_wsi(Path(args.hest_data_dir), args.sample_id)
    slide, backend = _open_slide(wsi_path)
    print(f"  {wsi_path} (reader={backend}, dimensions={slide.dimensions})")

    x0 = int(hole_center[0] - window)
    y0 = int(hole_center[1] - window)
    size = int(2 * window)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    size = min(size, slide.dimensions[0] - x0, slide.dimensions[1] - y0)
    print(f"Reading a real {size}x{size} pixel region at level 0, origin=({x0},{y0})...")
    region = np.array(slide.read_region((x0, y0), 0, (size, size)).convert("RGB"))
    slide.close()

    # Per-pixel context/query/boundary-zeroed assignment via nearest real
    # spot (Voronoi-style) -- the natural pixel-resolution extension of the
    # per-spot mask; there is no finer ground truth than "which spot is
    # this pixel part of, and is that spot context/query/image-zeroed".
    print("Assigning every pixel to its nearest real spot (context/query/boundary)...")
    spot_status = np.zeros(len(coords_xy), dtype=np.uint8)  # 0=ctx real image, 1=ctx image zeroed, 2=query
    spot_status[context_idx[~safe_image]] = 1
    spot_status[query_idx] = 2
    tree = cKDTree(coords_xy)
    yy, xx = np.meshgrid(np.arange(y0, y0 + size), np.arange(x0, x0 + size), indexing="ij")
    pixel_coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype("float64")
    _, nearest_spot = tree.query(pixel_coords, k=1)
    pixel_status = spot_status[nearest_spot].reshape(size, size)

    masked = region.copy()
    masked[pixel_status >= 1] = 0  # black out query AND boundary-image-zeroed regions

    from PIL import Image
    Image.fromarray(masked).save(output_dir / "masked_tissue.png")
    print(f"  wrote {output_dir / 'masked_tissue.png'}")

    print("Building the spots-overlaid version...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(masked, extent=[x0, x0 + size, y0 + size, y0])
    in_window = (
        (coords_xy[:, 0] >= x0) & (coords_xy[:, 0] <= x0 + size)
        & (coords_xy[:, 1] >= y0) & (coords_xy[:, 1] <= y0 + size)
    )
    colors = np.array(["#2E86AB", "#F4A261", "#E63946"])[spot_status]
    ax.scatter(coords_xy[in_window, 0], coords_xy[in_window, 1], s=10,
               c=colors[in_window], edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xlim(x0, x0 + size)
    ax.set_ylim(y0 + size, y0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{args.sample_id}, seed={args.seed}: REAL WSI crop, hole masked at pixel resolution.\n"
        f"blue = context (real image), orange = context (image zeroed, "
        f"strict_broken_region), red = query (fully hidden)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "masked_tissue_with_spots.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  wrote {output_dir / 'masked_tissue_with_spots.png'}")
    print(f"\nDone. Both files in {output_dir} are built from the real WSI -- "
          f"pull the directory to your local machine to inspect/share.")


if __name__ == "__main__":
    main()
