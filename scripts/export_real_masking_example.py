#!/usr/bin/env python3
"""Export REAL data used for masking -- not a synthetic/illustrative
diagram (see the scatter-plot figure sent in chat earlier) but the actual
boolean context/query masks and actual H&E patch images for one real
HEST-1k sample, produced by calling the exact same masking function
(src/data/masking.py's random_dropout_patches) with the exact same params
the transport-suite configs use.

Must be run on a machine with the real HEST-1k data downloaded
(data/raw/hest1k/st/{sample}.h5ad, data/raw/hest1k/patches/{sample}.h5) --
this repo's own sandbox does not have that data, so this script cannot be
run/tested there; run it on st-a100.

Writes to --output-dir:
  mask.npz                        -- context_mask, query_mask (bool [N]),
                                      coords_xy [N,2], barcodes [N] (real
                                      spot IDs) -- everything needed to
                                      exactly reproduce this masking draw
  context_patches/<barcode>.png    -- real H&E patches for N context spots
                                      nearest the hole (what every model,
                                      including harmonic, actually sees)
  query_patches_ground_truth/<barcode>.png
                                    -- real H&E patches for N query spots
                                      (the TRUE tissue -- hidden from every
                                      model in the suite, never used as
                                      input; only used to score predictions)
  query_patches_as_seen_by_model/<barcode>.png
                                    -- all-black 224x224 images -- literally
                                      what image_mode=target_zero replaces
                                      the query patches with as model input
  composite_real_patches.png       -- real H&E patches placed at their real
                                      spatial coordinates around the hole,
                                      with query spots shown as solid black
                                      squares (matches image_mode=target_zero)

Usage:
  python3 -m scripts.export_real_masking_example --sample-id INT8 \
      --hest-data-dir data/raw/hest1k --seed 960000 \
      --output-dir results/mask_exports/INT8_seed960000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-data-dir", default="data/raw/hest1k")
    parser.add_argument("--sample-id", default="INT8")
    parser.add_argument("--seed", type=int, default=960000,
                         help="Matches evaluation.test_seed in the transport-suite "
                              "configs by default -- pass a different int for a "
                              "different draw.")
    # Exact masking params from configs/recovery_suite/*_transport_*.yaml's
    # masking.params section -- change these only if reproducing a DIFFERENT
    # config's masking block.
    parser.add_argument("--n-patches", type=int, default=1)
    parser.add_argument("--radius-range", type=float, nargs=2, default=(3.0, 6.0))
    parser.add_argument("--radius-unit", default="spot_spacing", choices=["spot_spacing", "coordinate"])
    parser.add_argument("--shape", default="mixed", choices=["circle", "ellipse", "irregular", "mixed"])
    parser.add_argument("--n-example-patches", type=int, default=12,
                         help="How many real context/query patch PNGs to save individually.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from src.data.loaders import (
        load_hest_sample, basic_qc_and_normalize, load_hest_patches, align_patches_to_adata,
    )
    from src.data import masking

    output_dir = Path(args.output_dir)
    (output_dir / "context_patches").mkdir(parents=True, exist_ok=True)
    (output_dir / "query_patches_ground_truth").mkdir(parents=True, exist_ok=True)
    (output_dir / "query_patches_as_seen_by_model").mkdir(parents=True, exist_ok=True)

    print(f"Loading real {args.sample_id} expression + H&E patches...")
    adata = load_hest_sample(args.hest_data_dir, args.sample_id)
    adata = basic_qc_and_normalize(adata)
    patches, barcodes = load_hest_patches(args.hest_data_dir, args.sample_id)
    adata, patches = align_patches_to_adata(adata, patches, barcodes)
    print(f"  {adata.n_obs} real spots with matching H&E patches, {patches.shape}")

    coords_xy = np.asarray(adata.obsm["spatial"][:, :2], dtype="float64")
    slice_ids = np.zeros(adata.n_obs, dtype=int)
    spot_barcodes = adata.obs_names.to_numpy()

    print(f"Calling the REAL src.data.masking.random_dropout_patches with the exact "
          f"transport-suite params (n_patches={args.n_patches}, "
          f"radius_range={tuple(args.radius_range)}, radius_unit={args.radius_unit!r}, "
          f"shape={args.shape!r}, seed={args.seed})...")
    context_mask, query_mask = masking.random_dropout_patches(
        coords_xy, slice_ids, n_patches=args.n_patches,
        radius_range=tuple(args.radius_range), radius_unit=args.radius_unit,
        shape=args.shape, seed=args.seed,
    )
    print(f"  context: {context_mask.sum()} real spots, query/masked: {query_mask.sum()} real spots")

    np.savez(
        output_dir / "mask.npz",
        context_mask=context_mask, query_mask=query_mask,
        coords_xy=coords_xy, barcodes=spot_barcodes,
        sample_id=args.sample_id, seed=args.seed,
    )
    print(f"  wrote {output_dir / 'mask.npz'} (load with np.load(..., allow_pickle=True))")

    def save_png(patch: np.ndarray, path: Path) -> None:
        from PIL import Image
        Image.fromarray(patch).save(path)

    # Real H&E patches for the N context/query spots closest to the hole --
    # the most visually informative ones (border of the missing region).
    query_idx = np.where(query_mask)[0]
    context_idx = np.where(context_mask)[0]
    hole_center = coords_xy[query_idx].mean(axis=0)
    ctx_by_dist = context_idx[np.argsort(np.linalg.norm(coords_xy[context_idx] - hole_center, axis=1))]
    qry_by_dist = query_idx[np.argsort(np.linalg.norm(coords_xy[query_idx] - hole_center, axis=1))]

    n = min(args.n_example_patches, len(ctx_by_dist), len(qry_by_dist))
    for i in ctx_by_dist[:n]:
        save_png(patches[i], output_dir / "context_patches" / f"{spot_barcodes[i]}.png")
    for i in qry_by_dist[:n]:
        save_png(patches[i], output_dir / "query_patches_ground_truth" / f"{spot_barcodes[i]}.png")
        save_png(np.zeros_like(patches[i]), output_dir / "query_patches_as_seen_by_model" / f"{spot_barcodes[i]}.png")
    print(f"  wrote {n} real context patches, {n} real query ground-truth patches, "
          f"{n} all-zero query patches (image_mode=target_zero)")

    print("Building composite (real patches placed at real coordinates)...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    neighbour_d, _ = cKDTree(coords_xy).query(coords_xy, k=2)
    spacing = float(np.median(neighbour_d[:, 1]))
    half = spacing * 0.5

    # Crop window must be generous relative to the REALIZED hole radius (up
    # to radius_range[1] spot-spacings), not a fixed constant -- a window
    # only slightly bigger than the hole makes any hole look like it
    # dominates the whole slide, which is misleading (real fix, 2026-07-23:
    # a first version used window=9*spacing against a hole that can itself
    # be up to 6*spacing in radius, leaving almost no visible context ring).
    realized_hole_radius = float(np.linalg.norm(coords_xy[query_idx] - hole_center, axis=1).max())
    window = max(realized_hole_radius * 2.2, spacing * 12)
    pct_of_slide = 100.0 * query_mask.sum() / len(coords_xy)

    fig = plt.figure(figsize=(10, 10.6))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.82])
    in_window = np.linalg.norm(coords_xy - hole_center, axis=1) < window
    for i in np.where(in_window & context_mask)[0]:
        x, y = coords_xy[i]
        ax.imshow(patches[i], extent=[x - half, x + half, y - half, y + half], zorder=1)
    for i in np.where(in_window & query_mask)[0]:
        x, y = coords_xy[i]
        ax.imshow(np.zeros_like(patches[0]), extent=[x - half, x + half, y - half, y + half], zorder=2)
    ax.set_xlim(hole_center[0] - window, hole_center[0] + window)
    ax.set_ylim(hole_center[1] - window, hole_center[1] + window)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{args.sample_id}, seed={args.seed}: real H&E patches, real coordinates "
        f"(local crop, {window / spacing:.0f} spot-spacings shown each way)\n"
        f"Black = query spots as image_mode=target_zero delivers them (GEX hidden too). "
        f"This hole = {query_mask.sum()}/{len(coords_xy)} spots on the whole slide "
        f"({pct_of_slide:.1f}%)", fontsize=10,
    )

    # Full-slide inset for honest scale: where does this crop actually sit
    # relative to the ENTIRE tissue section, and how small is the hole
    # really once the whole slide is in view.
    inset = fig.add_axes([0.68, 0.83, 0.28, 0.15])
    inset.scatter(coords_xy[context_mask, 0], coords_xy[context_mask, 1], s=1.5, c="#2E86AB", linewidths=0)
    inset.scatter(coords_xy[query_mask, 0], coords_xy[query_mask, 1], s=1.5, c="#E63946", linewidths=0)
    from matplotlib.patches import Rectangle
    inset.add_patch(Rectangle(
        (hole_center[0] - window, hole_center[1] - window), 2 * window, 2 * window,
        fill=False, edgecolor="black", linewidth=1.0,
    ))
    inset.invert_yaxis()
    inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_title("whole slide (box = crop above)", fontsize=7.5)

    fig.savefig(output_dir / "composite_real_patches.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  wrote {output_dir / 'composite_real_patches.png'}")
    print(f"  this hole covers {query_mask.sum()}/{len(coords_xy)} spots on the whole slide "
          f"({pct_of_slide:.1f}%)")
    print(f"\nDone. Everything in {output_dir} is real data from {args.sample_id} -- "
          f"pull the whole directory to your local machine to inspect/share.")


if __name__ == "__main__":
    main()
