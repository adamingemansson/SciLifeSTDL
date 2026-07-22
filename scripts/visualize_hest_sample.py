#!/usr/bin/env python3
"""Audit and visualize a HEST sample using its barcode-aligned AnnData.

The HEST ``.h5ad`` contains both full-resolution spot coordinates and a
downscaled H&E image plus the exact scale factor connecting them.  Using that
embedded pair avoids guessing coordinate transforms from an independently
resized thumbnail.  An optional evaluation mask bank can be overlaid to show
the exact observed context and missing query region used by an experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def _embedded_image_and_scale(adata):
    spatial = adata.uns.get("spatial")
    if spatial is None or not len(spatial):
        raise ValueError("AnnData has no adata.uns['spatial'] image metadata")
    library_id = next(iter(spatial))
    library = spatial[library_id]
    images = library.get("images", {})
    scalefactors = library.get("scalefactors", {})
    choices = (
        ("downscaled_fullres", "tissue_downscaled_fullres_scalef"),
        ("hires", "tissue_hires_scalef"),
        ("lowres", "tissue_lowres_scalef"),
    )
    for image_key, scale_key in choices:
        if image_key in images and scale_key in scalefactors:
            image = np.asarray(images[image_key])
            scale = float(scalefactors[scale_key])
            return library_id, image_key, image, scale, scalefactors
    raise ValueError(
        "No embedded spatial image has a corresponding coordinate scale factor; "
        f"available images={list(images)}, scalefactors={list(scalefactors)}"
    )


def _gene_values(adata, gene: str | None) -> tuple[np.ndarray, str]:
    if gene is None:
        matrix = adata.X
        values = np.asarray(matrix.sum(axis=1)).reshape(-1)
        return np.log1p(values), "log1p total counts"
    if gene not in adata.var_names:
        matches = [name for name in map(str, adata.var_names) if name.lower() == gene.lower()]
        if len(matches) != 1:
            raise KeyError(
                f"gene {gene!r} is absent from {adata.n_vars} genes; "
                "matching is case-insensitive only when unique"
            )
        gene = matches[0]
    column = adata[:, gene].X
    if hasattr(column, "toarray"):
        column = column.toarray()
    values = np.asarray(column).reshape(-1)
    return np.log1p(np.clip(values, 0, None)), f"log1p {gene}"


def _mask_record(mask_bank: Path, split: str, index: int) -> dict:
    payload = json.loads(mask_bank.read_text())
    if payload.get("kind") == "training_seed_schedule":
        raise ValueError(f"{mask_bank} is a training seed schedule, not an evaluation mask bank")
    matches = [
        record for record in payload.get("records", [])
        if str(record.get("split")) == split and int(record.get("index", -1)) == index
    ]
    if len(matches) != 1:
        available = sorted({
            (str(record.get("split")), int(record.get("index", -1)))
            for record in payload.get("records", [])
        })
        raise ValueError(
            f"mask {split}[{index}] was not found in {mask_bank}; available={available}"
        )
    return matches[0]


def _mask_arrays(record: dict, obs_names) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(list(map(str, obs_names)))
    context_names = set(map(str, record["context_obs_names"]))
    query_names = set(map(str, record["query_obs_names"]))
    context = np.asarray([name in context_names for name in names])
    query = np.asarray([name in query_names for name in names])
    if int(context.sum()) != len(context_names) or int(query.sum()) != len(query_names):
        raise ValueError("mask bank barcodes do not match this AnnData observation set")
    if np.any(context & query):
        raise ValueError("mask bank contains overlapping context/query spots")
    return context, query


def _style_axis(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(image, origin="upper")
    axis.set_title(title)
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw/hest1k")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--gene", help="Gene to color; default is log1p total counts")
    parser.add_argument("--mask-bank", type=Path)
    parser.add_argument("--mask-split", choices=("validation", "test"), default="test")
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=9.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    sample_path = root / "st" / f"{args.sample_id}.h5ad"
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    adata = ad.read_h5ad(sample_path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"{sample_path} has no adata.obsm['spatial'] coordinates")
    coords_fullres = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords_fullres.shape != (adata.n_obs, 2) or not np.isfinite(coords_fullres).all():
        raise ValueError(f"invalid spatial coordinates: {coords_fullres.shape}")
    if not adata.obs_names.is_unique:
        raise ValueError("spot barcodes are not unique")

    library_id, image_key, image, scale, scalefactors = _embedded_image_and_scale(adata)
    coords = coords_fullres * scale
    height, width = image.shape[:2]
    inside = (
        (coords[:, 0] >= 0) & (coords[:, 0] < width)
        & (coords[:, 1] >= 0) & (coords[:, 1] < height)
    )
    if not inside.all():
        raise ValueError(
            f"coordinate/image alignment failed: {(~inside).sum()}/{adata.n_obs} spots "
            "fall outside the embedded image"
        )

    values, value_label = _gene_values(adata, args.gene)
    context = query = None
    record = None
    if args.mask_bank is not None:
        record = _mask_record(args.mask_bank, args.mask_split, args.mask_index)
        context, query = _mask_arrays(record, adata.obs_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    _style_axis(axes[0], image, f"{args.sample_id}: H&E and measured spots")
    axes[0].scatter(
        coords[:, 0], coords[:, 1], s=args.point_size,
        facecolors="none", edgecolors="#00d4ff", linewidths=0.55, alpha=0.8,
    )

    if record is None:
        second_title = f"{args.sample_id}: {value_label}"
        _style_axis(axes[1], image, second_title)
        points = axes[1].scatter(
            coords[:, 0], coords[:, 1], c=values, cmap="magma",
            s=args.point_size * 1.25, linewidths=0, alpha=0.9,
        )
        colorbar = fig.colorbar(points, ax=axes[1], fraction=0.046, pad=0.02)
        colorbar.set_label(value_label)
    else:
        second_title = (
            f"{args.sample_id}: {args.mask_split} mask {args.mask_index} "
            f"(seed {record['seed']})"
        )
        _style_axis(axes[1], image, second_title)
        other = ~(context | query)
        axes[1].scatter(
            coords[other, 0], coords[other, 1], s=args.point_size * 0.6,
            c="#b7b7b7", linewidths=0, alpha=0.35,
        )
        axes[1].scatter(
            coords[context, 0], coords[context, 1], s=args.point_size,
            c="#00d4ff", linewidths=0, alpha=0.75,
        )
        axes[1].scatter(
            coords[query, 0], coords[query, 1], s=args.point_size * 1.8,
            c="#ff3b30", linewidths=0, alpha=0.95,
        )
        axes[1].legend(handles=[
            Line2D([], [], marker="o", linestyle="", color="#00d4ff", label=f"context ({context.sum()})"),
            Line2D([], [], marker="o", linestyle="", color="#ff3b30", label=f"missing query ({query.sum()})"),
            Line2D([], [], marker="o", linestyle="", color="#b7b7b7", label=f"unused ({other.sum()})"),
        ], loc="lower right", framealpha=0.85)

    output = args.output or Path("reports/data_audit") / f"{args.sample_id}_overview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    spot_diameter = scalefactors.get("spot_diameter_fullres")
    print(f"sample: {args.sample_id}")
    print(f"h5ad: {sample_path}")
    print(f"spots x genes: {adata.n_obs} x {adata.n_vars}")
    print(f"embedded image: library={library_id}, key={image_key}, shape={image.shape}")
    print(f"fullres-to-image scale: {scale:.12g}")
    print(f"spot diameter fullres: {float(spot_diameter):.3f}" if spot_diameter is not None else "spot diameter fullres: unavailable")
    print(f"coordinate x range fullres: [{coords_fullres[:, 0].min():.1f}, {coords_fullres[:, 0].max():.1f}]")
    print(f"coordinate y range fullres: [{coords_fullres[:, 1].min():.1f}, {coords_fullres[:, 1].max():.1f}]")
    print(f"spots inside embedded image: {inside.sum()}/{adata.n_obs}")
    if record is not None:
        print(f"mask: {args.mask_split}[{args.mask_index}], context={context.sum()}, query={query.sum()}")
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
