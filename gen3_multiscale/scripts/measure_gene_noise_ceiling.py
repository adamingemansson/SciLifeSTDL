#!/usr/bin/env python3
"""Per-gene achievable-PCC ceiling for each slide, by count splitting.

Answers the question every current number begs: is a low per-gene PCC a model
failure, or the most anyone could have scored on a gene that noisy? Reported
scores can then be divided by this ceiling instead of read against 1.0.

    python -m gen3_multiscale.scripts.measure_gene_noise_ceiling \\
        --config /path/to/an_arm_config.yaml --split validation \\
        --output /path/to/noise_ceiling.json

Runs on raw counts straight from HEST, restricted to the manifest's declared
barcodes and gene panel, and normalises each split half with the manifest's
own recorded transform so the estimate lives in the model's target space.
CPU only, no GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data import loaders
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.noise_ceiling import gene_noise_ceiling


def _normalizer(build_args: dict):
    """Re-apply the manifest's own target-space transform to a count matrix.

    No per-cell filter is applied: splitting halves each spot's depth, so a
    min_genes filter would drop different spots from each half and the two
    would stop describing the same spots.
    """
    target_sum = float(build_args["expression_target_sum"])
    transform = str(build_args["expression_transform"])
    if "log1p" not in transform:
        raise ValueError(f"noise ceiling assumes a log1p target transform, got {transform!r}")

    def normalize(counts: np.ndarray) -> np.ndarray:
        totals = counts.sum(axis=1, keepdims=True)
        scaled = counts / np.where(totals > 0, totals, 1.0) * target_sum
        return np.log1p(scaled)

    return normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--n-top-genes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = resolved_config(args.config)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    gene_names = [str(gene) for gene in manifest["gene_panel"]]
    sample_ids = sorted(manifest[f"{args.split}_sample_ids"])
    if args.max_slides is not None:
        sample_ids = sample_ids[: args.max_slides]
    if not sample_ids:
        raise ValueError(f"no {args.split} slides to measure")

    normalize = _normalizer(manifest["build_args"])
    hest_dir = Path(manifest["hest_data_dir"])
    rng = np.random.default_rng(args.seed)
    records = []
    for position, sample_id in enumerate(sample_ids, start=1):
        record = manifest["samples"][sample_id]
        adata = loaders.load_hest_sample(
            hest_dir, sample_id, organ=record["organ"], tech=record["tech"],
        )
        keep = [str(b) for b in record["barcodes"]]
        adata = adata[keep, gene_names]
        result = gene_noise_ceiling(adata.X, normalize=normalize, rng=rng)
        ceiling = result["ceiling"]
        # Rank by the ceiling itself: "of the genes that ARE measurable here,
        # how measurable are they" is the number that bounds a report.
        finite = ceiling[~np.isnan(ceiling)]
        top = np.sort(finite)[::-1][: args.n_top_genes]
        records.append({
            "sample_id": sample_id,
            "n_spots": int(adata.shape[0]),
            "n_scored_genes": result["n_scored_genes"],
            "mean_ceiling_all_genes": float(np.nanmean(ceiling)),
            "mean_ceiling_top_genes": float(np.mean(top)) if top.size else float("nan"),
            "fraction_genes_ceiling_below_0.1": float(np.mean(finite < 0.1)) if finite.size else float("nan"),
            "ceiling_by_gene": {
                gene: float(value)
                for gene, value in zip(gene_names, ceiling)
                if not np.isnan(value)
            },
        })
        print(
            f"[{position}/{len(sample_ids)}] {sample_id}  spots={adata.shape[0]}  "
            f"ceiling all={records[-1]['mean_ceiling_all_genes']:.4f} "
            f"top{args.n_top_genes}={records[-1]['mean_ceiling_top_genes']:.4f}  "
            f"unmeasurable(<0.1)={records[-1]['fraction_genes_ceiling_below_0.1']:.1%}",
            flush=True,
        )
        del adata

    print("\n=== across-slide means ===", flush=True)
    print(
        f"ceiling all genes  = {np.nanmean([r['mean_ceiling_all_genes'] for r in records]):.4f}\n"
        f"ceiling top{args.n_top_genes}    = {np.nanmean([r['mean_ceiling_top_genes'] for r in records]):.4f}\n"
        f"genes with ceiling < 0.1 = "
        f"{np.nanmean([r['fraction_genes_ceiling_below_0.1'] for r in records]):.1%}",
        flush=True,
    )

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps({
        "kind": "gene_noise_ceiling_by_count_splitting",
        "config_path": str(args.config),
        "split": args.split,
        "seed": int(args.seed),
        "n_top_genes": int(args.n_top_genes),
        "per_slide": records,
    }, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(f"\nreport saved to {path}", flush=True)


if __name__ == "__main__":
    main()
