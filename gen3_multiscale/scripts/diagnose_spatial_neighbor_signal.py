#!/usr/bin/env python3
"""How much expression is predictable from SPATIAL NEIGHBOURS alone?

This is a premise check, not a model. Both the iterative spatial refinement
(Architecture 3) and the self-supervised spatial prior (Architecture 4) rest on
one assumption -- that a spot's expression is predictable from its neighbours'
expression -- and that assumption has never been measured on this data at this
panel size. STFlow's entire reported margin comes from exactly this mechanism,
but on a small benchmark panel; we predict ~17,000 genes.

The estimator is deliberately parameter-free: hide a fraction of spots, then
predict each hidden spot as the MEAN OF ITS OBSERVED NEIGHBOURS. No training,
no capacity, no tuning, so the result cannot be confounded by optimisation.
It is a lower bound on what a learned neighbour-attending module could reach,
and it settles the ambiguity a short pretraining probe cannot: a trained module
scoring ~0 after few steps is indistinguishable from "undertrained" and "no
signal exists", whereas this number distinguishes them directly.

Three predictors are scored side by side on the identical held-out spots:

* ``neighbor_mean`` -- the real question.
* ``slide_mean`` -- each gene's mean over the OBSERVED spots. This is what MSE
  training converges to first, and it must score ~0; it is the control that
  proves the metric is not rewarding a constant.
* ``shuffled_neighbor_mean`` -- the neighbour-mean predictor computed on a
  random permutation of the coordinates. It holds the estimator and the
  expression distribution fixed while destroying only the spatial relationship,
  so the gap between it and ``neighbor_mean`` is the part attributable to
  SPACE rather than to the marginal statistics of expression.

Metrics are the project's own ``pearson_per_gene``: per-gene Pearson across the
held-out spots WITHIN one slide, genes with no variance in the truth excluded,
a constant prediction for a variable gene scoring 0 rather than vanishing.
Reported over the full panel and, separately, over the highest-variance genes
of that slide, because the headline numbers in this study are HVG panels and a
full-panel mean is dominated by near-silent genes.

    python -m gen3_multiscale.scripts.diagnose_spatial_neighbor_signal \\
        --config /path/to/an_arm_config.yaml --max-slides 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from gen3_multiscale.conditional_wae.spatial_refinement import padded_neighbor_graph
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.metrics import pearson_per_gene


def neighbor_mean_prediction(expression: np.ndarray, coords: np.ndarray,
                             observed: np.ndarray, *, k_neighbors: int
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Predict every spot as the mean of its OBSERVED neighbours.

    Returns ``(prediction, has_observed_neighbor)``. Hidden neighbours are
    excluded from each average -- including them would leak the answer, since
    a hidden spot would then be predicted partly from other hidden spots'
    true values, which no real inference could see.
    """
    indices, valid = padded_neighbor_graph(coords, k_neighbors)
    indices = indices.numpy()
    valid = valid.numpy() & observed[indices]
    weights = valid.astype(np.float64)
    counts = weights.sum(axis=1)
    has_neighbor = counts > 0
    safe_counts = np.where(has_neighbor, counts, 1.0)
    gathered = expression[indices]
    totals = np.einsum("nk,nkg->ng", weights, gathered.astype(np.float64))
    return totals / safe_counts[:, None], has_neighbor


def _panel_summary(prediction: np.ndarray, truth: np.ndarray,
                   top_genes: np.ndarray) -> dict:
    per_gene = pearson_per_gene(prediction, truth)
    scored = per_gene[~np.isnan(per_gene)]
    top = per_gene[top_genes]
    top_scored = top[~np.isnan(top)]
    return {
        "all_genes_pcc": float(np.mean(scored)) if scored.size else float("nan"),
        "n_scored_genes": int(scored.size),
        "top_variance_genes_pcc": float(np.mean(top_scored)) if top_scored.size else float("nan"),
    }


def diagnose_slide(expression: np.ndarray, coords: np.ndarray, *,
                   mask_fraction: float, k_neighbors: int, n_top_genes: int,
                   rng: np.random.Generator) -> dict:
    n_spots = expression.shape[0]
    n_hidden = max(1, min(n_spots - 1, int(round(n_spots * mask_fraction))))
    hidden = np.zeros(n_spots, dtype=bool)
    hidden[rng.permutation(n_spots)[:n_hidden]] = True
    observed = ~hidden

    prediction, has_neighbor = neighbor_mean_prediction(
        expression, coords, observed, k_neighbors=k_neighbors,
    )
    # Score only hidden spots that actually had an observed neighbour; a spot
    # with none is a graph fact, not a prediction failure, and averaging a
    # fabricated value for it would quietly bias the result.
    scored = hidden & has_neighbor
    truth = expression[scored].astype(np.float64)
    if truth.shape[0] < 3:
        raise ValueError("too few scorable hidden spots on this slide")

    variance = truth.var(axis=0)
    top_genes = np.argsort(variance)[::-1][:n_top_genes]

    slide_mean = np.repeat(
        expression[observed].astype(np.float64).mean(axis=0, keepdims=True),
        truth.shape[0], axis=0,
    )
    shuffled_coords = coords[rng.permutation(n_spots)]
    shuffled_prediction, shuffled_has_neighbor = neighbor_mean_prediction(
        expression, shuffled_coords, observed, k_neighbors=k_neighbors,
    )
    shuffled_scored = hidden & shuffled_has_neighbor

    return {
        "n_spots": int(n_spots),
        "n_hidden_scored": int(truth.shape[0]),
        "n_hidden_without_observed_neighbor": int((hidden & ~has_neighbor).sum()),
        "neighbor_mean": _panel_summary(prediction[scored], truth, top_genes),
        "slide_mean": _panel_summary(slide_mean, truth, top_genes),
        "shuffled_neighbor_mean": _panel_summary(
            shuffled_prediction[shuffled_scored],
            expression[shuffled_scored].astype(np.float64),
            top_genes,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Any arm config naming the dataset manifest")
    parser.add_argument("--max-slides", type=int, default=8)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--n-top-genes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    config = resolved_config(args.config)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    sample_ids = sorted(manifest["train_sample_ids"])[: args.max_slides]
    if not sample_ids:
        raise ValueError("no train slides to diagnose")

    rng = np.random.default_rng(args.seed)
    records = []
    for position, sample_id in enumerate(sample_ids, start=1):
        adata = example_builder.load_expression_for_model_target_space(manifest, sample_id)
        expression = np.asarray(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32,
        )
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        record = diagnose_slide(
            expression, coords, mask_fraction=args.mask_fraction,
            k_neighbors=args.k_neighbors, n_top_genes=args.n_top_genes, rng=rng,
        )
        record["sample_id"] = sample_id
        records.append(record)
        print(
            f"[{position}/{len(sample_ids)}] {sample_id}  "
            f"spots={record['n_spots']} scored={record['n_hidden_scored']}\n"
            f"    neighbor_mean            all={record['neighbor_mean']['all_genes_pcc']:+.4f}  "
            f"top{args.n_top_genes}={record['neighbor_mean']['top_variance_genes_pcc']:+.4f}\n"
            f"    shuffled_neighbor_mean   all={record['shuffled_neighbor_mean']['all_genes_pcc']:+.4f}  "
            f"top{args.n_top_genes}={record['shuffled_neighbor_mean']['top_variance_genes_pcc']:+.4f}\n"
            f"    slide_mean (control)     all={record['slide_mean']['all_genes_pcc']:+.4f}  "
            f"top{args.n_top_genes}={record['slide_mean']['top_variance_genes_pcc']:+.4f}",
            flush=True,
        )
        del expression, coords, adata

    print("\n=== across-slide means ===", flush=True)
    for predictor in ("neighbor_mean", "shuffled_neighbor_mean", "slide_mean"):
        all_genes = np.nanmean([r[predictor]["all_genes_pcc"] for r in records])
        top = np.nanmean([r[predictor]["top_variance_genes_pcc"] for r in records])
        print(f"{predictor:24s} all={all_genes:+.4f}  top{args.n_top_genes}={top:+.4f}", flush=True)

    if args.output:
        report = {
            "kind": "spatial_neighbor_signal_diagnostic",
            "config_path": str(args.config),
            "mask_fraction": float(args.mask_fraction),
            "k_neighbors": int(args.k_neighbors),
            "n_top_genes": int(args.n_top_genes),
            "seed": int(args.seed),
            "per_slide": records,
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
        os.replace(temporary, path)
        print(f"\nreport saved to {path}", flush=True)


if __name__ == "__main__":
    main()
