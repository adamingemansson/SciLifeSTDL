#!/usr/bin/env python3
"""Pretrain the spatial expression refiner on TRAIN-split expression alone.

Stage 1 of Architecture 4. No image, no tile-encoder cache, no paired
supervision: the refiner is shown real expression fields with a random quarter
of their spots hidden and must reconstruct the hidden spots from their
neighbours. That teaches it the spatial autocorrelation, tissue-domain
structure and gene-gene coupling of real ST data, which is exactly the
knowledge the image pathway was measured NOT to supply (the latent's
contribution to our trained arms correlates 0.002-0.025 with the conditional
mean's error -- it is orthogonal noise, not missing biology).

    python -m gen3_multiscale.scripts.pretrain_conditional_wae_spatial_prior \\
        --config /path/to/an_arm_config.yaml \\
        --output /path/to/spatial_prior.pt \\
        --rounds 6 --steps-per-slide 4 --device cuda:0

The refiner is built THROUGH the arm's own `_build_model`, so its dimensions
are the ones that arm will actually instantiate -- not a second, independently
configured guess that could drift. `data.spatial_prior_path` is stripped from
the config first: this script PRODUCES that artifact, so consuming it here
would be circular.

Slides are loaded one at a time via `load_expression_for_model_target_space`,
which reads expression only and never touches H&E patches, so peak RAM is one
slide (~170 MB at full panel) rather than the whole corpus.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.spatial_prior import (
    pretrain_spatial_prior_streaming,
    save_spatial_prior,
)
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data import example_builder
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.training.train_conditional_wae import _build_model


def _slide_loader(manifest: dict):
    def load(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
        adata = example_builder.load_expression_for_model_target_space(manifest, sample_id)
        expression = np.asarray(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, dtype=np.float32,
        )
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if coords.shape[0] != expression.shape[0]:
            raise ValueError(
                f"{sample_id}: {coords.shape[0]} coordinates for {expression.shape[0]} spots"
            )
        return expression, coords

    return load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="The arm config whose refiner geometry this prior must match. Its "
             "model.params.n_refinement_steps must be > 0.",
    )
    parser.add_argument("--output", required=True, help="Destination .pt artifact")
    parser.add_argument("--rounds", type=int, default=6, help="Passes over the training corpus")
    parser.add_argument("--steps-per-slide", type=int, default=4)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-top-genes", type=int, default=200,
        help="Size of the highest-variance gene set the reported PCC is averaged over, "
             "alongside the full panel. The full-panel mean is dominated by near-silent "
             "genes and moves very little; the headline metrics in this study are HVG "
             "panels, so this is the number to watch.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--validation-slides", type=int, default=4,
        help="How many VALIDATION-split slides to score after every round. These are "
             "never trained on, which is what makes the number meaningful: the "
             "per-slide metric printed during training is measured on the slides just "
             "optimised, so memorisation and genuine spatial learning look identical "
             "there. 0 disables the held-out pass.",
    )
    parser.add_argument(
        "--max-slides", type=int,
        help="Optional cap on how many TRAIN slides to use, applied after sorting "
             "for determinism. For quick sanity runs only.",
    )
    args = parser.parse_args()

    config = copy.deepcopy(resolved_config(args.config))
    # This script produces the prior; consuming one here would be circular.
    config.setdefault("data", {}).pop("spatial_prior_path", None)
    static_audit_conditional_wae_config(config)
    if int(config["model"]["params"].get("n_refinement_steps", 0)) < 1:
        raise ValueError(
            f"{args.config}: model.params.n_refinement_steps is 0, so this arm has no spatial "
            "refiner to pretrain"
        )

    dataset_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    gene_names = list(dataset_manifest["gene_panel"])
    train_sample_ids = sorted(dataset_manifest["train_sample_ids"])
    if not train_sample_ids:
        raise ValueError("dataset manifest has zero train_sample_ids")
    if args.max_slides is not None:
        if args.max_slides < 1:
            raise ValueError("--max-slides must be positive")
        train_sample_ids = train_sample_ids[: args.max_slides]
    if args.validation_slides < 0:
        raise ValueError("--validation-slides must be non-negative")
    validation_ids = sorted(dataset_manifest.get("validation_sample_ids") or [])[: args.validation_slides]
    overlap = set(validation_ids) & set(train_sample_ids)
    if overlap:
        raise ValueError(f"validation slides {sorted(overlap)} also appear in the training split")

    device = torch.device(args.device)
    model = _build_model(config, len(gene_names), gene_names=gene_names)
    refiner = model.spatial_refiner
    if refiner is None:
        raise ValueError("the configured arm did not construct a spatial refiner")
    # Only the refiner is trained here; the rest of the model exists solely to
    # fix the refiner's dimensions and is released before the loop.
    refiner = refiner.to(device)
    del model

    print(
        f"pretraining spatial prior on {len(train_sample_ids)} TRAIN slides "
        f"({len(gene_names)} genes) for {args.rounds} rounds x {args.steps_per_slide} steps/slide",
        flush=True,
    )
    history = pretrain_spatial_prior_streaming(
        refiner, train_sample_ids, _slide_loader(dataset_manifest),
        rounds=args.rounds, steps_per_slide=args.steps_per_slide,
        mask_fraction=args.mask_fraction, learning_rate=args.learning_rate,
        seed=args.seed, n_top_genes=args.n_top_genes,
        validation_ids=validation_ids, device=device,
    )

    provenance = {
        "config_path": str(args.config),
        "gen3_manifest_path": str(config["data"]["gen3_manifest_path"]),
        "n_train_samples": len(train_sample_ids),
        "train_sample_ids": train_sample_ids,
        "validation_sample_ids": validation_ids,
        "rounds": int(args.rounds),
        "steps_per_slide": int(args.steps_per_slide),
        "mask_fraction": float(args.mask_fraction),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "n_top_genes": int(args.n_top_genes),
        "history": history,
    }
    path = save_spatial_prior(
        refiner.to("cpu"), args.output, gene_names=gene_names, provenance=provenance,
    )
    print(f"\nspatial prior saved to {path}", flush=True)
    top = args.n_top_genes
    print(
        f"{'round':>5}  {'train_all':>9} {'train_top' + str(top):>12}  "
        f"{'held_all':>9} {'held_top' + str(top):>12}",
        flush=True,
    )

    def _mean(records, key):
        values = [r[key] for r in records]
        return np.nanmean(values) if values else float("nan")

    for round_index in range(1, args.rounds + 1):
        rows = [r for r in history if r["round"] == round_index]
        train_rows = [r for r in rows if r["split"] == "train"]
        held_rows = [r for r in rows if r["split"] == "validation"]
        if not rows:
            continue
        print(
            f"{round_index:5d}  {_mean(train_rows, 'masked_spot_pearson'):+9.4f} "
            f"{_mean(train_rows, 'masked_spot_pearson_top_genes'):+12.4f}  "
            f"{_mean(held_rows, 'masked_spot_pearson'):+9.4f} "
            f"{_mean(held_rows, 'masked_spot_pearson_top_genes'):+12.4f}",
            flush=True,
        )
    if not validation_ids:
        print(
            "\nNOTE: --validation-slides 0, so every number above is measured on slides "
            "this run trained on. It cannot distinguish learning from memorisation.",
            flush=True,
        )
    Path(str(path) + ".provenance.json").write_text(
        json.dumps({k: v for k, v in provenance.items() if k != "history"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
