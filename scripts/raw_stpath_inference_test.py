"""
Raw STPath inference test (2026-07-20) — "how well does STPath itself,
completely disconnected from any of our own generative pipeline, fill in
masked gene expression?" Direct request following the held-out
generalization test's near-zero result: does STPath's own real,
pretrained prediction head do any better on the exact same task, using
its own native in-context-learning mechanism (real expression for
context spots, get predictions for the rest) with NO FM-OT, NO custom
decoder, NO velocity_net, NO training on our data at all?

Uses STPathContextEncoder.predict_raw_expression (src/models/
stpath_encoder.py) — reuses the SAME already-verified token-construction
code STPathContextEncoder.forward() uses for every other config in this
project, but keeps prediction_head's FIRST return value (STPath's own
real gene-expression prediction) instead of discarding it for our decoder.

Uses the SAME held-out split (masking.held_out_mask) as
scripts/run_parallel_8gpu_heldout_generalization_test.sh, same
heldout_seed=20260720, so this result is DIRECTLY comparable to that
batch's numbers — same 145 held-out spots, same "never seen this exact
task before" framing (STPath needs no training here at all, so "held-out"
just means "the spots we score on", but keeping the same split makes the
gene target set and spot identities directly comparable).

Requires the real `stpath` package installed (pip install -e . from a
cloned github.com/Graph-and-Geometric-Learning/STPath) + einops==0.8.0,
the real pretrained weights (huggingface.co/tlhuang/STPath), and the
bundled gene vocabulary file (symbol2ensembl.json in that repo) — see
src/models/stpath_encoder.py's own module docstring for the full setup.
Set the three env vars below or edit the paths directly.

Usage:
  STPATH_GENE_VOC_PATH=/path/to/symbol2ensembl.json \
  STPATH_MODEL_WEIGHT_PATH=/path/to/stpath_weights.pt \
  python -m scripts.raw_stpath_inference_test --sample-id INT1
"""
import argparse
import os

import numpy as np
import torch

from src.training.train import load_adata, get_gigapath_features, _images_tensor
from src.data import masking
from src.data.loaders import load_hest_patches, align_patches_to_adata
from src.evaluation import metrics as ev
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="INT1")
    parser.add_argument("--hest-data-dir", default="data/raw/hest1k")
    parser.add_argument("--organ-type", default="Kidney")  # INT1 is ccRCC
    parser.add_argument("--tech-type", default="Visium")
    parser.add_argument("--heldout-fraction", type=float, default=0.15)
    parser.add_argument("--heldout-seed", type=int, default=20260720)  # matches the 8-job batch
    args = parser.parse_args()

    gene_voc_path = os.environ.get("STPATH_GENE_VOC_PATH")
    model_weight_path = os.environ.get("STPATH_MODEL_WEIGHT_PATH")
    assert gene_voc_path and model_weight_path, (
        "Set STPATH_GENE_VOC_PATH and STPATH_MODEL_WEIGHT_PATH env vars first "
        "(see this script's own module docstring for where to get them)."
    )

    cfg = OmegaConf.create({
        "data": {
            "source": "hest1k", "hest_data_dir": args.hest_data_dir,
            "sample_id": args.sample_id, "min_genes": 200, "min_cells": 3,
        },
    })
    print(f"Loading {args.sample_id}...")
    adata = load_adata(cfg)
    coords3d = np.stack([adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1],
                          np.zeros(adata.n_obs)], axis=1).astype("float32")
    expr = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype="float32")
    gene_names = adata.var_names.tolist()
    print(f"  {adata.n_obs} spots, {len(gene_names)} genes")

    print("Loading H&E patches + Gigapath features...")
    patches, barcodes = load_hest_patches(args.hest_data_dir, args.sample_id)
    features = get_gigapath_features(cfg, patches, barcodes)
    _, gigapath_images = align_patches_to_adata(adata, features, barcodes)

    heldout = masking.held_out_mask(adata.n_obs, args.heldout_fraction, args.heldout_seed)
    context_mask, query_mask = ~heldout, heldout
    print(f"  context: {context_mask.sum()} spots, held-out query: {query_mask.sum()} spots "
          f"(same split as the 8-job held-out generalization batch)")

    print("Loading real pretrained STPath (frozen)...")
    from src.models.stpath_encoder import STPathContextEncoder
    encoder = STPathContextEncoder(
        gene_names=gene_names, gene_voc_path=gene_voc_path, model_weight_path=model_weight_path,
        organ_type=args.organ_type, tech_type=args.tech_type, pretrained=True,
    )
    encoder.eval()
    print(f"  {len(encoder._valid_gene_pos)}/{len(gene_names)} of our genes matched STPath's own vocabulary")

    context_coords = torch.tensor(coords3d[context_mask], dtype=torch.float32)
    query_coords = torch.tensor(coords3d[query_mask], dtype=torch.float32)
    context_expr = torch.tensor(expr[context_mask], dtype=torch.float32)
    context_images = _images_tensor(gigapath_images, context_mask)
    query_images = _images_tensor(gigapath_images, query_mask)

    print("Running STPath's own native prediction head (no training, no fine-tuning, "
          "no wrapping generative model)...")
    raw_pred = encoder.predict_raw_expression(
        context_coords, context_expr, query_coords, context_images, query_images,
    ).cpu().numpy()

    # target: our real expression, log1p'd (matching STPath's own ge_tokens convention,
    # see forward()'s "expr = torch.log1p(context_expression)[:, self._valid_gene_pos]"),
    # restricted to the same valid_gene_pos columns predict_raw_expression already returns
    target = np.log1p(expr[query_mask][:, encoder._valid_gene_pos])

    pcc = np.nanmean(ev.pearson_per_gene(raw_pred, target))
    rmse = ev.rmse(raw_pred, target)
    print("")
    print("=== RAW STPath inference (no training on our data at all) ===")
    print(f"mean PCC:  {pcc:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"n_query:   {query_mask.sum()} spots, {len(encoder._valid_gene_pos)} genes")
    print("")
    print("Compare directly against the held-out generalization batch's numbers "
          "(same split, same spots): standard-eval PCC ~0.25-0.45 (memorized), "
          "held-out PCC ~0.02 (our own trained models on unseen spots). If this raw "
          "STPath number is meaningfully above ~0.02, STPath's own pretrained in-context "
          "mechanism IS doing real work that our wrapping pipeline currently discards "
          "or fails to exploit. If it's ALSO near zero, the bottleneck may be more "
          "fundamental (data/task mismatch, not just our own decoder).")


if __name__ == "__main__":
    main()
