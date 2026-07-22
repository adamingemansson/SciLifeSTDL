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

RAW-COUNTS FIX (2026-07-20, from a real research pass into STPath's actual
repo — stpath/data/normalize_utils.py, stpath/app/pipeline/inference.py):
STPath expects RAW counts for context expression (it applies torch.log1p
internally) and ground truth is np.log1p(RAW counts), with NO
normalize_total anywhere in its own pipeline. This project's own
load_adata/basic_qc_and_normalize (src/data/loaders.py) applies
sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p IN PLACE and never
preserves raw counts — feeding that output into STPath would silently
double-transform + library-size-normalize data STPath was never trained
to expect. This script loads raw counts directly (bypassing
basic_qc_and_normalize's normalization steps, keeping only its QC
filtering) specifically for this reason — see _load_raw_counts_adata below.

Also per that same research: STPath's own repo reports its bundled ccRCC
kidney demo (INT2, i.e. the SAME organ/technology as our INT1) gets
zero-shot PCC=0.156, in-context (5% context) PCC=0.245 on top-50 HVGs —
the paper itself states "STPath currently struggles... CCRCC... scarcity
of relevant data in the pretraining set." So a low number here may
reflect a genuine, documented STPath weakness on this exact organ, not
necessarily a bug. Sanity check: rerun with --sample-id INT2 if available
locally and compare against those two reference numbers directly.

Usage:
  STPATH_GENE_VOC_PATH=/path/to/symbol2ensembl.json \
  STPATH_MODEL_WEIGHT_PATH=/path/to/stpath_weights.pt \
  python -m scripts.raw_stpath_inference_test --sample-id INT1
"""
import argparse
import os

import numpy as np
import torch

from src.data import loaders as data_loaders
from src.training.train import get_gigapath_features, _images_tensor
from src.data import masking
from src.data.loaders import load_hest_patches, align_patches_to_adata
from src.evaluation import metrics as ev
from omegaconf import OmegaConf


def _load_raw_counts_adata(cfg):
    """Same QC filtering as basic_qc_and_normalize (src/data/loaders.py),
    WITHOUT normalize_total/log1p — STPath expects raw counts as input
    (it applies its own internal log1p, no total-count normalization at
    all, see this module's own docstring). basic_qc_and_normalize
    normalizes in place and never preserves raw counts, so this can't
    reuse it directly -- duplicates just the two QC filter calls."""
    import scanpy as sc
    adata = data_loaders.load_hest_sample(cfg.data.hest_data_dir, cfg.data.sample_id)
    sc.pp.filter_cells(adata, min_genes=cfg.data.min_genes)
    sc.pp.filter_genes(adata, min_cells=cfg.data.min_cells)
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", default="INT1")
    parser.add_argument("--hest-data-dir", default="data/raw/hest1k")
    parser.add_argument("--organ-type", default="Kidney")  # INT1 is ccRCC
    parser.add_argument("--tech-type", default="Visium")
    parser.add_argument("--heldout-fraction", type=float, default=0.15)
    parser.add_argument("--heldout-seed", type=int, default=20260720)  # matches the 8-job batch
    parser.add_argument("--top-hvg", type=int, default=50,
                         help="Restrict PCC/RMSE to the top-N highly-variable genes after "
                              "log1p normalization -- matches STFlow's real published "
                              "evaluation convention (and most of this literature: STNet, "
                              "HisToGene, BLEEP, etc. all evaluate this way, not averaged "
                              "over the whole panel). Most of a ~16,500-gene panel is "
                              "lowly-expressed/near-constant across spots -- averaging PCC "
                              "over ALL of them dilutes whatever real signal exists in the "
                              "genes that actually vary spatially. Set to 0 to disable and "
                              "evaluate the full panel instead (the original behavior).")
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
    print(f"Loading {args.sample_id} (RAW counts -- see this script's own module docstring "
          f"for why, NOT this project's usual normalize_total+log1p pipeline)...")
    adata = _load_raw_counts_adata(cfg)
    print(f"  {adata.n_obs} spots, {adata.n_vars} genes (before H&E-patch alignment)")

    print("Loading H&E patches + Gigapath features...")
    patches, barcodes = load_hest_patches(args.hest_data_dir, args.sample_id)
    features = get_gigapath_features(cfg, patches, barcodes)
    # align_patches_to_adata drops spots with no matching H&E patch (a normal
    # partial gap in HEST-1k's own patch extraction) and returns a FILTERED
    # adata -- everything downstream (coords, expr, heldout_mask) must be
    # derived from THIS adata, not the original, or masks/arrays end up
    # different lengths (real bug hit on the first run of this script: a
    # 1080 vs 1031 shape mismatch in _images_tensor).
    adata, gigapath_images = align_patches_to_adata(adata, features, barcodes)

    coords3d = np.stack([adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1],
                          np.zeros(adata.n_obs)], axis=1).astype("float32")
    expr = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X, dtype="float32")
    gene_names = adata.var_names.tolist()
    print(f"  {adata.n_obs} spots after alignment, {len(gene_names)} genes")

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
    valid_gene_names = [gene_names[i] for i in encoder._valid_gene_pos]
    target = np.log1p(expr[query_mask][:, encoder._valid_gene_pos])

    eval_pred, eval_target, n_genes_used = raw_pred, target, len(valid_gene_names)
    if args.top_hvg > 0:
        import scanpy as sc
        # sc.pp.highly_variable_genes' default flavor ("seurat") expects
        # LOGARITHMIZED input -- adata here is raw counts (see this script's
        # own raw-counts fix above), so log1p a COPY first. Matches STPath's
        # own "log1pv2" convention (log1p of raw counts, no normalize_total)
        # for internal consistency with how STPath itself was evaluated,
        # not this project's usual normalize_total+log1p pipeline.
        hvg_adata = adata.copy()
        sc.pp.log1p(hvg_adata)
        sc.pp.highly_variable_genes(hvg_adata, n_top_genes=min(args.top_hvg, adata.n_vars))
        hvg_names = set(hvg_adata.var_names[hvg_adata.var["highly_variable"]].tolist())
        # restrict to whichever of the top-N HVGs ALSO had a match in STPath's own
        # vocabulary (valid_gene_names) -- can't score a gene STPath never predicted
        keep_idx = [i for i, g in enumerate(valid_gene_names) if g in hvg_names]
        print(f"  top-{args.top_hvg} HVGs: {len(keep_idx)}/{args.top_hvg} also matched "
              f"STPath's vocabulary (the rest of the {args.top_hvg} weren't in "
              f"_valid_gene_pos, see the 'symbols not in the tokenizer' warning above)")
        eval_pred, eval_target, n_genes_used = raw_pred[:, keep_idx], target[:, keep_idx], len(keep_idx)

    pcc = np.nanmean(ev.pearson_per_gene(eval_pred, eval_target))
    rmse = ev.rmse(eval_pred, eval_target)
    print("")
    hvg_label = f"top-{args.top_hvg} HVGs" if args.top_hvg > 0 else "full panel"
    print(f"=== RAW STPath inference (no training on our data at all) -- {hvg_label} ===")
    print(f"mean PCC:  {pcc:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"n_query:   {query_mask.sum()} spots, {n_genes_used} genes")
    print("")
    print("Compare directly against the held-out generalization batch's numbers "
          "(same split, same spots, though THOSE were full-panel PCC, not HVG-restricted --"
          " re-run that batch's eval with the same HVG restriction for a truly fair "
          "comparison): standard-eval PCC ~0.25-0.45 (memorized, full-panel), held-out "
          "PCC ~0.02 (our own trained models on unseen spots, full-panel). If this raw "
          "STPath HVG number is meaningfully above ~0.02, STPath's own pretrained "
          "in-context mechanism IS doing real work our wrapping pipeline discards or "
          "fails to exploit. If it's ALSO near zero, the bottleneck is more fundamental "
          "than just gene selection.")


if __name__ == "__main__":
    main()
