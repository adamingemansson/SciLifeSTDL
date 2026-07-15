"""
Full model comparison (docs/architecture_plan.md, docs/model_schematics.md
step 9, task #15): trains VAE (floor), WAE-GAN, FM-OT, and VQ-VAE+AR on the
same pilot sample with each model's own tuned config
(configs/exp_hest1k_*.yaml), evaluating each one (plus the zero-parameter
interp_baseline, added for free) on the SAME held-out masking draw (a seed
disjoint from every model's own training seeds), so the comparison is
fair — then FREES it before training the next one (2026-07-15: an earlier
version accumulated every trained model in memory and only evaluated them
all at the end, which crashed on a real machine once several
Gigapath/STPath-backed configs — each carrying a ~4.4GB frozen encoder —
were resident simultaneously; see _free()/_evaluate() below). Reports the
full metric suite in one table:
pointwise (PCC, RMSE, nonzero AUC), spatial-arrangement-aware
distributional (ST-FID via pool_knn_neighborhood + PCA, task #14), and
cell-type plausibility (task #13, Leiden pseudo-labels since INT1 has no
curated cell-type column).

Run with:
    python -m src.evaluation.run_comparison \\
        configs/exp_hest1k_vae_baseline.yaml \\
        configs/exp_hest1k_wae_gan.yaml \\
        configs/exp_hest1k_fm_ot.yaml \\
        configs/exp_hest1k_vqvae_ar.yaml

H&E (task #17/#20): configs may set data.use_images: true to also load
HEST-1k's H&E patches (needs model.params.image_encoder_type set to "cnn"
or "gigapath" too, or the images are loaded but unused). All configs
passed to one invocation should agree on data.use_images (and point at
the same underlying sample) — this script loads images once, from the
first config, and reuses them for the shared held-out eval draw; it does
NOT support mixing image-enabled and expression-only configs within a
single comparison run. Run expression-only and H&E sweeps as separate
invocations instead.
"""
from __future__ import annotations
import argparse
import gc
import os

# Same fix as src/training/train.py (2026-07-15): must be set before any
# MPS op is dispatched in the process, not just before torch.linalg.eigh
# (STPath's own SpatialTransformer, see src/models/stpath_encoder.py)
# actually runs. Importing src.training.train below sets this too, but
# that's an incidental side effect of import order — set it explicitly
# here so this script doesn't depend on that staying true.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.models.registry import build_model
from src.training.train import (
    load_adata, make_context_query_split, MaskedContextQueryDataset,
    _collate_identity, _load_images, _images_tensor, inject_stpath_gene_names,
)
from src.evaluation import metrics as ev
from src.evaluation.cell_type_classifier import cluster_pseudo_labels, CellTypePlausibilityClassifier

EVAL_SEED = 999_999  # disjoint from every per-model config's own training seed range


def _train_model(cfg_path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        # dotlist overrides applied to EVERY config in this comparison run,
        # e.g. ["training.epochs=2"] for a quick smoke test (2026-07-15,
        # checking all 18 task #19 configs run end-to-end before committing
        # to full-length training on each)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    torch.manual_seed(cfg.training.seed)
    adata = load_adata(cfg)
    from src.data import loaders
    # adata may come back as a SUBSET (align_patches_to_adata drops spots
    # with no matching H&E patch, a normal gap, not an error) - always use
    # the returned adata downstream, not the pre-image-loading one
    adata, images = _load_images(cfg, adata)  # images is None unless cfg.data.use_images is set
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_stpath_gene_names(model_cfg, adata)
    model = build_model(model_cfg)

    if list(model.parameters()):
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed, images=images,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=_collate_identity)
        trainer = pl.Trainer(
            max_epochs=1, accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False, logger=False,
        )
        trainer.fit(model, dataloader)

    model.eval()
    return model, cfg, adata, coords3d, expr, slice_ids, images


def _free(model) -> None:
    """Release a trained model's memory before moving to the next config.
    Added 2026-07-15 after a real RAM crash: this function used to just
    accumulate every trained model in a dict and evaluate them all at the
    very end, so N image-heavy configs (each carrying a ~4.4GB frozen
    Gigapath copy) stayed resident simultaneously — see the module
    docstring's H&E section and stpath_encoder.py's tile_encoder comment
    for the other half of this fix. `del` + `gc.collect()` alone doesn't
    reliably free GPU/MPS-resident tensors, hence the explicit cache
    clears below."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _evaluate(model, shared_eval: dict) -> tuple:
    with torch.no_grad():
        out = model.sample(shared_eval["context"], shared_eval["query"])
    pred = out["expression"].detach().cpu().numpy()
    target_expression = shared_eval["target_expression"]

    pcc = np.nanmean(ev.pearson_per_gene(pred, target_expression))
    rmse = ev.rmse(pred, target_expression)
    auc = ev.nonzero_auc(pred, target_expression)

    gen_patches = ev.pool_knn_neighborhood(
        shared_eval["coords3d"][shared_eval["query_mask"]], pred, k=shared_eval["query_k"]
    )
    fid = ev.st_fid(shared_eval["real_embed"], shared_eval["pca"].transform(gen_patches))

    plausibility = shared_eval["clf"].plausibility_accuracy(pred, shared_eval["true_query_labels"])
    return pcc, rmse, auc, fid, plausibility


def _build_shared_eval(cfg, adata, coords3d, expr, slice_ids, images,
                        k_neighborhood: int, pca_components: int) -> dict:
    """Built once, from the FIRST config's data — see module docstring on
    why mixed image-enabled/expression-only runs aren't supported. Every
    later config's own (coords3d, expr, slice_ids) must match this one for
    the shared held-out draw to mean the same thing across models (true
    whenever every config in one invocation points at the same sample)."""
    context_mask, query_mask = make_context_query_split(coords3d, slice_ids, cfg.masking, EVAL_SEED)
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(expr[context_mask], dtype=torch.float32),
    }
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    if images is not None:
        context["images"] = _images_tensor(images, context_mask)
        query["images"] = _images_tensor(images, query_mask)
    target_expression = expr[query_mask]

    # cell-type plausibility (task #13): Leiden pseudo-labels over the
    # whole real sample, classifier trained only on this draw's context
    labels_all = cluster_pseudo_labels(adata)
    clf = CellTypePlausibilityClassifier(seed=0).fit(expr[context_mask], labels_all[context_mask])
    true_query_labels = labels_all[query_mask]

    # ST-FID (task #14): patch-pooled PCA fit on real context data only
    from sklearn.decomposition import PCA
    context_patches = ev.pool_knn_neighborhood(coords3d[context_mask], expr[context_mask], k=k_neighborhood)
    n_components = min(pca_components, context_patches.shape[0] - 1, context_patches.shape[1])
    pca = PCA(n_components=n_components).fit(context_patches)
    query_k = min(k_neighborhood, int(query_mask.sum()))
    real_patches = ev.pool_knn_neighborhood(coords3d[query_mask], target_expression, k=query_k)
    real_embed = pca.transform(real_patches)

    return {
        "context": context, "query": query, "target_expression": target_expression,
        "clf": clf, "true_query_labels": true_query_labels,
        "pca": pca, "real_embed": real_embed, "query_k": query_k,
        "coords3d": coords3d, "query_mask": query_mask,
    }


def main(model_config_paths: list[str], k_neighborhood: int = 8, pca_components: int = 10,
         overrides: list[str] | None = None):
    rows = []
    interp_row = None
    shared_eval = None

    for path in model_config_paths:
        model, cfg, adata, coords3d, expr, slice_ids, images = _train_model(path, overrides)

        if shared_eval is None:
            shared_eval = _build_shared_eval(
                cfg, adata, coords3d, expr, slice_ids, images, k_neighborhood, pca_components
            )
            interp_model = build_model({"name": "interp_baseline", "params": {}})
            interp_row = ("interp_baseline", *_evaluate(interp_model, shared_eval))
            _free(interp_model)

        # keyed by experiment_name, not cfg.model.name: multiple configs can
        # share a registered model name (e.g. fm_ot's OT and EDM path_type
        # variants both register as "fm_ot") and must stay distinct rows
        rows.append((cfg.experiment_name, *_evaluate(model, shared_eval)))
        _free(model)  # evaluate-then-free, not accumulate-then-evaluate — see _free's comment
        print(f"trained + evaluated: {cfg.experiment_name}")

    rows.append(interp_row)

    header = f"{'model':<18}{'PCC':>10}{'RMSE':>10}{'AUC':>10}{'ST-FID':>10}{'plausible':>12}"
    print("\n" + header)
    print("-" * len(header))
    for name, pcc, rmse, auc, fid, plaus in rows:
        print(f"{name:<18}{pcc:>10.4f}{rmse:>10.4f}{auc:>10.4f}{fid:>10.4f}{plaus:>12.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=str,
                         help="paths to trainable models' configs (VAE/WAE-GAN/FM-OT/VQ-VAE+AR); "
                              "interp_baseline is added automatically")
    parser.add_argument("--override", nargs="*", default=[],
                         help="dotlist overrides applied to EVERY config, "
                              "e.g. --override training.epochs=2 for a quick smoke test")
    args = parser.parse_args()
    main(args.configs, overrides=args.override)
