"""
Full model comparison (docs/architecture_plan.md, docs/model_schematics.md
step 9, task #15): trains VAE (floor), WAE-GAN, FM-OT, and VQ-VAE+AR on the
same pilot sample with each model's own tuned config
(configs/exp_hest1k_*.yaml), then evaluates ALL of them — plus the
zero-parameter interp_baseline, added for free — on the SAME held-out
masking draw (a seed disjoint from every model's own training seeds), so
the comparison is fair. Reports the full metric suite in one table:
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


def _train_model(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
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


def main(model_config_paths: list[str], k_neighborhood: int = 8, pca_components: int = 10):
    trained = {}
    shared = None
    for path in model_config_paths:
        model, cfg, adata, coords3d, expr, slice_ids, images = _train_model(path)
        # keyed by experiment_name, not cfg.model.name: multiple configs can
        # share a registered model name (e.g. fm_ot's OT and EDM path_type
        # variants both register as "fm_ot") and must stay distinct rows
        trained[cfg.experiment_name] = model
        if shared is None:
            # images taken from the FIRST config only — see module docstring
            # on why mixed image-enabled/expression-only runs aren't supported
            shared = (adata, coords3d, expr, slice_ids, cfg.masking, images)
        print(f"trained: {cfg.experiment_name}")

    adata, coords3d, expr, slice_ids, masking_cfg, images = shared
    trained["interp_baseline"] = build_model({"name": "interp_baseline", "params": {}})

    # one shared held-out draw, used identically for every model
    context_mask, query_mask = make_context_query_split(coords3d, slice_ids, masking_cfg, EVAL_SEED)
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

    rows = []
    for name, model in trained.items():
        with torch.no_grad():
            out = model.sample(context, query)
        pred = out["expression"].detach().cpu().numpy()

        pcc = np.nanmean(ev.pearson_per_gene(pred, target_expression))
        rmse = ev.rmse(pred, target_expression)
        auc = ev.nonzero_auc(pred, target_expression)

        gen_patches = ev.pool_knn_neighborhood(coords3d[query_mask], pred, k=query_k)
        fid = ev.st_fid(real_embed, pca.transform(gen_patches))

        plausibility = clf.plausibility_accuracy(pred, true_query_labels)
        rows.append((name, pcc, rmse, auc, fid, plausibility))

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
    args = parser.parse_args()
    main(args.configs)
