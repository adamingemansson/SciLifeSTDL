"""
Full model comparison (docs/architecture_plan.md, docs/model_schematics.md
step 9, task #15): trains VAE (floor), WAE-GAN, FM-OT, and VQ-VAE+AR on the
same pilot sample with each model's own tuned config
(configs/exp_hest1k_*.yaml), evaluating each one (plus the zero-parameter
interp_baseline, added for free) on the SAME held-out masking draw (a seed
disjoint from every model's own training seeds), so the comparison is
fair — then FREES it (`del model` in main() itself, not inside a helper —
see _release_torch_memory's docstring for a real Python-scoping bug found
here) before training the next one (2026-07-15: an earlier version
accumulated every trained model in memory and only evaluated them all at
the end, which crashed on a real machine once several Gigapath/STPath-
backed configs — each carrying a ~4.4GB frozen encoder — were resident
simultaneously). Reports the full metric suite in one table:
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

H&E (task #17/#18/#20): configs may set data.use_images: true to also load
HEST-1k's H&E patches (needs model.params.image_encoder_type set to "cnn"
or "gigapath", or context_encoder_type set to "stpath" — or the images
are loaded but unused). All configs passed to one invocation should agree
on data.use_images (and point at the same underlying sample); it does NOT
support mixing image-enabled and expression-only configs within a single
comparison run — run expression-only and H&E sweeps as separate
invocations. WITHIN one H&E-enabled invocation, mixing CNN/Gigapath/STPath
configs together IS supported (that's the point of task #19's matrix):
_build_shared_eval computes both raw patches and precomputed Gigapath
features once, and each model gets whichever format it actually expects
(see _images_for_model) — an earlier version only had whichever format
the first config's own training happened to produce, which crashed on
real hardware when a Gigapath/STPath model got raw patches instead
(2026-07-15, see _images_for_model's docstring).
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
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.models.registry import build_model
from src.training.train import (
    load_adata, make_context_query_split, MaskedContextQueryDataset,
    _load_images, _images_tensor, inject_stpath_gene_names,
    get_gigapath_features, save_trained_model, make_dataloader,
    _downsample_patches, get_novae_features, inject_novae_dim, inject_stpath_novae_dim,
    inject_coord_scale, inject_decoder_gene_names, PeriodicCheckpointCallback,
)
from src.evaluation import metrics as ev
from src.evaluation.cell_type_classifier import cluster_pseudo_labels, CellTypePlausibilityClassifier

EVAL_SEED = 999_999  # disjoint from every per-model config's own training seed range


def _cached_load_adata(cfg, adata_cache: dict):
    """load_adata(cfg), cached across configs within ONE
    run_comparison.py invocation. Real inefficiency found 2026-07-15 (a
    long gap observed between models starting during a real run on an
    SSH session): _train_model used to call load_adata(cfg) fresh for
    EVERY config — rereading the .h5ad from disk and rerunning scanpy's
    full QC/normalize pipeline every single time — even though every
    config in one invocation is REQUIRED to point at the same underlying
    sample (this module's own docstring), so the result is always
    identical. Worse on NFS-mounted storage (a real case hit — repeated
    file reads carry more latency there than local disk).

    Safe to return the SAME object, not a fresh .copy() per use: nothing
    downstream mutates it in place — _load_images/align_patches_to_adata
    return NEW filtered objects via their own .copy() calls, and
    cluster_pseudo_labels copies internally too
    (cell_type_classifier.py). If a future change ever DOES mutate an
    AnnData in place somewhere downstream, that mutation would leak
    across configs sharing this cache — worth keeping in mind before
    adding any such code."""
    if cfg.data.get("source") == "hest1k":
        key = ("hest1k", str(cfg.data.hest_data_dir), cfg.data.sample_id,
               cfg.data.min_genes, cfg.data.min_cells)
    else:
        key = ("multi_slice", tuple(cfg.data.paths), tuple(cfg.data.get("z_positions") or []),
               cfg.data.min_genes, cfg.data.min_cells)
    if key not in adata_cache:
        adata_cache[key] = load_adata(cfg)
    return adata_cache[key]


def _train_model(cfg_path: str, overrides: list[str] | None = None, adata_cache: dict | None = None,
                  skip_training: bool = False):
    """NOTE (2026-07-16): single-sample only — does not yet support
    cfg.data.sample_ids (multi-sample training + organ/tech conditioning,
    see src/training/train.py's _main_multi_sample/load_multi_sample_data).
    This function's shared-eval-set/adata-caching machinery below assumes
    exactly one adata per config; extending it to multi-sample comparisons
    is real follow-up work, not done here — use
    `python -m src.training.train --config <multisample config>` directly
    for now (see configs/exp_hest1k_fm_ot_multisample.yaml)."""
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        # dotlist overrides applied to EVERY config in this comparison run,
        # e.g. ["training.epochs=2"] for a quick smoke test (2026-07-15,
        # checking all 18 task #19 configs run end-to-end before committing
        # to full-length training on each)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    torch.manual_seed(cfg.training.seed)
    adata = _cached_load_adata(cfg, adata_cache if adata_cache is not None else {})
    from src.data import loaders
    # adata may come back as a SUBSET (align_patches_to_adata drops spots
    # with no matching H&E patch, a normal gap, not an error) - always use
    # the returned adata downstream, not the pre-image-loading one
    adata, images = _load_images(cfg, adata)  # images is None unless cfg.data.use_images is set
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()

    # gene_encoder_type="novae" (2026-07-15) / stpath_new_gene_encoder_type=
    # "novae" (2026-07-16, Route B): same reasoning as
    # src/training/train.py's main() — precompute once, row-aligned to
    # THIS config's (possibly image-QC-filtered) adata, before building
    # the model, so the right *_novae_dim can be injected into model_cfg.
    train_model_params = cfg.model.get("params", {})
    train_context_encoder_type = train_model_params.get("context_encoder_type", "builtin")
    context_gene_features = None
    context_novae_features = None
    if train_context_encoder_type == "builtin" and train_model_params.get("gene_encoder_type") == "novae":
        context_gene_features = get_novae_features(cfg, adata)
    elif (train_context_encoder_type == "stpath"
          and train_model_params.get("stpath_new_gene_encoder_type") in ("novae", "both")):
        context_novae_features = get_novae_features(cfg, adata)
    elif (train_context_encoder_type == "storm_lite"
          and train_model_params.get("gene_encoder_type") in ("novae", "both")):
        context_novae_features = get_novae_features(cfg, adata)

    # .get() with the same default every real config's own YAML comment
    # documents, not a bare attribute access — configs that don't declare
    # checkpoint_dir (e.g. this file's own test_run_comparison.py synthetic
    # configs) must still work, not crash on a missing key
    checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")

    if skip_training:
        # Re-evaluate an ALREADY-trained model without retraining
        # (2026-07-15, added specifically so a new/expanded metric set -
        # e.g. st_mmd - can be checked against real completed multi-
        # thousand-epoch runs in minutes, not by re-running training from
        # scratch). Requires this config's checkpoint_dir to already have
        # a real checkpoint from an earlier save_trained_model() call —
        # raises a clear FileNotFoundError (via load_trained_model) if
        # not, rather than silently falling back to training.
        from src.training.train import load_trained_model
        print(f"--skip-training: loading {cfg.experiment_name} from {checkpoint_dir} (not retraining)")
        model, _gene_names = load_trained_model(checkpoint_dir)
        model.eval()
        return model, cfg, adata, coords3d, expr, slice_ids, images

    # 2026-07-17: RandomFourierFeatures real-scale bug fix — see
    # inject_coord_scale's own docstring
    coord_scale = float(coords3d[:, :2].std())

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_stpath_gene_names(model_cfg, adata)
    inject_decoder_gene_names(model_cfg, adata)
    inject_coord_scale(model_cfg, coord_scale)
    if context_gene_features is not None:
        inject_novae_dim(model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        if train_context_encoder_type == "storm_lite":
            inject_novae_dim(model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(model_cfg, context_novae_features.shape[1])
    model = build_model(model_cfg)
    # UNRESOLVED copy, saved (not model_cfg above) so a STPath config's
    # ${oc.env:STPATH_GENE_VOC_PATH}/${oc.env:STPATH_MODEL_WEIGHT_PATH}
    # interpolations stay literal in the checkpoint rather than getting
    # baked in as THIS machine's resolved path — see
    # train.py load_trained_model's docstring for the real cross-machine
    # bug this fixes (2026-07-15).
    unresolved_model_cfg = OmegaConf.to_container(cfg.model, resolve=False)
    inject_stpath_gene_names(unresolved_model_cfg, adata)
    inject_decoder_gene_names(unresolved_model_cfg, adata)
    inject_coord_scale(unresolved_model_cfg, coord_scale)
    if context_gene_features is not None:
        inject_novae_dim(unresolved_model_cfg, context_gene_features.shape[1])
    if context_novae_features is not None:
        if train_context_encoder_type == "storm_lite":
            inject_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])
        else:
            inject_stpath_novae_dim(unresolved_model_cfg, context_novae_features.shape[1])

    if list(model.parameters()):
        # augment_coords (2026-07-16, src/data/augmentation.py) applies
        # ONLY to the training draws here, never to the shared held-out
        # eval set built elsewhere in this file (that set must stay fixed/
        # unaugmented so every model in a comparison run is scored against
        # the exact same query points — see this module's own shared-eval
        # comments).
        # organ/tech (2026-07-17, real bug fix — see _build_shared_eval's
        # own comment on this same fix for the full reasoning: these were
        # NEVER populated anywhere in this file before, silently leaving
        # context["tech"]/query["tech"] None for every training step of
        # every config ever run through this script).
        organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
        tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed, images=images,
            context_gene_features=context_gene_features,
            context_novae_features=context_novae_features,
            organ=organ, tech=tech,
            augment=cfg.training.get("augment_coords", False),
        )
        dataloader = make_dataloader(dataset, cfg)
        # PeriodicCheckpointCallback (2026-07-17): opt-in via
        # training.checkpoint_every_n_steps, unset by default — see that
        # class's own docstring.
        callbacks = []
        checkpoint_every_n_steps = cfg.training.get("checkpoint_every_n_steps")
        if checkpoint_every_n_steps:
            callbacks.append(PeriodicCheckpointCallback(
                unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir,
                save_every_n_steps=checkpoint_every_n_steps,
            ))
        trainer = pl.Trainer(
            max_epochs=1, accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False, logger=False,
            callbacks=callbacks,
        )
        trainer.fit(model, dataloader)
        saved_path = save_trained_model(model, unresolved_model_cfg, adata.var_names.tolist(), checkpoint_dir)
        if saved_path is not None:
            print(f"Saved trained model (weights + config + gene names) to {saved_path.parent}")

    model.eval()
    return model, cfg, adata, coords3d, expr, slice_ids, images


def _release_torch_memory() -> None:
    """gc.collect() + device cache clear. Real bug found 2026-07-15 in an
    earlier version of this fix: it was a function `_free(model)` that
    called `del model` on its OWN parameter — that only removes the
    binding inside THAT function's stack frame, not the caller's. Python
    has no block scoping, so main()'s own `model` variable (assigned once
    per loop iteration) stayed alive for the rest of the function
    regardless of calling that helper — meaning gc.collect()/empty_cache()
    ran while the object was still referenced and could free nothing,
    every single call. The actual `del model` MUST happen in the caller's
    own scope (see main() below) — a callee can never delete a variable
    it doesn't own. Real symptom that exposed this: the user watched RAM
    climb to 10GB across just 4 small expression-only models (tens of MB
    each), which this bug alone doesn't fully explain either — worth
    re-checking after this fix whether growth continues or plateaus."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _images_for_model(model_params: dict, shared_eval: dict):
    """Pick the image FORMAT this particular model actually expects.
    Real bug found 2026-07-15: a single comparison run can mix CNN,
    Gigapath, and STPath configs (that's the whole point of task #19's
    matrix) — CNN wants raw patches, Gigapath/STPath want precomputed
    features. Blindly reusing whatever format the FIRST config's own
    training happened to produce meant a Gigapath/STPath model sometimes
    got raw patches at eval time, which forced its encoder into an
    unbatched full-ViT forward pass over the whole ~700-900 point eval
    set at once (no batching, unlike precompute_gigapath_features) — a
    real ~25GB RAM crash. See _build_shared_eval, which now computes
    BOTH formats once up front so this never happens."""
    if shared_eval.get("raw_images") is None and shared_eval.get("gigapath_images") is None:
        return None
    image_encoder_type = model_params.get("image_encoder_type", "none")
    context_encoder_type = model_params.get("context_encoder_type", "builtin")
    if image_encoder_type == "none" and context_encoder_type not in ("stpath", "storm_lite"):
        return None  # this model doesn't use images at all (e.g. interp_baseline)
    needs_gigapath = image_encoder_type == "gigapath" or context_encoder_type in ("stpath", "storm_lite")
    return shared_eval["gigapath_images"] if needs_gigapath else shared_eval["raw_images"]


def _gene_features_for_model(model_params: dict, shared_eval: dict):
    """Pick the gene-expression FEATURE FORMAT this particular model
    expects for its context — same reasoning/pattern as
    _images_for_model: a single comparison run can mix gene_encoder_type
    values across configs ("raw"/"mlp" want raw expression fed straight
    in; "novae" wants shared_eval's precomputed Novae features), so
    shared_eval carries both and each model picks its own. Returns None
    for "raw"/"mlp" (meaning: use raw expr, exactly as before this switch
    existed) or shared_eval["novae_features"] for "novae" — ONLY for
    context_encoder_type="builtin" (default): storm_lite reuses the same
    gene_encoder_type param name but needs the ADDITIVE channel instead
    (see _stpath_novae_features_for_model below), never a replacement."""
    if (model_params.get("context_encoder_type", "builtin") == "builtin"
            and model_params.get("gene_encoder_type") == "novae"):
        return shared_eval["novae_features"]
    return None


def _stpath_novae_features_for_model(model_params: dict, shared_eval: dict):
    """ADDITIVE Novae-features counterpart to _gene_features_for_model
    above (2026-07-16) — genuinely different from it, not reusable: this
    is an ADDITIVE channel (context["novae_features"]), never a
    replacement for context["expression"] (STPath's own gene_embed
    pathway, and StormLiteContextEncoder's own raw-expression path in
    "both" mode, both always need real raw expression too — see
    _build_masked_item's context_novae_features docstring in train.py
    for the full reasoning). Covers TWO real callers despite the STPath-
    specific name (kept for git history continuity, not renamed): STPath's
    Route-B residual (context_encoder_type="stpath" +
    stpath_new_gene_encoder_type in ("novae","both")) AND
    StormLiteContextEncoder (context_encoder_type="storm_lite" +
    gene_encoder_type in ("novae","both"))."""
    if (model_params.get("context_encoder_type") == "stpath"
            and model_params.get("stpath_new_gene_encoder_type") in ("novae", "both")):
        return shared_eval["novae_features"]
    if (model_params.get("context_encoder_type") == "storm_lite"
            and model_params.get("gene_encoder_type") in ("novae", "both")):
        return shared_eval["novae_features"]
    return None


def _evaluate(model, model_params: dict, shared_eval: dict) -> tuple:
    context_mask, query_mask = shared_eval["context_mask"], shared_eval["query_mask"]
    coords3d, expr = shared_eval["coords3d"], shared_eval["expr"]
    gene_features = _gene_features_for_model(model_params, shared_eval)
    context_expr_source = expr if gene_features is None else gene_features
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(context_expr_source[context_mask], dtype=torch.float32),
    }
    stpath_novae_features = _stpath_novae_features_for_model(model_params, shared_eval)
    if stpath_novae_features is not None:
        context["novae_features"] = torch.tensor(
            stpath_novae_features[context_mask], dtype=torch.float32
        )
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    context["organ"], context["tech"] = shared_eval["organ"], shared_eval["tech"]
    query["organ"], query["tech"] = shared_eval["organ"], shared_eval["tech"]
    images = _images_for_model(model_params, shared_eval)
    if images is not None:
        context["images"] = _images_tensor(images, context_mask)
        query["images"] = _images_tensor(images, query_mask)

    with torch.no_grad():
        out = model.sample(context, query)
    pred = out["expression"].detach().cpu().numpy()
    target_expression = shared_eval["target_expression"]

    pcc = np.nanmean(ev.pearson_per_gene(pred, target_expression))
    rmse = ev.rmse(pred, target_expression)
    auc = ev.nonzero_auc(pred, target_expression)

    gen_patches = ev.pool_knn_neighborhood(coords3d[query_mask], pred, k=shared_eval["query_k"])
    gen_embed = shared_eval["pca"].transform(gen_patches)
    fid = ev.st_fid(shared_eval["real_embed"], gen_embed)
    # MMD (2026-07-15): drops st_fid's Gaussian assumption on the pooled
    # PCA embedding space - same real_embed/gen_embed pair, so this is
    # free (no extra forward pass, no extra fitting), and comparing the
    # two numbers tells you whether that Gaussian assumption is actually
    # reasonable for this data or distorting the ST-FID number.
    mmd = ev.st_mmd(shared_eval["real_embed"], gen_embed)

    plausibility = shared_eval["clf"].plausibility_accuracy(pred, shared_eval["true_query_labels"])
    return pcc, rmse, auc, fid, mmd, plausibility


def _evaluate_shuffled_images(model, model_params: dict, shared_eval: dict, seed: int = 0) -> tuple | None:
    """Diagnostic (2026-07-16, from project research into why CNN/Gigapath
    conditioning showed weak/inconsistent PCC effects): randomly permute
    WHICH query location gets which histology patch at eval time, keeping
    everything else (context, coords, real target expression) identical,
    then re-run _evaluate's own metric computation. If a model is
    genuinely using its query image to predict that query's expression,
    shuffling should make PCC/RMSE meaningfully worse. If PCC barely
    moves, the model is ignoring its image conditioning regardless of
    which image encoder it uses — disambiguates "the image encoder is
    weak" from "the fusion mechanism never learns to use images at all".

    Only shuffles QUERY images (not context) — context images/expression
    stay real/aligned throughout, since the question is specifically
    whether a PREDICTION at query i reflects query i's own histology, not
    whether the model can use context information at all. Returns None
    for models that don't use images (nothing to shuffle)."""
    images = _images_for_model(model_params, shared_eval)
    if images is None:
        return None
    context_mask, query_mask = shared_eval["context_mask"], shared_eval["query_mask"]
    coords3d = shared_eval["coords3d"]
    gene_features = _gene_features_for_model(model_params, shared_eval)
    context_expr_source = shared_eval["expr"] if gene_features is None else gene_features

    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(context_expr_source[context_mask], dtype=torch.float32),
        "images": _images_tensor(images, context_mask),
    }
    stpath_novae_features = _stpath_novae_features_for_model(model_params, shared_eval)
    if stpath_novae_features is not None:
        context["novae_features"] = torch.tensor(
            stpath_novae_features[context_mask], dtype=torch.float32
        )
    query_images = _images_tensor(images, query_mask)
    perm = np.random.default_rng(seed).permutation(query_images.shape[0])
    query = {
        "coords": torch.tensor(coords3d[query_mask], dtype=torch.float32),
        "images": query_images[perm],
    }
    context["organ"], context["tech"] = shared_eval["organ"], shared_eval["tech"]
    query["organ"], query["tech"] = shared_eval["organ"], shared_eval["tech"]

    with torch.no_grad():
        out = model.sample(context, query)
    pred = out["expression"].detach().cpu().numpy()
    target_expression = shared_eval["target_expression"]

    pcc = np.nanmean(ev.pearson_per_gene(pred, target_expression))
    rmse = ev.rmse(pred, target_expression)
    auc = ev.nonzero_auc(pred, target_expression)
    gen_patches = ev.pool_knn_neighborhood(coords3d[query_mask], pred, k=shared_eval["query_k"])
    gen_embed = shared_eval["pca"].transform(gen_patches)
    fid = ev.st_fid(shared_eval["real_embed"], gen_embed)
    mmd = ev.st_mmd(shared_eval["real_embed"], gen_embed)
    plausibility = shared_eval["clf"].plausibility_accuracy(pred, shared_eval["true_query_labels"])
    return pcc, rmse, auc, fid, mmd, plausibility


def _cnn_image_patch_size(model_config_paths: list[str], overrides: list[str] | None) -> int | None:
    """Scan every config in this invocation for a CNN-branch
    image_patch_size (every he_cnn config in this project uses the same
    value). Needed so _build_shared_eval's raw_images fallback reload
    (triggered when the FIRST config in the list isn't itself CNN)
    downsamples to the SAME resolution the CNN was actually trained on
    (see _downsample_patches, train.py) — not the native 224x224, which
    would be a real train/eval resolution mismatch. Returns None if no
    config in this invocation uses "cnn" (nothing to match)."""
    for path in model_config_paths:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        params = cfg.model.get("params", {})
        if params.get("image_encoder_type") == "cnn":
            return params.get("image_patch_size", 224)
    return None


def _any_config_uses_novae(model_config_paths: list[str], overrides: list[str] | None) -> bool:
    """Scan every config in this invocation for gene_encoder_type: "novae"
    (builtin encoder) OR stpath_new_gene_encoder_type: "novae" (STPath
    Route-B residual, 2026-07-16) — same "scan all configs before building
    shared_eval" pattern as _cnn_image_patch_size, for the same reason:
    shared_eval is built once from the FIRST config, but a later config in
    the same run may be the one that actually needs Novae features."""
    for path in model_config_paths:
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        params = cfg.model.get("params", {})
        params_context_encoder_type = params.get("context_encoder_type", "builtin")
        if (params_context_encoder_type == "builtin" and params.get("gene_encoder_type") == "novae"):
            return True
        if (params_context_encoder_type == "stpath"
                and params.get("stpath_new_gene_encoder_type") in ("novae", "both")):
            return True
        if (params_context_encoder_type == "storm_lite"
                and params.get("gene_encoder_type") in ("novae", "both")):
            return True
    return False


def _build_shared_eval(cfg, adata, coords3d, expr, slice_ids, images,
                        k_neighborhood: int, pca_components: int,
                        cnn_patch_size: int | None = None,
                        compute_novae: bool = False) -> dict:
    """Built once, from the FIRST config's data — see module docstring on
    why mixed image-enabled/expression-only runs aren't supported. Every
    later config's own (coords3d, expr, slice_ids) must match this one for
    the shared held-out draw to mean the same thing across models (true
    whenever every config in one invocation points at the same sample).

    Computes BOTH raw patches and precomputed Gigapath features when
    images are enabled at all — not just whichever format the first
    config's own training happened to use — since later configs in the
    same run may need the other format (see _images_for_model). Same
    reasoning for compute_novae/novae_features (see
    _any_config_uses_novae/_gene_features_for_model) — Novae is
    independent of images entirely (a pure gene-expression encoder), so
    it's computed based on its own scan of the invocation's configs, not
    tied to whether images are enabled."""
    context_mask, query_mask = make_context_query_split(coords3d, slice_ids, cfg.masking, EVAL_SEED)
    target_expression = expr[query_mask]

    novae_features = get_novae_features(cfg, adata) if compute_novae else None

    raw_images, gigapath_images = None, None
    if images is not None:
        if images.ndim == 4:
            raw_images = images
        else:
            gigapath_images = images
        if raw_images is None or gigapath_images is None:
            from src.data import loaders as _loaders
            patches, barcodes = _loaders.load_hest_patches(cfg.data.hest_data_dir, cfg.data.sample_id)
            if raw_images is None:
                _, raw_images = _loaders.align_patches_to_adata(adata, patches, barcodes)
                if cnn_patch_size is not None and cnn_patch_size != raw_images.shape[1]:
                    raw_images = _downsample_patches(raw_images, cnn_patch_size)
            if gigapath_images is None:
                features = get_gigapath_features(cfg, patches, barcodes)
                _, gigapath_images = _loaders.align_patches_to_adata(adata, features, barcodes)

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
        "context_mask": context_mask, "query_mask": query_mask,
        "coords3d": coords3d, "expr": expr, "target_expression": target_expression,
        "raw_images": raw_images, "gigapath_images": gigapath_images,
        "novae_features": novae_features,
        "clf": clf, "true_query_labels": true_query_labels,
        "pca": pca, "real_embed": real_embed, "query_k": query_k,
        # 2026-07-17: real bug fix — organ/tech were NEVER populated
        # anywhere in this file, meaning context["tech"]/query["tech"]
        # were always None for every config ever run through this script,
        # regardless of whether adata.obs actually had real values (HEST-1k
        # samples always have "tech"="Visium"). Harmless for every
        # existing consumer (OrganTechEmbedding is gated on organ_vocab/
        # tech_vocab being set at construction, which no single-sample
        # config here does) but a REAL crash for LLOKIStyleDecoder
        # (decoder_type="lloki"), whose forward() requires a valid,
        # non-None tech string — this is what actually broke it on real
        # data (see docs/results_log.md's 2026-07-17 entry).
        "organ": str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None,
        "tech": str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None,
    }


def main(model_config_paths: list[str], k_neighborhood: int = 8, pca_components: int = 10,
         overrides: list[str] | None = None, skip_training: bool = False,
         shuffle_diagnostic: bool = False):
    # 2026-07-16: same reasoning as train.py's main() — free Tensor Core
    # speedup on Ampere+ GPUs, set once here rather than per-config inside
    # _train_model (it's a process-wide torch setting, not per-model).
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    rows = []
    shuffle_rows = []
    interp_row = None
    shared_eval = None
    adata_cache: dict = {}  # shared across every config in this invocation — see _cached_load_adata
    cnn_patch_size = _cnn_image_patch_size(model_config_paths, overrides)
    uses_novae = _any_config_uses_novae(model_config_paths, overrides)

    for path in model_config_paths:
        model, cfg, adata, coords3d, expr, slice_ids, images = _train_model(
            path, overrides, adata_cache, skip_training
        )
        model_params = cfg.model.get("params", {})

        if shared_eval is None:
            shared_eval = _build_shared_eval(
                cfg, adata, coords3d, expr, slice_ids, images, k_neighborhood, pca_components,
                cnn_patch_size, compute_novae=uses_novae,
            )
            interp_model = build_model({"name": "interp_baseline", "params": {}})
            interp_row = ("interp_baseline", *_evaluate(interp_model, {}, shared_eval))
            del interp_model  # must del the CALLER's own reference — see _release_torch_memory
            _release_torch_memory()

        # keyed by experiment_name, not cfg.model.name: multiple configs can
        # share a registered model name (e.g. fm_ot's OT and EDM path_type
        # variants both register as "fm_ot") and must stay distinct rows
        rows.append((cfg.experiment_name, *_evaluate(model, model_params, shared_eval)))
        if shuffle_diagnostic:
            # must run BEFORE `del model` below — same model instance, one
            # extra forward pass with permuted query images
            shuffled = _evaluate_shuffled_images(model, model_params, shared_eval)
            if shuffled is not None:
                shuffle_rows.append((cfg.experiment_name, *shuffled))
        del model  # evaluate-then-free, not accumulate-then-evaluate — see _release_torch_memory
        _release_torch_memory()
        print(f"trained + evaluated: {cfg.experiment_name}")

    rows.append(interp_row)

    header = f"{'model':<18}{'PCC':>10}{'RMSE':>10}{'AUC':>10}{'ST-FID':>10}{'ST-MMD':>10}{'plausible':>12}"
    print("\n" + header)
    print("-" * len(header))
    for name, pcc, rmse, auc, fid, mmd, plaus in rows:
        print(f"{name:<18}{pcc:>10.4f}{rmse:>10.4f}{auc:>10.4f}{fid:>10.4f}{mmd:>10.4f}{plaus:>12.4f}")

    if shuffle_diagnostic and shuffle_rows:
        print(f"\nShuffle-image diagnostic (query histology patches randomly "
              f"permuted at eval time — see _evaluate_shuffled_images docstring):")
        print(header)
        print("-" * len(header))
        for name, pcc, rmse, auc, fid, mmd, plaus in shuffle_rows:
            print(f"{name:<18}{pcc:>10.4f}{rmse:>10.4f}{auc:>10.4f}{fid:>10.4f}{mmd:>10.4f}{plaus:>12.4f}")
        print("Compare each row above against its normal counterpart in the "
              "main table: if PCC/RMSE barely change under shuffling, that "
              "model is not actually using its image conditioning to predict "
              "expression, regardless of which image encoder it uses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=str,
                         help="paths to trainable models' configs (VAE/WAE-GAN/FM-OT/VQ-VAE+AR); "
                              "interp_baseline is added automatically")
    parser.add_argument("--override", nargs="*", default=[],
                         help="dotlist overrides applied to EVERY config, "
                              "e.g. --override training.epochs=2 for a quick smoke test")
    parser.add_argument("--skip-training", action="store_true",
                         help="load each config's already-saved weights from its "
                              "training.checkpoint_dir instead of retraining from scratch — "
                              "fast re-evaluation with a new/expanded metric set against "
                              "completed runs. Requires a prior run of this same config (without "
                              "--skip-training) to have actually saved a checkpoint there.")
    parser.add_argument("--shuffle-diagnostic", action="store_true",
                         help="also evaluate every image-using model with query histology "
                              "patches randomly permuted, to check whether it's actually using "
                              "image conditioning at all (see _evaluate_shuffled_images). Adds "
                              "one extra forward pass per image-using model; no-op for models "
                              "that don't use images.")
    args = parser.parse_args()
    main(args.configs, overrides=args.override, skip_training=args.skip_training,
         shuffle_diagnostic=args.shuffle_diagnostic)
