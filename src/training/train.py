"""
Minimal training/eval loop skeleton showing how the pieces connect:
data -> masking -> model (from registry) -> Lightning Trainer -> metrics.

MaskedContextQueryDataset draws a FRESH random masking split per training
step (a different seed per index), so an epoch sees many different
damaged/held-out regions instead of one repeated split — the real point of
masking.random_dropout_patches already being seed-parameterized. Replaces
the earlier _SingleBatchDataset placeholder now that a real pilot dataset
(HEST-1k, docs/dataset_notes.md) can be loaded.

Run with: python -m src.training.train --config configs/base_config.yaml
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

# Must be set before `import torch` / any MPS usage, not just before the
# specific op that needs it — PyTorch reads this once when the MPS
# fallback mechanism initializes, not per-op. Confirmed 2026-07-15: setting
# it later in src/models/stpath_encoder.py (imported lazily, well after
# this process had already touched MPS via the Gigapath precompute step)
# was too late and the crash still happened. STPath's own SpatialTransformer
# uses torch.linalg.eigh (fa.py create_frame), not implemented on MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import loaders, masking
from src.models.registry import build_model
from src.evaluation import metrics as ev


def save_trainable_state_dict(model, checkpoint_dir: str, filename: str = "trainable_weights.pt") -> Path | None:
    """Save only trainable parameters, not frozen backbones (Gigapath/
    STPath). Those are identical to the public pretrained weights and get
    reloaded fresh from HuggingFace/the local weight file every time a
    model is rebuilt (see conditioning.py/stpath_encoder.py __init__) —
    saving them again here would be pure redundancy: a STPath-conditioned
    model's full state_dict() is ~4.7GB (1.2B frozen params) vs a few
    tens of MB for just the trainable ones. No-op (returns None) for
    parameter-free models like interp_baseline.

    Added 2026-07-15 after a real question: training runs weren't saving
    ANYTHING before this, meaning a completed 10000-step run's weights
    were gone the moment run_comparison.py's _free() deleted the model
    object — any later use (e.g. testing on a newly downloaded sample)
    would have required retraining from scratch."""
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    if not trainable_names:
        return None
    state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
    path = Path(checkpoint_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def save_trained_model(model, model_cfg: dict, gene_names: list, checkpoint_dir: str) -> Path | None:
    """save_trainable_state_dict's weights alone aren't enough to actually
    reuse a trained model later — reconstructing it needs (1) the exact
    architecture (model_cfg, so build_model() rebuilds something
    load_state_dict-compatible) and (2) which gene each of the n_genes
    fixed output positions corresponds to.

    (2) matters because these models (WAE-GAN/FM-OT/VQ-VAE+AR) use a
    dense fixed-width decoder — position i always means "the i-th gene in
    whatever adata.var_names order was used at construction time," with
    no notion of gene IDENTITY built into the architecture (unlike
    STPath, which tokenizes genes by a fixed vocabulary and is naturally
    panel-agnostic — see stpath_encoder.py). A DIFFERENT HEST-1k sample
    will not have an identical gene panel after its own independent QC,
    so testing a saved model against new data needs to align genes by
    NAME, not position — impossible without recording gene_names here.
    Discussed 2026-07-15 when planning cross-sample testing; this is the
    save half, load_trained_model below is the load half."""
    weights_path = save_trainable_state_dict(model, checkpoint_dir)
    if weights_path is None:
        return None
    out_dir = weights_path.parent
    with open(out_dir / "model_cfg.json", "w") as f:
        json.dump(model_cfg, f)
    with open(out_dir / "gene_names.json", "w") as f:
        json.dump(list(gene_names), f)
    return weights_path


def load_trained_model(checkpoint_dir: str):
    """Reconstruct a model saved by save_trained_model: rebuild an
    architecturally-identical, freshly-initialized model from the saved
    model_cfg (so frozen backbones like Gigapath/STPath get re-loaded
    from their own real pretrained weights, exactly as they were during
    training — see conditioning.py/stpath_encoder.py __init__, neither of
    which was ever saved here in the first place), then load the saved
    trainable-only weights on top. strict=False since frozen/buffer
    entries in the fresh model's state_dict were never in the saved file
    by design — but every TRAINABLE parameter the fresh architecture
    expects must be present, checked explicitly rather than silently
    trusting load_state_dict's missing-keys list (which conflates
    "expected to be missing" with "genuinely lost").

    Returns (model, gene_names) — gene_names is the exact ordered list
    output position i corresponds to; callers must align any new sample's
    genes to this list BY NAME before calling model.sample()."""
    from src.models.registry import build_model
    in_dir = Path(checkpoint_dir)
    with open(in_dir / "model_cfg.json") as f:
        model_cfg = json.load(f)
    with open(in_dir / "gene_names.json") as f:
        gene_names = json.load(f)
    model = build_model(model_cfg)
    state = torch.load(in_dir / "trainable_weights.pt", map_location="cpu")
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    missing_trainable = trainable_names - set(state.keys())
    assert not missing_trainable, (
        f"saved weights at {in_dir} are missing trainable parameters this "
        f"model architecture expects: {missing_trainable} (model_cfg mismatch?)"
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, gene_names


def make_context_query_split(coords3d: np.ndarray, slice_ids: np.ndarray, masking_cfg, seed: int):
    """One random context/query mask draw. Factored out of
    MaskedContextQueryDataset so anything that needs the raw boolean masks
    directly (e.g. src/evaluation/run_comparison.py, task #15, which needs
    to index the source AnnData the same way) doesn't have to duplicate
    the strategy branching logic."""
    strategy = masking_cfg.strategy
    if strategy == "hold_out_slice":
        rng = np.random.default_rng(seed)
        held_out = rng.choice(np.unique(slice_ids))
        context_mask, query_mask = masking.hold_out_slice(coords3d[:, 2], held_out, slice_ids)
    elif strategy == "random_dropout_patches":
        context_mask, query_mask = masking.random_dropout_patches(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    else:
        raise ValueError(f"Unknown masking strategy {strategy}")
    return context_mask, query_mask


def _images_tensor(images: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Slice + tensor-ify per-spot image data for one masking draw.
    Standalone (not a Dataset method) so src/evaluation/run_comparison.py's
    shared held-out eval draw can build the same tensors without going
    through MaskedContextQueryDataset.

    `images` is either raw uint8 patches [N, H, W, 3] (image_encoder_type
    "cnn" — the CNN is trainable, so caching its output would be wrong)
    or precomputed Gigapath features [N, gigapath_dim] float32
    (image_encoder_type "gigapath"/"stpath" — see
    src/models/conditioning.py precompute_gigapath_features, computed
    once by _load_images below rather than per training step). Dispatches
    on ndim so callers don't need to know which case they're in."""
    selected = images[mask]
    if selected.ndim == 4:  # raw uint8 patches [n, H, W, 3]
        return torch.tensor(selected, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return torch.tensor(selected, dtype=torch.float32)  # already-precomputed features [n, feat_dim]


def _build_masked_item(coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                        masking_cfg, images: np.ndarray | None, seed: int) -> dict:
    """One {context, query, target_expression} training item for a SINGLE
    sample's data. Factored out of MaskedContextQueryDataset.__getitem__
    (2026-07-15) so MultiSampleMaskedContextQueryDataset below can reuse
    the exact same masking-draw logic per-sample, rather than duplicating
    it — the only new thing multi-sample training needs is WHICH sample
    to draw from each item, not a different way of drawing from one."""
    context_mask, query_mask = make_context_query_split(coords3d, slice_ids, masking_cfg, seed)
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(expr[context_mask], dtype=torch.float32),
    }
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    if images is not None:
        context["images"] = _images_tensor(images, context_mask)
        query["images"] = _images_tensor(images, query_mask)
    target_expression = torch.tensor(expr[query_mask], dtype=torch.float32)
    return {"context": context, "query": query, "target_expression": target_expression}


class MaskedContextQueryDataset(Dataset):
    """Each item = one fresh random context/query split over the same
    underlying AnnData. batch_size stays 1 at the DataLoader level since
    each split has a different N_context/N_query — collating variable-sized
    point clouds isn't handled here yet."""

    def __init__(self, coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                 masking_cfg, n_items: int, base_seed: int = 0,
                 images: np.ndarray | None = None):
        self.coords3d = coords3d
        self.expr = expr
        self.slice_ids = slice_ids
        self.masking_cfg = masking_cfg
        self.n_items = n_items
        self.base_seed = base_seed
        # optional per-spot image data (task #17/#18/#20), already aligned
        # to coords3d/expr's row order by the caller — either raw H&E
        # patches [N, H, W, 3] uint8 (image_encoder_type "cnn") or
        # precomputed Gigapath features [N, gigapath_dim] float32
        # ("gigapath"/"stpath" — see _load_images in this file). See
        # _images_tensor above for how the two cases are distinguished.
        self.images = images

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.base_seed + idx
        return _build_masked_item(
            self.coords3d, self.expr, self.slice_ids, self.masking_cfg, self.images, seed
        )


class MultiSampleMaskedContextQueryDataset(Dataset):
    """Multi-sample generalization of MaskedContextQueryDataset (task
    #19 follow-up scaffolding, 2026-07-15 — not yet wired into
    train.py's/run_comparison.py's CLI, which still train on one sample;
    this is the dataset half of extending to real multi-sample training).

    Each item picks ONE sample (uniformly at random, reseeded per item)
    and draws its masking split from JUST that sample via
    _build_masked_item — the same logic MaskedContextQueryDataset uses,
    not a reimplementation. Deliberately does NOT pool samples into one
    shared coordinate space: independent HEST-1k samples (different
    patients/sections) have no real spatial relationship to each other,
    so letting a k-NN/attention context encoder draw "neighbors" across
    samples would be meaningless — see load_multi_sample's docstring in
    src/data/loaders.py for the full reasoning (this mirrors that
    function's design: keep samples separate, never concatenate their
    coordinate spaces).

    samples: list of (coords3d, expr, slice_ids, images) tuples, one per
    already-loaded/QC'd/gene-aligned sample (see
    src/data/loaders.py load_multi_sample for the loading half — it
    returns a list of gene-aligned AnnData; callers derive these tuples
    from that list the same way _load_data already does for one sample).
    images is None throughout in the current scaffolding — H&E/Gigapath/
    STPath support across multiple samples is real follow-up work, not
    built here yet, since it adds real extra complexity (per-sample
    Gigapath caching, image alignment) better scoped once this base case
    (expression-only, multi-sample) is validated."""

    def __init__(self, samples: list[tuple], masking_cfg, n_items: int, base_seed: int = 0):
        assert samples, "samples must be non-empty"
        self.samples = samples
        self.masking_cfg = masking_cfg
        self.n_items = n_items
        self.base_seed = base_seed

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.base_seed + idx
        # separate RNG draw for "which sample" vs. the masking split
        # itself (seed + 1, passed to _build_masked_item) so the two
        # choices aren't spuriously correlated through a shared seed
        sample_idx = int(np.random.default_rng(seed).integers(len(self.samples)))
        coords3d, expr, slice_ids, images = self.samples[sample_idx]
        return _build_masked_item(coords3d, expr, slice_ids, self.masking_cfg, images, seed + 1)


def _collate_identity(batch_list):
    return batch_list[0]


def load_adata(cfg):
    """QC'd AnnData for this config's data section — factored out so
    src/evaluation/run_comparison.py (task #15) can get the AnnData object
    itself (needed for cell-type clustering), not just the derived arrays
    _load_data returns."""
    if cfg.data.get("source") == "hest1k":
        adata = loaders.load_hest_sample(cfg.data.hest_data_dir, cfg.data.sample_id)
    else:
        adata = loaders.load_multi_slice(cfg.data.paths, cfg.data.z_positions)
    return loaders.basic_qc_and_normalize(
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells
    )


def _gigapath_cache_path(cfg) -> Path:
    """Where precomputed Gigapath features for this sample get cached
    across runs (see _load_images) — next to the HEST-1k data itself so
    it's obvious it belongs to that sample, not somewhere in /tmp that
    would silently vanish between sessions."""
    return Path(cfg.data.hest_data_dir) / "gigapath_cache" / f"{cfg.data.sample_id}.npz"


def get_gigapath_features(cfg, patches: np.ndarray, barcodes: np.ndarray) -> np.ndarray:
    """Load cached Gigapath features for these patches (see
    _gigapath_cache_path) if available, else compute + cache them.
    Factored out of _load_images (2026-07-15) so
    src/evaluation/run_comparison.py's shared eval-image construction can
    reuse the exact same cache, instead of only ever having whichever
    image format the FIRST config in a comparison run happened to
    produce — see run_comparison.py's _build_shared_eval for why that
    was a real bug (raw patches fed into a Gigapath/STPath model at eval
    time forced an unbatched full-ViT forward pass over the whole eval
    set at once, a real ~25GB RAM crash)."""
    cache_path = _gigapath_cache_path(cfg)
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["barcodes"], barcodes):
            print(f"get_gigapath_features: loaded cached features for "
                  f"{cached['features'].shape[0]} spots from {cache_path} "
                  f"(delete this file to force a recompute).")
            return cached["features"]
        print(f"get_gigapath_features: cache at {cache_path} covers a different "
              f"barcode set than the current patches file — recomputing.")
    from src.models.conditioning import precompute_gigapath_features, _default_device
    print(f"Precomputing Gigapath features for {patches.shape[0]} spots on "
          f"{_default_device()} (one-time cost, cached to {cache_path} "
          f"so future runs skip this step)...")
    features = precompute_gigapath_features(patches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, barcodes=barcodes)
    return features


def _load_images(cfg, adata):
    """Optional H&E patches (task #17) — only loaded when
    cfg.data.use_images is set, since every existing pilot config stays
    expression-only by default. Returns (adata, images): adata may come
    back as a SUBSET of the input — align_patches_to_adata() drops spots
    with no matching patch (a normal partial gap in HEST-1k's own patch
    extraction, not an error) — so callers must use the returned adata,
    not their original one, for everything downstream.

    For image_encoder_type "gigapath" or context_encoder_type "stpath",
    images comes back as PRECOMPUTED Gigapath features [N, gigapath_dim]
    float32, not raw patches — computed once via precompute_gigapath_features()
    rather than recomputed from raw pixels on every training step (task
    #18/#20's real bug, 2026-07-15: the naive per-step version made a real
    STPath training run essentially hang).

    That in-memory cache only helped WITHIN one run though — every fresh
    `python -m src.training.train` invocation still re-ran the ~1.1B-param
    frozen ViT over every patch from scratch (2026-07-15, user question:
    "why is gigapath recomputing every run? cant it be saved?"). Since
    Gigapath is frozen, its output for a given raw patch is fixed forever,
    so it's cached to disk at _gigapath_cache_path(cfg) (features computed
    for the FULL raw barcode set from the .h5 file, before any adata
    filtering — align_patches_to_adata reorders/subsets an array purely by
    barcode lookup, so it works identically whether that array is raw
    patches or a cached feature matrix, meaning this cache stays valid
    even if QC settings change which spots survive downstream). Delete the
    cache file to force a recompute (e.g. after changing which patches
    file is on disk).

    For "cnn" (or no image use), images stays raw uint8 patches
    [N, 224, 224, 3] — the CNN is trainable so its output can't be cached."""
    if not cfg.data.get("use_images", False):
        return adata, None
    patches, barcodes = loaders.load_hest_patches(cfg.data.hest_data_dir, cfg.data.sample_id)

    model_params = cfg.model.get("params", {})
    uses_frozen_gigapath = (
        model_params.get("image_encoder_type") == "gigapath"
        or model_params.get("context_encoder_type") == "stpath"
    )
    if uses_frozen_gigapath:
        features = get_gigapath_features(cfg, patches, barcodes)
        adata, images = loaders.align_patches_to_adata(adata, features, barcodes)
    else:
        adata, images = loaders.align_patches_to_adata(adata, patches, barcodes)
    return adata, images


def _load_data(cfg) -> tuple:
    adata = load_adata(cfg)
    adata, images = _load_images(cfg, adata)
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()
    return adata, coords3d, expr, slice_ids, images


def load_multi_sample_data(cfg) -> list[tuple]:
    """Multi-sample counterpart to _load_data (scaffolding, 2026-07-15;
    not yet called from main() — this loads the data,
    MultiSampleMaskedContextQueryDataset above is the dataset that
    consumes it, wiring both into an actual training run/config is
    later, real work). Reads cfg.data.sample_ids (a list), not the
    single-sample configs' cfg.data.sample_id.

    Deliberately expression-only for now (images always None per
    sample) — see MultiSampleMaskedContextQueryDataset's docstring for
    why H&E support is scoped separately. Uses
    loaders.load_multi_sample for the actual loading/QC/shared-gene-panel
    alignment (not reimplemented here) — this function's only job is
    converting that list of AnnData into the (coords3d, expr, slice_ids,
    images) tuples MultiSampleMaskedContextQueryDataset expects, the same
    conversion _load_data already does for the single-sample case."""
    adatas = loaders.load_multi_sample(
        cfg.data.hest_data_dir, list(cfg.data.sample_ids),
        min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells,
    )
    samples = []
    for adata in adatas:
        coords3d = loaders.get_coords_3d(adata)
        expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        slice_ids = adata.obs["slice_id"].to_numpy()
        samples.append((coords3d, expr, slice_ids, None))
    return samples


def inject_stpath_gene_names(model_cfg: dict, adata) -> None:
    """If a config sets context_encoder_type: "stpath" (task #18),
    auto-derive stpath_gene_names from the loaded AnnData's var_names
    rather than requiring ~16570 gene symbols hardcoded into a YAML file.
    Mutates model_cfg["params"] in place; no-op for every other config."""
    params = model_cfg.get("params", {})
    if params.get("context_encoder_type") == "stpath" and "stpath_gene_names" not in params:
        params["stpath_gene_names"] = adata.var_names.tolist()


def main(cfg_path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        # dotlist overrides, e.g. ["training.epochs=2"] — smoke-testing a
        # config without editing the file itself (2026-07-15, checking all
        # 18 task #19 configs actually run before committing to full-length
        # training on each)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    torch.manual_seed(cfg.training.seed)

    adata, coords3d, expr, slice_ids, images = _load_data(cfg)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    inject_stpath_gene_names(model_cfg, adata)
    model = build_model(model_cfg)

    # Train (skipped entirely for parameter-free baselines like interp_baseline) --
    if list(model.parameters()):
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed, images=images,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=_collate_identity)
        trainer = pl.Trainer(
            max_epochs=1,  # one pass over `n_items` fresh masking draws == old epoch count
            accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, dataloader)
        # .get() with the same default every real config's own YAML comment
        # documents, not a bare attribute access — configs that don't
        # declare checkpoint_dir (e.g. tests/test_run_comparison.py's
        # synthetic configs) must still work, not crash on a missing key
        checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
        saved_path = save_trained_model(model, model_cfg, adata.var_names.tolist(), checkpoint_dir)
        if saved_path is not None:
            print(f"Saved trained model (weights + config + gene names) to {saved_path.parent}")

    # Evaluate on a held-out masking draw not seen during training -----------
    eval_item = MaskedContextQueryDataset(
        coords3d, expr, slice_ids, cfg.masking,
        n_items=1, base_seed=cfg.training.seed + cfg.training.epochs + 1, images=images,
    )[0]
    context, query = eval_item["context"], eval_item["query"]
    target_expression = eval_item["target_expression"].numpy()

    model.eval()
    with torch.no_grad():
        output = model.sample(context, query)
    pred = output["expression"].detach().cpu().numpy()
    pcc = ev.pearson_per_gene(pred, target_expression)
    print(f"mean PCC: {np.nanmean(pcc):.4f}")
    print(f"RMSE: {ev.rmse(pred, target_expression):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--override", nargs="*", default=[],
                         help="dotlist config overrides, e.g. --override training.epochs=2")
    args = parser.parse_args()
    main(args.config, args.override)
