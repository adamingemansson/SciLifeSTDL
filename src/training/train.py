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
        context_mask, query_mask = make_context_query_split(
            self.coords3d, self.slice_ids, self.masking_cfg, seed
        )
        context = {
            "coords": torch.tensor(self.coords3d[context_mask], dtype=torch.float32),
            "expression": torch.tensor(self.expr[context_mask], dtype=torch.float32),
        }
        query = {"coords": torch.tensor(self.coords3d[query_mask], dtype=torch.float32)}
        if self.images is not None:
            context["images"] = _images_tensor(self.images, context_mask)
            query["images"] = _images_tensor(self.images, query_mask)
        target_expression = torch.tensor(self.expr[query_mask], dtype=torch.float32)
        return {"context": context, "query": query, "target_expression": target_expression}


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
