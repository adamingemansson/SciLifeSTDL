"""Train/evaluate Novae-conditioned models with a fixed context-only mask bank.

The ordinary comparison entry point now evaluates Novae leak-free, but its
training path still precomputes Novae on the intact sample. This wrapper builds
and caches Novae embeddings on context-only AnnData objects for a deterministic
mask bank, cycles those exact masks during training, and delegates evaluation
to ``src.evaluation.run_comparison``.
"""
from __future__ import annotations

import argparse
import os
from typing import Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data import loaders
from src.data.augmentation import augment_coords_xy
from src.evaluation import run_comparison as rc
from src.models.registry import build_model
from src.training.train import (
    EMACallback,
    PeriodicCheckpointCallback,
    PeriodicPrintCallback,
    _build_masked_item,
    _cap_context_mask,
    _load_images,
    get_novae_features_context_only,
    inject_coord_scale,
    inject_decoder_gene_names,
    inject_novae_dim,
    inject_single_sample_n_genes,
    inject_storm_lite_tokenizer_gene_names,
    inject_stpath_gene_names,
    inject_stpath_novae_dim,
    load_pretrained_weights_into,
    load_trained_model,
    make_context_query_split,
    make_dataloader,
    save_trained_model,
)

NovaeMode = Literal["replace", "additive"]
_ORIGINAL_TRAIN_MODEL = rc._train_model


def _load_cfg(path: str, overrides: list[str] | None = None):
    cfg = OmegaConf.load(path)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides)) if overrides else cfg


def _novae_mode(params) -> NovaeMode | None:
    encoder = params.get("context_encoder_type", "builtin")
    if encoder == "builtin" and params.get("gene_encoder_type") == "novae":
        return "replace"
    if encoder == "stpath" and params.get("stpath_new_gene_encoder_type") in ("novae", "both"):
        return "additive"
    if encoder == "storm_lite" and params.get("gene_encoder_type") in ("novae", "both", "tokenizer_novae"):
        return "additive"
    return None


def _bank_spec(cfg) -> tuple[int, int]:
    size = int(cfg.training.get("novae_mask_bank_size", 0))
    seed = int(cfg.training.get("novae_mask_bank_seed", 700_000))
    if size < 1:
        raise ValueError("training.novae_mask_bank_size must be >= 1")
    return size, seed


def _context_mask(coords3d, slice_ids, masking_cfg, seed: int, augment: bool):
    coords = augment_coords_xy(coords3d, seed=seed + 2) if augment else coords3d
    mask, _ = make_context_query_split(coords, slice_ids, masking_cfg, seed)
    return _cap_context_mask(mask, getattr(masking_cfg, "max_context_points", None), seed)


def _load_sample(cfg, adata_cache: dict | None = None):
    adata = rc._cached_load_adata(cfg, adata_cache if adata_cache is not None else {})
    adata, images = _load_images(cfg, adata)
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()
    organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
    tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
    return adata, images, coords3d, expr, slice_ids, organ, tech


def _bank(cfg, adata, coords3d, slice_ids, start=0, count=None, keep=True):
    size, base_seed = _bank_spec(cfg)
    if start < 0 or start > size:
        raise ValueError(f"start {start} outside [0, {size}]")
    stop = size if count is None else min(size, start + max(0, count))
    augment = bool(cfg.training.get("augment_coords", False))
    result = {}
    print(f"Fixed Novae bank [{start}, {stop})/{size}, base_seed={base_seed}")
    for index in range(start, stop):
        seed = base_seed + index
        mask = _context_mask(coords3d, slice_ids, cfg.masking, seed, augment)
        features = get_novae_features_context_only(cfg, adata, mask)
        if keep:
            result[seed] = features
        print(f"bank[{index:03d}] seed={seed} context={int(mask.sum())} dim={features.shape[1]}")
    return result


class FixedNovaeMaskBankDataset(Dataset):
    def __init__(self, coords3d, expr, slice_ids, masking_cfg, n_items, bank,
                 mode: NovaeMode, images=None, organ=None, tech=None, augment=False):
        if not bank:
            raise ValueError("fixed Novae bank is empty")
        self.coords3d, self.expr, self.slice_ids = coords3d, expr, slice_ids
        self.masking_cfg, self.n_items = masking_cfg, int(n_items)
        self.bank, self.seeds, self.mode = bank, tuple(sorted(bank)), mode
        self.images, self.organ, self.tech, self.augment = images, organ, tech, augment

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.seeds[idx % len(self.seeds)]
        features = self.bank[seed]
        return _build_masked_item(
            self.coords3d, self.expr, self.slice_ids, self.masking_cfg,
            self.images, seed,
            context_gene_features=features if self.mode == "replace" else None,
            context_novae_features=features if self.mode == "additive" else None,
            organ=self.organ, tech=self.tech, augment=self.augment,
        )


def _inject_novae(model_cfg: dict, mode: NovaeMode, dim: int):
    params = model_cfg.get("params", {})
    if mode == "additive" and params.get("context_encoder_type") == "stpath":
        inject_stpath_novae_dim(model_cfg, dim)
    else:
        inject_novae_dim(model_cfg, dim)


def _train_model(cfg_path, overrides=None, adata_cache=None, skip_training=False):
    cfg = _load_cfg(cfg_path, overrides)
    mode = _novae_mode(cfg.model.get("params", {}))
    if mode is None:
        return _ORIGINAL_TRAIN_MODEL(cfg_path, overrides, adata_cache, skip_training)

    torch.manual_seed(cfg.training.seed)
    adata, images, coords3d, expr, slice_ids, organ, tech = _load_sample(cfg, adata_cache)
    checkpoint_dir = cfg.training.get("checkpoint_dir", f"results/checkpoints/{cfg.experiment_name}")
    if skip_training:
        model, _ = load_trained_model(checkpoint_dir)
        model.eval()
        return model, cfg, adata, coords3d, expr, slice_ids, images

    bank = _bank(cfg, adata, coords3d, slice_ids)
    novae_dim = next(iter(bank.values())).shape[1]
    coord_scale = float(coords3d[:, :2].std())

    def prepare(resolve: bool):
        model_cfg = OmegaConf.to_container(cfg.model, resolve=resolve)
        inject_single_sample_n_genes(model_cfg, adata)
        inject_stpath_gene_names(model_cfg, adata)
        inject_decoder_gene_names(model_cfg, adata)
        inject_storm_lite_tokenizer_gene_names(model_cfg, adata)
        inject_coord_scale(model_cfg, coord_scale)
        _inject_novae(model_cfg, mode, novae_dim)
        return model_cfg

    model_cfg = prepare(True)
    unresolved_cfg = prepare(False)
    model = build_model(model_cfg)
    init_dir = cfg.training.get("init_checkpoint_dir")
    if init_dir:
        load_pretrained_weights_into(model, init_dir)

    dataset = FixedNovaeMaskBankDataset(
        coords3d, expr, slice_ids, cfg.masking, cfg.training.epochs,
        bank, mode, images=images, organ=organ, tech=tech,
        augment=bool(cfg.training.get("augment_coords", False)),
    )
    callbacks = []
    checkpoint_every = cfg.training.get("checkpoint_every_n_steps")
    if checkpoint_every:
        callbacks.append(PeriodicCheckpointCallback(
            unresolved_cfg, adata.var_names.tolist(), checkpoint_dir, checkpoint_every))
    print_every = cfg.training.get("log_print_every_n_steps")
    if print_every:
        callbacks.append(PeriodicPrintCallback(print_every))
    ema_decay = cfg.training.get("ema_decay")
    ema = EMACallback(ema_decay) if ema_decay else None
    if ema:
        callbacks.append(ema)

    print(f"Training with leak-free fixed Novae bank: {len(bank)} masks, mode={mode}")
    trainer = pl.Trainer(
        max_epochs=1, accelerator="auto", logger=False,
        enable_checkpointing=False, callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=cfg.training.log_every_n_steps,
    )
    trainer.fit(model, make_dataloader(dataset, cfg))
    if ema:
        ema.apply_to_model(model)
    saved = save_trained_model(model, unresolved_cfg, adata.var_names.tolist(), checkpoint_dir)
    if saved is not None:
        print(f"Saved leak-free fixed-Novae model to {saved.parent}")
    model.eval()
    return model, cfg, adata, coords3d, expr, slice_ids, images


def _precompute(path, overrides, start, count):
    cfg = _load_cfg(path, overrides)
    if _novae_mode(cfg.model.get("params", {})) is None:
        raise ValueError(f"{path} does not use Novae")
    adata, _, coords3d, _, slice_ids, _, _ = _load_sample(cfg)
    _bank(cfg, adata, coords3d, slice_ids, start=start, count=count, keep=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--shuffle-diagnostic", action="store_true")
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--precompute-start", type=int, default=0)
    parser.add_argument("--precompute-count", type=int)
    args = parser.parse_args()
    if args.precompute_only:
        if len(args.configs) != 1:
            parser.error("--precompute-only accepts exactly one config")
        _precompute(args.configs[0], args.override, args.precompute_start, args.precompute_count)
        return
    rc._train_model = _train_model
    rc.main(args.configs, overrides=args.override, skip_training=args.skip_training,
            shuffle_diagnostic=args.shuffle_diagnostic)


if __name__ == "__main__":
    main()
