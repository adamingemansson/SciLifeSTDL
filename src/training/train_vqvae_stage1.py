"""
Real-data training script for VQ-VAE stage 1 (src/models/vqvae.py) —
plain reconstruction, no masking/context-query split (deliberately
unconditioned, see vqvae.py docstring). Separate from
src/training/train.py since VQVAEStage1 doesn't implement
BaseGenerativeModel's context/query interface — it isn't a
location->expression generator yet, just the reconstruction stage
validated in isolation before stage 2 (autoregressive transformer, task
#12) wires it into the shared pipeline.

Run with:
    python -m src.training.train_vqvae_stage1 --config configs/exp_hest1k_vqvae_stage1.yaml
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import loaders
from src.models.vqvae import VQVAEStage1


class ExpressionDataset(Dataset):
    """Plain per-spot expression vectors — no coords, no masking."""

    def __init__(self, expr: np.ndarray):
        self.expr = expr

    def __len__(self):
        return self.expr.shape[0]

    def __getitem__(self, idx):
        return {"expression": torch.tensor(self.expr[idx], dtype=torch.float32)}


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    torch.manual_seed(cfg.training.seed)

    adata = loaders.load_hest_sample(cfg.data.hest_data_dir, cfg.data.sample_id)
    adata = loaders.basic_qc_and_normalize(
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells
    )
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()

    # held-out spots for a real reconstruction check, never used in training
    n = expr.shape[0]
    rng = np.random.default_rng(cfg.training.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = VQVAEStage1(**model_cfg)

    train_dataset = ExpressionDataset(expr[train_idx])
    dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        # devices=1 (2026-07-23, real incident -- see train.py's identical
        # fix): a bare, unscoped invocation (no CUDA_VISIBLE_DEVICES) would
        # otherwise let accelerator="auto" launch DDP across every visible
        # GPU rather than the one this job is meant to use.
        devices=1,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dataloader)

    model.eval()
    with torch.no_grad():
        val_expr = torch.tensor(expr[val_idx], dtype=torch.float32)
        x_hat, idx, _ = model(val_expr)
    recon_rmse = torch.sqrt(torch.mean((x_hat - val_expr) ** 2)).item()
    n_used = torch.unique(idx).numel()
    print(f"held-out recon RMSE: {recon_rmse:.4f}")
    print(f"held-out codebook usage: {n_used / model.vq.codebook_size:.4f} "
          f"({n_used}/{model.vq.codebook_size} codes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
