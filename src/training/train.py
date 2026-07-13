"""
Minimal training/eval loop skeleton showing how the pieces connect:
data -> masking -> model (from registry) -> Lightning Trainer -> metrics.

Wraps the single context/query split as a repeating dataset so
pytorch_lightning.Trainer can drive it uniformly across model families —
including WAE-GAN's manual multi-optimizer step, which needs an attached
Trainer to work at all (self.optimizers() is only valid inside one).

This is intentionally thin — replace _SingleBatchDataset with a real
per-cell/mini-batch Dataset once a real pilot dataset is chosen (Phase 4,
see docs/project_outline.md). The model interface does not need to change
when that happens.
Run with: python -m src.training.train --config configs/base_config.yaml
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import loaders, masking
from src.models.registry import build_model
from src.evaluation import metrics as ev


class _SingleBatchDataset(Dataset):
    """Yields the same context/query batch `n_steps` times, so Trainer's
    dataloader loop maps onto the old script's manual epoch loop. Placeholder
    until real mini-batching exists (see module docstring)."""

    def __init__(self, batch: dict, n_steps: int):
        self.batch = batch
        self.n_steps = n_steps

    def __len__(self):
        return self.n_steps

    def __getitem__(self, idx):
        return self.batch


def _collate_identity(batch_list):
    return batch_list[0]


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    torch.manual_seed(cfg.training.seed)

    # 1. Load data ---------------------------------------------------------
    adata = loaders.load_multi_slice(cfg.data.paths, cfg.data.z_positions)
    adata = loaders.basic_qc_and_normalize(
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells
    )
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()

    # 2. Build train/query split via the masking simulator ------------------
    if cfg.masking.strategy == "hold_out_slice":
        held_out = np.unique(slice_ids)[-1]
        context_mask, query_mask = masking.hold_out_slice(
            coords3d[:, 2], held_out, slice_ids
        )
    elif cfg.masking.strategy == "random_dropout_patches":
        context_mask, query_mask = masking.random_dropout_patches(
            coords3d[:, :2], slice_ids, seed=cfg.training.seed,
            **cfg.masking.params,
        )
    else:
        raise ValueError(f"Unknown masking strategy {cfg.masking.strategy}")

    # 3. Build model from registry -------------------------------------------
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(model_cfg)

    # 4. Package tensors -------------------------------------------------------
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(expr[context_mask], dtype=torch.float32),
    }
    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    target_expression = expr[query_mask]
    batch = {
        "context": context,
        "query": query,
        "target_expression": torch.tensor(target_expression, dtype=torch.float32),
    }

    # 5. Train (skipped entirely for parameter-free baselines like interp_baseline) --
    if list(model.parameters()):
        dataset = _SingleBatchDataset(batch, n_steps=cfg.training.epochs)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=_collate_identity)
        trainer = pl.Trainer(
            max_epochs=1,  # one pass over `n_steps` repeats of the batch == old epoch count
            accelerator="auto",
            log_every_n_steps=cfg.training.log_every_n_steps,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.fit(model, dataloader)

    # 6. Evaluate ---------------------------------------------------------------
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
    args = parser.parse_args()
    main(args.config)
