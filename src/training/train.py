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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import loaders, masking
from src.models.registry import build_model
from src.evaluation import metrics as ev


class MaskedContextQueryDataset(Dataset):
    """Each item = one fresh random context/query split over the same
    underlying AnnData. batch_size stays 1 at the DataLoader level since
    each split has a different N_context/N_query — collating variable-sized
    point clouds isn't handled here yet."""

    def __init__(self, coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
                 masking_cfg, n_items: int, base_seed: int = 0):
        self.coords3d = coords3d
        self.expr = expr
        self.slice_ids = slice_ids
        self.masking_cfg = masking_cfg
        self.n_items = n_items
        self.base_seed = base_seed

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        seed = self.base_seed + idx
        strategy = self.masking_cfg.strategy
        if strategy == "hold_out_slice":
            rng = np.random.default_rng(seed)
            held_out = rng.choice(np.unique(self.slice_ids))
            context_mask, query_mask = masking.hold_out_slice(
                self.coords3d[:, 2], held_out, self.slice_ids
            )
        elif strategy == "random_dropout_patches":
            context_mask, query_mask = masking.random_dropout_patches(
                self.coords3d[:, :2], self.slice_ids, seed=seed, **self.masking_cfg.params
            )
        else:
            raise ValueError(f"Unknown masking strategy {strategy}")

        context = {
            "coords": torch.tensor(self.coords3d[context_mask], dtype=torch.float32),
            "expression": torch.tensor(self.expr[context_mask], dtype=torch.float32),
        }
        query = {"coords": torch.tensor(self.coords3d[query_mask], dtype=torch.float32)}
        target_expression = torch.tensor(self.expr[query_mask], dtype=torch.float32)
        return {"context": context, "query": query, "target_expression": target_expression}


def _collate_identity(batch_list):
    return batch_list[0]


def _load_data(cfg) -> tuple:
    if cfg.data.get("source") == "hest1k":
        adata = loaders.load_hest_sample(cfg.data.hest_data_dir, cfg.data.sample_id)
    else:
        adata = loaders.load_multi_slice(cfg.data.paths, cfg.data.z_positions)
    adata = loaders.basic_qc_and_normalize(
        adata, min_genes=cfg.data.min_genes, min_cells=cfg.data.min_cells
    )
    coords3d = loaders.get_coords_3d(adata)
    expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    slice_ids = adata.obs["slice_id"].to_numpy()
    return coords3d, expr, slice_ids


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    torch.manual_seed(cfg.training.seed)

    coords3d, expr, slice_ids = _load_data(cfg)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(model_cfg)

    # Train (skipped entirely for parameter-free baselines like interp_baseline) --
    if list(model.parameters()):
        dataset = MaskedContextQueryDataset(
            coords3d, expr, slice_ids, cfg.masking,
            n_items=cfg.training.epochs, base_seed=cfg.training.seed,
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
        n_items=1, base_seed=cfg.training.seed + cfg.training.epochs + 1,
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
    args = parser.parse_args()
    main(args.config)
