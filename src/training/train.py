"""
Minimal training/eval loop skeleton showing how the pieces connect:
data -> masking -> model (from registry) -> loss -> metrics.

This is intentionally thin — flesh it out (proper DataLoader/batching,
checkpointing, logging backend) once a real dataset + model are chosen.
Run with: python -m src.training.train --config configs/base_config.yaml
"""
from __future__ import annotations
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data import loaders, masking
from src.models.registry import build_model
from src.evaluation import metrics as ev


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
    device = cfg.training.device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 4. Package tensors -------------------------------------------------------
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32, device=device),
        "expression": torch.tensor(expr[context_mask], dtype=torch.float32, device=device),
    }
    query = {
        "coords": torch.tensor(coords3d[query_mask], dtype=torch.float32, device=device),
    }
    target_expression = expr[query_mask]

    # 5. Forward + (if learned) train loop -------------------------------------
    optimizer = None
    if list(model.parameters()):
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    for epoch in range(cfg.training.epochs if optimizer else 1):
        output = model(context, query)
        if optimizer is not None:
            batch = {"target_expression": torch.tensor(
                target_expression, dtype=torch.float32, device=device)}
            loss = model.loss(batch, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % cfg.training.log_every_n_steps == 0:
                print(f"epoch {epoch}: loss={loss.item():.4f}")

    # 6. Evaluate ---------------------------------------------------------------
    pred = output["expression"].detach().cpu().numpy()
    pcc = ev.pearson_per_gene(pred, target_expression)
    print(f"mean PCC: {np.nanmean(pcc):.4f}")
    print(f"RMSE: {ev.rmse(pred, target_expression):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
