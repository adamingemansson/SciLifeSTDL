"""Pretrain the residual-expression autoencoder used by residual FM.

The target is deliberately **not** absolute expression.  For every clean
training/validation mask we first construct the same graph-harmonic anchor used
by :class:`ResidualFlowMatchingOT`, then train on
``query_expression - harmonic_anchor``.  Test slides and test masks are never
loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.data.mask_bank import (
    cap_context_mask,
    ensure_mask_bank,
    make_split,
    record_masks,
    split_records,
)
from src.models.spatial_baselines import harmonic_interpolate


class ExpressionAutoencoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _dense(adata) -> np.ndarray:
    return np.asarray(
        adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),
        dtype=np.float32,
    )


def _residual_for_mask(
    coords3d: np.ndarray,
    expression: np.ndarray,
    context_mask: np.ndarray,
    query_mask: np.ndarray,
    *,
    harmonic_k: int,
    harmonic_ridge: float,
) -> np.ndarray:
    """Return query residuals using observed context expression only."""
    context_mask = np.asarray(context_mask, dtype=bool)
    query_mask = np.asarray(query_mask, dtype=bool)
    if not context_mask.any() or not query_mask.any():
        raise ValueError("residual pretraining mask has an empty context or query")
    coords = torch.from_numpy(np.asarray(coords3d, dtype=np.float32))
    expr = torch.from_numpy(np.asarray(expression, dtype=np.float32))
    with torch.inference_mode():
        anchor = harmonic_interpolate(
            coords[context_mask], expr[context_mask], coords[query_mask],
            k=int(harmonic_k), ridge=float(harmonic_ridge),
        )
        residual = expr[query_mask] - anchor
    return residual.cpu().numpy().astype(np.float32, copy=False)


def _select_rows(rows: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or len(rows) <= int(limit):
        return rows
    idx = np.random.default_rng(int(seed)).choice(len(rows), size=int(limit), replace=False)
    return rows[np.sort(idx)]


def _collect_seeded_residuals(
    adata,
    masking_cfg,
    seeds: Iterable[int],
    *,
    harmonic_k: int,
    harmonic_ridge: float,
    max_rows: int,
    max_rows_per_mask: int | None,
) -> np.ndarray:
    from src.data.loaders import get_coords_3d

    coords3d = get_coords_3d(adata)
    slice_ids = adata.obs["slice_id"].to_numpy()
    expression = _dense(adata)
    chunks: list[np.ndarray] = []
    n_rows = 0
    max_context = masking_cfg.get("max_context_points", None)
    for seed in map(int, seeds):
        context, query = make_split(coords3d, slice_ids, masking_cfg, seed)
        context = cap_context_mask(context, max_context, seed)
        residual = _residual_for_mask(
            coords3d, expression, context, query,
            harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
        )
        residual = _select_rows(residual, max_rows_per_mask, seed + 17)
        remaining = int(max_rows) - n_rows
        if remaining <= 0:
            break
        chunks.append(residual[:remaining])
        n_rows += min(len(residual), remaining)
    if not chunks:
        raise RuntimeError("no residual rows were generated from training masks")
    return np.concatenate(chunks, axis=0)


def _collect_bank_residuals(
    adata,
    masking_cfg,
    bank: dict,
    split: str,
    *,
    harmonic_k: int,
    harmonic_ridge: float,
    max_rows: int,
    max_rows_per_mask: int | None,
) -> np.ndarray:
    from src.data.loaders import get_coords_3d

    coords3d = get_coords_3d(adata)
    expression = _dense(adata)
    chunks: list[np.ndarray] = []
    n_rows = 0
    for record in split_records(bank, split):
        context, query = record_masks(record, adata.obs_names)
        residual = _residual_for_mask(
            coords3d, expression, context, query,
            harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
        )
        residual = _select_rows(
            residual, max_rows_per_mask, int(record["seed"]) + 31
        )
        remaining = int(max_rows) - n_rows
        if remaining <= 0:
            break
        chunks.append(residual[:remaining])
        n_rows += min(len(residual), remaining)
    if not chunks:
        raise RuntimeError(f"no residual rows were generated from {split!r} masks")
    return np.concatenate(chunks, axis=0)


def _mask_bank_for_adata(cfg, adata, sample_id: str | None = None) -> dict:
    from src.data.loaders import get_coords_3d

    evaluation = cfg.get("evaluation", {})
    coords3d = get_coords_3d(adata)
    slice_ids = adata.obs["slice_id"].to_numpy()
    counts = {
        "validation": int(evaluation.get("n_validation_masks", 4)),
        "test": int(evaluation.get("n_test_masks", 8)),
    }
    seeds = {
        "validation": int(evaluation.get("validation_seed", 700_000)),
        "test": int(evaluation.get("test_seed", 900_000)),
    }
    if sample_id is None:
        path = Path(evaluation.get(
            "mask_bank_path", f"results/mask_banks/{cfg.data.sample_id}.json"
        ))
    else:
        path = Path(evaluation.get("mask_bank_dir", "results/mask_banks")) / f"{sample_id}.json"
    return ensure_mask_bank(
        path, coords3d, slice_ids, adata.obs_names, cfg.masking, counts, seeds
    )


def _load_single_residual_arrays(cfg):
    from src.training.train import load_adata, _load_images

    adata = load_adata(cfg)
    # Match downstream spot inclusion exactly when image coverage is required,
    # even though the autoencoder itself does not consume images.
    adata, _ = _load_images(cfg, adata)
    training = cfg.training
    params = cfg.model
    harmonic_k = int(params.get("harmonic_k", 8))
    harmonic_ridge = float(params.get("harmonic_ridge", 1e-4))
    n_masks = int(training.get("residual_training_masks", 128))
    base_seed = int(training.get("residual_training_mask_seed", 100_000))
    train = _collect_seeded_residuals(
        adata, cfg.masking, range(base_seed, base_seed + n_masks),
        harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
        max_rows=int(training.get("max_training_residual_rows", 8192)),
        max_rows_per_mask=training.get("max_rows_per_mask", 128),
    )
    bank = _mask_bank_for_adata(cfg, adata)
    validation = _collect_bank_residuals(
        adata, cfg.masking, bank, "validation",
        harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
        max_rows=int(training.get("max_validation_residual_rows", 2048)),
        max_rows_per_mask=training.get("max_rows_per_mask", 128),
    )
    return train, validation, adata.var_names.tolist(), {
        "fit_sample_ids": [str(cfg.data.sample_id)],
        "validation_sample_ids": [str(cfg.data.sample_id)],
        "expression_preprocessing": adata.uns.get("expression_preprocessing", {}),
        "harmonic_k": harmonic_k,
        "harmonic_ridge": harmonic_ridge,
    }


def _load_multi_residual_arrays(cfg):
    from src.training.train import load_multi_sample_data

    train_ids = [str(x) for x in cfg.data.get("train_sample_ids", [])]
    validation_ids = [str(x) for x in cfg.data.get("validation_sample_ids", [])]
    if not train_ids:
        raise ValueError("multi-sample residual pretraining requires data.train_sample_ids")
    if not validation_ids:
        raise ValueError(
            "multi-sample residual pretraining requires held-out data.validation_sample_ids; "
            "do not split rows from the training slides"
        )

    # The panel is selected from training slides only. Validation slides are
    # aligned to that fixed panel and test slides are never loaded.
    train_samples, train_adatas = load_multi_sample_data(cfg, sample_ids=train_ids)
    gene_names = train_adatas[0].var_names.tolist()
    _val_samples, val_adatas = load_multi_sample_data(
        cfg, sample_ids=validation_ids, reference_gene_names=gene_names
    )

    training = cfg.training
    params = cfg.model
    harmonic_k = int(params.get("harmonic_k", 8))
    harmonic_ridge = float(params.get("harmonic_ridge", 1e-4))
    n_masks_per_sample = int(training.get("residual_training_masks_per_sample", 32))
    base_seed = int(training.get("residual_training_mask_seed", 100_000))
    total_train_rows = int(training.get("max_training_residual_rows", 16384))
    total_val_rows = int(training.get("max_validation_residual_rows", 4096))
    rows_per_train_sample = max(1, total_train_rows // len(train_adatas))
    rows_per_val_sample = max(1, total_val_rows // len(val_adatas))
    per_mask = training.get("max_rows_per_mask", 128)

    train_chunks = []
    for sample_index, adata in enumerate(train_adatas):
        sample_seed = base_seed + sample_index * 100_000
        train_chunks.append(_collect_seeded_residuals(
            adata, cfg.masking,
            range(sample_seed, sample_seed + n_masks_per_sample),
            harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
            max_rows=rows_per_train_sample, max_rows_per_mask=per_mask,
        ))
    val_chunks = []
    for sample_id, adata in zip(validation_ids, val_adatas):
        bank = _mask_bank_for_adata(cfg, adata, sample_id=sample_id)
        val_chunks.append(_collect_bank_residuals(
            adata, cfg.masking, bank, "validation",
            harmonic_k=harmonic_k, harmonic_ridge=harmonic_ridge,
            max_rows=rows_per_val_sample, max_rows_per_mask=per_mask,
        ))

    preprocessing = train_adatas[0].uns.get("expression_preprocessing", {})
    return (
        np.concatenate(train_chunks, axis=0)[:total_train_rows],
        np.concatenate(val_chunks, axis=0)[:total_val_rows],
        gene_names,
        {
            "fit_sample_ids": train_ids,
            "validation_sample_ids": validation_ids,
            "expression_preprocessing": preprocessing,
            "harmonic_k": harmonic_k,
            "harmonic_ridge": harmonic_ridge,
        },
    )


def _load_arrays(cfg):
    if cfg.get("masking") is None:
        raise ValueError("residual autoencoder pretraining requires a masking section")
    if cfg.data.get("train_sample_ids"):
        return _load_multi_residual_arrays(cfg)
    return _load_single_residual_arrays(cfg)


def _evaluate(model, x: torch.Tensor, batch_size: int, device: torch.device):
    model.eval()
    sq_error, count = 0.0, 0
    preds, targets = [], []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(x), batch_size=batch_size):
            batch = batch.to(device)
            pred = model(batch)
            sq_error += float(((pred - batch) ** 2).sum().cpu())
            count += batch.numel()
            preds.append(pred.cpu())
            targets.append(batch.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    pred_c = pred - pred.mean(0, keepdim=True)
    target_c = target - target.mean(0, keepdim=True)
    denom = torch.sqrt((pred_c**2).sum(0) * (target_c**2).sum(0)).clamp_min(1e-8)
    pcc = ((pred_c * target_c).sum(0) / denom).nanmean().item()
    return float(np.sqrt(sq_error / max(1, count))), float(pcc)


def main(config_path: str):
    cfg = OmegaConf.load(config_path)
    torch.manual_seed(int(cfg.training.get("seed", 0)))
    train_np, val_np, gene_names, target_metadata = _load_arrays(cfg)
    print(
        f"residual AE data: train={train_np.shape}, validation={val_np.shape}, "
        f"genes={len(gene_names)}"
    )
    n_genes = train_np.shape[1]
    hidden_dim = int(cfg.model.get("hidden_dim", 256))
    latent_dim = int(cfg.model.get("latent_dim", 32))
    batch_size = int(cfg.training.get("batch_size", 256))
    max_steps = int(cfg.training.get("steps", 20000))
    validate_every = int(cfg.training.get("validate_every_n_steps", 500))
    patience = int(cfg.training.get("patience_checks", 10))
    lr = float(cfg.training.get("lr", 1e-3))
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = ExpressionAutoencoder(n_genes, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_tensor = torch.from_numpy(train_np)
    val_tensor = torch.from_numpy(val_np)
    generator = torch.Generator().manual_seed(int(cfg.training.get("seed", 0)))
    loader = DataLoader(
        TensorDataset(train_tensor), batch_size=batch_size, shuffle=True,
        drop_last=False, generator=generator,
    )
    iterator = iter(loader)
    best_rmse = float("inf")
    best_pcc = float("nan")
    best_state = None
    bad_checks = 0
    history = []

    model.train()
    for step in range(1, max_steps + 1):
        try:
            (batch,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (batch,) = next(iterator)
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % validate_every == 0 or step == max_steps:
            val_rmse, val_pcc = _evaluate(model, val_tensor, batch_size, device)
            improved = val_rmse < best_rmse - float(cfg.training.get("min_delta", 1e-5))
            history.append({
                "step": step, "train_mse": float(loss.detach().cpu()),
                "validation_rmse": val_rmse, "validation_pcc": val_pcc,
                "improved": improved,
            })
            print(
                f"step={step} train_mse={loss.item():.6f} "
                f"val_rmse={val_rmse:.6f} val_pcc={val_pcc:.4f}"
            )
            if improved:
                best_rmse, best_pcc = val_rmse, val_pcc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
            model.train()
            if bad_checks >= patience:
                print(f"early stopping after {bad_checks} non-improving checks")
                break

    if best_state is None:
        raise RuntimeError("autoencoder training produced no validation checkpoint")
    model.load_state_dict(best_state)
    out_dir = Path(cfg.training.get(
        "checkpoint_dir", "results/checkpoints/audit_suite/expression_autoencoder"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    required_max_rmse = float(cfg.training.get("required_max_validation_rmse", float("inf")))
    required_min_pcc = float(cfg.training.get("required_min_validation_pcc", -float("inf")))
    checkpoint = {
        "format_version": 2,
        "target_type": "harmonic_residual",
        "n_genes": n_genes,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "gene_names": gene_names,
        "harmonic_k": int(target_metadata["harmonic_k"]),
        "harmonic_ridge": float(target_metadata["harmonic_ridge"]),
        "expression_preprocessing": target_metadata["expression_preprocessing"],
        "fit_sample_ids": target_metadata["fit_sample_ids"],
        "validation_sample_ids": target_metadata["validation_sample_ids"],
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "validation_rmse": best_rmse,
        "validation_pcc": best_pcc,
        "n_training_residual_rows": int(len(train_np)),
        "n_validation_residual_rows": int(len(val_np)),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml", resolve=True)
    quality = {
        "target_type": "harmonic_residual",
        "validation_rmse": best_rmse,
        "validation_pcc": best_pcc,
        "required_max_validation_rmse": required_max_rmse,
        "required_min_validation_pcc": required_min_pcc,
        "passed": bool(best_rmse <= required_max_rmse and best_pcc >= required_min_pcc),
    }
    (out_dir / "quality_gate.json").write_text(json.dumps(quality, indent=2))
    if not quality["passed"]:
        raise RuntimeError(
            "residual expression autoencoder failed its quality gate: "
            f"RMSE={best_rmse:.6f} (required <= {required_max_rmse}), "
            f"PCC={best_pcc:.4f} (required >= {required_min_pcc}). "
            "Residual-flow jobs are intentionally blocked."
        )
    tmp = out_dir / "best_autoencoder.pt.tmp"
    torch.save(checkpoint, tmp)
    tmp.replace(out_dir / "best_autoencoder.pt")
    print(
        f"saved {out_dir / 'best_autoencoder.pt'} "
        f"(residual val_rmse={best_rmse:.6f}, val_pcc={best_pcc:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
