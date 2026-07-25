"""Architecture 3, Stage A — pretrain the denoising transcriptome
autoencoder on every observed spot from every training slide.

No masking, no images, no context/query split — this is plain
self-supervised representation learning on expression alone, independent
of the spatial task. Cheap relative to Stage B (train_arch3_stage_b.py):
run this first.

Usage:
    python3 -m gen2_architectures.training.train_arch3_stage_a \\
        --config gen2_architectures/configs/arch3_stage_a_pretrain.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder, corrupt_expression
from gen2_architectures.models.components import StagedGeneLoss
from gen2_architectures.data import loaders
from gen2_architectures.training import checkpoint, data_prep


def main(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    data_prep.apply_sample_selection(cfg)
    device = torch.device(cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_ids = list(cfg.data.train_sample_ids)
    if not train_ids:
        raise ValueError("data.train_sample_ids must be non-empty")
    print(f"loading {len(train_ids)} training sample(s) for Stage A pretraining...")
    adatas = loaders.load_multi_sample(
        cfg.data.hest_data_dir, train_ids,
        min_genes=cfg.data.get("min_genes", 200), min_cells=cfg.data.get("min_cells", 3),
        organs=[cfg.data.get("organ_by_sample", {}).get(str(sid)) for sid in train_ids] or None,
        techs=[cfg.data.get("tech_by_sample", {}).get(str(sid)) for sid in train_ids] or None,
        expression_transform=cfg.data.get("expression_transform", "normalize_log1p"),
        expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
    )
    gene_names = adatas[0].var_names.tolist()
    n_genes = len(gene_names)
    # Pool every observed spot from every training slide into one big
    # expression matrix -- Stage A has no notion of "which sample/slide"
    # a row came from, it's pure per-spot representation learning.
    pooled = np.concatenate(
        [a.X if isinstance(a.X, np.ndarray) else a.X.toarray() for a in adatas], axis=0
    ).astype(np.float32)
    print(f"pooled {pooled.shape[0]} spots x {n_genes} genes across {len(adatas)} training slide(s)")

    params = dict(cfg.model.get("params", {}))
    model = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, **params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Stage A autoencoder built: {n_params:,} parameters")

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    start_step = 0
    if (checkpoint_dir / "model_config.json").exists():
        checkpoint.load_trainable_state(model, checkpoint_dir)
        start_step = checkpoint.load_training_state(checkpoint_dir).get("step", 0)
        print(f"resumed from checkpoint at step {start_step}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.get("lr", 1e-4)))
    loss_fn = StagedGeneLoss(**dict(cfg.training.get("loss", {})))

    total_steps = int(cfg.training.total_steps)
    batch_size = int(cfg.training.get("batch_size", 256))
    grad_clip = float(cfg.training.get("gradient_clip_val", 1.0))
    log_every = int(cfg.training.get("log_every_n_steps", 100))
    checkpoint_every = int(cfg.training.get("checkpoint_every_n_steps", 5000))
    mask_fraction = float(cfg.training.get("mask_fraction", 0.2))
    gaussian_std = float(cfg.training.get("gaussian_std", 0.0))

    pooled_t = torch.from_numpy(pooled)
    rng = random.Random(int(cfg.training.get("seed", 0)))
    n_spots_total = pooled_t.shape[0]
    model.train()
    for step in range(start_step, total_steps):
        idx = torch.tensor(rng.sample(range(n_spots_total), min(batch_size, n_spots_total)))
        clean = pooled_t[idx].to(device)
        corrupted = corrupt_expression(clean, seed=step, mask_fraction=mask_fraction, gaussian_std=gaussian_std)

        recon = model(corrupted)
        progress = step / max(1, total_steps)
        result = loss_fn(recon, clean, progress)  # reconstruct the CLEAN target from corrupted input
        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0:
            print(
                f"step {step}/{total_steps} loss={result['loss'].item():.4f} mse={result['mse'].item():.4f} "
                f"pearson_penalty={result['pearson_penalty'].item():.4f} pearson_weight={result['pearson_weight']:.2f}"
            )
        if step > 0 and step % checkpoint_every == 0:
            checkpoint.save_checkpoint(
                model, {"n_genes": n_genes, "params": params}, gene_names, checkpoint_dir, step,
            )

    checkpoint.save_checkpoint(model, {"n_genes": n_genes, "params": params}, gene_names, checkpoint_dir, total_steps)
    print(f"Stage A pretraining complete. Checkpoint at {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
