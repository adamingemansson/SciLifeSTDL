"""Training entrypoint for Architectures 1, 2, and 4 — all three share the
same per-step shape (draw one masked item from a random training sample,
forward through a single model, staged MSE->Pearson loss, backprop), and
differ only in model construction and whether a scFoundation feature
provider is needed. Architecture 3 (Stage A pretraining, Stage B latent
transformer) has a different enough loop shape to warrant its own scripts
(train_arch3_stage_a.py, train_arch3_stage_b.py).

Usage:
    python3 -m gen2_architectures.training.train_local_neighborhood \\
        --config gen2_architectures/configs/arch1_gpt_baseline.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen2_architectures.data import loaders
from gen2_architectures.data.masked_item import build_masked_item
from gen2_architectures.models.arch1_gpt_baseline import Architecture1
from gen2_architectures.models.arch2_scfoundation import Architecture2
from gen2_architectures.models.arch4_stpath_hybrid import Architecture4
from gen2_architectures.models.components import StagedGeneLoss
from gen2_architectures.training import checkpoint, data_prep, diagnostics, evaluate
from gen2_architectures.training.validation import move_to_device


def _model_config_dict(cfg, scfoundation_dim: int | None) -> dict:
    params = dict(cfg.model.get("params", {}))
    return {"architecture": str(cfg.model.architecture), "params": params, "scfoundation_dim": scfoundation_dim}


def build_model(cfg, gene_names: list[str], scfoundation_dim: int | None):
    architecture = str(cfg.model.architecture)
    params = dict(cfg.model.get("params", {}))
    if architecture == "1":
        return Architecture1(n_genes=len(gene_names), **params)
    if architecture == "2":
        if scfoundation_dim is None:
            raise ValueError("Architecture 2 requires scfoundation_dim (scFoundation features unavailable)")
        return Architecture2(n_genes=len(gene_names), scfoundation_dim=scfoundation_dim, **params)
    if architecture == "4":
        if scfoundation_dim is None:
            raise ValueError("Architecture 4 requires scfoundation_dim (scFoundation features unavailable)")
        return Architecture4(gene_names=gene_names, scfoundation_dim=scfoundation_dim, **params)
    raise ValueError(f"train_local_neighborhood.py handles architectures '1'/'2'/'4', got {architecture!r}")


def _sample_organ_tech(adata) -> tuple[str | None, str | None]:
    organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
    tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None
    return organ, tech


def main(config_path: str, smoke_steps: int | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    data_prep.apply_sample_selection(cfg)
    data_prep.apply_smoke_override(cfg, smoke_steps)
    architecture = str(cfg.model.architecture)
    device = torch.device(cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    needs_scfoundation = architecture in ("2", "4")

    train_ids = list(cfg.data.train_sample_ids)
    validation_ids = list(cfg.data.get("validation_sample_ids", []))
    test_ids = list(cfg.data.get("test_sample_ids", []))
    if not train_ids:
        raise ValueError("data.train_sample_ids must be non-empty")

    print(f"loading {len(train_ids)} training sample(s)...")
    train_adatas, train_images = data_prep.load_multi_sample_with_images(cfg, train_ids)
    gene_names = train_adatas[0].var_names.tolist()
    data_prep.apply_coord_scale(cfg, train_adatas)

    held_out_adatas: dict[str, tuple] = {}
    for split_name, ids in (("validation", validation_ids), ("test", test_ids)):
        if not ids:
            continue
        print(f"loading {len(ids)} {split_name} sample(s), aligned to the training gene panel...")
        kept_ids, adatas, images_list = data_prep.load_held_out_samples_with_images(cfg, ids, gene_names)
        if split_name == "validation":
            validation_ids = kept_ids
        else:
            test_ids = kept_ids
        for sid, adata, images in zip(kept_ids, adatas, images_list):
            held_out_adatas[str(sid)] = (adata, images, split_name)

    scfoundation_providers: dict[str, object] = {}
    scfoundation_dim = None
    if needs_scfoundation:
        print("building scFoundation context feature providers (frozen checkpoint, one-time per-sample precompute + cache)...")
        all_ids = train_ids + validation_ids + test_ids
        all_adatas = list(train_adatas) + [held_out_adatas[str(sid)][0] for sid in validation_ids + test_ids]
        for sid, adata in zip(all_ids, all_adatas):
            scfoundation_providers[str(sid)] = data_prep.build_scfoundation_provider(cfg, adata, str(sid))
        scfoundation_dim = data_prep._probe_context_feature_dim(
            cfg, train_adatas[0], scfoundation_providers[str(train_ids[0])]
        )
        print(f"scFoundation feature width: {scfoundation_dim}")

    model = build_model(cfg, gene_names, scfoundation_dim).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model built: architecture {architecture}, {n_trainable:,} trainable / {n_total:,} total parameters")

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    start_step = 0
    if (checkpoint_dir / "model_config.json").exists():
        checkpoint.load_trainable_state(model, checkpoint_dir)
        start_step = checkpoint.load_training_state(checkpoint_dir).get("step", 0)
        print(f"resumed from checkpoint at step {start_step}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            f"architecture {architecture} has zero trainable parameters with this config "
            "(check gene_encoder_type / finetune flags) — nothing to train"
        )
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.training.get("lr", 1e-4)))
    loss_fn = StagedGeneLoss(**dict(cfg.training.get("loss", {})))
    diag_stats = diagnostics.attach_diagnostic_hooks(model)

    total_steps = int(cfg.training.total_steps)
    grad_clip = float(cfg.training.get("gradient_clip_val", 1.0))
    log_every = int(cfg.training.get("log_every_n_steps", 50))
    checkpoint_every = int(cfg.training.get("checkpoint_every_n_steps", 2000))
    checkpoint_keep_last = int(cfg.training.get("checkpoint_keep_last", 2))
    eval_every = int(cfg.training.get("eval_every_n_steps", 5000))
    image_mode = str(cfg.training.get("image_mode", "target_zero"))
    context_gex_mode = str(cfg.training.get("context_gex_mode", "full"))
    augment = bool(cfg.training.get("augment_coords", False))

    rng = random.Random(int(cfg.training.get("seed", 0)))
    model.train()
    for step in range(start_step, total_steps):
        sid_idx = rng.randrange(len(train_ids))
        sid = str(train_ids[sid_idx])
        adata, images = train_adatas[sid_idx], train_images[sid_idx]
        coords3d = loaders.get_coords_3d(adata)
        expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        organ, tech = _sample_organ_tech(adata)
        provider = scfoundation_providers.get(sid)

        item = build_masked_item(
            coords3d, expr, adata.obs["slice_id"].to_numpy(), cfg.masking, images, seed=step,
            context_gene_feature_provider=provider if architecture == "2" else None,
            context_extra_feature_provider=provider if architecture == "4" else None,
            organ=organ, tech=tech, augment=augment,
            image_mode=image_mode, context_gex_mode=context_gex_mode,
        )
        item = move_to_device(item, device)
        pred = model(item["context"], item["query"])
        target = item["target_expression"]
        target_idx = getattr(model, "_decoder_target_col_idx", None)
        if target_idx is not None:
            target = target[:, target_idx]

        progress = step / max(1, total_steps)
        result = loss_fn(pred, target, progress)
        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        if step % log_every == 0:
            diag = diagnostics.collect_diagnostics(model, diag_stats)
            print(
                f"step {step}/{total_steps} sample={sid} n_query={target.shape[0]} "
                f"loss={result['loss'].item():.4f} mse={result['mse'].item():.4f} "
                f"pearson_penalty={result['pearson_penalty'].item():.4f} "
                f"pearson_weight={result['pearson_weight']:.2f} {diagnostics.format_diagnostics(diag)}"
            )
        if step > 0 and step % checkpoint_every == 0:
            checkpoint.save_checkpoint(
                model, _model_config_dict(cfg, scfoundation_dim), gene_names, checkpoint_dir, step,
                keep_last=checkpoint_keep_last,
            )
        if step > 0 and step % eval_every == 0 and validation_ids:
            model.eval()
            attn_entropy = diagnostics.compute_attention_entropy(model, diag_stats)
            if attn_entropy is not None:
                print(f"  [val step {step}] attention_entropy={attn_entropy:.3f} nats")
            for sid in validation_ids:
                adata, images, _ = held_out_adatas[str(sid)]
                gene_inputs = {
                    "context_gene_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "2" else None,
                    "context_extra_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "4" else None,
                }
                metrics = evaluate.evaluate_sample(model, cfg, adata, images, gene_inputs, str(sid), "validation", checkpoint_dir)
                primary = metrics["image_modes"][metrics["primary_image_mode"]]["summary"]
                print(f"  [val step {step}] {sid}: PCC={primary['pcc']['mean']:.4f} RMSE={primary['rmse']['mean']:.4f}")
            model.train()

    checkpoint.save_checkpoint(
        model, _model_config_dict(cfg, scfoundation_dim), gene_names, checkpoint_dir, total_steps,
        keep_last=checkpoint_keep_last,
    )

    if test_ids:
        print("running final held-out test evaluation...")
        model.eval()
        for sid in test_ids:
            adata, images, _ = held_out_adatas[str(sid)]
            gene_inputs = {
                "context_gene_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "2" else None,
                "context_extra_feature_provider": scfoundation_providers.get(str(sid)) if architecture == "4" else None,
            }
            metrics = evaluate.evaluate_sample(model, cfg, adata, images, gene_inputs, str(sid), "test", checkpoint_dir)
            primary = metrics["image_modes"][metrics["primary_image_mode"]]["summary"]
            print(f"TEST {sid}: PCC={primary['pcc']['mean']:.4f} RMSE={primary['rmse']['mean']:.4f}")
            if "pcc_raw_log1p" in primary:
                print(f"       notebook-comparable PCC (raw log1p space): {primary['pcc_raw_log1p']['mean']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke_steps", type=int, default=None,
                         help="run only this many steps (overriding config), with checkpoint/eval/log intervals scaled down to match -- for a quick real-hardware smoke test before the full run")
    args = parser.parse_args()
    main(args.config, smoke_steps=args.smoke_steps)
