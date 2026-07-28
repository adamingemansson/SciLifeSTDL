"""Architecture 3, Stage B — spatial transformer predicting a latent code,
built on top of a Stage-A checkpoint (train_arch3_stage_a.py, must be run
first). Same masked-item-per-step loop shape as
train_local_neighborhood.py, with the two-term loss GPT review answer #2
specifies: latent_loss (predicted latent vs Stage-A's own encoder applied
to the TRUE query expression, teacher-forced) + gene_loss (StagedGeneLoss
on the decoded prediction).

Usage:
    python3 -m gen2_architectures.training.train_arch3_stage_b \\
        --config gen2_architectures/configs/arch3_stage_b_spatial.yaml
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen2_architectures.data import loaders
from gen2_architectures.data.masked_item import build_masked_item
from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder
from gen2_architectures.models.arch3_stage_b_latent_transformer import Architecture3StageB
from gen2_architectures.models.components import StagedGeneLoss
from gen2_architectures.training import checkpoint, data_prep, diagnostics, evaluate
from gen2_architectures.training.validation import move_to_device


def _load_stage_a(stage_a_checkpoint_dir: str, gene_names: list[str]) -> DenoisingTranscriptomeAutoencoder:
    import json

    stage_a_dir = Path(stage_a_checkpoint_dir)
    stage_a_config = json.loads((stage_a_dir / "model_config.json").read_text())
    n_genes = len(gene_names)
    if stage_a_config["n_genes"] != n_genes:
        raise ValueError(
            f"Stage A was pretrained on {stage_a_config['n_genes']} genes, but this run's "
            f"training gene panel has {n_genes}. Stage A and Stage B must share the exact "
            "same gene panel (same train_sample_ids / QC settings) or the autoencoder's "
            "input/output width is meaningless here."
        )
    # 2026-07-27 bugfix: the count check above only caught a WIDTH mismatch.
    # Two panels can have the same width but different genes or a different
    # order -- same n_genes, silently wrong semantics, since the
    # autoencoder's input/output columns are positional, not name-keyed.
    # Compare the exact ordered gene list against Stage A's own saved
    # gene_names.json, not just its width.
    stage_a_gene_names_path = stage_a_dir / "gene_names.json"
    if stage_a_gene_names_path.is_file():
        stage_a_gene_names = json.loads(stage_a_gene_names_path.read_text())
        if list(stage_a_gene_names) != list(gene_names):
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(stage_a_gene_names, gene_names)) if a != b),
                min(len(stage_a_gene_names), len(gene_names)),
            )
            raise ValueError(
                f"Stage A's gene panel ({stage_a_gene_names_path}) has the same width "
                f"({n_genes}) as this run's, but the ORDERED gene identities differ "
                f"(first mismatch at index {first_diff}: "
                f"{stage_a_gene_names[first_diff] if first_diff < len(stage_a_gene_names) else '<end>'!r} vs "
                f"{gene_names[first_diff] if first_diff < len(gene_names) else '<end>'!r}). "
                "Stage A and Stage B must share the exact same ordered gene panel."
            )
    autoencoder = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, **stage_a_config["params"])
    checkpoint.load_trainable_state(autoencoder, stage_a_checkpoint_dir)
    return autoencoder


def main(
    config_path: str, smoke_steps: int | None = None, max_wall_clock_hours_override: float | None = None,
    skip_final_eval: bool = False,
) -> None:
    cfg = OmegaConf.load(config_path)
    data_prep.seed_everything(int(cfg.training.get("seed", 0)))
    data_prep.apply_sample_selection(cfg)
    data_prep.apply_smoke_override(cfg, smoke_steps)
    if max_wall_clock_hours_override is not None:
        cfg.training.max_wall_clock_hours = max_wall_clock_hours_override
    device = torch.device(cfg.training.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

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

    autoencoder = _load_stage_a(cfg.model.stage_a_checkpoint_dir, gene_names).to(device)
    # OmegaConf.to_container (not dict(...)) -- a shallow dict() leaves
    # nested list-valued params (organ_vocab, tech_vocab, ...) as
    # OmegaConf ListConfig objects, which json.dump cannot serialize --
    # see train_local_neighborhood.py::_model_config_dict's own comment
    # for the real bug this fixes (hit on the actual training server).
    params = OmegaConf.to_container(cfg.model.get("params", {}), resolve=True)
    model = Architecture3StageB(autoencoder, **params).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage B built: {n_trainable:,} trainable / {n_total:,} total parameters")

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    start_step = 0
    if (checkpoint_dir / "model_config.json").exists():
        checkpoint.load_trainable_state(model, checkpoint_dir)
        start_step = checkpoint.load_training_state(checkpoint_dir).get("step", 0)
        print(f"resumed from checkpoint at step {start_step}")

    param_groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "autoencoder" not in n],
                      "lr": float(cfg.training.get("lr", 1e-4))}]
    autoencoder_params = [p for n, p in model.named_parameters() if p.requires_grad and "autoencoder" in n]
    if autoencoder_params:
        param_groups.append({
            "params": autoencoder_params,
            "lr": float(cfg.training.get("lr", 1e-4)) * model.autoencoder_lr_multiplier,
        })
    optimizer = torch.optim.AdamW(param_groups)
    loss_fn = StagedGeneLoss(**dict(cfg.training.get("loss", {})))
    latent_loss_weight = float(cfg.training.get("latent_loss_weight", 1.0))
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
    # 2026-07-27 (GPT-audit-flagged): see train_local_neighborhood.py's own
    # comment -- context patches near the mask boundary could still see
    # real pixels from inside the supposedly destroyed region unless this
    # is enabled.
    strict_broken_region = bool(cfg.get("data", {}).get("strict_broken_region", False))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    wall_clock_deadline = data_prep.resolve_wall_clock_deadline(cfg)
    progress_fn = data_prep.make_progress_fn(wall_clock_deadline, total_steps)

    rng = random.Random(int(cfg.training.get("seed", 0)))
    model.train()
    last_step = start_step
    for step in range(start_step, total_steps):
        last_step = step
        if wall_clock_deadline is not None and time.monotonic() >= wall_clock_deadline:
            print(f"step {step}/{total_steps}: max_wall_clock_hours budget reached, stopping training early")
            break
        sid_idx = rng.randrange(len(train_ids))
        adata, images = train_adatas[sid_idx], train_images[sid_idx]
        coords3d = loaders.get_coords_3d(adata)
        expr = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
        organ = str(adata.obs["organ"].iloc[0]) if "organ" in adata.obs else None
        tech = str(adata.obs["tech"].iloc[0]) if "tech" in adata.obs else None

        item = build_masked_item(
            coords3d, expr, adata.obs["slice_id"].to_numpy(), cfg.masking, images, seed=step,
            organ=organ, tech=tech, augment=augment,
            image_mode=image_mode, context_gex_mode=context_gex_mode,
            strict_broken_region=strict_broken_region,
        )
        item = move_to_device(item, device)
        out = model(item["context"], item["query"])
        target = item["target_expression"]

        true_latent = model.true_latent(target)
        latent_loss = torch.nn.functional.mse_loss(out["predicted_latent"], true_latent)
        progress = progress_fn(step)
        gene_result = loss_fn(out["predicted_expression"], target, progress)
        total_loss = latent_loss_weight * latent_loss + gene_result["loss"]

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        if not data_prep.is_finite_update(total_loss, grad_norm):
            print(
                f"step {step}/{total_steps}: non-finite loss/grad_norm "
                f"(loss={total_loss.item()}, grad_norm={grad_norm.item()}) -- "
                f"skipping this optimizer step to avoid corrupting the model",
                flush=True,
            )
            continue
        optimizer.step()

        if step % log_every == 0:
            diag = diagnostics.collect_diagnostics(model, diag_stats)
            predicted_latent_norm = out["predicted_latent"].detach().float().norm(dim=-1).mean().item()
            predicted_expression_norm = out["predicted_expression"].detach().float().norm(dim=-1).mean().item()
            print(
                f"step {step}/{total_steps} n_query={target.shape[0]} total_loss={total_loss.item():.4f} "
                f"latent_loss={latent_loss.item():.4f} gene_mse={gene_result['mse'].item():.4f} "
                f"pearson_penalty={gene_result['pearson_penalty'].item():.4f} "
                f"predicted_latent_norm={predicted_latent_norm:.3f} predicted_expression_norm={predicted_expression_norm:.3f} "
                f"{diagnostics.format_diagnostics(diag)}"
            )
        if step > 0 and step % checkpoint_every == 0:
            checkpoint.save_checkpoint(
                model, {"stage_a_checkpoint_dir": str(cfg.model.stage_a_checkpoint_dir), "params": params},
                gene_names, checkpoint_dir, step, keep_last=checkpoint_keep_last,
            )
        if step > 0 and step % eval_every == 0 and validation_ids:
            model.eval()
            attn_entropy = diagnostics.compute_attention_entropy(model, diag_stats)
            if attn_entropy is not None:
                print(f"  [val step {step}] attention_entropy={attn_entropy:.3f} nats")
            for sid in validation_ids:
                v_adata, v_images, _ = held_out_adatas[str(sid)]
                metrics = evaluate.evaluate_sample(model, cfg, v_adata, v_images, {}, str(sid), "validation", checkpoint_dir)
                primary = metrics["image_modes"][metrics["primary_image_mode"]]["summary"]
                print(f"  [val step {step}] {sid}: PCC={primary['pcc']['mean']:.4f} RMSE={primary['rmse']['mean']:.4f}")
            model.train()

    # 2026-07-27 bugfix: see train_local_neighborhood.py's own comment --
    # save the actual last step reached, not the total_steps safety cap.
    checkpoint.save_checkpoint(
        model, {"stage_a_checkpoint_dir": str(cfg.model.stage_a_checkpoint_dir), "params": params},
        gene_names, checkpoint_dir, last_step, keep_last=checkpoint_keep_last,
    )

    if skip_final_eval:
        print(
            "--skip_final_eval set: skipping the final held-out test evaluation. "
            "The trained checkpoint above is already saved -- run it separately later "
            "(e.g. via a standalone evaluation script against this checkpoint_dir)."
        )
    elif test_ids:
        print("running final held-out test evaluation...")
        model.eval()
        for sid in test_ids:
            t_adata, t_images, _ = held_out_adatas[str(sid)]
            metrics = evaluate.evaluate_sample(model, cfg, t_adata, t_images, {}, str(sid), "test", checkpoint_dir)
            primary = metrics["image_modes"][metrics["primary_image_mode"]]["summary"]
            print(f"TEST {sid}: PCC={primary['pcc']['mean']:.4f} RMSE={primary['rmse']['mean']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke_steps", type=int, default=None,
                         help="run only this many steps (overriding config), with checkpoint/eval/log intervals scaled down to match -- for a quick real-hardware smoke test before the full run")
    parser.add_argument("--skip_final_eval", action="store_true",
                         help="skip the final held-out test evaluation entirely -- the checkpoint is still saved "
                              "(saving happens before evaluation), run evaluation separately later against it")
    args = parser.parse_args()
    main(args.config, smoke_steps=args.smoke_steps, skip_final_eval=args.skip_final_eval)
