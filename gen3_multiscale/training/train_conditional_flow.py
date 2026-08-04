#!/usr/bin/env python3
"""Real manifest/cache trainer for the matched MK conditional-flow suite."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_flow import (
    Architecture1ImageConditioner,
    ConditionalLatentFlow,
    ConditionalWAEMaskedGEXDataset,
)
from gen3_multiscale.conditional_flow.contract import static_audit_conditional_flow_config
from gen3_multiscale.conditional_wae.tensorboard import (
    ConditionalFlowTensorBoardLogger,
    ConditionalWAESnapshotAccumulator,
)
from gen3_multiscale.config_identity import config_identity_fingerprint, resolved_config
from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import (
    load_and_preflight_samples,
    save_gen3_preflight_report,
)
from gen3_multiscale.training.train import (
    _code_commit_hash,
    _worktree_diff_hash,
    dataset_manifest_fingerprint,
    expected_tile_encoder_provenance,
    save_best_checkpoint_bundle,
)
from gen3_multiscale.training.train_conditional_wae import (
    _atomic_json,
    _checkpoint_resume_state,
    _configure_cpu_threads,
    _select_train_item_with_minimum_queries,
    _stable_seed,
    _tensorboard_gene_indices,
    _validate,
)


def _build_model(config: dict, n_genes: int) -> ConditionalLatentFlow:
    model_cfg = config["model"]
    params = model_cfg["params"]
    conditioner = Architecture1ImageConditioner(
        n_genes=n_genes,
        image_feature_dim=int(params["image_feature_dim"]),
        gex_feature_dim=int(params["gex_feature_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        n_heads=int(params["n_heads"]),
        n_blocks=int(params["n_blocks"]),
        dense_threshold=int(params["dense_threshold"]),
        sparse_k=int(params["sparse_k"]),
        dropout=float(params.get("dropout", 0.1)),
    )
    loss = config["loss"]
    return ConditionalLatentFlow(
        n_genes,
        conditioner,
        coupling=model_cfg["coupling"],
        latent_dim=int(params["latent_dim"]),
        autoencoder_hidden_dim=int(params["autoencoder_hidden_dim"]),
        n_flow_blocks=int(params["n_flow_blocks"]),
        n_flow_samples=int(params["n_inference_samples"]),
        n_ode_steps=int(params["n_ode_steps"]),
        flow_weight=float(loss["flow_weight"]),
        conditional_mean_weight=float(loss["conditional_mean_weight"]),
        pcc_weight=float(loss["pcc_weight"]),
        ot_epsilon=float(params.get("ot_epsilon", 0.1)),
        ot_sinkhorn_iters=int(params.get("ot_sinkhorn_iters", 20)),
    )


def _manifest(config: dict, dataset_manifest: dict, preflight_report: dict) -> dict:
    return {
        "version": 1,
        "kind": "conditional_flow_supervisor_run",
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "coupling": config["model"]["coupling"],
        "image_mode": "full_visible",
        "query_gex_visible": False,
        "surrounding_gex_visible": bool(config["model"]["include_observed_gex"]),
        "config_identity_fingerprint": config_identity_fingerprint(config),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(dataset_manifest["gene_panel"]),
        "cache_content_fingerprint": preflight_report["cache_content_fingerprint"],
        "cache_content_by_sample": preflight_report["cache_content_by_sample"],
        "code_commit_hash": _code_commit_hash(),
        "code_worktree_diff_hash": _worktree_diff_hash(),
    }


def _verify_resume(old: dict, new: dict, *, allow_code_drift: bool) -> None:
    fields = (
        "kind", "arm", "task", "coupling", "image_mode", "query_gex_visible",
        "surrounding_gex_visible", "config_identity_fingerprint",
        "dataset_manifest_fingerprint", "gene_panel_hash",
        "cache_content_fingerprint", "cache_content_by_sample",
    )
    changed = [field for field in fields if old.get(field) != new.get(field)]
    if changed:
        raise ValueError(f"conditional-flow resume refused: changed identity fields {changed}")
    code_changed = (
        old.get("code_commit_hash") is None
        or new.get("code_commit_hash") is None
        or old.get("code_commit_hash") != new.get("code_commit_hash")
        or old.get("code_worktree_diff_hash") != new.get("code_worktree_diff_hash")
    )
    if code_changed and not allow_code_drift:
        raise ValueError(
            "conditional-flow resume refused: code state changed; pass --allow-code-drift "
            "only after reviewing the difference"
        )
    new["code_drift_acknowledged"] = bool(code_changed and allow_code_drift)


def run_conditional_flow_training(
    config_path: str, *, smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    config = resolved_config(config_path)
    static_audit_conditional_flow_config(config)
    data_cfg, training_cfg = config["data"], config["training"]
    cpu_threads = _configure_cpu_threads(training_cfg)
    print(f"CPU thread cap: {cpu_threads}", flush=True)
    dataset_manifest = load_dataset_manifest(data_cfg["gen3_manifest_path"])
    train_ids = list(dataset_manifest["train_sample_ids"])
    validation_ids = list(dataset_manifest["validation_sample_ids"])
    if smoke:
        train_ids = train_ids[:1]
        validation_ids = validation_ids[:1]
    cfg_om = OmegaConf.create(config)
    samples, preflight_report = load_and_preflight_samples(
        cfg_om, dataset_manifest, train_ids + validation_ids,
        expected_tile_encoder_provenance(config),
    )
    strata = config["masking"]["strata"]
    n_smoke_masks = max(1, len(strata))
    train_samples = {sample_id: samples[sample_id] for sample_id in train_ids}
    validation_samples = {sample_id: samples[sample_id] for sample_id in validation_ids}
    train_schedule = build_gen3_mask_schedule(
        dataset_manifest, train_samples, strata, "train",
        n_training_masks_per_sample=(
            n_smoke_masks if smoke else int(data_cfg.get("n_training_masks_per_sample", 500))
        ),
    )
    validation_schedule = build_gen3_mask_schedule(
        dataset_manifest, validation_samples, strata, "validation",
        split_counts={
            "validation": n_smoke_masks if smoke else int(data_cfg.get("n_validation_masks", 4)),
        },
        split_seeds={"validation": 700_000},
    ) if validation_samples else None
    base_train = Gen3SpatialFieldDataset(
        dataset_manifest, train_samples, train_schedule, strata, novae_enabled=False,
    )
    include_observed_gex = bool(config["model"]["include_observed_gex"])
    train_dataset = ConditionalWAEMaskedGEXDataset(
        base_train, include_observed_gex=include_observed_gex,
    )
    validation_dataset = None
    if validation_schedule is not None:
        base_validation = Gen3SpatialFieldDataset(
            dataset_manifest, validation_samples, validation_schedule, strata,
            novae_enabled=False,
        )
        boundary_report = base_validation.validate_boundary_schedule()
        print(
            f"validation boundary preflight: PASS ({boundary_report['n_items_checked']} masks)",
            flush=True,
        )
        validation_dataset = ConditionalWAEMaskedGEXDataset(
            base_validation, include_observed_gex=include_observed_gex,
        )

    seed = int(training_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loop_rng = random.Random(seed)
    device = torch.device(
        training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    gene_names = list(dataset_manifest["gene_panel"])
    model = _build_model(config, len(gene_names)).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_cfg = training_cfg.get("optimizer") or {}
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    run_manifest = _manifest(config, dataset_manifest, preflight_report)
    resume_step = 0
    if not smoke:
        has_checkpoint, old_manifest = _checkpoint_resume_state(checkpoint_dir)
        if old_manifest is not None:
            _verify_resume(old_manifest, run_manifest, allow_code_drift=allow_code_drift)
        if has_checkpoint:
            checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
            checkpoint_module.load_trainable_state(model, checkpoint_dir)
            if not checkpoint_module.load_optimizer_and_rng_state(
                optimizer, checkpoint_dir, rng=loop_rng,
            ):
                raise ValueError("conditional-flow checkpoint lacks optimizer/RNG state")
            resume_step = int(checkpoint_module.load_training_state(checkpoint_dir)["step"])
            print(f"resumed conditional flow from step {resume_step}", flush=True)
        save_gen3_preflight_report(preflight_report, checkpoint_dir / "preflight_report.json")
        _atomic_json(run_manifest, checkpoint_dir / "run_manifest.json")

    total_steps = int(training_cfg.get("total_steps", 1))
    step_target = resume_step + 1 if smoke else total_steps
    max_hours = float(training_cfg.get("max_wall_clock_hours", 8.0))
    log_every = max(1, int(training_cfg.get("log_every_n_steps", 50)))
    eval_every = max(1, int(training_cfg.get("eval_every_n_steps", 2000)))
    checkpoint_every = max(1, int(training_cfg.get("checkpoint_every_n_steps", 2000)))
    keep_last = int(training_cfg.get("checkpoint_keep_last", 2))
    clip_value = float(training_cfg.get("gradient_clip_val", 1.0))
    history_path = checkpoint_dir / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() and not smoke else []
    best_total = min((entry["total"] for entry in history), default=float("inf"))
    tensorboard_cfg = dict((config.get("evaluation") or {}).get("tensorboard") or {})
    tensorboard_logger = None
    if bool(tensorboard_cfg.get("enabled", False)) and not smoke:
        tensorboard_logger = ConditionalFlowTensorBoardLogger(
            tensorboard_cfg["log_dir"],
            max_spatial_samples=int(tensorboard_cfg.get("max_spatial_samples", 4)),
        )
        print(f"TensorBoard logging: {tensorboard_cfg['log_dir']}", flush=True)
    started = time.time()
    step = resume_step
    completion_reason = "total_steps_reached"
    model.train()
    try:
        while step < step_target:
            if (time.time() - started) / 3600 >= max_hours:
                completion_reason = "wall_clock_limit_reached"
                break
            inputs, target, _identity, skipped_small_masks = (
                _select_train_item_with_minimum_queries(
                    train_dataset, step=step, seed=seed,
                )
            )
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
            if skipped_small_masks and (step == resume_step or step % log_every == 0):
                print(
                    f"[step {step}] skipped {skipped_small_masks} undersized mask(s) "
                    "before selecting a mask with at least two query spots",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(
                inputs, target_tensor,
                generator=torch.Generator(device=device).manual_seed(seed + step),
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"[step {step}] non-finite total loss")
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_value)
            if not torch.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f"[step {step}] invalid gradient norm {grad_norm}")
            optimizer.step()
            step += 1
            if smoke or step % log_every == 0:
                print(
                    f"[step {step}] train: total={float(losses['total'].detach()):.6f}, "
                    f"reconstruction={float(losses['reconstruction_loss'].detach()):.6f}, "
                    f"conditional_mean={float(losses['conditional_mean_loss'].detach()):.6f}, "
                    f"flow={float(losses['flow_loss'].detach()):.6f}, "
                    f"grad_norm={float(grad_norm):.4f}",
                    flush=True,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.add_train_scalars(
                        step, losses, grad_norm=grad_norm,
                        learning_rate=optimizer.param_groups[0]["lr"],
                    )
            if validation_dataset is not None and (smoke or step % eval_every == 0):
                snapshot = None
                if tensorboard_logger is not None:
                    snapshot_every = max(
                        1, int(tensorboard_cfg.get("snapshot_every_n_evals", 5))
                    )
                    evaluation_number = max(1, step // eval_every)
                    if (evaluation_number - 1) % snapshot_every == 0:
                        snapshot = ConditionalWAESnapshotAccumulator(
                            sample_records={
                                sample_id: dataset_manifest["samples"][sample_id]
                                for sample_id in validation_ids
                            },
                            gene_names=gene_names,
                            logged_gene_indices=_tensorboard_gene_indices(
                                config, dataset_manifest, gene_names,
                                int(tensorboard_cfg.get("spatial_gene_count", 2)),
                            ),
                            max_points=int(tensorboard_cfg.get("embedding_max_points", 5000)),
                            max_points_per_item=int(
                                tensorboard_cfg.get("embedding_max_points_per_item", 64)
                            ),
                            thumbnail_max_points=int(
                                tensorboard_cfg.get("thumbnail_max_points", 512)
                            ),
                            thumbnail_size=int(tensorboard_cfg.get("thumbnail_size", 48)),
                        )
                entry = {
                    "step": step,
                    **_validate(
                        model, validation_dataset, device=device, seed=seed,
                        snapshot=snapshot, samples=validation_samples,
                    ),
                }
                print(
                    f"[step {step}] validation: total={entry['total']:.6f}, "
                    f"rmse={entry['rmse']:.6f}, "
                    f"conditional_mean_rmse={entry['conditional_mean_rmse']:.6f}",
                    flush=True,
                )
                if tensorboard_logger is not None:
                    tensorboard_logger.add_validation_scalars(step, entry)
                    if snapshot is not None:
                        tensorboard_logger.add_snapshot(step, snapshot)
                if not smoke:
                    history.append(entry)
                    _atomic_json(history, history_path)
                    if entry["total"] < best_total:
                        best_total = entry["total"]
                        save_best_checkpoint_bundle(
                            model, config, gene_names, checkpoint_dir / "best",
                            step=step, val_loss=entry["total"], run_manifest=run_manifest,
                        )
            if not smoke and step % checkpoint_every == 0 and step < step_target:
                checkpoint_module.save_checkpoint(
                    model, config, gene_names, checkpoint_dir, step,
                    extra_metadata={"completion_reason": "in_progress"},
                    keep_last=keep_last, optimizer=optimizer, rng=loop_rng,
                    run_manifest=run_manifest,
                )
        if not smoke and step > resume_step:
            checkpoint_module.save_checkpoint(
                model, config, gene_names, checkpoint_dir, step,
                extra_metadata={"completion_reason": completion_reason},
                keep_last=keep_last, optimizer=optimizer, rng=loop_rng,
                run_manifest=run_manifest,
            )
    finally:
        if tensorboard_logger is not None:
            tensorboard_logger.close()
    summary = {
        "ok": True,
        "smoke": smoke,
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "coupling": config["model"]["coupling"],
        "final_step": step,
        "elapsed_seconds": time.time() - started,
        "completion_reason": completion_reason,
        "checkpoint_dir": str(checkpoint_dir),
    }
    print(f"conditional flow training finished: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    run_conditional_flow_training(
        args.config, smoke=args.smoke, allow_code_drift=args.allow_code_drift,
    )


if __name__ == "__main__":
    main()
