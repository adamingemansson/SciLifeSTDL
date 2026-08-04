#!/usr/bin/env python3
"""Real manifest/cache trainer for the supervisor conditional-WAE suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAEMaskedGEXDataset,
    ConditionalWAE,
)
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.tensorboard import (
    ConditionalWAESnapshotAccumulator,
    ConditionalWAETensorBoardLogger,
)
from gen3_multiscale.config_identity import config_identity_fingerprint, resolved_config
from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.models.losses import rmse_pcc_reconstruction_loss
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples, save_gen3_preflight_report
from gen3_multiscale.training.train import (
    _code_commit_hash,
    _worktree_diff_hash,
    dataset_manifest_fingerprint,
    deterministic_train_index_for_step,
    expected_tile_encoder_provenance,
    save_best_checkpoint_bundle,
)


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _configure_cpu_threads(training_cfg: dict) -> int:
    """Apply the explicit per-trainer CPU cap before data/model work starts."""
    cpu_threads = int(
        os.environ.get(
            "SCILIFESTDL_CPU_THREADS",
            training_cfg.get("cpu_threads", 8),
        )
    )
    if cpu_threads < 1:
        raise ValueError("training.cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)
    return cpu_threads


def _select_train_item_with_minimum_queries(
    dataset,
    *,
    step: int,
    seed: int,
    minimum_query_spots: int = 2,
):
    """Deterministically bypass masks too small for latent distribution losses."""
    n_items = len(dataset)
    if n_items < 1:
        raise ValueError("conditional latent training dataset is empty")
    start = deterministic_train_index_for_step(step, n_items, seed)
    for offset in range(n_items):
        index = (start + offset) % n_items
        inputs, target, identity = dataset[index]
        if np.asarray(target).shape[0] >= minimum_query_spots:
            return inputs, target, identity, offset
    raise ValueError(
        "conditional latent training requires at least two query spots, but no "
        "eligible mask exists in the dataset"
    )


def _build_model(config: dict, n_genes: int) -> ConditionalWAE:
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
    return ConditionalWAE(
        n_genes,
        conditioner,
        regularizer=model_cfg["regularizer"],
        latent_dim=int(params["latent_dim"]),
        autoencoder_hidden_dim=int(params["autoencoder_hidden_dim"]),
        discriminator_hidden_dim=int(params["discriminator_hidden_dim"]),
        regularizer_weight=float(loss["regularizer_weight"]),
        conditional_mean_weight=float(loss["conditional_mean_weight"]),
        pcc_weight=float(loss["pcc_weight"]),
        n_inference_samples=int(params["n_inference_samples"]),
    )


def _manifest(config: dict, dataset_manifest: dict, preflight_report: dict) -> dict:
    return {
        "version": 1,
        "kind": "conditional_wae_supervisor_run",
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "regularizer": config["model"]["regularizer"],
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
    identity_fields = (
        "kind", "arm", "task", "regularizer", "image_mode",
        "query_gex_visible", "surrounding_gex_visible",
        "config_identity_fingerprint", "dataset_manifest_fingerprint",
        "gene_panel_hash", "cache_content_fingerprint", "cache_content_by_sample",
    )
    changed = [field for field in identity_fields if old.get(field) != new.get(field)]
    if changed:
        raise ValueError(f"conditional-WAE resume refused: changed identity fields {changed}")
    code_changed = (
        old.get("code_commit_hash") is None
        or new.get("code_commit_hash") is None
        or old.get("code_commit_hash") != new.get("code_commit_hash")
        or old.get("code_worktree_diff_hash") != new.get("code_worktree_diff_hash")
    )
    if code_changed and not allow_code_drift:
        raise ValueError(
            "conditional-WAE resume refused: code state changed; pass --allow-code-drift "
            "only after reviewing the difference"
        )
    new["code_drift_acknowledged"] = bool(code_changed and allow_code_drift)


def _stable_seed(seed: int, identity: dict) -> int:
    stable = ":".join(str(identity.get(key)) for key in (
        "sample_id", "stratum", "query_fingerprint",
    ))
    digest = hashlib.sha256(f"conditional-wae:{seed}:{stable}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63)


def _checkpoint_resume_state(checkpoint_dir: Path) -> tuple[bool, dict | None]:
    """Distinguish a real bundle from preflight-only root metadata."""
    has_checkpoint = (checkpoint_dir / "latest_bundle.json").is_file()
    old_manifest = checkpoint_module.load_checkpoint_run_manifest(checkpoint_dir)
    if has_checkpoint and old_manifest is None:
        raise ValueError("conditional-WAE checkpoint has no bundle-bound run manifest")
    return has_checkpoint, old_manifest


@torch.no_grad()
def _validate(model: ConditionalWAE, dataset, *, device: torch.device, seed: int,
              snapshot: ConditionalWAESnapshotAccumulator | None = None,
              samples: dict | None = None) -> dict:
    model.eval()
    totals, rmses, pcc_losses, conditional_mean_rmses = [], [], [], []
    for index in range(len(dataset)):
        inputs, target, identity = dataset[index]
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        generator = torch.Generator(device=device).manual_seed(_stable_seed(seed, identity))
        prediction = model.sample_predictive_distribution(inputs, generator=generator)
        total, rmse, pcc_loss = rmse_pcc_reconstruction_loss(
            prediction["predictive_mean"], target_tensor, pcc_weight=model.pcc_weight,
        )
        _mean_total, mean_rmse, _ = rmse_pcc_reconstruction_loss(
            prediction["conditional_mean_expression"], target_tensor, pcc_weight=model.pcc_weight,
        )
        totals.append(float(total))
        rmses.append(float(rmse))
        pcc_losses.append(float(pcc_loss))
        conditional_mean_rmses.append(float(mean_rmse))
        if snapshot is not None and not snapshot.full:
            if samples is None or inputs.sample_id not in samples:
                raise ValueError("TensorBoard snapshot requires the aligned validation sample")
            # Held-out target GEX is encoded only for a diagnostic Projector view.
            # It is never fed to sample_predictive_distribution or model selection.
            posterior_z = model.expression_encoder(target_tensor)
            snapshot.add(
                inputs=inputs, target=target_tensor, identity=identity,
                prediction=prediction, posterior_z=posterior_z,
                sample=samples[inputs.sample_id],
            )
    model.train()
    return {
        "total": float(np.mean(totals)),
        "rmse": float(np.mean(rmses)),
        "pcc_loss": float(np.mean(pcc_losses)),
        "conditional_mean_rmse": float(np.mean(conditional_mean_rmses)),
        "n_items": len(dataset),
    }


def _tensorboard_gene_indices(config: dict, dataset_manifest: dict,
                              gene_names: list[str], count: int) -> list[int]:
    if count < 1:
        return []
    artifact_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
    if not artifact_path:
        return list(range(min(count, len(gene_names))))
    artifact = load_train_derived_gene_panels(artifact_path, dataset_manifest)
    panel = artifact["panels"].get("train_log1p_variance_top50") or []
    positions = {gene: index for index, gene in enumerate(gene_names)}
    return [positions[gene] for gene in panel[:count]]


def run_conditional_wae_training(
    config_path: str, *, smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    config = resolved_config(config_path)
    static_audit_conditional_wae_config(config)
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
        cfg_om,
        dataset_manifest,
        train_ids + validation_ids,
        expected_tile_encoder_provenance(config),
    )
    strata = config["masking"]["strata"]
    n_smoke_masks = max(1, len(strata))
    train_samples = {sample_id: samples[sample_id] for sample_id in train_ids}
    validation_samples = {sample_id: samples[sample_id] for sample_id in validation_ids}
    train_schedule = build_gen3_mask_schedule(
        dataset_manifest,
        train_samples,
        strata,
        "train",
        n_training_masks_per_sample=(
            n_smoke_masks if smoke else int(data_cfg.get("n_training_masks_per_sample", 500))
        ),
    )
    validation_schedule = build_gen3_mask_schedule(
        dataset_manifest,
        validation_samples,
        strata,
        "validation",
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
            dataset_manifest, validation_samples, validation_schedule, strata, novae_enabled=False,
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
    device = torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    gene_names = list(dataset_manifest["gene_panel"])
    model = _build_model(config, len(gene_names)).to(device)
    generator_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("discriminator.")
    ]
    parameter_groups = [{"params": generator_parameters, "name": "generator"}]
    if model.discriminator is not None:
        parameter_groups.append({"params": list(model.discriminator.parameters()), "name": "discriminator"})
    optimizer_cfg = training_cfg.get("optimizer") or {}
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
                raise ValueError("conditional-WAE checkpoint lacks optimizer/RNG state")
            resume_step = int(checkpoint_module.load_training_state(checkpoint_dir)["step"])
            print(f"resumed conditional WAE from step {resume_step}", flush=True)
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
        tensorboard_logger = ConditionalWAETensorBoardLogger(
            tensorboard_cfg["log_dir"],
            max_spatial_samples=int(tensorboard_cfg.get("max_spatial_samples", 4)),
            purge_step=(resume_step if resume_step > 0 else None),
        )
        print(f"TensorBoard logging: {tensorboard_cfg['log_dir']}", flush=True)
    started = time.time()
    step = resume_step
    completion_reason = "total_steps_reached"
    model.train()
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
        discriminator_loss = None
        if model.discriminator is not None:
            optimizer.zero_grad(set_to_none=True)
            discriminator_loss = model.compute_discriminator_loss(
                target_tensor,
                generator=torch.Generator(device=device).manual_seed(seed + 2 * step),
            )
            if not torch.isfinite(discriminator_loss):
                raise RuntimeError(f"[step {step}] non-finite discriminator loss")
            discriminator_loss.backward()
            discriminator_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.discriminator.parameters(), clip_value,
            )
            if not torch.isfinite(discriminator_grad_norm):
                raise RuntimeError(f"[step {step}] non-finite discriminator gradient")
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses = model.compute_generator_losses(
            inputs,
            target_tensor,
            generator=torch.Generator(device=device).manual_seed(seed + 2 * step + 1),
        )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"[step {step}] non-finite generator loss")
        losses["total"].backward()
        generator_grad_norm = torch.nn.utils.clip_grad_norm_(generator_parameters, clip_value)
        if not torch.isfinite(generator_grad_norm) or generator_grad_norm <= 0:
            raise RuntimeError(f"[step {step}] invalid generator gradient norm {generator_grad_norm}")
        optimizer.step()
        step += 1
        if smoke or step % log_every == 0:
            pieces = [
                f"total={float(losses['total'].detach()):.6f}",
                f"reconstruction={float(losses['reconstruction_loss'].detach()):.6f}",
                f"conditional_mean={float(losses['conditional_mean_loss'].detach()):.6f}",
                f"prior={float(losses['prior_loss'].detach()):.6f}",
                f"grad_norm={float(generator_grad_norm):.4f}",
            ]
            if discriminator_loss is not None:
                pieces.append(f"discriminator={float(discriminator_loss.detach()):.6f}")
            print(f"[step {step}] train: " + ", ".join(pieces), flush=True)
            if tensorboard_logger is not None:
                tensorboard_logger.add_train_scalars(
                    step, losses, grad_norm=generator_grad_norm,
                    discriminator_loss=discriminator_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
        if validation_dataset is not None and (smoke or step % eval_every == 0):
            snapshot = None
            if tensorboard_logger is not None:
                snapshot_every = max(1, int(tensorboard_cfg.get("snapshot_every_n_evals", 5)))
                evaluation_number = max(1, step // eval_every)
                if (evaluation_number - 1) % snapshot_every == 0:
                    logged_gene_indices = _tensorboard_gene_indices(
                        config, dataset_manifest, gene_names,
                        int(tensorboard_cfg.get("spatial_gene_count", 2)),
                    )
                    snapshot = ConditionalWAESnapshotAccumulator(
                        sample_records={
                            sample_id: dataset_manifest["samples"][sample_id]
                            for sample_id in validation_ids
                        },
                        gene_names=gene_names,
                        logged_gene_indices=logged_gene_indices,
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
    summary = {
        "ok": True,
        "smoke": smoke,
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "regularizer": config["model"]["regularizer"],
        "final_step": step,
        "elapsed_seconds": time.time() - started,
        "completion_reason": completion_reason,
        "checkpoint_dir": str(checkpoint_dir),
    }
    if tensorboard_logger is not None:
        tensorboard_logger.close()
    print(f"conditional WAE training finished: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    run_conditional_wae_training(
        args.config, smoke=args.smoke, allow_code_drift=args.allow_code_drift,
    )


if __name__ == "__main__":
    main()
