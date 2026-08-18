#!/usr/bin/env python3
"""Prepare the controlled four-arm residual WAE-MMD factorial."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen4.uni2_spot_cache import require_uni2_spot_cache_coverage
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    tensorboard_block,
    whole_slide_validation_block,
)
from gen3_multiscale.training import checkpoint as checkpoint_module


ARM_ORDER = (
    "mk_rwae_standard_nofilm",
    "mk_rwae_standard_film",
    "mk_rwae_conditional_nofilm",
    "mk_rwae_conditional_film",
)
ARM_DESIGN = {
    "mk_rwae_standard_nofilm": ("standard", "none"),
    "mk_rwae_standard_film": ("standard", "film"),
    "mk_rwae_conditional_nofilm": ("conditional", "none"),
    "mk_rwae_conditional_film": ("conditional", "film"),
}


def prepare_mk_residual_wae_suite(
    *, deterministic_config: str, deterministic_checkpoint: str,
    manifest: str, train_gene_panels: str, output_root: str,
    hours: float = 8.0, gpus: tuple[int, ...] = (0, 2, 3, 5),
    cpu_threads: int = 8, latent_dim: int = 64,
) -> dict:
    if hours <= 0 or cpu_threads < 1 or latent_dim < 1:
        raise ValueError("hours, cpu_threads and latent_dim must be positive")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("gpus must contain exactly four distinct ids")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"{root} already exists; suite roots are immutable")

    source_config_path = Path(deterministic_config).expanduser().resolve()
    source_config = yaml.safe_load(source_config_path.read_text())
    source_model = source_config.get("model") or {}
    source_params = source_model.get("params") or {}
    if source_model.get("arm") != "mk_wb_parallel_gated":
        raise ValueError("deterministic_config must be mk_wb_parallel_gated")
    if not bool(source_params.get("deterministic_only", False)):
        raise ValueError("deterministic source config is not deterministic")
    if source_params.get("structured_composition") != "parallel_gated":
        raise ValueError("deterministic source must use parallel_gated composition")

    manifest_path = Path(manifest).expanduser().resolve()
    dataset_manifest = load_dataset_manifest(manifest_path)
    configured_manifest = Path(source_config["data"]["gen3_manifest_path"]).resolve()
    if configured_manifest != manifest_path:
        raise ValueError("deterministic source and requested suite use different manifests")
    gene_names = list(dataset_manifest["gene_panel"])
    if len(dataset_manifest.get("validation_sample_ids", [])) != 14:
        raise ValueError("expanded MK cohort must contain exactly 14 validation slides")

    panel_path = Path(train_gene_panels).expanduser().resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, dataset_manifest)
    required_panels = {
        "train_log1p_variance_top50", "train_log1p_variance_top200",
        "train_within_slide_variance_top50", "train_within_slide_variance_top200",
    }
    if not required_panels.issubset(panel_artifact["panels"]):
        raise ValueError("expanded global and within-slide panels are required")

    cache_root = Path(source_config["data"]["gen3_uni2_spot_feature_cache_dir"]).resolve()
    require_uni2_spot_cache_coverage(cache_root, dataset_manifest["samples"])
    identity = checkpoint_module.resolve_checkpoint_identity(deterministic_checkpoint)
    if identity.weights_sha256 is None:
        raise ValueError("deterministic source has no trainable weights")
    checkpoint_module.verify_gene_names(identity.resolved_dir, gene_names)
    bundled_config = json.loads((identity.resolved_dir / "model_config.json").read_text())
    if (bundled_config.get("model") or {}).get("arm") != "mk_wb_parallel_gated":
        raise ValueError("checkpoint bundle is not mk_wb_parallel_gated")
    if (bundled_config.get("model") or {}) != source_model:
        raise ValueError(
            "deterministic_config model does not exactly match the checkpoint's "
            "bundled model config"
        )

    root.mkdir(parents=True)
    for directory in ("configs", "checkpoints", "logs", "tensorboard", "control"):
        (root / directory).mkdir()

    seed = int((source_config.get("training") or {}).get("seed", 0))
    arms = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        prior_mode, encoder_conditioning = ARM_DESIGN[arm]
        config = copy.deepcopy(source_config)
        params = copy.deepcopy(source_params)
        params.update({
            "deterministic_only": False,
            "prior_mode": prior_mode,
            "encoder_conditioning": encoder_conditioning,
            "film_layers": ["first", "second"],
            "film_shared_generator": False,
            "latent_dim": int(latent_dim),
            "latent_residual_mode": "antithetic_zero_mean",
            "autoencoder_hidden_dim": int(params.get("autoencoder_hidden_dim", 1024)),
            "discriminator_hidden_dim": 256,
            "n_inference_samples": 8,
            "conditional_prior_hidden_dim": 256,
            "conditional_prior_context_weight": 1.0,
            "conditional_prior_anchor_weight": 0.1,
            "gene_encoder_source": "linear",
            "z_noise_std": 0.0,
            "freeze_deterministic_backbone": True,
            "deterministic_backbone_checkpoint": str(identity.resolved_dir),
            "deterministic_backbone_weights_sha256": identity.weights_sha256,
        })
        config["experiment_name"] = f"mk_residual_wae_{arm}_v1"
        config["model"] = {
            "arm": arm,
            "kind": "conditional_wae",
            "task": "he_to_st",
            "regularizer": "mmd",
            "include_observed_gex": False,
            "image_mode": "full_visible",
            "params": params,
        }
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.1,
            "conditional_mean_weight": 0.0,
            "local_gradient_weight": 0.0,
            "wide_gradient_weight": 0.0,
        }
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(
            root, dataset_manifest,
        )
        config["evaluation"]["structured_field_metrics"] = {
            "enabled": True,
            "all_split_slides": True,
            "chunk_size": 2048,
            "local_k": 6,
            "wide_k": 18,
            "nontrivial_gradient_threshold_training_sd": 0.25,
            "minimum_noise_ceiling": 0.05,
        }
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            "device": "cuda",
            "cpu_threads": int(cpu_threads),
            "seed": seed,
            "lr": 1e-4,
            "gradient_accumulation_steps": 1,
            "best_monitor": "generator_total",
            "optimizer": {"weight_decay": 0.01},
        })
        config["documented_divergences"] = [
            "model.arm", "model.regularizer", "model.params.deterministic_only",
            "model.params.prior_mode", "model.params.encoder_conditioning",
            "model.params.latent_residual_mode",
            "model.params.freeze_deterministic_backbone",
            "model.params.deterministic_backbone_checkpoint",
            "model.params.deterministic_backbone_weights_sha256",
            "loss.regularizer_weight", "loss.conditional_mean_weight",
            "training.best_monitor", "training.checkpoint_dir",
            "evaluation.tensorboard.log_dir",
        ]
        audit = static_audit_conditional_wae_config(config)
        config_path = root / "configs" / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(config_path), "gpu": int(gpu), "audit": audit,
            "checkpoint_dir": config["training"]["checkpoint_dir"],
            "tensorboard_log_dir": config["evaluation"]["tensorboard"]["log_dir"],
        }

    plan = {
        "kind": "mk_residual_wae_suite", "version": 1,
        "root": str(root), "arms": arms, "arm_order": list(ARM_ORDER),
        "deterministic_config": str(source_config_path),
        "deterministic_checkpoint": str(identity.resolved_dir),
        "deterministic_weights_sha256": identity.weights_sha256,
        "manifest": str(manifest_path), "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours), "cpu_threads_per_arm": int(cpu_threads),
        "latent_dim": int(latent_dim), "seed": seed,
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_MK_RESIDUAL_WAE_SUITE_ROOT.txt"
    temporary = pointer.with_name(pointer.name + ".tmp")
    temporary.write_text(str(root) + "\n")
    temporary.replace(pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic-config", required=True)
    parser.add_argument("--deterministic-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2,3,5")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    args = parser.parse_args()
    result = prepare_mk_residual_wae_suite(
        deterministic_config=args.deterministic_config,
        deterministic_checkpoint=args.deterministic_checkpoint,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root,
        hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        cpu_threads=args.cpu_threads,
        latent_dim=args.latent_dim,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
