#!/usr/bin/env python3
"""Prepare the controlled full-encoder/300-gene residual WAE-MMD screen."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen4.uni2_spot_cache import require_uni2_spot_cache_coverage
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    tensorboard_block, whole_slide_validation_block,
)
from gen3_multiscale.training import checkpoint as checkpoint_module


ARM_ORDER = (
    "mk_swae_hvg300_standard",
    "mk_swae_hvg300_conditional",
    "mk_swae_within300_standard",
    "mk_swae_within300_conditional",
)
DESIGNS = {
    "mk_swae_hvg300_standard": ("ranking", "pooled_variance", "standard"),
    "mk_swae_hvg300_conditional": ("ranking", "pooled_variance", "conditional"),
    "mk_swae_within300_standard": (
        "within_slide_ranking", "within_slide_variance", "standard",
    ),
    "mk_swae_within300_conditional": (
        "within_slide_ranking", "within_slide_variance", "conditional",
    ),
}


def _write_panel(path: Path, genes: list[str], *, source: str, artifact: dict) -> None:
    payload = {
        "version": 1,
        "kind": "mk_train_only_specialist_gene_panel",
        "source": source,
        "source_artifact_sha256": artifact["artifact_sha256"],
        "genes": genes,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_suite(*, deterministic_config: str, deterministic_checkpoint: str,
                  manifest: str, train_gene_panels: str, output_root: str,
                  hours: float = 8.0, gpus=(0, 2, 3, 5), cpu_threads: int = 8,
                  latent_dim: int = 64) -> dict:
    if hours <= 0 or cpu_threads < 1 or latent_dim < 1:
        raise ValueError("hours, cpu_threads, and latent_dim must be positive")
    gpus = tuple(int(gpu) for gpu in gpus)
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("exactly four distinct GPUs are required")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"{root} already exists; suite roots are immutable")

    source_path = Path(deterministic_config).expanduser().resolve()
    source = yaml.safe_load(source_path.read_text())
    source_model = source["model"]
    source_params = source_model["params"]
    if source_model.get("arm") != "mk_wb_parallel_gated":
        raise ValueError("source must be mk_wb_parallel_gated")
    if not source_params.get("deterministic_only") or source_params.get(
        "structured_composition"
    ) != "parallel_gated":
        raise ValueError("source is not the deterministic parallel-gated architecture")

    manifest_path = Path(manifest).expanduser().resolve()
    dataset = load_dataset_manifest(manifest_path)
    if Path(source["data"]["gen3_manifest_path"]).resolve() != manifest_path:
        raise ValueError("source and specialist suite use different manifests")
    if len(dataset.get("validation_sample_ids", [])) != 14:
        raise ValueError("expanded MK cohort must have 14 held-out slides")
    genes = [str(gene) for gene in dataset["gene_panel"]]
    panel_path = Path(train_gene_panels).expanduser().resolve()
    artifact = load_train_derived_gene_panels(panel_path, dataset)
    ranked = [str(row["gene"]) for row in artifact["ranking"][:300]]
    within = [str(row["gene"]) for row in artifact["within_slide_ranking"][:300]]
    if len(ranked) != 300 or len(within) != 300:
        raise ValueError("both train-only rankings must provide 300 genes")

    cache_root = Path(source["data"]["gen3_uni2_spot_feature_cache_dir"]).resolve()
    require_uni2_spot_cache_coverage(cache_root, dataset["samples"])
    identity = checkpoint_module.resolve_checkpoint_identity(deterministic_checkpoint)
    if identity.weights_sha256 is None:
        raise ValueError("source checkpoint has no trainable weights")
    checkpoint_module.verify_gene_names(identity.resolved_dir, genes)
    bundled = json.loads((identity.resolved_dir / "model_config.json").read_text())
    if bundled.get("model") != source_model:
        raise ValueError("source config does not exactly match checkpoint model config")

    root.mkdir(parents=True)
    for name in ("configs", "checkpoints", "logs", "tensorboard", "control", "panels"):
        (root / name).mkdir()
    pooled_panel = root / "panels" / "train_pooled_variance_top300.json"
    within_panel = root / "panels" / "train_within_slide_variance_top300.json"
    _write_panel(pooled_panel, ranked, source="pooled_variance", artifact=artifact)
    _write_panel(within_panel, within, source="within_slide_variance", artifact=artifact)

    seed = int(source["training"].get("seed", 0))
    arms = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        ranking_key, panel_source, prior_mode = DESIGNS[arm]
        selected = ranked if ranking_key == "ranking" else within
        selected_path = pooled_panel if ranking_key == "ranking" else within_panel
        config = copy.deepcopy(source)
        params = copy.deepcopy(source_params)
        params.update({
            "deterministic_only": False,
            "prior_mode": prior_mode,
            "encoder_conditioning": "none",
            "latent_dim": int(latent_dim),
            "latent_residual_mode": "free",
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
            "specialist_gene_names": selected,
            "specialist_panel_source": panel_source,
            "specialist_encoder_scope": "full_gene_panel",
        })
        config["experiment_name"] = f"mk_specialist_wae_{arm}_v1"
        config["model"] = {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st",
            "regularizer": "mmd", "include_observed_gex": False,
            "image_mode": "full_visible", "params": params,
        }
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.1,
            "conditional_mean_weight": 0.0,
            "specialist_prior_center_weight": 1.0,
            "local_gradient_weight": 0.0,
            "wide_gradient_weight": 0.0,
        }
        evaluation = config.setdefault("evaluation", {})
        evaluation["train_gene_panel_artifact"] = str(panel_path)
        evaluation["gene_panels"] = {
            ("specialist_hvg300" if panel_source == "pooled_variance"
             else "specialist_within300"): str(selected_path),
        }
        evaluation["tensorboard"] = tensorboard_block(root, arm)
        evaluation["whole_slide_validation"] = whole_slide_validation_block(root, dataset)
        # Never collide with an artifact from a previous cohort/seed.
        evaluation["whole_slide_validation"]["reference_projection_path"] = str(
            root / "reference_gex_projection"
        )
        evaluation["structured_field_metrics"] = {
            "enabled": True, "all_split_slides": True, "chunk_size": 2048,
            "local_k": 6, "wide_k": 18,
            "nontrivial_gradient_threshold_training_sd": 0.25,
            "minimum_noise_ceiling": 0.05,
        }
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            "device": "cuda", "cpu_threads": int(cpu_threads), "seed": seed,
            "lr": 1e-4, "gradient_accumulation_steps": 1,
            "best_monitor": "total", "optimizer": {"weight_decay": 0.01},
        })
        config["documented_divergences"] = [
            "model.arm", "model.regularizer", "model.params.deterministic_only",
            "model.params.prior_mode", "model.params.latent_residual_mode",
            "model.params.specialist_gene_names", "model.params.specialist_panel_source",
            "model.params.specialist_encoder_scope",
            "model.params.freeze_deterministic_backbone",
            "model.params.deterministic_backbone_checkpoint",
            "model.params.deterministic_backbone_weights_sha256",
            "loss.specialist_prior_center_weight", "training.best_monitor",
            "training.checkpoint_dir", "evaluation.gene_panels",
            "evaluation.tensorboard.log_dir",
        ]
        audit = static_audit_conditional_wae_config(config)
        config_path = root / "configs" / f"{arm}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(config_path), "gpu": int(gpu), "audit": audit,
            "checkpoint_dir": config["training"]["checkpoint_dir"],
            "specialist_panel": str(selected_path),
        }

    plan = {
        "kind": "mk_specialist_wae_suite", "version": 1,
        "root": str(root), "arm_order": list(ARM_ORDER), "arms": arms,
        "deterministic_config": str(source_path),
        "deterministic_checkpoint": str(identity.resolved_dir),
        "deterministic_weights_sha256": identity.weights_sha256,
        "manifest": str(manifest_path), "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours), "cpu_threads_per_arm": int(cpu_threads),
        "latent_dim": int(latent_dim), "seed": seed,
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_MK_SPECIALIST_WAE_SUITE_ROOT.txt"
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
    print(json.dumps(prepare_suite(
        deterministic_config=args.deterministic_config,
        deterministic_checkpoint=args.deterministic_checkpoint,
        manifest=args.manifest, train_gene_panels=args.train_gene_panels,
        output_root=args.output_root, hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",")),
        cpu_threads=args.cpu_threads, latent_dim=args.latent_dim,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
