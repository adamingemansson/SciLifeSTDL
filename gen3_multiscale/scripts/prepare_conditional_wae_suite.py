#!/usr/bin/env python3
"""Write four matched resolved configs; never starts training."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import (
    ARM_SPECS,
    static_audit_conditional_wae_config,
)
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels


def _source_repository_root(comparison_config: str | Path) -> Path:
    """Locate the repository against which relative resolved paths were valid."""
    path = Path(comparison_config).resolve()
    for candidate in path.parents:
        if (candidate / "gen3_multiscale").is_dir():
            return candidate
    raise ValueError(
        f"cannot locate source repository root above comparison config {path}"
    )


def _absolutize_existing_source_paths(value, source_root: Path):
    """Preserve source-config asset semantics when writing into another worktree.

    Only strings whose source-root-relative target actually exists are changed;
    ordinary enum/name/hash strings remain untouched.
    """
    if isinstance(value, dict):
        return {
            key: _absolutize_existing_source_paths(item, source_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_absolutize_existing_source_paths(item, source_root) for item in value]
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            candidate = source_root / path
            if candidate.exists():
                return str(candidate.resolve())
    return value


def whole_slide_validation_block(root: Path, dataset_manifest: dict, *, every_n_evals: int = 5) -> dict:
    """Shared `evaluation.whole_slide_validation` config block for every
    WAE-GAN ablation suite-prep script: full-slide (every spot, not just
    masked query rows) diagnostic validation, `max_slides` capped to one
    representative slide per organ present in validation (the trainer's
    `_select_whole_slide_sample_ids` round-robins across organs so every
    organ still gets covered even at this smaller count -- logging every
    single validation slide was needlessly expensive: same organ's slides
    are largely redundant for diagnostic purposes, and it multiplies
    TensorBoard image/embedding disk usage for no real added coverage),
    and ONE reference GEX-PCA/cluster basis path shared across every
    arm/suite written under `root.parent` -- so PC1/PC2/PC3 and cluster
    colors mean the same thing when comparing arms across different
    suites too, not just within one suite. `reference_projection_path` is
    load-or-build (see conditional_wae.reference_projection.
    ensure_reference_gex_projection): the first arm to start training
    builds it, every later arm/suite just verifies its identity and
    reuses it."""
    validation_ids = dataset_manifest["validation_sample_ids"]
    if not validation_ids:
        raise ValueError("dataset manifest has zero validation_sample_ids")
    samples = dataset_manifest.get("samples") or {}
    n_organs = len({str(samples[sid]["organ"]) for sid in validation_ids if sid in samples and "organ" in samples[sid]})
    max_slides = min(len(validation_ids), n_organs) if n_organs else len(validation_ids)
    return {
        "enabled": True,
        "every_n_evals": int(every_n_evals),
        "max_slides": max_slides,
        "chunk_size": 2048,
        "reference_projection_path": str(root.parent / "mk_wae_shared_reference_gex_projection"),
    }


def tensorboard_block(root: Path, arm: str) -> dict:
    """Shared `evaluation.tensorboard` config block for every WAE-GAN
    ablation suite-prep script, trimmed to a minimal-but-followable set:
    train/* scalars are OFF (too noisy to be useful step-to-step; the
    validation trio below is what actually answers "is this arm working"),
    validation/whole_slide scalars stay to just total/rmse/pcc_loss (no
    conditional_mean_rmse or the old per-panel/AUC breakdown), and the
    embedding/Projector snapshot -- the single biggest disk contributor
    per snapshot -- is capped far below its old 5000-point default while
    still logging often enough (every 5th validation, matching whole-slide
    cadence) to see a real trend."""
    return {
        "enabled": True,
        "log_dir": str(root / "tensorboard" / arm),
        "log_train_scalars": False,
        "snapshot_every_n_evals": 5,
        "embedding_max_points": 800,
        "embedding_max_points_per_item": 64,
        "thumbnail_max_points": 100,
        "thumbnail_size": 32,
        "max_spatial_samples": 4,
        "spatial_gene_count": 2,
    }


def prepare_conditional_wae_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    output_root: str, hours: float = 8.0,
) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; suite roots are immutable")
    source_root = _source_repository_root(comparison_config)
    base = yaml.safe_load(Path(comparison_config).read_text())
    base = _absolutize_existing_source_paths(base, source_root)
    if str((base.get("model") or {}).get("architecture", "")) != "1":
        raise ValueError("--comparison-config must be a resolved Gen3 Architecture 1 config")
    manifest_path = Path(manifest).resolve()
    dataset_manifest = load_dataset_manifest(manifest_path)
    panel_path = Path(train_gene_panels).resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, dataset_manifest)
    required_panels = {"train_log1p_variance_top50", "train_log1p_variance_top200"}
    missing = sorted(required_panels - set(panel_artifact["panels"]))
    if missing:
        raise ValueError(f"train-derived panel artifact is missing {missing}")

    root.mkdir(parents=True)
    for name in ("configs", "checkpoints", "logs", "tensorboard"):
        (root / name).mkdir()
    written = {}
    architecture1_params = dict((base.get("model") or {}).get("params") or {})
    shared_params = {
        key: architecture1_params[key]
        for key in (
            "image_feature_dim", "hidden_dim", "n_heads", "n_blocks",
            "dense_threshold", "sparse_k",
        )
    }
    shared_params.update({
        "gex_feature_dim": int((base.get("data") or {}).get("gex_feature_dim", 256)),
        "latent_dim": 256,
        "autoencoder_hidden_dim": 1024,
        "discriminator_hidden_dim": 256,
        "n_inference_samples": 8,
        "dropout": 0.1,
    })
    for arm, spec in ARM_SPECS.items():
        config = copy.deepcopy(base)
        config["experiment_name"] = f"conditional_wae_{arm}_v1"
        config["documented_divergences"] = [
            "model.arm", "model.task", "model.regularizer",
            "model.include_observed_gex", "training.checkpoint_dir",
        ]
        config["model"] = {
            "arm": arm,
            "kind": "conditional_wae",
            "task": spec.task,
            "regularizer": spec.regularizer,
            "include_observed_gex": spec.include_observed_gex,
            "image_mode": "full_visible",
            "params": copy.deepcopy(shared_params),
        }
        config["data"]["gen3_manifest_path"] = str(manifest_path)
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["evaluation"]["tensorboard"] = {
            "enabled": True,
            "log_dir": str(root / "tensorboard" / arm),
            # Scalars are written at every existing log/evaluation boundary.
            # Expensive Projector/maps reuse every fifth validation pass.
            "snapshot_every_n_evals": 5,
            "embedding_max_points": 5000,
            "embedding_max_points_per_item": 64,
            "thumbnail_max_points": 512,
            "thumbnail_size": 48,
            "max_spatial_samples": 4,
            "spatial_gene_count": 2,
        }
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.1,
            "conditional_mean_weight": 1.0,
        }
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            "batch_size": 1,
        })
        report = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[arm] = {"config": str(path), "audit": report}
    plan = {
        "kind": "conditional_wae_supervisor_suite",
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours),
        "arms": written,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_CONDITIONAL_WAE_SUITE_ROOT.txt"
    tmp = pointer.with_name(f"{pointer.name}.tmp.{os.getpid()}")
    tmp.write_text(str(root.resolve()) + "\n")
    os.replace(tmp, pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()
    plan = prepare_conditional_wae_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root,
        hours=args.hours,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
