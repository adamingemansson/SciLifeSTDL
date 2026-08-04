#!/usr/bin/env python3
"""Write four matched MK conditional-flow configs; never starts training."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from gen3_multiscale.conditional_flow.contract import (
    ARM_SPECS,
    static_audit_conditional_flow_config,
)
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    _absolutize_existing_source_paths,
    _source_repository_root,
)


def prepare_conditional_flow_suite(
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
        "n_flow_blocks": 2,
        "n_ode_steps": 20,
        "n_inference_samples": 8,
        "ot_epsilon": 0.1,
        "ot_sinkhorn_iters": 20,
        "dropout": 0.1,
    })
    written = {}
    for arm, spec in ARM_SPECS.items():
        config = copy.deepcopy(base)
        config["experiment_name"] = f"conditional_flow_{arm}_v1"
        config["documented_divergences"] = [
            "model.arm", "model.task", "model.coupling",
            "model.include_observed_gex", "training.checkpoint_dir",
        ]
        config["model"] = {
            "arm": arm,
            "kind": "conditional_latent_flow",
            "task": spec.task,
            "coupling": spec.coupling,
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
            # Match the WAE suite's 0.1 prior-regularizer weight so the
            # reconstruction/conditioning objectives retain the same scale.
            "flow_weight": 0.1,
            "conditional_mean_weight": 1.0,
        }
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            "batch_size": 1,
        })
        audit = static_audit_conditional_flow_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[arm] = {"config": str(path), "audit": audit}
    plan = {
        "kind": "conditional_flow_supervisor_suite",
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours),
        "arms": written,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_CONDITIONAL_FLOW_SUITE_ROOT.txt"
    temporary = pointer.with_name(f"{pointer.name}.tmp.{os.getpid()}")
    temporary.write_text(str(root.resolve()) + "\n")
    os.replace(temporary, pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(prepare_conditional_flow_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root,
        hours=args.hours,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
