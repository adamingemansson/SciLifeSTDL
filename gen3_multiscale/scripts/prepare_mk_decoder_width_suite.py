#!/usr/bin/env python3
"""Prepare a paired-seed 512-versus-1024 decoder-width screen."""
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


ARM_ORDER = (
    "mk_pg_width1024_seed1", "mk_pg_width1024_seed2",
    "mk_pg_width512_seed1", "mk_pg_width512_seed2",
)
ARM_DESIGN = {
    "mk_pg_width1024_seed1": (1024, 1),
    "mk_pg_width1024_seed2": (1024, 2),
    "mk_pg_width512_seed1": (512, 1),
    "mk_pg_width512_seed2": (512, 2),
}


def prepare_mk_decoder_width_suite(
    *, source_config: str, output_root: str, hours: float = 8.0,
    gpus: tuple[int, ...] = (0, 2, 3, 5), cpu_threads: int = 8,
) -> dict:
    if hours <= 0 or cpu_threads < 1:
        raise ValueError("hours and cpu_threads must be positive")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("gpus must contain exactly four distinct ids")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"{root} already exists; suite roots are immutable")
    source_path = Path(source_config).expanduser().resolve()
    source = yaml.safe_load(source_path.read_text())
    model = source.get("model") or {}
    params = model.get("params") or {}
    if model.get("arm") != "mk_wb_parallel_gated":
        raise ValueError("source_config must be mk_wb_parallel_gated")
    if not bool(params.get("deterministic_only", False)):
        raise ValueError("source_config must be deterministic")
    if params.get("structured_composition") != "parallel_gated":
        raise ValueError("source_config must use parallel_gated composition")

    manifest_path = Path(source["data"]["gen3_manifest_path"]).resolve()
    manifest = load_dataset_manifest(manifest_path)
    if len(manifest.get("validation_sample_ids", [])) != 14:
        raise ValueError("expanded MK cohort must contain exactly 14 validation slides")
    panel_path = Path(source["evaluation"]["train_gene_panel_artifact"]).resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, manifest)
    required = {
        "train_log1p_variance_top50", "train_log1p_variance_top200",
        "train_within_slide_variance_top50", "train_within_slide_variance_top200",
    }
    if not required.issubset(panel_artifact["panels"]):
        raise ValueError("expanded global and within-slide panels are required")
    cache_root = Path(source["data"]["gen3_uni2_spot_feature_cache_dir"]).resolve()
    require_uni2_spot_cache_coverage(cache_root, manifest["samples"])
    if not source["data"].get("centered_gene_structure_path"):
        raise ValueError("source_config is missing centered gene structure")

    root.mkdir(parents=True)
    for directory in ("configs", "checkpoints", "logs", "tensorboard", "control"):
        (root / directory).mkdir()
    arms = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        width, seed = ARM_DESIGN[arm]
        config = copy.deepcopy(source)
        config["experiment_name"] = f"mk_decoder_width_{arm}_v1"
        config["model"]["arm"] = arm
        config["model"]["params"]["autoencoder_hidden_dim"] = width
        config["training"].update({
            "seed": seed, "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000, "max_wall_clock_hours": float(hours),
            "device": "cuda", "cpu_threads": int(cpu_threads),
        })
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(
            root, manifest,
        )
        config["documented_divergences"] = [
            "experiment_name", "model.arm", "model.params.autoencoder_hidden_dim",
            "training.seed", "training.checkpoint_dir", "training.max_wall_clock_hours",
            "evaluation.tensorboard.log_dir",
        ]
        audit = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(path), "gpu": int(gpu), "width": width, "seed": seed,
            "checkpoint_dir": config["training"]["checkpoint_dir"], "audit": audit,
        }
    plan = {
        "kind": "mk_decoder_width_suite", "version": 1, "root": str(root),
        "source_config": str(source_path), "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path), "arm_order": list(ARM_ORDER),
        "arms": arms, "hours_per_arm": float(hours),
        "cpu_threads_per_arm": int(cpu_threads),
        "controlled_factors": ["model.params.autoencoder_hidden_dim", "training.seed"],
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_MK_DECODER_WIDTH_SUITE_ROOT.txt"
    temporary = pointer.with_name(pointer.name + ".tmp")
    temporary.write_text(str(root) + "\n")
    temporary.replace(pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2,3,5")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    plan = prepare_mk_decoder_width_suite(
        source_config=args.source_config, output_root=args.output_root,
        hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
