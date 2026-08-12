#!/usr/bin/env python3
"""Prepare four matched MK H&E-to-GEX architecture configs; never train.

The four cells are intentionally narrow:

* mk_local_wae_mmd: historical standard-prior WAE-MMD, local UNI2 only.
* mk_spatial_deterministic: UNI2 + coordinates + spatial transformer + head.
* mk_local_conditional_wae_mmd: local UNI2 with learned p(z|image).
* mk_spatial_conditional_wae_mmd: spatial context with learned p(z|context).

Every WAE uses the same plain MLP target-GEX encoder with two-layer FiLM.
All arms use the same data, masks, image cache, supervised RMSE/PCC loss,
optimizer and seed. Spatial refinement is disabled: “spatial” means only the
explicit coordinate/relative-geometry transformer being tested here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    _absolutize_existing_source_paths,
    _source_repository_root,
    resolve_data_locations,
    tensorboard_block,
    whole_slide_validation_block,
)


ARM_ORDER = (
    "mk_local_wae_mmd",
    "mk_spatial_deterministic",
    "mk_local_conditional_wae_mmd",
    "mk_spatial_conditional_wae_mmd",
)

ARM_DESIGN = {
    "mk_local_wae_mmd": {
        "conditioner_mode": "local", "prior_mode": "standard", "deterministic_only": False,
    },
    "mk_spatial_deterministic": {
        "conditioner_mode": "spatial", "prior_mode": "none", "deterministic_only": True,
    },
    "mk_local_conditional_wae_mmd": {
        "conditioner_mode": "local", "prior_mode": "conditional", "deterministic_only": False,
    },
    "mk_spatial_conditional_wae_mmd": {
        "conditioner_mode": "spatial", "prior_mode": "conditional", "deterministic_only": False,
    },
}


def prepare_mk_architecture_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    output_root: str, uni2_pinned_revision: str,
    uni2_spot_feature_cache_dir: str,
    hours: float = 8.0, gpus: tuple[int, ...] = (0, 2, 3, 5),
    cpu_threads: int = 12, latent_dim: int = 64,
    hest_data_dir: str | None = None, hest_cache_dir: str | None = None,
) -> dict:
    if hours <= 0 or latent_dim < 1 or cpu_threads < 1:
        raise ValueError("hours, latent_dim and cpu_threads must be positive")
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("gpus must contain four distinct GPU ids")
    if not uni2_pinned_revision:
        raise ValueError("uni2_pinned_revision must be a pinned UNI2 commit")
    uni2_input = Path(uni2_spot_feature_cache_dir).expanduser().resolve()
    uni2_cache = uni2_input.parent if uni2_input.name == "uni2_gen3_spot_cache" else uni2_input
    if not (uni2_cache / "uni2_gen3_spot_cache").is_dir():
        raise FileNotFoundError(
            "UNI2 cache root must contain uni2_gen3_spot_cache/: "
            f"{uni2_cache}"
        )

    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"{root} already exists; suite roots are immutable")
    source_root = _source_repository_root(comparison_config)
    base = yaml.safe_load(Path(comparison_config).read_text())
    base = _absolutize_existing_source_paths(base, source_root)
    base = resolve_data_locations(base, source_root, overrides={
        "hest_data_dir": hest_data_dir, "hest_cache_dir": hest_cache_dir,
    })
    if str((base.get("model") or {}).get("architecture", "")) != "1":
        raise ValueError("comparison_config must be a resolved Gen3 Architecture 1 config")

    manifest_path = Path(manifest).expanduser().resolve()
    dataset_manifest = load_dataset_manifest(manifest_path)
    panel_path = Path(train_gene_panels).expanduser().resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, dataset_manifest)
    required = {"train_log1p_variance_top50", "train_log1p_variance_top200"}
    if not required.issubset(panel_artifact["panels"]):
        raise ValueError(f"train gene panels are missing {sorted(required - set(panel_artifact['panels']))}")

    root.mkdir(parents=True)
    for directory in ("configs", "checkpoints", "logs", "tensorboard"):
        (root / directory).mkdir()

    architecture1 = dict((base.get("model") or {}).get("params") or {})
    shared = {
        key: architecture1[key] for key in (
            "image_feature_dim", "hidden_dim", "n_heads", "n_blocks",
            "dense_threshold", "sparse_k",
        )
    }
    shared.update({
        "gex_feature_dim": int((base.get("data") or {}).get("gex_feature_dim", 256)),
        "latent_dim": int(latent_dim),
        "autoencoder_hidden_dim": 1024,
        "discriminator_hidden_dim": 256,
        "n_inference_samples": 8,
        "dropout": 0.1,
        "gene_encoder_source": "linear",
        "encoder_conditioning": "film",
        "film_layers": ["first", "second"],
        "film_shared_generator": False,
        "z_noise_std": 0.0,
        "n_refinement_steps": 0,
        "likelihood": "gaussian_mse",
        "conditional_prior_hidden_dim": 256,
        "conditional_prior_context_weight": 1.0,
        "conditional_prior_anchor_weight": 0.1,
    })
    seed = int((base.get("training") or {}).get("seed", 0))
    arms = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        design = ARM_DESIGN[arm]
        params = copy.deepcopy(shared)
        params.update(design)
        regularizer = "none" if design["deterministic_only"] else "mmd"
        if design["deterministic_only"]:
            params["encoder_conditioning"] = "none"
            params.pop("film_layers")
            params.pop("film_shared_generator")

        config = copy.deepcopy(base)
        config["experiment_name"] = f"mk_four_architecture_{arm}_v1"
        config["model"] = {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st",
            "regularizer": regularizer, "include_observed_gex": False,
            "image_mode": "full_visible", "params": params,
        }
        config["documented_divergences"] = [
            "model.arm", "model.regularizer",
            "model.params.conditioner_mode", "model.params.prior_mode",
            "model.params.deterministic_only", "model.params.encoder_conditioning",
            "model.params.film_layers", "model.params.film_shared_generator",
            "training.checkpoint_dir", "evaluation.tensorboard.log_dir",
        ]
        config["data"].update({
            "gen3_manifest_path": str(manifest_path),
            "image_encoder": "uni2",
            "uni2_pinned_revision": str(uni2_pinned_revision),
            "gen3_uni2_spot_feature_cache_dir": str(uni2_cache),
            "retain_patches_in_memory": False,
        })
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.1,
            "conditional_mean_weight": 1.0,
        }
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(
            root, dataset_manifest,
        )
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            # The launcher pins the physical device with CUDA_VISIBLE_DEVICES;
            # inside that process the selected GPU is always logical cuda:0.
            "device": "cuda",
            "cpu_threads": int(cpu_threads),
            "seed": seed,
            "lr": 1e-4,
            "gradient_accumulation_steps": 1,
            "optimizer": {"weight_decay": 0.01},
        })
        audit = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(path), "gpu": int(gpu), "audit": audit,
            "tensorboard_log_dir": config["evaluation"]["tensorboard"]["log_dir"],
            "checkpoint_dir": config["training"]["checkpoint_dir"],
        }

    plan = {
        "kind": "mk_four_architecture_suite", "version": 1,
        "arms": arms, "arm_order": list(ARM_ORDER),
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path), "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours), "cpu_threads_per_arm": int(cpu_threads),
        "latent_dim": int(latent_dim),
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_MK_FOUR_ARCHITECTURE_SUITE_ROOT.txt"
    temporary = pointer.with_name(f"{pointer.name}.tmp")
    temporary.write_text(str(root) + "\n")
    temporary.replace(pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--uni2-pinned-revision", required=True)
    parser.add_argument("--uni2-spot-feature-cache-dir", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2,3,5")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--hest-cache-dir")
    args = parser.parse_args()
    gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    plan = prepare_mk_architecture_suite(
        comparison_config=args.comparison_config, manifest=args.manifest,
        train_gene_panels=args.train_gene_panels, output_root=args.output_root,
        uni2_pinned_revision=args.uni2_pinned_revision,
        uni2_spot_feature_cache_dir=args.uni2_spot_feature_cache_dir,
        hours=args.hours, gpus=gpus, cpu_threads=args.cpu_threads,
        latent_dim=args.latent_dim, hest_data_dir=args.hest_data_dir,
        hest_cache_dir=args.hest_cache_dir,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
