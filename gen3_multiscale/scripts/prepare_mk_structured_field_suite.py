#!/usr/bin/env python3
"""Prepare the four deterministic MK structured-field ablations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.structured_field import (
    load_centered_gene_structure_artifact,
)
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen4.uni2_spot_cache import require_uni2_spot_cache_coverage
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    _absolutize_existing_source_paths,
    _source_repository_root,
    resolve_data_locations,
    tensorboard_block,
    whole_slide_validation_block,
)


ARM_ORDER = (
    "mk_field_within",
    "mk_field_between",
    "mk_field_gradient",
    "mk_field_combined",
)
ARM_DESIGN = {
    "mk_field_within": {
        "conditioner_mode": "local", "use_centered_gene_structure": True,
        "n_refinement_steps": 0, "local_gradient_weight": 0.0,
        "wide_gradient_weight": 0.0,
    },
    "mk_field_between": {
        "conditioner_mode": "spatial", "use_centered_gene_structure": False,
        "n_refinement_steps": 1, "local_gradient_weight": 0.0,
        "wide_gradient_weight": 0.0,
    },
    "mk_field_gradient": {
        "conditioner_mode": "local", "use_centered_gene_structure": False,
        "n_refinement_steps": 0, "local_gradient_weight": 0.025,
        "wide_gradient_weight": 0.025,
    },
    "mk_field_combined": {
        "conditioner_mode": "spatial", "use_centered_gene_structure": True,
        "n_refinement_steps": 1, "local_gradient_weight": 0.025,
        "wide_gradient_weight": 0.025,
    },
}
COMPOSITION_ARM_ORDER = (
    "mk_wb_serial",
    "mk_bw_serial",
    "mk_wbw_sandwich",
    "mk_wb_parallel_gated",
)
COMPOSITION_ARM_DESIGN = {
    arm: {
        "conditioner_mode": "spatial",
        "use_centered_gene_structure": True,
        "n_refinement_steps": 1,
        "local_gradient_weight": 0.0,
        "wide_gradient_weight": 0.0,
        "structured_composition": composition,
    }
    for arm, composition in zip(COMPOSITION_ARM_ORDER, (
        "within_then_between",
        "between_then_within",
        "within_between_within",
        "parallel_gated",
    ))
}
FAMILY_SPECS = {
    "deterministic": {
        "arm_order": ARM_ORDER, "regularizer": "none",
        "prior_mode": "none", "deterministic_only": True,
    },
    "standard_wae": {
        "arm_order": ("wae_within", "wae_between", "wae_gradient", "wae_combined"),
        "regularizer": "mmd", "prior_mode": "standard", "deterministic_only": False,
    },
    "conditional_wae": {
        "arm_order": ("cwae_within", "cwae_between", "cwae_gradient", "cwae_combined"),
        "regularizer": "mmd", "prior_mode": "conditional", "deterministic_only": False,
    },
    "deterministic_composition": {
        "arm_order": COMPOSITION_ARM_ORDER, "regularizer": "none",
        "prior_mode": "none", "deterministic_only": True,
    },
}


def _design_for_arm(arm: str) -> dict:
    if arm in COMPOSITION_ARM_DESIGN:
        return COMPOSITION_ARM_DESIGN[arm]
    suffix = arm.removeprefix("mk_field_").removeprefix("cwae_").removeprefix("wae_")
    source = f"mk_field_{suffix}"
    if source not in ARM_DESIGN:
        raise ValueError(f"unknown structured-field suffix for arm {arm!r}")
    return ARM_DESIGN[source]


def prepare_mk_structured_field_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    centered_gene_structure: str, output_root: str,
    uni2_pinned_revision: str, uni2_spot_feature_cache_dir: str,
    hours: float = 8.0, gpus: tuple[int, ...] = (0, 2, 3, 5),
    cpu_threads: int = 12, hest_data_dir: str | None = None,
    hest_cache_dir: str | None = None, family: str = "deterministic",
    latent_dim: int = 64,
) -> dict:
    if hours <= 0 or cpu_threads < 1 or latent_dim < 1:
        raise ValueError("hours, cpu_threads and latent_dim must be positive")
    if family not in FAMILY_SPECS:
        raise ValueError(f"family must be one of {sorted(FAMILY_SPECS)}")
    family_spec = FAMILY_SPECS[family]
    arm_order = family_spec["arm_order"]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("gpus must contain four distinct ids")
    if not uni2_pinned_revision:
        raise ValueError("uni2_pinned_revision must be pinned")
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
    require_uni2_spot_cache_coverage(
        uni2_cache, dataset_manifest["samples"],
    )
    panel_path = Path(train_gene_panels).expanduser().resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, dataset_manifest)
    required_panels = {"train_log1p_variance_top50", "train_log1p_variance_top200"}
    if not required_panels.issubset(panel_artifact["panels"]):
        raise ValueError("training-derived HVG-50/HVG-200 panels are required")

    structure_path = Path(centered_gene_structure).expanduser().resolve()
    structure = load_centered_gene_structure_artifact(
        structure_path, list(dataset_manifest["gene_panel"]),
    )
    if structure.metadata["train_sample_ids"] != sorted(
        str(value) for value in dataset_manifest["train_sample_ids"]
    ):
        raise ValueError("centered gene structure was fit against a different training split")

    root.mkdir(parents=True)
    for directory in ("configs", "checkpoints", "logs", "tensorboard"):
        (root / directory).mkdir()

    arch1 = dict((base.get("model") or {}).get("params") or {})
    shared = {key: arch1[key] for key in (
        "image_feature_dim", "hidden_dim", "n_heads", "n_blocks",
        "dense_threshold", "sparse_k",
    )}
    shared.update({
        "gex_feature_dim": int((base.get("data") or {}).get("gex_feature_dim", 256)),
        "autoencoder_hidden_dim": 1024,
        "n_inference_samples": 1,
        "dropout": 0.1,
        "deterministic_only": True,
        "prior_mode": "none",
        "encoder_conditioning": "none",
        "gene_structure_hidden_dim": 64,
        "refinement_k_neighbors": 6,
        "refinement_hidden_dim": 256,
        "refinement_gex_feature_dim": 256,
        "local_gradient_k": 6,
        "wide_gradient_k": 18,
        "likelihood": "gaussian_mse",
        "latent_dim": int(latent_dim),
        "discriminator_hidden_dim": 256,
        "gene_encoder_source": "linear",
        "conditional_prior_hidden_dim": 256,
        "conditional_prior_context_weight": 1.0,
        "conditional_prior_anchor_weight": 0.1,
        "z_noise_std": 0.0,
        "structured_composition": "within_then_between",
    })
    seed = int((base.get("training") or {}).get("seed", 0))
    arms = {}
    for arm, gpu in zip(arm_order, gpus):
        design = _design_for_arm(arm)
        params = copy.deepcopy(shared)
        params.update({
            key: value for key, value in design.items()
            if key not in {"local_gradient_weight", "wide_gradient_weight"}
        })
        config = copy.deepcopy(base)
        params.update({
            "deterministic_only": bool(family_spec["deterministic_only"]),
            "prior_mode": str(family_spec["prior_mode"]),
            "n_inference_samples": 1 if family_spec["deterministic_only"] else 8,
            "encoder_conditioning": "none" if family_spec["deterministic_only"] else "film",
        })
        if not family_spec["deterministic_only"]:
            params.update({
                "film_layers": ["first", "second"],
                "film_shared_generator": False,
            })
        config["experiment_name"] = f"mk_structured_field_{family}_{arm}_v1"
        config["model"] = {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st",
            "regularizer": family_spec["regularizer"], "include_observed_gex": False,
            "image_mode": "full_visible", "params": params,
        }
        config["data"].update({
            "gen3_manifest_path": str(manifest_path),
            "image_encoder": "uni2",
            "uni2_pinned_revision": str(uni2_pinned_revision),
            "gen3_uni2_spot_feature_cache_dir": str(uni2_cache),
            "centered_gene_structure_path": str(structure_path),
            "centered_gene_structure_basis_sha256": structure.metadata["basis_sha256"],
            "retain_patches_in_memory": False,
        })
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.0 if family_spec["deterministic_only"] else 0.1,
            "conditional_mean_weight": 1.0,
            "local_gradient_weight": design["local_gradient_weight"],
            "wide_gradient_weight": design["wide_gradient_weight"],
        }
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(
            root, dataset_manifest,
        )
        # Final evaluation, unlike the bounded TensorBoard visualisation
        # block above, uses EVERY held-out slide and predicts every tissue
        # spot exactly once.  These metrics test the specific coexpression,
        # between-spot and gradient claims varied by this four-arm screen.
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
            "optimizer": {"weight_decay": 0.01},
        })
        config["documented_divergences"] = [
            "model.arm", "model.regularizer", "model.params.conditioner_mode",
            "model.params.prior_mode", "model.params.deterministic_only",
            "model.params.encoder_conditioning", "model.params.film_layers",
            "model.params.film_shared_generator",
            "model.params.use_centered_gene_structure",
            "model.params.n_refinement_steps", "loss.local_gradient_weight",
            "model.params.structured_composition",
            "loss.wide_gradient_weight", "training.checkpoint_dir",
            "evaluation.tensorboard.log_dir",
            "evaluation.structured_field_metrics",
        ]
        audit = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(path), "gpu": int(gpu), "audit": audit,
            "checkpoint_dir": config["training"]["checkpoint_dir"],
            "tensorboard_log_dir": config["evaluation"]["tensorboard"]["log_dir"],
        }

    plan = {
        "kind": "mk_structured_field_suite", "version": 1,
        "family": family, "arms": arms, "arm_order": list(arm_order),
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path), "train_gene_panels": str(panel_path),
        "centered_gene_structure": str(structure_path),
        "hours_per_arm": float(hours), "cpu_threads_per_arm": int(cpu_threads),
        "latent_dim": int(latent_dim),
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / f"LATEST_MK_STRUCTURED_FIELD_{family.upper()}_SUITE_ROOT.txt"
    temporary = pointer.with_name(f"{pointer.name}.tmp")
    temporary.write_text(str(root) + "\n")
    temporary.replace(pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--centered-gene-structure", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--uni2-pinned-revision", required=True)
    parser.add_argument("--uni2-spot-feature-cache-dir", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2,3,5")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--family", choices=tuple(FAMILY_SPECS), default="deterministic")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--hest-cache-dir")
    args = parser.parse_args()
    plan = prepare_mk_structured_field_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        centered_gene_structure=args.centered_gene_structure,
        output_root=args.output_root,
        uni2_pinned_revision=args.uni2_pinned_revision,
        uni2_spot_feature_cache_dir=args.uni2_spot_feature_cache_dir,
        hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        cpu_threads=args.cpu_threads,
        hest_data_dir=args.hest_data_dir,
        hest_cache_dir=args.hest_cache_dir,
        family=args.family,
        latent_dim=args.latent_dim,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
