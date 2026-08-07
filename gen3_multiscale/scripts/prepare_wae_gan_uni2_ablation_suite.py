#!/usr/bin/env python3
"""Write two matched image-encoder-backbone ablation configs: the existing
GigaPath control vs. a UNI2-h-backed arm.

Both arms share IDENTICAL training hyperparameters and model dimensions
(matching wae_he_gan_control exactly: lr=1e-4, no gradient accumulation,
full latent/hidden dims) and the same task/regularizer/
include_observed_gex/dataset/masking/losses/seed as every other WAE-GAN
arm. Only `data.image_encoder` (and its matching pinned-revision field)
differs between arms -- this is a backbone ablation, not a training-
hyperparameter or conditioning-design ablation. Never starts training.

Requires the UNI2 spot-feature cache to already exist for every sample in
the manifest (see scripts/precompute_gen3_uni2_spot_features.py) --
neither this script nor training itself ever calls the UNI2 tile encoder.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    _absolutize_existing_source_paths,
    _source_repository_root,
    tensorboard_block,
    whole_slide_validation_block,
)

ARM_ORDER = ("wae_he_gan_uni2_control", "wae_he_gan_uni2")


def prepare_wae_gan_uni2_ablation_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    output_root: str, uni2_pinned_revision: str, hours: float = 8.0,
    gpus: tuple[int, ...] = (0, 2), cpu_threads: int = 12,
    uni2_spot_feature_cache_dir: str | None = None,
) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if len(gpus) != len(ARM_ORDER) or len(set(gpus)) != len(gpus):
        raise ValueError(f"gpus must list {len(ARM_ORDER)} unique GPU ids, one per arm")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    if not uni2_pinned_revision:
        raise ValueError("uni2_pinned_revision must be set to the pinned MahmoodLab/UNI2-h commit SHA")
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
    # Matches wae_he_gan_control exactly -- same architecture/dims as the
    # existing production WAE-GAN control arm. image_feature_dim stays
    # 1536 for both arms: UNI2-h's real output dim happens to equal
    # GigaPath's, so no dimension change is needed to swap backbones.
    shared_params.update({
        "gex_feature_dim": int((base.get("data") or {}).get("gex_feature_dim", 256)),
        "latent_dim": 256,
        "autoencoder_hidden_dim": 1024,
        "discriminator_hidden_dim": 256,
        "n_inference_samples": 8,
        "dropout": 0.1,
    })
    base_seed = int((base.get("training") or {}).get("seed", 0))

    written = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        params = copy.deepcopy(shared_params)

        config = copy.deepcopy(base)
        config["experiment_name"] = f"wae_gan_uni2_ablation_{arm}_v1"
        config["model"] = {
            "arm": arm,
            "kind": "conditional_wae",
            "task": "he_to_st",
            "regularizer": "gan",
            "include_observed_gex": False,
            "image_mode": "full_visible",
            "params": params,
        }
        config["data"]["gen3_manifest_path"] = str(manifest_path)
        if arm == "wae_he_gan_uni2_control":
            config["data"]["image_encoder"] = "gigapath"
            config["documented_divergences"] = [
                "model.arm", "training.checkpoint_dir", "training.device",
                "evaluation.tensorboard.log_dir",
            ]
        else:
            config["data"]["image_encoder"] = "uni2"
            config["data"]["uni2_pinned_revision"] = str(uni2_pinned_revision)
            if uni2_spot_feature_cache_dir:
                config["data"]["gen3_uni2_spot_feature_cache_dir"] = str(uni2_spot_feature_cache_dir)
            config["documented_divergences"] = [
                "model.arm", "data.image_encoder", "data.uni2_pinned_revision",
                "data.gen3_uni2_spot_feature_cache_dir",
                "training.checkpoint_dir", "training.device", "evaluation.tensorboard.log_dir",
            ]
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(root, dataset_manifest)
        config["loss"] = {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.1,
            "conditional_mean_weight": 1.0,
        }
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            "device": f"cuda:{gpu}",
            "cpu_threads": int(cpu_threads),
            "seed": base_seed,
            "lr": 1e-4,
            "gradient_accumulation_steps": 1,
            "optimizer": {"weight_decay": 0.01},
        })
        report = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[arm] = {
            "config": str(path),
            "audit": report,
            "gpu": gpu,
            "image_encoder": config["data"]["image_encoder"],
            "tensorboard_log_dir": config["evaluation"]["tensorboard"]["log_dir"],
            "checkpoint_dir": config["training"]["checkpoint_dir"],
        }

    # Matched-except-declared-fields invariant: the uni2 arm's resolved
    # config must be identical to control except the fields its own spec
    # explicitly overrides.
    control_config = yaml.safe_load((root / "configs" / "wae_he_gan_uni2_control.yaml").read_text())
    uni2_config = yaml.safe_load((root / "configs" / "wae_he_gan_uni2.yaml").read_text())
    _assert_matched_except_declared(control_config, uni2_config, "wae_he_gan_uni2")

    plan = {
        "kind": "wae_gan_uni2_ablation_suite",
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours),
        "cpu_threads_per_arm": int(cpu_threads),
        "gpus": list(gpus),
        "uni2_pinned_revision": str(uni2_pinned_revision),
        "arms": written,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_WAE_GAN_UNI2_ABLATION_SUITE_ROOT.txt"
    tmp = pointer.with_name(f"{pointer.name}.tmp.{os.getpid()}")
    tmp.write_text(str(root.resolve()) + "\n")
    os.replace(tmp, pointer)
    return plan


_ALLOWED_DIVERGENT_KEYS = {
    ("model", "arm"),
    ("data", "image_encoder"),
    ("data", "uni2_pinned_revision"),
    ("data", "gen3_uni2_spot_feature_cache_dir"),
    ("training", "checkpoint_dir"),
    ("training", "device"),
    ("evaluation", "tensorboard", "log_dir"),
    ("experiment_name",),
    ("documented_divergences",),
}


def _assert_matched_except_declared(control: dict, other: dict, arm: str, path: tuple = ()) -> None:
    keys = sorted(set(control) | set(other))
    for key in keys:
        current_path = path + (key,)
        if current_path in _ALLOWED_DIVERGENT_KEYS:
            continue
        control_value = control.get(key)
        other_value = other.get(key)
        if isinstance(control_value, dict) and isinstance(other_value, dict):
            _assert_matched_except_declared(control_value, other_value, arm, current_path)
        elif control_value != other_value:
            raise ValueError(
                f"{arm}: undeclared divergence from control at "
                f"{'.'.join(current_path)}: {control_value!r} != {other_value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--uni2-pinned-revision", required=True,
        help="MANDATORY: the pinned, immutable MahmoodLab/UNI2-h Hugging Face commit SHA the "
             "UNI2 spot-feature cache was built from (scripts/precompute_gen3_uni2_spot_features.py).",
    )
    parser.add_argument(
        "--uni2-spot-feature-cache-dir", default=None,
        help="Optional override for where the uni2 arm looks up its spot-feature cache "
             "(gen3_uni2_spot_feature_cache_dir). Defaults to the same hest_cache_dir/"
             "hest_data_dir convention the GigaPath cache already uses -- set this only if the "
             "UNI2 cache lives under a different root (e.g. built from a different checkout). "
             "IMPORTANT: pass the PARENT of uni2_gen3_spot_cache/, not that directory itself -- "
             "e.g. for a cache at /data/.../hest1k/uni2_gen3_spot_cache/, pass /data/.../hest1k "
             "(the loader appends uni2_gen3_spot_cache/<sample_id>.npz itself).",
    )
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2", help="comma-separated GPU ids, one per arm")
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    plan = prepare_wae_gan_uni2_ablation_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root,
        uni2_pinned_revision=args.uni2_pinned_revision,
        hours=args.hours,
        gpus=gpus,
        cpu_threads=args.cpu_threads,
        uni2_spot_feature_cache_dir=args.uni2_spot_feature_cache_dir,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
