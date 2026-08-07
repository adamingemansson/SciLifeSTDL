#!/usr/bin/env python3
"""Write four matched configs for a gene-encoder x FiLM factorial ablation,
all on the WAE-MMD (Wasserstein) regularizer -- no GAN arm in this suite.

Compares scFoundation's frozen per-gene embedding table (real expression @
frozen_table.T -> trainable projection into hidden_dim, sourced from the
already-fit scFoundation-derived gene-coexpression basis artifact reused
here as the table) against the plain from-scratch Linear(n_genes,
hidden_dim) MLP encoder every OTHER WAE arm already uses ("mlp" --
gene_encoder_source="linear", no table/basis file at all). Crossed with
`model.params.encoder_conditioning` (FiLM on/off) so the encoder-source
effect and the FiLM effect are each independently visible, not confounded:

    arm                                     gene encoder     FiLM
    wae_he_mmd_geneencoder_scfoundation_film   scFoundation    yes
    wae_he_mmd_geneencoder_scfoundation_nofilm scFoundation    no
    wae_he_mmd_geneencoder_mlp_film            plain MLP       yes
    wae_he_mmd_geneencoder_mlp_nofilm          plain MLP       no

"wae_he_mmd_geneencoder_mlp_nofilm" is therefore a genuine no-intervention
control (plain encoder, no FiLM, just regularizer=mmd/latent_dim=64) --
the other three each add exactly one real change on top of it.

All four share IDENTICAL latent_dim=64, autoencoder dims, task/
include_observed_gex/dataset/masking/losses/seed/image-encoder. Only
model.params.gene_encoder_source/encoder_conditioning/film_layers and
data.gene_encoder_table_path (scFoundation arms only) differ between arms,
each declared explicitly -- this is a gene-representation x conditioning-
design ablation, not a training-hyperparameter or backbone ablation. Never
starts training.

Requires the scFoundation-derived gene-coexpression basis artifact to
already exist (see
scripts/fit_conditional_wae_gene_coexpression_basis_from_scfoundation.py)
-- neither this script nor training itself ever fits one. The plain-MLP
arms need no basis file at all.

All four arms default to UNI2 as the image encoder (not GigaPath) --
this project's standing preference for new suites going forward. Requires
the UNI2 spot-feature cache to already exist for every sample in the
manifest (see scripts/precompute_gen3_uni2_spot_features.py).
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

ARM_ORDER = (
    "wae_he_mmd_geneencoder_scfoundation_film",
    "wae_he_mmd_geneencoder_scfoundation_nofilm",
    "wae_he_mmd_geneencoder_mlp_film",
    "wae_he_mmd_geneencoder_mlp_nofilm",
)


def prepare_wae_mmd_geneencoder_ablation_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    output_root: str, scfoundation_basis_path: str, uni2_pinned_revision: str,
    latent_dim: int = 64, hours: float = 8.0, gpus: tuple[int, ...] = (0, 1, 2, 3),
    cpu_threads: int = 12, uni2_spot_feature_cache_dir: str | None = None,
) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if len(gpus) != len(ARM_ORDER) or len(set(gpus)) != len(gpus):
        raise ValueError(f"gpus must list {len(ARM_ORDER)} unique GPU ids, one per arm")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    if latent_dim < 1:
        raise ValueError("latent_dim must be positive")
    if not scfoundation_basis_path:
        raise ValueError("scfoundation_basis_path must be set to an already-fit basis artifact")
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
    shared_params.update({
        "gex_feature_dim": int((base.get("data") or {}).get("gex_feature_dim", 256)),
        "latent_dim": int(latent_dim),
        "autoencoder_hidden_dim": 1024,
        "discriminator_hidden_dim": 256,  # unused for regularizer="mmd" (no discriminator), kept for schema
        "n_inference_samples": 8,
        "dropout": 0.1,
        "encoder_conditioning": "film",
        "film_layers": ["first", "second"],
        "film_shared_generator": False,
    })
    base_seed = int((base.get("training") or {}).get("seed", 0))

    written = {}
    for arm, gpu in zip(ARM_ORDER, gpus):
        use_scfoundation = "scfoundation" in arm
        use_film = arm.endswith("_film")
        params = copy.deepcopy(shared_params)
        params["gene_encoder_source"] = "frozen_table" if use_scfoundation else "linear"
        if not use_film:
            params["encoder_conditioning"] = "none"
            del params["film_layers"]
            del params["film_shared_generator"]

        config = copy.deepcopy(base)
        config["experiment_name"] = f"wae_mmd_geneencoder_ablation_{arm}_v1"
        config["model"] = {
            "arm": arm,
            "kind": "conditional_wae",
            "task": "he_to_st",
            "regularizer": "mmd",
            "include_observed_gex": False,
            "image_mode": "full_visible",
            "params": params,
        }
        config["data"]["gen3_manifest_path"] = str(manifest_path)
        config["data"]["image_encoder"] = "uni2"
        config["data"]["uni2_pinned_revision"] = str(uni2_pinned_revision)
        if uni2_spot_feature_cache_dir:
            config["data"]["gen3_uni2_spot_feature_cache_dir"] = str(uni2_spot_feature_cache_dir)
        divergences = [
            "model.arm", "model.params.gene_encoder_source",
            "training.checkpoint_dir", "training.device", "evaluation.tensorboard.log_dir",
        ]
        if use_scfoundation:
            config["data"]["gene_encoder_table_path"] = str(scfoundation_basis_path)
            divergences.append("data.gene_encoder_table_path")
        if use_film:
            divergences += [
                "model.params.encoder_conditioning", "model.params.film_layers",
                "model.params.film_shared_generator",
            ]
        else:
            divergences += ["model.params.encoder_conditioning"]
        config["documented_divergences"] = divergences
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
            "gene_encoder_source": params["gene_encoder_source"],
            "encoder_conditioning": params.get("encoder_conditioning", "none"),
            "tensorboard_log_dir": config["evaluation"]["tensorboard"]["log_dir"],
            "checkpoint_dir": config["training"]["checkpoint_dir"],
        }

    # Matched-except-declared-fields invariant, checked pairwise against a
    # single reference arm (the first, scfoundation+film): every OTHER arm
    # must match it except for the specific keys ITS OWN divergence list
    # declares. This is a genuine 2x2 factorial (no shared "control"), so
    # each comparison's allowed-divergent-key set differs depending on
    # which factor(s) that arm actually varies relative to the reference.
    reference_arm = ARM_ORDER[0]
    reference_config = yaml.safe_load((root / "configs" / f"{reference_arm}.yaml").read_text())
    for arm in ARM_ORDER[1:]:
        arm_config = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
        allowed = _allowed_divergent_keys_between(reference_arm, arm)
        _assert_matched_except_declared(reference_config, arm_config, arm, allowed)

    plan = {
        "kind": "wae_mmd_geneencoder_ablation_suite",
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path),
        "hours_per_arm": float(hours),
        "cpu_threads_per_arm": int(cpu_threads),
        "latent_dim": int(latent_dim),
        "gpus": list(gpus),
        "scfoundation_basis_path": str(scfoundation_basis_path),
        "uni2_pinned_revision": str(uni2_pinned_revision),
        "arms": written,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_WAE_MMD_GENEENCODER_ABLATION_SUITE_ROOT.txt"
    tmp = pointer.with_name(f"{pointer.name}.tmp.{os.getpid()}")
    tmp.write_text(str(root.resolve()) + "\n")
    os.replace(tmp, pointer)
    return plan


_BASE_ALLOWED_DIVERGENT_KEYS = {
    ("model", "arm"),
    ("model", "params", "gene_encoder_source"),
    ("data", "gene_encoder_table_path"),
    ("training", "checkpoint_dir"),
    ("training", "device"),
    ("evaluation", "tensorboard", "log_dir"),
    ("experiment_name",),
    ("documented_divergences",),
}
_FILM_ALLOWED_DIVERGENT_KEYS = {
    ("model", "params", "encoder_conditioning"),
    ("model", "params", "film_layers"),
    ("model", "params", "film_shared_generator"),
}


def _allowed_divergent_keys_between(reference_arm: str, other_arm: str) -> set[tuple[str, ...]]:
    reference_film = reference_arm.endswith("_film")
    other_film = other_arm.endswith("_film")
    allowed = set(_BASE_ALLOWED_DIVERGENT_KEYS)
    if reference_film != other_film:
        allowed |= _FILM_ALLOWED_DIVERGENT_KEYS
    return allowed


def _assert_matched_except_declared(
    control: dict, other: dict, arm: str, allowed_keys: set[tuple[str, ...]], path: tuple = (),
) -> None:
    keys = sorted(set(control) | set(other))
    for key in keys:
        current_path = path + (key,)
        if current_path in allowed_keys:
            continue
        control_value = control.get(key)
        other_value = other.get(key)
        if isinstance(control_value, dict) and isinstance(other_value, dict):
            _assert_matched_except_declared(control_value, other_value, arm, allowed_keys, current_path)
        elif control_value != other_value:
            raise ValueError(
                f"{arm}: undeclared divergence from {ARM_ORDER[0]} at "
                f"{'.'.join(current_path)}: {control_value!r} != {other_value!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--scfoundation-basis-path", required=True,
        help="MANDATORY: path to an already-fit scFoundation-derived gene coexpression basis "
             "artifact (scripts/fit_conditional_wae_gene_coexpression_basis_from_scfoundation.py) "
             "-- reused here as the frozen table for the scFoundation-encoder arms.",
    )
    parser.add_argument(
        "--uni2-pinned-revision", required=True,
        help="MANDATORY: the pinned, immutable MahmoodLab/UNI2-h Hugging Face commit SHA the "
             "UNI2 spot-feature cache was built from (scripts/precompute_gen3_uni2_spot_features.py). "
             "Every arm in this suite uses UNI2, not GigaPath.",
    )
    parser.add_argument(
        "--uni2-spot-feature-cache-dir", default=None,
        help="Optional override for where every arm looks up its UNI2 spot-feature cache. "
             "Defaults to the same hest_cache_dir/hest_data_dir convention every other Gen3 "
             "spot-feature cache uses.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids, one per arm")
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(","))
    plan = prepare_wae_mmd_geneencoder_ablation_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root,
        scfoundation_basis_path=args.scfoundation_basis_path,
        uni2_pinned_revision=args.uni2_pinned_revision,
        uni2_spot_feature_cache_dir=args.uni2_spot_feature_cache_dir,
        latent_dim=args.latent_dim,
        hours=args.hours,
        gpus=gpus,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
