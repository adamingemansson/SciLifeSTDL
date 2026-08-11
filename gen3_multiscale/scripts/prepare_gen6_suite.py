#!/usr/bin/env python3
"""Prepare immutable, matched 8-hour configs for Gen6 deterministic arms.

This command writes configs and a run plan only.  It never starts training.
It deliberately clones the resolved comparison config's data, masking,
evaluation and optimizer sections so the component screen differs only in
the documented model components and common Gen6 RMSE+PCC objective.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen6.contract import GEN6_ARM_SPECS, REFINED_SPATIAL_FIELD_ARMS
from gen3_multiscale.gen6.preflight import static_audit_gen6_config


_DETERMINISTIC_ARMS = tuple(f"gen6{letter}" for letter in "abcdefghijmn")

# Refinement depth/neighbourhood per arm. Two points rather than one because a
# single positive result at one setting is indistinguishable from luck, and
# the two settings bracket the neighbour-mean baseline's own k.
_REFINEMENT_PARAMS = {
    "gen6m": {"n_refinement_steps": 2, "refinement_k_neighbors": 6},
    "gen6n": {"n_refinement_steps": 4, "refinement_k_neighbors": 12},
}

if set(_REFINEMENT_PARAMS) != REFINED_SPATIAL_FIELD_ARMS:
    raise RuntimeError(
        "refinement settings and the arm contract disagree: "
        f"{sorted(_REFINEMENT_PARAMS)} vs {sorted(REFINED_SPATIAL_FIELD_ARMS)}"
    )
if not REFINED_SPATIAL_FIELD_ARMS <= set(_DETERMINISTIC_ARMS):
    raise RuntimeError("a refinement arm is missing from the deterministic suite")


def _pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        if not name or not value or name in result:
            raise ValueError(f"invalid or duplicate fingerprint {raw!r}")
        result[name] = value
    return result


def prepare_gen6_suite(*, comparison_config: str, manifest: str, train_gene_panels: str,
                       output_root: str, fingerprints: dict[str, str],
                       hours: float = 8.0,
                       retain_patches_in_memory: bool = True) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; Gen6 suite roots are immutable")
    base = yaml.safe_load(Path(comparison_config).read_text())
    if str((base.get("model") or {}).get("kind", "")) != "conditioner":
        raise ValueError(
            "--comparison-config must be a resolved deterministic conditioner config; "
            "flow/latent-flow configs carry staged parameters that are not a matched Gen6 base"
        )
    dataset_manifest = load_dataset_manifest(manifest)
    panel_path = Path(train_gene_panels).resolve()
    panel_artifact = load_train_derived_gene_panels(panel_path, dataset_manifest)
    required_panels = {
        "train_log1p_variance_top50",
        "train_log1p_variance_top200",
    }
    missing_panels = sorted(required_panels - set(panel_artifact["panels"]))
    if missing_panels:
        raise ValueError(
            "Gen6 evaluation requires train-derived HVG-50 and HVG-200 panels; "
            f"artifact is missing {missing_panels}"
        )
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "checkpoints").mkdir()
    (root / "logs").mkdir()
    written = {}
    for arm in _DETERMINISTIC_ARMS:
        spec = GEN6_ARM_SPECS[arm]
        config = copy.deepcopy(base)
        config["experiment_name"] = f"gen6_component_screen_{arm}_v1"
        config["documented_divergences"] = [
            "model.arm", "model.kind", "model.params", "loss.primary_mode", "loss.pcc_weight",
        ]
        params = dict((config.get("model") or {}).get("params") or {})
        params.pop("gene_basis_rank", None)
        # Cleared unconditionally: the contract, not an inherited base config,
        # decides which arms refine. build_gen6_model rejects the field on any
        # arm that does not declare it.
        for key in ("n_refinement_steps", "refinement_k_neighbors",
                    "refinement_hidden_dim", "refinement_gex_feature_dim"):
            params.pop(key, None)
        params.update({"init_seed": 0, "fusion_heads": 4})
        params.update(_REFINEMENT_PARAMS.get(arm, {}))
        if spec.uses_scfoundation:
            params.setdefault("gex_context_embedding_dim", 3072)
        config["model"] = {"arm": arm, "kind": "conditioner", "params": params}
        config.setdefault("loss", {})
        config["loss"].update({"primary_mode": "rmse_pcc", "pcc_weight": 0.1})
        config["data"]["gen3_manifest_path"] = str(manifest)
        config["data"]["retain_patches_in_memory"] = bool(retain_patches_in_memory)
        config.setdefault("evaluation", {})
        config["evaluation"]["train_gene_panel_artifact"] = str(panel_path)
        config["training"]["checkpoint_dir"] = str(root / "checkpoints" / arm)
        config["training"]["total_steps"] = 100_000_000
        config["training"]["max_wall_clock_hours"] = float(hours)
        config["required_fingerprints"].update(fingerprints)
        report = static_audit_gen6_config(config)
        if not report["ready_for_real_training"]:
            raise ValueError(f"{arm} unresolved launch fields: {report['unset_required_fingerprints']}")
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[arm] = {"config": str(path), "spec": spec.to_dict()}
    plan = {
        "kind": "gen6_deterministic_component_screen",
        "comparison_config": str(Path(comparison_config).resolve()),
        "manifest": str(Path(manifest).resolve()), "hours_per_arm": float(hours),
        "train_gene_panels": {
            "path": str(panel_path),
            "artifact_sha256": panel_artifact["artifact_sha256"],
            "panels": sorted(panel_artifact["panels"]),
        },
        "arms": written,
        "stage_two": {
            "gen6k": "freeze Gen6-C and apply learned-latent minibatch-OT flow",
            "gen6l": "freeze the same Gen6-C checkpoint and train its conditional WAE-GAN",
            "gen6p": "gen6k with hard OT assignment (prepare_gen6_generators.py)",
            "gen6o": "gen6l with a supervised conditional-mean head (prepare_gen6_generators.py)",
        },
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_GEN6_SUITE_ROOT.txt"
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
    parser.add_argument("--fingerprint", action="append", default=[])
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument(
        "--release-patches-after-verification", action="store_true",
        help="Free each slide's raw H&E patch array once the spot-feature cache has "
             "re-hashed it against its stored patch_content_sha256. Training reads only "
             "the verified cached features afterwards, so this drops ~376 MB per "
             "2,500-spot slide per process with no effect on any metric. With four "
             "concurrent GPU queues that is the difference between comfortable and "
             "saturating the shared box.",
    )
    args = parser.parse_args()
    plan = prepare_gen6_suite(
        comparison_config=args.comparison_config, manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        output_root=args.output_root, fingerprints=_pairs(args.fingerprint), hours=args.hours,
        retain_patches_in_memory=not args.release_patches_after_verification,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
