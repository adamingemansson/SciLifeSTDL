#!/usr/bin/env python3
"""Prepare eight deterministic gene/field mechanism arms."""
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
    "mk_pg_genequery", "mk_pg_field", "mk_pg_identity", "mk_pg_spectrum",
    "mk_pg_genequery_field", "mk_pg_genequery_structure",
    "mk_pg_field_structure", "mk_pg_full",
)
ARM_DESIGN = {
    # gene-query, field/amplitude/differential, map identity, spectrum
    "mk_pg_genequery": (True, False, False, False),
    "mk_pg_field": (False, True, False, False),
    "mk_pg_identity": (False, False, True, False),
    "mk_pg_spectrum": (False, False, False, True),
    "mk_pg_genequery_field": (True, True, False, False),
    "mk_pg_genequery_structure": (True, False, True, True),
    "mk_pg_field_structure": (False, True, True, True),
    "mk_pg_full": (True, True, True, True),
}


def _structured_panel(artifact: dict) -> list[str]:
    """Stable union of global-HVG and within-slide-HVG panels."""
    panels = artifact["panels"]
    names = []
    seen = set()
    for key in (
        "train_log1p_variance_top200",
        "train_within_slide_variance_top200",
    ):
        for gene in panels[key]:
            gene = str(gene)
            if gene not in seen:
                seen.add(gene)
                names.append(gene)
    return names


def prepare_mk_gene_field_suite(
    *, source_config: str, output_root: str, hours: float = 8.0,
    gpus: tuple[int, ...] = (0, 2, 3, 5), cpu_threads: int = 6,
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
        "train_log1p_variance_top200", "train_within_slide_variance_top200",
    }
    if not required.issubset(panel_artifact["panels"]):
        raise ValueError("expanded global and within-slide panels are required")
    structured_genes = _structured_panel(panel_artifact)
    if len(structured_genes) < 200:
        raise ValueError("structured loss panel unexpectedly contains fewer than 200 genes")
    cache_root = Path(source["data"]["gen3_uni2_spot_feature_cache_dir"]).resolve()
    require_uni2_spot_cache_coverage(cache_root, manifest["samples"])
    if not source["data"].get("centered_gene_structure_path"):
        raise ValueError("source_config is missing centered gene structure")

    root.mkdir(parents=True)
    for directory in ("configs", "checkpoints", "logs", "tensorboard", "control"):
        (root / directory).mkdir()
    arms = {}
    gpu_assignments = tuple(gpus) + tuple(gpus)
    for arm, gpu in zip(ARM_ORDER, gpu_assignments):
        gene_query, field, identity, spectrum = ARM_DESIGN[arm]
        config = copy.deepcopy(source)
        config["experiment_name"] = f"mk_gene_field_{arm}_v1"
        config["model"]["arm"] = arm
        run_params = config["model"]["params"]
        run_params.update({
            "decoder_kind": (
                "structured_gene_query" if gene_query else "mlp"
            ),
            "gene_query_dim": 256,
            "structured_loss_gene_names": structured_genes,
            "gene_identity_temperature": 0.1,
            "spectrum_max_rank": 64,
        })
        config["loss"].update({
            "local_gradient_weight": 0.025 if field else 0.0,
            "wide_gradient_weight": 0.025 if field else 0.0,
            "field_mean_weight": 0.05 if field else 0.0,
            "field_amplitude_weight": 0.05 if field else 0.0,
            "gene_identity_weight": 0.01 if identity else 0.0,
            "spectrum_weight": 0.02 if spectrum else 0.0,
        })
        config["training"].update({
            "checkpoint_dir": str(root / "checkpoints" / arm),
            "total_steps": 100_000_000,
            "max_wall_clock_hours": float(hours),
            # This is a fixed-duration mechanism screen.  Model selection
            # still uses the saved best validation checkpoint, but one arm
            # may not receive less optimization simply because its curve is
            # noisier early in training.
            "early_stopping": None,
            "device": "cuda", "cpu_threads": int(cpu_threads),
        })
        config["evaluation"]["tensorboard"] = tensorboard_block(root, arm)
        config["evaluation"]["whole_slide_validation"] = whole_slide_validation_block(
            root, manifest,
        )
        config["documented_divergences"] = [
            "experiment_name", "model.arm", "model.params.decoder_kind",
            "model.params.gene_query_dim",
            "model.params.structured_loss_gene_names",
            "model.params.gene_identity_temperature",
            "model.params.spectrum_max_rank", "loss.local_gradient_weight",
            "loss.wide_gradient_weight", "loss.field_mean_weight",
            "loss.field_amplitude_weight", "loss.gene_identity_weight",
            "loss.spectrum_weight", "training.checkpoint_dir",
            "training.max_wall_clock_hours", "training.early_stopping",
            "evaluation.tensorboard.log_dir",
        ]
        audit = static_audit_conditional_wae_config(config)
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        arms[arm] = {
            "config": str(path), "gpu": int(gpu),
            "checkpoint_dir": config["training"]["checkpoint_dir"],
            "design": {
                "gene_query": gene_query, "field": field,
                "identity": identity, "spectrum": spectrum,
            },
            "audit": audit,
        }
    plan = {
        "kind": "mk_gene_field_suite", "version": 1, "root": str(root),
        "source_config": str(source_path), "manifest": str(manifest_path),
        "train_gene_panels": str(panel_path), "arm_order": list(ARM_ORDER),
        "arms": arms, "hours_per_arm": float(hours),
        "cpu_threads_per_arm": int(cpu_threads),
        "concurrent_processes": 8,
        "gpu_process_counts": {str(gpu): 2 for gpu in gpus},
        "structured_loss_panel_size": len(structured_genes),
        "controlled_factors": [
            "model.params.decoder_kind", "loss.field_mean_weight",
            "loss.field_amplitude_weight", "loss.local_gradient_weight",
            "loss.wide_gradient_weight", "loss.gene_identity_weight",
            "loss.spectrum_weight",
        ],
    }
    (root / "suite_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_MK_GENE_FIELD_SUITE_ROOT.txt"
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
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(prepare_mk_gene_field_suite(
        source_config=args.source_config, output_root=args.output_root,
        hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        cpu_threads=args.cpu_threads,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
