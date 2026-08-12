#!/usr/bin/env python3
"""Prepare all four MK batches and a two-wave, eight-at-once launch plan.

Nothing is trained by this module. Wave 1 contains Batch A + B and wave 2
contains Batch C + D. Each wave therefore has exactly two processes on each
of GPUs 0,2,3,5. Every four-arm batch writes to its own TensorBoard root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gen3_multiscale.scripts.prepare_mk_architecture_suite import (
    prepare_mk_architecture_suite,
)
from gen3_multiscale.scripts.prepare_mk_structured_field_suite import (
    prepare_mk_structured_field_suite,
)


BATCHES = (
    ("batch_a_architecture_controls", "architecture_controls"),
    ("batch_b_deterministic_structured", "deterministic"),
    ("batch_c_standard_wae_structured", "standard_wae"),
    ("batch_d_conditional_wae_structured", "conditional_wae"),
)


def _arm_records(batch_name: str, root: Path, plan: dict) -> list[dict]:
    return [
        {
            "arm": arm,
            "batch": batch_name,
            "gpu": int(plan["arms"][arm]["gpu"]),
            "config": str(Path(plan["arms"][arm]["config"]).resolve()),
            "checkpoint_dir": str(Path(plan["arms"][arm]["checkpoint_dir"]).resolve()),
            "log": str((root / "logs" / f"{arm}.log").resolve()),
        }
        for arm in plan["arm_order"]
    ]


def prepare_mk_16_arm_suite(
    *, comparison_config: str, manifest: str, train_gene_panels: str,
    centered_gene_structure: str, output_root: str,
    uni2_pinned_revision: str, uni2_spot_feature_cache_dir: str,
    hours: float = 8.0, gpus: tuple[int, ...] = (0, 2, 3, 5),
    cpu_threads_per_arm: int = 6, latent_dim: int = 64,
    hest_data_dir: str | None = None, hest_cache_dir: str | None = None,
) -> dict:
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("gpus must contain exactly four distinct ids")
    if hours <= 0 or cpu_threads_per_arm < 1 or latent_dim < 1:
        raise ValueError("hours, cpu_threads_per_arm and latent_dim must be positive")
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"{root} already exists; master suite roots are immutable")
    batches_root = root / "batches"
    batches_root.mkdir(parents=True)

    common = dict(
        comparison_config=comparison_config,
        manifest=manifest,
        train_gene_panels=train_gene_panels,
        uni2_pinned_revision=uni2_pinned_revision,
        uni2_spot_feature_cache_dir=uni2_spot_feature_cache_dir,
        hours=hours,
        gpus=gpus,
        cpu_threads=cpu_threads_per_arm,
        hest_data_dir=hest_data_dir,
        hest_cache_dir=hest_cache_dir,
    )
    batch_plans: dict[str, dict] = {}
    batch_roots: dict[str, Path] = {}
    for batch_name, family in BATCHES:
        batch_root = batches_root / batch_name
        if family == "architecture_controls":
            plan = prepare_mk_architecture_suite(
                output_root=str(batch_root), latent_dim=latent_dim, **common,
            )
        else:
            plan = prepare_mk_structured_field_suite(
                output_root=str(batch_root), centered_gene_structure=centered_gene_structure,
                family=family, latent_dim=latent_dim, **common,
            )
        batch_plans[batch_name] = plan
        batch_roots[batch_name] = batch_root

    wave_batches = (
        ("wave_1", ("batch_a_architecture_controls", "batch_b_deterministic_structured")),
        ("wave_2", ("batch_c_standard_wae_structured", "batch_d_conditional_wae_structured")),
    )
    waves = []
    for wave_name, names in wave_batches:
        arms = []
        for name in names:
            arms.extend(_arm_records(name, batch_roots[name], batch_plans[name]))
        gpu_counts = {str(gpu): sum(record["gpu"] == gpu for record in arms) for gpu in gpus}
        if set(gpu_counts.values()) != {2}:
            raise ValueError(f"{wave_name}: expected exactly two arms per GPU, got {gpu_counts}")
        waves.append({"name": wave_name, "batches": list(names), "arms": arms,
                      "concurrent_processes": 8, "gpu_process_counts": gpu_counts})

    tensorboard_groups = {
        name: {
            "logdir": str((batch_roots[name] / "tensorboard").resolve()),
            "arms": list(batch_plans[name]["arm_order"]),
        }
        for name, _family in BATCHES
    }
    master = {
        "kind": "mk_16_arm_two_wave_suite",
        "version": 1,
        "root": str(root),
        "hours_per_arm": float(hours),
        "cpu_threads_per_arm": int(cpu_threads_per_arm),
        "gpus": list(gpus),
        "waves": waves,
        "tensorboard_groups": tensorboard_groups,
        "batch_plan_paths": {
            name: str((batch_roots[name] / "suite_plan.json").resolve())
            for name, _family in BATCHES
        },
    }
    (root / "master_plan.json").write_text(json.dumps(master, indent=2, sort_keys=True))
    (root / "control").mkdir()
    pointer = root.parent / "LATEST_MK_16_ARM_SUITE_ROOT.txt"
    temporary = pointer.with_name(f"{pointer.name}.tmp")
    temporary.write_text(str(root) + "\n")
    temporary.replace(pointer)
    return master


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
    parser.add_argument("--cpu-threads-per-arm", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--hest-cache-dir")
    args = parser.parse_args()
    plan = prepare_mk_16_arm_suite(
        comparison_config=args.comparison_config,
        manifest=args.manifest,
        train_gene_panels=args.train_gene_panels,
        centered_gene_structure=args.centered_gene_structure,
        output_root=args.output_root,
        uni2_pinned_revision=args.uni2_pinned_revision,
        uni2_spot_feature_cache_dir=args.uni2_spot_feature_cache_dir,
        hours=args.hours,
        gpus=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
        cpu_threads_per_arm=args.cpu_threads_per_arm,
        latent_dim=args.latent_dim,
        hest_data_dir=args.hest_data_dir,
        hest_cache_dir=args.hest_cache_dir,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
