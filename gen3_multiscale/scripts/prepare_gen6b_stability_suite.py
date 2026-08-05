#!/usr/bin/env python3
"""Prepare three immutable, seed-matched Gen6-B configs for Adam's
Gen6-B stability/overfitting/seed-reproducibility audit, Phase 3.

This command writes configs and a run plan only -- it never starts
training, exactly like `prepare_gen6_suite.py`. Unlike that script
(which builds one config per DIFFERENT arm from a comparison base),
this one clones a single REAL, already-resolved `gen6b.yaml` verbatim
three times, touching ONLY `training.seed`, `training.checkpoint_dir`,
`training.max_wall_clock_hours`, `training.early_stopping` and
`training.tensorboard.log_dir`. Every other field (manifest, patient
split, masking strata, loss, optimizer, model.arm/params, gene panels)
is required to be byte-identical across the three seed configs --
`_assert_matched_except` enforces that invariant itself before writing
anything, rather than merely documenting it.

Gen6-B's validation masks are already deterministic and seed-INDEPENDENT
by construction: `training/train.py::run_training` builds the validation
schedule with a hardcoded `split_seeds={"validation": 700_000}`, never
`training.seed` -- so "same deterministic validation masks across all
three seeds" holds automatically and needs no extra config field here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml


def _is_allowed(path: tuple[str, ...], allowed_diff_paths: set[tuple[str, ...]]) -> bool:
    """A path is allowed if it, or any of its ancestor paths, is in the
    allowed set -- e.g. allowing ("training", "tensorboard") also allows
    the nested ("training", "tensorboard", "log_dir") difference under it."""
    return any(path[: len(prefix)] == prefix for prefix in allowed_diff_paths)


def _assert_matched_except(
    base: dict, variant: dict, *, allowed_diff_paths: set[tuple[str, ...]], context: str,
) -> None:
    """Recursively diff two config dicts; raise on any difference whose
    path is not explicitly allowed. `context` names the two configs being
    compared in the error message (e.g. "seed0 vs seed1")."""

    def walk(a, b, path: tuple[str, ...]) -> None:
        if _is_allowed(path, allowed_diff_paths):
            return
        if isinstance(a, dict) or isinstance(b, dict):
            if not isinstance(a, dict) or not isinstance(b, dict):
                if a != b:
                    raise ValueError(
                        f"{context}: configs must be identical except {sorted(allowed_diff_paths)}, "
                        f"but {'.'.join(path)} differs in shape: {a!r} != {b!r}"
                    )
                return
            for key in sorted(set(a) | set(b)):
                walk(a.get(key), b.get(key), path + (key,))
            return
        if a != b:
            raise ValueError(
                f"{context}: configs must be identical except {sorted(allowed_diff_paths)}, but "
                f"{'.'.join(path)} differs: {a!r} != {b!r}"
            )

    walk(base, variant, ())


_ALLOWED_DIFF_PATHS = {
    ("training", "seed"),
    ("training", "checkpoint_dir"),
    ("training", "max_wall_clock_hours"),
    ("training", "early_stopping"),
    ("training", "tensorboard"),
}


def prepare_gen6b_stability_suite(
    *, reference_config: str, seeds: list[int], output_root: str,
    hours: float = 8.0,
    monitor: str = "validation_total", mode: str = "min",
    patience_validations: int = 8, min_delta: float = 0.0001,
) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a set of at least two distinct integers")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; stability-suite roots are immutable")
    base = yaml.safe_load(Path(reference_config).read_text())
    if str((base.get("model") or {}).get("arm", "")) != "gen6b":
        raise ValueError("--reference-config must be a resolved gen6b.yaml (model.arm == 'gen6b')")
    if "training" not in base or "checkpoint_dir" not in base["training"]:
        raise ValueError("--reference-config must already have a resolved training.checkpoint_dir")

    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "checkpoints").mkdir()
    (root / "tensorboard").mkdir()
    (root / "logs").mkdir()

    written: dict[str, dict] = {}
    for seed in seeds:
        config = copy.deepcopy(base)
        config["training"]["seed"] = int(seed)
        config["training"]["checkpoint_dir"] = str(root / "checkpoints" / f"seed{seed}")
        config["training"]["max_wall_clock_hours"] = float(hours)
        config["training"]["early_stopping"] = {
            "monitor": monitor, "mode": mode,
            "patience_validations": int(patience_validations), "min_delta": float(min_delta),
        }
        config["training"]["tensorboard"] = {"log_dir": str(root / "tensorboard" / f"seed{seed}")}
        _assert_matched_except(
            base, config, allowed_diff_paths=_ALLOWED_DIFF_PATHS, context=f"reference vs seed{seed}",
        )
        path = root / "configs" / f"gen6b_seed{seed}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written[str(seed)] = {
            "config": str(path),
            "checkpoint_dir": config["training"]["checkpoint_dir"],
            "tensorboard_log_dir": config["training"]["tensorboard"]["log_dir"],
            "log_path": str(root / "logs" / f"gen6b_seed{seed}.log"),
            "pid_path": str(root / "logs" / f"gen6b_seed{seed}.pid"),
        }

    # Cross-check every WRITTEN config against every other -- not just
    # each against `base` -- so a bug that only diverged from `base` in
    # an allowed way but ALSO diverged between two variants in a
    # disallowed way cannot slip through.
    variant_items = list(written.items())
    for i in range(len(variant_items)):
        for j in range(i + 1, len(variant_items)):
            seed_i, seed_j = variant_items[i][0], variant_items[j][0]
            config_i = yaml.safe_load(Path(variant_items[i][1]["config"]).read_text())
            config_j = yaml.safe_load(Path(variant_items[j][1]["config"]).read_text())
            _assert_matched_except(
                config_i, config_j, allowed_diff_paths=_ALLOWED_DIFF_PATHS,
                context=f"seed{seed_i} vs seed{seed_j}",
            )

    plan = {
        "kind": "gen6b_stability_seed_suite",
        "reference_config": str(Path(reference_config).resolve()),
        "arm": "gen6b",
        "seeds": list(seeds),
        "hours_per_run": float(hours),
        "early_stopping": {
            "monitor": monitor, "mode": mode,
            "patience_validations": int(patience_validations), "min_delta": float(min_delta),
        },
        "runs": written,
        "launch_commands": {
            seed: f"python -m gen3_multiscale.training.train --config {info['config']}"
            for seed, info in written.items()
        },
        "gpu_selection": (
            "select 3 genuinely free GPUs from {0,2,3,5} (check via nvidia-smi before "
            "launching); one seed per GPU; apply a reasonable per-process CPU-thread cap "
            "(e.g. SCILIFESTDL_CPU_THREADS=8); never kill or interfere with unrelated processes"
        ),
        "tensorboard_launch": (
            f"python -m tensorboard.main --logdir {root / 'tensorboard'} "
            "--host 127.0.0.1 --port <dynamically-selected-free-port>"
        ),
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    pointer = root.parent / "LATEST_GEN6B_STABILITY_SUITE_ROOT.txt"
    tmp = pointer.with_name(f"{pointer.name}.tmp.{os.getpid()}")
    tmp.write_text(str(root.resolve()) + "\n")
    os.replace(tmp, pointer)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", required=True, help="A real, resolved gen6b.yaml")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--monitor", default="validation_total")
    parser.add_argument("--mode", default="min")
    parser.add_argument("--patience-validations", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    args = parser.parse_args()
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    plan = prepare_gen6b_stability_suite(
        reference_config=args.reference_config, seeds=seeds, output_root=args.output_root,
        hours=args.hours, monitor=args.monitor, mode=args.mode,
        patience_validations=args.patience_validations, min_delta=args.min_delta,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
