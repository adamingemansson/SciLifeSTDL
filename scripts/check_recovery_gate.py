#!/usr/bin/env python3
"""Fail closed when the narrow recovery stage has not restored learning."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


RESIDUAL_RUNS = [
    *(f"recovery_hr_builtin_seed{seed}" for seed in range(3)),
    *(f"recovery_hr_concat_seed{seed}" for seed in range(3)),
]
FLAGSHIP = "recovery_fixed_novae_flagship_seed10"


def full_mean(payload: dict, metric: str, mode: str = "full") -> float:
    value = payload["image_modes"][mode]["summary"][metric]["mean"]
    return float(value)


def check_residuals(root: Path) -> list[str]:
    failures = []
    for run in RESIDUAL_RUNS:
        gate_path = root / run / "quality_gate.json"
        if not gate_path.exists():
            failures.append(f"{run}: missing quality_gate.json")
            continue
        gate = json.loads(gate_path.read_text())
        if gate.get("passed") is not True:
            failures.append(f"{run}: anchor-improvement gate did not pass")
        correction = gate.get("correction_rms")
        minimum = float(gate.get("minimum_correction_rms", 0.0))
        if correction is None or not math.isfinite(float(correction)) or float(correction) < minimum:
            failures.append(f"{run}: correction RMS is absent/nonfinite/below {minimum:g}")

    return failures


def check_flagship(root: Path) -> list[str]:
    failures = []
    metrics_path = root / FLAGSHIP / "audit_test_metrics.json"
    if not metrics_path.exists():
        failures.append(f"{FLAGSHIP}: missing audit_test_metrics.json")
    else:
        metrics = json.loads(metrics_path.read_text())
        for name in ("pcc", "rmse", "nonzero_auc"):
            value = full_mean(metrics, name)
            if not math.isfinite(value):
                failures.append(f"{FLAGSHIP}: {name} is nonfinite")
        if full_mean(metrics, "nonzero_auc") <= 0.5:
            failures.append(f"{FLAGSHIP}: nonzero AUC is at/below chance")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/checkpoints/recovery_suite"))
    parser.add_argument(
        "--mode", choices=("all", "residuals", "flagship"), default="all",
        help="Select which independent recovery branch must pass.",
    )
    args = parser.parse_args()
    failures = []
    if args.mode in {"all", "residuals"}:
        failures.extend(check_residuals(args.root))
    if args.mode in {"all", "flagship"}:
        failures.extend(check_flagship(args.root))
    if failures:
        print("RECOVERY GATE: BLOCKED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"RECOVERY GATE ({args.mode}): PASS")
    if args.mode == "flagship":
        print("The clean FM flagship is finite and noncollapsed; FM component ablations may proceed.")
    elif args.mode == "residuals":
        print("The direct residual branch beat its harmonic anchors.")
    else:
        print("Both independent recovery branches passed.")


if __name__ == "__main__":
    main()
