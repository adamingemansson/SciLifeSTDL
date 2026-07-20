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


def check(root: Path) -> list[str]:
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
        modes = metrics.get("image_modes", {})
        if "shuffled" not in modes:
            failures.append(f"{FLAGSHIP}: shuffled-image diagnostic is missing")
        else:
            delta = abs(full_mean(metrics, "rmse", "full") - full_mean(metrics, "rmse", "shuffled"))
            if delta <= 1e-6:
                failures.append(f"{FLAGSHIP}: full and shuffled H&E RMSE are identical")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/checkpoints/recovery_suite"))
    args = parser.parse_args()
    failures = check(args.root)
    if failures:
        print("RECOVERY GATE: BLOCKED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("RECOVERY GATE: PASS")
    print("The residual models beat their anchors and the current-audit flagship is noncollapsed.")


if __name__ == "__main__":
    main()

