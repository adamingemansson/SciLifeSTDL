#!/usr/bin/env python3
"""Gen6-B stability/overfitting audit, Phase 1.

Read-only inspection of an ALREADY-COMPLETED (or in-progress) training
run's `validation_history.json`, retained checkpoint history, and
(optionally) its captured stdout training log. Every number in the
returned report is either read directly from one of those real files or
derived from them by an explicit, documented computation -- there is no
code path here that invents a value for a checkpoint step that was not
actually retained on disk (`list_checkpoint_bundles`, not a fabricated
full step range), and a missing/empty `validation_history.json` raises
rather than silently reporting an empty audit as if it were real.

`--train-log` is optional: `validation_history.json` alone is enough for
the validation-side analysis (validation-total trend, best/final step,
relative degradation, recent slope, retained-checkpoint coverage). The
gradient-norm distribution, non-finite-skip/crash detection, and the
train/validation gap additionally need the captured stdout log, since
`train.py`'s per-step `grad_norm` and train-loss components are only
ever printed (`_log_step`), never persisted to a JSON file.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.early_stopping import EarlyStoppingConfig, early_stopping_status

_TRAIN_LINE_RE = re.compile(r"^\[step (\d+)\] train: (.+)$")
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
_NON_FINITE_RE = re.compile(
    r"\[step (\d+)\] (non-finite total loss|non-finite gradient norm) \((.+?)\)"
)


def _linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Ordinary-least-squares slope of ys against xs; None if fewer than
    two points or xs are degenerate (all identical, zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return numerator / denominator


def _parse_training_log(log_path: Path) -> dict:
    """Extract per-step train `total`/`grad_norm` and any recorded
    non-finite-loss/gradient crash from a real captured stdout log."""
    train_steps: list[int] = []
    train_totals: list[float] = []
    grad_norms: list[float] = []
    grad_norm_steps: list[int] = []
    non_finite_events: list[dict] = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = _NON_FINITE_RE.search(line)
        if match:
            non_finite_events.append({
                "step": int(match.group(1)), "kind": match.group(2), "detail": match.group(3),
            })
            continue
        match = _TRAIN_LINE_RE.match(line)
        if not match:
            continue
        step = int(match.group(1))
        fields = {k: float(v) for k, v in _KV_RE.findall(match.group(2))}
        if "total" in fields:
            train_steps.append(step)
            train_totals.append(fields["total"])
        if "grad_norm" in fields:
            grad_norm_steps.append(step)
            grad_norms.append(fields["grad_norm"])
    return {
        "train_steps": train_steps, "train_totals": train_totals,
        "grad_norm_steps": grad_norm_steps, "grad_norms": grad_norms,
        "non_finite_events": non_finite_events,
    }


def _gradient_norm_summary(grad_norm_steps: list[int], grad_norms: list[float], *, spike_multiplier: float) -> dict:
    if not grad_norms:
        return {"n": 0}
    median = statistics.median(grad_norms)
    spike_threshold = median * spike_multiplier if median > 0 else max(grad_norms)
    spikes = [
        {"step": step, "grad_norm": value}
        for step, value in zip(grad_norm_steps, grad_norms) if value > spike_threshold
    ]
    return {
        "n": len(grad_norms),
        "min": min(grad_norms), "max": max(grad_norms),
        "mean": statistics.fmean(grad_norms), "median": median,
        "stdev": statistics.pstdev(grad_norms) if len(grad_norms) > 1 else 0.0,
        "spike_multiplier": spike_multiplier, "spike_threshold": spike_threshold,
        "n_spikes": len(spikes), "spike_fraction": len(spikes) / len(grad_norms),
        "spikes": spikes,
    }


def _train_validation_gap(
    validation_steps: list[int], validation_totals: list[float],
    train_steps: list[int], train_totals: list[float], *, moving_average_window: int,
) -> list[dict]:
    """At each validation step, the trailing moving average of every
    parsed train-loss point at or before that step, and the gap
    (train_moving_average - validation_total). A gap that shrinks toward
    zero or goes negative and then WIDENS again over time (validation
    loss pulling away from an ever-improving train loss) is the classic
    overfitting signature this pairing is meant to expose."""
    gap_series = []
    for v_step, v_total in zip(validation_steps, validation_totals):
        eligible = [t for s, t in zip(train_steps, train_totals) if s <= v_step]
        if not eligible:
            continue
        window = eligible[-moving_average_window:]
        train_ma = sum(window) / len(window)
        gap_series.append({
            "validation_step": v_step, "validation_total": v_total,
            "train_moving_average": train_ma, "gap": train_ma - v_total,
            "n_train_points_in_window": len(window),
        })
    return gap_series


def audit_gen6b_training_stability(
    checkpoint_dir: str | Path, *,
    train_log_path: str | Path | None = None,
    recent_window: int = 8,
    moving_average_window: int = 20,
    spike_multiplier: float = 5.0,
    early_stopping_patience_validations: int = 8,
    early_stopping_min_delta: float = 0.0001,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    history_path = checkpoint_dir / "validation_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"{history_path} does not exist -- this checkpoint_dir has no recorded validation "
            "history to audit; refusing to fabricate a stability verdict without it"
        )
    validation_history = json.loads(history_path.read_text())
    if not validation_history:
        raise ValueError(f"{history_path} exists but records zero validations")
    steps = [int(entry["step"]) for entry in validation_history]
    totals = [float(entry["total"]) for entry in validation_history]
    if steps != sorted(steps):
        raise ValueError(f"{history_path}: validation steps are not monotonically ascending: {steps}")

    best_index = min(range(len(totals)), key=lambda i: totals[i])
    best_step, best_value = steps[best_index], totals[best_index]
    final_step, final_value = steps[-1], totals[-1]
    relative_degradation = (
        (final_value - best_value) / best_value if best_value not in (0, None) else float("nan")
    )
    recent_steps, recent_totals = steps[-recent_window:], totals[-recent_window:]
    recent_slope = _linear_slope([float(s) for s in recent_steps], recent_totals)

    retained_steps = sorted(checkpoint_module.list_checkpoint_history(checkpoint_dir))
    has_best_bundle = (checkpoint_dir / "best" / "latest_bundle.json").is_file()
    validation_by_step = dict(zip(steps, totals))
    step_spacing = steps[1] - steps[0] if len(steps) > 1 else max(best_step, 1)
    near_minimum_window = max(step_spacing * 3, 1)
    retained_near_minimum = [
        {
            "step": step, "validation_total": validation_by_step.get(step),
            "has_validation_entry": step in validation_by_step,
        }
        for step in retained_steps
        if abs(step - best_step) <= near_minimum_window
    ]
    if best_step not in retained_steps and not has_best_bundle:
        best_checkpoint_availability = "NOT_RETAINED_ON_DISK"
    elif has_best_bundle:
        best_checkpoint_availability = "best_bundle_present"
    else:
        best_checkpoint_availability = "present_in_history_only"

    es_config = EarlyStoppingConfig(
        monitor="validation_total", mode="min",
        patience_validations=early_stopping_patience_validations, min_delta=early_stopping_min_delta,
    )
    retrospective_early_stopping = early_stopping_status(validation_history, es_config)

    log_summary = None
    gradient_norm_summary = {"n": 0}
    train_validation_gap = []
    non_finite_events = []
    if train_log_path is not None:
        log_path = Path(train_log_path)
        if not log_path.is_file():
            raise FileNotFoundError(f"--train-log {log_path} does not exist")
        parsed = _parse_training_log(log_path)
        non_finite_events = parsed["non_finite_events"]
        gradient_norm_summary = _gradient_norm_summary(
            parsed["grad_norm_steps"], parsed["grad_norms"], spike_multiplier=spike_multiplier,
        )
        train_validation_gap = _train_validation_gap(
            steps, totals, parsed["train_steps"], parsed["train_totals"],
            moving_average_window=moving_average_window,
        )
        log_summary = {
            "n_train_lines_parsed": len(parsed["train_steps"]),
            "n_grad_norm_values_parsed": len(parsed["grad_norms"]),
            "first_step": parsed["train_steps"][0] if parsed["train_steps"] else None,
            "last_step": parsed["train_steps"][-1] if parsed["train_steps"] else None,
        }

    # Explicit, documented verdict logic -- never a bare "looks fine".
    # A crash or a real non-finite event always wins as "unstable". A
    # high gradient-norm-spike fraction (default: more than 5% of parsed
    # steps exceed 5x the run's own median grad_norm) is also unstable.
    # Otherwise, retrospectively applying the SAME early-stopping rule
    # this audit's Phase 2 implements (default patience=8, min_delta=
    # 1e-4) to the full validation history: triggering it with real
    # degradation (not just a flat plateau) is "mildly_overfit"; a flat
    # trigger is "plateaued"; not triggering it at all, with a
    # non-positive recent slope, is "stable".
    if non_finite_events or gradient_norm_summary.get("spike_fraction", 0.0) > 0.05:
        verdict = "unstable"
    elif retrospective_early_stopping["should_stop"] and relative_degradation > 0.01:
        verdict = "mildly_overfit"
    elif retrospective_early_stopping["should_stop"]:
        verdict = "plateaued"
    elif recent_slope is not None and recent_slope > 0:
        verdict = "plateaued"
    else:
        verdict = "stable"

    return {
        "version": 1,
        "kind": "gen6b_stability_audit_report",
        "checkpoint_dir": str(checkpoint_dir),
        "train_log_path": str(train_log_path) if train_log_path is not None else None,
        "n_validations": len(steps),
        "validation_step_range": [steps[0], steps[-1]],
        "best_step": best_step, "best_validation_total": best_value,
        "final_step": final_step, "final_validation_total": final_value,
        "relative_degradation_final_vs_best": relative_degradation,
        "recent_window": recent_window,
        "recent_validation_slope": recent_slope,
        "retained_checkpoint_steps": retained_steps,
        "has_best_bundle": has_best_bundle,
        "best_checkpoint_availability": best_checkpoint_availability,
        "retained_checkpoints_near_minimum": retained_near_minimum,
        "retrospective_early_stopping": {
            **retrospective_early_stopping,
            "patience_validations": early_stopping_patience_validations,
            "min_delta": early_stopping_min_delta,
        },
        "log_summary": log_summary,
        "gradient_norm_summary": gradient_norm_summary,
        "non_finite_events": non_finite_events,
        "train_validation_gap": train_validation_gap,
        "verdict": verdict,
        "verdict_rationale": {
            "unstable": "non-finite loss/gradient event(s) in the log, or >5% of parsed grad_norm "
                        "values exceed 5x the run's own median",
            "mildly_overfit": "retrospective early stopping (patience/min_delta as configured) would "
                              "have triggered AND final validation is >1% worse than the best",
            "plateaued": "retrospective early stopping would have triggered without real degradation, "
                        "or the recent validation slope is non-negative (not improving)",
            "stable": "none of the above -- still improving or not yet at a plateau/instability signal",
        }[verdict],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-log", default=None, help="Captured stdout log (optional)")
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--moving-average-window", type=int, default=20)
    parser.add_argument("--spike-multiplier", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience-validations", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    parser.add_argument("--output", default=None, help="Write the JSON report here as well as stdout")
    args = parser.parse_args()
    report = audit_gen6b_training_stability(
        args.checkpoint_dir, train_log_path=args.train_log,
        recent_window=args.recent_window, moving_average_window=args.moving_average_window,
        spike_multiplier=args.spike_multiplier,
        early_stopping_patience_validations=args.early_stopping_patience_validations,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
