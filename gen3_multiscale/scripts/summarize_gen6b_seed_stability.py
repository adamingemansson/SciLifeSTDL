#!/usr/bin/env python3
"""Cross-seed summary for the Gen6-B stability audit, Phase 3 final
evaluation.

Reads three REAL `gen3_evaluator.py` JSON reports (one per seed,
produced by `python -m gen3_multiscale.evaluation.gen3_evaluator` against
each seed's own best checkpoint) plus each seed's `validation_history.json`
for best-step/wall-clock context, and reports mean +/- SD across seeds
for all-gene, CCRCC-50, HVG-50 and HVG-200 PCC/RMSE/AUC, paired deltas
against the mean/nearest-neighbour/harmonic baselines, and whether the
seed-induced spread is larger than the spread previously observed among
Gen6 B-J. Those Gen6 B-J reference numbers are supplied by the caller
(`--reference-arm-pcc NAME=VALUE`), never hardcoded here -- this
project's own reported numbers have already changed once mid-session,
and baking a stale comparison point into code would silently go wrong
the next time they do.

Refuses to compare seeds that were not evaluated on the same number of
items (`n_items` must match across all three reports) -- the entire
point of "matched" seeds is a like-for-like comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

_DEFAULT_PANELS = ("hvg_50", "hvg_200", "CCRCC_var_50genes")
_DEFAULT_METRICS = ("pcc", "rmse", "nonzero_auc")
_BASELINES = ("mean", "nearest_neighbor", "harmonic")


def _mean_sd(values: list) -> dict:
    finite = [float(v) for v in values if v is not None and v == v]  # drop None and NaN
    return {
        "n": len(finite), "n_total": len(values),
        "mean": statistics.fmean(finite) if finite else None,
        "sd": statistics.pstdev(finite) if len(finite) > 1 else (0.0 if finite else None),
        "values": values,
    }


def _all_gene_metric(report: dict, metric: str) -> float | None:
    return (
        report.get("per_arm_patient_aggregated_metrics", {})
        .get("model", {}).get(metric, {}).get("patient_mean")
    )


def _panel_metric(report: dict, panel: str, metric: str) -> float | None:
    panel_block = report.get("per_panel_patient_aggregated_metrics", {}).get(panel)
    if not panel_block:
        return None
    return panel_block.get("model", {}).get(metric, {}).get("patient_mean")


def _paired_delta(report: dict, baseline: str, delta_metric: str) -> float | None:
    deltas = report.get("per_arm_paired_delta_vs_model", {}).get(baseline)
    if not deltas:
        return None
    return deltas.get(delta_metric, {}).get("patient_mean")


def _best_step_and_value(validation_history_path: Path) -> dict:
    if not validation_history_path.is_file():
        return {"best_step": None, "best_validation_total": None, "final_step": None}
    history = json.loads(validation_history_path.read_text())
    if not history:
        return {"best_step": None, "best_validation_total": None, "final_step": None}
    best_entry = min(history, key=lambda e: e["total"])
    return {
        "best_step": int(best_entry["step"]), "best_validation_total": float(best_entry["total"]),
        "final_step": int(history[-1]["step"]),
    }


def _wall_clock_hours_from_mtimes(checkpoint_dir: Path) -> float | None:
    """Best-effort approximation from bundle file modification times --
    NOT authoritative (see this module's docstring); prefer
    `--training-summary` per seed when exact elapsed_seconds is needed."""
    history_dir = checkpoint_dir / "history"
    if not history_dir.is_dir():
        return None
    mtimes = [entry.stat().st_mtime for entry in history_dir.iterdir() if entry.is_dir()]
    if len(mtimes) < 2:
        return None
    return (max(mtimes) - min(mtimes)) / 3600.0


def summarize_gen6b_seed_stability(
    *, evaluation_reports: dict[str, str], checkpoint_dirs: dict[str, str] | None = None,
    training_summaries: dict[str, str] | None = None,
    panels: tuple[str, ...] = _DEFAULT_PANELS, metrics: tuple[str, ...] = _DEFAULT_METRICS,
    reference_arm_pcc: dict[str, float] | None = None,
) -> dict:
    seeds = sorted(evaluation_reports)
    if len(seeds) < 2:
        raise ValueError("need at least two seeds' evaluation reports to summarize spread")
    reports = {seed: json.loads(Path(evaluation_reports[seed]).read_text()) for seed in seeds}

    n_items = {seed: reports[seed].get("n_items") for seed in seeds}
    if len(set(n_items.values())) != 1:
        raise ValueError(
            f"seeds were evaluated on different n_items, not a like-for-like comparison: {n_items}"
        )
    dataset_fingerprints = {
        seed: reports[seed].get("checkpoint_identity", {}).get("dataset_manifest_fingerprint")
        for seed in seeds
    }
    if len(set(dataset_fingerprints.values())) != 1:
        raise ValueError(
            f"seeds do not share the same dataset_manifest_fingerprint, refusing to compare: "
            f"{dataset_fingerprints}"
        )

    all_gene = {
        metric: _mean_sd([_all_gene_metric(reports[s], metric) for s in seeds]) for metric in metrics
    }
    by_panel = {
        panel: {
            metric: _mean_sd([_panel_metric(reports[s], panel, metric) for s in seeds])
            for metric in metrics
        }
        for panel in panels
    }
    delta_metric_by_metric = {"pcc": "pcc_delta", "rmse": "rmse_delta"}
    paired_deltas = {
        baseline: {
            metric: _mean_sd([_paired_delta(reports[s], baseline, delta_metric) for s in seeds])
            for metric, delta_metric in delta_metric_by_metric.items()
        }
        for baseline in _BASELINES
    }

    run_context: dict[str, dict] = {}
    for seed in seeds:
        checkpoint_dir = Path(checkpoint_dirs[seed]) if checkpoint_dirs and seed in checkpoint_dirs else None
        context: dict = {"evaluation_report": evaluation_reports[seed]}
        if checkpoint_dir is not None:
            context.update(_best_step_and_value(checkpoint_dir / "validation_history.json"))
            approx_hours = _wall_clock_hours_from_mtimes(checkpoint_dir)
            context["wall_clock_hours_approx_from_bundle_mtimes"] = approx_hours
        if training_summaries and seed in training_summaries:
            summary = json.loads(Path(training_summaries[seed]).read_text())
            context["training_summary"] = summary
            context["elapsed_seconds"] = summary.get("elapsed_seconds")
            context["completion_reason"] = summary.get("completion_reason")
        run_context[seed] = context

    best_steps = [run_context[s].get("best_step") for s in seeds if run_context[s].get("best_step") is not None]
    elapsed = [
        run_context[s].get("elapsed_seconds") for s in seeds
        if run_context[s].get("elapsed_seconds") is not None
    ]
    variability = {
        "best_step": _mean_sd(best_steps) if best_steps else None,
        "elapsed_seconds": _mean_sd(elapsed) if elapsed else None,
    }

    seed_spread_vs_reference = None
    if reference_arm_pcc:
        reference_values = list(reference_arm_pcc.values())
        reference_spread = max(reference_values) - min(reference_values) if len(reference_values) >= 2 else None
        seed_pcc_spread = None
        pcc_values = [v for v in all_gene["pcc"]["values"] if v is not None]
        if len(pcc_values) >= 2:
            seed_pcc_spread = max(pcc_values) - min(pcc_values)
        seed_spread_vs_reference = {
            "reference_arm_pcc": reference_arm_pcc,
            "reference_pcc_spread": reference_spread,
            "seed_pcc_spread": seed_pcc_spread,
            "seed_spread_exceeds_reference_spread": (
                seed_pcc_spread is not None and reference_spread is not None
                and seed_pcc_spread > reference_spread
            ),
        }

    return {
        "version": 1,
        "kind": "gen6b_stability_seed_summary",
        "seeds": seeds,
        "n_items": n_items[seeds[0]],
        "dataset_manifest_fingerprint": dataset_fingerprints[seeds[0]],
        "all_gene_patient_mean_across_seeds": all_gene,
        "per_panel_patient_mean_across_seeds": by_panel,
        "paired_delta_patient_mean_across_seeds": paired_deltas,
        "run_context": run_context,
        "best_step_and_elapsed_seconds_variability": variability,
        "seed_spread_vs_gen6_b_through_j_reference": seed_spread_vs_reference,
    }


def _kv_pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"expected NAME=VALUE, got {raw!r}")
        name, value = raw.split("=", 1)
        result[name] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-report", action="append", required=True, metavar="SEED=PATH",
        help="Repeat for each seed, e.g. --evaluation-report 0=/path/seed0_eval.json",
    )
    parser.add_argument("--checkpoint-dir", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--training-summary", action="append", default=[], metavar="SEED=PATH")
    parser.add_argument("--panel", action="append", default=list(_DEFAULT_PANELS))
    parser.add_argument("--metric", action="append", default=list(_DEFAULT_METRICS))
    parser.add_argument(
        "--reference-arm-pcc", action="append", default=[], metavar="ARM=VALUE",
        help="e.g. --reference-arm-pcc gen6a=0.0587 --reference-arm-pcc gen6b=0.045384",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    reference_arm_pcc = {name: float(value) for name, value in _kv_pairs(args.reference_arm_pcc).items()}
    summary = summarize_gen6b_seed_stability(
        evaluation_reports=_kv_pairs(args.evaluation_report),
        checkpoint_dirs=_kv_pairs(args.checkpoint_dir) or None,
        training_summaries=_kv_pairs(args.training_summary) or None,
        panels=tuple(args.panel), metrics=tuple(args.metric),
        reference_arm_pcc=reference_arm_pcc or None,
    )
    text = json.dumps(summary, indent=2, sort_keys=True, default=str)
    print(text)
    if args.output:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
