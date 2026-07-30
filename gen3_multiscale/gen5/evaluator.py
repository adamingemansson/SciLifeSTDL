"""Thin paired evaluator -- GEN5_CONTRACT.md sections 7 (gate 10), 9.

Reuses `evaluation/metrics.py` primitives directly (`pearson_per_gene`,
`rmse`, `aggregate_patient_metrics`) rather than reimplementing metric
math. Takes ALREADY-COMPUTED prediction arrays for each method being
compared (the caller is responsible for actually running each model --
this module has no model-construction logic of its own) -- a full CLI
matching `evaluation/gen3_evaluator.py`'s own is explicit future work
(GEN5_CONTRACT.md section 9), since it needs a real trained checkpoint to
be meaningful.
"""
from __future__ import annotations

import numpy as np

from gen3_multiscale.evaluation.metrics import aggregate_patient_metrics, pearson_per_gene, rmse


def _per_item_metrics(predictions: list[np.ndarray], true_expression: list[np.ndarray]) -> list[dict[str, float]]:
    per_item = []
    for pred, true in zip(predictions, true_expression):
        per_gene = pearson_per_gene(np.asarray(pred, dtype=np.float32), np.asarray(true, dtype=np.float32))
        per_item.append({
            "pcc": float(np.nanmean(per_gene)),
            "rmse": float(rmse(np.asarray(pred, dtype=np.float32), np.asarray(true, dtype=np.float32))),
        })
    return per_item


def compare_gen5_arm(
    *,
    gen5_predictions: list[np.ndarray],
    true_expression: list[np.ndarray],
    patient_ids: list[str],
    gen4_predictions: list[np.ndarray] | None = None,
    deterministic_predictions: list[np.ndarray] | None = None,
    mean_baseline_predictions: list[np.ndarray] | None = None,
    nearest_neighbor_predictions: list[np.ndarray] | None = None,
    harmonic_predictions: list[np.ndarray] | None = None,
) -> dict:
    """Every `*_predictions` list must be item-aligned with
    `true_expression`/`patient_ids` (same length, same order) -- the
    caller's responsibility, matching every other paired-comparison
    function already in this codebase
    (`evaluation/gen3_evaluator.py`'s own baseline comparisons)."""
    n_items = len(true_expression)
    if len(patient_ids) != n_items or len(gen5_predictions) != n_items:
        raise ValueError("gen5_predictions/true_expression/patient_ids must all have the same length")

    methods = {"gen5": gen5_predictions}
    for name, preds in (
        ("gen4", gen4_predictions), ("deterministic_conditioner", deterministic_predictions),
        ("mean_baseline", mean_baseline_predictions), ("nearest_neighbor_baseline", nearest_neighbor_predictions),
        ("harmonic_baseline", harmonic_predictions),
    ):
        if preds is not None:
            if len(preds) != n_items:
                raise ValueError(f"{name} predictions has {len(preds)} items, expected {n_items}")
            methods[name] = preds

    report: dict = {"n_items": n_items, "per_method": {}}
    per_method_item_metrics: dict[str, list[dict[str, float]]] = {}
    for name, preds in methods.items():
        per_item = _per_item_metrics(preds, true_expression)
        per_method_item_metrics[name] = per_item
        report["per_method"][name] = {
            "patient_aggregated": aggregate_patient_metrics(per_item, patient_ids),
            "mean_pcc": float(np.mean([m["pcc"] for m in per_item])),
            "mean_rmse": float(np.mean([m["rmse"] for m in per_item])),
        }

    gen5_items = per_method_item_metrics["gen5"]
    report["paired_delta_vs"] = {}
    for name, items in per_method_item_metrics.items():
        if name == "gen5":
            continue
        pcc_deltas = [g["pcc"] - o["pcc"] for g, o in zip(gen5_items, items)]
        rmse_deltas = [g["rmse"] - o["rmse"] for g, o in zip(gen5_items, items)]
        report["paired_delta_vs"][name] = {
            "mean_pcc_delta": float(np.mean(pcc_deltas)), "mean_rmse_delta": float(np.mean(rmse_deltas)),
        }
    return report
