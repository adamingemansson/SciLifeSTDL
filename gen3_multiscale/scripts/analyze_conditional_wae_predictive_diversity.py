#!/usr/bin/env python3
"""Audit whether conditional-WAE prior draws are diverse and useful.

This is a checkpoint-only, H&E-only whole-slide diagnostic.  It never exposes
target GEX to the predictor and never modifies training state.  The report
separates four questions that a mean prediction alone cannot answer:

1. Are independent prior draws effectively identical?
2. If they differ, is the variation large relative to biological variation?
3. Does the stochastic mean correct deterministic point-prediction errors?
4. How many draws are needed before the ensemble mean has converged?

Large expression matrices are accumulated online.  Exact final per-gene
statistics are computed in bounded gene chunks; pairwise-draw and ensemble
convergence statistics use a deterministic, explicitly reported subsample of
spot-gene values so that 64 draws remain practical on full slides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae.contract import (
    static_audit_conditional_wae_config,
)
from gen3_multiscale.conditional_wae.data import build_conditional_wae_example
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import load_configured_gene_panels
from gen3_multiscale.evaluation.metrics import resolve_gene_panels
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_wae import (
    _build_model,
    _manifest,
    _stable_seed,
    _verify_resume,
)


DEFAULT_MAP_GENES = ("COL1A1", "DCN", "KRT8", "TOP2A", "FABP1", "IGKC")


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _pearson_1d(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).reshape(-1)
    right = np.asarray(right).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("Pearson inputs must be matching vectors with at least two values")
    # Sufficient statistics avoid two full float64 copies for a 5k x 17k
    # whole-slide expression field.
    n = left.size
    sum_left = sum_right = sum_left2 = sum_right2 = sum_cross = 0.0
    for start in range(0, n, 2_000_000):
        end = min(start + 2_000_000, n)
        x = left[start:end].astype(np.float64, copy=False)
        y = right[start:end].astype(np.float64, copy=False)
        sum_left += float(np.sum(x, dtype=np.float64))
        sum_right += float(np.sum(y, dtype=np.float64))
        sum_left2 += float(np.dot(x, x))
        sum_right2 += float(np.dot(y, y))
        sum_cross += float(np.dot(x, y))
    covariance = sum_cross - sum_left * sum_right / n
    left_ss = max(sum_left2 - sum_left * sum_left / n, 0.0)
    right_ss = max(sum_right2 - sum_right * sum_right / n, 0.0)
    denominator = math.sqrt(left_ss * right_ss)
    return float(covariance / denominator) if denominator > 1e-12 else float("nan")


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("RMSE inputs must be matching non-empty arrays")
    left = left.reshape(-1)
    right = right.reshape(-1)
    squared_error = 0.0
    for start in range(0, left.size, 2_000_000):
        end = min(start + 2_000_000, left.size)
        difference = (
            left[start:end].astype(np.float64, copy=False)
            - right[start:end].astype(np.float64, copy=False)
        )
        squared_error += float(np.dot(difference, difference))
    return math.sqrt(squared_error / left.size)


def _parse_positive_ints(raw: str, *, maximum: int) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] < 1:
        raise ValueError("ensemble sizes must be positive comma-separated integers")
    values = [value for value in values if value <= maximum]
    if maximum not in values:
        values.append(maximum)
    return sorted(values)


def _column_diagnostics(
    point: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Exact per-gene diagnostics without full-panel float64 copies."""
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
        point, predictive_mean, predictive_std, target,
    ))
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("diagnostic arrays must be matching [spots, genes] matrices")
    n_genes = arrays[0].shape[1]
    result = {
        name: np.full(n_genes, np.nan, dtype=np.float64)
        for name in (
            "point_pcc", "predictive_mean_pcc", "mean_shift_residual_pcc",
            "point_rmse", "predictive_mean_rmse", "predictive_std_mean",
            "target_std", "point_std", "predictive_mean_std",
        )
    }
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        p, m, s, y = (array[:, start:end].astype(np.float64) for array in arrays)
        residual = y - p
        shift = m - p

        def per_column_pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            ac = a - a.mean(0, keepdims=True)
            bc = b - b.mean(0, keepdims=True)
            denominator = np.sqrt(np.sum(ac * ac, axis=0) * np.sum(bc * bc, axis=0))
            out = np.full(a.shape[1], np.nan, dtype=np.float64)
            valid = denominator > 1e-12
            out[valid] = np.sum(ac[:, valid] * bc[:, valid], axis=0) / denominator[valid]
            return np.clip(out, -1.0, 1.0)

        result["point_pcc"][start:end] = per_column_pcc(p, y)
        result["predictive_mean_pcc"][start:end] = per_column_pcc(m, y)
        result["mean_shift_residual_pcc"][start:end] = per_column_pcc(shift, residual)
        result["point_rmse"][start:end] = np.sqrt(np.mean(np.square(p - y), axis=0))
        result["predictive_mean_rmse"][start:end] = np.sqrt(np.mean(np.square(m - y), axis=0))
        result["predictive_std_mean"][start:end] = np.mean(s, axis=0)
        result["target_std"][start:end] = np.std(y, axis=0)
        result["point_std"][start:end] = np.std(p, axis=0)
        result["predictive_mean_std"][start:end] = np.std(m, axis=0)
    return result


def _panel_summary(
    diagnostics: dict[str, np.ndarray],
    point: np.ndarray,
    predictive_mean: np.ndarray,
    target: np.ndarray,
    panel_indices: dict[str, np.ndarray],
) -> dict[str, dict[str, float | None]]:
    panels = {"all_genes": np.arange(target.shape[1], dtype=np.int64), **panel_indices}
    output = {}
    for name, indices in panels.items():
        indices = np.asarray(indices, dtype=np.int64)
        point_pcc = diagnostics["point_pcc"][indices]
        mean_pcc = diagnostics["predictive_mean_pcc"][indices]
        alignment = diagnostics["mean_shift_residual_pcc"][indices]
        output[name] = {
            "deterministic_mean_gene_pcc": _finite(np.nanmean(point_pcc)),
            "predictive_mean_gene_pcc": _finite(np.nanmean(mean_pcc)),
            "predictive_minus_deterministic_pcc": _finite(
                np.nanmean(mean_pcc) - np.nanmean(point_pcc)
            ),
            "deterministic_rmse": _finite(_rmse(point[:, indices], target[:, indices])),
            "predictive_mean_rmse": _finite(
                _rmse(predictive_mean[:, indices], target[:, indices])
            ),
            "deterministic_minus_predictive_rmse": _finite(
                _rmse(point[:, indices], target[:, indices])
                - _rmse(predictive_mean[:, indices], target[:, indices])
            ),
            "mean_predictive_std": _finite(
                np.nanmean(diagnostics["predictive_std_mean"][indices])
            ),
            "mean_shift_residual_pcc": _finite(np.nanmean(alignment)),
        }
    return output


def _draw_pair_records(draw_values: np.ndarray) -> list[dict]:
    records = []
    for left in range(draw_values.shape[0]):
        for right in range(left + 1, draw_values.shape[0]):
            records.append({
                "left_draw": left + 1,
                "right_draw": right + 1,
                "sampled_value_pcc": _finite(_pearson_1d(draw_values[left], draw_values[right])),
                "sampled_value_rmse": _finite(_rmse(draw_values[left], draw_values[right])),
            })
    return records


def _write_tsv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _plot_gene_maps(
    output_dir: Path,
    *,
    arm: str,
    sample_id: str,
    coords: np.ndarray,
    gene_names: list[str],
    gene_indices: list[int],
    target: np.ndarray,
    point: np.ndarray,
    predictive_mean: np.ndarray,
    predictive_std: np.ndarray,
    selected_draws: np.ndarray,
) -> None:
    if not gene_indices:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_size = max(3.0, min(10.0, 28_000.0 / max(coords.shape[0], 1)))
    draw_count = min(selected_draws.shape[0], 4)
    for local_index, gene_index in enumerate(gene_indices):
        gene = gene_names[gene_index]
        values = [
            ("Target", target[:, gene_index]),
            ("Deterministic", point[:, gene_index]),
            ("WAE mean", predictive_mean[:, gene_index]),
            ("WAE std", predictive_std[:, gene_index]),
        ]
        values.extend(
            (f"Draw {draw + 1}", selected_draws[draw, :, local_index])
            for draw in range(draw_count)
        )
        figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        vmax_expression = float(np.quantile(np.concatenate([
            target[:, gene_index], predictive_mean[:, gene_index],
        ]), 0.995))
        vmax_expression = max(vmax_expression, 1e-6)
        for axis, (title, field) in zip(axes.reshape(-1), values):
            vmax = float(np.quantile(field, 0.995)) if title == "WAE std" else vmax_expression
            scatter = axis.scatter(
                coords[:, 0], coords[:, 1], c=field, s=marker_size,
                cmap="magma" if title == "WAE std" else "viridis",
                vmin=0.0, vmax=max(vmax, 1e-6), linewidths=0,
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.axis("off")
            figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.02)
        for axis in axes.reshape(-1)[len(values):]:
            axis.axis("off")
        figure.suptitle(f"{arm} | {sample_id} | {gene} | prior-draw diversity")
        figure.savefig(output_dir / f"{sample_id}__{gene}.png", dpi=160)
        plt.close(figure)


def _load_bundle(
    config_path: str,
    checkpoint_dir: str,
    *,
    split: str,
    device_name: str,
    use_best: bool,
    allow_code_drift: bool,
):
    config = resolved_config(config_path)
    static_audit_conditional_wae_config(config)
    if config["model"]["task"] != "he_to_st" or config["model"]["include_observed_gex"]:
        raise ValueError("predictive-diversity analysis requires H&E-only he_to_st inference")
    dataset_manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    split_ids = list(dataset_manifest[f"{split}_sample_ids"])
    samples, preflight = load_and_preflight_samples(
        OmegaConf.create(config), dataset_manifest, split_ids,
        expected_tile_encoder_provenance(config),
    )
    requested = Path(checkpoint_dir) / "best" if use_best else Path(checkpoint_dir)
    old_manifest = checkpoint_module.load_checkpoint_run_manifest(requested)
    if old_manifest is None:
        raise ValueError(f"{requested}: no bundle-bound run manifest")
    current_manifest = _manifest(config, dataset_manifest, preflight)
    old_cache = old_manifest.get("cache_content_by_sample") or {}
    for sample_id, identity in preflight["cache_content_by_sample"].items():
        if sample_id in old_cache and old_cache[sample_id] != identity:
            raise ValueError(f"{sample_id}: cache content differs from checkpoint training")
    current_manifest["cache_content_fingerprint"] = old_manifest["cache_content_fingerprint"]
    current_manifest["cache_content_by_sample"] = old_cache
    _verify_resume(old_manifest, current_manifest, allow_code_drift=allow_code_drift)
    gene_names = list(dataset_manifest["gene_panel"])
    checkpoint_module.verify_gene_names(requested, gene_names)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(gene_names), gene_names=gene_names).to(device)
    checkpoint_module.load_trainable_state(
        model, requested, reconstructed_buffer_names={"per_gene_scale"},
    )
    model.eval()
    if not bool(getattr(model, "has_latent_model", True)):
        raise ValueError("checkpoint has no latent model; there are no WAE draws to diagnose")
    if model.distributional_head is not None:
        raise ValueError("predictive-diversity analysis currently requires gaussian_mse")
    return config, dataset_manifest, split_ids, samples, requested, model, device, gene_names


@torch.no_grad()
def analyze_slide(
    model,
    sample,
    *,
    gene_names: list[str],
    panels: dict[str, list[str]],
    arm: str,
    organ: str,
    n_draws: int,
    ensemble_sizes: list[int],
    chunk_size: int,
    pairwise_max_values: int,
    seed: int,
    map_genes: list[str],
    map_dir: Path,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    inputs, target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    context = model.image_conditioner(inputs)
    n_rows = int(context.shape[0])
    # Use the model's public inference methods.  In particular, residual WAE
    # arms symmetrize *after* nonlinear structured refinement; manually doing
    # decode() then refine_prediction() would measure a different model.
    point = model.predict_point_from_context(context, inputs)
    if bool(getattr(model, "has_specialist_latent_head", False)):
        prior_center = model._prior_mean(context)
        point = model.decode_latent_from_context(prior_center, context, inputs)

    total_values = int(point.numel())
    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(
        total_values, size=min(pairwise_max_values, total_values), replace=False,
    )).astype(np.int64)
    sampled_index_tensor = torch.as_tensor(sampled_indices, device=point.device)
    target_flat = np.asarray(target, dtype=np.float32).reshape(-1)
    sampled_target = target_flat[sampled_indices]
    sampled_point = point.reshape(-1).index_select(0, sampled_index_tensor).cpu().numpy()
    draw_values = np.empty((n_draws, sampled_indices.size), dtype=np.float32)

    selected_gene_indices = [gene_names.index(gene) for gene in map_genes if gene in gene_names]
    selected_draws = np.empty(
        (min(n_draws, 4), n_rows, len(selected_gene_indices)), dtype=np.float32,
    )
    predictive_sum = torch.zeros_like(point)
    predictive_sum_squared = torch.zeros_like(point)
    generator = torch.Generator(device=context.device).manual_seed(seed)
    ensemble_records = []
    draw_records = []
    residual_sample = sampled_target - sampled_point
    antithetic = getattr(model, "latent_residual_mode", "free") == "antithetic_zero_mean"
    prior_mean = model._prior_mean(context) if antithetic else None
    mirrored_z = None
    for draw_index in range(n_draws):
        # Reproduce sample_predictive_distribution's canonical ordering:
        # z, its mirror around the prior mean, then (for odd counts) the prior
        # center.  Every even ensemble size therefore has the exact same
        # finite-sample mean used by ordinary residual-WAE evaluation.
        if antithetic and n_draws % 2 == 1 and draw_index == n_draws - 1:
            z = prior_mean
        elif antithetic and draw_index % 2 == 1:
            if mirrored_z is None:
                raise RuntimeError("antithetic mirror was not prepared")
            z = mirrored_z
            mirrored_z = None
        else:
            z = model.sample_inference_latent(context, generator=generator)
            if antithetic:
                mirrored_z = 2.0 * prior_mean - z
        complete = model.decode_latent_from_context(z, context, inputs)
        predictive_sum.add_(complete)
        predictive_sum_squared.add_(complete.square())
        sampled = complete.reshape(-1).index_select(0, sampled_index_tensor).cpu().numpy()
        draw_values[draw_index] = sampled
        if draw_index < selected_draws.shape[0] and selected_gene_indices:
            selected_draws[draw_index] = complete[:, selected_gene_indices].cpu().numpy()
        shift = sampled - sampled_point
        draw_records.append({
            "sample_id": str(sample.sample_id),
            "organ": organ,
            "draw": draw_index + 1,
            "draw_vs_deterministic_pcc": _finite(_pearson_1d(sampled, sampled_point)),
            "draw_vs_deterministic_rmse": _finite(_rmse(sampled, sampled_point)),
            "draw_vs_target_pcc": _finite(_pearson_1d(sampled, sampled_target)),
            "draw_vs_target_rmse": _finite(_rmse(sampled, sampled_target)),
            "stochastic_shift_vs_residual_pcc": _finite(_pearson_1d(shift, residual_sample)),
        })
        count = draw_index + 1
        if count in ensemble_sizes:
            ensemble = draw_values[:count].mean(axis=0)
            ensemble_records.append({
                "sample_id": str(sample.sample_id),
                "organ": organ,
                "n_draws": count,
                "sampled_value_pcc_vs_target": _finite(_pearson_1d(ensemble, sampled_target)),
                "sampled_value_rmse_vs_target": _finite(_rmse(ensemble, sampled_target)),
                "sampled_value_pcc_vs_deterministic": _finite(
                    _pearson_1d(ensemble, sampled_point)
                ),
                "sampled_value_rmse_vs_deterministic": _finite(
                    _rmse(ensemble, sampled_point)
                ),
            })
        print(
            f"predictive diversity: sample={sample.sample_id} draw={count}/{n_draws}",
            flush=True,
        )

    predictive_mean_t = predictive_sum / n_draws
    variance_t = (predictive_sum_squared / n_draws - predictive_mean_t.square()).clamp_min(0)
    predictive_std_t = variance_t.sqrt()
    point_np = point.cpu().numpy().astype(np.float32)
    predictive_mean = predictive_mean_t.cpu().numpy().astype(np.float32)
    predictive_std = predictive_std_t.cpu().numpy().astype(np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    diagnostics = _column_diagnostics(point_np, predictive_mean, predictive_std, target_np)
    panel_indices, _ = resolve_gene_panels(gene_names, panels) if panels else ({}, {})
    panel_metrics = _panel_summary(
        diagnostics, point_np, predictive_mean, target_np, panel_indices,
    )
    pair_records = _draw_pair_records(draw_values)
    for record in pair_records:
        record.update(sample_id=str(sample.sample_id), organ=organ)

    pair_pcc = np.asarray([
        row["sampled_value_pcc"] for row in pair_records
        if row["sampled_value_pcc"] is not None
    ], dtype=np.float64)
    pair_rmse = np.asarray([row["sampled_value_rmse"] for row in pair_records])
    final_sampled_mean = draw_values.mean(0)
    uncertainty_sample = predictive_std.reshape(-1)[sampled_indices]
    final_abs_error_sample = np.abs(sampled_target - final_sampled_mean)
    residual = target_np - point_np
    shift = predictive_mean - point_np
    residual_alignment = _pearson_1d(shift, residual)
    deterministic_rmse = _rmse(point_np, target_np)
    mean_rmse = _rmse(predictive_mean, target_np)
    target_sd = float(np.std(target_np))
    summary = {
        "sample_id": str(sample.sample_id),
        "organ": organ,
        "n_spots": n_rows,
        "n_genes": len(gene_names),
        "n_draws": n_draws,
        "pairwise_subsample_values": int(sampled_indices.size),
        "pairwise_draw_pcc_mean": _finite(np.mean(pair_pcc)),
        "pairwise_draw_pcc_min": _finite(np.min(pair_pcc)),
        "pairwise_draw_rmse_mean": _finite(np.mean(pair_rmse)),
        "exact_pairwise_draw_rmse": _finite(math.sqrt(
            2.0 * n_draws / (n_draws - 1.0) * float(torch.mean(variance_t))
        )),
        "predictive_std_mean": _finite(float(torch.mean(predictive_std_t))),
        "predictive_std_q50_sampled": _finite(np.quantile(uncertainty_sample, 0.50)),
        "predictive_std_q90_sampled": _finite(np.quantile(uncertainty_sample, 0.90)),
        "predictive_std_to_target_std_ratio": _finite(
            float(torch.mean(predictive_std_t)) / max(target_sd, 1e-12)
        ),
        "uncertainty_abs_error_pcc_sampled": _finite(
            _pearson_1d(uncertainty_sample, final_abs_error_sample)
        ),
        "deterministic_rmse": _finite(deterministic_rmse),
        "predictive_mean_rmse": _finite(mean_rmse),
        "deterministic_minus_predictive_rmse": _finite(deterministic_rmse - mean_rmse),
        "mean_shift_residual_pcc": _finite(residual_alignment),
        "mean_shift_rmse": _finite(_rmse(predictive_mean, point_np)),
        "panel_metrics": panel_metrics,
    }
    per_gene_rows = []
    for index, gene in enumerate(gene_names):
        per_gene_rows.append({
            "sample_id": str(sample.sample_id),
            "organ": organ,
            "gene": gene,
            **{name: _finite(values[index]) for name, values in diagnostics.items()},
        })
    _plot_gene_maps(
        map_dir, arm=arm, sample_id=str(sample.sample_id),
        coords=np.asarray(sample.full_sample_coords, dtype=np.float32),
        gene_names=gene_names, gene_indices=selected_gene_indices,
        target=target_np, point=point_np, predictive_mean=predictive_mean,
        predictive_std=predictive_std, selected_draws=selected_draws,
    )
    return summary, ensemble_records, draw_records, pair_records, per_gene_rows


def analyze(
    *,
    config_path: str,
    checkpoint_dir: str,
    output_root: str,
    split: str,
    device_name: str,
    use_best: bool,
    allow_code_drift: bool,
    n_draws: int,
    ensemble_sizes: list[int],
    chunk_size: int,
    pairwise_max_values: int,
    seed: int,
    map_genes: list[str],
) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if n_draws < 2:
        raise ValueError("n_draws must be at least two")
    if chunk_size < 1 or pairwise_max_values < 100:
        raise ValueError("chunk_size must be positive and pairwise_max_values at least 100")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "maps").mkdir()
    (
        config, manifest, split_ids, samples, requested_checkpoint,
        model, _device, gene_names,
    ) = _load_bundle(
        config_path, checkpoint_dir, split=split, device_name=device_name,
        use_best=use_best, allow_code_drift=allow_code_drift,
    )
    panels = load_configured_gene_panels(config, manifest)
    arm = str(config["model"].get("arm") or Path(config_path).stem)
    slide_rows = []
    convergence_rows = []
    draw_rows = []
    pair_rows = []
    per_gene_rows = []
    for slide_index, sample_id in enumerate(split_ids):
        slide_seed = _stable_seed(seed, {
            "sample_id": sample_id,
            "stratum": "whole_slide_predictive_diversity",
            "query_fingerprint": "every_spot_exactly_once",
        })
        result = analyze_slide(
            model, samples[sample_id], gene_names=gene_names, panels=panels,
            arm=arm, organ=str(manifest["samples"][sample_id]["organ"]),
            n_draws=n_draws, ensemble_sizes=ensemble_sizes,
            chunk_size=chunk_size, pairwise_max_values=pairwise_max_values,
            seed=slide_seed, map_genes=map_genes, map_dir=root / "maps",
        )
        slide, convergence, draws, pairs, genes = result
        slide_rows.append(slide)
        convergence_rows.extend(convergence)
        draw_rows.extend(draws)
        pair_rows.extend(pairs)
        per_gene_rows.extend(genes)
        print(f"completed slide {slide_index + 1}/{len(split_ids)}: {sample_id}", flush=True)

    flat_slide_rows = []
    for row in slide_rows:
        flat = {key: value for key, value in row.items() if key != "panel_metrics"}
        flat_slide_rows.append(flat)
    _write_tsv(root / "per_slide.tsv", flat_slide_rows)
    _write_tsv(root / "ensemble_convergence.tsv", convergence_rows)
    _write_tsv(root / "per_draw.tsv", draw_rows)
    _write_tsv(root / "pairwise_draw_similarity.tsv", pair_rows)
    _write_tsv(root / "per_gene_per_slide.tsv", per_gene_rows)

    scalar_keys = [
        key for key, value in flat_slide_rows[0].items()
        if isinstance(value, (int, float)) and key not in {
            "n_spots", "n_genes", "n_draws", "pairwise_subsample_values",
        }
    ]
    macro = {}
    for key in scalar_keys:
        values = [row[key] for row in flat_slide_rows if row[key] is not None]
        macro[key] = _finite(np.mean(values)) if values else None
    report = {
        "kind": "conditional_wae_whole_slide_predictive_diversity",
        "arm": arm,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "checkpoint": str(requested_checkpoint.resolve()),
        "checkpoint_choice": "best" if use_best else "latest",
        "split": split,
        "n_slides": len(split_ids),
        "n_draws": n_draws,
        "ensemble_sizes": ensemble_sizes,
        "pairwise_max_values_per_slide": pairwise_max_values,
        "target_gex_visible_to_model": False,
        "same_slide_and_spots_across_draws": True,
        "seed": seed,
        "map_genes": map_genes,
        "per_slide": slide_rows,
        "slide_macro_mean": macro,
        "interpretation": {
            "identical_draws": "pairwise PCC near 1 and pairwise RMSE/predictive std near 0",
            "unhelpful_noise": "nonzero draw variance but stochastic shift does not align with deterministic residual and ensemble does not improve target metrics",
            "useful_latent": "mean shift aligns positively with deterministic residual and ensemble improves PCC/RMSE",
            "calibrated_uncertainty": "predictive std positively tracks absolute prediction error",
        },
        "files": {
            "per_slide": "per_slide.tsv",
            "ensemble_convergence": "ensemble_convergence.tsv",
            "per_draw": "per_draw.tsv",
            "pairwise_draw_similarity": "pairwise_draw_similarity.tsv",
            "per_gene_per_slide": "per_gene_per_slide.tsv",
            "maps": "maps/",
        },
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (root / "COMPLETE").write_text("PASS\n")
    print(json.dumps({"report": str(root / "report.json"), "macro": macro}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-draws", type=int, default=64)
    parser.add_argument("--ensemble-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--pairwise-max-values", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--genes", default=",".join(DEFAULT_MAP_GENES))
    parser.add_argument("--latest", action="store_true", help="use latest, not best, checkpoint")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    if args.n_draws < 2:
        parser.error("--n-draws must be at least two")
    analyze(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_root=args.output_root,
        split=args.split,
        device_name=args.device,
        use_best=not args.latest,
        allow_code_drift=args.allow_code_drift,
        n_draws=args.n_draws,
        ensemble_sizes=_parse_positive_ints(args.ensemble_sizes, maximum=args.n_draws),
        chunk_size=args.chunk_size,
        pairwise_max_values=args.pairwise_max_values,
        seed=args.seed,
        map_genes=[gene.strip() for gene in args.genes.split(",") if gene.strip()],
    )


if __name__ == "__main__":
    main()
