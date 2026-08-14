#!/usr/bin/env python3
"""Leakage-safe frozen-feature PCA/MLP benchmark on the project split.

This is the nonlinear companion to ``frozen_feature_ridge_evaluator``. The
image encoder remains frozen, PCA is fit from bounded training-only samples,
and a small MLP learns normalized-log1p expression with MSE. Validation is
used only for checkpoint selection. The final report predicts every held-out
spot exactly once and uses the identical point and structured metric code as
ridge and MK.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.frozen_feature_ridge_evaluator import (
    _atomic_json,
    _aggregate_point_metrics_by_group,
    _candidate_indices,
    _dense_expression,
    _fit_covariance_pca,
    _flatten_structured_panel,
    _load_sample,
    _panel_metrics,
    _positive_gene_scale,
    _stable_indices,
)
from gen3_multiscale.evaluation.metrics import aggregate_patient_metrics, resolve_gene_panels
from gen3_multiscale.evaluation.structured_field_metrics import structured_field_metrics
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.data import spot_feature_cache
from gen3_multiscale.gen4 import uni2_spot_cache


class FrozenFeatureMLP(nn.Module):
    """One-hidden-layer nonlinear probe; the pathology encoder stays frozen."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1:
            raise ValueError("MLP dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _cache_contract(cfg, image_encoder: str, required_ids: list[str]) -> dict:
    if image_encoder == "uni2":
        return uni2_spot_cache.require_uni2_spot_cache_coverage(
            uni2_spot_cache.cfg_cache_root(cfg), required_ids,
        )
    missing = [
        sample_id for sample_id in required_ids
        if not spot_feature_cache._cache_path(cfg, sample_id).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"GigaPath spot-feature cache covers {len(required_ids) - len(missing)}/"
            f"{len(required_ids)} samples; first missing={missing[:15]}"
        )
    return {
        "cache_dir": str(spot_feature_cache._cache_path(cfg, required_ids[0]).parent),
        "required_samples": len(required_ids), "missing_samples": [],
    }


def _fit_pca(
    cfg, manifest: dict, train_ids: list[str], *, image_policy: str,
    spots_per_slide: int, seed: int, components: int, device: str,
):
    feature_sum = feature_cross_product = None
    n_rows = 0
    provenance = {}
    for position, sample_id in enumerate(train_ids):
        sample = _load_sample(cfg, manifest, sample_id)
        candidates = _candidate_indices(sample, image_policy)
        chosen = candidates[_stable_indices(sample_id, len(candidates), spots_per_slide, seed)]
        rows = np.asarray(sample.precomputed_spot_features[chosen], dtype=np.float64)
        if feature_sum is None:
            feature_sum = np.zeros(rows.shape[1], dtype=np.float64)
            feature_cross_product = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
        feature_sum += rows.sum(axis=0)
        feature_cross_product += rows.T @ rows
        n_rows += len(rows)
        provenance[sample_id] = sample.tile_encoder_provenance["spot_features"]
        print(
            f"PCA sampling: {position + 1}/{len(train_ids)} sample={sample_id} rows={len(rows)}",
            flush=True,
        )
    reference = provenance[train_ids[0]]
    mismatch = [name for name, value in provenance.items() if value != reference]
    if mismatch:
        raise ValueError(f"spot-feature provenance differs across samples: {mismatch[:10]}")
    if components > min(n_rows - 1, len(feature_sum)):
        raise ValueError("pca_components exceeds the training-only sampled rank")
    return (
        _fit_covariance_pca(
            feature_sum, feature_cross_product, n_rows, components, device,
        ),
        reference,
        n_rows,
    )


def _sample_rows(sample_id: str, sample, limit: int, seed: int) -> np.ndarray:
    candidates = _candidate_indices(sample, "zero")
    return candidates[_stable_indices(sample_id, len(candidates), limit, seed)]


@torch.no_grad()
def _validation_mse(
    model: nn.Module, pca, cfg, manifest: dict, sample_ids: list[str], *,
    rows_per_slide: int, batch_size: int, seed: int, device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    n_values = 0
    for sample_id in sample_ids:
        sample = _load_sample(cfg, manifest, sample_id)
        chosen = _sample_rows(sample_id, sample, rows_per_slide, seed)
        x = pca.transform(sample.precomputed_spot_features[chosen]).astype(np.float32)
        y = _dense_expression(sample.adata.X)[chosen]
        for start in range(0, len(chosen), batch_size):
            end = min(start + batch_size, len(chosen))
            predicted = model(torch.as_tensor(x[start:end], device=device)).cpu().numpy()
            difference = predicted - y[start:end]
            squared_error += float(np.square(difference, dtype=np.float64).sum())
            n_values += int(difference.size)
    if n_values == 0:
        raise ValueError("validation produced zero values")
    return squared_error / n_values


def _save_checkpoint(
    path: Path, *, model: nn.Module, pca, per_gene_scale: np.ndarray,
    gene_names: list[str], metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "pca_mean": pca.mean_, "pca_components": pca.components_,
        "per_gene_scale": per_gene_scale, "gene_names": gene_names,
        "metadata": metadata,
    }, temporary)
    os.replace(temporary, path)


def evaluate_frozen_feature_mlp(
    *, config_path: str, output: str, image_encoder: str,
    manifest_path: str | None = None, train_gene_panels_path: str | None = None,
    cache_dir: str | None = None, hest_data_dir: str | None = None,
    split: str = "validation", pca_components: int = 256,
    pca_spots_per_slide: int = 128, train_spots_per_slide: int = 2048,
    validation_spots_per_slide: int = 512, hidden_dim: int = 512,
    dropout: float = 0.1, epochs: int = 30, batch_size: int = 64,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    patience_epochs: int = 5, minimum_delta: float = 1e-5,
    max_wall_clock_hours: float = 8.0,
    seed: int = 0, local_k: int = 6, wide_k: int = 18,
    missing_image_policy: str = "zero", device_str: str = "cuda",
    linear_algebra_device: str = "cuda",
) -> dict:
    started = time.perf_counter()
    if image_encoder not in {"uni2", "gigapath"}:
        raise ValueError("image_encoder must be 'uni2' or 'gigapath'")
    if missing_image_policy != "zero":
        raise ValueError("the primary exact-split MLP currently requires missing_image_policy='zero'")
    if min(pca_components, pca_spots_per_slide, train_spots_per_slide,
           validation_spots_per_slide, hidden_dim, epochs, batch_size) < 1:
        raise ValueError("all dimensions, counts, and epochs must be positive")
    if (learning_rate <= 0 or weight_decay < 0 or patience_epochs < 1
            or minimum_delta < 0 or max_wall_clock_hours <= 0):
        raise ValueError("invalid optimizer or early-stopping settings")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device_str!r} requested but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = resolved_config(config_path)
    cfg = OmegaConf.create(config)
    if manifest_path is not None:
        cfg.data.gen3_manifest_path = str(Path(manifest_path).expanduser().resolve())
    if hest_data_dir is not None:
        cfg.data.hest_data_dir = str(Path(hest_data_dir).expanduser().resolve())
    cfg.data.image_encoder = image_encoder
    cfg.data.retain_patches_in_memory = False
    cfg.data.use_histology_features = False
    cfg.data.slide_context_source = "disabled"
    if cache_dir is not None:
        key = "gen3_uni2_spot_feature_cache_dir" if image_encoder == "uni2" else "gen3_spot_feature_cache_dir"
        cfg.data[key] = str(Path(cache_dir).expanduser().resolve())

    manifest = load_dataset_manifest(str(cfg.data.gen3_manifest_path))
    if hest_data_dir is not None:
        manifest = dict(manifest)
        manifest["hest_data_dir"] = str(Path(hest_data_dir).expanduser().resolve())
    train_ids = [str(value) for value in manifest["train_sample_ids"]]
    split_ids = [str(value) for value in manifest[f"{split}_sample_ids"]]
    gene_names = [str(value) for value in manifest["gene_panel"]]
    if not train_ids or not split_ids or not gene_names:
        raise ValueError("manifest train/split IDs and gene panel must be non-empty")
    cache_coverage = _cache_contract(cfg, image_encoder, train_ids + split_ids)
    panel_payload = (
        load_train_derived_gene_panels(train_gene_panels_path, manifest)
        if train_gene_panels_path else {"panels": {}}
    )
    panels = dict(panel_payload.get("panels") or {})
    panel_indices, panel_metadata = resolve_gene_panels(gene_names, panels) if panels else ({}, {})

    pca, feature_provenance, n_pca_rows = _fit_pca(
        cfg, manifest, train_ids, image_policy=missing_image_policy,
        spots_per_slide=pca_spots_per_slide, seed=seed,
        components=pca_components, device=linear_algebra_device,
    )
    model = FrozenFeatureMLP(
        pca_components, hidden_dim, len(gene_names), dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    output_path = Path(output).expanduser().resolve()
    checkpoint_path = output_path.with_suffix(".model.pt")
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    target_sum = np.zeros(len(gene_names), dtype=np.float64)
    target_sum_squared = np.zeros(len(gene_names), dtype=np.float64)
    n_scale_rows = 0
    completion_reason = "epochs_completed"

    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + 10_000 * epoch)
        ordered_ids = [train_ids[index] for index in rng.permutation(len(train_ids))]
        train_squared_error = 0.0
        train_values = 0
        for slide_position, sample_id in enumerate(ordered_ids):
            sample = _load_sample(cfg, manifest, sample_id)
            chosen = _sample_rows(
                sample_id, sample, train_spots_per_slide,
                seed + 100_000 * epoch + slide_position,
            )
            x = pca.transform(sample.precomputed_spot_features[chosen]).astype(np.float32)
            y = _dense_expression(sample.adata.X)[chosen]
            if epoch == 1:
                target_sum += y.sum(axis=0)
                target_sum_squared += np.square(y, dtype=np.float64).sum(axis=0)
                n_scale_rows += len(y)
            order = rng.permutation(len(chosen))
            for start in range(0, len(order), batch_size):
                batch = order[start:min(start + batch_size, len(order))]
                x_batch = torch.as_tensor(x[batch], device=device)
                y_batch = torch.as_tensor(y[batch], device=device)
                optimizer.zero_grad(set_to_none=True)
                predicted = model(x_batch)
                loss = torch.mean(torch.square(predicted - y_batch))
                loss.backward()
                optimizer.step()
                train_squared_error += float(loss.detach()) * int(y_batch.numel())
                train_values += int(y_batch.numel())
        validation_mse = _validation_mse(
            model, pca, cfg, manifest, split_ids,
            rows_per_slide=validation_spots_per_slide, batch_size=batch_size,
            seed=seed + 77, device=device,
        )
        train_mse = train_squared_error / train_values
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse})
        print(
            f"[frozen MLP epoch {epoch}/{epochs}] train_mse={train_mse:.8f} "
            f"validation_mse={validation_mse:.8f}", flush=True,
        )
        if validation_mse < best_validation_mse - minimum_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            scale, _ = _positive_gene_scale(target_sum, target_sum_squared, n_scale_rows)
            _save_checkpoint(
                checkpoint_path, model=model, pca=pca, per_gene_scale=scale,
                gene_names=gene_names,
                metadata={"epoch": epoch, "validation_mse": validation_mse},
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience_epochs:
                print(f"Early stopping after epoch {epoch}; best epoch={best_epoch}", flush=True)
                completion_reason = "early_stopping"
                break
        if time.perf_counter() - started >= max_wall_clock_hours * 3600.0:
            print(
                f"Wall-clock limit reached after epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            completion_reason = "wall_clock_limit_reached"
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    per_gene_scale = np.asarray(checkpoint["per_gene_scale"], dtype=np.float32)

    records = []
    with torch.no_grad():
        for position, sample_id in enumerate(split_ids):
            sample = _load_sample(cfg, manifest, sample_id)
            evaluated = _candidate_indices(sample, missing_image_policy)
            x = pca.transform(sample.precomputed_spot_features[evaluated]).astype(np.float32)
            chunks = [
                model(torch.as_tensor(x[start:min(start + batch_size, len(x))], device=device))
                .cpu().numpy().astype(np.float32)
                for start in range(0, len(x), batch_size)
            ]
            predicted = np.concatenate(chunks, axis=0)
            target = _dense_expression(sample.adata.X)[evaluated]
            coords = np.asarray(sample.full_sample_coords[evaluated], dtype=np.float64)
            point = _panel_metrics(predicted, target, panel_indices)
            structured = structured_field_metrics(
                predicted, target, coords, per_gene_scale,
                panel_indices=panel_indices, local_k=local_k, wide_k=wide_k,
            )
            records.append({
                "sample_id": sample_id,
                "patient_id": str(sample.patient_id),
                "organ": str(manifest["samples"][sample_id]["organ"]),
                "technology": str(manifest["samples"][sample_id]["tech"]),
                "n_manifest_spots": int(sample.adata.n_obs),
                "n_evaluated_spots": int(len(evaluated)),
                "n_spots_with_he": int(np.count_nonzero(sample.image_source_available)),
                "image_coverage_fraction": float(np.mean(sample.image_source_available)),
                "point_metrics": point,
                "structured_field": structured,
            })
            print(
                f"Whole-slide evaluation: {position + 1}/{len(split_ids)} "
                f"sample={sample_id} spots={len(evaluated)}", flush=True,
            )

    patient_ids = [row["patient_id"] for row in records]
    point_aggregated = {
        panel: aggregate_patient_metrics(
            [row["point_metrics"][panel] for row in records], patient_ids,
        )
        for panel in records[0]["point_metrics"]
    }
    structured_aggregated = {
        panel: aggregate_patient_metrics(
            [_flatten_structured_panel(row["structured_field"]["panels"][panel]) for row in records],
            patient_ids,
        )
        for panel in records[0]["structured_field"]["panels"]
    }
    by_organ = _aggregate_point_metrics_by_group(records, "organ")
    by_technology = _aggregate_point_metrics_by_group(records, "technology")
    elapsed = time.perf_counter() - started
    peak_gpu_memory = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )
    report = {
        "version": 1,
        "kind": "frozen_feature_pca_mlp_whole_slide_benchmark",
        "benchmark_track": "expanded_cohort_exact_split",
        "config_path": str(Path(config_path).expanduser().resolve()),
        "manifest_path": str(Path(str(cfg.data.gen3_manifest_path)).resolve()),
        "split": split,
        "image_encoder": image_encoder,
        "feature_provenance": feature_provenance,
        "cache_coverage": cache_coverage,
        "missing_image_policy": missing_image_policy,
        "target_space": manifest.get("build_args", {}).get("expression_transform"),
        "expression_target_sum": manifest.get("build_args", {}).get("expression_target_sum"),
        "pca_components": int(pca_components),
        "pca_method": "train_only_bounded_covariance_eigh",
        "linear_algebra_device": linear_algebra_device,
        "pca_spots_per_slide": int(pca_spots_per_slide),
        "train_spots_per_slide_per_epoch": int(train_spots_per_slide),
        "validation_spots_per_slide": int(validation_spots_per_slide),
        "hidden_dim": int(hidden_dim), "dropout": float(dropout),
        "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
        "epochs_requested": int(epochs), "epochs_completed": len(history),
        "max_wall_clock_hours": float(max_wall_clock_hours),
        "completion_reason": completion_reason,
        "best_epoch": int(best_epoch), "best_validation_mse": float(best_validation_mse),
        "training_history": history,
        "n_train_samples": len(train_ids), "n_pca_rows": int(n_pca_rows),
        "n_validation_samples": len(split_ids),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "runtime_seconds": float(elapsed), "peak_gpu_memory_mib": peak_gpu_memory,
        "gene_panel_metadata": panel_metadata,
        "point_metrics_patient_aggregated": point_aggregated,
        "structured_metrics_patient_aggregated": structured_aggregated,
        "point_metrics_by_organ": by_organ,
        "point_metrics_by_technology": by_technology,
        "per_slide_records": records,
        "model_artifact": str(checkpoint_path),
        "comparability_notes": {
            "primary": "whole-slide gene-wise PCC, patient macro average",
            "image_encoder_frozen": True,
            "target_gex_visible_to_model": False,
            "panel_selection": "training-only", "target_smoothing": False,
        },
    }
    _atomic_json(output_path, report)
    print(f"frozen-feature MLP benchmark saved to {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-encoder", choices=("uni2", "gigapath"), default="uni2")
    parser.add_argument("--manifest")
    parser.add_argument("--train-gene-panels")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--pca-spots-per-slide", type=int, default=128)
    parser.add_argument("--train-spots-per-slide", type=int, default=2048)
    parser.add_argument("--validation-spots-per-slide", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience-epochs", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=1e-5)
    parser.add_argument("--max-wall-clock-hours", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-k", type=int, default=6)
    parser.add_argument("--wide-k", type=int, default=18)
    parser.add_argument("--missing-image-policy", choices=("zero",), default="zero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--linear-algebra-device", default="cuda")
    args = parser.parse_args()
    evaluate_frozen_feature_mlp(
        config_path=args.config, output=args.output,
        image_encoder=args.image_encoder, manifest_path=args.manifest,
        train_gene_panels_path=args.train_gene_panels, cache_dir=args.cache_dir,
        hest_data_dir=args.hest_data_dir, split=args.split,
        pca_components=args.pca_components,
        pca_spots_per_slide=args.pca_spots_per_slide,
        train_spots_per_slide=args.train_spots_per_slide,
        validation_spots_per_slide=args.validation_spots_per_slide,
        hidden_dim=args.hidden_dim, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, patience_epochs=args.patience_epochs,
        minimum_delta=args.minimum_delta,
        max_wall_clock_hours=args.max_wall_clock_hours, seed=args.seed,
        local_k=args.local_k, wide_k=args.wide_k,
        missing_image_policy=args.missing_image_policy,
        device_str=args.device, linear_algebra_device=args.linear_algebra_device,
    )


if __name__ == "__main__":
    main()
