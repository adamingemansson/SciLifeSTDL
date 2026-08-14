#!/usr/bin/env python3
"""Leakage-safe frozen-feature PCA/ridge benchmark on the project split.

This is the most important control for the MK models: it asks how far a
linear decoder can get from the exact frozen spot embeddings they consume.
PCA and ridge are fit using training samples only. Every held-out slide is
then predicted in full, one spot exactly once, and scored with the same point
and structured-field metric implementation as the learned MK architectures.

The implementation deliberately uses bounded train-only PCA sampling and
ridge sufficient statistics. It never concatenates the full 17k-gene target
matrix across hundreds of slides in RAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy import sparse
import torch

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics,
    comparable_expression_metrics,
    nonzero_auc,
    resolve_gene_panels,
)
from gen3_multiscale.evaluation.structured_field_metrics import structured_field_metrics
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.data import spot_feature_cache
from gen3_multiscale.gen4 import uni2_spot_cache
from gen3_multiscale.training.gen3_dataset import load_gen3_sample_data


def _stable_indices(sample_id: str, n_rows: int, limit: int, seed: int) -> np.ndarray:
    if n_rows < 1:
        return np.empty(0, dtype=np.int64)
    count = min(int(limit), int(n_rows))
    digest = hashlib.sha256(f"ridge:{seed}:{sample_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(n_rows, size=count, replace=False)).astype(np.int64)


def _dense_expression(matrix) -> np.ndarray:
    array = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("expression target must be a finite rank-2 matrix")
    return array


def _available(sample) -> np.ndarray:
    return np.flatnonzero(np.asarray(sample.image_source_available, dtype=bool))


def _candidate_indices(sample, missing_image_policy: str) -> np.ndarray:
    if missing_image_policy == "zero":
        return np.arange(sample.adata.n_obs, dtype=np.int64)
    if missing_image_policy == "exclude":
        return _available(sample)
    raise ValueError("missing_image_policy must be 'zero' or 'exclude'")


def _load_sample(cfg, manifest: dict, sample_id: str):
    return load_gen3_sample_data(cfg, manifest, sample_id, require_dense_wsi=False)


def _point_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        **comparable_expression_metrics(predicted, target),
        "auc": nonzero_auc(predicted, target),
    }


def _panel_metrics(
    predicted: np.ndarray, target: np.ndarray, panel_indices: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result = {"all_genes": _point_metrics(predicted, target)}
    for name, indices in panel_indices.items():
        result[name] = _point_metrics(predicted[:, indices], target[:, indices])
    return result


def _flatten_structured_panel(panel: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section, values in panel.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if name.startswith("n_") or name.endswith("threshold_training_sd"):
                continue
            if isinstance(value, (int, float)):
                flat[f"{section}.{name}"] = float(value)
    return flat


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


class _CovariancePCA:
    """Small PCA projection fitted from bounded sufficient statistics."""

    def __init__(self, mean: np.ndarray, components: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.components_ = np.asarray(components, dtype=np.float32)

    def transform(self, features: np.ndarray) -> np.ndarray:
        centered = np.asarray(features, dtype=np.float32) - self.mean_[None, :]
        return centered @ self.components_.T


def _fit_covariance_pca(
    feature_sum: np.ndarray,
    feature_cross_product: np.ndarray,
    n_rows: int,
    n_components: int,
    device: str,
) -> _CovariancePCA:
    """Fit PCA without invoking NumPy/SciPy LAPACK.

    The server's MKL build has repeatedly segfaulted in ``SLASWP`` during
    randomized SVD. Computing the small feature covariance explicitly and
    using torch's eigensolver on CUDA avoids that failure while producing the
    same principal subspace from the exact bounded train-only sample.
    """
    if n_rows < 2:
        raise ValueError("PCA requires at least two sampled training rows")
    mean = np.asarray(feature_sum, dtype=np.float64) / float(n_rows)
    covariance = (
        np.asarray(feature_cross_product, dtype=np.float64)
        - float(n_rows) * np.outer(mean, mean)
    ) / float(n_rows - 1)
    covariance = 0.5 * (covariance + covariance.T)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"linear algebra device {device!r} requested but CUDA is unavailable")
    covariance_tensor = torch.as_tensor(
        covariance, dtype=torch.float64, device=target_device,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)
    order = torch.argsort(eigenvalues, descending=True)[:n_components]
    components = eigenvectors[:, order].T.cpu().numpy()
    del covariance_tensor, eigenvalues, eigenvectors
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return _CovariancePCA(mean, components)


def _solve_ridge(
    xtx: np.ndarray, xty: np.ndarray, penalty: np.ndarray, device: str,
) -> np.ndarray:
    """Solve ridge normal equations without the server's unstable MKL LAPACK."""
    target_device = torch.device(device)
    lhs = torch.as_tensor(xtx + penalty, dtype=torch.float64, device=target_device)
    rhs = torch.as_tensor(xty, dtype=torch.float64, device=target_device)
    solution = torch.linalg.solve(lhs, rhs).cpu().numpy()
    del lhs, rhs
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return solution


def _positive_gene_scale(
    target_sum: np.ndarray,
    target_sum_squared: np.ndarray,
    n_rows: int,
    *,
    floor: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Return finite positive train-only gene scales for spatial metrics.

    Some genes can be constant in the bounded ridge sample. Their empirical
    standard deviation is exactly zero, but the structured-field metrics use
    the scale as a divisor and therefore require a strictly positive value.
    Flooring only those degenerate scales preserves every non-degenerate
    training standard deviation and matches the training-side convention.
    """
    if n_rows < 1:
        raise ValueError("gene-scale estimation requires at least one row")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("gene-scale floor must be finite and positive")
    mean = np.asarray(target_sum, dtype=np.float64) / float(n_rows)
    variance = np.maximum(
        np.asarray(target_sum_squared, dtype=np.float64) / float(n_rows)
        - np.square(mean),
        0.0,
    )
    scale = np.sqrt(variance)
    if not np.isfinite(scale).all():
        raise ValueError("train-only per-gene scales contain non-finite values")
    n_floored = int(np.count_nonzero(scale < floor))
    return np.clip(scale, floor, None).astype(np.float32), n_floored


def _save_fit_artifact(
    path: Path,
    *,
    pca: _CovariancePCA,
    coefficients: np.ndarray,
    per_gene_scale: np.ndarray,
    gene_names: list[str],
) -> None:
    """Atomically persist the expensive fit before held-out evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pca_mean=pca.mean_.astype(np.float32),
            pca_components=pca.components_.astype(np.float32),
            coefficients=np.asarray(coefficients, dtype=np.float32),
            per_gene_scale=np.asarray(per_gene_scale, dtype=np.float32),
            gene_names=np.asarray(gene_names),
        )
    os.replace(temporary, path)


def evaluate_frozen_feature_ridge(
    *, config_path: str, output: str, image_encoder: str,
    manifest_path: str | None = None, train_gene_panels_path: str | None = None,
    cache_dir: str | None = None, hest_data_dir: str | None = None,
    split: str = "validation", pca_components: int = 256,
    pca_spots_per_slide: int = 128, ridge_spots_per_slide: int = 2048,
    ridge_alpha: float = 1.0, seed: int = 0, local_k: int = 6,
    wide_k: int = 18, missing_image_policy: str = "zero",
    linear_algebra_device: str = "cpu",
) -> dict:
    if image_encoder not in {"uni2", "gigapath"}:
        raise ValueError("image_encoder must be 'uni2' or 'gigapath'")
    if pca_components < 1 or pca_spots_per_slide < 1 or ridge_spots_per_slide < 1:
        raise ValueError("PCA components and per-slide sample limits must be positive")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    if missing_image_policy not in {"zero", "exclude"}:
        raise ValueError("missing_image_policy must be 'zero' or 'exclude'")

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
        key = (
            "gen3_uni2_spot_feature_cache_dir"
            if image_encoder == "uni2" else "gen3_spot_feature_cache_dir"
        )
        cfg.data[key] = str(Path(cache_dir).expanduser().resolve())

    manifest = load_dataset_manifest(str(cfg.data.gen3_manifest_path))
    if hest_data_dir is not None:
        # The loader uses both cfg.data.hest_data_dir (provenance check) and
        # manifest["hest_data_dir"] (actual h5ad/patch resolution). Keep them
        # identical when a relocated but content-identical dataset is used.
        manifest = dict(manifest)
        manifest["hest_data_dir"] = str(Path(hest_data_dir).expanduser().resolve())
    train_ids = [str(value) for value in manifest["train_sample_ids"]]
    split_ids = [str(value) for value in manifest[f"{split}_sample_ids"]]
    gene_names = [str(value) for value in manifest["gene_panel"]]
    if not train_ids or not split_ids or not gene_names:
        raise ValueError("manifest train/split IDs and gene panel must be non-empty")

    required_ids = train_ids + split_ids
    if image_encoder == "uni2":
        cache_coverage = uni2_spot_cache.require_uni2_spot_cache_coverage(
            uni2_spot_cache.cfg_cache_root(cfg), required_ids,
        )
    else:
        missing_cache = [
            sample_id for sample_id in required_ids
            if not spot_feature_cache._cache_path(cfg, sample_id).is_file()
        ]
        if missing_cache:
            raise FileNotFoundError(
                f"GigaPath spot-feature cache covers "
                f"{len(required_ids) - len(missing_cache)}/{len(required_ids)} samples; "
                f"first missing={missing_cache[:15]}"
            )
        cache_coverage = {
            "cache_dir": str(spot_feature_cache._cache_path(cfg, required_ids[0]).parent),
            "required_samples": len(required_ids),
            "missing_samples": [],
        }

    panel_payload = (
        load_train_derived_gene_panels(train_gene_panels_path, manifest)
        if train_gene_panels_path else {"panels": {}}
    )
    panels = dict(panel_payload.get("panels") or {})
    panel_indices, panel_metadata = resolve_gene_panels(gene_names, panels) if panels else ({}, {})

    # Pass 1: bounded, equal-per-slide train-only covariance statistics for
    # PCA. This avoids retaining/concatenating the sampled feature matrix and
    # avoids the server's unstable MKL randomized-SVD path.
    feature_sum = None
    feature_cross_product = None
    n_pca_rows = 0
    provenance_by_sample = {}
    for position, sample_id in enumerate(train_ids):
        sample = _load_sample(cfg, manifest, sample_id)
        candidates = _candidate_indices(sample, missing_image_policy)
        chosen = candidates[_stable_indices(
            sample_id, len(candidates), pca_spots_per_slide, seed,
        )]
        rows = np.asarray(sample.precomputed_spot_features[chosen], dtype=np.float64)
        if feature_sum is None:
            feature_sum = np.zeros(rows.shape[1], dtype=np.float64)
            feature_cross_product = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
        feature_sum += rows.sum(axis=0)
        feature_cross_product += rows.T @ rows
        n_pca_rows += len(rows)
        provenance_by_sample[sample_id] = sample.tile_encoder_provenance["spot_features"]
        print(
            f"PCA sampling: {position + 1}/{len(train_ids)} sample={sample_id} rows={len(chosen)}",
            flush=True,
        )
        del sample, rows
    reference_provenance = provenance_by_sample[train_ids[0]]
    mismatched = [
        sample_id for sample_id, value in provenance_by_sample.items()
        if value != reference_provenance
    ]
    if mismatched:
        raise ValueError(f"spot-feature provenance differs across samples: {mismatched[:10]}")
    maximum_components = min(n_pca_rows - 1, len(feature_sum))
    if pca_components > maximum_components:
        raise ValueError(
            f"pca_components={pca_components} exceeds train-only sample rank {maximum_components}"
        )
    print(
        f"PCA eigensolve: rows={n_pca_rows} features={len(feature_sum)} "
        f"components={pca_components} device={linear_algebra_device}",
        flush=True,
    )
    pca = _fit_covariance_pca(
        feature_sum, feature_cross_product, n_pca_rows,
        pca_components, linear_algebra_device,
    )
    del feature_sum, feature_cross_product

    # Pass 2: sufficient statistics for multi-output ridge and train-only
    # per-gene scales. The intercept is not regularized.
    augmented_dim = pca_components + 1
    xtx = np.zeros((augmented_dim, augmented_dim), dtype=np.float64)
    xty = np.zeros((augmented_dim, len(gene_names)), dtype=np.float64)
    target_sum = np.zeros(len(gene_names), dtype=np.float64)
    target_sum_squared = np.zeros(len(gene_names), dtype=np.float64)
    n_train_rows = 0
    for position, sample_id in enumerate(train_ids):
        sample = _load_sample(cfg, manifest, sample_id)
        candidates = _candidate_indices(sample, missing_image_policy)
        chosen = candidates[_stable_indices(
            sample_id, len(candidates), ridge_spots_per_slide, seed + 1,
        )]
        x = pca.transform(sample.precomputed_spot_features[chosen]).astype(np.float64)
        x_augmented = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
        y = _dense_expression(sample.adata.X)[chosen].astype(np.float64)
        xtx += x_augmented.T @ x_augmented
        xty += x_augmented.T @ y
        target_sum += y.sum(axis=0)
        target_sum_squared += np.square(y).sum(axis=0)
        n_train_rows += len(y)
        print(
            f"Ridge statistics: {position + 1}/{len(train_ids)} sample={sample_id} rows={len(y)}",
            flush=True,
        )
        del sample, x, x_augmented, y
    penalty = np.eye(augmented_dim, dtype=np.float64) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    print(f"Ridge solve: device={linear_algebra_device}", flush=True)
    coefficients = _solve_ridge(xtx, xty, penalty, linear_algebra_device)
    per_gene_scale, n_gene_scales_floored = _positive_gene_scale(
        target_sum, target_sum_squared, n_train_rows,
    )
    print(
        f"Train-only gene scales: {n_gene_scales_floored}/{len(gene_names)} "
        "constant genes floored to 1e-6",
        flush=True,
    )

    # Persist the expensive train-only PCA/ridge fit before any held-out
    # reporting. A later metric or plotting failure must not discard the fit.
    output_path = Path(output).expanduser().resolve()
    model_path = output_path.with_suffix(".model.npz")
    _save_fit_artifact(
        model_path,
        pca=pca,
        coefficients=coefficients,
        per_gene_scale=per_gene_scale,
        gene_names=gene_names,
    )
    print(f"Ridge fit artifact saved to {model_path}", flush=True)

    records = []
    for position, sample_id in enumerate(split_ids):
        sample = _load_sample(cfg, manifest, sample_id)
        available = _available(sample)
        evaluated = _candidate_indices(sample, missing_image_policy)
        x = pca.transform(sample.precomputed_spot_features[evaluated]).astype(np.float64)
        x_augmented = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
        predicted = (x_augmented @ coefficients).astype(np.float32)
        target = _dense_expression(sample.adata.X)[evaluated]
        coords = np.asarray(sample.full_sample_coords[evaluated], dtype=np.float64)
        point = _panel_metrics(predicted, target, panel_indices)
        structured = structured_field_metrics(
            predicted, target, coords, per_gene_scale,
            panel_indices=panel_indices, local_k=local_k, wide_k=wide_k,
        )
        record = {
            "sample_id": sample_id,
            "patient_id": str(sample.patient_id),
            "organ": str(manifest["samples"][sample_id]["organ"]),
            "technology": str(manifest["samples"][sample_id]["tech"]),
            "n_manifest_spots": int(sample.adata.n_obs),
            "n_evaluated_spots": int(len(evaluated)),
            "n_spots_with_he": int(len(available)),
            "image_coverage_fraction": float(len(available) / sample.adata.n_obs),
            "point_metrics": point,
            "structured_field": structured,
        }
        records.append(record)
        print(
            f"Whole-slide evaluation: {position + 1}/{len(split_ids)} "
            f"sample={sample_id} spots={len(evaluated)}",
            flush=True,
        )
        del sample, x, x_augmented, predicted, target

    patient_ids = [row["patient_id"] for row in records]
    point_aggregated = {
        panel: aggregate_patient_metrics(
            [row["point_metrics"][panel] for row in records], patient_ids,
        )
        for panel in records[0]["point_metrics"]
    }
    structured_aggregated = {
        panel: aggregate_patient_metrics(
            [
                _flatten_structured_panel(row["structured_field"]["panels"][panel])
                for row in records
            ],
            patient_ids,
        )
        for panel in records[0]["structured_field"]["panels"]
    }
    by_organ = {}
    for organ in sorted({row["organ"] for row in records}):
        organ_records = [row for row in records if row["organ"] == organ]
        organ_patients = [row["patient_id"] for row in organ_records]
        by_organ[organ] = {
            panel: aggregate_patient_metrics(
                [row["point_metrics"][panel] for row in organ_records], organ_patients,
            )
            for panel in organ_records[0]["point_metrics"]
        }
    by_technology = {}
    for technology in sorted({row["technology"] for row in records}):
        technology_records = [row for row in records if row["technology"] == technology]
        technology_patients = [row["patient_id"] for row in technology_records]
        by_technology[technology] = {
            panel: aggregate_patient_metrics(
                [row["point_metrics"][panel] for row in technology_records],
                technology_patients,
            )
            for panel in technology_records[0]["point_metrics"]
        }

    report = {
        "version": 1,
        "kind": "frozen_feature_pca_ridge_whole_slide_benchmark",
        "benchmark_track": "expanded_cohort_exact_split",
        "config_path": str(Path(config_path).expanduser().resolve()),
        "manifest_path": str(Path(str(cfg.data.gen3_manifest_path)).resolve()),
        "split": split,
        "image_encoder": image_encoder,
        "feature_provenance": reference_provenance,
        "cache_coverage": cache_coverage,
        "missing_image_policy": missing_image_policy,
        "target_space": manifest.get("build_args", {}).get("expression_transform"),
        "expression_target_sum": manifest.get("build_args", {}).get("expression_target_sum"),
        "pca_components": int(pca_components),
        "pca_method": "train_only_bounded_covariance_eigh",
        "linear_algebra_device": linear_algebra_device,
        "pca_spots_per_slide": int(pca_spots_per_slide),
        "ridge_spots_per_slide": int(ridge_spots_per_slide),
        "ridge_alpha": float(ridge_alpha),
        "n_train_samples": len(train_ids),
        "n_train_rows_for_ridge": int(n_train_rows),
        "n_gene_scales_floored": int(n_gene_scales_floored),
        "n_validation_samples": len(split_ids),
        "gene_panel_metadata": panel_metadata,
        "point_metrics_patient_aggregated": point_aggregated,
        "structured_metrics_patient_aggregated": structured_aggregated,
        "point_metrics_by_organ": by_organ,
        "point_metrics_by_technology": by_technology,
        "per_slide_records": records,
        "model_artifact": str(model_path),
        "comparability_notes": {
            "primary": "whole-slide gene-wise PCC, patient macro average",
            "missing_he": (
                "spots without a real H&E patch use the cache's explicit zero feature, matching MK"
                if missing_image_policy == "zero"
                else "spots without a real H&E patch are excluded; coverage is reported"
            ),
            "panel_selection": "training-only",
            "target_smoothing": False,
        },
    }
    _atomic_json(output_path, report)
    print(f"frozen-feature ridge benchmark saved to {output_path}", flush=True)
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
    parser.add_argument("--ridge-spots-per-slide", type=int, default=2048)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-k", type=int, default=6)
    parser.add_argument("--wide-k", type=int, default=18)
    parser.add_argument(
        "--missing-image-policy", choices=("zero", "exclude"), default="zero",
        help="Use zero-placeholder cache rows (MK-comparable default) or exclude missing H&E spots.",
    )
    parser.add_argument(
        "--linear-algebra-device", default="cpu",
        help="Torch device for PCA eigensolve and ridge solve; use cuda on the server to avoid MKL.",
    )
    args = parser.parse_args()
    evaluate_frozen_feature_ridge(
        config_path=args.config,
        output=args.output,
        image_encoder=args.image_encoder,
        manifest_path=args.manifest,
        train_gene_panels_path=args.train_gene_panels,
        cache_dir=args.cache_dir,
        hest_data_dir=args.hest_data_dir,
        split=args.split,
        pca_components=args.pca_components,
        pca_spots_per_slide=args.pca_spots_per_slide,
        ridge_spots_per_slide=args.ridge_spots_per_slide,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        local_k=args.local_k,
        wide_k=args.wide_k,
        missing_image_policy=args.missing_image_policy,
        linear_algebra_device=args.linear_algebra_device,
    )


if __name__ == "__main__":
    main()
