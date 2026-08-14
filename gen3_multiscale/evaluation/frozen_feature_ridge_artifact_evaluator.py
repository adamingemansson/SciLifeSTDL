#!/usr/bin/env python3
"""Re-score a saved frozen-feature ridge fit without refitting train data.

The expensive PCA/ridge fit is immutable. This evaluator validates that the
saved artifact, original fit report, current manifest, and frozen feature
cache describe the same experiment, then evaluates every held-out spot once
with the current shared metric implementation. It is intended for adding new
metrics (for example spatial SSIM) without repeating the 242-slide fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.frozen_feature_ridge_evaluator import (
    _CovariancePCA,
    _aggregate_point_metrics_by_group,
    _atomic_json,
    _candidate_indices,
    _dense_expression,
    _flatten_structured_panel,
    _load_sample,
    _panel_metrics,
)
from gen3_multiscale.evaluation.metrics import aggregate_patient_metrics, resolve_gene_panels
from gen3_multiscale.evaluation.per_gene_diagnostics import PerGeneDiagnosticsAccumulator
from gen3_multiscale.evaluation.structured_field_metrics import structured_field_metrics
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.data import spot_feature_cache
from gen3_multiscale.gen4 import uni2_spot_cache


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_and_validate_fit(
    artifact_path: Path, fit_report: dict, *, gene_names: list[str],
    manifest_path: Path, image_encoder: str, missing_image_policy: str,
) -> tuple[_CovariancePCA, np.ndarray, np.ndarray]:
    """Load a fit only when all identity-bearing fields match exactly."""
    if fit_report.get("kind") != "frozen_feature_pca_ridge_whole_slide_benchmark":
        raise ValueError("fit report is not a frozen-feature PCA/ridge report")
    if fit_report.get("image_encoder") != image_encoder:
        raise ValueError("image encoder differs from the saved ridge fit")
    if fit_report.get("missing_image_policy") != missing_image_policy:
        raise ValueError("missing-image policy differs from the saved ridge fit")
    reported_manifest = Path(str(fit_report.get("manifest_path") or "")).expanduser().resolve()
    if reported_manifest != manifest_path.resolve():
        raise ValueError(
            f"manifest differs from saved ridge fit: {reported_manifest} != {manifest_path}"
        )
    reported_artifact = Path(str(fit_report.get("model_artifact") or "")).expanduser().resolve()
    if reported_artifact != artifact_path.resolve():
        raise ValueError(
            f"artifact differs from fit report: {reported_artifact} != {artifact_path}"
        )

    with np.load(artifact_path, allow_pickle=False) as payload:
        required = {
            "pca_mean", "pca_components", "coefficients",
            "per_gene_scale", "gene_names",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"ridge artifact is incomplete; missing={missing}")
        artifact_genes = [str(value) for value in payload["gene_names"].tolist()]
        mean = np.asarray(payload["pca_mean"], dtype=np.float32)
        components = np.asarray(payload["pca_components"], dtype=np.float32)
        coefficients = np.asarray(payload["coefficients"], dtype=np.float32)
        per_gene_scale = np.asarray(payload["per_gene_scale"], dtype=np.float32)

    if artifact_genes != gene_names:
        raise ValueError("ridge artifact gene panel/order differs from the manifest")
    if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != len(mean):
        raise ValueError("ridge artifact PCA arrays have incompatible shapes")
    if coefficients.shape != (components.shape[0] + 1, len(gene_names)):
        raise ValueError("ridge artifact coefficient shape is incompatible")
    if per_gene_scale.shape != (len(gene_names),):
        raise ValueError("ridge artifact per-gene scale shape is incompatible")
    arrays = (mean, components, coefficients, per_gene_scale)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("ridge artifact contains non-finite values")
    if not np.all(per_gene_scale > 0):
        raise ValueError("ridge artifact per-gene scales must be positive")
    reported_components = int(fit_report.get("pca_components", -1))
    if reported_components != components.shape[0]:
        raise ValueError("fit report PCA dimension differs from the artifact")
    return _CovariancePCA(mean, components), coefficients, per_gene_scale


def evaluate_frozen_feature_ridge_artifact(
    *, config_path: str, manifest_path: str, train_gene_panels_path: str,
    artifact_path: str, fit_report_path: str, output: str,
    image_encoder: str = "uni2", cache_dir: str | None = None,
    hest_data_dir: str | None = None, split: str = "validation",
    local_k: int = 6, wide_k: int = 18, missing_image_policy: str = "zero",
    per_gene_diagnostics_output: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if image_encoder not in {"uni2", "gigapath"}:
        raise ValueError("image_encoder must be 'uni2' or 'gigapath'")
    if missing_image_policy not in {"zero", "exclude"}:
        raise ValueError("missing_image_policy must be 'zero' or 'exclude'")

    manifest_file = Path(manifest_path).expanduser().resolve()
    artifact_file = Path(artifact_path).expanduser().resolve()
    fit_report_file = Path(fit_report_path).expanduser().resolve()
    for label, path in (
        ("manifest", manifest_file), ("artifact", artifact_file),
        ("fit report", fit_report_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    cfg = OmegaConf.create(resolved_config(config_path))
    cfg.data.gen3_manifest_path = str(manifest_file)
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

    manifest = load_dataset_manifest(str(manifest_file))
    if hest_data_dir is not None:
        manifest = dict(manifest)
        manifest["hest_data_dir"] = str(Path(hest_data_dir).expanduser().resolve())
    split_ids = [str(value) for value in manifest[f"{split}_sample_ids"]]
    gene_names = [str(value) for value in manifest["gene_panel"]]
    if not split_ids or not gene_names:
        raise ValueError("manifest split IDs and gene panel must be non-empty")

    if image_encoder == "uni2":
        cache_coverage = uni2_spot_cache.require_uni2_spot_cache_coverage(
            uni2_spot_cache.cfg_cache_root(cfg), split_ids,
        )
    else:
        missing = [
            sample_id for sample_id in split_ids
            if not spot_feature_cache._cache_path(cfg, sample_id).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"GigaPath cache covers {len(split_ids) - len(missing)}/{len(split_ids)} "
                f"held-out samples; first missing={missing[:15]}"
            )
        cache_coverage = {
            "cache_dir": str(spot_feature_cache._cache_path(cfg, split_ids[0]).parent),
            "required_samples": len(split_ids), "missing_samples": [],
        }

    fit_report = _load_json(fit_report_file)
    pca, coefficients, per_gene_scale = _load_and_validate_fit(
        artifact_file, fit_report, gene_names=gene_names,
        manifest_path=manifest_file, image_encoder=image_encoder,
        missing_image_policy=missing_image_policy,
    )
    panels_payload = load_train_derived_gene_panels(train_gene_panels_path, manifest)
    panels = dict(panels_payload.get("panels") or {})
    panel_indices, panel_metadata = resolve_gene_panels(gene_names, panels)

    expected_provenance = fit_report.get("feature_provenance")
    if not isinstance(expected_provenance, dict) or not expected_provenance:
        raise ValueError("fit report lacks frozen-feature provenance")
    records: list[dict[str, Any]] = []
    per_gene_accumulator = (
        PerGeneDiagnosticsAccumulator(gene_names)
        if per_gene_diagnostics_output else None
    )
    for position, sample_id in enumerate(split_ids):
        sample = _load_sample(cfg, manifest, sample_id)
        actual_provenance = sample.tile_encoder_provenance["spot_features"]
        if actual_provenance != expected_provenance:
            raise ValueError(f"frozen-feature provenance mismatch for {sample_id}")
        evaluated = _candidate_indices(sample, missing_image_policy)
        features = pca.transform(sample.precomputed_spot_features[evaluated]).astype(np.float64)
        augmented = np.concatenate(
            [features, np.ones((len(features), 1), dtype=np.float64)], axis=1,
        )
        predicted = (augmented @ coefficients).astype(np.float32)
        target = _dense_expression(sample.adata.X)[evaluated]
        coords = np.asarray(sample.full_sample_coords[evaluated], dtype=np.float64)
        records.append({
            "sample_id": sample_id,
            "patient_id": str(sample.patient_id),
            "organ": str(manifest["samples"][sample_id]["organ"]),
            "technology": str(manifest["samples"][sample_id]["tech"]),
            "n_manifest_spots": int(sample.adata.n_obs),
            "n_evaluated_spots": int(len(evaluated)),
            "n_spots_with_he": int(np.count_nonzero(sample.image_source_available)),
            "image_coverage_fraction": float(np.mean(sample.image_source_available)),
            "point_metrics": _panel_metrics(predicted, target, panel_indices),
            "structured_field": structured_field_metrics(
                predicted, target, coords, per_gene_scale,
                panel_indices=panel_indices, local_k=local_k, wide_k=wide_k,
            ),
        })
        if per_gene_accumulator is not None:
            per_gene_accumulator.add_slide(
                sample_id=sample_id, patient_id=str(sample.patient_id),
                organ=str(manifest["samples"][sample_id]["organ"]),
                predicted=predicted, target=target, coords=coords, local_k=local_k,
            )
        print(
            f"Whole-slide re-evaluation: {position + 1}/{len(split_ids)} "
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
            [_flatten_structured_panel(row["structured_field"]["panels"][panel])
             for row in records],
            patient_ids,
        )
        for panel in records[0]["structured_field"]["panels"]
    }
    per_gene_path = None
    if per_gene_accumulator is not None:
        per_gene_path = per_gene_accumulator.save(
            per_gene_diagnostics_output,
            provenance={
                "method": f"{image_encoder}_pca_ridge",
                "report_output": str(Path(output).expanduser().resolve()),
                "split": split, "target_space": "normalize_total_then_log1p",
                "primary_prediction": "frozen_feature_ridge",
            },
        )
    report = {
        **{
            key: value for key, value in fit_report.items()
            if key not in {
                "point_metrics_patient_aggregated",
                "structured_metrics_patient_aggregated", "point_metrics_by_organ",
                "point_metrics_by_technology", "per_slide_records",
            }
        },
        "version": max(2, int(fit_report.get("version", 1))),
        "fit_reused": True,
        "fit_report_source": str(fit_report_file),
        "model_artifact": str(artifact_file),
        "cache_coverage": cache_coverage,
        "gene_panel_metadata": panel_metadata,
        "point_metrics_patient_aggregated": point_aggregated,
        "structured_metrics_patient_aggregated": structured_aggregated,
        "point_metrics_by_organ": _aggregate_point_metrics_by_group(records, "organ"),
        "point_metrics_by_technology": _aggregate_point_metrics_by_group(
            records, "technology",
        ),
        "per_slide_records": records,
        "per_gene_diagnostics_path": str(per_gene_path) if per_gene_path else None,
        "reevaluation_runtime_seconds": float(time.perf_counter() - started),
        "parameter_count": int(coefficients.size),
        "comparability_notes": {
            **dict(fit_report.get("comparability_notes") or {}),
            "fit_reused_without_refitting": True,
            "metric_implementation": "current shared whole-slide benchmark metrics",
        },
    }
    output_path = Path(output).expanduser().resolve()
    _atomic_json(output_path, report)
    print(f"re-scored ridge benchmark saved to {output_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--fit-artifact", required=True)
    parser.add_argument("--fit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-encoder", choices=("uni2", "gigapath"), default="uni2")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--local-k", type=int, default=6)
    parser.add_argument("--wide-k", type=int, default=18)
    parser.add_argument("--missing-image-policy", choices=("zero", "exclude"), default="zero")
    parser.add_argument("--per-gene-diagnostics-output")
    args = parser.parse_args()
    evaluate_frozen_feature_ridge_artifact(
        config_path=args.config, manifest_path=args.manifest,
        train_gene_panels_path=args.train_gene_panels,
        artifact_path=args.fit_artifact, fit_report_path=args.fit_report,
        output=args.output, image_encoder=args.image_encoder,
        cache_dir=args.cache_dir, hest_data_dir=args.hest_data_dir,
        split=args.split, local_k=args.local_k, wide_k=args.wide_k,
        missing_image_policy=args.missing_image_policy,
        per_gene_diagnostics_output=args.per_gene_diagnostics_output,
    )


if __name__ == "__main__":
    main()
