#!/usr/bin/env python3
"""Measure train-to-held-out H&E stain shift and its association with MK error.

This is a diagnostic, not stain normalization.  The reference distribution is
fit from training slides only; held-out slides never influence the reference.
Patches are sampled one slide at a time so the command has bounded memory use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr
from skimage.color import rgb2hed

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data import loaders
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest


FEATURE_NAMES = tuple(
    [f"rgb_{channel}_{stat}" for channel in ("r", "g", "b") for stat in ("mean", "std")]
    + [f"hed_{channel}_{stat}" for channel in ("h", "e") for stat in ("mean", "std")]
)


def _stable_rng(seed: int, sample_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def patch_stain_features(patches: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Return one RGB/HED summary vector per patch."""
    values = np.asarray(patches)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"patches must have shape [N,H,W,3], got {values.shape}")
    if values.shape[0] < 1:
        raise ValueError("at least one patch is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = values.astype(np.float32)
    if values.max() > 1.5:
        values /= 255.0
    values = np.clip(values, 0.0, 1.0)
    rows = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        rgb_mean = batch.mean(axis=(1, 2))
        rgb_std = batch.std(axis=(1, 2))
        hed = rgb2hed(batch)
        hed_mean = hed[..., :2].mean(axis=(1, 2))
        hed_std = hed[..., :2].std(axis=(1, 2))
        rows.append(np.column_stack([
            rgb_mean[:, 0], rgb_std[:, 0],
            rgb_mean[:, 1], rgb_std[:, 1],
            rgb_mean[:, 2], rgb_std[:, 2],
            hed_mean[:, 0], hed_std[:, 0],
            hed_mean[:, 1], hed_std[:, 1],
        ]))
    result = np.concatenate(rows).astype(np.float64)
    if result.shape != (len(values), len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("non-finite or malformed stain features")
    return result


def robust_reference(train_slide_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_slide_vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("training stain vectors must be a finite [slides,features] matrix")
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center, scale


def stain_distance(vector: np.ndarray, center: np.ndarray, scale: np.ndarray) -> float:
    z = (np.asarray(vector) - np.asarray(center)) / np.asarray(scale)
    return float(np.sqrt(np.mean(np.square(z))))


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _load_performance(report_path: Path, expected_ids: list[str]) -> dict[str, dict[str, float]]:
    report = json.loads(report_path.read_text())
    whole = report.get("whole_slide_structured_field_evaluation") or {}
    records = whole.get("per_slide_records") or []
    by_id = {str(row["sample_id"]): row for row in records}
    if set(by_id) != set(expected_ids):
        raise ValueError(
            "evaluation report does not contain exactly the manifest's held-out slides: "
            f"missing={sorted(set(expected_ids) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(expected_ids))}"
        )
    result = {}
    for sample_id, row in by_id.items():
        point = (row.get("point_metrics") or {}).get("all_genes") or {}
        result[sample_id] = {
            "all_gene_pcc": float(point["pcc"]),
            "all_gene_rmse": float(point["rmse"]),
        }
    return result


def _load_template_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {str(row["sample_id"]): row for row in csv.DictReader(handle, delimiter="\t")}


def _association(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    x = np.asarray([row["stain_distance"] for row in rows], dtype=np.float64)
    y = np.asarray([row[outcome] for row in rows], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    rho = float("nan")
    p = float("nan")
    if finite.sum() >= 3 and np.std(x[finite]) > 0 and np.std(y[finite]) > 0:
        statistic = spearmanr(x[finite], y[finite])
        rho, p = float(statistic.statistic), float(statistic.pvalue)
    return {"outcome": outcome, "spearman_rho": rho, "p_value": p, "n_slides": int(finite.sum())}


def analyze(
    *, config_path: str, report_path: str, output_dir: str,
    max_patches_per_slide: int = 256, seed: int = 0,
    template_diagnostics: str | None = None,
) -> dict[str, Path]:
    if max_patches_per_slide < 8:
        raise ValueError("max_patches_per_slide must be at least 8")
    config = resolved_config(config_path)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    train_ids = list(map(str, manifest["train_sample_ids"]))
    validation_ids = list(map(str, manifest["validation_sample_ids"]))
    if len(train_ids) != 242 or len(validation_ids) != 14:
        raise ValueError(
            f"this study requires the 242/14 cohort, got {len(train_ids)}/{len(validation_ids)}"
        )
    performance = _load_performance(Path(report_path).expanduser().resolve(), validation_ids)
    template = _load_template_rows(
        Path(template_diagnostics).expanduser().resolve() if template_diagnostics else None
    )
    slide_vectors: dict[str, np.ndarray] = {}
    slide_rows: list[dict[str, Any]] = []
    for split, sample_ids in (("train", train_ids), ("validation", validation_ids)):
        for index, sample_id in enumerate(sample_ids):
            patches, _patch_barcodes = loaders.load_hest_patches(
                manifest["hest_data_dir"], sample_id,
            )
            if len(patches) < 1:
                raise ValueError(f"{sample_id}: no real H&E patches")
            rng = _stable_rng(seed, sample_id)
            chosen = rng.choice(
                len(patches),
                size=min(max_patches_per_slide, len(patches)),
                replace=False,
            )
            features = patch_stain_features(np.asarray(patches)[chosen])
            vector = features.mean(axis=0)
            slide_vectors[sample_id] = vector
            row: dict[str, Any] = {
                "split": split,
                "sample_id": sample_id,
                "patient_id": str(manifest["samples"][sample_id]["patient_id"]),
                "organ": str(manifest["samples"][sample_id]["organ"]),
                "n_available_patches": int(len(patches)),
                "n_sampled_patches": int(chosen.size),
            }
            row.update({name: float(value) for name, value in zip(FEATURE_NAMES, vector)})
            slide_rows.append(row)
            print(f"stain diagnostic: {split} {index + 1}/{len(sample_ids)} sample={sample_id}", flush=True)
            del patches, features
    center, scale = robust_reference(np.stack([slide_vectors[sample_id] for sample_id in train_ids]))
    validation_rows = []
    for row in slide_rows:
        row["stain_distance"] = stain_distance(slide_vectors[row["sample_id"]], center, scale)
        if row["split"] == "validation":
            row.update(performance[row["sample_id"]])
            for key, value in template.get(row["sample_id"], {}).items():
                if key not in row:
                    try:
                        row[f"template_{key}"] = float(value)
                    except ValueError:
                        row[f"template_{key}"] = value
            validation_rows.append(row)
    outcomes = ["all_gene_pcc", "all_gene_rmse"]
    if validation_rows and template:
        outcomes.extend([
            key for key in validation_rows[0]
            if key.startswith("template_") and isinstance(validation_rows[0][key], float)
        ])
    associations = [_association(validation_rows, outcome) for outcome in outcomes]
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    slide_path = root / "stain_features_by_slide.tsv"
    validation_path = root / "held_out_stain_performance.tsv"
    association_path = root / "stain_performance_associations.tsv"
    _write_tsv(slide_path, slide_rows)
    _write_tsv(validation_path, validation_rows)
    _write_tsv(association_path, associations)
    manifest_path = root / "stain_domain_manifest.json"
    manifest_path.write_text(json.dumps({
        "kind": "mk_stain_domain_shift_diagnostic",
        "config": str(Path(config_path).expanduser().resolve()),
        "evaluation_report": str(Path(report_path).expanduser().resolve()),
        "reference_fit_split": "train_only",
        "n_train_slides": len(train_ids),
        "n_validation_slides": len(validation_ids),
        "max_patches_per_slide": max_patches_per_slide,
        "seed": seed,
        "feature_names": FEATURE_NAMES,
        "reference_center": center.tolist(),
        "reference_scale": scale.tolist(),
        "outputs": [str(slide_path), str(validation_path), str(association_path)],
    }, indent=2) + "\n")
    return {"slides": slide_path, "validation": validation_path, "associations": association_path, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-patches-per-slide", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--template-diagnostics")
    args = parser.parse_args()
    outputs = analyze(
        config_path=args.config, report_path=args.report, output_dir=args.output_dir,
        max_patches_per_slide=args.max_patches_per_slide, seed=args.seed,
        template_diagnostics=args.template_diagnostics,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
