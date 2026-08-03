"""Zero-shot released-STPath baseline on Gen3's exact fixed masks.

This is deliberately a standalone evaluator, not a Gen3/4/5 architecture:
no parameter is trained or selected on the validation split.  It reuses the
same manifest-selected patients, deterministic validation/test mask schedule,
precomputed GigaPath spot features, named gene panels, and patient-level
aggregation as :mod:`gen3_multiscale.evaluation.gen3_evaluator`.

The Gen3 task hides both expression and H&E inside the query hole.  Released
STPath normally predicts a spot while seeing that spot's H&E feature.  To keep
this comparison honest, query image features are therefore all zeros here.
That is an out-of-distribution adaptation of STPath, so the report records it
explicitly as ``query_image_mode='zero_missing_tissue'``.  A morphology-visible
STPath score would answer an easier, different question and is intentionally
not offered by this CLI.

STPath's released head predicts log1p(raw counts), whereas Gen3's primary
space is library-size-normalized log1p expression.  The report contains both:

* ``normalized_log1p``: clamp impossible negative predicted counts to zero,
  invert log1p, normalize each prediction over the manifest gene panel to the
  configured target sum, then log1p.  These are the values comparable to the
  current Gen3 PCC/RMSE table.
* ``native_log1p_raw_supported_genes``: released-head output versus log1p raw
  counts, restricted to genes supported by STPath.  This is faithful to
  STPath's own preprocessing but its RMSE is not comparable to Gen3 RMSE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import (
    load_configured_gene_panels,
    per_item_reconstruction_metrics,
)
from gen3_multiscale.evaluation.metrics import aggregate_patient_metrics, gene_panel_metrics
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    dataset_manifest_fingerprint,
    expected_tile_encoder_provenance,
    resolved_config,
)


def stpath_log1p_to_normalized_log1p(
    prediction_log1p_raw: np.ndarray, *, target_sum: float,
) -> np.ndarray:
    """Map released-STPath output into Gen3's normalized-log1p space.

    Negative values cannot represent log1p counts and are conservatively
    mapped to zero counts.  Rows with zero predicted mass stay zero.
    """
    prediction = np.asarray(prediction_log1p_raw, dtype=np.float64)
    if prediction.ndim != 2:
        raise ValueError(f"prediction_log1p_raw must be 2-D, got {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("released STPath prediction contains non-finite values")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError(f"target_sum must be finite and positive, got {target_sum}")

    counts = np.expm1(np.maximum(prediction, 0.0))
    if not np.isfinite(counts).all():
        raise ValueError("released STPath prediction overflowed while converting log1p to counts")
    totals = counts.sum(axis=1, keepdims=True)
    scale = np.divide(
        float(target_sum), totals,
        out=np.zeros_like(totals, dtype=np.float64), where=totals > 0,
    )
    return np.log1p(counts * scale).astype(np.float32)


def scatter_supported_genes(
    supported_values: np.ndarray, supported_positions: list[int] | np.ndarray, n_genes: int,
) -> np.ndarray:
    """Place STPath-supported columns into manifest-panel order.

    Unsupported genes receive zero, which is the honest prediction from a
    model that cannot emit them.  Coverage is always recorded in the report.
    """
    values = np.asarray(supported_values, dtype=np.float32)
    positions = np.asarray(supported_positions, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != positions.shape[0]:
        raise ValueError("supported_values columns must match supported_positions")
    if positions.size and (positions.min() < 0 or positions.max() >= n_genes):
        raise ValueError("supported_positions contains an out-of-range gene index")
    if np.unique(positions).shape[0] != positions.shape[0]:
        raise ValueError("supported_positions contains duplicate gene indices")
    output = np.zeros((values.shape[0], int(n_genes)), dtype=np.float32)
    output[:, positions] = values
    return output


def _rows(matrix, positions: np.ndarray) -> np.ndarray:
    selected = matrix[positions]
    if hasattr(selected, "toarray"):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_stpath_zero_shot(
    config_path: str | Path,
    *,
    gene_vocab_path: str | Path,
    model_weight_path: str | Path,
    split: str = "validation",
    n_masks_per_sample: int = 8,
    device_str: str = "cpu",
    allow_test: bool = False,
) -> dict:
    if split == "test" and not allow_test:
        raise ValueError("split='test' requires allow_test=True; use validation during development")
    if split not in {"validation", "test"}:
        raise ValueError(f"split must be validation or test, got {split!r}")
    if n_masks_per_sample <= 0:
        raise ValueError("n_masks_per_sample must be positive")
    for label, path in (("gene vocabulary", gene_vocab_path), ("model weights", model_weight_path)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"STPath {label} is missing: {path}")

    config = resolved_config(config_path)
    cfg = OmegaConf.create(config)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    split_ids = list(manifest[f"{split}_sample_ids"])
    if not split_ids:
        raise ValueError(f"dataset manifest has zero {split} samples")

    samples, preflight_report = load_and_preflight_samples(
        cfg, manifest, split_ids, expected_tile_encoder_provenance(config),
    )
    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(
        manifest, samples, strata, role=split,
        split_counts={split: int(n_masks_per_sample)},
        split_seeds={split: 700_000 if split == "validation" else 900_000},
    )
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, strata)
    dataset.validate_boundary_schedule()

    gene_names = list(manifest["gene_panel"])
    gene_panels = load_configured_gene_panels(config, manifest)
    device = torch.device(device_str)
    from src.models.stpath_encoder import STPathContextEncoder

    encoder = STPathContextEncoder(
        gene_names=gene_names,
        gene_voc_path=str(gene_vocab_path),
        model_weight_path=str(model_weight_path),
        organ_type="Others",  # replaced per sample immediately before every forward
        tech_type="Visium",
        hidden_dim=256,
        device=str(device),
        new_gene_encoder_type="none",
        pretrained=True,
        input_already_log1p=False,  # context values below are true raw counts
    ).to(device)
    encoder.eval()

    supported_positions = np.asarray(encoder._valid_gene_pos, dtype=np.int64)
    supported_position_set = set(supported_positions.tolist())
    supported_genes = [gene_names[i] for i in supported_positions]
    target_sum = float(config["data"].get("expression_target_sum", 1e4))
    normalized_items: list[dict] = []
    native_items: list[dict] = []
    normalized_panel_items: dict[str, list[dict]] = {name: [] for name in gene_panels}
    patient_ids: list[str] = []
    per_item_records: list[dict] = []

    for idx in range(len(dataset)):
        inputs, targets = dataset[idx]
        identity = dataset.item_identity(idx)
        sample = samples[identity["sample_id"]]
        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        position = {barcode: i for i, barcode in enumerate(obs_names)}
        context_pos = np.asarray([position[str(x)] for x in inputs.observed_barcodes], dtype=np.int64)
        query_pos = np.asarray([position[str(x)] for x in inputs.query_barcodes], dtype=np.int64)
        if "raw_counts" not in sample.adata.layers:
            raise ValueError(f"{sample.sample_id}: adata.layers['raw_counts'] is required for STPath")
        context_raw = _rows(sample.adata.layers["raw_counts"], context_pos)
        query_raw_log1p = np.log1p(_rows(sample.adata.layers["raw_counts"], query_pos))

        # Use original slide coordinates. SpatialFieldInputs coordinates are
        # centered/normalized for Gen3 and are not STPath's native contract.
        context_coords = sample.full_sample_coords[context_pos]
        query_coords = sample.full_sample_coords[query_pos]
        context_images = np.asarray(inputs.observed_gigapath_features, dtype=np.float32)
        query_images = np.zeros((query_pos.shape[0], context_images.shape[1]), dtype=np.float32)

        record = manifest["samples"][sample.sample_id]
        encoder.organ_type = str(record["organ"])
        encoder.tech_type = str(record.get("st_technology") or "Visium")
        with torch.no_grad():
            pred_supported_native = encoder(
                torch.as_tensor(context_coords, dtype=torch.float32, device=device),
                torch.as_tensor(context_raw, dtype=torch.float32, device=device),
                torch.as_tensor(query_coords, dtype=torch.float32, device=device),
                torch.as_tensor(context_images, dtype=torch.float32, device=device),
                torch.as_tensor(query_images, dtype=torch.float32, device=device),
                context_image_available=torch.as_tensor(
                    inputs.observed_image_available, dtype=torch.bool, device=device,
                ),
                query_image_available=None,  # keep literal zero query features; do not use a learned token
                return_official_predictions=True,
            ).detach().cpu().numpy().astype(np.float32)

        pred_supported_normalized = stpath_log1p_to_normalized_log1p(
            pred_supported_native, target_sum=target_sum,
        )
        pred_normalized = scatter_supported_genes(
            pred_supported_normalized, supported_positions, len(gene_names),
        )
        true_normalized = np.asarray(targets.query_expression, dtype=np.float32)
        normalized_metrics = per_item_reconstruction_metrics(pred_normalized, true_normalized)
        native_metrics = per_item_reconstruction_metrics(
            pred_supported_native, query_raw_log1p[:, supported_positions],
        )
        normalized_items.append(normalized_metrics)
        native_items.append(native_metrics)
        patient_ids.append(str(inputs.patient_id))

        panel_metrics = gene_panel_metrics(pred_normalized, true_normalized, gene_names, gene_panels)
        for panel_name, metrics in panel_metrics.items():
            normalized_panel_items[panel_name].append(metrics)
        per_item_records.append({
            "idx": idx,
            "sample_id": sample.sample_id,
            "patient_id": str(inputs.patient_id),
            "stratum": identity["stratum"],
            "query_fingerprint": identity["query_fingerprint"],
            "normalized_log1p": normalized_metrics,
            "native_log1p_raw_supported_genes": native_metrics,
            "gene_panels": panel_metrics,
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(dataset):
            print(f"[STPath evaluation {idx + 1}/{len(dataset)}] {sample.sample_id}", flush=True)

    return {
        "version": 1,
        "kind": "gen3_zero_shot_stpath_evaluation_report",
        "config_path": str(config_path),
        "split": split,
        "n_samples": len(split_ids),
        "n_items": len(dataset),
        "n_masks_per_sample": int(n_masks_per_sample),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(manifest),
        "mask_schedule_fingerprint": hashlib.sha256(
            json.dumps(schedule.reports, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "cache_preflight_report": preflight_report,
        "stpath": {
            "model_weight_path": str(model_weight_path),
            "model_weight_sha256": _sha256_file(model_weight_path),
            "gene_vocab_path": str(gene_vocab_path),
            "gene_vocab_sha256": _sha256_file(gene_vocab_path),
            "pretrained": True,
            "fine_tuned_on_current_data": False,
            "query_expression_visible": False,
            "query_image_mode": "zero_missing_tissue",
            "query_image_contract_warning": (
                "Released STPath normally consumes query-spot H&E; zero features are an "
                "out-of-distribution but leakage-free adaptation to the missing-tissue task."
            ),
            "native_output_space": "log1p_raw_counts",
            "primary_output_space": "normalize_total_then_log1p",
            "normalization_target_sum": target_sum,
            "supported_gene_count": len(supported_genes),
            "manifest_gene_count": len(gene_names),
            "supported_gene_fraction": len(supported_genes) / len(gene_names),
            "unsupported_genes": [g for i, g in enumerate(gene_names) if i not in supported_position_set],
        },
        "normalized_log1p_patient_aggregated_metrics": aggregate_patient_metrics(
            normalized_items, patient_ids,
        ),
        "native_log1p_raw_supported_genes_patient_aggregated_metrics": aggregate_patient_metrics(
            native_items, patient_ids,
        ),
        "per_panel_patient_aggregated_metrics": {
            panel: aggregate_patient_metrics(items, patient_ids)
            for panel, items in normalized_panel_items.items()
        },
        "per_item_records": per_item_records,
    }


def _save_report(report: dict, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stpath-gene-vocab", required=True)
    parser.add_argument("--stpath-model-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--n-masks-per-sample", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    report = evaluate_stpath_zero_shot(
        args.config,
        gene_vocab_path=args.stpath_gene_vocab,
        model_weight_path=args.stpath_model_weights,
        split=args.split,
        n_masks_per_sample=args.n_masks_per_sample,
        device_str=args.device,
        allow_test=args.allow_test,
    )
    path = _save_report(report, args.output)
    primary = report["normalized_log1p_patient_aggregated_metrics"]
    print(f"evaluation report saved to {path}")
    print(f"STPath normalized-log1p PCC patient_mean={primary['pcc']['patient_mean']:.6f}")
    print(f"STPath normalized-log1p RMSE patient_mean={primary['rmse']['patient_mean']:.6f}")
    print(f"STPath nonzero AUC patient_mean={primary['nonzero_auc']['patient_mean']:.6f}")


if __name__ == "__main__":
    main()
