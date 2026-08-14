"""Frozen released-STPath baselines for the two supervisor prediction tasks.

No parameter is trained or selected here.  Both modes reuse the Gen3 fixed
mask schedule and report in the same normalized-log1p space as the conditional
WAE/flow experiments:

``he_to_st``
    All expression tokens on a slide are masked; real H&E and coordinates are
    visible.  The slide prediction is computed once and indexed by each fixed
    query mask.

``he_plus_st_to_st``
    Real H&E is visible at context and query spots; expression is visible only
    outside the query mask.  Query expression is always represented by
    STPath's learned mask token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import (
    load_configured_gene_panels,
    per_item_reconstruction_metrics,
)
from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics,
    comparable_expression_metrics,
    gene_panel_metrics,
    nonzero_auc,
    resolve_gene_panels,
)
from gen3_multiscale.evaluation.schedule_contract import fixed_mask_evaluation_metadata
from gen3_multiscale.evaluation.stpath_zero_shot_evaluator import (
    _rows,
    _sha256_file,
    scatter_supported_genes,
    stpath_log1p_to_normalized_log1p,
)
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    dataset_manifest_fingerprint,
    expected_tile_encoder_provenance,
    resolved_config,
)


TASKS = ("he_to_st", "he_plus_st_to_st")


def _whole_slide_panel_metrics(
    predicted: np.ndarray, target: np.ndarray,
    panel_indices: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    def score(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        return {
            **comparable_expression_metrics(left, right),
            "auc": nonzero_auc(left, right),
        }

    result = {"all_genes": score(predicted, target)}
    for panel, indices in panel_indices.items():
        result[panel] = score(predicted[:, indices], target[:, indices])
    return result


def _stpath_task_positions(
    task: str, *, n_spots: int, context_pos: np.ndarray, query_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return model context rows, model query rows, and rows to score.

    For H&E-only prediction every slide row is a masked-expression query, and
    ``score_pos`` maps the fixed mask back into that full-slide prediction.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if task == "he_to_st":
        return (
            np.empty(0, dtype=np.int64),
            np.arange(int(n_spots), dtype=np.int64),
            np.asarray(query_pos, dtype=np.int64),
        )
    return (
        np.asarray(context_pos, dtype=np.int64),
        np.asarray(query_pos, dtype=np.int64),
        np.arange(len(query_pos), dtype=np.int64),
    )


def evaluate_stpath_supervisor_zero_shot(
    config_path: str | Path,
    *,
    task: str,
    gene_vocab_path: str | Path,
    model_weight_path: str | Path,
    manifest_path: str | Path | None = None,
    train_gene_panels_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    hest_data_dir: str | Path | None = None,
    split: str = "validation",
    n_masks_per_sample: int = 8,
    device_str: str = "cpu",
    allow_test: bool = False,
) -> dict:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
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
    data_config = config.setdefault("data", {})
    if manifest_path is not None:
        data_config["gen3_manifest_path"] = str(Path(manifest_path).expanduser().resolve())
    if hest_data_dir is not None:
        data_config["hest_data_dir"] = str(Path(hest_data_dir).expanduser().resolve())
    # Released STPath was trained with GigaPath morphology embeddings.  Force
    # that contract here so an otherwise compatible-width UNI2 cache cannot
    # silently enter the benchmark through a comparison config.
    data_config["image_encoder"] = "gigapath"
    data_config["retain_patches_in_memory"] = False
    data_config["use_histology_features"] = False
    data_config["slide_context_source"] = "disabled"
    if cache_dir is not None:
        data_config["gen3_spot_feature_cache_dir"] = str(
            Path(cache_dir).expanduser().resolve()
        )
    if train_gene_panels_path is not None:
        evaluation_config = config.setdefault("evaluation", {})
        # The explicit expanded-cohort artifact replaces, rather than merges
        # with, panels embedded in an older comparison config.
        evaluation_config["gene_panels"] = {}
        evaluation_config["train_gene_panel_artifact"] = str(
            Path(train_gene_panels_path).expanduser().resolve()
        )
    cfg = OmegaConf.create(config)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    if hest_data_dir is not None:
        manifest = dict(manifest)
        manifest["hest_data_dir"] = str(Path(hest_data_dir).expanduser().resolve())
    split_ids = list(manifest[f"{split}_sample_ids"])
    if not split_ids:
        raise ValueError(f"dataset manifest has zero {split} samples")

    samples, preflight_report = load_and_preflight_samples(
        cfg, manifest, split_ids, expected_tile_encoder_provenance(config),
    )
    strata = config["masking"]["strata"]
    schedule = build_gen3_mask_schedule(
        manifest,
        samples,
        strata,
        role=split,
        split_counts={split: int(n_masks_per_sample)},
        split_seeds={split: 700_000 if split == "validation" else 900_000},
    )
    dataset = Gen3SpatialFieldDataset(manifest, samples, schedule, strata)
    dataset.validate_boundary_schedule()

    gene_names = list(manifest["gene_panel"])
    gene_panels = load_configured_gene_panels(config, manifest)
    panel_indices, panel_metadata = (
        resolve_gene_panels(gene_names, gene_panels) if gene_panels else ({}, {})
    )
    device = torch.device(device_str)
    from src.models.stpath_encoder import STPathContextEncoder

    encoder = STPathContextEncoder(
        gene_names=gene_names,
        gene_voc_path=str(gene_vocab_path),
        model_weight_path=str(model_weight_path),
        organ_type="Others",
        tech_type="Visium",
        hidden_dim=256,
        device=str(device),
        new_gene_encoder_type="none",
        pretrained=True,
        input_already_log1p=False,
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
    he_only_prediction_cache: dict[str, np.ndarray] = {}

    for idx in range(len(dataset)):
        inputs, targets = dataset[idx]
        identity = dataset.item_identity(idx)
        sample = samples[identity["sample_id"]]
        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        barcode_to_pos = {barcode: i for i, barcode in enumerate(obs_names)}
        context_pos = np.asarray(
            [barcode_to_pos[str(x)] for x in inputs.observed_barcodes], dtype=np.int64,
        )
        query_pos = np.asarray(
            [barcode_to_pos[str(x)] for x in inputs.query_barcodes], dtype=np.int64,
        )
        if "raw_counts" not in sample.adata.layers:
            raise ValueError(f"{sample.sample_id}: adata.layers['raw_counts'] is required for STPath")

        model_context_pos, model_query_pos, score_pos = _stpath_task_positions(
            task,
            n_spots=len(obs_names),
            context_pos=context_pos,
            query_pos=query_pos,
        )
        query_raw_log1p = np.log1p(_rows(sample.adata.layers["raw_counts"], query_pos))

        if task == "he_to_st" and sample.sample_id in he_only_prediction_cache:
            pred_supported_native = he_only_prediction_cache[sample.sample_id][score_pos]
        else:
            context_raw = _rows(sample.adata.layers["raw_counts"], model_context_pos)
            context_coords = np.asarray(sample.full_sample_coords[model_context_pos], dtype=np.float32)
            model_query_coords = np.asarray(sample.full_sample_coords[model_query_pos], dtype=np.float32)
            context_images = np.asarray(
                sample.precomputed_spot_features[model_context_pos], dtype=np.float32,
            )
            query_images = np.asarray(
                sample.precomputed_spot_features[model_query_pos], dtype=np.float32,
            )
            context_available = np.asarray(
                sample.image_source_available[model_context_pos], dtype=bool,
            )
            query_available = np.asarray(
                sample.image_source_available[model_query_pos], dtype=bool,
            )

            record = manifest["samples"][sample.sample_id]
            encoder.organ_type = str(record["organ"])
            encoder.tech_type = str(record.get("st_technology") or "Visium")
            with torch.no_grad():
                full_prediction = encoder(
                    torch.as_tensor(context_coords, dtype=torch.float32, device=device),
                    torch.as_tensor(context_raw, dtype=torch.float32, device=device),
                    torch.as_tensor(model_query_coords, dtype=torch.float32, device=device),
                    torch.as_tensor(context_images, dtype=torch.float32, device=device),
                    torch.as_tensor(query_images, dtype=torch.float32, device=device),
                    context_image_available=torch.as_tensor(
                        context_available, dtype=torch.bool, device=device,
                    ),
                    query_image_available=torch.as_tensor(
                        query_available, dtype=torch.bool, device=device,
                    ),
                    return_official_predictions=True,
                ).detach().cpu().numpy().astype(np.float32)
            if task == "he_to_st":
                he_only_prediction_cache[sample.sample_id] = full_prediction
            pred_supported_native = full_prediction[score_pos]

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
            print(f"[STPath {task} evaluation {idx + 1}/{len(dataset)}] {sample.sample_id}", flush=True)

    whole_slide_point_evaluation = None
    if task == "he_to_st":
        whole_slide_records = []
        for sample_id in split_ids:
            sample = samples[sample_id]
            pred_supported_native = he_only_prediction_cache[sample_id]
            pred_supported_normalized = stpath_log1p_to_normalized_log1p(
                pred_supported_native, target_sum=target_sum,
            )
            predicted = scatter_supported_genes(
                pred_supported_normalized, supported_positions, len(gene_names),
            )
            target = _rows(sample.adata.X, np.arange(sample.adata.n_obs, dtype=np.int64))
            whole_slide_records.append({
                "sample_id": sample_id,
                "patient_id": str(sample.patient_id),
                "organ": str(manifest["samples"][sample_id]["organ"]),
                "technology": str(
                    manifest["samples"][sample_id].get("tech")
                    or manifest["samples"][sample_id].get("st_technology")
                    or "unknown"
                ),
                "n_spots": int(sample.adata.n_obs),
                "point_metrics": _whole_slide_panel_metrics(
                    predicted, target, panel_indices,
                ),
            })
        whole_slide_patients = [row["patient_id"] for row in whole_slide_records]
        point_aggregated = {
            panel: aggregate_patient_metrics(
                [row["point_metrics"][panel] for row in whole_slide_records],
                whole_slide_patients,
            )
            for panel in whole_slide_records[0]["point_metrics"]
        }
        whole_slide_point_evaluation = {
            "scope": "all_held_out_slides_every_spot_exactly_once",
            "primary_prediction": "pretrained_stpath_h_and_e_only",
            "target_gex_visible_to_model": False,
            "surrounding_gex_visible_to_model": False,
            "target_space": "normalize_total_then_log1p",
            "gene_panel_metadata": panel_metadata,
            "per_slide_records": whole_slide_records,
            "point_metrics_patient_aggregated": point_aggregated,
        }

    return {
        "version": 1,
        "kind": "stpath_supervisor_zero_shot_evaluation_report",
        "task": task,
        "config_path": str(config_path),
        "manifest_path": str(config["data"]["gen3_manifest_path"]),
        "train_gene_panels_path": (
            str(Path(train_gene_panels_path).expanduser().resolve())
            if train_gene_panels_path is not None else None
        ),
        "gigapath_cache_dir": str(data_config.get("gen3_spot_feature_cache_dir", "")),
        **fixed_mask_evaluation_metadata(
            split=split, sample_ids=split_ids, strata=strata,
            masks_per_stratum_per_sample=n_masks_per_sample,
            actual_n_items=len(dataset),
        ),
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
            "surrounding_expression_visible": task == "he_plus_st_to_st",
            "query_image_mode": "visible_real",
            "all_expression_tokens_masked": task == "he_to_st",
            "he_only_full_slide_prediction_reused_across_masks": task == "he_to_st",
            "native_output_space": "log1p_raw_counts",
            "primary_output_space": "normalize_total_then_log1p",
            "normalization_target_sum": target_sum,
            "supported_gene_count": len(supported_genes),
            "manifest_gene_count": len(gene_names),
            "supported_gene_fraction": len(supported_genes) / len(gene_names),
            "unsupported_genes": [
                gene for i, gene in enumerate(gene_names) if i not in supported_position_set
            ],
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
        "whole_slide_point_evaluation": whole_slide_point_evaluation,
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
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--stpath-gene-vocab", required=True)
    parser.add_argument("--stpath-model-weights", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--train-gene-panels")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--n-masks-per-stratum-per-sample", "--n-masks-per-sample",
        dest="n_masks_per_sample", type=int, default=8,
        help="Masks for each configured stratum of each sample; legacy spelling retained.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    report = evaluate_stpath_supervisor_zero_shot(
        args.config,
        task=args.task,
        gene_vocab_path=args.stpath_gene_vocab,
        model_weight_path=args.stpath_model_weights,
        manifest_path=args.manifest,
        train_gene_panels_path=args.train_gene_panels,
        cache_dir=args.cache_dir,
        hest_data_dir=args.hest_data_dir,
        split=args.split,
        n_masks_per_sample=args.n_masks_per_sample,
        device_str=args.device,
        allow_test=args.allow_test,
    )
    path = _save_report(report, args.output)
    primary = report["normalized_log1p_patient_aggregated_metrics"]
    print(f"evaluation report saved to {path}")
    print(f"task={report['task']}")
    print(f"STPath normalized-log1p PCC patient_mean={primary['pcc']['patient_mean']:.6f}")
    print(f"STPath normalized-log1p RMSE patient_mean={primary['rmse']['patient_mean']:.6f}")
    print(f"STPath nonzero AUC patient_mean={primary['nonzero_auc']['patient_mean']:.6f}")


if __name__ == "__main__":
    main()
