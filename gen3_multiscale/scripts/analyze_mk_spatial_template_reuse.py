#!/usr/bin/env python3
"""Analyze smoothing, amplitude shrinkage, rank and template reuse on MK slides.

The command reruns cached-feature, H&E-only whole-slide inference for one
trained model.  It processes one held-out slide at a time and persists only
compact TSV/JSON diagnostics, never full all-gene prediction matrices.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.template_reuse_diagnostics import (
    multiscale_pcc_per_gene,
    selected_template_diagnostics,
    variance_diagnostics,
)
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.scripts.plot_mk_gene_spatial_maps import (
    _load_model_and_samples,
    _select_split_sample_ids,
    resolve_model_source,
)
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train_conditional_wae import _stable_seed


def _finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _finite_fraction_below(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values < threshold)) if values.size else float("nan")


def _resolve_panels(config: dict, manifest: dict, gene_names: list[str]) -> dict[str, np.ndarray]:
    panels = {"all_genes": np.arange(len(gene_names), dtype=np.int64)}
    artifact_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
    if not artifact_path:
        return panels
    artifact = load_train_derived_gene_panels(artifact_path, manifest)
    positions = {gene: index for index, gene in enumerate(gene_names)}
    for name, genes in artifact.get("panels", {}).items():
        indices = [positions[gene] for gene in genes if gene in positions]
        if indices:
            panels[str(name)] = np.asarray(indices, dtype=np.int64)
    return panels


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _macro_summary(rows: list[dict], group_key: str) -> list[dict]:
    result = []
    for group in sorted({str(row[group_key]) for row in rows}):
        selected = [row for row in rows if str(row[group_key]) == group]
        entry = {group_key: group, "n_slides": len(selected)}
        for key in selected[0]:
            if key in {group_key, "sample_id", "organ"}:
                continue
            values = []
            for row in selected:
                try:
                    values.append(float(row[key]))
                except (TypeError, ValueError):
                    pass
            if values:
                entry[f"{key}_macro_mean"] = _finite_mean(np.asarray(values))
        result.append(entry)
    return result


def _slide_associations(slide_rows: list[dict]) -> list[dict]:
    outcomes = (
        "exact_pcc_mean", "blur1_gain", "blur2_gain",
        "predicted_to_target_std_ratio_median", "entropy_effective_rank_ratio",
        "same_gene_top1_fraction",
    )
    attributes = (
        "n_spots", "image_available_fraction", "image_embedding_mean_norm",
        "image_embedding_within_slide_rms",
    )
    rows = []
    for outcome in outcomes:
        for attribute in attributes:
            left = np.asarray([row[outcome] for row in slide_rows], dtype=np.float64)
            right = np.asarray([row[attribute] for row in slide_rows], dtype=np.float64)
            finite = np.isfinite(left) & np.isfinite(right)
            rho = float("nan")
            if finite.sum() >= 3 and np.std(left[finite]) > 0 and np.std(right[finite]) > 0:
                rho = float(spearmanr(left[finite], right[finite]).statistic)
            rows.append({
                "outcome": outcome, "slide_attribute": attribute,
                "spearman_rho": rho, "n_slides": int(finite.sum()),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--results-root", default="gen3_multiscale/results")
    parser.add_argument("--run-root")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--report")
    parser.add_argument(
        "--checkpoint-choice", choices=("best", "best_whole_slide", "latest"),
        default="best_whole_slide",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prediction-role", choices=("point", "prior"), default="point")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--metric-chunk-size", type=int, default=128)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--template-genes", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if min(
        args.n_samples, args.chunk_size, args.metric_chunk_size,
        args.k_neighbors, args.template_genes,
    ) < 1 or args.template_genes < 2:
        parser.error("counts must be positive and --template-genes must be >=2")

    repo = Path(args.repo).expanduser().resolve()
    results_root = Path(args.run_root or args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = repo / results_root
    results_root = results_root.resolve()
    source = resolve_model_source(
        repo=repo, results_root=results_root, model_name=args.model_name,
        config_path=Path(args.config) if args.config else None,
        checkpoint_path=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        report_path=Path(args.report) if args.report else None,
        checkpoint_choice=args.checkpoint_choice,
    )
    config = resolved_config(str(source.config_path))
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    available_ids, sample_ids = _select_split_sample_ids(
        manifest, split=args.split, requested=list(args.sample_id),
    )
    model, samples, loaded_manifest, gene_names, _, weights_sha256 = _load_model_and_samples(
        source, split=args.split, device_name=args.device,
        allow_code_drift=args.allow_code_drift, sample_ids=sample_ids,
    )
    if list(loaded_manifest[f"{args.split}_sample_ids"]) != available_ids:
        raise RuntimeError("loaded split identity changed")
    panels = _resolve_panels(config, manifest, gene_names)
    identity = checkpoint_module.resolve_checkpoint_identity(source.checkpoint_path)
    step = int(checkpoint_module.load_training_state(identity.resolved_dir)["step"])
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root else results_root / "spatial_template_reuse"
    )
    output_dir = output_root / f"{source.model_name}__{weights_sha256[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    slide_rows: list[dict] = []
    panel_rows: list[dict] = []
    per_gene_rows: list[dict] = []
    detail_rows: list[dict] = []
    for slide_index, sample_id in enumerate(sample_ids):
        sample = samples[sample_id]
        result = predict_whole_slide(
            model, sample, chunk_size=args.chunk_size,
            n_samples=args.n_samples if args.prediction_role == "prior" else 1,
            seed=_stable_seed(int(config["training"].get("seed", 0)), {
                "sample_id": sample_id,
                "stratum": "spatial_template_reuse_diagnostic",
                "query_fingerprint": "every_spot_exactly_once",
            }),
        )
        predicted_tensor = (
            result["predictive_mean"] if args.prediction_role == "prior"
            else result["point_prediction"]
        )
        predicted = predicted_tensor.detach().cpu().numpy().astype(np.float32)
        target = np.asarray(result["target"], dtype=np.float32)
        coords = np.asarray(result["coords"], dtype=np.float32)
        pcc = multiscale_pcc_per_gene(
            predicted, target, coords, k_neighbors=args.k_neighbors,
            chunk_size=args.metric_chunk_size,
        )
        variance = variance_diagnostics(predicted, target)
        selected_indices, templates = selected_template_diagnostics(
            predicted, target, max_genes=args.template_genes,
        )
        features = np.asarray(sample.precomputed_spot_features, dtype=np.float64)
        feature_mean = features.mean(axis=0)
        feature_centered = features - feature_mean
        exact_mean = _finite_mean(pcc["exact"])
        slide_row = {
            "sample_id": sample_id,
            "organ": str(manifest["samples"][sample_id]["organ"]),
            "n_spots": int(predicted.shape[0]),
            "n_genes": int(predicted.shape[1]),
            "exact_pcc_mean": exact_mean,
            "blur1_pcc_mean": _finite_mean(pcc["blur1"]),
            "blur2_pcc_mean": _finite_mean(pcc["blur2"]),
            "blur1_gain": _finite_mean(pcc["blur1"]) - exact_mean,
            "blur2_gain": _finite_mean(pcc["blur2"]) - exact_mean,
            "predicted_to_target_std_ratio_median": _finite_median(
                variance["predicted_to_target_std_ratio"]
            ),
            "fraction_genes_std_ratio_below_0_25": _finite_fraction_below(
                variance["predicted_to_target_std_ratio"], 0.25,
            ),
            "fraction_genes_std_ratio_below_0_50": _finite_fraction_below(
                variance["predicted_to_target_std_ratio"], 0.50,
            ),
            "image_available_fraction": float(np.mean(sample.image_source_available)),
            "image_embedding_mean_norm": float(np.linalg.norm(feature_mean)),
            "image_embedding_within_slide_rms": float(
                np.sqrt(np.mean(np.square(feature_centered)))
            ),
            **templates,
        }
        slide_rows.append(slide_row)
        detail_rows.append({
            "sample_id": sample_id,
            "selection": "top target-variance genes on this held-out slide; diagnostic only",
            "selected_gene_indices": selected_indices.tolist(),
            "selected_gene_names": [gene_names[index] for index in selected_indices],
        })
        for panel_name, indices in panels.items():
            exact = _finite_mean(pcc["exact"][indices])
            blur1 = _finite_mean(pcc["blur1"][indices])
            blur2 = _finite_mean(pcc["blur2"][indices])
            ratios = variance["predicted_to_target_std_ratio"][indices]
            panel_rows.append({
                "sample_id": sample_id, "organ": slide_row["organ"],
                "panel": panel_name, "n_genes": int(indices.size),
                "exact_pcc_mean": exact, "blur1_pcc_mean": blur1,
                "blur2_pcc_mean": blur2, "blur1_gain": blur1 - exact,
                "blur2_gain": blur2 - exact,
                "predicted_to_target_std_ratio_median": _finite_median(ratios),
            })
        for gene_index, gene in enumerate(gene_names):
            per_gene_rows.append({
                "sample_id": sample_id, "organ": slide_row["organ"], "gene": gene,
                "exact_pcc": float(pcc["exact"][gene_index]),
                "blur1_pcc": float(pcc["blur1"][gene_index]),
                "blur2_pcc": float(pcc["blur2"][gene_index]),
                "blur1_gain": float(pcc["blur1"][gene_index] - pcc["exact"][gene_index]),
                "blur2_gain": float(pcc["blur2"][gene_index] - pcc["exact"][gene_index]),
                "target_std": float(variance["target_std"][gene_index]),
                "predicted_std": float(variance["predicted_std"][gene_index]),
                "predicted_to_target_std_ratio": float(
                    variance["predicted_to_target_std_ratio"][gene_index]
                ),
            })
        print(
            f"template diagnostic: {slide_index + 1}/{len(sample_ids)} "
            f"sample={sample_id} exact={exact_mean:.4f} "
            f"blur2={slide_row['blur2_pcc_mean']:.4f} "
            f"rank_ratio={slide_row['entropy_effective_rank_ratio']:.3f}",
            flush=True,
        )
        del result, predicted_tensor, predicted, target, pcc, variance, feature_centered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_tsv(output_dir / "slide_diagnostics.tsv", slide_rows)
    _write_tsv(output_dir / "panel_slide_diagnostics.tsv", panel_rows)
    _write_tsv(output_dir / "per_gene_slide_diagnostics.tsv", per_gene_rows)
    _write_tsv(output_dir / "panel_macro_summary.tsv", _macro_summary(panel_rows, "panel"))
    _write_tsv(output_dir / "slide_attribute_associations.tsv", _slide_associations(slide_rows))
    (output_dir / "template_gene_selections.json").write_text(
        json.dumps(detail_rows, indent=2, sort_keys=True)
    )
    provenance = {
        "kind": "mk_spatial_template_reuse_diagnostics",
        "version": 1,
        "model_name": source.model_name,
        "config_path": str(source.config_path),
        "checkpoint_path": str(source.checkpoint_path),
        "checkpoint_step": step,
        "weights_sha256": weights_sha256,
        "split": args.split,
        "sample_ids": sample_ids,
        "prediction_role": args.prediction_role,
        "target_gex_visible_to_model": False,
        "k_neighbors": args.k_neighbors,
        "blur_definition": "self-plus-kNN mean applied to prediction and target",
        "template_gene_count": args.template_genes,
        "template_selection_is_benchmark_metric": False,
        "output_files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"Diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
