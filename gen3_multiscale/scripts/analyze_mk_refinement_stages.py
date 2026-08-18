#!/usr/bin/env python3
"""Audit where spatial smoothing/template reuse enters a deterministic MK model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.conditional_wae.whole_slide import (
    predict_deterministic_refinement_stages,
)
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.template_reuse_diagnostics import (
    multiscale_pcc_per_gene,
    selected_template_diagnostics,
    variance_diagnostics,
)
from gen3_multiscale.scripts.analyze_mk_spatial_template_reuse import (
    _finite_mean,
    _finite_median,
    _macro_summary,
    _resolve_panels,
    _write_tsv,
)
from gen3_multiscale.scripts.plot_mk_gene_spatial_maps import (
    _load_model_and_samples,
    _select_split_sample_ids,
    resolve_model_source,
)
from gen3_multiscale.training import checkpoint as checkpoint_module


STAGE_ORDER = ("base", "within_only", "between_only", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mk_wb_parallel_gated")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--results-root", default="gen3_multiscale/results")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-choice", choices=("best", "best_whole_slide", "latest"),
                        default="best_whole_slide")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--metric-chunk-size", type=int, default=128)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--template-genes", type=int, default=256)
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if min(args.chunk_size, args.metric_chunk_size, args.k_neighbors) < 1:
        parser.error("chunk and neighbour counts must be positive")
    if args.template_genes < 2:
        parser.error("--template-genes must be at least two")

    repo = Path(args.repo).expanduser().resolve()
    results_root = Path(args.results_root).expanduser()
    if not results_root.is_absolute():
        results_root = repo / results_root
    source = resolve_model_source(
        repo=repo, results_root=results_root.resolve(), model_name=args.model_name,
        config_path=Path(args.config) if args.config else None,
        checkpoint_path=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        report_path=None, checkpoint_choice=args.checkpoint_choice,
    )
    config = resolved_config(str(source.config_path))
    params = (config.get("model") or {}).get("params") or {}
    if (config.get("model") or {}).get("arm") != "mk_wb_parallel_gated":
        raise ValueError("stage audit is defined for mk_wb_parallel_gated")
    if not bool(params.get("deterministic_only", False)):
        raise ValueError("stage audit requires the deterministic checkpoint")
    if params.get("structured_composition") != "parallel_gated":
        raise ValueError("stage audit requires parallel_gated composition")

    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    available, sample_ids = _select_split_sample_ids(
        manifest, split=args.split, requested=list(args.sample_id),
    )
    model, samples, loaded_manifest, gene_names, _, weights_sha256 = _load_model_and_samples(
        source, split=args.split, device_name=args.device,
        allow_code_drift=args.allow_code_drift, sample_ids=sample_ids,
    )
    if list(loaded_manifest[f"{args.split}_sample_ids"]) != available:
        raise RuntimeError("loaded split identity changed")
    panels = _resolve_panels(config, manifest, gene_names)
    identity = checkpoint_module.resolve_checkpoint_identity(source.checkpoint_path)
    step = int(checkpoint_module.load_training_state(identity.resolved_dir)["step"])

    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    slide_rows: list[dict] = []
    panel_rows: list[dict] = []
    gates = None
    for slide_index, sample_id in enumerate(sample_ids):
        result = predict_deterministic_refinement_stages(
            model, samples[sample_id], chunk_size=args.chunk_size,
        )
        target = np.asarray(result["target"], dtype=np.float32)
        coords = np.asarray(result["coords"], dtype=np.float32)
        gates = result["composition_gates"]
        progress = []
        for stage in STAGE_ORDER:
            predicted = result["stages"][stage].detach().cpu().numpy().astype(np.float32)
            pcc = multiscale_pcc_per_gene(
                predicted, target, coords, k_neighbors=args.k_neighbors,
                chunk_size=args.metric_chunk_size,
            )
            variance = variance_diagnostics(predicted, target)
            _, templates = selected_template_diagnostics(
                predicted, target, max_genes=args.template_genes,
            )
            exact = _finite_mean(pcc["exact"])
            blur1 = _finite_mean(pcc["blur1"])
            blur2 = _finite_mean(pcc["blur2"])
            slide_rows.append({
                "stage": stage, "sample_id": sample_id,
                "organ": str(manifest["samples"][sample_id]["organ"]),
                "n_spots": int(predicted.shape[0]),
                "exact_pcc_mean": exact, "blur1_pcc_mean": blur1,
                "blur2_pcc_mean": blur2, "blur1_gain": blur1 - exact,
                "blur2_gain": blur2 - exact,
                "rmse": float(np.sqrt(np.mean(np.square(predicted - target)))),
                "predicted_to_target_std_ratio_median": _finite_median(
                    variance["predicted_to_target_std_ratio"]
                ),
                **templates,
            })
            for panel, indices in panels.items():
                panel_exact = _finite_mean(pcc["exact"][indices])
                panel_blur1 = _finite_mean(pcc["blur1"][indices])
                panel_blur2 = _finite_mean(pcc["blur2"][indices])
                panel_rows.append({
                    "stage": stage, "sample_id": sample_id,
                    "organ": str(manifest["samples"][sample_id]["organ"]),
                    "panel": panel, "n_genes": int(indices.size),
                    "exact_pcc_mean": panel_exact,
                    "blur1_pcc_mean": panel_blur1,
                    "blur2_pcc_mean": panel_blur2,
                    "blur1_gain": panel_blur1 - panel_exact,
                    "blur2_gain": panel_blur2 - panel_exact,
                    "predicted_to_target_std_ratio_median": _finite_median(
                        variance["predicted_to_target_std_ratio"][indices]
                    ),
                })
            progress.append(f"{stage}={exact:.4f}")
        print(
            f"stage audit: {slide_index + 1}/{len(sample_ids)} sample={sample_id} "
            + " ".join(progress), flush=True,
        )
        del result, target
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_tsv(output / "stage_slide_diagnostics.tsv", slide_rows)
    _write_tsv(output / "stage_panel_slide_diagnostics.tsv", panel_rows)
    _write_tsv(output / "stage_macro_summary.tsv", _macro_summary(slide_rows, "stage"))
    stage_panel_macro = []
    for stage in STAGE_ORDER:
        stage_rows = _macro_summary(
            [row for row in panel_rows if row["stage"] == stage], "panel"
        )
        for row in stage_rows:
            row["stage"] = stage
        stage_panel_macro.extend(stage_rows)
    _write_tsv(output / "stage_panel_macro_summary.tsv", stage_panel_macro)
    provenance = {
        "kind": "mk_deterministic_refinement_stage_audit", "version": 1,
        "model_name": source.model_name, "config_path": str(source.config_path),
        "checkpoint_path": str(identity.resolved_dir), "checkpoint_step": step,
        "weights_sha256": weights_sha256, "split": args.split,
        "sample_ids": sample_ids, "target_gex_visible_to_model": False,
        "stages": list(STAGE_ORDER),
        "composition_gates": None if gates is None else gates.cpu().tolist(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    print(f"Stage audit saved to {output}", flush=True)


if __name__ == "__main__":
    main()
