#!/usr/bin/env python3
"""Batch-render standardized MK spatial maps for a tissue marker panel.

The model and requested held-out slides are loaded once, and each slide is
predicted once.  The requested marker genes are then rendered into an
organ/gene folder tree using the exact target/prediction/error visualization
and metrics used by ``plot_mk_gene_spatial_maps``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.scripts.plot_mk_gene_spatial_maps import (
    _load_model_and_samples,
    _render,
    _safe_name,
    _select_split_sample_ids,
    resolve_model_source,
)
from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train_conditional_wae import _stable_seed


def _resolve_panel(path: Path, gene_names: list[str], organs: set[str]) -> dict:
    raw = json.loads(path.read_text())
    panel = {
        "common": list(raw.get("common") or []) + list(raw.get("hard_immune_controls") or []),
        "by_organ": dict(raw.get("by_organ") or {}),
    }
    known = set(gene_names)
    resolved = {"common": [], "by_organ": {}, "missing": {}}
    for scope, requested in [("common", panel["common"])]:
        resolved[scope] = [gene for gene in requested if gene in known]
        resolved["missing"][scope] = [gene for gene in requested if gene not in known]
    by_casefold = {key.casefold(): key for key in panel["by_organ"]}
    for organ in sorted(organs):
        key = by_casefold.get(organ.casefold())
        requested = list(panel["by_organ"].get(key, [])) if key else []
        resolved["by_organ"][organ] = [gene for gene in requested if gene in known]
        resolved["missing"][organ] = [gene for gene in requested if gene not in known]
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name")
    parser.add_argument("--marker-panel", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--results-root", default="gen3_multiscale/results")
    parser.add_argument("--run-root")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--report")
    parser.add_argument(
        "--checkpoint-choice", choices=("best", "best_whole_slide", "latest"),
        default="best",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--prediction-role", choices=("point", "prior"), default="point")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--spot-size-scale", type=float, default=6.0,
        help="Scatter-marker area multiplier; marker-panel default is 6x the legacy plots.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.n_samples < 1 or args.chunk_size < 1 or args.spot_size_scale <= 0:
        parser.error("--n-samples, --chunk-size, and --spot-size-scale must be positive")

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
    available, sample_ids = _select_split_sample_ids(
        manifest, split=args.split, requested=list(args.sample_id),
    )
    genes = list(manifest["gene_panel"])
    organs = {str(manifest["samples"][sample]["organ"]) for sample in sample_ids}
    panel = _resolve_panel(Path(args.marker_panel).expanduser().resolve(), genes, organs)
    union = sorted(set(panel["common"]).union(*map(set, panel["by_organ"].values())))
    if not union:
        raise ValueError("none of the requested markers is in the model gene panel")
    positions = np.asarray([genes.index(gene) for gene in union], dtype=np.int64)

    print(f"Resolved model: {source.model_name}")
    print(f"Slides: {len(sample_ids)}/{len(available)}")
    print(f"Markers present: {len(union)}")
    model, samples, loaded_manifest, loaded_genes, _, weights_sha = _load_model_and_samples(
        source, split=args.split, device_name=args.device,
        allow_code_drift=args.allow_code_drift, sample_ids=sample_ids,
    )
    if loaded_genes != genes or loaded_manifest[f"{args.split}_sample_ids"] != available:
        raise RuntimeError("model load changed the manifest identity")
    identity = checkpoint_module.resolve_checkpoint_identity(source.checkpoint_path)
    step = int(checkpoint_module.load_training_state(identity.resolved_dir)["step"])

    records = []
    for index, sample_id in enumerate(sample_ids):
        result = predict_whole_slide(
            model, samples[sample_id], chunk_size=args.chunk_size,
            n_samples=args.n_samples if args.prediction_role == "prior" else 1,
            seed=_stable_seed(int(config["training"].get("seed", 0)), {
                "sample_id": sample_id,
                "stratum": "tissue_marker_panel",
                "query_fingerprint": "every_spot_exactly_once",
            }),
        )
        predicted = result["predictive_mean"] if args.prediction_role == "prior" else result["point_prediction"]
        records.append({
            "sample_id": sample_id,
            "organ": str(manifest["samples"][sample_id]["organ"]),
            "coords": np.asarray(result["coords"], dtype=np.float32),
            "target": np.asarray(result["target"][:, positions], dtype=np.float32),
            "prediction": predicted[:, positions].detach().cpu().numpy().astype(np.float32),
        })
        print(
            f"marker-panel inference: {index + 1}/{len(sample_ids)} "
            f"sample={sample_id} organ={records[-1]['organ']}", flush=True,
        )
        del result, predicted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_root = Path(args.output_root).expanduser().resolve() / _safe_name(source.model_name)
    output_root.mkdir(parents=True, exist_ok=True)
    gene_to_column = {gene: index for index, gene in enumerate(union)}
    scopes = {"common": (panel["common"], records)}
    scopes.update({
        organ: (markers, [row for row in records if row["organ"] == organ])
        for organ, markers in panel["by_organ"].items()
    })
    rendered = 0
    for scope, (markers, scoped_records) in scopes.items():
        for gene in markers:
            column = gene_to_column[gene]
            gene_records = [{
                "sample_id": row["sample_id"], "coords": row["coords"],
                "target": row["target"][:, column],
                "prediction": row["prediction"][:, column],
            } for row in scoped_records]
            if not gene_records:
                continue
            _render(
                gene_records,
                output_dir=output_root / _safe_name(scope) / _safe_name(gene),
                model_name=source.model_name, gene=gene,
                prediction_role=args.prediction_role, checkpoint_step=step,
                marker_size_scale=args.spot_size_scale,
            )
            rendered += len(gene_records)

    (output_root / "marker_panel_resolved.json").write_text(json.dumps({
        **panel,
        "model_name": source.model_name,
        "config_path": str(source.config_path),
        "checkpoint_path": str(source.checkpoint_path),
        "checkpoint_step": step,
        "weights_sha256": weights_sha,
        "split": args.split,
        "sample_ids": sample_ids,
        "prediction_role": args.prediction_role,
        "spot_size_scale": args.spot_size_scale,
        "target_gex_visible_to_model": False,
        "rendered_slide_gene_maps": rendered,
    }, indent=2, sort_keys=True))
    print(f"Saved {rendered} standardized slide/gene maps to {output_root}")


if __name__ == "__main__":
    main()
