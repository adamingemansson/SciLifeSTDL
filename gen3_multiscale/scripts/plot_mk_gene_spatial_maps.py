#!/usr/bin/env python3
"""Create standardized whole-slide target/prediction/error maps for one gene.

The command resolves a trained MK conditional-WAE/deterministic arm, performs
exact whole-slide H&E-only inference, and writes one three-panel PNG per held-
out slide plus a multipage PDF and a TSV of slide-level metrics.  A compact
NPZ cache stores only the requested gene, coordinates and provenance; full
17k-gene prediction matrices are never persisted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae.contract import (
    static_audit_conditional_wae_config,
)
from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide
from gen3_multiscale.config_identity import resolved_config
from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import expected_tile_encoder_provenance
from gen3_multiscale.training.train_conditional_wae import (
    _build_model,
    _manifest,
    _stable_seed,
    _verify_resume,
)


@dataclass(frozen=True)
class ModelSource:
    model_name: str
    config_path: Path
    checkpoint_path: Path
    report_path: Path | None


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not cleaned:
        raise ValueError(f"cannot form a safe path component from {value!r}")
    return cleaned


def _absolute_from_repo(value: str | Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def _checkpoint_for_config(config: dict, choice: str, repo: Path) -> Path:
    root = _absolute_from_repo(config["training"]["checkpoint_dir"], repo)
    if choice == "latest":
        return root
    return root / choice


def _checkpoint_is_complete(path: Path) -> bool:
    try:
        identity = checkpoint_module.resolve_checkpoint_identity(path)
    except Exception:
        return False
    return bool(identity.weights_sha256 and (identity.resolved_dir / "model_config.json").is_file())


def _load_config_model_name(path: Path) -> tuple[dict, str]:
    config = resolved_config(str(path))
    name = str((config.get("model") or {}).get("arm") or "")
    if not name:
        raise ValueError(f"{path}: model.arm is missing")
    return config, name


def _require_he_to_st_config(config: dict, path: Path) -> None:
    model = config.get("model") or {}
    if model.get("task") != "he_to_st" or bool(model.get("include_observed_gex", False)):
        raise ValueError(
            f"{path}: spatial gene maps require H&E-only he_to_st inference with "
            "query and surrounding GEX hidden"
        )


def resolve_model_source(
    *, repo: Path, results_root: Path, model_name: str | None,
    config_path: Path | None, checkpoint_path: Path | None,
    report_path: Path | None, checkpoint_choice: str,
) -> ModelSource:
    """Resolve one exact config/checkpoint without silently mixing runs."""
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report = json.loads(report_path.read_text())
        if report.get("kind") != "conditional_wae_supervisor_evaluation":
            raise ValueError(f"{report_path}: unsupported report kind {report.get('kind')!r}")
        config_path = _absolute_from_repo(report["config_path"], repo)
        checkpoint_path = _absolute_from_repo(report["checkpoint_dir"], repo)

    if config_path is not None:
        config_path = config_path.expanduser().resolve()
        config, resolved_name = _load_config_model_name(config_path)
        if model_name is not None and model_name != resolved_name:
            raise ValueError(
                f"requested model {model_name!r}, but config contains {resolved_name!r}"
            )
        model_name = resolved_name
        if checkpoint_path is None:
            checkpoint_path = _checkpoint_for_config(config, checkpoint_choice, repo)
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not _checkpoint_is_complete(checkpoint_path):
            raise ValueError(f"checkpoint is incomplete: {checkpoint_path}")
        return ModelSource(model_name, config_path, checkpoint_path, report_path)

    if model_name is None:
        raise ValueError("provide --model-name, --config, or --report")

    candidates: list[tuple[int, Path, Path]] = []
    for path in results_root.rglob("*.yaml"):
        try:
            config, candidate_name = _load_config_model_name(path)
            if candidate_name != model_name:
                continue
            checkpoint = _checkpoint_for_config(config, checkpoint_choice, repo)
            if not _checkpoint_is_complete(checkpoint):
                continue
            identity = checkpoint_module.resolve_checkpoint_identity(checkpoint)
            step = int(checkpoint_module.load_training_state(identity.resolved_dir)["step"])
            candidates.append((step, path.resolve(), checkpoint.resolve()))
        except Exception:
            continue
    if not candidates:
        raise ValueError(
            f"no complete {checkpoint_choice!r} checkpoint found for model {model_name!r} "
            f"under {results_root}"
        )
    candidates.sort(key=lambda item: (item[0], item[1].stat().st_mtime, str(item[1])))
    best_step = candidates[-1][0]
    tied = [entry for entry in candidates if entry[0] == best_step]
    selected = max(tied, key=lambda item: (item[1].stat().st_mtime, str(item[1])))
    return ModelSource(model_name, selected[1], selected[2], None)


def _resolve_gene(gene_names: list[str], requested: str) -> tuple[str, int]:
    if requested in gene_names:
        return requested, gene_names.index(requested)
    matches = [gene for gene in gene_names if gene.casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0], gene_names.index(matches[0])
    raise ValueError(f"gene {requested!r} is not uniquely present in the model gene panel")


def _load_model_and_samples(
    source: ModelSource, *, split: str, device_name: str,
    allow_code_drift: bool,
) -> tuple[torch.nn.Module, dict, dict, list[str], torch.device, str]:
    config = resolved_config(str(source.config_path))
    _require_he_to_st_config(config, source.config_path)
    static_audit_conditional_wae_config(config)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    sample_ids = list(manifest[f"{split}_sample_ids"])
    if not sample_ids:
        raise ValueError(f"manifest has no {split} samples")
    cfg_om = OmegaConf.create(config)
    samples, preflight = load_and_preflight_samples(
        cfg_om, manifest, sample_ids, expected_tile_encoder_provenance(config),
    )

    old_manifest = checkpoint_module.load_checkpoint_run_manifest(source.checkpoint_path)
    if old_manifest is None:
        raise ValueError(f"{source.checkpoint_path}: no bundle-bound run manifest")
    current_manifest = _manifest(config, manifest, preflight)
    old_cache = old_manifest.get("cache_content_by_sample") or {}
    for sample_id, identity in preflight["cache_content_by_sample"].items():
        if sample_id in old_cache and old_cache[sample_id] != identity:
            raise ValueError(f"{sample_id}: cache differs from the training checkpoint")
    current_manifest["cache_content_fingerprint"] = old_manifest[
        "cache_content_fingerprint"
    ]
    current_manifest["cache_content_by_sample"] = old_cache
    _verify_resume(old_manifest, current_manifest, allow_code_drift=allow_code_drift)

    gene_names = list(manifest["gene_panel"])
    checkpoint_module.verify_gene_names(source.checkpoint_path, gene_names)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = _build_model(config, len(gene_names), gene_names=gene_names).to(device)
    checkpoint_module.load_trainable_state(
        model, source.checkpoint_path,
        reconstructed_buffer_names={"per_gene_scale"},
    )
    model.eval()
    identity = checkpoint_module.resolve_checkpoint_identity(source.checkpoint_path)
    if identity.weights_sha256 is None:
        raise ValueError("resolved checkpoint has no weight hash")
    return model, samples, manifest, gene_names, device, identity.weights_sha256


def _cache_metadata(
    source: ModelSource, *, weights_sha256: str, gene: str, split: str,
    sample_ids: list[str], prediction_role: str, n_samples: int,
) -> dict:
    return {
        "version": 1,
        "kind": "mk_single_gene_whole_slide_prediction_cache",
        "model_name": source.model_name,
        "config_path": str(source.config_path),
        "checkpoint_path": str(source.checkpoint_path),
        "weights_sha256": weights_sha256,
        "gene": gene,
        "split": split,
        "sample_ids": sample_ids,
        "prediction_role": prediction_role,
        "n_samples": int(n_samples),
        "target_space": "normalize_total_then_log1p",
        "target_gex_visible_to_model": False,
    }


def _save_cache(path: Path, metadata: dict, records: list[dict]) -> None:
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "sample_ids": np.asarray([row["sample_id"] for row in records]),
    }
    for index, row in enumerate(records):
        arrays[f"coords_{index:03d}"] = np.asarray(row["coords"], dtype=np.float32)
        arrays[f"target_{index:03d}"] = np.asarray(row["target"], dtype=np.float32)
        arrays[f"prediction_{index:03d}"] = np.asarray(
            row["prediction"], dtype=np.float32,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _load_cache(path: Path, expected: dict) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            if metadata != expected:
                return None
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            return [
                {
                    "sample_id": sample_id,
                    "coords": np.asarray(payload[f"coords_{index:03d}"], dtype=np.float32),
                    "target": np.asarray(payload[f"target_{index:03d}"], dtype=np.float32),
                    "prediction": np.asarray(
                        payload[f"prediction_{index:03d}"], dtype=np.float32,
                    ),
                }
                for index, sample_id in enumerate(sample_ids)
            ]
    except Exception:
        return None


def _predict_gene(
    model: torch.nn.Module, samples: dict, sample_ids: list[str], *,
    gene_index: int, prediction_role: str, n_samples: int, chunk_size: int,
    seed: int,
) -> list[dict]:
    records = []
    for index, sample_id in enumerate(sample_ids):
        result = predict_whole_slide(
            model, samples[sample_id], chunk_size=chunk_size,
            n_samples=n_samples if prediction_role == "prior" else 1,
            seed=_stable_seed(seed, {
                "sample_id": sample_id,
                "stratum": "single_gene_whole_slide_plot",
                "query_fingerprint": "every_spot_exactly_once",
            }),
        )
        predicted = (
            result["predictive_mean"] if prediction_role == "prior"
            else result["point_prediction"]
        )
        records.append({
            "sample_id": sample_id,
            "coords": np.asarray(result["coords"], dtype=np.float32),
            "target": np.asarray(result["target"][:, gene_index], dtype=np.float32),
            "prediction": predicted[:, gene_index].detach().cpu().numpy().astype(np.float32),
        })
        print(
            f"gene-map inference: {index + 1}/{len(sample_ids)} "
            f"sample={sample_id} spots={len(records[-1]['target'])}",
            flush=True,
        )
        del result, predicted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def _finite_percentile(values: np.ndarray, percent: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("plot values contain no finite observations")
    return float(np.percentile(finite, percent))


def _global_scales(records: list[dict]) -> tuple[float, float, float]:
    # Target-only limits make two models for the same gene/split directly
    # comparable. Predictions are clipped visually, never for metrics.
    target = np.concatenate([row["target"] for row in records])
    low = _finite_percentile(target, 2.0)
    high = _finite_percentile(target, 98.0)
    if not high > low:
        pad = max(abs(low) * 0.05, 1.0e-3)
        low, high = low - pad, high + pad
    return low, high, high - low


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    if np.std(target) > 0 and np.std(prediction) > 0:
        pcc = float(np.corrcoef(target, prediction)[0, 1])
    else:
        pcc = float("nan")
    return {
        "pcc": pcc,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def _render(
    records: list[dict], *, output_dir: Path, model_name: str, gene: str,
    prediction_role: str, checkpoint_step: int | None,
) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    value_low, value_high, error_high = _global_scales(records)
    pdf_path = output_dir / "all_slides.pdf"
    summary_path = output_dir / "slide_metrics.tsv"
    rows = []

    with PdfPages(pdf_path) as pdf:
        for record in records:
            target = record["target"]
            prediction = record["prediction"]
            coords = record["coords"]
            metrics = _metrics(target, prediction)
            rows.append({
                "sample_id": record["sample_id"],
                "n_spots": len(target),
                **metrics,
            })
            marker_size = max(1.0, min(12.0, 12000.0 / max(len(target), 1)))
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
            panels = (
                (target, "Target", "viridis", value_low, value_high),
                (prediction, "Prediction", "viridis", value_low, value_high),
                (np.abs(prediction - target), "Absolute error", "magma", 0.0, error_high),
            )
            for axis, (values, title, cmap, low, high) in zip(axes, panels):
                scatter = axis.scatter(
                    coords[:, 0], coords[:, 1], c=values, s=marker_size,
                    cmap=cmap, vmin=low, vmax=high, linewidths=0,
                )
                fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
                axis.invert_yaxis()
                axis.set_aspect("equal", adjustable="datalim")
                axis.set_title(title)
                axis.set_axis_off()
            step_label = f" | step {checkpoint_step}" if checkpoint_step is not None else ""
            fig.suptitle(
                f"{model_name} | {record['sample_id']} | {gene} | {prediction_role}{step_label}\n"
                f"PCC={metrics['pcc']:.4f}   RMSE={metrics['rmse']:.4f}   "
                f"MAE={metrics['mae']:.4f}   n={len(target)}",
            )
            png = output_dir / f"{_safe_name(record['sample_id'])}.png"
            fig.savefig(png, dpi=200, facecolor="white")
            pdf.savefig(fig, dpi=200, facecolor="white")
            plt.close(fig)

    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_id", "n_spots", "pcc", "rmse", "mae"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    return pdf_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name")
    parser.add_argument("--gene", required=True)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-code-drift", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if args.n_samples < 1 or args.chunk_size < 1:
        parser.error("--n-samples and --chunk-size must be positive")

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
    print(f"Resolved model: {source.model_name}")
    print(f"Config: {source.config_path}")
    print(f"Checkpoint: {source.checkpoint_path}")

    config = resolved_config(str(source.config_path))
    _require_he_to_st_config(config, source.config_path)
    manifest = load_dataset_manifest(config["data"]["gen3_manifest_path"])
    gene_names = list(manifest["gene_panel"])
    gene, gene_index = _resolve_gene(gene_names, args.gene)
    available_ids = list(manifest[f"{args.split}_sample_ids"])
    sample_ids = list(args.sample_id) if args.sample_id else available_ids
    unknown = sorted(set(sample_ids) - set(available_ids))
    if unknown:
        raise ValueError(f"requested samples are not in the {args.split} split: {unknown}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("--sample-id values must be unique")

    identity = checkpoint_module.resolve_checkpoint_identity(source.checkpoint_path)
    if identity.weights_sha256 is None:
        raise ValueError("resolved checkpoint has no weight hash")
    weights_sha256 = identity.weights_sha256
    checkpoint_step = int(checkpoint_module.load_training_state(identity.resolved_dir)["step"])
    scope = "all" if sample_ids == available_ids else hashlib.sha256(
        "\n".join(sample_ids).encode()
    ).hexdigest()[:10]
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root else results_root / "gene_spatial_maps"
    )
    output_dir = (
        output_root / _safe_name(source.model_name)
        / f"{_safe_name(gene)}__{args.prediction_role}__{weights_sha256[:12]}__{scope}"
    )
    metadata = _cache_metadata(
        source, weights_sha256=weights_sha256, gene=gene, split=args.split,
        sample_ids=sample_ids, prediction_role=args.prediction_role,
        n_samples=args.n_samples if args.prediction_role == "prior" else 1,
    )
    cache_path = output_dir / "single_gene_values.npz"
    records = None if args.force_recompute else _load_cache(cache_path, metadata)
    if records is None:
        model, samples, loaded_manifest, loaded_gene_names, _, loaded_weights = (
            _load_model_and_samples(
                source, split=args.split, device_name=args.device,
                allow_code_drift=args.allow_code_drift,
            )
        )
        if loaded_gene_names != gene_names or loaded_weights != weights_sha256:
            raise RuntimeError("model load changed the resolved gene/checkpoint identity")
        if list(loaded_manifest[f"{args.split}_sample_ids"]) != available_ids:
            raise RuntimeError("model load changed the resolved split identity")
        records = _predict_gene(
            model, samples, sample_ids, gene_index=gene_index,
            prediction_role=args.prediction_role, n_samples=args.n_samples,
            chunk_size=args.chunk_size,
            seed=int(config["training"].get("seed", 0)),
        )
        _save_cache(cache_path, metadata, records)
        print(f"Saved compact single-gene cache: {cache_path}")
    else:
        print(f"Reused compact single-gene cache: {cache_path}")

    pdf, summary = _render(
        records, output_dir=output_dir, model_name=source.model_name,
        gene=gene, prediction_role=args.prediction_role,
        checkpoint_step=checkpoint_step,
    )
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(json.dumps({
        **metadata,
        "checkpoint_step": checkpoint_step,
        "value_scale": "global target-only 2nd-98th percentile",
        "error_scale": "zero to global robust target range",
        "visual_clipping_only": True,
        "raw_values_used_for_metrics": True,
        "cache_path": str(cache_path),
        "pdf_path": str(pdf),
        "summary_path": str(summary),
    }, indent=2, sort_keys=True))
    print(f"Plots: {output_dir}")
    print(f"PDF: {pdf}")
    print(f"Metrics: {summary}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
