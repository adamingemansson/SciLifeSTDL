"""Collect compact, reviewable audit artifacts without copying model weights."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

COMPACT_NAMES = {
    "audit_test_metrics.json",
    "heldout_sample_summary.json",
    "validation_history.json",
    "run_manifest.json",
    "resolved_config.yaml",
    "sample_split.json",
    "history.json",
    "quality_gate.json",
    "metrics.csv",
}


def _primary_summary(metrics: dict) -> dict:
    modes = metrics.get("image_modes", {})
    primary_mode = str(metrics.get("primary_image_mode", "full"))
    selected = modes.get(primary_mode) or (next(iter(modes.values())) if modes else {})
    summary = selected.get("summary", {})
    def mean(key):
        value = summary.get(key, {})
        return value.get("mean") if isinstance(value, dict) else None
    return {
        "experiment_name": metrics.get("experiment_name"),
        "primary_image_mode": primary_mode,
        "context_gex_mode": metrics.get("context_gex_mode", "full"),
        "modality_ablation": metrics.get("modality_ablation", "both"),
        "pcc": mean("pcc"),
        "rmse": mean("rmse"),
        "nonzero_auc": mean("nonzero_auc"),
        "st_fid": mean("st_fid"),
        "st_mmd": mean("st_mmd"),
        "spatial_domain_plausibility": mean("spatial_domain_plausibility"),
        "predictive_std": mean("predictive_std"),
        "interval90_coverage": mean("interval90_coverage"),
        "n_test_masks": metrics.get("n_test_masks"),
        "n_samples_per_mask": metrics.get("n_samples_per_mask"),
        "effective_pca_components": metrics.get("effective_pca_components"),
        "n_evaluated_genes": metrics.get("n_evaluated_genes"),
    }


def collect(
    checkpoint_root: Path,
    output_dir: Path,
    log_root: Path | None,
    experiment_prefix: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    copied = []
    for experiment_dir in sorted(p for p in checkpoint_root.glob("*") if p.is_dir()):
        if experiment_prefix and not experiment_dir.name.startswith(experiment_prefix):
            continue
        destination = output_dir / experiment_dir.name
        for source in sorted(experiment_dir.iterdir()):
            include = source.name in COMPACT_NAMES or (
                source.name.startswith("audit_test_metrics_") and source.suffix == ".json"
            )
            if not include or not source.is_file():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / source.name)
            copied.append(str(source))
        metrics_path = experiment_dir / "audit_test_metrics.json"
        if metrics_path.exists():
            rows.append(_primary_summary(json.loads(metrics_path.read_text())))
        heldout = experiment_dir / "heldout_sample_summary.json"
        if heldout.exists():
            payload = json.loads(heldout.read_text())
            full = payload.get("full") or next(iter(payload.values()))
            rows.append({
                "experiment_name": experiment_dir.name,
                "pcc": full.get("pcc_mean"),
                "rmse": full.get("rmse_mean"),
                "evaluation_scope": "heldout_samples",
            })

    fields = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "collection_manifest.json").write_text(json.dumps({
        "checkpoint_root": str(checkpoint_root),
        "log_root": str(log_root) if log_root else None,
        "experiment_prefix": experiment_prefix,
        "copied_files": copied,
        "note": "Model weights and raw logs are deliberately excluded from this compact report.",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--experiment-prefix")
    args = parser.parse_args()
    collect(args.checkpoint_root, args.output_dir, args.log_root, args.experiment_prefix)
