#!/usr/bin/env python3
"""Train Gen5's shared expression autoencoder on manifest training data.

Expression is staged through a disk-backed matrix and transferred to the
GPU one mini-batch at a time. Only training samples can reach optimizer
updates; validation samples are read afterward for an independent
reconstruction-ceiling diagnostic. Test samples are never read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.data.dataset_manifest import (
    load_dataset_manifest,
    verify_content_provenance,
    verify_metadata_csv_provenance,
)
from gen3_multiscale.data.example_builder import load_expression_for_model_target_space
from gen3_multiscale.evaluation.metrics import resolve_gene_panels
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen5.autoencoder import save_expression_autoencoder_checkpoint
from gen3_multiscale.gen5.autoencoder_training import train_expression_autoencoder
from gen3_multiscale.training.train import (
    _code_commit_hash,
    _worktree_diff_hash,
    dataset_manifest_fingerprint,
    file_sha256,
)


def _stream_reconstruction_metrics(
    autoencoder,
    matrices,
    *,
    batch_size: int,
    gene_panels: dict[str, list[str]] | None = None,
) -> dict:
    """Exact full-row RMSE/PCC with O(n_genes) accumulator memory."""
    n_genes = autoencoder.n_genes
    sum_true = np.zeros(n_genes, dtype=np.float64)
    sum_pred = np.zeros(n_genes, dtype=np.float64)
    sum_true2 = np.zeros(n_genes, dtype=np.float64)
    sum_pred2 = np.zeros(n_genes, dtype=np.float64)
    sum_cross = np.zeros(n_genes, dtype=np.float64)
    squared_error_by_gene = np.zeros(n_genes, dtype=np.float64)
    n_rows = 0
    device = next(autoencoder.parameters()).device
    autoencoder.eval()
    with torch.no_grad():
        for matrix in matrices:
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != n_genes:
                raise ValueError(
                    f"reconstruction matrix must be [N, {n_genes}], got {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("reconstruction matrix contains non-finite values")
            for start in range(0, matrix.shape[0], batch_size):
                true = matrix[start:start + batch_size]
                pred = autoencoder(
                    torch.as_tensor(true, dtype=torch.float32, device=device)
                ).cpu().numpy()
                true64 = true.astype(np.float64, copy=False)
                pred64 = pred.astype(np.float64, copy=False)
                delta = pred64 - true64
                squared_error_by_gene += np.sum(delta * delta, axis=0)
                sum_true += np.sum(true64, axis=0)
                sum_pred += np.sum(pred64, axis=0)
                sum_true2 += np.sum(true64 * true64, axis=0)
                sum_pred2 += np.sum(pred64 * pred64, axis=0)
                sum_cross += np.sum(true64 * pred64, axis=0)
                n_rows += true.shape[0]
    if n_rows == 0:
        raise ValueError("cannot evaluate autoencoder reconstruction on zero rows")
    covariance = sum_cross - (sum_true * sum_pred / n_rows)
    true_ss = sum_true2 - (sum_true * sum_true / n_rows)
    pred_ss = sum_pred2 - (sum_pred * sum_pred / n_rows)
    denominator = np.sqrt(np.maximum(true_ss, 0.0) * np.maximum(pred_ss, 0.0))
    valid = denominator > 0
    pcc = np.full(n_genes, np.nan, dtype=np.float64)
    pcc[valid] = covariance[valid] / denominator[valid]
    report = {
        "n_rows": int(n_rows),
        "n_genes": int(n_genes),
        "rmse": float(np.sqrt(np.sum(squared_error_by_gene) / (n_rows * n_genes))),
        "pcc_mean": float(np.nanmean(pcc)),
        "n_valid_pcc_genes": int(np.sum(valid)),
    }
    if gene_panels:
        panel_indices, panel_metadata = resolve_gene_panels(
            list(autoencoder.gene_names), gene_panels,
        )
        report["gene_panels"] = {
            panel_name: {
                "pcc_mean": float(np.nanmean(pcc[idx])),
                "rmse": float(
                    np.sqrt(np.sum(squared_error_by_gene[idx]) / (n_rows * len(idx)))
                ),
                "n_valid_pcc_genes": int(np.sum(valid[idx])),
                **panel_metadata[panel_name],
            }
            for panel_name, idx in panel_indices.items()
        }
    return report


def train_from_manifest(
    manifest_path: str,
    output_checkpoint: str,
    *,
    latent_dim: int = 256,
    hidden_dim: int = 1024,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
    train_gene_panel_artifact: str | None = None,
) -> dict:
    manifest = load_dataset_manifest(manifest_path)
    verify_metadata_csv_provenance(manifest)
    train_ids = list(manifest["train_sample_ids"])
    if not train_ids:
        raise ValueError("dataset manifest has no training samples")
    gene_names = list(manifest["gene_panel"])
    gene_panels = None
    train_panel_identity = None
    if train_gene_panel_artifact:
        panel_artifact = load_train_derived_gene_panels(
            train_gene_panel_artifact, manifest,
        )
        gene_panels = dict(panel_artifact["panels"])
        train_panel_identity = {
            "path": str(train_gene_panel_artifact),
            "artifact_sha256": panel_artifact["artifact_sha256"],
            "dataset_manifest_fingerprint": panel_artifact[
                "dataset_manifest_fingerprint"
            ],
            "method": panel_artifact["method"],
        }
    n_rows = sum(len(manifest["samples"][sid]["barcodes"]) for sid in train_ids)
    if n_rows < 2:
        raise ValueError("training split contains fewer than two spots")

    output_path = Path(output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memmap_path = output_path.with_name(f"{output_path.name}.expression.tmp.{os.getpid()}.npy")
    expression = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(n_rows, len(gene_names)),
    )
    offset = 0
    try:
        for sample_id in train_ids:
            verify_content_provenance(manifest["hest_data_dir"], manifest, sample_id)
            adata = load_expression_for_model_target_space(manifest, sample_id)
            values = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            values = np.asarray(values, dtype=np.float32)
            expression[offset:offset + values.shape[0]] = values
            offset += values.shape[0]
        if offset != n_rows:
            raise RuntimeError(f"wrote {offset} rows, expected {n_rows} from the manifest")
        expression.flush()

        autoencoder, training_report = train_expression_autoencoder(
            expression,
            gene_names,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )
        # Report a deterministic training reconstruction diagnostic on a
        # bounded prefix; this is not model selection and never touches
        # validation/test data.
        diagnostic_rows = min(n_rows, 256)
        training_diagnostic = _stream_reconstruction_metrics(
            autoencoder,
            [np.asarray(expression[:diagnostic_rows])],
            batch_size=batch_size,
            gene_panels=gene_panels,
        )
    finally:
        # Drop the memmap before unlinking on Windows-like semantics too.
        try:
            del expression
        except UnboundLocalError:
            pass
        memmap_path.unlink(missing_ok=True)

    validation_ids = list(manifest.get("validation_sample_ids") or [])
    if not validation_ids:
        raise ValueError(
            "dataset manifest has no validation samples; refusing to save an autoencoder "
            "without an independent reconstruction-capacity diagnostic"
        )
    def _validation_matrices():
        for sample_id in validation_ids:
            verify_content_provenance(manifest["hest_data_dir"], manifest, sample_id)
            adata = load_expression_for_model_target_space(manifest, sample_id)
            values = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            yield np.asarray(values, dtype=np.float32)

    validation_diagnostic = _stream_reconstruction_metrics(
        autoencoder,
        _validation_matrices(),
        batch_size=batch_size,
        gene_panels=gene_panels,
    )

    code_identity = f"{_code_commit_hash()}:{_worktree_diff_hash()}"
    manifest_fp = dataset_manifest_fingerprint(manifest)
    preprocessing_spec = json.dumps(manifest["build_args"], sort_keys=True, default=str)
    save_expression_autoencoder_checkpoint(
        autoencoder,
        output_path,
        dataset_manifest_fingerprint=manifest_fp,
        preprocessing_spec=preprocessing_spec,
        code_identity=code_identity,
    )
    report = {
        "kind": "gen5_expression_autoencoder_training_report",
        "checkpoint_path": str(output_path),
        "checkpoint_sha256": file_sha256(output_path),
        "dataset_manifest_fingerprint": manifest_fp,
        "train_sample_ids": sorted(train_ids),
        "n_rows": n_rows,
        "n_genes": len(gene_names),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "train_gene_panel_artifact": train_panel_identity,
        "training": {
            "n_epochs": training_report.n_epochs,
            "final_train_loss": training_report.final_train_loss,
            "loss_per_epoch": training_report.loss_per_epoch,
        },
        "training_prefix_reconstruction": training_diagnostic,
        "validation_reconstruction": {
            **validation_diagnostic,
            "sample_ids": sorted(validation_ids),
        },
    }
    report_path = Path(f"{output_path}.report.json")
    tmp = report_path.with_name(f"{report_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(tmp, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-gene-panel-artifact",
        help=(
            "Optional immutable training-derived panel artifact. When given, "
            "the reconstruction ceiling report also includes the same HVG-50/200 "
            "views used by Gen3/Gen4/Gen5 evaluation."
        ),
    )
    args = parser.parse_args()
    report = train_from_manifest(
        args.manifest,
        args.output_checkpoint,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        train_gene_panel_artifact=args.train_gene_panel_artifact,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
