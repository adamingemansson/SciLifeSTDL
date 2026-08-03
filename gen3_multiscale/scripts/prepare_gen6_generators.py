#!/usr/bin/env python3
"""Prepare Gen6-K/L after a deterministic Gen6 conditioner is selected.

Writes immutable configs only; never starts training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

from gen3_multiscale.gen6.contract import get_gen6_arm_spec
from gen3_multiscale.gen6.preflight import static_audit_gen6_config
from gen3_multiscale.gen5.autoencoder import (
    load_expression_autoencoder_checkpoint,
    verify_expression_autoencoder_gene_names,
)
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train import dataset_manifest_fingerprint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(*, conditioner_checkpoint: str, autoencoder_checkpoint: str,
            output_root: str, hours: float = 8.0) -> dict:
    if hours <= 0:
        raise ValueError("hours must be positive")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    identity = checkpoint_module.resolve_checkpoint_identity(conditioner_checkpoint)
    config_path = identity.resolved_dir / "model_config.json"
    if not config_path.is_file():
        raise ValueError(f"{identity.resolved_dir}: missing model_config.json")
    base = json.loads(config_path.read_text())
    conditioner_arm = str((base.get("model") or {}).get("arm", ""))
    selected = get_gen6_arm_spec(conditioner_arm)
    if selected.staged_conditioner or conditioner_arm != "gen6c":
        raise ValueError("Gen6-K/L require the deterministic Gen6-C conditioner checkpoint")
    manifest_path = Path(str((base.get("data") or {}).get("gen3_manifest_path") or ""))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"conditioner config manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    gene_names = list(manifest["gene_panel"])
    manifest_fingerprint = dataset_manifest_fingerprint(manifest)
    autoencoder_path = Path(autoencoder_checkpoint).resolve()
    autoencoder, ae_payload = load_expression_autoencoder_checkpoint(
        autoencoder_path, dataset_manifest_fingerprint=manifest_fingerprint,
    )
    verify_expression_autoencoder_gene_names(autoencoder, gene_names)
    report_path = Path(f"{autoencoder_path}.report.json")
    if not report_path.is_file():
        raise FileNotFoundError(f"missing autoencoder reconstruction report: {report_path}")
    ae_report = json.loads(report_path.read_text())
    expected_report = {
        "kind": "gen5_expression_autoencoder_training_report",
        "checkpoint_sha256": _sha256(autoencoder_path),
        "dataset_manifest_fingerprint": manifest_fingerprint,
    }
    mismatches = {
        key: (ae_report.get(key), value)
        for key, value in expected_report.items() if ae_report.get(key) != value
    }
    if mismatches:
        raise ValueError(f"autoencoder report identity mismatch: {mismatches}")
    validation = ae_report.get("validation_reconstruction") or {}
    for metric in ("rmse", "pcc_mean"):
        if not math.isfinite(float(validation.get(metric, float("nan")))):
            raise ValueError(f"autoencoder validation_reconstruction.{metric} is not finite")
    expected_train_ids = sorted(manifest.get("train_sample_ids") or [])
    expected_validation_ids = sorted(manifest.get("validation_sample_ids") or [])
    if sorted(ae_report.get("train_sample_ids") or []) != expected_train_ids:
        raise ValueError("autoencoder report training sample IDs do not match the manifest")
    if sorted(validation.get("sample_ids") or []) != expected_validation_ids:
        raise ValueError("autoencoder report validation sample IDs do not match the manifest")
    dimension_fields = {
        "n_genes": len(gene_names),
        "latent_dim": int(ae_payload["latent_dim"]),
        "hidden_dim": int(ae_payload["hidden_dim"]),
    }
    dimension_mismatches = {
        name: (ae_report.get(name), expected)
        for name, expected in dimension_fields.items()
        if int(ae_report.get(name, -1)) != expected
    }
    if dimension_mismatches:
        raise ValueError(f"autoencoder report dimension mismatch: {dimension_mismatches}")
    (root / "configs").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "logs").mkdir()
    paths = {}
    for arm, kind in (("gen6k", "latent_flow"), ("gen6l", "wae_gan")):
        config = copy.deepcopy(base)
        params = dict(config["model"].get("params") or {})
        params.update({
            "conditioner_arm": conditioner_arm, "n_flow_samples": 16,
            "n_ode_steps": 20, "latent_dim": int(ae_payload["latent_dim"]),
        })
        if arm == "gen6k":
            params.update({
                "autoencoder_hidden_dim": int(ae_payload["hidden_dim"]),
                "n_flow_blocks": 2, "ot_epsilon": 0.1, "ot_sinkhorn_iters": 20,
            })
        else:
            params.update({
                "wae_hidden_dim": 1024, "discriminator_hidden_dim": 256,
                "adversarial_weight": 0.1, "discriminator_weight": 1.0,
            })
        config["experiment_name"] = f"gen6_component_screen_{arm}_v1"
        config["model"] = {"arm": arm, "kind": kind, "params": params}
        config["required_fingerprints"]["gen6_conditioner_checkpoint"] = str(
            Path(conditioner_checkpoint).resolve()
        )
        if arm == "gen6k":
            config["required_fingerprints"]["expression_autoencoder_checkpoint"] = str(
                autoencoder_path
            )
        config["training"]["checkpoint_dir"] = str(root / "checkpoints" / arm)
        config["training"]["total_steps"] = 100_000_000
        config["training"]["max_wall_clock_hours"] = float(hours)
        report = static_audit_gen6_config(config)
        if not report["ready_for_real_training"]:
            raise ValueError(f"{arm} unresolved fields: {report['unset_required_fingerprints']}")
        path = root / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths[arm] = str(path)
    plan = {
        "conditioner_arm": conditioner_arm, "conditioner_checkpoint": str(conditioner_checkpoint),
        "conditioner_weights_sha256": identity.weights_sha256,
        "autoencoder_checkpoint": str(autoencoder_path),
        "autoencoder_checkpoint_sha256": expected_report["checkpoint_sha256"],
        "hours_per_arm": float(hours), "configs": paths,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioner-checkpoint", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(prepare(
        conditioner_checkpoint=args.conditioner_checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        output_root=args.output_root, hours=args.hours,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
