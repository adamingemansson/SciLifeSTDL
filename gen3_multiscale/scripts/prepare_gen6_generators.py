#!/usr/bin/env python3
"""Prepare Gen6-K/L after a deterministic Gen6 conditioner is selected.

Writes immutable configs only; never starts training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from gen3_multiscale.gen6.contract import get_gen6_arm_spec
from gen3_multiscale.gen6.preflight import static_audit_gen6_config
from gen3_multiscale.training import checkpoint as checkpoint_module


def prepare(*, conditioner_checkpoint: str, basis_path: str,
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
    if selected.staged_conditioner:
        raise ValueError("the selected checkpoint is not a deterministic Gen6 conditioner")
    basis = Path(basis_path)
    provenance_path = Path(f"{basis}.provenance.json")
    if not basis.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Gen6-K requires a basis and matching .provenance.json")
    provenance = json.loads(provenance_path.read_text())
    expected = {
        "kind": "gen6_residual_basis_provenance", "conditioner_arm": conditioner_arm,
        "conditioner_checkpoint_sha256": identity.weights_sha256,
        "conditioner_checkpoint_step": identity.step,
        "conditioner_checkpoint_bundle_id": identity.bundle_dir,
        "conditioner_checkpoint_manifest_sha256": identity.manifest_sha256,
    }
    mismatch = {key: (provenance.get(key), value) for key, value in expected.items()
                if provenance.get(key) != value}
    if mismatch:
        raise ValueError(f"basis does not belong to selected conditioner: {mismatch}")
    (root / "configs").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "logs").mkdir()
    paths = {}
    for arm, kind in (("gen6k", "flow"), ("gen6l", "wae_gan")):
        config = copy.deepcopy(base)
        params = dict(config["model"].get("params") or {})
        params.update({
            "conditioner_arm": conditioner_arm, "n_flow_samples": 16,
            "n_ode_steps": 20, "latent_dim": 256,
        })
        if arm == "gen6k":
            params.update({
                "gene_basis_rank": int(provenance["rank"]), "n_flow_blocks": 2,
                "ot_epsilon": 0.1, "ot_sinkhorn_iters": 20,
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
            config["required_fingerprints"]["gene_residual_basis"] = str(basis.resolve())
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
        "basis": str(basis), "hours_per_arm": float(hours), "configs": paths,
    }
    (root / "run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioner-checkpoint", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    args = parser.parse_args()
    print(json.dumps(prepare(
        conditioner_checkpoint=args.conditioner_checkpoint, basis_path=args.basis,
        output_root=args.output_root, hours=args.hours,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
