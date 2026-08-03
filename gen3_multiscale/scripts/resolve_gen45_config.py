#!/usr/bin/env python3
"""Resolve one Gen4/Gen5 template into a runnable immutable config file."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from gen3_multiscale.gen4.preflight import static_audit_gen4_config
from gen3_multiscale.gen5.preflight import static_audit_gen5_config


def _pairs(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=VALUE, got {value!r}")
        name, resolved = value.split("=", 1)
        if not name or not resolved:
            raise ValueError(f"expected non-empty NAME=VALUE, got {value!r}")
        if name in result:
            raise ValueError(f"duplicate override for {name!r}")
        result[name] = resolved
    return result


def resolve_gen45_config(
    base_config: str,
    output_config: str,
    *,
    manifest: str,
    checkpoint_dir: str,
    fingerprints: dict[str, str],
    data_overrides: dict[str, str],
    evaluation_overrides: dict[str, str] | None = None,
    total_steps: int | None = None,
    max_wall_clock_hours: float | None = None,
    device: str = "cuda",
) -> dict:
    output = Path(output_config)
    if output.exists():
        raise FileExistsError(f"{output} already exists; write a new resolved config")
    config = yaml.safe_load(Path(base_config).read_text())
    config["data"]["gen3_manifest_path"] = str(manifest)
    config["training"]["checkpoint_dir"] = str(checkpoint_dir)
    config["training"]["device"] = str(device)
    if total_steps is not None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        config["training"]["total_steps"] = int(total_steps)
    if max_wall_clock_hours is not None:
        if max_wall_clock_hours <= 0:
            raise ValueError("max_wall_clock_hours must be positive")
        config["training"]["max_wall_clock_hours"] = float(max_wall_clock_hours)
    config["required_fingerprints"].update(fingerprints)
    config["data"].update(data_overrides)
    config["evaluation"].update(evaluation_overrides or {})

    kind = str((config.get("model") or {}).get("kind"))
    from gen3_multiscale.gen6.contract import is_gen6_config

    if is_gen6_config(config):
        from gen3_multiscale.gen6.preflight import static_audit_gen6_config

        report = static_audit_gen6_config(config)
    else:
        report = (
            static_audit_gen5_config(config)
            if kind == "latent_flow"
            else static_audit_gen4_config(config)
        )
    if not report["ready_for_real_training"]:
        raise ValueError(
            "resolved config still has unset launch-critical fields: "
            f"{report['unset_required_fingerprints']}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False))
    os.replace(tmp, output)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--fingerprint", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--data", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Set an evaluation field, e.g. train_gene_panel_artifact=/path/train_gene_panels.json",
    )
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--max-wall-clock-hours", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = resolve_gen45_config(
        args.base_config,
        args.output_config,
        manifest=args.manifest,
        checkpoint_dir=args.checkpoint_dir,
        fingerprints=_pairs(args.fingerprint),
        data_overrides=_pairs(args.data),
        evaluation_overrides=_pairs(args.evaluation),
        total_steps=args.total_steps,
        max_wall_clock_hours=args.max_wall_clock_hours,
        device=args.device,
    )
    print(f"resolved runnable config: {args.output_config}")
    print(yaml.safe_dump(config["model"], sort_keys=False))


if __name__ == "__main__":
    main()
