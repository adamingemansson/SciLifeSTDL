#!/usr/bin/env python3
"""Prepare four immutable, comparable Gen5 full-run configs.

This command does not train anything.  It verifies the shared autoencoder,
the validation-selected Gen4 conditioner roots, the train-derived gene
panels, and the frozen-feature identities already present in the cache,
then writes one resolved config per primary Gen5 arm plus a compact run
plan.  Long GPU runs remain an explicit separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.gen4.preflight import audit_gen4_manifest_cache_coverage
from gen3_multiscale.gen4.trainer_adapter import _resolve_gen4_arm
from gen3_multiscale.gen5.autoencoder import (
    load_expression_autoencoder_checkpoint,
    verify_expression_autoencoder_gene_names,
)
from gen3_multiscale.scripts.resolve_gen45_config import resolve_gen45_config
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.train import dataset_manifest_fingerprint


_PRIMARY_ARMS = ("gen5c", "gen5b", "gen5d", "gen5e")
_GEN4_ARM = {
    "gen5c": "gen4c",
    "gen5b": "gen4b",
    "gen5d": "gen4d",
    "gen5e": "gen4e",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(values: list[str], *, expected: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected ARM=PATH, got {value!r}")
        arm, path = value.split("=", 1)
        if arm not in expected or not path:
            raise ValueError(f"expected one of {sorted(expected)}=PATH, got {value!r}")
        if arm in result:
            raise ValueError(f"duplicate conditioner for {arm}")
        result[arm] = path
    missing = sorted(expected - set(result))
    if missing:
        raise ValueError(f"missing --conditioner entries for {missing}")
    return result


def _scalar(cache: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in cache.files:
        raise ValueError(f"cache is missing identity field {name!r}")
    return str(np.asarray(cache[name]).item())


def _cache_identities(cache_root: Path, sample_id: str) -> tuple[dict, dict, str]:
    uni2_path = cache_root / "uni2_gen3_spot_cache" / f"{sample_id}.npz"
    scf_path = cache_root / "scfoundation_gen3_spot_cache" / f"{sample_id}.npz"
    gigapath_path = cache_root / "gigapath_gen3_spot_cache" / f"{sample_id}.npz"
    for path in (uni2_path, scf_path, gigapath_path):
        if not path.is_file():
            raise FileNotFoundError(f"required cache identity source is missing: {path}")
    with np.load(uni2_path, allow_pickle=False) as cache:
        uni2 = {
            "uni2_revision": _scalar(cache, "uni2_pinned_revision"),
            "uni2_package_version": _scalar(cache, "uni2_package_version"),
            "uni2_preprocessing_spec": _scalar(cache, "uni2_preprocessing_spec"),
            "output_dim": int(np.asarray(cache["uni2_output_dim"]).item()),
        }
    with np.load(scf_path, allow_pickle=False) as cache:
        scfoundation = {
            "scfoundation_package_version": _scalar(cache, "scfoundation_package_version"),
            "scfoundation_preprocessing_spec": _scalar(cache, "scfoundation_preprocessing_spec"),
            "output_dim": int(np.asarray(cache["scfoundation_output_dim"]).item()),
        }
    with np.load(gigapath_path, allow_pickle=False) as cache:
        tile_revision = _scalar(cache, "tile_encoder_hf_revision")
    if len(tile_revision) != 40 or any(c not in "0123456789abcdef" for c in tile_revision):
        raise ValueError(f"GigaPath cache has an invalid pinned revision: {tile_revision!r}")
    return uni2, scfoundation, tile_revision


def _verified_autoencoder_report(
    checkpoint_path: str | Path,
    *,
    checkpoint_sha256: str,
    manifest_fingerprint: str,
    validation_sample_ids: list[str],
    train_panel_sha256: str,
) -> dict:
    """Require the independent reconstruction ceiling before Gen5 prep.

    A numerically valid autoencoder checkpoint alone does not show that
    its 256-dimensional bottleneck reconstructs held-out expression well
    enough for meaningful Gen5 comparisons.  Bind the report to the exact
    checkpoint bytes, manifest, validation split, and train-derived panels.
    """
    report_path = Path(f"{checkpoint_path}.report.json")
    if not report_path.is_file():
        raise FileNotFoundError(
            f"missing autoencoder reconstruction report: {report_path}; "
            "run train_gen5_autoencoder to completion before preparing Gen5"
        )
    report = json.loads(report_path.read_text())
    expected = {
        "kind": "gen5_expression_autoencoder_training_report",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_fingerprint": manifest_fingerprint,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(
                f"{report_path}: {name}={report.get(name)!r} does not match {value!r}"
            )
    panel_identity = report.get("train_gene_panel_artifact") or {}
    if panel_identity.get("artifact_sha256") != train_panel_sha256:
        raise ValueError(
            f"{report_path}: train-derived panel identity does not match this Gen5 suite"
        )
    validation = report.get("validation_reconstruction") or {}
    if sorted(validation.get("sample_ids") or []) != sorted(validation_sample_ids):
        raise ValueError(f"{report_path}: validation sample IDs do not match the manifest")
    for name in ("rmse", "pcc_mean"):
        try:
            value = float(validation[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{report_path}: invalid validation_reconstruction.{name}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{report_path}: validation_reconstruction.{name} is not finite")
    return report


def _verified_conditioner(path: str, expected_arm: str) -> dict:
    root = Path(path)
    if root.name != "best":
        raise ValueError(
            f"{expected_arm}: conditioner path must be the validation-selected best/ root, got {root}"
        )
    identity = checkpoint_module.resolve_checkpoint_identity(root)
    config_path = identity.resolved_dir / "model_config.json"
    if not config_path.is_file():
        raise ValueError(f"{identity.resolved_dir}: missing model_config.json")
    config = json.loads(config_path.read_text())
    if str((config.get("model") or {}).get("kind")) != "conditioner":
        raise ValueError(f"{root}: selected Gen4 artifact is not a conditioner")
    actual_arm = _resolve_gen4_arm(config)
    if actual_arm != expected_arm:
        raise ValueError(f"{root}: conditioner arm {actual_arm!r} != expected {expected_arm!r}")
    return {
        "path": str(root),
        "resolved_bundle": str(identity.resolved_dir),
        "step": identity.step,
        "weights_sha256": identity.weights_sha256,
        "manifest_sha256": identity.manifest_sha256,
    }


def prepare_gen5_suite(
    *,
    manifest_path: str,
    cache_root: str,
    train_gene_panels: str,
    autoencoder_checkpoint: str,
    conditioners: dict[str, str],
    output_root: str,
    uni2_checkpoint: str,
    scfoundation_checkpoint: str,
    scfoundation_vocab: str,
    stpath_checkpoint: str,
    stpath_gene_vocab: str,
    gigapath_checkpoint: str,
    hest_data_dir: str | None = None,
    total_steps: int = 100_000_000,
    max_wall_clock_hours: float = 24.0,
    device: str = "cuda",
) -> dict:
    manifest = load_dataset_manifest(manifest_path)
    manifest_fp = dataset_manifest_fingerprint(manifest)
    gene_names = list(manifest["gene_panel"])
    panel_artifact = load_train_derived_gene_panels(train_gene_panels, manifest)
    autoencoder_sha256 = _sha256(autoencoder_checkpoint)
    autoencoder, autoencoder_payload = load_expression_autoencoder_checkpoint(
        autoencoder_checkpoint,
        dataset_manifest_fingerprint=manifest_fp,
    )
    verify_expression_autoencoder_gene_names(autoencoder, gene_names)
    if int(autoencoder_payload["latent_dim"]) != 256:
        raise ValueError(
            f"shared autoencoder latent_dim={autoencoder_payload['latent_dim']}, but Gen5 configs require 256"
        )
    autoencoder_report = _verified_autoencoder_report(
        autoencoder_checkpoint,
        checkpoint_sha256=autoencoder_sha256,
        manifest_fingerprint=manifest_fp,
        validation_sample_ids=list(manifest.get("validation_sample_ids") or []),
        train_panel_sha256=panel_artifact["artifact_sha256"],
    )

    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"{output} already exists; Gen5 run roots are immutable")
    cache_path = Path(cache_root).resolve()
    sample_ids = sorted(manifest["samples"])
    if not sample_ids:
        raise ValueError("manifest has no samples")
    uni2_identity, scf_identity, tile_revision = _cache_identities(
        cache_path, sample_ids[0],
    )
    if uni2_identity.pop("output_dim") != 1536:
        raise ValueError("UNI2 cache output width is not 1536")
    if scf_identity.pop("output_dim") != 3072:
        raise ValueError("scFoundation cache output width is not 3072")

    for path in (
        uni2_checkpoint,
        scfoundation_checkpoint,
        scfoundation_vocab,
        stpath_checkpoint,
        stpath_gene_vocab,
        gigapath_checkpoint,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required model artifact is missing: {path}")

    conditioner_info = {
        arm: _verified_conditioner(conditioners[arm], _GEN4_ARM[arm])
        for arm in _PRIMARY_ARMS
    }
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        configs: dict[str, str] = {}
        common_data = {
            "hest_data_dir": str(hest_data_dir or manifest["hest_data_dir"]),
            "hest_cache_dir": str(cache_path),
        }
        common = {
            "expression_autoencoder_checkpoint": str(Path(autoencoder_checkpoint).resolve()),
        }
        uni2 = {
            "uni2_checkpoint": str(Path(uni2_checkpoint).resolve()),
            **uni2_identity,
        }
        scf = {
            "scfoundation_checkpoint": str(Path(scfoundation_checkpoint).resolve()),
            "scfoundation_vocab": str(Path(scfoundation_vocab).resolve()),
            **scf_identity,
        }
        stpath = {
            "stpath_checkpoint": str(Path(stpath_checkpoint).resolve()),
            "stpath_gene_vocab": str(Path(stpath_gene_vocab).resolve()),
        }
        by_arm = {
            "gen5c": {**uni2, **scf},
            "gen5b": {
                "gigapath_checkpoint": str(Path(gigapath_checkpoint).resolve()),
                **scf,
            },
            "gen5d": stpath,
            "gen5e": {**stpath, **uni2, **scf},
        }
        for arm in _PRIMARY_ARMS:
            final_config = output / "configs" / f"{arm}.yaml"
            staged_config = staging / "configs" / f"{arm}.yaml"
            fingerprints = {
                **common,
                **by_arm[arm],
                # `best/` is a mutable validation-selection pointer.
                # Resolve it once during preparation and publish the
                # immutable bundle, so a later Gen4 resume cannot move
                # this Gen5 run to different conditioner weights.
                "gen4_conditioner_checkpoint": conditioner_info[arm]["resolved_bundle"],
            }
            data_overrides = dict(common_data)
            if arm in {"gen5b", "gen5d", "gen5e"}:
                data_overrides["tile_encoder_revision"] = tile_revision
            config = resolve_gen45_config(
                f"gen3_multiscale/configs/gen5/{arm}.yaml",
                str(staged_config),
                manifest=str(Path(manifest_path).resolve()),
                checkpoint_dir=str(output / f"checkpoints_{arm}"),
                fingerprints=fingerprints,
                data_overrides=data_overrides,
                evaluation_overrides={
                    "train_gene_panel_artifact": str(Path(train_gene_panels).resolve()),
                },
                total_steps=total_steps,
                max_wall_clock_hours=max_wall_clock_hours,
                device=device,
            )
            # Exact cache coverage/provenance for every manifest sample,
            # before publishing any run root.
            audit_gen4_manifest_cache_coverage(cache_path, sample_ids, config)
            configs[arm] = str(final_config)

        plan = {
            "kind": "gen5_full_run_plan",
            "manifest": str(Path(manifest_path).resolve()),
            "dataset_manifest_fingerprint": manifest_fp,
            "train_gene_panels": {
                "path": str(Path(train_gene_panels).resolve()),
                "artifact_sha256": panel_artifact["artifact_sha256"],
            },
            "autoencoder": {
                "path": str(Path(autoencoder_checkpoint).resolve()),
                "file_sha256": autoencoder_sha256,
                "latent_dim": autoencoder_payload["latent_dim"],
                "hidden_dim": autoencoder_payload["hidden_dim"],
                "validation_reconstruction": autoencoder_report[
                    "validation_reconstruction"
                ],
            },
            "conditioners": conditioner_info,
            "configs": configs,
            "arms": list(_PRIMARY_ARMS),
            "total_steps": int(total_steps),
            "max_wall_clock_hours": float(max_wall_clock_hours),
        }
        (staging / "logs").mkdir()
        (staging / "run_plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n"
        )
        os.replace(staging, output)
        return plan
    except Exception:
        # The staging tree belongs exclusively to this process.  Remove
        # it so a corrected invocation can retry, while the immutable
        # final run root remains entirely absent on failure.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--train-gene-panels", required=True)
    parser.add_argument("--autoencoder-checkpoint", required=True)
    parser.add_argument(
        "--conditioner",
        action="append",
        default=[],
        metavar="GEN5_ARM=/path/to/gen4/best",
        help="Repeat for gen5c, gen5b, gen5d, and gen5e.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--uni2-checkpoint", required=True)
    parser.add_argument("--scfoundation-checkpoint", required=True)
    parser.add_argument("--scfoundation-vocab", required=True)
    parser.add_argument("--stpath-checkpoint", required=True)
    parser.add_argument("--stpath-gene-vocab", required=True)
    parser.add_argument("--gigapath-checkpoint", required=True)
    parser.add_argument("--hest-data-dir")
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--max-wall-clock-hours", type=float, default=24.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    plan = prepare_gen5_suite(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        train_gene_panels=args.train_gene_panels,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        conditioners=_pairs(args.conditioner, expected=set(_PRIMARY_ARMS)),
        output_root=args.output_root,
        uni2_checkpoint=args.uni2_checkpoint,
        scfoundation_checkpoint=args.scfoundation_checkpoint,
        scfoundation_vocab=args.scfoundation_vocab,
        stpath_checkpoint=args.stpath_checkpoint,
        stpath_gene_vocab=args.stpath_gene_vocab,
        gigapath_checkpoint=args.gigapath_checkpoint,
        hest_data_dir=args.hest_data_dir,
        total_steps=args.total_steps,
        max_wall_clock_hours=args.max_wall_clock_hours,
        device=args.device,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
