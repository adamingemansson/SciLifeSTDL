#!/usr/bin/env python3
"""The first piece of the deployment/orchestration system Adam asked for
across several audit rounds -- see `gen3_multiscale/CONTRACT.md` sections
53-54's "What remains honestly undone" -- "a dedicated config-resolution
CLI that refuses unresolved/null fields and records resolved-config
hashes." Every other piece of that ask (a standalone synchronized-init
verification command, one consolidated experiment-preflight command, the
real staged orchestrator with Stages A-D, dedicated failure/recovery
tests per stage, six runnable deliverables plus a generated run-plan
JSON, disk/RAM/GPU estimates) remains unbuilt; this script is a
deliberately bounded, independently useful, independently testable first
step, not a claim that the orchestrator itself now exists.

The four committed `gen3_multiscale/configs/architectureN.yaml` files are
templates, not runnable configs: several fields are `null` by design
(`data.gen3_manifest_path`, `data.tile_encoder_revision`,
`training.synchronized_init_dir`, and conditionally
`required_fingerprints.gigapath_checkpoint`/`gene_residual_basis`/
`architecture3_conditioner_checkpoint`) because they describe a specific
DEPLOYMENT, not the architecture itself, and are deliberately not
committed to the repo. Today an operator fills these in by hand, directly
editing a copy of the template -- error-prone (a forgotten field is
silently `None` until `train.py` fails deep inside a run) and produces no
durable, hashed record of exactly what was resolved. This script replaces
that manual edit with one command: apply explicit deployment overrides to
a base template, fail closed if anything the SPECIFIC architecture/config
combination actually requires is still unresolved, drop dead fields the
real trainer never reads (see below) rather than leave them as misleading
nulls, and write the fully-resolved config plus a durable identity
sidecar to an IMMUTABLE output path (refuses to silently overwrite an
existing resolved config -- a resolved config is the input other
orchestration steps hash-bind to, so overwriting it after other steps
have started trusting it would silently invalidate their own identity
checks).

Dead fields, confirmed by direct inspection of `training/train.py`, never
read by the real trainer and therefore dropped from the resolved output
rather than left as always-unresolvable nulls: `model.params.n_genes` and
`model.params.gex_feature_dim` (the trainer computes both directly --
`n_genes = len(gene_names)` from the dataset manifest's own gene panel,
`gex_feature_dim` from `data.gex_feature_dim`, a real, already-set,
non-null field -- and passes them as explicit kwargs to
`model_factory.build_architecture`, never reading `model.params.n_genes`/
`gex_feature_dim` at all); `required_fingerprints.gene_vocabulary`,
`train_mask_bank`, `validation_mask_bank`, `test_mask_bank` (leftovers
from before Step 6's real per-sample data pipeline and mask-schedule
generator existed -- `train.py` derives gene names from
`dataset_manifest["gene_panel"]` and builds its own mask schedule via
`build_gen3_mask_schedule`, consulting none of these four config keys).

    python -m gen3_multiscale.scripts.resolve_experiment_config \\
        --base-config gen3_multiscale/configs/architecture3.yaml \\
        --output gen3_multiscale/results/my_experiment/resolved_architecture3.yaml \\
        --gen3-manifest-path /path/to/dataset_manifest.json \\
        --tile-encoder-revision <pinned 40-hex HF commit SHA> \\
        --synchronized-init-dir /path/to/synchronized_init \\
        --gigapath-checkpoint /path/to/slide_encoder.pt   # required iff model.params.use_global_slide
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from gen3_multiscale.training.train import config_fingerprint, config_identity_fingerprint

# Confirmed dead by direct inspection of train.py -- see this module's
# own docstring for the full explanation of why each is never read.
_DEAD_MODEL_PARAM_FIELDS = ("n_genes", "gex_feature_dim")
_DEAD_REQUIRED_FINGERPRINT_FIELDS = (
    "gene_vocabulary", "train_mask_bank", "validation_mask_bank", "test_mask_bank",
)


def resolve_experiment_config(
    base_config_path: str | Path,
    *,
    gen3_manifest_path: str,
    tile_encoder_revision: str,
    synchronized_init_dir: str,
    checkpoint_dir: str | None = None,
    gigapath_checkpoint: str | None = None,
    gene_residual_basis: str | None = None,
    architecture3_conditioner_checkpoint: str | None = None,
) -> dict:
    """Load `base_config_path` (one of the committed `configs/
    architectureN.yaml` templates, or any config sharing its schema),
    apply the given deployment overrides, drop the confirmed-dead fields
    above, and return the fully-resolved config dict. Raises `ValueError`
    (fail closed) listing every field this SPECIFIC architecture/config
    still requires that was not supplied -- never returns a config with a
    silently-still-null required field. Does not write anything to disk;
    see `resolve_and_save_experiment_config` for the persisted, hashed,
    immutable-output version this module's CLI actually uses."""
    base_config_path = Path(base_config_path)
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = dict(config.get("data") or {})
    data_cfg["gen3_manifest_path"] = str(gen3_manifest_path)
    data_cfg["tile_encoder_revision"] = str(tile_encoder_revision)
    config["data"] = data_cfg

    training_cfg = dict(config.get("training") or {})
    training_cfg["synchronized_init_dir"] = str(synchronized_init_dir)
    if checkpoint_dir is not None:
        training_cfg["checkpoint_dir"] = str(checkpoint_dir)
    config["training"] = training_cfg

    model_params = dict((config.get("model") or {}).get("params") or {})
    for field in _DEAD_MODEL_PARAM_FIELDS:
        model_params.pop(field, None)
    config["model"] = dict(config.get("model") or {})
    config["model"]["params"] = model_params

    required_fingerprints = dict(config.get("required_fingerprints") or {})
    for field in _DEAD_REQUIRED_FINGERPRINT_FIELDS:
        required_fingerprints.pop(field, None)
    if gigapath_checkpoint is not None:
        required_fingerprints["gigapath_checkpoint"] = str(gigapath_checkpoint)
    if gene_residual_basis is not None:
        required_fingerprints["gene_residual_basis"] = str(gene_residual_basis)
    if architecture3_conditioner_checkpoint is not None:
        required_fingerprints["architecture3_conditioner_checkpoint"] = str(architecture3_conditioner_checkpoint)
    config["required_fingerprints"] = required_fingerprints

    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    missing: list[str] = []
    if not data_cfg.get("gen3_manifest_path"):
        missing.append("data.gen3_manifest_path")
    if not data_cfg.get("tile_encoder_revision"):
        missing.append("data.tile_encoder_revision")
    if not training_cfg.get("synchronized_init_dir"):
        missing.append("training.synchronized_init_dir")
    if model_params.get("use_global_slide") and not required_fingerprints.get("gigapath_checkpoint"):
        missing.append("required_fingerprints.gigapath_checkpoint (required: model.params.use_global_slide=true)")
    if architecture_id == "4":
        if not required_fingerprints.get("gene_residual_basis"):
            missing.append("required_fingerprints.gene_residual_basis (required: model.architecture=4)")
        if not required_fingerprints.get("architecture3_conditioner_checkpoint"):
            missing.append(
                "required_fingerprints.architecture3_conditioner_checkpoint (required: model.architecture=4)"
            )
    if missing:
        raise ValueError(
            f"{base_config_path}: cannot resolve this config -- still missing required field(s): {missing}. "
            "Every deployment-specific field this architecture/config combination actually needs must be "
            "supplied before a resolved config is considered runnable"
        )
    return config


def resolve_and_save_experiment_config(
    base_config_path: str | Path, output_path: str | Path, *, force: bool = False, **override_kwargs,
) -> dict:
    """`resolve_experiment_config` plus a durable, hashed, IMMUTABLE
    write. `output_path` must not already exist unless `force=True` --
    other orchestration steps (synchronized-init preparation, preflight,
    training itself) hash-bind to a resolved config's identity fingerprint
    once it exists; silently overwriting it out from under them would
    invalidate those bindings without anything noticing. Writes two
    files: `output_path` itself (the resolved config, loadable directly
    by `train.py`/`fit_architecture4_residual_basis.py`/`gen3_evaluator.py`
    exactly like any other config), and `<output_path>.resolved_identity.json`
    (the base config path, every override applied, and both
    `config_fingerprint`/`config_identity_fingerprint` of the resolved
    result -- a durable record of exactly what was resolved and its
    identity, independent of the resolved YAML's own bytes surviving
    unmodified). Returns the resolved config dict."""
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists -- refusing to overwrite a resolved config (other steps may "
            "already hash-bind to its identity). Pass force=True if you genuinely intend to replace it"
        )
    resolved = resolve_experiment_config(base_config_path, **override_kwargs)

    identity_record = {
        "version": 1,
        "kind": "gen3_resolved_experiment_config_identity",
        "base_config_path": str(base_config_path),
        "overrides": {k: (str(v) if v is not None else None) for k, v in override_kwargs.items()},
        "config_fingerprint": config_fingerprint(resolved),
        "config_identity_fingerprint": config_identity_fingerprint(resolved),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    output_tmp.write_text(yaml.safe_dump(resolved, sort_keys=False))
    os.replace(output_tmp, output_path)

    identity_path = output_path.with_name(f"{output_path.name}.resolved_identity.json")
    identity_tmp = identity_path.with_name(f"{identity_path.name}.tmp.{os.getpid()}")
    identity_tmp.write_text(json.dumps(identity_record, indent=2, sort_keys=True, default=str))
    os.replace(identity_tmp, identity_path)

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", required=True, help="One of configs/architectureN.yaml, or any config sharing its schema")
    parser.add_argument("--output", required=True, help="Immutable output path for the resolved config YAML")
    parser.add_argument("--gen3-manifest-path", required=True)
    parser.add_argument("--tile-encoder-revision", required=True)
    parser.add_argument("--synchronized-init-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None, help="Overrides the base config's own training.checkpoint_dir if given")
    parser.add_argument("--gigapath-checkpoint", default=None, help="Required iff model.params.use_global_slide=true")
    parser.add_argument("--gene-residual-basis", default=None, help="Required iff model.architecture=4")
    parser.add_argument("--architecture3-conditioner-checkpoint", default=None, help="Required iff model.architecture=4")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing resolved config at --output")
    args = parser.parse_args()

    try:
        resolve_and_save_experiment_config(
            args.base_config, args.output, force=args.force,
            gen3_manifest_path=args.gen3_manifest_path, tile_encoder_revision=args.tile_encoder_revision,
            synchronized_init_dir=args.synchronized_init_dir, checkpoint_dir=args.checkpoint_dir,
            gigapath_checkpoint=args.gigapath_checkpoint, gene_residual_basis=args.gene_residual_basis,
            architecture3_conditioner_checkpoint=args.architecture3_conditioner_checkpoint,
        )
    except Exception as exc:
        print(f"resolve_experiment_config failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"resolved config written to {args.output}")


if __name__ == "__main__":
    main()
