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
combination actually requires is still unresolved or malformed, drop dead
fields the real trainer never reads (see below) rather than leave them as
misleading nulls, and write the fully-resolved config as a real
TRANSACTIONAL BUNDLE -- a staged-then-atomically-renamed directory
containing `config.yaml` and `identity.json` together, never as two
independently-written loose files.

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

Codex re-audit of commit 7a2d819, findings #4 and #5, both addressed
here:

  #4: "The resolver's 'immutable pair' is not transactional or verified.
  It writes YAML first and the identity sidecar second. A crash can
  leave only one artifact, and --force can overwrite them
  inconsistently. The sidecar also lacks the exact YAML-file SHA256, and
  no verified loader currently recomputes the fingerprints before
  consuming it." Confirmed real -- fixed by making a resolved config a
  real BUNDLE DIRECTORY (mirroring `training/checkpoint.py`'s own
  already-established staging-directory-then-atomic-rename pattern, the
  same discipline this codebase already uses everywhere else an
  artifact's identity matters): `resolve_and_save_experiment_config`
  writes `config.yaml` and `identity.json` (LAST, so its presence is
  itself the "this bundle is complete" signal) into a staging directory,
  then performs ONE atomic `os.rename` of the whole directory into
  place -- a crash at any point before that rename leaves the FINAL
  bundle path untouched (nothing partial ever appears there), and the
  final directory can never contain the config without its
  identity.json or vice versa. `identity.json` now also records
  `config_yaml_sha256` (the exact bytes of the bundled `config.yaml`)
  and `base_template_sha256` (the exact bytes of the base template file
  resolution started from), neither previously recorded. A new
  `load_verified_resolved_config` is the one function anything in the
  future orchestrator should use to CONSUME a resolved bundle: it
  recomputes `config_yaml_sha256` from the bundle's own `config.yaml`
  bytes, and recomputes both `config_fingerprint`/
  `config_identity_fingerprint` from the LOADED config dict, comparing
  every one against `identity.json`'s recorded values before returning
  anything -- a bundle whose files disagree with its own recorded
  identity (corruption, a hand-edit, a partially-applied patch) is
  refused, never silently trusted.

  #5: "Reject None, blank strings, unknown architecture IDs, and
  non-40-hex tile revisions. Avoid --force; generate a new immutable
  resolved-config bundle instead." Confirmed real: `data.gen3_manifest_
  path`/`tile_encoder_revision`/`training.synchronized_init_dir` were
  stringified UNCONDITIONALLY (`str(gen3_manifest_path)`, etc.) before
  the "is this still missing" check ran -- a caller passing `None`
  directly (bypassing argparse's own `required=True`, which only guards
  the CLI, not a direct Python call) produced the literal STRING
  `"None"`, which is TRUTHY and therefore silently passed the `if not
  data_cfg.get(...)` missing-field check. Fixed: every deployment field
  is now validated as a non-None, non-blank string BEFORE being used,
  never stringified first and validated after. `tile_encoder_revision`
  is now required to match `^[0-9a-f]{40}$` (an exact, pinned, lowercase
  40-hex Hugging Face commit SHA -- the same format `train.py`'s own
  `expected_tile_encoder_provenance` already assumes it to be) rather
  than accepted as any non-empty string. `model.architecture` (read from
  the loaded base template itself, not a CLI argument, but a real
  defensive check against a malformed/tampered template) must be one of
  `"1"`/`"2"`/`"3"`/`"4"`. `force`/`--force` is REMOVED entirely --
  `resolve_and_save_experiment_config` always refuses an already-existing
  output bundle path unconditionally; the correct way to "replace" a
  resolved config is to write a genuinely NEW, distinctly-named bundle,
  never to overwrite an old one other orchestration steps may already
  hash-bind to.

    python -m gen3_multiscale.scripts.resolve_experiment_config \\
        --base-config gen3_multiscale/configs/architecture3.yaml \\
        --output-dir gen3_multiscale/results/my_experiment/resolved_architecture3 \\
        --gen3-manifest-path /path/to/dataset_manifest.json \\
        --tile-encoder-revision <pinned 40-hex HF commit SHA> \\
        --synchronized-init-dir /path/to/synchronized_init \\
        --gigapath-checkpoint /path/to/slide_encoder.pt   # required iff model.params.use_global_slide
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
_VALID_ARCHITECTURE_IDS = frozenset({"1", "2", "3", "4"})
_TILE_ENCODER_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_SCHEMA_VERSION = 2


def _require_non_blank_string(value, field_name: str) -> str:
    """Codex re-audit of commit 7a2d819, finding #5: reject `None` and
    blank/whitespace-only strings EXPLICITLY, before any stringification
    -- `str(None) == "None"`, a truthy string that would otherwise
    silently sail through a later `if not value:` check."""
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    still requires that was not supplied, malformed, or blank -- never
    returns a config with a silently-still-null or silently-`"None"`-
    stringified required field. Does not write anything to disk; see
    `resolve_and_save_experiment_config` for the persisted, hashed,
    transactional-bundle version this module's CLI actually uses."""
    base_config_path = Path(base_config_path)
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    architecture_id = str((config.get("model") or {}).get("architecture", ""))
    if architecture_id not in _VALID_ARCHITECTURE_IDS:
        raise ValueError(
            f"{base_config_path}: model.architecture={architecture_id!r} is not one of "
            f"{sorted(_VALID_ARCHITECTURE_IDS)} -- refusing to resolve a config for an unknown architecture"
        )

    gen3_manifest_path = _require_non_blank_string(gen3_manifest_path, "gen3_manifest_path")
    tile_encoder_revision = _require_non_blank_string(tile_encoder_revision, "tile_encoder_revision")
    if not _TILE_ENCODER_REVISION_RE.match(tile_encoder_revision):
        raise ValueError(
            f"tile_encoder_revision must be a pinned, lowercase 40-character hexadecimal Hugging Face "
            f"commit SHA, got {tile_encoder_revision!r}"
        )
    synchronized_init_dir = _require_non_blank_string(synchronized_init_dir, "synchronized_init_dir")
    if checkpoint_dir is not None:
        checkpoint_dir = _require_non_blank_string(checkpoint_dir, "checkpoint_dir")
    if gigapath_checkpoint is not None:
        gigapath_checkpoint = _require_non_blank_string(gigapath_checkpoint, "gigapath_checkpoint")
    if gene_residual_basis is not None:
        gene_residual_basis = _require_non_blank_string(gene_residual_basis, "gene_residual_basis")
    if architecture3_conditioner_checkpoint is not None:
        architecture3_conditioner_checkpoint = _require_non_blank_string(
            architecture3_conditioner_checkpoint, "architecture3_conditioner_checkpoint",
        )

    data_cfg = dict(config.get("data") or {})
    data_cfg["gen3_manifest_path"] = gen3_manifest_path
    data_cfg["tile_encoder_revision"] = tile_encoder_revision
    config["data"] = data_cfg

    training_cfg = dict(config.get("training") or {})
    training_cfg["synchronized_init_dir"] = synchronized_init_dir
    if checkpoint_dir is not None:
        training_cfg["checkpoint_dir"] = checkpoint_dir
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
        required_fingerprints["gigapath_checkpoint"] = gigapath_checkpoint
    if gene_residual_basis is not None:
        required_fingerprints["gene_residual_basis"] = gene_residual_basis
    if architecture3_conditioner_checkpoint is not None:
        required_fingerprints["architecture3_conditioner_checkpoint"] = architecture3_conditioner_checkpoint
    config["required_fingerprints"] = required_fingerprints

    missing: list[str] = []
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
    base_config_path: str | Path, output_dir: str | Path, **override_kwargs,
) -> dict:
    """`resolve_experiment_config` plus a durable, hashed, TRANSACTIONAL
    write. `output_dir` is the resolved config's BUNDLE directory --
    must not already exist; there is no `force` override (Codex re-audit
    of commit 7a2d819, finding #5: "avoid --force; generate a new
    immutable resolved-config bundle instead") -- other orchestration
    steps hash-bind to a resolved bundle's identity once it exists, so
    "replacing" one always means writing a genuinely new, distinctly-
    named bundle, never overwriting an old one out from under whatever
    already trusts it.

    Writes into a staging directory first, then performs ONE atomic
    `os.rename` into `output_dir` (mirrors `training/checkpoint.py`'s own
    staging-then-atomic-rename bundle discipline) -- a crash at any point
    before that rename leaves `output_dir` itself completely untouched;
    the bundle directory contains exactly `config.yaml` (the resolved
    config, loadable directly by `train.py`/`fit_architecture4_residual_
    basis.py`/`gen3_evaluator.py` exactly like any other config) and
    `identity.json`, written LAST so its presence is itself proof the
    bundle is complete (base config path, every override applied,
    `config_yaml_sha256` -- the exact bytes of the bundled config.yaml
    -- `base_template_sha256` -- the exact bytes of the base template
    file resolution started from -- and both `config_fingerprint`/
    `config_identity_fingerprint` of the resolved result). Returns the
    resolved config dict; see `load_verified_resolved_config` for the
    verified read path a consumer should use instead of loading
    `config.yaml` directly."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists -- refusing to overwrite a resolved config bundle (other steps "
            "may already hash-bind to its identity). Write a new, distinctly-named bundle instead"
        )
    resolved = resolve_experiment_config(base_config_path, **override_kwargs)
    base_config_path = Path(base_config_path)
    config_yaml_bytes = yaml.safe_dump(resolved, sort_keys=False).encode("utf-8")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.staging.{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"stale staging directory {staging_dir} already exists -- refusing to proceed")
    staging_dir.mkdir(parents=True)
    try:
        config_path = staging_dir / "config.yaml"
        config_path.write_bytes(config_yaml_bytes)

        identity_record = {
            "version": _IDENTITY_SCHEMA_VERSION,
            "kind": "gen3_resolved_experiment_config_identity",
            "base_config_path": str(base_config_path),
            "base_template_sha256": _file_sha256(base_config_path),
            "config_yaml_sha256": hashlib.sha256(config_yaml_bytes).hexdigest(),
            "overrides": {k: v for k, v in override_kwargs.items()},
            "config_fingerprint": config_fingerprint(resolved),
            "config_identity_fingerprint": config_identity_fingerprint(resolved),
        }
        identity_path = staging_dir / "identity.json"
        identity_path.write_text(json.dumps(identity_record, indent=2, sort_keys=True, default=str))

        os.rename(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return resolved


def load_verified_resolved_config(bundle_dir: str | Path) -> dict:
    """The one function anything in the future orchestrator should use
    to CONSUME a resolved config bundle written by
    `resolve_and_save_experiment_config` -- Codex re-audit of commit
    7a2d819, finding #4: "Add load_verified_resolved_config() that
    checks file hash and recomputes both semantic fingerprints. The
    future orchestrator must use this loader." Recomputes
    `config_yaml_sha256` from the bundle's own `config.yaml` bytes, and
    recomputes both `config_fingerprint`/`config_identity_fingerprint`
    from the LOADED config dict, comparing every one against `identity
    .json`'s recorded values before returning anything -- a bundle whose
    files disagree with its own recorded identity (corruption, a
    hand-edit, a partially-applied patch) is refused, never silently
    trusted. Returns the verified, resolved config dict."""
    bundle_dir = Path(bundle_dir)
    config_path = bundle_dir / "config.yaml"
    identity_path = bundle_dir / "identity.json"
    if not config_path.is_file() or not identity_path.is_file():
        raise ValueError(
            f"{bundle_dir} is not a complete resolved-config bundle -- missing config.yaml and/or "
            "identity.json. Never partially written by resolve_and_save_experiment_config (which "
            "writes both, atomically, or neither); this bundle was corrupted, hand-edited, or is not "
            "a resolved-config bundle at all"
        )
    config_yaml_bytes = config_path.read_bytes()
    identity = json.loads(identity_path.read_text())

    actual_config_yaml_sha256 = hashlib.sha256(config_yaml_bytes).hexdigest()
    recorded_config_yaml_sha256 = identity.get("config_yaml_sha256")
    if actual_config_yaml_sha256 != recorded_config_yaml_sha256:
        raise ValueError(
            f"{bundle_dir}: config.yaml's actual sha256 ({actual_config_yaml_sha256!r}) does not match "
            f"identity.json's recorded config_yaml_sha256 ({recorded_config_yaml_sha256!r}) -- refusing "
            "to trust a resolved-config bundle whose own files disagree with its recorded identity"
        )

    resolved = yaml.safe_load(config_yaml_bytes)
    actual_config_fingerprint = config_fingerprint(resolved)
    recorded_config_fingerprint = identity.get("config_fingerprint")
    if actual_config_fingerprint != recorded_config_fingerprint:
        raise ValueError(
            f"{bundle_dir}: the loaded config's recomputed config_fingerprint "
            f"({actual_config_fingerprint!r}) does not match identity.json's recorded value "
            f"({recorded_config_fingerprint!r}) -- refusing to trust this bundle"
        )
    actual_config_identity_fingerprint = config_identity_fingerprint(resolved)
    recorded_config_identity_fingerprint = identity.get("config_identity_fingerprint")
    if actual_config_identity_fingerprint != recorded_config_identity_fingerprint:
        raise ValueError(
            f"{bundle_dir}: the loaded config's recomputed config_identity_fingerprint "
            f"({actual_config_identity_fingerprint!r}) does not match identity.json's recorded value "
            f"({recorded_config_identity_fingerprint!r}) -- refusing to trust this bundle"
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", required=True, help="One of configs/architectureN.yaml, or any config sharing its schema")
    parser.add_argument("--output-dir", required=True, help="Bundle directory for the resolved config -- must not already exist")
    parser.add_argument("--gen3-manifest-path", required=True)
    parser.add_argument("--tile-encoder-revision", required=True)
    parser.add_argument("--synchronized-init-dir", required=True)
    parser.add_argument("--checkpoint-dir", default=None, help="Overrides the base config's own training.checkpoint_dir if given")
    parser.add_argument("--gigapath-checkpoint", default=None, help="Required iff model.params.use_global_slide=true")
    parser.add_argument("--gene-residual-basis", default=None, help="Required iff model.architecture=4")
    parser.add_argument("--architecture3-conditioner-checkpoint", default=None, help="Required iff model.architecture=4")
    args = parser.parse_args()

    try:
        resolve_and_save_experiment_config(
            args.base_config, args.output_dir,
            gen3_manifest_path=args.gen3_manifest_path, tile_encoder_revision=args.tile_encoder_revision,
            synchronized_init_dir=args.synchronized_init_dir, checkpoint_dir=args.checkpoint_dir,
            gigapath_checkpoint=args.gigapath_checkpoint, gene_residual_basis=args.gene_residual_basis,
            architecture3_conditioner_checkpoint=args.architecture3_conditioner_checkpoint,
        )
    except Exception as exc:
        print(f"resolve_experiment_config failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"resolved config bundle written to {args.output_dir}")


if __name__ == "__main__":
    main()
