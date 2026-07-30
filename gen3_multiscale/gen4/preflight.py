"""Gen4 static preflight/audit -- GEN4_CONTRACT.md section 9/12.

Validates a resolved Gen4 config's schema/consistency WITHOUT touching a
GPU, downloading a checkpoint, or requiring real weights to be loaded --
the mandatory gate a smoke run refuses to start without passing. Where an
on-disk cache already exists, its recorded provenance/shape is
cross-checked against the config's own declared dims; a cache that does
not exist yet is reported as a named, explicit gap, never silently
skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from gen3_multiscale.gen4.model_factory import ARM_TABLE

_REQUIRED_TOP_LEVEL_KEYS = {"model", "masking", "data", "evaluation", "required_fingerprints", "training"}

# Per-arm required_fingerprints keys, mirroring configs/gen4/*.yaml's own
# required_fingerprints blocks -- GEN4_CONTRACT.md section 6/7/8.
#
# Item 2 (six-launch-blocker audit): "compare cache checkpoint/revision/
# vocabulary/preprocessing identities against the resolved experiment" --
# uni2_package_version/uni2_preprocessing_spec and
# scfoundation_package_version/scfoundation_preprocessing_spec are NEW
# keys (the prior schema only pinned checkpoint+revision/vocab, silently
# never checking package version or preprocessing spec at all).
_ARM_REQUIRED_FINGERPRINTS = {
    "gen4a": {"uni2_checkpoint", "uni2_revision", "uni2_package_version", "uni2_preprocessing_spec"},
    "gen4b": {"gigapath_checkpoint", "scfoundation_checkpoint", "scfoundation_vocab", "scfoundation_package_version", "scfoundation_preprocessing_spec"},
    "gen4c": {
        "uni2_checkpoint", "uni2_revision", "uni2_package_version", "uni2_preprocessing_spec",
        "scfoundation_checkpoint", "scfoundation_vocab", "scfoundation_package_version", "scfoundation_preprocessing_spec",
    },
    "gen4d": {"stpath_checkpoint", "stpath_gene_vocab", "gigapath_checkpoint"},
    "gen4e": {
        "stpath_checkpoint", "stpath_gene_vocab", "gigapath_checkpoint",
        "uni2_checkpoint", "uni2_revision", "uni2_package_version", "uni2_preprocessing_spec",
        "scfoundation_checkpoint", "scfoundation_vocab", "scfoundation_package_version", "scfoundation_preprocessing_spec",
    },
}

# Which per-sample cache modality(ies) each arm actually needs coverage
# for (Item 2's "exact manifest-derived cache coverage" -- distinct from
# `_ARM_REQUIRED_FINGERPRINTS` above, which only checks the CONFIG has
# the keys; this drives which real, per-sample on-disk cache
# `audit_gen4_manifest_cache_coverage` requires to exist for EVERY
# manifest sample the resolved experiment will actually use).
_ARM_CACHE_MODALITIES = {
    "gen4a": {"uni2"},
    "gen4b": {"scfoundation"},
    "gen4c": {"uni2", "scfoundation"},
    "gen4d": set(),  # STPath/GigaPath consumed live inside the conditioner -- no per-sample spot cache to audit here
    "gen4e": {"uni2", "scfoundation"},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_audit_gen4_config(config: dict) -> dict:
    """Raises ValueError on the first structural problem found (fail
    closed); returns a report dict of what it checked on success."""
    missing_top = _REQUIRED_TOP_LEVEL_KEYS - set(config)
    if missing_top:
        raise ValueError(f"config is missing required top-level section(s): {sorted(missing_top)}")

    model_cfg = config["model"]
    arm = str(model_cfg.get("arm", ""))
    if arm not in ARM_TABLE:
        raise ValueError(f"model.arm must be one of {sorted(ARM_TABLE)}, got {arm!r}")
    kind = str(model_cfg.get("kind", ""))
    if kind not in {"conditioner", "flow"}:
        raise ValueError(f"model.kind must be 'conditioner' or 'flow', got {kind!r}")

    params = model_cfg.get("params") or {}
    if not params.get("n_genes") is None and int(params["n_genes"]) <= 0:
        raise ValueError("model.params.n_genes must be positive when set")
    image_feature_dim = params.get("image_feature_dim")
    if image_feature_dim is not None and int(image_feature_dim) <= 0:
        raise ValueError("model.params.image_feature_dim must be positive")
    if arm in {"gen4b", "gen4c", "gen4e"}:
        context_dim = params.get("gex_context_embedding_dim")
        if not context_dim or int(context_dim) <= 0:
            raise ValueError(f"arm {arm!r} requires a positive model.params.gex_context_embedding_dim")

    strata = (config.get("masking") or {}).get("strata")
    if not strata:
        raise ValueError("masking.strata must be a non-empty list")

    fingerprints = config.get("required_fingerprints") or {}
    expected = _ARM_REQUIRED_FINGERPRINTS[arm]
    missing_keys = expected - set(fingerprints)
    if missing_keys:
        raise ValueError(f"arm {arm!r} config is missing required_fingerprints key(s): {sorted(missing_keys)}")
    unset = sorted(key for key in expected if not fingerprints.get(key))
    if kind == "flow":
        for key in ("gene_residual_basis", "gen4_conditioner_checkpoint"):
            if key not in fingerprints:
                raise ValueError(f"a flow config is missing required_fingerprints.{key}")
            if not fingerprints.get(key):
                unset.append(key)

    return {
        "arm": arm, "kind": kind, "checked_required_fingerprints": sorted(expected),
        "unset_required_fingerprints": sorted(set(unset)),
        "ready_for_real_training": len(unset) == 0,
    }


def audit_uni2_cache_matches_config(cache_root: str | Path, sample_id: str, config: dict, *, require_exists: bool = False) -> dict:
    """If a UNI2 spot-feature cache already exists for this sample, its
    recorded FULL identity (output_dim, revision, checkpoint content
    hash, package version, preprocessing spec) must match the config's
    own declared values -- comparison only happens for fingerprint keys
    the config actually SETS (an unset/null key means "not yet pinned,"
    consistent with required_fingerprints' own documented-placeholder
    convention; `static_audit_gen4_config` is what enforces every key is
    eventually set for a "ready" config).

    `require_exists=False` (default): reports (does not raise) when the
    cache simply does not exist yet -- an expected pre-training state
    when called standalone. `require_exists=True` (used by
    `audit_gen4_manifest_cache_coverage`, Item 2's "missing ... caches
    must fail before model construction" gate): a missing cache is a
    hard FileNotFoundError."""
    from gen3_multiscale.gen4.uni2_spot_cache import _cache_path

    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        if require_exists:
            raise FileNotFoundError(
                f"UNI2 spot-feature cache required for {sample_id!r} (arm needs image_feature_source="
                f"'precomputed' from UNI2) but missing: {path}. Build it with "
                "gen4.uni2_spot_cache.build_uni2_spot_feature_cache before training."
            )
        return {"cache_exists": False, "path": str(path)}
    cached = np.load(path, allow_pickle=False)
    fingerprints = config.get("required_fingerprints") or {}
    declared_dim = int((config["model"].get("params") or {}).get("image_feature_dim") or 0)
    declared_revision = fingerprints.get("uni2_revision")
    declared_checkpoint_path = fingerprints.get("uni2_checkpoint")
    declared_package_version = fingerprints.get("uni2_package_version")
    declared_preprocessing_spec = fingerprints.get("uni2_preprocessing_spec")
    actual_dim = int(np.asarray(cached["uni2_output_dim"]).item())
    actual_revision = str(cached["uni2_pinned_revision"])
    actual_checkpoint_sha256 = str(cached["uni2_checkpoint_sha256"])
    actual_package_version = str(cached["uni2_package_version"])
    actual_preprocessing_spec = str(cached["uni2_preprocessing_spec"])
    mismatches = []
    if declared_dim and actual_dim != declared_dim:
        mismatches.append(f"image_feature_dim: config={declared_dim} cache={actual_dim}")
    if declared_revision and actual_revision != declared_revision:
        mismatches.append(f"uni2_revision: config={declared_revision} cache={actual_revision}")
    if declared_package_version and actual_package_version != declared_package_version:
        mismatches.append(f"uni2_package_version: config={declared_package_version} cache={actual_package_version}")
    if declared_preprocessing_spec and actual_preprocessing_spec != declared_preprocessing_spec:
        mismatches.append(f"uni2_preprocessing_spec: config={declared_preprocessing_spec} cache={actual_preprocessing_spec}")
    if declared_checkpoint_path:
        checkpoint_path = Path(str(declared_checkpoint_path))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"required_fingerprints.uni2_checkpoint={checkpoint_path} does not exist")
        expected_checkpoint_sha256 = _sha256_file(checkpoint_path)
        if expected_checkpoint_sha256 != actual_checkpoint_sha256:
            mismatches.append(
                f"uni2_checkpoint content sha256: config file={expected_checkpoint_sha256} cache={actual_checkpoint_sha256}"
            )
    if mismatches:
        raise ValueError(f"UNI2 cache {path} does not match config: {'; '.join(mismatches)}")
    return {
        "cache_exists": True, "path": str(path), "output_dim": actual_dim, "revision": actual_revision,
        "checkpoint_sha256": actual_checkpoint_sha256, "package_version": actual_package_version,
        "preprocessing_spec": actual_preprocessing_spec,
    }


def audit_scfoundation_cache_matches_config(cache_root: str | Path, sample_id: str, config: dict, *, require_exists: bool = False) -> dict:
    """Same discipline as audit_uni2_cache_matches_config, applied to
    scFoundation's own FULL identity (output_dim, vocabulary sha256,
    checkpoint content hash, package version, preprocessing spec)."""
    from gen3_multiscale.gen4.scfoundation_cache import _cache_path

    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        if require_exists:
            raise FileNotFoundError(
                f"scFoundation spot-feature cache required for {sample_id!r} but missing: {path}. Build it "
                "with gen4.scfoundation_cache.build_scfoundation_spot_feature_cache before training."
            )
        return {"cache_exists": False, "path": str(path)}
    cached = np.load(path, allow_pickle=False)
    fingerprints = config.get("required_fingerprints") or {}
    declared_dim = int((config["model"].get("params") or {}).get("gex_context_embedding_dim") or 0)
    declared_vocab = fingerprints.get("scfoundation_vocab")
    declared_checkpoint_path = fingerprints.get("scfoundation_checkpoint")
    declared_package_version = fingerprints.get("scfoundation_package_version")
    declared_preprocessing_spec = fingerprints.get("scfoundation_preprocessing_spec")
    actual_dim = int(np.asarray(cached["scfoundation_output_dim"]).item())
    actual_vocab_sha256 = str(cached["scfoundation_vocab_sha256"])
    actual_checkpoint_sha256 = str(cached["scfoundation_checkpoint_sha256"])
    actual_package_version = str(cached["scfoundation_package_version"])
    actual_preprocessing_spec = str(cached["scfoundation_preprocessing_spec"])
    mismatches = []
    if declared_dim and actual_dim != declared_dim:
        mismatches.append(f"gex_context_embedding_dim: config={declared_dim} cache={actual_dim}")
    if declared_package_version and actual_package_version != declared_package_version:
        mismatches.append(f"scfoundation_package_version: config={declared_package_version} cache={actual_package_version}")
    if declared_preprocessing_spec and actual_preprocessing_spec != declared_preprocessing_spec:
        mismatches.append(f"scfoundation_preprocessing_spec: config={declared_preprocessing_spec} cache={actual_preprocessing_spec}")
    if declared_vocab:
        vocab_path = Path(str(declared_vocab))
        if vocab_path.is_file():
            expected_vocab_sha256 = _sha256_file(vocab_path)
            if expected_vocab_sha256 != actual_vocab_sha256:
                mismatches.append(
                    f"scfoundation_vocab content sha256: config file={expected_vocab_sha256} cache={actual_vocab_sha256}"
                )
        elif str(declared_vocab) != actual_vocab_sha256:
            # Not every deployment stores a local vocab FILE at this path (some
            # pin a bare sha256/revision string directly) -- fall back to a
            # direct string comparison rather than silently skipping the check.
            mismatches.append(f"scfoundation_vocab: config={declared_vocab} cache={actual_vocab_sha256}")
    if declared_checkpoint_path:
        checkpoint_path = Path(str(declared_checkpoint_path))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"required_fingerprints.scfoundation_checkpoint={checkpoint_path} does not exist")
        expected_checkpoint_sha256 = _sha256_file(checkpoint_path)
        if expected_checkpoint_sha256 != actual_checkpoint_sha256:
            mismatches.append(
                f"scfoundation_checkpoint content sha256: config file={expected_checkpoint_sha256} cache={actual_checkpoint_sha256}"
            )
    if mismatches:
        raise ValueError(f"scFoundation cache {path} does not match config: {'; '.join(mismatches)}")
    return {
        "cache_exists": True, "path": str(path), "output_dim": actual_dim, "vocab_sha256": actual_vocab_sha256,
        "checkpoint_sha256": actual_checkpoint_sha256, "package_version": actual_package_version,
        "preprocessing_spec": actual_preprocessing_spec,
    }


def audit_gen4_manifest_cache_coverage(cache_root: str | Path, sample_ids: list[str], config: dict) -> dict:
    """Item 2 (six-launch-blocker audit): "Make preflight require exact
    manifest-derived cache coverage ... Missing or wrong-modality caches
    must fail before model construction." Unlike the two per-sample audit
    functions above (soft-by-default, for ad hoc/manual checks), THIS is
    the mandatory pre-training gate: for the resolved experiment's arm
    and its REAL, resolved manifest sample_ids (never a hardcoded/partial
    list), every modality that arm actually consumes
    (`_ARM_CACHE_MODALITIES`) must have a real, identity-matching cache
    for EVERY sample -- a single missing or mismatched cache raises
    before this function returns, and therefore before the caller may
    proceed to model construction."""
    if not sample_ids:
        raise ValueError("audit_gen4_manifest_cache_coverage requires at least one resolved manifest sample_id")
    arm = str((config.get("model") or {}).get("arm", ""))
    modalities = _ARM_CACHE_MODALITIES.get(arm)
    if modalities is None:
        raise ValueError(f"unknown model.arm {arm!r} -- expected one of {sorted(_ARM_CACHE_MODALITIES)}")
    checked = {"arm": arm, "modalities": sorted(modalities), "n_samples": len(sample_ids), "samples": {}}
    for sample_id in sample_ids:
        sample_report = {}
        if "uni2" in modalities:
            sample_report["uni2"] = audit_uni2_cache_matches_config(cache_root, sample_id, config, require_exists=True)
        if "scfoundation" in modalities:
            sample_report["scfoundation"] = audit_scfoundation_cache_matches_config(cache_root, sample_id, config, require_exists=True)
        checked["samples"][sample_id] = sample_report
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    report = static_audit_gen4_config(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
