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
import json
from pathlib import Path

import numpy as np
import yaml

from gen3_multiscale.gen4.model_factory import ARM_TABLE

_REQUIRED_TOP_LEVEL_KEYS = {"model", "masking", "data", "evaluation", "required_fingerprints", "training"}

# Per-arm required_fingerprints keys, mirroring configs/gen4/*.yaml's own
# required_fingerprints blocks -- GEN4_CONTRACT.md section 6/7/8.
_ARM_REQUIRED_FINGERPRINTS = {
    "gen4a": {"uni2_checkpoint", "uni2_revision"},
    "gen4b": {"gigapath_checkpoint", "scfoundation_checkpoint", "scfoundation_vocab"},
    "gen4c": {"uni2_checkpoint", "uni2_revision", "scfoundation_checkpoint", "scfoundation_vocab"},
    "gen4d": {"stpath_checkpoint", "stpath_gene_vocab", "gigapath_checkpoint"},
}


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
    if arm in {"gen4b", "gen4c"}:
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


def audit_uni2_cache_matches_config(cache_root: str | Path, sample_id: str, config: dict) -> dict:
    """Soft check: if a UNI2 spot-feature cache already exists for this
    sample, its recorded output_dim/revision must match the config's own
    declared values. Reports (does not raise) when the cache simply does
    not exist yet -- that is an expected pre-training state, not a
    config error."""
    from gen3_multiscale.gen4.uni2_spot_cache import _cache_path

    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        return {"cache_exists": False, "path": str(path)}
    cached = np.load(path, allow_pickle=False)
    declared_dim = int((config["model"].get("params") or {}).get("image_feature_dim") or 0)
    declared_revision = (config.get("required_fingerprints") or {}).get("uni2_revision")
    actual_dim = int(np.asarray(cached["uni2_output_dim"]).item())
    actual_revision = str(cached["uni2_pinned_revision"])
    mismatches = []
    if declared_dim and actual_dim != declared_dim:
        mismatches.append(f"image_feature_dim: config={declared_dim} cache={actual_dim}")
    if declared_revision and actual_revision != declared_revision:
        mismatches.append(f"uni2_revision: config={declared_revision} cache={actual_revision}")
    if mismatches:
        raise ValueError(f"UNI2 cache {path} does not match config: {'; '.join(mismatches)}")
    return {"cache_exists": True, "path": str(path), "output_dim": actual_dim, "revision": actual_revision}


def audit_scfoundation_cache_matches_config(cache_root: str | Path, sample_id: str, config: dict) -> dict:
    from gen3_multiscale.gen4.scfoundation_cache import _cache_path

    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        return {"cache_exists": False, "path": str(path)}
    cached = np.load(path, allow_pickle=False)
    declared_dim = int((config["model"].get("params") or {}).get("gex_context_embedding_dim") or 0)
    actual_dim = int(np.asarray(cached["scfoundation_output_dim"]).item())
    if declared_dim and actual_dim != declared_dim:
        raise ValueError(
            f"scFoundation cache {path} output_dim={actual_dim} does not match config "
            f"gex_context_embedding_dim={declared_dim}"
        )
    return {"cache_exists": True, "path": str(path), "output_dim": actual_dim}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    report = static_audit_gen4_config(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
