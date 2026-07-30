"""Gen5 static preflight/audit -- GEN5_CONTRACT.md section 8. Mirrors
`gen4/preflight.py`'s discipline; reuses `gen4.preflight._ARM_REQUIRED_FINGERPRINTS`
for the conditioning-side fingerprint requirements and adds the
autoencoder/flow-specific ones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from gen3_multiscale.gen4.preflight import _ARM_REQUIRED_FINGERPRINTS
from gen3_multiscale.gen5.model_factory import GEN5_TO_GEN4_ARM

_REQUIRED_TOP_LEVEL_KEYS = {"model", "masking", "data", "evaluation", "required_fingerprints", "training"}
_GEN5_EXTRA_REQUIRED_FINGERPRINTS = {"expression_autoencoder_checkpoint", "gen4_conditioner_checkpoint"}


def static_audit_gen5_config(config: dict) -> dict:
    missing_top = _REQUIRED_TOP_LEVEL_KEYS - set(config)
    if missing_top:
        raise ValueError(f"config is missing required top-level section(s): {sorted(missing_top)}")

    model_cfg = config["model"]
    arm = str(model_cfg.get("arm", ""))
    if arm not in GEN5_TO_GEN4_ARM:
        raise ValueError(f"model.arm must be one of {sorted(GEN5_TO_GEN4_ARM)}, got {arm!r}")
    if str(model_cfg.get("kind", "")) != "latent_flow":
        raise ValueError(f"model.kind must be 'latent_flow', got {model_cfg.get('kind')!r}")

    params = model_cfg.get("params") or {}
    latent_dim = params.get("latent_dim")
    if latent_dim is not None and int(latent_dim) <= 0:
        raise ValueError("model.params.latent_dim must be positive when set")
    if arm in {"gen5b", "gen5c", "gen5e"}:
        context_dim = params.get("gex_context_embedding_dim")
        if not context_dim or int(context_dim) <= 0:
            raise ValueError(f"arm {arm!r} requires a positive model.params.gex_context_embedding_dim")

    strata = (config.get("masking") or {}).get("strata")
    if not strata:
        raise ValueError("masking.strata must be a non-empty list")

    fingerprints = config.get("required_fingerprints") or {}
    expected = set(_ARM_REQUIRED_FINGERPRINTS[GEN5_TO_GEN4_ARM[arm]]) | _GEN5_EXTRA_REQUIRED_FINGERPRINTS
    missing_keys = expected - set(fingerprints)
    if missing_keys:
        raise ValueError(f"arm {arm!r} config is missing required_fingerprints key(s): {sorted(missing_keys)}")
    unset = sorted(key for key in expected if not fingerprints.get(key))

    return {
        "arm": arm, "kind": "latent_flow", "checked_required_fingerprints": sorted(expected),
        "unset_required_fingerprints": unset, "ready_for_real_training": len(unset) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(static_audit_gen5_config(config), indent=2))


if __name__ == "__main__":
    main()
