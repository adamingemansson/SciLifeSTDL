"""Fail-closed static and manifest-cache preflight for Gen6."""
from __future__ import annotations

from pathlib import Path

from gen3_multiscale.gen4.preflight import (
    _sha256_file, audit_scfoundation_cache_matches_config,
    audit_uni2_cache_matches_config, audit_uni2_dense_cache_matches_config,
)
from gen3_multiscale.gen6.contract import get_gen6_arm_spec

_REQUIRED_TOP_LEVEL = {"model", "masking", "data", "evaluation", "required_fingerprints", "training"}
_UNI2_IDENTITIES = {
    "uni2_checkpoint", "uni2_revision", "uni2_package_version", "uni2_preprocessing_spec",
}
_SCF_IDENTITIES = {
    "scfoundation_checkpoint", "scfoundation_vocab",
    "scfoundation_package_version", "scfoundation_preprocessing_spec",
}


def required_gen6_fingerprints(config: dict) -> set[str]:
    spec = get_gen6_arm_spec((config.get("model") or {}).get("arm", ""))
    required: set[str] = set()
    if spec.uses_uni2_spot:
        required |= _UNI2_IDENTITIES
    if spec.uses_scfoundation:
        required |= _SCF_IDENTITIES
    if spec.uses_gigapath_dense:
        required.add("gigapath_checkpoint")
    if spec.uses_stpath:
        required |= {"stpath_checkpoint", "stpath_gene_vocab"}
    if spec.staged_conditioner:
        required.add("gen6_conditioner_checkpoint")
        selected_arm = str(((config.get("model") or {}).get("params") or {}).get("conditioner_arm", ""))
        selected = get_gen6_arm_spec(selected_arm)
        if selected.uses_uni2_spot:
            required |= _UNI2_IDENTITIES
        if selected.uses_scfoundation:
            required |= _SCF_IDENTITIES
        if selected.uses_gigapath_dense:
            required.add("gigapath_checkpoint")
        if selected.uses_stpath:
            required |= {"stpath_checkpoint", "stpath_gene_vocab"}
    if spec.staged_autoencoder:
        required.add("expression_autoencoder_checkpoint")
    return required


def static_audit_gen6_config(config: dict) -> dict:
    missing = _REQUIRED_TOP_LEVEL - set(config)
    if missing:
        raise ValueError(f"Gen6 config is missing top-level sections: {sorted(missing)}")
    model = config["model"]
    arm = str(model.get("arm", ""))
    spec = get_gen6_arm_spec(arm)
    kind = str(model.get("kind", ""))
    expected_kind = {
        "latent_ot_flow": "latent_flow", "wae_gan": "wae_gan",
    }.get(spec.generator, "conditioner")
    if kind != expected_kind:
        raise ValueError(f"{arm} requires model.kind={expected_kind!r}, got {kind!r}")
    if not (config.get("masking") or {}).get("strata"):
        raise ValueError("masking.strata must be non-empty")
    params = model.get("params") or {}
    selected_spec = spec
    if spec.staged_conditioner:
        selected_spec = get_gen6_arm_spec(str(params.get("conditioner_arm", "")))
        if selected_spec.staged_conditioner:
            raise ValueError("conditioner_arm must name one of gen6a-gen6j")
        if selected_spec.arm != "gen6c":
            raise ValueError("Gen6-K/L require model.params.conditioner_arm='gen6c'")
        for name in ("latent_dim", "n_flow_samples"):
            if int(params.get(name, 0)) <= 0:
                raise ValueError(f"{arm} requires positive model.params.{name}")
        if spec.generator == "latent_ot_flow":
            if int(params.get("n_ode_steps", 0)) <= 0:
                raise ValueError(f"{arm} requires positive model.params.n_ode_steps")
            if float(params.get("ot_epsilon", 0.0)) <= 0:
                raise ValueError(f"{arm} requires positive model.params.ot_epsilon")
            if int(params.get("ot_sinkhorn_iters", 0)) <= 0:
                raise ValueError(f"{arm} requires positive model.params.ot_sinkhorn_iters")
        if spec.generator == "wae_gan":
            for name in ("adversarial_weight", "discriminator_weight"):
                if float(params.get(name, -1.0)) < 0:
                    raise ValueError(f"{arm} requires non-negative model.params.{name}")
    if selected_spec.uses_scfoundation and int(params.get("gex_context_embedding_dim") or 0) <= 0:
        raise ValueError(f"{arm} requires positive model.params.gex_context_embedding_dim")
    required = required_gen6_fingerprints(config)
    fingerprints = config.get("required_fingerprints") or {}
    missing_keys = required - set(fingerprints)
    if missing_keys:
        raise ValueError(f"{arm} is missing required_fingerprints keys: {sorted(missing_keys)}")
    unset = sorted(name for name in required if not fingerprints.get(name))
    if selected_spec.uses_gigapath_spot:
        revision = str((config.get("data") or {}).get("tile_encoder_revision") or "")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            unset.append("data.tile_encoder_revision")
    return {
        "arm": arm, "kind": kind, "spec": spec.to_dict(),
        "checked_required_fingerprints": sorted(required),
        "unset_required_fingerprints": sorted(set(unset)),
        "ready_for_real_training": not unset,
    }


def audit_gen6_manifest_cache_coverage(
    cache_root, sample_ids: list[str], config: dict, *, require_staged_artifacts: bool = True,
) -> dict:
    if not sample_ids:
        raise ValueError("Gen6 cache preflight requires manifest-selected samples")
    requested_spec = get_gen6_arm_spec((config.get("model") or {}).get("arm", ""))
    spec = requested_spec
    if spec.staged_conditioner:
        spec = get_gen6_arm_spec(str(((config.get("model") or {}).get("params") or {}).get("conditioner_arm", "")))
    fingerprints = config.get("required_fingerprints") or {}
    if requested_spec.staged_conditioner and require_staged_artifacts:
        conditioner_path = Path(str(fingerprints.get("gen6_conditioner_checkpoint") or ""))
        if not conditioner_path.exists():
            raise FileNotFoundError(
                "required_fingerprints.gen6_conditioner_checkpoint does not exist: "
                f"{conditioner_path}"
            )
    if requested_spec.staged_autoencoder and require_staged_artifacts:
        autoencoder_path = Path(str(fingerprints.get("expression_autoencoder_checkpoint") or ""))
        if not autoencoder_path.is_file():
            raise FileNotFoundError(
                "required_fingerprints.expression_autoencoder_checkpoint does not exist: "
                f"{autoencoder_path}"
            )
    direct_artifacts = set()
    if spec.uses_gigapath_dense:
        direct_artifacts.add("gigapath_checkpoint")
    if spec.uses_stpath:
        direct_artifacts |= {"stpath_checkpoint", "stpath_gene_vocab"}
    for name in sorted(direct_artifacts):
        path = Path(str(fingerprints.get(name) or ""))
        if not path.is_file():
            raise FileNotFoundError(f"required_fingerprints.{name} does not exist: {path}")
    artifact_hashes = {}
    for name in ("uni2_checkpoint", "scfoundation_checkpoint", "scfoundation_vocab"):
        path = Path(str(fingerprints.get(name) or ""))
        if path.is_file():
            artifact_hashes[name] = _sha256_file(path)
    report = {"arm": spec.arm, "n_samples": len(sample_ids), "samples": {}}
    for sample_id in sample_ids:
        row = {}
        if spec.uses_uni2_spot:
            row["uni2"] = audit_uni2_cache_matches_config(
                cache_root, sample_id, config, require_exists=True,
                artifact_file_sha256=artifact_hashes,
            )
        if spec.uses_uni2_dense:
            row["uni2_dense"] = audit_uni2_dense_cache_matches_config(
                cache_root, sample_id, config, require_exists=True,
                artifact_file_sha256=artifact_hashes,
            )
        if spec.uses_scfoundation:
            row["scfoundation"] = audit_scfoundation_cache_matches_config(
                cache_root, sample_id, config, require_exists=True,
                artifact_file_sha256=artifact_hashes,
            )
        report["samples"][sample_id] = row
    return report
