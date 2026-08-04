"""Static scientific contract for the supervisor conditional-WAE suite."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalWAEArmSpec:
    task: str
    regularizer: str
    include_observed_gex: bool


ARM_SPECS = {
    "wae_he_mmd": ConditionalWAEArmSpec("he_to_st", "mmd", False),
    "wae_he_gan": ConditionalWAEArmSpec("he_to_st", "gan", False),
    "wae_he_st_mmd": ConditionalWAEArmSpec("he_plus_st_to_st", "mmd", True),
    "wae_he_st_gan": ConditionalWAEArmSpec("he_plus_st_to_st", "gan", True),
}


def static_audit_conditional_wae_config(config: dict) -> dict:
    model = config.get("model") or {}
    arm = str(model.get("arm", ""))
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown conditional-WAE arm {arm!r}")
    spec = ARM_SPECS[arm]
    if model.get("kind") != "conditional_wae":
        raise ValueError("model.kind must be 'conditional_wae'")
    if model.get("task") != spec.task or model.get("regularizer") != spec.regularizer:
        raise ValueError(f"{arm}: task/regularizer do not match the immutable arm contract")
    if bool(model.get("include_observed_gex")) != spec.include_observed_gex:
        raise ValueError(f"{arm}: include_observed_gex does not match the task contract")
    if model.get("image_mode") != "full_visible":
        raise ValueError("model.image_mode must be 'full_visible'; query H&E may not be masked")
    params = model.get("params") or {}
    if int(params.get("image_feature_dim", 0)) < 1:
        raise ValueError("model.params.image_feature_dim must be positive")
    for field in ("latent_dim", "hidden_dim", "gex_feature_dim", "autoencoder_hidden_dim"):
        if int(params.get(field, 0)) < 1:
            raise ValueError(f"model.params.{field} must be positive")
    data = config.get("data") or {}
    if not data.get("gen3_manifest_path"):
        raise ValueError("data.gen3_manifest_path must point to an immutable manifest")
    if not data.get("tile_encoder_revision"):
        raise ValueError("data.tile_encoder_revision must be pinned")
    training = config.get("training") or {}
    if not training.get("checkpoint_dir"):
        raise ValueError("training.checkpoint_dir is required")
    loss = config.get("loss") or {}
    for field in ("pcc_weight", "regularizer_weight", "image_mean_weight"):
        if float(loss.get(field, -1)) < 0:
            raise ValueError(f"loss.{field} must be non-negative")
    return {
        "passed": True,
        "arm": arm,
        "task": spec.task,
        "regularizer": spec.regularizer,
        "query_he_visible": True,
        "query_gex_visible": False,
        "surrounding_gex_visible": spec.include_observed_gex,
    }
