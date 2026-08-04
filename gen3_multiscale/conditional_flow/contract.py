"""Static scientific contract for the matched MK conditional-flow suite."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalFlowArmSpec:
    task: str
    coupling: str
    include_observed_gex: bool


ARM_SPECS = {
    "flow_he": ConditionalFlowArmSpec("he_to_st", "independent", False),
    "flow_he_ot": ConditionalFlowArmSpec("he_to_st", "sinkhorn_ot", False),
    "flow_he_st": ConditionalFlowArmSpec("he_plus_st_to_st", "independent", True),
    "flow_he_st_ot": ConditionalFlowArmSpec("he_plus_st_to_st", "sinkhorn_ot", True),
}


def static_audit_conditional_flow_config(config: dict) -> dict:
    model = config.get("model") or {}
    arm = str(model.get("arm", ""))
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown conditional-flow arm {arm!r}")
    spec = ARM_SPECS[arm]
    if model.get("kind") != "conditional_latent_flow":
        raise ValueError("model.kind must be 'conditional_latent_flow'")
    if model.get("task") != spec.task or model.get("coupling") != spec.coupling:
        raise ValueError(f"{arm}: task/coupling do not match the immutable arm contract")
    if bool(model.get("include_observed_gex")) != spec.include_observed_gex:
        raise ValueError(f"{arm}: include_observed_gex does not match the task contract")
    if model.get("image_mode") != "full_visible":
        raise ValueError("model.image_mode must be 'full_visible'; query H&E may not be masked")
    params = model.get("params") or {}
    for field in (
        "image_feature_dim", "latent_dim", "hidden_dim", "gex_feature_dim",
        "autoencoder_hidden_dim", "n_flow_blocks", "n_ode_steps",
        "n_inference_samples",
    ):
        if int(params.get(field, 0)) < 1:
            raise ValueError(f"model.params.{field} must be positive")
    if spec.coupling == "sinkhorn_ot":
        if float(params.get("ot_epsilon", 0)) <= 0:
            raise ValueError("model.params.ot_epsilon must be positive")
        if int(params.get("ot_sinkhorn_iters", 0)) < 1:
            raise ValueError("model.params.ot_sinkhorn_iters must be positive")
    data = config.get("data") or {}
    if not data.get("gen3_manifest_path"):
        raise ValueError("data.gen3_manifest_path must point to an immutable manifest")
    if not data.get("tile_encoder_revision"):
        raise ValueError("data.tile_encoder_revision must be pinned")
    if not (config.get("training") or {}).get("checkpoint_dir"):
        raise ValueError("training.checkpoint_dir is required")
    loss = config.get("loss") or {}
    for field in ("pcc_weight", "flow_weight", "conditional_mean_weight"):
        if float(loss.get(field, -1)) < 0:
            raise ValueError(f"loss.{field} must be non-negative")
    return {
        "passed": True,
        "arm": arm,
        "task": spec.task,
        "coupling": spec.coupling,
        "query_he_visible": True,
        "query_gex_visible": False,
        "surrounding_gex_visible": spec.include_observed_gex,
    }
