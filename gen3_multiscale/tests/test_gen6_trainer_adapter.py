import torch

from gen3_multiscale.gen4.trainer_adapter import build_gen6_model_for_inference
from gen3_multiscale.gen6.generative import Gen6LatentOTFlowModel, Gen6WAEGANModel


def _config(arm, kind="conditioner", conditioner_arm=None):
    params = {
        "image_feature_dim": 8, "gex_context_embedding_dim": 5,
        "hidden_dim": 16, "n_heads": 4, "n_blocks": 1,
        "dense_threshold": 20, "fusion_heads": 4,
    }
    if conditioner_arm:
        params.update({"conditioner_arm": conditioner_arm,
                       "latent_dim": 4, "wae_hidden_dim": 16})
    return {
        "model": {"arm": arm, "kind": kind, "params": params},
        "data": {"gex_feature_dim": 4}, "training": {"seed": 0},
        "required_fingerprints": {},
    }


def test_shared_inference_adapter_constructs_gen6_conditioner():
    model, info = build_gen6_model_for_inference(
        _config("gen6c"), gene_names=[f"g{i}" for i in range(6)],
        device=torch.device("cpu"), smoke=True,
    )
    assert model.arm_spec.arm == "gen6c"
    assert info["kind"] == "conditioner"


def test_construction_only_smoke_builds_both_staged_generator_types():
    flow, flow_info = build_gen6_model_for_inference(
        _config("gen6k", "latent_flow", "gen6c"), gene_names=[f"g{i}" for i in range(6)],
        device=torch.device("cpu"), smoke=True,
    )
    wae, wae_info = build_gen6_model_for_inference(
        _config("gen6l", "wae_gan", "gen6c"), gene_names=[f"g{i}" for i in range(6)],
        device=torch.device("cpu"), smoke=True,
    )
    assert isinstance(flow, Gen6LatentOTFlowModel) and flow_info["kind"] == "latent_flow"
    assert isinstance(wae, Gen6WAEGANModel) and wae_info["kind"] == "wae_gan"


def _staged_config(arm, kind, **extra_params):
    config = _config(arm, kind, "gen6c")
    config["model"]["params"].update(extra_params)
    return config


def test_staged_builder_carries_the_new_generator_settings_into_the_model():
    """These four params are the entire difference between gen6o/gen6p and
    their controls. If the builder drops one, the arm trains as a duplicate of
    its control and the result reads as 'the change did nothing'."""
    flow, _ = build_gen6_model_for_inference(
        _staged_config("gen6p", "latent_flow", ot_assignment="hard"),
        gene_names=[f"g{i}" for i in range(6)], device=torch.device("cpu"), smoke=True,
    )
    wae, _ = build_gen6_model_for_inference(
        _staged_config(
            "gen6o", "wae_gan",
            conditional_mean_weight=1.0, latent_spatial_correlation=0.5,
        ),
        gene_names=[f"g{i}" for i in range(6)], device=torch.device("cpu"), smoke=True,
    )
    assert flow.ot_assignment == "hard"
    assert wae.conditional_mean_weight == 1.0
    assert wae.latent_spatial_correlation == 0.5


def test_staged_defaults_reproduce_the_arms_that_already_ran():
    """gen6k and gen6l predate these knobs. Their configs carry no value for
    conditional_mean_weight, so the default must leave the head without
    gradient rather than silently changing a trained arm's objective."""
    wae, _ = build_gen6_model_for_inference(
        _config("gen6l", "wae_gan", "gen6c"), gene_names=[f"g{i}" for i in range(6)],
        device=torch.device("cpu"), smoke=True,
    )
    assert wae.conditional_mean_weight == 0.0
    assert wae.latent_spatial_correlation == 0.0
