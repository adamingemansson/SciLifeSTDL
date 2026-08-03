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
