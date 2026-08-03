import pytest
import torch

from gen3_multiscale.gen6.conditioner import Gen6Conditioner
from gen3_multiscale.gen6.model_factory import build_gen6_model


def _config(arm):
    return {
        "model": {"arm": arm, "kind": "conditioner", "params": {
            "hidden_dim": 32, "n_heads": 4, "n_blocks": 1,
            "dense_threshold": 16, "gex_context_embedding_dim": 11,
            "image_feature_dim": 13, "fusion_heads": 4,
        }},
        "required_fingerprints": {},
    }


@pytest.mark.parametrize("arm", ["gen6b", "gen6c", "gen6f", "gen6g", "gen6h", "gen6i"])
def test_standalone_deterministic_arms_construct(arm):
    model = build_gen6_model(
        _config(arm), gene_names=[f"g{i}" for i in range(7)],
        gex_feature_dim=9, image_feature_dim=13,
        gex_context_embedding_dim=11, seed=4,
    )
    assert isinstance(model, Gen6Conditioner)
    assert model.arm_spec.arm == arm


def test_gigapath_arm_fails_closed_without_longnet():
    with pytest.raises(ValueError, match="GigaPath slide encoder"):
        build_gen6_model(
            _config("gen6d"), gene_names=["a", "b"],
            gex_feature_dim=9, image_feature_dim=13, seed=1,
        )


def test_generative_arms_cannot_silently_construct_as_conditioners():
    for arm in ("gen6k", "gen6l"):
        with pytest.raises(ValueError, match="staged generative builder"):
            build_gen6_model(
                _config(arm), gene_names=["a", "b"],
                gex_feature_dim=9, image_feature_dim=13, seed=1,
            )
