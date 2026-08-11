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
    for arm in ("gen6k", "gen6l", "gen6o", "gen6p"):
        with pytest.raises(ValueError, match="staged generative builder"):
            build_gen6_model(
                _config(arm), gene_names=["a", "b"],
                gex_feature_dim=9, image_feature_dim=13, seed=1,
            )


def _refining_config(arm, steps=2, k=3):
    config = _config(arm)
    config["model"]["params"].update({
        "n_refinement_steps": steps, "refinement_k_neighbors": k,
        "refinement_hidden_dim": 16, "refinement_gex_feature_dim": 8,
    })
    return config


@pytest.mark.parametrize("arm", ["gen6m", "gen6n"])
def test_refinement_arms_construct_with_a_refiner(arm):
    model = build_gen6_model(
        _refining_config(arm), gene_names=[f"g{i}" for i in range(7)],
        gex_feature_dim=9, image_feature_dim=13,
        gex_context_embedding_dim=11, seed=4,
    )
    assert isinstance(model, Gen6Conditioner)
    assert model.expression_refiner is not None
    assert model.n_refinement_steps >= 1


@pytest.mark.parametrize("arm", ["gen6m", "gen6n"])
def test_a_refinement_arm_without_steps_is_refused(arm):
    """Silently building gen6m as a plain gen6c would produce a full 8-hour
    run whose name says 'refinement' and whose weights contain none."""
    with pytest.raises(ValueError, match="n_refinement_steps"):
        build_gen6_model(
            _config(arm), gene_names=[f"g{i}" for i in range(7)],
            gex_feature_dim=9, image_feature_dim=13,
            gex_context_embedding_dim=11, seed=4,
        )


def test_a_control_arm_cannot_be_turned_into_a_refinement_arm_by_a_stray_param():
    with pytest.raises(ValueError, match="must not set"):
        build_gen6_model(
            _refining_config("gen6c"), gene_names=[f"g{i}" for i in range(7)],
            gex_feature_dim=9, image_feature_dim=13,
            gex_context_embedding_dim=11, seed=4,
        )


def test_a_refinement_arm_shares_every_non_refiner_weight_with_its_control():
    """gen6m is gen6c plus a refiner and nothing else. The refiner is built
    under a forked RNG precisely so this holds; without it, every module
    constructed afterwards (query geometry, for one) would silently draw from
    a shifted stream and the comparison would confound refinement with a
    different random initialisation."""
    gene_names = [f"g{i}" for i in range(7)]
    kwargs = dict(
        gene_names=gene_names, gex_feature_dim=9, image_feature_dim=13,
        gex_context_embedding_dim=11, seed=4,
    )
    refining = build_gen6_model(_refining_config("gen6m"), **kwargs)
    control = build_gen6_model(_config("gen6c"), **kwargs)
    control_parameters = dict(control.named_parameters())
    shared = {
        name: parameter for name, parameter in refining.named_parameters()
        if not name.startswith("expression_refiner.")
    }
    assert set(shared) == set(control_parameters)
    assert shared, "expected the arms to share weights"
    for name, parameter in shared.items():
        torch.testing.assert_close(
            parameter, control_parameters[name], msg=lambda m, n=name: f"{n}: {m}",
        )
