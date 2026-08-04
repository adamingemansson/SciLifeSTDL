import copy

import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_flow import (
    Architecture1ImageConditioner,
    ConditionalLatentFlow,
    FullImageExpressionInputs,
)
from gen3_multiscale.conditional_flow.contract import (
    ARM_SPECS,
    static_audit_conditional_flow_config,
)
from gen3_multiscale.conditional_wae.tensorboard import ConditionalFlowTensorBoardLogger


def _inputs(*, task_ii=False):
    rng = np.random.default_rng(0)
    n, n_genes = 12, 7
    query = np.zeros(n, dtype=bool) if task_ii else np.ones(n, dtype=bool)
    if task_ii:
        query[:5] = True
    expression = rng.normal(size=(n, n_genes)).astype(np.float32)
    observed = np.flatnonzero(~query)
    inputs = FullImageExpressionInputs(
        sample_id="slide",
        image_features=rng.normal(size=(n, 16)).astype(np.float32),
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        image_available=np.ones(n, dtype=bool),
        query_mask=query,
        observed_expression=expression[observed] if task_ii else None,
        observed_expression_indices=observed if task_ii else None,
        expression_available=~query if task_ii else None,
    )
    return inputs, torch.as_tensor(expression[query])


def _model(coupling):
    return ConditionalLatentFlow(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        coupling=coupling, latent_dim=5, autoencoder_hidden_dim=20,
        n_flow_blocks=1, n_flow_samples=3, n_ode_steps=2,
        ot_sinkhorn_iters=4,
    )


@pytest.mark.parametrize("coupling", ["independent", "sinkhorn_ot"])
@pytest.mark.parametrize("task_ii", [False, True])
def test_conditional_flow_one_step_and_prior_only_inference(coupling, task_ii):
    model = _model(coupling)
    inputs, target = _inputs(task_ii=task_ii)
    losses = model.compute_losses(
        inputs, target, generator=torch.Generator().manual_seed(1),
    )
    assert all(torch.isfinite(losses[key]) for key in (
        "total", "reconstruction_loss", "conditional_mean_loss", "flow_loss",
    ))
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.image_conditioner.parameters())
    assert any(parameter.grad is not None for parameter in model.expression_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.residual_decoder.parameters())
    assert any(parameter.grad is not None for parameter in model.velocity_network.parameters())
    prediction = model.sample_predictive_distribution(
        inputs, generator=torch.Generator().manual_seed(2),
    )
    assert prediction["predictive_mean"].shape == target.shape
    assert prediction["predictive_samples"].shape == (3, *target.shape)
    assert prediction["latent_samples"].shape == (3, target.shape[0], 5)
    assert not torch.equal(prediction["predictive_samples"][0], prediction["predictive_samples"][1])


def test_flow_objective_cannot_collapse_the_learned_target_encoder_directly():
    model = _model("independent")
    inputs, target = _inputs()
    losses = model.compute_losses(
        inputs, target, generator=torch.Generator().manual_seed(3),
    )
    losses["flow_loss"].backward()
    assert all(parameter.grad is None for parameter in model.expression_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.velocity_network.parameters())


def test_coupling_ablation_starts_from_identical_weights():
    torch.manual_seed(11)
    independent = _model("independent")
    torch.manual_seed(11)
    optimal_transport = _model("sinkhorn_ot")
    assert independent.state_dict().keys() == optimal_transport.state_dict().keys()
    for name, value in independent.state_dict().items():
        torch.testing.assert_close(value, optimal_transport.state_dict()[name])


def test_matched_arm_contract_is_complete_and_fail_closed():
    assert list(ARM_SPECS) == ["flow_he", "flow_he_ot", "flow_he_st", "flow_he_st_ot"]
    config = {
        "model": {
            "arm": "flow_he_st_ot", "kind": "conditional_latent_flow",
            "task": "he_plus_st_to_st", "coupling": "sinkhorn_ot",
            "include_observed_gex": True, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
                "n_flow_blocks": 1, "n_ode_steps": 2,
                "n_inference_samples": 3, "ot_epsilon": 0.1,
                "ot_sinkhorn_iters": 4,
            },
        },
        "data": {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"},
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "flow_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    report = static_audit_conditional_flow_config(config)
    assert report["query_gex_visible"] is False
    assert report["surrounding_gex_visible"] is True
    broken = copy.deepcopy(config)
    broken["model"]["include_observed_gex"] = False
    with pytest.raises(ValueError, match="task contract"):
        static_audit_conditional_flow_config(broken)


def test_flow_tensorboard_uses_flow_specific_tags(tmp_path):
    class Writer:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, tag, value, step):
            self.scalars[tag] = (value, step)

        def flush(self):
            pass

        def close(self):
            pass

    writer = Writer()
    logger = ConditionalFlowTensorBoardLogger(tmp_path, writer=writer)
    values = {key: torch.tensor(1.0) for key in (
        "total", "reconstruction_loss", "reconstruction_rmse",
        "reconstruction_pcc_loss", "conditional_mean_loss",
        "conditional_mean_rmse", "flow_loss",
    )}
    logger.add_train_scalars(10, values, grad_norm=torch.tensor(2.0), learning_rate=1e-4)
    assert "train/flow" in writer.scalars
    assert "train/prior" not in writer.scalars
    assert writer.scalars["train/flow"] == (1.0, 10)
