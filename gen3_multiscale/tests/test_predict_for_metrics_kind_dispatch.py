"""Integration audit finding #1 (CONFIRMED real, fixed): `predict_for_
metrics`/`compute_step_losses`/`compute_deterministic_reconstruction_
losses` used to dispatch on Gen3's numeric `architecture_id == "4"`. A
Gen4 flow config has no `model.architecture` at all, so the prior check
silently fell through to the `model(inputs)` branch -- which for
`Gen4ResidualFlowModel` returns the FROZEN CONDITIONER's own output (by
design, `forward()` just is `self.conditioner(inputs)`), never the
flow model's real predictive-mean prediction. `Gen5LatentFlowModel` has
no `forward()` at all, so the same fallback would have raised
`NotImplementedError` outright.

These tests exercise the FIXED, `model.kind`-dispatched functions
directly against real `Gen4ResidualFlowModel`/`Gen5LatentFlowModel`
instances (never a Gen3 architecture) and prove: (1) the returned
"expression" genuinely comes from `sample_predictive_distribution`, not
the conditioner pass -- demonstrated by showing it is SENSITIVE to the
velocity network's own weights while the conditioner-only prediction is
not; (2) `compute_step_losses` computes a real, finite `flow_loss` via
`model.compute_losses` for both `"flow"` and `"latent_flow"`; (3) none
of this ever calls `model(inputs)` on a `Gen5LatentFlowModel` (which
would raise `NotImplementedError`)."""
from __future__ import annotations

import torch

from gen3_multiscale.gen4.flow import Gen4ResidualFlowModel
from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.tests._gen4_fixtures import GEN4_MODEL_KWARGS, synthetic_gen4_inputs
from gen3_multiscale.tests._gen5_fixtures import GEN5_MODEL_KWARGS, tiny_autoencoder
from gen3_multiscale.training.train import (
    compute_deterministic_reconstruction_losses, compute_step_losses, model_kind_for_architecture_id,
    predict_for_metrics,
)

N_GENES, GEX_DIM, IMAGE_DIM = 6, 4, 8


def _gene_basis_for(n_genes, rank=4, seed=99):
    import numpy as np

    rng = np.random.default_rng(seed)
    residuals = rng.normal(size=(30, n_genes))
    gene_names = [f"g{i}" for i in range(n_genes)]
    return fit_gene_residual_basis(residuals, gene_names, rank=rank), gene_names


def test_model_kind_for_architecture_id_maps_gen3_architecture_four_to_flow():
    assert model_kind_for_architecture_id("4") == "flow"
    assert model_kind_for_architecture_id(4) == "flow"
    for other in ("1", "2", "3", ""):
        assert model_kind_for_architecture_id(other) == "conditioner"


def _gen4_flow_model(seed: int, gene_basis, gene_names) -> Gen4ResidualFlowModel:
    torch.manual_seed(seed)
    return Gen4ResidualFlowModel(
        n_genes=N_GENES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=4, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )


def test_predict_for_metrics_flow_expression_is_the_predictive_mean_not_the_conditioner():
    """The core regression proof: two Gen4ResidualFlowModel instances
    with an IDENTICAL conditioner but DIFFERENT (differently-seeded)
    velocity networks must produce the SAME conditioner-only expression
    but DIFFERENT `predict_for_metrics(kind="flow", ...)["expression"]`
    -- the prior, buggy fallback (`model(inputs)`, i.e. the conditioner
    pass) would have returned IDENTICAL "expression" for both models,
    since it never touches the velocity network at all."""
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    gene_basis, gene_names = _gene_basis_for(N_GENES)

    model1 = _gen4_flow_model(0, gene_basis, gene_names)
    model2 = _gen4_flow_model(1, gene_basis, gene_names)
    model2.conditioner.load_state_dict(model1.conditioner.state_dict())
    model1.eval()
    model2.eval()

    conditioner_pred1 = predict_for_metrics("flow", model1, inputs, generator=torch.Generator().manual_seed(7))
    conditioner_pred2 = predict_for_metrics("flow", model2, inputs, generator=torch.Generator().manual_seed(7))

    # Same frozen conditioner -> identical conditioner-only diagnostic.
    assert torch.allclose(
        conditioner_pred1["conditioner_only_expression"], conditioner_pred2["conditioner_only_expression"],
    )
    # Different velocity networks -> the REAL, reported "expression" (the
    # predictive mean) must differ -- proving it is a genuine function of
    # the flow weights, not a silent conditioner pass-through.
    assert not torch.allclose(conditioner_pred1["expression"], conditioner_pred2["expression"])
    # And the reported "expression" must NOT equal the conditioner-only
    # value either -- confirming this is not merely re-reporting forward().
    assert not torch.allclose(conditioner_pred1["expression"], conditioner_pred1["conditioner_only_expression"])


def test_predict_for_metrics_latent_flow_never_calls_forward_and_uses_predictive_mean():
    """Gen5LatentFlowModel has NO forward() at all -- calling model(inputs)
    directly raises NotImplementedError. predict_for_metrics(kind=
    "latent_flow", ...) must never do that; it must go through
    model.conditioner(inputs) + model.sample_predictive_distribution
    exactly like the "flow" branch."""
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=4)
    gene_names = [f"g{i}" for i in range(N_GENES)]
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)

    torch.manual_seed(0)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=gene_names, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=4, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    model.eval()

    # Confirm the premise directly: model(inputs) is not implemented.
    import pytest

    with pytest.raises(NotImplementedError):
        model(inputs)

    prediction = predict_for_metrics("latent_flow", model, inputs, generator=torch.Generator().manual_seed(3))
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    assert prediction["expression"].shape == target.shape
    assert "conditioner_only_expression" in prediction
    assert "predictive_std" in prediction and "predictive_samples" in prediction


def test_compute_step_losses_flow_and_latent_flow_produce_real_finite_flow_loss():
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
    gene_basis, gene_names = _gene_basis_for(N_GENES)

    gen4_model = _gen4_flow_model(0, gene_basis, gene_names)
    gen4_losses = compute_step_losses(
        "flow", gen4_model, inputs, target, query_coords, gradient_weight=0.05, k_neighbors=6,
    )
    assert torch.isfinite(gen4_losses["total"]) and torch.isfinite(gen4_losses["flow_loss"])

    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=4)
    torch.manual_seed(0)
    gen5_model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=gene_names, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=4, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    gen5_losses = compute_step_losses(
        "latent_flow", gen5_model, inputs, target, query_coords, gradient_weight=0.05, k_neighbors=6,
    )
    assert torch.isfinite(gen5_losses["total"]) and torch.isfinite(gen5_losses["flow_loss"])


def test_compute_step_losses_conditioner_kind_never_touches_flow_apparatus():
    """A "conditioner" kind must use plain model(inputs) -- proven here
    with a Gen4Conditioner directly (no velocity network exists at all)."""
    from gen3_multiscale.gen4.conditioner import Gen4Conditioner

    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
    torch.manual_seed(0)
    model = Gen4Conditioner(
        n_genes=N_GENES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    losses = compute_step_losses("conditioner", model, inputs, target, query_coords, gradient_weight=0.05, k_neighbors=6)
    assert "flow_loss" not in losses
    assert torch.isfinite(losses["total"])


def test_compute_deterministic_reconstruction_losses_dispatches_flow_kind_through_predictive_mean():
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
    gene_basis, gene_names = _gene_basis_for(N_GENES)
    model = _gen4_flow_model(0, gene_basis, gene_names)
    model.eval()
    result = compute_deterministic_reconstruction_losses(
        "flow", model, inputs, target, query_coords, gradient_weight=0.05, k_neighbors=6,
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.isfinite(result["total"])
    assert "conditioner_only_total" in result  # secondary diagnostic present for flow kinds
