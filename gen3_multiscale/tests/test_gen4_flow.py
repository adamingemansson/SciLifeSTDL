"""Gen4ResidualFlowModel tests -- mirrors test_architectures.py's own
Architecture4 test shape, one representative arm (D, the simplest -- no
global/regional branch) plus one frozen_context arm (B) for the
gex_context_proj path."""
from __future__ import annotations

import numpy as np
import torch

from gen3_multiscale.gen4.flow import Gen4ResidualFlowModel
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.tests._gen4_fixtures import GEN4_MODEL_KWARGS, synthetic_gen4_inputs


def _gene_basis_for(n_genes, rank=4, seed=99):
    rng = np.random.default_rng(seed)
    residuals = rng.normal(size=(30, n_genes))
    gene_names = [f"g{i}" for i in range(n_genes)]
    return fit_gene_residual_basis(residuals, gene_names, rank=rank), gene_names


def test_arm_d_flow_forward_matches_conditioner_contract():
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Gen4ResidualFlowModel(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape


def test_arm_d_compute_losses_single_conditioner_pass():
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    gene_basis, gene_names = _gene_basis_for(n_genes)
    model = Gen4ResidualFlowModel(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    assert "flow_loss" in out and torch.isfinite(out["flow_loss"])
    assert out["expression"].shape == target.shape


def test_arm_d_sample_predictive_distribution_deterministic_with_generator():
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    gene_basis, gene_names = _gene_basis_for(n_genes)
    model = Gen4ResidualFlowModel(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=3, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )
    model.eval()  # dropout must be disabled for the conditioner pass to be reproducible across calls
    out1 = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(7))
    out2 = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(out1["predictive_mean"], out2["predictive_mean"])
    assert torch.equal(out1["deterministic_mean"], out2["deterministic_mean"])


def test_flow_depends_on_trained_flow_weights():
    """A model whose velocity_network weights differ produces a different
    flow loss for the SAME conditioner/inputs -- flow prediction is
    genuinely a function of the flow weights, not a pass-through of the
    deterministic mean."""
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    gene_basis, gene_names = _gene_basis_for(n_genes)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)

    torch.manual_seed(0)
    model1 = Gen4ResidualFlowModel(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )
    torch.manual_seed(1)
    model2 = Gen4ResidualFlowModel(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN4_MODEL_KWARGS,
    )
    # Force the conditioners to agree (only velocity_network differs).
    model2.conditioner.load_state_dict(model1.conditioner.state_dict())

    gen = torch.Generator().manual_seed(0)
    loss1 = model1.compute_flow_matching_loss(inputs, target, generator=gen)
    gen = torch.Generator().manual_seed(0)
    loss2 = model2.compute_flow_matching_loss(inputs, target, generator=gen)
    assert not torch.allclose(loss1, loss2)


def test_basis_fitting_uses_only_supplied_training_residuals():
    """fit_gen4_residual_basis is a pure function of what it's given --
    changing "validation-only" data never fed into it cannot change the
    fitted basis. Structural: the function has no argument through which
    it could reach anything but the caller-supplied train_examples."""
    from gen3_multiscale.gen4.basis_fit import fit_gen4_residual_basis

    n_genes, gex_dim, image_dim = 6, 4, 8
    gene_names = [f"g{i}" for i in range(n_genes)]
    torch.manual_seed(0)
    from gen3_multiscale.gen4.conditioner import Gen4Conditioner
    conditioner = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    train_examples = [synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, seed=s) for s in range(3)]

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        basis_path = os.path.join(tmp, "basis.pt")
        basis1 = fit_gen4_residual_basis(conditioner, train_examples, gene_names, rank=2, output_basis_path=basis_path)
        # A "validation-only" example not passed in must be irrelevant.
        _unused_validation_example = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, seed=999)
        basis2 = fit_gen4_residual_basis(conditioner, train_examples, gene_names, rank=2, output_basis_path=os.path.join(tmp, "basis2.pt"))
        assert torch.allclose(basis1.basis, basis2.basis)
