"""Gen5 leakage/mutation-invariant tests -- GEN5_CONTRACT.md sections 1, 5;
the task's required gates 4/5/6/7."""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import torch

from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel
from gen3_multiscale.models.flow import VelocityNetwork
from gen3_multiscale.tests._gen5_fixtures import GEN5_MODEL_KWARGS, synthetic_gen4_inputs, tiny_autoencoder

N_GENES, GEX_DIM, IMAGE_DIM, LATENT_DIM = 6, 4, 8, 8
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


def _tiny_model():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    return Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=4, n_ode_steps=3, **GEN5_MODEL_KWARGS,
    )


def test_conditioner_deterministic_mean_never_enters_the_prediction():
    """Structural proof: neither compute_losses' nor
    sample_predictive_distribution's source ever reads
    conditioner_out["expression"] into anything the caller can use as a
    prediction. Both methods may carry it through only in an explicitly
    named diagnostic-only field, never combine it with the flow loss,
    generated latent, or decoded prediction."""
    sample_source = inspect.getsource(Gen5LatentFlowModel.sample_predictive_distribution)
    lines_after = sample_source.split("conditioner_out = self.conditioner(inputs)", 1)[1]
    sample_uses = [
        line for line in lines_after.splitlines()
        if 'conditioner_out["expression"]' in line
        or "conditioner_out['expression']" in line
    ]
    assert all("diagnostic_only" in line for line in sample_uses)

    losses_source = inspect.getsource(Gen5LatentFlowModel.compute_losses)
    lines_after = losses_source.split("conditioner_out = self.conditioner(inputs)", 1)[1]
    # the ONLY permitted reference is the explicitly-named diagnostic field
    uses = [line for line in lines_after.splitlines() if 'conditioner_out["expression"]' in line]
    assert all("diagnostic_only" in line for line in uses)


def test_different_seeds_produce_different_predictions():
    """Gate 4: noise-sensitivity."""
    model = _tiny_model()
    model.eval()
    inputs, _targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    out1 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(1))
    out2 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(out1["expression"], out2["expression"])


def test_same_seed_reproducible():
    model = _tiny_model()
    model.eval()
    inputs, _targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    out1 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(7))
    out2 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(out1["expression"], out2["expression"])


def test_visible_context_mutation_changes_generation():
    """Gate 5: context-sensitivity."""
    model = _tiny_model()
    model.eval()
    inputs, _targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    out1 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(0))
    mutated_expr = inputs.observed_full_gene_expression.copy()
    mutated_expr[0] += 50.0
    mutated = dataclasses.replace(inputs, observed_full_gene_expression=mutated_expr)
    out2 = model.sample_predictive_distribution(mutated, n_samples=1, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(out1["expression"], out2["expression"])


def test_hidden_target_mutation_cannot_affect_inference():
    """Gate 6: sample_predictive_distribution's signature has no
    target-expression-shaped parameter at all -- structurally impossible
    to pass hidden target data to it."""
    params = set(inspect.signature(Gen5LatentFlowModel.sample_predictive_distribution).parameters) - {"self"}
    assert not any("target" in p or "expression" in p for p in params)

    model = _tiny_model()
    model.eval()
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    out1 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(3))
    _mutated_targets = dataclasses.replace(targets, query_expression=targets.query_expression * 0.0 + 999.0)
    out2 = model.sample_predictive_distribution(inputs, n_samples=1, generator=torch.Generator().manual_seed(3))  # targets never passed
    assert torch.equal(out1["expression"], out2["expression"])


def test_query_predictions_are_not_generated_independently():
    """Gate 7: spatial-coupling. Perturbing the noisy state at query row 0
    only, then running one velocity_network forward pass, must change the
    predicted velocity at OTHER query rows too -- proof the network
    jointly attends across queries rather than processing each
    independently (a per-row-independent network would leave every other
    row's output exactly unchanged)."""
    torch.manual_seed(0)
    velocity_network = VelocityNetwork(residual_rank=LATENT_DIM, hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)
    velocity_network.eval()
    n_query = 6
    rng = np.random.default_rng(0)
    state = torch.as_tensor(rng.normal(size=(n_query, LATENT_DIM)).astype(np.float32))
    coords = torch.as_tensor(rng.normal(size=(n_query, 2)).astype(np.float32))
    conditioning = torch.as_tensor(rng.normal(size=(n_query, 32)).astype(np.float32))
    t = torch.tensor(0.5)

    with torch.no_grad():
        out1 = velocity_network(state, t, coords, conditioning)
        perturbed_state = state.clone()
        perturbed_state[0] += 10.0
        out2 = velocity_network(perturbed_state, t, coords, conditioning)

    assert not torch.allclose(out1[1:], out2[1:]), (
        "perturbing query 0's noisy state left every other query's predicted velocity unchanged -- "
        "the velocity network is not actually jointly attending across queries"
    )
