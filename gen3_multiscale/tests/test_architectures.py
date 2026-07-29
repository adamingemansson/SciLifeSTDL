"""Phase 6 (multiscale spatial-field handoff): end-to-end integration
tests for Architecture 1/2/3, built on synthetic square-grid data (same
pattern as test_boundary_graph.py's Phase 2 tests) so the WHOLE pipeline
-- token projection, boundary extraction, backbone, transport head -- is
exercised together for the first time, not just each piece in isolation."""
import dataclasses

import numpy as np
import pytest
import torch

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example
from gen3_multiscale.models.architectures import Architecture1, Architecture2, Architecture3, Architecture4
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis


def _synthetic_inputs(n_genes=6, gex_dim=4, image_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    n = 15
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n), np.arange(lo, lo + n))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    dist = np.linalg.norm(grid, axis=1)
    observed_coords, query_coords = grid[dist > 2.5], grid[dist <= 2.5]
    n_observed, n_query = observed_coords.shape[0], query_coords.shape[0]

    result = extract_boundary_and_local_context(observed_coords, query_coords, k_neighbors=6, local_k=6, max_rings=3)

    inputs = SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed_coords.astype(np.float32),
        query_coords=query_coords.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, image_dim)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=result.query_local_neighbor_idx,
        boundary_idx=result.boundary_idx,
        boundary_ring=result.boundary_ring,
        query_depth_to_boundary=result.query_depth_to_boundary,
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    validate_spatial_field_example(inputs, targets)  # the pipeline must produce a well-formed example
    return inputs, targets, n_genes, gex_dim, image_dim


_MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)


def test_architecture_1_forward_shape_and_no_anchor():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert torch.isfinite(out["expression"]).all()
    assert out["anchor_expression"] is None
    assert model.transport_head.blend_logit is None


def test_architecture_2_forward_has_a_real_harmonic_anchor_and_stays_close_to_it_at_init():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture2(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    out = model(inputs)
    assert out["anchor_expression"] is not None
    assert out["anchor_expression"].shape == targets.query_expression.shape
    dist_to_anchor = (out["expression"] - out["anchor_expression"]).abs().mean()
    dist_to_candidate = (out["expression"] - out["candidate_expression"]).abs().mean()
    assert dist_to_anchor < dist_to_candidate  # blend_logit_init keeps the fresh model close to harmonic


def test_architecture_3_forward_shape_with_global_gex_pool():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture3(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert torch.isfinite(out["expression"]).all()
    assert out["anchor_expression"] is None  # still anchor-free
    assert model.gex_pool is not None


def test_architecture_3_raises_not_implemented_for_unwired_regional_he():
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture3(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, **_MODEL_KWARGS,
    )
    try:
        model(inputs)
        assert False, "expected a NotImplementedError"
    except NotImplementedError as exc:
        assert "regional H&E" in str(exc)


def test_architecture_3_uses_global_gex_pool_expression_in_the_transport_candidate_pool():
    """Regression test for a real, confirmed bug (Codex audit finding #6
    against commit 386bcf4): InducedGlobalGEXPool computes a genuine
    value-preserving candidate per inducing token, but an earlier version
    of _SharedFieldArchitecture.forward() only forwarded its `hidden`
    output into the backbone's attention and silently discarded
    `expression` -- the real GEX-mixture candidates were computed and
    then thrown away, never reaching the transport head. Monkeypatches
    the pool to return two DIFFERENT expression matrices for the SAME
    hidden/geometry and asserts the final prediction changes -- proving
    `expression` is genuinely part of the transport candidate pool now."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture3(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)

    real_forward = model.gex_pool.forward
    captured = {}

    def _patched(observed_hidden, observed_expression):
        out = dict(real_forward(observed_hidden, observed_expression))
        out["expression"] = captured["expression"]
        return out

    model.gex_pool.forward = _patched
    n_inducing = model.gex_pool.n_inducing

    captured["expression"] = torch.full((n_inducing, n_genes), 7.0)
    out_a = model(inputs)
    captured["expression"] = torch.full((n_inducing, n_genes), -7.0)
    out_b = model(inputs)

    assert not torch.allclose(out_a["expression"], out_b["expression"])


def test_architecture_3_shared_transport_candidates_are_never_broadcast_per_query():
    """Regression test for a real, confirmed OOM bug (Codex audit finding
    #5 against commit 386bcf4): boundary + global-GEX candidates must
    reach GeneValueTransportHead as SHARED [S, G] tensors, never
    broadcast to [Nq, S, G] -- for realistic query/boundary/gene counts
    the broadcast form could allocate tens of GB in one forward pass."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture3(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)

    real_forward = model.transport_head.forward
    captured = {}

    def _patched(*args, **kwargs):
        captured.update(kwargs)
        return real_forward(*args, **kwargs)

    model.transport_head.forward = _patched
    model(inputs)

    n_boundary = inputs.boundary_idx.shape[0]
    n_inducing = model.gex_pool.n_inducing
    n_shared = n_boundary + n_inducing
    assert captured["shared_candidate_expression"].shape == (n_shared, n_genes)  # NOT [Nq, n_shared, n_genes]
    assert captured["shared_candidate_hidden"].shape[0] == n_shared


def test_forward_output_tensors_live_on_the_models_own_device():
    """Regression test for a real, confirmed device bug (Codex audit
    finding #3 against commit 386bcf4): forward() used to build several
    tensors (modality flags, boundary-ring scatter, hole-geometry area
    proxy, the harmonic anchor) without an explicit device, which would
    silently stay on CPU even for a model moved to CUDA and crash or
    produce a device-mismatched result. Every intermediate now threads
    `next(self.parameters()).device` through explicitly -- verified here
    by checking the actual output device matches the model's own
    parameter device (this environment has no CUDA to move the model to,
    but the same code path is exercised regardless of which device it
    resolves to)."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture2(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    model_device = next(model.parameters()).device
    out = model(inputs)
    assert out["expression"].device == model_device
    assert out["anchor_expression"].device == model_device


def test_gene_encoder_is_genuinely_wired_into_the_model():
    """Regression test for a real, confirmed bug (2nd Codex re-audit of
    commit 547f51e): the trainable WeightedGeneExpressionEncoder module
    existed but nothing called it, and gene_encoder_type: weighted_linear
    in every config had no effect on the constructed model at all. Fixed:
    the encoder is now owned and called by the model itself, from
    observed_full_gene_expression -- the only gene array
    SpatialFieldInputs carries (a 3rd-round audit flagged and this
    project then removed the interim separate, unused conditioning
    field entirely -- see SpatialFieldInputs' own docstring). Verified:
    (1) gradients reach gene_encoder.projection.weight from a full
    forward+backward pass; (2) changing observed_full_gene_expression
    changes the output."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)

    out = model(inputs)
    out["expression"].sum().backward()
    assert model.gene_encoder.projection.weight.grad is not None
    assert torch.isfinite(model.gene_encoder.projection.weight.grad).all()
    assert not torch.allclose(
        model.gene_encoder.projection.weight.grad, torch.zeros_like(model.gene_encoder.projection.weight.grad),
    )

    perturbed = dataclasses.replace(
        inputs, observed_full_gene_expression=inputs.observed_full_gene_expression * 0.0 + 3.0,
    )
    with torch.no_grad():
        out_perturbed = model(perturbed)
    assert not torch.allclose(out["expression"], out_perturbed["expression"])


def test_gradients_flow_end_to_end_through_architecture_1():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    out = model(inputs)
    loss = torch.nn.functional.mse_loss(out["expression"], torch.as_tensor(targets.query_expression))
    loss.backward()

    checked = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite gradient in {name}"
            checked += 1
    assert checked > 10, "expected gradients across many parameters, got suspiciously few"
    # spot check a few specific modules that must receive gradient
    assert model.spot_token.gex_proj.weight.grad is not None
    assert model.transport_head.gene_head_logits.grad is not None
    assert model.backbone.blocks[0].query_self_attn.query_proj.weight.grad is not None


def test_architecture_1_and_2_share_identical_initialization_for_the_same_seed():
    """"Confirm common state-dict modules initialize identically across
    arms for the same seed" (Phase 6 item 4). Architecture 1 and 2 differ
    ONLY in use_anchor_blend, which adds one deterministically-filled
    (not randomly sampled) blend_logit parameter -- every OTHER parameter
    must be byte-identical given the same seed, since nothing about
    skipping a non-random tensor's creation can perturb the shared torch
    RNG stream."""
    torch.manual_seed(7)
    inputs, _, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(7)
    model1 = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    torch.manual_seed(7)
    model2 = Architecture2(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    shared_names = set(params1) & set(params2)
    assert len(shared_names) > 10
    for name in shared_names:
        assert torch.equal(params1[name], params2[name]), f"parameter {name} differs between Arch1 and Arch2"
    assert set(params2) - set(params1) == {"transport_head.blend_logit"}


def test_architecture_1_and_3_share_identical_token_projection_initialization():
    """A more modest, honest version of the same gate for Architecture 3:
    it constructs EXTRA randomly-initialized modules (gex_pool, the
    backbone's global-GEX cross-attention) that Architecture 1 doesn't
    have, which genuinely does shift the shared torch RNG stream for
    everything constructed afterward -- so full-model byte-identity isn't
    a meaningful claim here. What IS shared and must still match: the
    token projection modules (spot_token, query_token), constructed
    before any architecture-specific divergence."""
    torch.manual_seed(3)
    inputs, _, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(3)
    model1 = Architecture1(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)
    torch.manual_seed(3)
    model3 = Architecture3(n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS)

    for name, param1 in model1.spot_token.named_parameters():
        param3 = dict(model3.spot_token.named_parameters())[name]
        assert torch.equal(param1, param3), f"spot_token.{name} differs between Arch1 and Arch3"
    for name, param1 in model1.query_token.named_parameters():
        param3 = dict(model3.query_token.named_parameters())[name]
        assert torch.equal(param1, param3), f"query_token.{name} differs between Arch1 and Arch3"


def _gene_basis_for(n_genes, rank=4, seed=99):
    rng = np.random.default_rng(seed)
    residuals = rng.normal(size=(30, n_genes))
    gene_names = [f"g{i}" for i in range(n_genes)]
    return fit_gene_residual_basis(residuals, gene_names, rank=rank), gene_names


def test_architecture_4_forward_matches_the_conditioner_contract():
    """Architecture4's plain forward() runs ONLY the deterministic
    conditioner -- the same contract Architecture3 has, per the handoff's
    "Report Architecture 4's deterministic mean using the same path as
    Architecture 3."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert out["anchor_expression"] is None  # Architecture 4 remains anchor-free


def test_architecture_4_compute_flow_matching_loss_rejects_a_shape_mismatched_target():
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): "Both flow-loss methods should also move and
    validate target_expression against the model's actual device/dtype
    and check shape/finiteness." Before this fix, a wrong-shaped target
    would silently broadcast inside the residual subtraction instead of
    failing at the point of the actual mistake."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    wrong_shape_target = torch.as_tensor(targets.query_expression)[:, :-1]  # drop one gene column
    with pytest.raises(ValueError, match="must match"):
        model.compute_flow_matching_loss(inputs, wrong_shape_target)


def test_architecture_4_compute_flow_matching_loss_rejects_a_non_finite_target():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    bad_target = torch.as_tensor(targets.query_expression).clone()
    bad_target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model.compute_flow_matching_loss(inputs, bad_target)


def test_architecture_4_compute_flow_matching_loss_accepts_a_raw_numpy_target():
    """Regression test for a real, confirmed gap (7th Codex re-audit of
    commit 2782ff0): SpatialFieldTargets.query_expression -- the
    natural, real source of this argument -- is typed and documented as
    a plain np.ndarray throughout data/example.py. The previous
    `target_expression.to(...)` call would raise AttributeError on a
    genuine numpy array (numpy arrays have no `.to()` method); a real
    trainer passing targets.query_expression directly, exactly as the
    schema documents, would have crashed here."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    assert isinstance(targets.query_expression, np.ndarray)
    loss = model.compute_flow_matching_loss(inputs, targets.query_expression)
    assert torch.isfinite(loss)


def test_architecture_4_compute_flow_matching_loss_accepts_a_target_on_a_different_dtype():
    """A target passed as float64 (a common default from raw numpy/anndata
    conversion) must be moved onto the model's own dtype, not rejected or
    silently mismatched deep inside an einsum."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    float64_target = torch.as_tensor(targets.query_expression, dtype=torch.float64)
    loss = model.compute_flow_matching_loss(inputs, float64_target)
    assert torch.isfinite(loss)


def test_architecture_4_flow_loss_gradients_reach_only_the_velocity_network():
    """"Initially stop gradients from the flow loss into the deterministic
    conditioner" -- verified directly: backpropagating ONLY the flow loss
    must leave every conditioner parameter's .grad as None."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    loss = model.compute_flow_matching_loss(inputs, torch.as_tensor(targets.query_expression))
    assert torch.isfinite(loss)
    loss.backward()

    for name, param in model.conditioner.named_parameters():
        assert param.grad is None, f"conditioner.{name} received a gradient from the flow loss"
    velocity_grads = [p.grad for p in model.velocity_network.parameters() if p.grad is not None]
    assert len(velocity_grads) > 0
    assert all(torch.isfinite(g).all() for g in velocity_grads)


def test_architecture_4_predictive_distribution_shapes_and_diversity():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, n_flow_samples=5, n_ode_steps=4, **_MODEL_KWARGS,
    )
    n_query = targets.query_expression.shape[0]
    out = model.sample_predictive_distribution(inputs)
    assert out["predictive_samples"].shape == (5, n_query, n_genes)
    assert out["predictive_mean"].shape == (n_query, n_genes)
    assert out["predictive_std"].shape == (n_query, n_genes)
    assert out["expression"] is out["predictive_mean"]
    assert torch.isfinite(out["predictive_mean"]).all()
    assert (out["predictive_std"] > 0).any()  # samples must actually differ somewhere
    assert not torch.allclose(out["predictive_samples"][0], out["predictive_samples"][1])


def test_architecture_4_predictive_std_is_finite_zero_not_nan_for_a_single_sample():
    """Regression test for a real, confirmed bug (Codex audit finding
    against commit c02a5d1): torch.Tensor.std()'s default unbiased=True
    divides by (n_samples - 1), which is zero degrees of freedom for a
    single sample and returns all-NaN (confirmed directly via
    torch.randn(1, 5).std(dim=0)). A single draw has a well-defined
    population std of exactly 0, so predictive_std must be finite and
    all-zero when n_samples=1, never NaN."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, n_flow_samples=1, n_ode_steps=2, **_MODEL_KWARGS,
    )
    out = model.sample_predictive_distribution(inputs, n_samples=1)
    assert out["predictive_samples"].shape[0] == 1
    assert torch.isfinite(out["predictive_std"]).all()
    assert torch.allclose(out["predictive_std"], torch.zeros_like(out["predictive_std"]))


def test_architecture_4_compute_losses_runs_the_conditioner_exactly_once_and_stays_consistent():
    """Regression test for a real, confirmed bug (Codex audit finding
    against commit c02a5d1): calling forward() and
    compute_flow_matching_loss() separately in one training step invokes
    self.conditioner(inputs) twice; with dropout active, the two calls
    draw different dropout masks, so the flow loss's deterministic_mean
    would silently disagree with the mean the reconstruction loss was
    computed against. compute_losses() must derive both from a single
    conditioner pass, and (with dropout disabled via eval() so the
    comparison is meaningful) must produce the same deterministic
    "expression" that forward() alone would, and the same flow loss
    compute_flow_matching_loss() alone would (since eval-mode dropout is
    a no-op, both call patterns become deterministic and comparable)."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    model.eval()
    target_expression = torch.as_tensor(targets.query_expression)

    torch.manual_seed(1)
    combined = model.compute_losses(inputs, target_expression)
    assert "flow_loss" in combined
    assert torch.isfinite(combined["flow_loss"])
    assert combined["expression"].shape == target_expression.shape

    torch.manual_seed(1)
    forward_out = model(inputs)
    assert torch.allclose(combined["expression"], forward_out["expression"])

    torch.manual_seed(1)
    separate_flow_loss = model.compute_flow_matching_loss(inputs, target_expression)
    assert torch.allclose(combined["flow_loss"], separate_flow_loss)


def test_architecture_4_registers_the_gene_basis_as_a_buffer_not_a_plain_tensor():
    """Regression test for a real, confirmed device bug (Codex audit
    finding #3 against commit 386bcf4): GeneResidualBasis is a plain
    frozen dataclass, not an nn.Module, so gene_basis.basis alone would
    NOT move when this Architecture4 instance is sent to a CUDA device --
    every call to compute_flow_matching_loss/sample_predictive_distribution
    would then silently mix a CPU-resident basis with CUDA-resident
    activations. Verified structurally: the basis matrix must appear in
    named_buffers(), matching gene_basis.basis exactly at construction."""
    n_genes = 6
    gene_basis, gene_names = _gene_basis_for(n_genes)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=4, image_feature_dim=8,
        gene_basis=gene_basis, gene_names=gene_names, **_MODEL_KWARGS,
    )
    buffers = dict(model.named_buffers())
    assert "_gene_basis_matrix" in buffers
    assert torch.allclose(buffers["_gene_basis_matrix"], gene_basis.basis)


def test_architecture_4_flow_and_sampling_output_tensors_live_on_the_models_own_device():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names, n_flow_samples=2, n_ode_steps=2, **_MODEL_KWARGS,
    )
    model_device = next(model.parameters()).device
    loss = model.compute_flow_matching_loss(inputs, torch.as_tensor(targets.query_expression))
    assert loss.device == model_device
    out = model.sample_predictive_distribution(inputs)
    assert out["predictive_mean"].device == model_device


def test_architecture_4_rejects_a_gene_basis_fit_on_a_different_panel():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    wrong_basis, _ = _gene_basis_for(n_genes)
    wrong_gene_names = [f"different_gene_{i}" for i in range(n_genes)]
    try:
        Architecture4(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            gene_basis=wrong_basis, gene_names=wrong_gene_names, **_MODEL_KWARGS,
        )
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "gene panel" in str(exc)
