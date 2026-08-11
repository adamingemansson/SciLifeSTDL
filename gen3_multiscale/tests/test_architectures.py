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


_STUB_CHECKPOINT_SHA256 = "deadbeef" * 8


def _with_synthetic_wsi_context(inputs, image_dim, n_tiles=12, grid_bound=10.0, seed=1):
    """Attach a hand-built, in-bounds WSI context to an existing
    SpatialFieldInputs -- LongNet coords deliberately far outside
    [-1, 1] (real GigaPath target-MPP coordinates are large) while
    regional coords stay inside grid_bound, matching how
    example_builder.py produces two genuinely different-scale frames
    from the same tiles (17th Codex re-audit, Step 5 Part 2 launch
    blocker #1: the two frames must never be interchangeable)."""
    rng = np.random.default_rng(seed)
    longnet_coords = rng.uniform(10_000.0, 20_000.0, size=(n_tiles, 2)).astype(np.float32)
    regional_coords = rng.uniform(-grid_bound + 0.5, grid_bound - 0.5, size=(n_tiles, 2)).astype(np.float32)
    features = rng.normal(size=(n_tiles, image_dim)).astype(np.float32)
    return dataclasses.replace(
        inputs,
        wsi_tile_longnet_coords=longnet_coords,
        wsi_tile_regional_coords=regional_coords,
        wsi_tile_features=features,
        full_slide_coord_bounds=(-grid_bound, grid_bound, -grid_bound, grid_bound),
        slide_cache_namespace="unit-test-slide-abc123",
    )


class _StubSlideEncoder(torch.nn.Module):
    """Duck-typed stand-in for FrozenGigaPathSlideEncoder -- same
    forward(tile_features, tile_coords, cache_namespace) -> [output_dim]
    contract, no real checkpoint needed, matching every other pluggable-
    component test stub in this codebase. Exposes checkpoint_sha256
    (17th Codex re-audit, Step 5 Part 2, "Important before Step 6/7") so
    _SharedFieldArchitecture's real cross-verification against a
    caller-supplied gigapath_checkpoint_sha256 has something real to
    check -- defaults to the SAME value every test's
    gigapath_checkpoint_sha256= call sites use, so construction succeeds
    unless a test deliberately passes a mismatched value."""

    def __init__(self, tile_feature_dim: int, output_dim: int, checkpoint_sha256: str = _STUB_CHECKPOINT_SHA256):
        super().__init__()
        self.proj = torch.nn.Linear(tile_feature_dim, output_dim)
        self.checkpoint_sha256 = checkpoint_sha256
        self.calls = []

    def forward(self, tile_features, tile_coords, cache_namespace):
        self.calls.append(cache_namespace)
        return self.proj(tile_features.mean(dim=0))


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


def test_architecture_3_regional_he_without_wsi_context_raises_a_clear_error():
    """16th Codex re-audit (Step 5 Part 2): regional/global H&E is now
    genuinely wired -- use_regional_he=True on an example with no WSI
    context (wsi_tile_features is None, e.g. built without a
    slide_context) must fail loudly and specifically, not silently
    proceed with zero regional tokens or a confusing generic error."""
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    model = Architecture3(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, **_MODEL_KWARGS,
    )
    with pytest.raises(ValueError, match="use_regional_he=True requires wsi_tile_features"):
        model(inputs)


def test_architecture_3_use_global_slide_requires_a_real_slide_encoder_and_checksum_at_construction():
    """16th Codex re-audit's complete cache-key requirement (tile-cache
    content hash + visible-tile identity + GigaPath checkpoint SHA256 +
    model architecture/version): use_global_slide=True must fail
    CLOSED at construction time -- never silently default -- when either
    the real slide_encoder or its checkpoint SHA256 is missing."""
    n_genes, gex_dim, image_dim = 6, 4, 8
    with pytest.raises(ValueError, match="requires a real slide_encoder"):
        Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            use_global_slide=True, **_MODEL_KWARGS,
        )
    with pytest.raises(ValueError, match="requires a non-empty gigapath_checkpoint_sha256"):
        Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            use_global_slide=True, slide_encoder=_StubSlideEncoder(image_dim, 8),
            gigapath_checkpoint_sha256="   ", **_MODEL_KWARGS,
        )


def test_architecture_3_use_global_slide_rejects_a_slide_encoder_with_no_checkpoint_sha256_attribute():
    """17th Codex re-audit (Step 5 Part 2, "Important before Step 6/7"),
    CONFIRMED real: a slide_encoder that doesn't expose checkpoint_sha256
    at all (e.g. an incorrectly-typed object) must be rejected explicitly
    -- there is nothing to verify a caller's gigapath_checkpoint_sha256
    claim against otherwise."""
    n_genes, gex_dim, image_dim = 6, 4, 8

    class _NoChecksumEncoder(torch.nn.Module):
        def forward(self, tile_features, tile_coords, cache_namespace):
            return tile_features.mean(dim=0)

    with pytest.raises(ValueError, match="requires slide_encoder to expose checkpoint_sha256"):
        Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            use_global_slide=True, slide_encoder=_NoChecksumEncoder(),
            gigapath_checkpoint_sha256=_STUB_CHECKPOINT_SHA256, **_MODEL_KWARGS,
        )


def test_architecture_3_use_global_slide_rejects_a_gigapath_checkpoint_sha256_that_does_not_match_the_real_checkpoint():
    """17th Codex re-audit (Step 5 Part 2, "Important before Step 6/7"),
    CONFIRMED real: gigapath_checkpoint_sha256 used to be a caller-
    supplied string trusted blindly -- a caller could pass ANY unrelated
    string and it would silently poison the LongNet cache namespace with
    a false checkpoint identity. FrozenGigaPathSlideEncoder now exposes
    its own real checkpoint_sha256 (computed from the actual file bytes
    at construction); construction must fail closed when the caller's
    claim disagrees with it."""
    n_genes, gex_dim, image_dim = 6, 4, 8
    mismatched_encoder = _StubSlideEncoder(image_dim, 8, checkpoint_sha256="totally-unrelated-string")
    with pytest.raises(ValueError, match="does not match"):
        Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            use_global_slide=True, slide_encoder=mismatched_encoder,
            gigapath_checkpoint_sha256=_STUB_CHECKPOINT_SHA256, **_MODEL_KWARGS,
        )


def test_forward_device_selection_ignores_an_independently_relocated_slide_encoder():
    """17th Codex re-audit (Step 5 Part 2, "Important before Step 6/7"),
    CONFIRMED real: `next(self.parameters()).device` picks whatever
    parameter is registered FIRST -- self.slide_encoder is registered
    before self.gene_encoder/spot_token/backbone/transport_head, and the
    real FrozenGigaPathSlideEncoder.forward() independently moves ITS
    OWN frozen submodule onto CUDA lazily, per call. A CPU-resident
    learned model could then silently pick a stale/independently-moved
    device on the NEXT forward() call. Verified here without needing
    real CUDA: registers a stray module holding a `meta`-device
    parameter FIRST (reproducing the exact registration-order scenario,
    via a plain slide_encoder kwarg -- use_global_slide stays False, so
    _global_slide_vector is never actually called), and confirms
    forward() still resolves the real (cpu) device rather than picking
    up `meta` from naive next(self.parameters())."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    torch.manual_seed(0)
    stray_encoder = torch.nn.Module()
    stray_encoder.weight = torch.nn.Parameter(torch.zeros(2, device="meta"))
    model = Architecture1(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        slide_encoder=stray_encoder, **_MODEL_KWARGS,
    )
    # Sanity: confirms the premise -- naive next(model.parameters()) really
    # would pick up the stray encoder's meta-device parameter first.
    assert next(model.parameters()).device.type == "meta"

    out = model(inputs)  # must not crash, must resolve the real cpu device
    assert out["expression"].device.type == "cpu"
    assert torch.isfinite(out["expression"]).all()


def test_architecture_3_regional_he_and_global_slide_forward_end_to_end_with_synthetic_wsi_context():
    """Real end-to-end forward pass with both branches wired to genuine
    (synthetic) WSI data -- proves the whole chain (pool_regional_tokens
    -> regional_token_proj -> backbone cross-attention;
    FrozenGigaPathSlideEncoder-shaped stub -> backbone FiLM) produces a
    finite, correctly-shaped prediction, not just that it doesn't crash
    on missing data."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs(image_dim=8)
    inputs = _with_synthetic_wsi_context(inputs, image_dim)
    torch.manual_seed(0)
    global_slide_dim = 8
    slide_encoder = _StubSlideEncoder(image_dim, global_slide_dim)
    model = Architecture3(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, use_global_slide=True, global_slide_dim=global_slide_dim,
        slide_encoder=slide_encoder, gigapath_checkpoint_sha256="deadbeef" * 8,
        regional_grid_size=2, **_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert torch.isfinite(out["expression"]).all()
    assert len(slide_encoder.calls) == 1
    # The cache namespace passed to the slide encoder binds the data
    # layer's own identity (slide_cache_namespace) with the MODEL's
    # checkpoint SHA256 and architecture/version -- not just one or the
    # other (16th Codex re-audit's complete cache-key requirement).
    assert "unit-test-slide-abc123" in slide_encoder.calls[0]
    assert "deadbeef" in slide_encoder.calls[0]
    assert model.model_architecture_version in slide_encoder.calls[0]


def test_architecture_3_regional_he_uses_longnet_vs_regional_coordinate_frames_correctly():
    """16th/17th Codex re-audits (Step 5 Part 2): regional attention must
    use wsi_tile_regional_coords (the same centered/normalized frame as
    query_coords), never wsi_tile_longnet_coords (real, large-magnitude
    GigaPath LongNet target-MPP coordinates) -- feeding LongNet
    coordinates into compute_relative_geometry against normalized
    query_coords would produce huge, meaningless relative-geometry
    values. Verified directly: the LongNet stub call always receives the
    LongNet-frame coordinates (large magnitude), confirming the two
    frames are never swapped."""
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs(image_dim=8)
    inputs = _with_synthetic_wsi_context(inputs, image_dim)
    torch.manual_seed(0)
    slide_encoder = _StubSlideEncoder(image_dim, 8)
    captured_longnet_coords = {}
    real_forward = slide_encoder.forward

    def _patched(tile_features, tile_coords, cache_namespace):
        captured_longnet_coords["coords"] = tile_coords.clone()
        return real_forward(tile_features, tile_coords, cache_namespace)

    slide_encoder.forward = _patched
    model = Architecture3(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, use_global_slide=True, global_slide_dim=8,
        slide_encoder=slide_encoder, gigapath_checkpoint_sha256=_STUB_CHECKPOINT_SHA256,
        regional_grid_size=2, **_MODEL_KWARGS,
    )
    model(inputs)
    # wsi_tile_longnet_coords was sampled from [10_000, 20_000); regional
    # coords from roughly [-10, 10). If the frames were ever swapped, the
    # LongNet stub would see small-magnitude values instead.
    assert captured_longnet_coords["coords"].abs().min() > 1000.0


def test_architecture_3_regional_and_global_he_never_enter_the_gex_value_candidate_pool():
    """Direct structural proof of the 16th Codex re-audit's "regional/
    global H&E enters hidden conditioning only" requirement: the
    transport head's shared_candidate_expression/shared_candidate_hidden
    pools must have EXACTLY the same size whether or not
    use_regional_he/use_global_slide are enabled (with use_global_gex
    held fixed) -- proving neither branch ever appends a row to the real
    GEX value-candidate pool, only to the backbone's separate
    block_kwargs hidden-conditioning path."""
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs(image_dim=8)
    inputs_with_wsi = _with_synthetic_wsi_context(inputs, image_dim)

    def _captured_shared_expression_shape(**extra_kwargs):
        torch.manual_seed(0)
        model = Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
            **extra_kwargs, **_MODEL_KWARGS,
        )
        real_forward = model.transport_head.forward
        captured = {}

        def _patched(*args, **kwargs):
            captured.update(kwargs)
            return real_forward(*args, **kwargs)

        model.transport_head.forward = _patched
        example = inputs_with_wsi if extra_kwargs else inputs
        model(example)
        return captured["shared_candidate_expression"].shape, captured["shared_candidate_hidden"].shape

    baseline_expr_shape, baseline_hidden_shape = _captured_shared_expression_shape()
    wsi_expr_shape, wsi_hidden_shape = _captured_shared_expression_shape(
        use_regional_he=True, use_global_slide=True, global_slide_dim=8,
        slide_encoder=_StubSlideEncoder(image_dim, 8), gigapath_checkpoint_sha256=_STUB_CHECKPOINT_SHA256,
        regional_grid_size=2,
    )
    assert wsi_expr_shape == baseline_expr_shape
    assert wsi_hidden_shape == baseline_hidden_shape


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


def test_architecture_4_threads_regional_he_and_global_slide_into_its_conditioner():
    """16th Codex re-audit (Step 5 Part 2), CONFIRMED real gap: a prior
    version silently omitted use_regional_he/use_global_slide and every
    slide-encoder param when constructing self.conditioner, so
    Architecture3's own kwargs.setdefault(False) always won regardless
    of what Architecture4's caller asked for -- "Architecture 4 reuses
    Architecture 3's exact conditioner" was never actually true for
    these two flags. Verified directly: self.conditioner really has the
    flags set, AND a real forward pass with synthetic WSI context
    succeeds end to end through Architecture4's own forward()."""
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs(image_dim=8)
    inputs = _with_synthetic_wsi_context(inputs, image_dim)
    gene_basis, gene_names = _gene_basis_for(n_genes)
    torch.manual_seed(0)
    model = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
        use_regional_he=True, use_global_slide=True, global_slide_dim=8,
        slide_encoder=_StubSlideEncoder(image_dim, 8), gigapath_checkpoint_sha256=_STUB_CHECKPOINT_SHA256,
        regional_grid_size=2, **_MODEL_KWARGS,
    )
    assert model.conditioner.use_regional_he is True
    assert model.conditioner.use_global_slide is True
    assert model.conditioner.regional_grid_size == 2
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert torch.isfinite(out["expression"]).all()


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


def _refining(n_refinement_steps, n_genes, gex_dim, image_dim):
    torch.manual_seed(0)
    return Architecture1(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        n_refinement_steps=n_refinement_steps, refinement_k_neighbors=4,
        refinement_hidden_dim=16, refinement_gex_feature_dim=8, **_MODEL_KWARGS,
    )


def test_refinement_defaults_to_a_strict_no_op():
    """Every existing architecture inherits this path, so the default must
    construct no module and leave the forward output bit-identical."""
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    model = _refining(0, n_genes, gex_dim, image_dim)
    assert model.expression_refiner is None
    torch.manual_seed(0)
    baseline = Architecture1(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS,
    )
    model.eval(), baseline.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs)["expression"], baseline(inputs)["expression"])


def test_refinement_starts_at_the_identity_then_changes_the_prediction():
    """The refiner's update head is zero-initialised, so enabling it cannot
    degrade a prediction before any refinement gradient has been taken."""
    inputs, _targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    model = _refining(3, n_genes, gex_dim, image_dim)
    assert model.expression_refiner is not None
    torch.manual_seed(0)
    baseline = Architecture1(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, **_MODEL_KWARGS,
    )
    model.eval(), baseline.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs)["expression"], baseline(inputs)["expression"])
        for parameter in model.expression_refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
        refined = model(inputs)["expression"]
    assert not torch.allclose(refined, baseline(inputs)["expression"])
    assert refined.shape == baseline(inputs)["expression"].shape


def test_refinement_gradients_reach_the_refiner():
    inputs, targets, n_genes, gex_dim, image_dim = _synthetic_inputs()
    model = _refining(2, n_genes, gex_dim, image_dim)
    out = model(inputs)
    loss = torch.nn.functional.mse_loss(
        out["expression"], torch.as_tensor(targets.query_expression),
    )
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.expression_refiner.parameters()
    )
