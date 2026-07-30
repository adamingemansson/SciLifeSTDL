"""Gen4Conditioner construction/forward tests -- all four arms' actual
config shape (GEN4_CONTRACT.md section 2), tiny CPU-fast dims."""
from __future__ import annotations

import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.tests._gen4_fixtures import (
    GEN4_MODEL_KWARGS, Gen4STPathStub, synthetic_gen4_inputs, with_synthetic_uni2_features, with_synthetic_wsi_context,
)


def test_arm_a_uni2_pool_conditioner_forward():
    """gex_feature_source='weighted_linear', global_context_source='uni2_pool'."""
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    inputs = with_synthetic_wsi_context(inputs, image_dim, wsi_tile_feature_provenance="uni2")
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=image_dim, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16,
        uni2_global_pool=pool, **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert model.gex_feature_source == "weighted_linear"
    assert model.gex_context_proj is None
    # the trainable pool is a real registered submodule (appears in the optimizer's param set)
    assert any(p is model.slide_encoder.inducing_query for p in model.parameters())


def test_arm_b_frozen_context_gigapath_conditioner_forward():
    """gex_feature_source='frozen_context', global_context_source='gigapath'."""
    n_genes, gex_dim, image_dim, context_dim = 6, 4, 8, 5
    inputs, targets = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, gex_context_dim=context_dim)
    inputs = with_synthetic_wsi_context(inputs, image_dim)

    class _StubSlideEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.checkpoint_sha256 = "deadbeef" * 8
            self.proj = torch.nn.Linear(image_dim, 16)

        def forward(self, tile_features, tile_coords, cache_namespace):
            return self.proj(tile_features.mean(dim=0, keepdim=True)).squeeze(0)

    slide_encoder = _StubSlideEncoder()
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="frozen_context", gex_context_embedding_dim=context_dim,
        use_regional_he=True, global_context_source="gigapath", global_slide_dim=16,
        slide_encoder=slide_encoder, gigapath_checkpoint_sha256="deadbeef" * 8, **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    # gene_encoder is constructed (inherited unmodified) but inert for this arm
    assert not any(p.requires_grad for p in model.gene_encoder.parameters())
    assert model.gex_context_proj is not None
    assert any(p.requires_grad for p in model.gex_context_proj.parameters())


def test_arm_c_uni2_pool_frozen_context_conditioner_forward():
    n_genes, gex_dim, image_dim, context_dim = 6, 4, 8, 5
    inputs, targets = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, gex_context_dim=context_dim)
    inputs = with_synthetic_wsi_context(inputs, image_dim, wsi_tile_feature_provenance="uni2")
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=image_dim, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="frozen_context", gex_context_embedding_dim=context_dim,
        use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16,
        uni2_global_pool=pool, **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape


def test_arm_d_stpath_context_conditioner_forward_and_gradients():
    """Arm D: STPath's live context-only encoder replaces BOTH per-spot
    encoders -- image_feature_source='stpath_context' and
    gex_feature_source='stpath_joint' together, no separate
    WeightedGeneExpressionEncoder path, no precomputed numpy embedding
    (Codex audit finding #3, fixed). The stub's trainable `proj` layer
    must receive a real, nonzero gradient after backward() -- the
    concrete regression test for the previous no_grad()-at-data-build-time
    bug that left it a frozen random projection forever."""
    n_genes, gex_dim, image_dim = 6, 4, 8
    inputs, targets = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim)
    stpath_stub = Gen4STPathStub(n_genes=n_genes, hidden_dim=image_dim)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="stpath_joint", image_feature_source="stpath_context", stpath_encoder=stpath_stub,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert model.slide_encoder is None
    assert model.stpath_encoder is stpath_stub
    assert any(p is stpath_stub.proj.weight for p in model.parameters())

    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(out["expression"], target)
    loss.backward()
    assert stpath_stub.proj.weight.grad is not None
    assert torch.any(stpath_stub.proj.weight.grad != 0)


def test_arm4_hybrid_conditioner_forward_and_gradients():
    """Arm 4 (intended-design hybrid): STPath's joint context token fused
    with UNI2's own per-spot morphology token (image slot) and frozen
    scFoundation observed-GEX context (GEX slot) via small trainable
    fusion layers -- image_feature_source='hybrid_context' and
    gex_feature_source='hybrid_context' together. STPath's trainable
    proj layer AND the fusion layers must all receive real, nonzero
    gradients after backward()."""
    n_genes, gex_dim, image_dim, context_dim = 6, 4, 8, 5
    inputs, targets = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, gex_context_dim=context_dim)
    inputs = with_synthetic_uni2_features(inputs, image_dim)
    stpath_stub = Gen4STPathStub(n_genes=n_genes, hidden_dim=image_dim)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="hybrid_context", image_feature_source="hybrid_context",
        gex_context_embedding_dim=context_dim, stpath_encoder=stpath_stub,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert model.slide_encoder is None
    assert model.stpath_encoder is stpath_stub
    assert any(p is stpath_stub.proj.weight for p in model.parameters())

    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    loss = torch.nn.functional.mse_loss(out["expression"], target)
    loss.backward()
    assert stpath_stub.proj.weight.grad is not None
    assert torch.any(stpath_stub.proj.weight.grad != 0)
    for name, module in (("hybrid_scf_proj", model.hybrid_scf_proj), ("hybrid_image_fusion", model.hybrid_image_fusion), ("hybrid_gex_fusion", model.hybrid_gex_fusion)):
        for parameter in module.parameters():
            assert parameter.grad is not None, f"{name} received no gradient"


def test_arm4_hybrid_requires_matched_image_and_gex_sources():
    import pytest
    stpath_stub = Gen4STPathStub(n_genes=6, hidden_dim=8)
    with pytest.raises(ValueError, match="must be used"):
        Gen4Conditioner(
            n_genes=6, gex_feature_dim=4, image_feature_dim=8,
            image_feature_source="hybrid_context", stpath_encoder=stpath_stub,
            use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
        )


def test_arm4_hybrid_requires_observed_uni2_features():
    import pytest
    n_genes, gex_dim, image_dim, context_dim = 6, 4, 8, 5
    inputs, _targets = synthetic_gen4_inputs(n_genes=n_genes, gex_dim=gex_dim, image_dim=image_dim, gex_context_dim=context_dim)
    stpath_stub = Gen4STPathStub(n_genes=n_genes, hidden_dim=image_dim)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="hybrid_context", image_feature_source="hybrid_context",
        gex_context_embedding_dim=context_dim, stpath_encoder=stpath_stub,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    with pytest.raises(ValueError, match="observed_uni2_features"):
        model(inputs)  # observed_uni2_features left unset (None)


def test_uni2_pool_rejects_wsi_tile_features_without_uni2_provenance():
    """Codex audit finding #4: Gen3's existing dense-WSI tile cache is
    GigaPath-encoded, not UNI2-encoded -- no real UNI2 dense-WSI cache
    builder exists yet. Without `wsi_tile_feature_provenance == "uni2"`
    explicitly set, arm A/C must refuse to consume `wsi_tile_features`
    rather than silently treating GigaPath-shaped features as UNI2
    features."""
    import pytest
    inputs, _targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    inputs = with_synthetic_wsi_context(inputs, image_dim)  # provenance left unset (None) -- the real-data default
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=image_dim, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16,
        uni2_global_pool=pool, **GEN4_MODEL_KWARGS,
    )
    with pytest.raises(ValueError, match="wsi_tile_feature_provenance"):
        model(inputs)


def test_arm_d_requires_matched_image_and_gex_sources():
    import pytest
    stpath_stub = Gen4STPathStub(n_genes=6, hidden_dim=8)
    with pytest.raises(ValueError, match="must be used"):
        Gen4Conditioner(
            n_genes=6, gex_feature_dim=4, image_feature_dim=8,
            image_feature_source="stpath_context", stpath_encoder=stpath_stub,
            use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
        )


def test_frozen_context_requires_embedding_dim():
    import pytest
    with pytest.raises(ValueError, match="gex_context_embedding_dim"):
        Gen4Conditioner(
            n_genes=6, gex_feature_dim=4, image_feature_dim=8, gex_feature_source="frozen_context",
            **GEN4_MODEL_KWARGS,
        )


def test_frozen_context_forward_requires_the_input_field_set():
    import pytest
    inputs, _targets = synthetic_gen4_inputs()  # no gex_context_dim -> context_gex_embedding is None
    model = Gen4Conditioner(
        n_genes=6, gex_feature_dim=4, image_feature_dim=8, gex_feature_source="frozen_context",
        gex_context_embedding_dim=5, **GEN4_MODEL_KWARGS,
    )
    with pytest.raises(ValueError, match="context_gex_embedding"):
        model(inputs)


def test_unknown_arm_flags_raise():
    import pytest
    with pytest.raises(ValueError, match="gex_feature_source"):
        Gen4Conditioner(n_genes=6, gex_feature_dim=4, gex_feature_source="bogus", **GEN4_MODEL_KWARGS)
    with pytest.raises(ValueError, match="global_context_source"):
        Gen4Conditioner(n_genes=6, gex_feature_dim=4, global_context_source="bogus", **GEN4_MODEL_KWARGS)
