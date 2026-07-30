"""Gen4Conditioner construction/forward tests -- all four arms' actual
config shape (GEN4_CONTRACT.md section 2), tiny CPU-fast dims."""
from __future__ import annotations

import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.tests._gen4_fixtures import GEN4_MODEL_KWARGS, synthetic_gen4_inputs, with_synthetic_wsi_context


def test_arm_a_uni2_pool_conditioner_forward():
    """gex_feature_source='weighted_linear', global_context_source='uni2_pool'."""
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    inputs = with_synthetic_wsi_context(inputs, image_dim)
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
    inputs = with_synthetic_wsi_context(inputs, image_dim)
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=image_dim, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gex_feature_source="frozen_context", gex_context_embedding_dim=context_dim,
        use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16,
        uni2_global_pool=pool, **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape


def test_arm_d_no_global_or_regional_branch_conditioner_forward():
    """Arm D: STPath's context representation already occupies the image
    slot (populated by gen4/stpath_example.py, not tested here) -- the
    conditioner itself just needs use_regional_he=False,
    global_context_source='none'."""
    inputs, targets = synthetic_gen4_inputs()
    n_genes, gex_dim, image_dim = 6, 4, 8
    model = Gen4Conditioner(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )
    out = model(inputs)
    assert out["expression"].shape == targets.query_expression.shape
    assert model.slide_encoder is None


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
