"""Gen5LatentFlowModel construction/forward tests + gate 3 (fixed-mask
overfit)."""
from __future__ import annotations

import torch

from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel
from gen3_multiscale.tests._gen5_fixtures import (
    GEN5_MODEL_KWARGS, Gen4STPathStub, synthetic_gen4_inputs, tiny_autoencoder, with_synthetic_wsi_context,
)

N_GENES, GEX_DIM, IMAGE_DIM, CONTEXT_DIM, LATENT_DIM = 6, 4, 8, 5, 8
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


def test_arm_a_uni2_pool_construction_and_forward():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    inputs = with_synthetic_wsi_context(inputs, IMAGE_DIM, wsi_tile_feature_provenance="uni2")
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=IMAGE_DIM, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16,
        uni2_global_pool=pool, n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["flow_loss"])
    sample_out = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(1))
    assert sample_out["expression"].shape == target.shape


def test_arm_b_frozen_context_gigapath_construction_and_forward():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM, gex_context_dim=CONTEXT_DIM)
    inputs = with_synthetic_wsi_context(inputs, IMAGE_DIM)

    class _StubSlideEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.checkpoint_sha256 = "deadbeef" * 8
            self.proj = torch.nn.Linear(IMAGE_DIM, 16)

        def forward(self, tile_features, tile_coords, cache_namespace):
            return self.proj(tile_features.mean(dim=0, keepdim=True)).squeeze(0)

    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, gex_feature_source="frozen_context", gex_context_embedding_dim=CONTEXT_DIM,
        use_regional_he=True, global_context_source="gigapath", global_slide_dim=16,
        slide_encoder=_StubSlideEncoder(), gigapath_checkpoint_sha256="deadbeef" * 8,
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["flow_loss"])


def test_arm_c_uni2_pool_frozen_context_construction_and_forward():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM, gex_context_dim=CONTEXT_DIM)
    inputs = with_synthetic_wsi_context(inputs, IMAGE_DIM, wsi_tile_feature_provenance="uni2")
    pool = MaskAwareCoordinateAttentionPool(tile_feature_dim=IMAGE_DIM, output_dim=16, hidden_dim=16, n_heads=2)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, gex_feature_source="frozen_context", gex_context_embedding_dim=CONTEXT_DIM,
        use_regional_he=True, global_context_source="uni2_pool", global_slide_dim=16, uni2_global_pool=pool,
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["flow_loss"])


def test_arm_d_stpath_context_construction_and_forward():
    """The REAL arm D: image_feature_source='stpath_context' +
    gex_feature_source='stpath_joint', mirroring Gen4's fixed arm D
    (GEN5_CONTRACT.md section 2's "matched conditioning systems"
    requirement). `compute_losses` detaches `query_hidden` from the
    conditioner (gen5/latent_flow.py -- the same "train flow against a
    frozen conditioner" discipline Gen4's own flow stage uses), so
    flow_loss.backward() correctly produces NO gradient on STPath's
    projection here; the real regression test for gradients actually
    reaching STPath's trainable projection lives at the CONDITIONER level
    (test_gen4_conditioner.py::test_arm_d_stpath_context_conditioner_forward_and_gradients,
    which Gen5 reuses unmodified via Gen4Conditioner). This test checks
    the flow model still constructs/forwards correctly with arm D's wiring
    and that stpath_encoder is a real registered submodule."""
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    stpath_stub = Gen4STPathStub(n_genes=N_GENES, hidden_dim=IMAGE_DIM)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, gex_feature_source="stpath_joint", image_feature_source="stpath_context",
        stpath_encoder=stpath_stub, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    assert model.conditioner.stpath_encoder is stpath_stub
    assert any(p is stpath_stub.proj.weight for p in model.parameters())
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["flow_loss"])
    out["flow_loss"].backward()
    assert stpath_stub.proj.weight.grad is None  # detached-conditioner discipline: no flow-stage gradient reaches it


def test_autoencoder_gene_mismatch_rejected():
    import pytest
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    with pytest.raises(ValueError, match="does not match the panel"):
        Gen5LatentFlowModel(
            n_genes=N_GENES, gene_names=[f"different{i}" for i in range(N_GENES)], gex_feature_dim=GEX_DIM,
            image_feature_dim=IMAGE_DIM, autoencoder=autoencoder, use_regional_he=False,
            global_context_source="none", n_flow_blocks=1, **GEN5_MODEL_KWARGS,
        )


def test_autoencoder_parameters_are_frozen_on_construction():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, **GEN5_MODEL_KWARGS,
    )
    assert not any(p.requires_grad for p in model.autoencoder.parameters())


def test_freeze_conditioner_disables_gradients():
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, **GEN5_MODEL_KWARGS,
    )
    assert any(p.requires_grad for p in model.conditioner.parameters())
    model.freeze_conditioner()
    assert not any(p.requires_grad for p in model.conditioner.parameters())


def test_latent_flow_overfits_a_fixed_mask():
    """Gate 3: many optimizer steps against one fixed mask should drive
    the flow loss down substantially (a real capacity check on the flow
    apparatus, analogous to Gen3/Gen4's own overfit gates)."""
    autoencoder = tiny_autoencoder(n_genes=N_GENES, latent_dim=LATENT_DIM)
    inputs, targets = synthetic_gen4_inputs(n_genes=N_GENES, gex_dim=GEX_DIM, image_dim=IMAGE_DIM)
    model = Gen5LatentFlowModel(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, use_regional_he=False, global_context_source="none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **GEN5_MODEL_KWARGS,
    )
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.velocity_network.parameters(), lr=5e-3)
    generator = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(400):
        out = model.compute_losses(inputs, target, generator=generator)
        optimizer.zero_grad()
        out["flow_loss"].backward()
        optimizer.step()
        losses.append(float(out["flow_loss"].item()))
    # Flow-matching loss is stochastic (fresh t/x0 draw every step), so
    # compare smoothed windows rather than single first/last points.
    early_mean = sum(losses[:20]) / 20
    late_mean = sum(losses[-20:]) / 20
    assert late_mean < early_mean * 0.6
