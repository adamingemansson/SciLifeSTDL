import torch

from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder, corrupt_expression
from gen2_architectures.models.arch3_stage_b_latent_transformer import Architecture3StageB
from gen2_architectures.models.components import StagedGeneLoss


def test_corrupt_expression_deterministic_given_seed():
    x = torch.randn(16, 200)
    a = corrupt_expression(x, seed=42, mask_fraction=0.2)
    b = corrupt_expression(x, seed=42, mask_fraction=0.2)
    assert torch.equal(a, b)


def test_corrupt_expression_different_seeds_differ():
    x = torch.randn(16, 200)
    a = corrupt_expression(x, seed=1, mask_fraction=0.2)
    b = corrupt_expression(x, seed=2, mask_fraction=0.2)
    assert not torch.equal(a, b)


def test_corrupt_expression_masks_approximately_the_requested_fraction():
    x = torch.ones(1000, 200)  # no real zeros, so zeroed count == masked count exactly
    corrupted = corrupt_expression(x, seed=0, mask_fraction=0.2)
    frac_zeroed = (corrupted == 0).float().mean().item()
    assert 0.15 < frac_zeroed < 0.25


def test_corrupt_expression_identity_when_no_corruption_requested():
    x = torch.randn(8, 50)
    assert torch.equal(corrupt_expression(x, seed=0, mask_fraction=0.0, gaussian_std=0.0), x)


def test_corrupt_expression_gaussian_noise_changes_values():
    x = torch.randn(8, 50)
    corrupted = corrupt_expression(x, seed=0, mask_fraction=0.0, gaussian_std=0.5)
    assert not torch.equal(corrupted, x)


def test_autoencoder_encode_decode_shapes_and_gradient():
    n_genes = 100
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    x = torch.randn(8, n_genes)
    latent = ae.encode(x)
    assert latent.shape == (8, 16)
    recon = ae.decode(latent)
    assert recon.shape == (8, n_genes)
    assert torch.equal(ae(x), recon)
    loss = torch.nn.functional.mse_loss(recon, x)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ae.parameters())


def _synthetic_context(n_context: int, n_genes: int) -> dict:
    return {
        "coords": torch.randn(n_context, 2) * 500,
        "expression": torch.randn(n_context, n_genes),
        "images": torch.randn(n_context, 1536),
        "image_available": torch.ones(n_context, dtype=torch.bool),
    }


def test_stage_b_forward_shapes():
    n_genes = 50
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    model = Architecture3StageB(ae, feat_dim=32, coord_dim=16, conf_dim=8, hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    out = model(context, query)
    assert out["predicted_latent"].shape == (5, 16)
    assert out["predicted_expression"].shape == (5, n_genes)


def test_stage_b_frozen_autoencoder_receives_no_gradient():
    n_genes = 50
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    model = Architecture3StageB(ae, finetune_autoencoder=False, feat_dim=32, coord_dim=16, conf_dim=8,
                                 hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    true_expr = torch.randn(5, n_genes)
    out = model(context, query)
    true_latent = model.true_latent(true_expr)
    loss_fn = StagedGeneLoss()
    latent_loss = torch.nn.functional.mse_loss(out["predicted_latent"], true_latent)
    gene_loss = loss_fn(out["predicted_expression"], true_expr, progress=0.9)
    (latent_loss + gene_loss["loss"]).backward()
    ae_grad = sum(1 for n, p in model.named_parameters() if "autoencoder" in n and p.grad is not None and p.grad.abs().sum() > 0)
    assert ae_grad == 0


def test_stage_b_finetune_reenables_gradient_even_on_a_previously_frozen_instance():
    """Real bug caught during development: finetune_autoencoder=True must
    actively RE-ENABLE grad, not merely skip disabling it -- a Stage-A
    checkpoint's requires_grad state is not part of state_dict and could
    arrive already frozen from a prior caller."""
    n_genes = 50
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    Architecture3StageB(ae, finetune_autoencoder=False, feat_dim=32, coord_dim=16, conf_dim=8,
                         hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20)  # freezes `ae` in place
    assert all(not p.requires_grad for p in ae.parameters())

    model2 = Architecture3StageB(ae, finetune_autoencoder=True, feat_dim=32, coord_dim=16, conf_dim=8,
                                  hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20)
    assert all(p.requires_grad for p in ae.parameters())
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    true_expr = torch.randn(5, n_genes)
    out = model2(context, query)
    loss_fn = StagedGeneLoss()
    gene_loss = loss_fn(out["predicted_expression"], true_expr, progress=0.9)
    gene_loss["loss"].backward()
    ae_params = [p for n, p in model2.named_parameters() if "autoencoder" in n]
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ae_params)


def test_stage_b_true_latent_has_no_gradient():
    n_genes = 50
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    model = Architecture3StageB(ae, finetune_autoencoder=True, feat_dim=32, coord_dim=16, conf_dim=8,
                                 hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20)
    true_expr = torch.randn(5, n_genes)
    true_latent = model.true_latent(true_expr)
    assert not true_latent.requires_grad
