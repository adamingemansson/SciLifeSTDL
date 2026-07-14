"""
Smoke test for VQ-VAE stage 1 (src/models/vqvae.py). Tiny synthetic data,
no real dataset or Lightning Trainer needed — calls forward()/
training_step() directly to check shapes/gradients/no-NaN and that the
codebook isn't collapsing to a single code on random data.

Run with: python -m tests.test_vqvae_stage1
"""
import torch

from src.models.vqvae import VQVAEStage1


def test_forward_shapes():
    torch.manual_seed(0)
    model = VQVAEStage1(n_genes=50, latent_dim=8, hidden_dim=32, codebook_size=16)
    expression = torch.rand(20, 50)

    x_hat, idx, vq_loss = model(expression)
    assert x_hat.shape == (20, 50), x_hat.shape
    assert idx.shape == (20,)
    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(vq_loss)
    print(f"[forward] OK — recon shape {tuple(x_hat.shape)}, "
          f"{torch.unique(idx).numel()}/{model.vq.codebook_size} codes used")


def test_training_step_updates_weights():
    torch.manual_seed(0)
    model = VQVAEStage1(n_genes=50, latent_dim=8, hidden_dim=32, codebook_size=16)
    opt = model.configure_optimizers()
    model.log_dict = lambda *args, **kwargs: None

    batch = {"expression": torch.rand(20, 50)}
    before_enc = model.encoder[0].weight.clone()
    before_codebook = model.vq.embed.clone()

    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    after_enc = model.encoder[0].weight
    after_codebook = model.vq.embed
    assert not torch.allclose(before_enc, after_enc), "encoder weights did not change"
    assert not torch.allclose(before_codebook, after_codebook), "codebook (EMA buffer) did not change"
    assert torch.isfinite(after_enc).all()
    assert torch.isfinite(after_codebook).all()
    print("[training_step] OK — encoder updated (gradient) + codebook updated (EMA), no NaNs")


def test_dead_code_reset_prevents_collapse():
    """Regression check for the real collapse seen on HEST-1k data
    (docs/model_schematics.md, 2026-07-14): repeatedly encode a batch that
    only ever hits a couple of codes and confirm the dead-code reset keeps
    other codes from going permanently unused."""
    torch.manual_seed(0)
    model = VQVAEStage1(n_genes=50, latent_dim=8, hidden_dim=32, codebook_size=16)
    expression = torch.rand(64, 50)
    for _ in range(50):
        model(expression)
    usage = model.vq.ema_cluster_size
    assert (usage >= model.vq.dead_code_threshold).all(), (
        f"codebook collapsed — some codes never revived: {usage.tolist()}"
    )
    print("[dead_code_reset] OK — no code stuck below the dead-code threshold")


if __name__ == "__main__":
    test_forward_shapes()
    test_training_step_updates_weights()
    test_dead_code_reset_prevents_collapse()
    print("\nAll VQ-VAE stage 1 smoke tests passed.")
