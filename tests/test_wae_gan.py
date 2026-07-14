"""
Smoke test for WAE-GAN with conditioning wired in (src/models/registry.py).
Tiny synthetic data, no real dataset or Lightning Trainer needed — calls
sample() and training_step() directly to check shapes/gradients/no-NaN.

Run with: python -m tests.test_wae_gan
"""
import torch

from src.models.registry import WAEGAN


def _make_batch(n_context, n_query, n_genes, coord_dim):
    context = {
        "coords": torch.randn(n_context, coord_dim),
        "expression": torch.rand(n_context, n_genes),
    }
    query = {"coords": torch.randn(n_query, coord_dim)}
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


def test_sample():
    torch.manual_seed(0)
    model = WAEGAN(n_genes=50, coord_dim=3, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16)
    batch = _make_batch(n_context=100, n_query=20, n_genes=50, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (20, 50), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()
    print(f"[sample] OK — output shape {tuple(out['expression'].shape)}")


def test_training_step_updates_weights():
    torch.manual_seed(0)
    model = WAEGAN(n_genes=50, coord_dim=2, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16)
    opt_ae, opt_disc = model.configure_optimizers()
    # bypass Lightning Trainer wiring for this isolated smoke test — optimizers(),
    # manual_backward(), and log_dict() all normally require an attached Trainer
    model.optimizers = lambda: (opt_ae, opt_disc)
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *args, **kwargs: None

    batch = _make_batch(n_context=60, n_query=15, n_genes=50, coord_dim=2)
    before = model.decoder[0].weight.clone()

    model.training_step(batch, batch_idx=0)

    after = model.decoder[0].weight
    assert not torch.allclose(before, after), "decoder weights did not change — optimizer step had no effect"
    assert torch.isfinite(after).all(), "decoder weights contain NaN/Inf after one training step"
    print("[training_step] OK — decoder weights updated, no NaNs")


if __name__ == "__main__":
    test_sample()
    test_training_step_updates_weights()
    print("\nAll WAE-GAN smoke tests passed.")
