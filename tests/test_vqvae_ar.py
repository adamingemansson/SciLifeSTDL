"""
Smoke test for VQ-VAE + autoregressive transformer (stage 2,
src/models/registry.py VQVAEAutoregressive). Tiny synthetic data, no real
dataset or Lightning Trainer needed — calls sample() and training_step()
directly to check shapes/gradients/no-NaN, and that sample() actually
returns output aligned back to the original (not Morton-sorted) query
order.

Run with: python -m tests.test_vqvae_ar
"""
import torch

from src.models.registry import VQVAEAutoregressive


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
    model = VQVAEAutoregressive(n_genes=50, coord_dim=3, cond_hidden_dim=32,
                                 latent_dim=8, ae_hidden_dim=32, codebook_size=16,
                                 transformer_dim=32, n_transformer_layers=2, n_heads=2,
                                 max_seq_len=64)
    model.eval()
    batch = _make_batch(n_context=60, n_query=10, n_genes=50, coord_dim=3)

    with torch.no_grad():
        out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (10, 50), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()
    # output must be aligned back to the original query order, not left in
    # Morton order — coords passed through unchanged is the contract check
    assert torch.equal(out["coords"], batch["query"]["coords"])
    # regression check for the real collapse seen on HEST-1k data
    # (docs/model_schematics.md, 2026-07-14): greedy argmax generated the
    # identical token for every query point regardless of conditioning
    unique_rows = torch.unique(out["expression"], dim=0)
    assert unique_rows.shape[0] > 1, "sample() collapsed to one repeated output for every query point"
    print(f"[sample] OK — output shape {tuple(out['expression'].shape)}, "
          f"aligned to original query order")


def test_training_step_updates_weights():
    torch.manual_seed(0)
    model = VQVAEAutoregressive(n_genes=50, coord_dim=2, cond_hidden_dim=32,
                                 latent_dim=8, ae_hidden_dim=32, codebook_size=16,
                                 transformer_dim=32, n_transformer_layers=2, n_heads=2,
                                 max_seq_len=64)
    opt = model.configure_optimizers()
    model.log_dict = lambda *args, **kwargs: None

    batch = _make_batch(n_context=40, n_query=12, n_genes=50, coord_dim=2)
    before_enc = model.encoder[0].weight.clone()
    before_transformer = next(model.transformer.parameters()).clone()

    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    after_enc = model.encoder[0].weight
    after_transformer = next(model.transformer.parameters())
    assert not torch.allclose(before_enc, after_enc), "encoder weights did not change"
    assert not torch.allclose(before_transformer, after_transformer), "transformer weights did not change"
    assert torch.isfinite(after_enc).all()
    assert torch.isfinite(after_transformer).all()
    print("[training_step] OK — encoder + transformer updated, no NaNs")


if __name__ == "__main__":
    test_sample()
    test_training_step_updates_weights()
    print("\nAll VQ-VAE+AR smoke tests passed.")
