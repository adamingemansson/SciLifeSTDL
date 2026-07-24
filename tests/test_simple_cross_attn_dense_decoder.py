"""Test for simple_cross_attn_dense_decoder (2026-07-24): same
transport-vs-decoder isolation as stpath_backbone_simple_gene, applied to
the cross-attention encoder. No external `stpath` dependency (unlike the
STPath-backbone variant) so this one runs for real, not mocked.

Run with: python -m tests.test_simple_cross_attn_dense_decoder
"""
import torch

from src.models.conditioning import _GIGAPATH_FEAT_DIM
from src.models.registry import build_model


def test_simple_cross_attn_dense_decoder():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden = 40, 8, 20, 32
    model = build_model({
        "name": "simple_cross_attn_dense_decoder",
        "params": {
            "n_genes": n_genes, "hidden_dim": hidden, "n_heads": 4,
            "n_layers": 2, "knn_k": 5, "lr": 1e-3,
        },
    })
    context = {
        "coords": torch.randn(n_context, 3), "expression": torch.rand(n_context, n_genes),
        "images": torch.randn(n_context, _GIGAPATH_FEAT_DIM),
    }
    query = {"coords": torch.randn(n_query, 3), "images": torch.randn(n_query, _GIGAPATH_FEAT_DIM)}

    out = model.sample(context, query)
    assert out["expression"].shape == (n_query, n_genes)
    assert torch.isfinite(out["expression"]).all()

    batch = {"context": context, "query": query, "target_expression": torch.rand(n_query, n_genes)}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters()), (
        "gradient did not reach the decoder head"
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), (
        "gradient did not reach the encoder"
    )
    assert model.configure_optimizers() is not None
    print(f"[simple_cross_attn_dense_decoder] OK — sample/training_step/backward/optimizer, loss={loss.item():.4f}")


if __name__ == "__main__":
    test_simple_cross_attn_dense_decoder()
    print("\nAll simple_cross_attn_dense_decoder tests passed.")
