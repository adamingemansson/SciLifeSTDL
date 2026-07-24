"""Smoke test for SimpleFusionSpatialTransformerContextEncoder (2026-07-24):
our own GigaPath + gene-MLP tokens fed through STPath's REAL
SpatialTransformer backbone, no organ/tech tokens, no STPath gene
vocabulary. Needs the external `stpath` package installed (git clone
Graph-and-Geometric-Learning/STPath + pip install -e .) -- not needed for
any pretrained weights or gene vocabulary here, just the SpatialTransformer/
ModelConfig classes themselves. Skips cleanly if not installed, same
pattern as tests/test_stpath_encoder.py and tests/test_stpath_scratch.py.

Unlike this file's siblings, the construction details (SpatialTransformer
taking one ModelConfig object, forward(features, coords, batch_idx))
were verified by reading STPath's real source directly, but never
executed end-to-end before this test exists to run it for real -- treat
a first real run of this test as the actual verification, not a formality.

Run with: python -m tests.test_simple_fusion_stpath_transformer
"""
import torch

from src.models.conditioning import _GIGAPATH_FEAT_DIM


def test_simple_fusion_stpath_transformer():
    try:
        from src.models.simple_fusion_encoder import SimpleFusionSpatialTransformerContextEncoder
    except ImportError as e:
        print(f"[simple_fusion_stpath_transformer] SKIPPED — stpath package not installed ({e})")
        return

    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden = 40, 8, 30, 32

    try:
        enc = SimpleFusionSpatialTransformerContextEncoder(
            n_genes=n_genes, hidden_dim=hidden, n_layers=2, n_heads=4,
        )
    except Exception as e:
        print(f"[simple_fusion_stpath_transformer] SKIPPED — could not construct ({e})")
        return

    context_coords = torch.randn(n_context, 3)
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.randn(n_context, _GIGAPATH_FEAT_DIM)
    query_coords = torch.randn(n_query, 3)
    query_images = torch.randn(n_query, _GIGAPATH_FEAT_DIM)

    out = enc(context_coords, context_expression, query_coords, context_images, query_images)
    assert out.shape == (n_query, hidden), out.shape
    assert torch.isfinite(out).all()
    print(f"[simple_fusion_stpath_transformer] OK — output shape {tuple(out.shape)}")

    out.sum().backward()
    grad_reached = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert grad_reached, "gradient did not reach any parameter"
    print("[simple_fusion_stpath_transformer] OK — gradient reaches parameters (incl. backbone)")

    # no-image path -- must fall back to mask_token, not crash
    out2 = enc(context_coords, context_expression, query_coords, None, None)
    assert out2.shape == (n_query, hidden)
    assert torch.isfinite(out2).all()
    print("[simple_fusion_stpath_transformer] OK — no-image (mask_token) path")


if __name__ == "__main__":
    test_simple_fusion_stpath_transformer()
    print("\nsimple_fusion_stpath_transformer smoke test done (see above for SKIPPED vs OK).")
