"""Smoke + end-to-end tests for the two new lightweight context encoders
(2026-07-24): SimpleFusionContextEncoder (GigaPath + gene-MLP tokens,
uniform mean-pool over k nearest context spots, no attention -- the
"simple fusion" arm) and SimpleCrossAttentionContextEncoder (identical
token construction, one learned cross-attention layer instead of a fixed
average -- the "with transformer" arm). Both are registered as
context_transport_regressor conditioning_mode/context_encoder_type
options 'simple_fusion' / 'simple_cross_attn'; see registry.py.

Run with: python -m tests.test_simple_fusion_encoder
"""
import torch

from src.models.conditioning import _GIGAPATH_FEAT_DIM
from src.models.registry import build_model
from src.models.simple_fusion_encoder import (
    SimpleCrossAttentionContextEncoder,
    SimpleFusionContextEncoder,
)


def _run_encoder_case(cls, label: str):
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden = 40, 8, 30, 32
    enc = cls(n_genes=n_genes, hidden_dim=hidden, knn_k=5)
    context_coords = torch.randn(n_context, 3)
    context_expression = torch.rand(n_context, n_genes)
    query_coords = torch.randn(n_query, 3)

    # No images at all -- must fall back to the learned mask_token, not crash.
    out = enc(context_coords, context_expression, query_coords, None, None)
    assert out.shape == (n_query, hidden), f"[{label}] bad shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"[{label}] no-image output has NaN/Inf"

    # Precomputed Gigapath-shaped features (rank-2 -> fast cached path).
    context_images = torch.randn(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.randn(n_query, _GIGAPATH_FEAT_DIM)
    out2 = enc(context_coords, context_expression, query_coords, context_images, query_images)
    assert out2.shape == (n_query, hidden)
    assert torch.isfinite(out2).all(), f"[{label}] with-image output has NaN/Inf"

    out2.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()), (
        f"[{label}] no gradient reached any parameter"
    )
    print(f"[{label}] OK — shape {tuple(out2.shape)}, gradient reaches parameters")


def _run_registry_case(mode: str):
    torch.manual_seed(0)
    n_context, n_query, n_genes = 40, 8, 20
    model = build_model({
        "name": "context_transport_regressor",
        "params": {
            "n_genes": n_genes, "coord_dim": 3, "cond_hidden_dim": 32,
            "score_hidden_dim": 16, "transport_k": 8,
            "conditioning_mode": mode, "context_encoder_type": mode, "lr": 1e-3,
        },
    })
    context = {
        "coords": torch.randn(n_context, 3),
        "expression": torch.rand(n_context, n_genes),
        "images": torch.randn(n_context, _GIGAPATH_FEAT_DIM),
    }
    query = {
        "coords": torch.randn(n_query, 3),
        "images": torch.randn(n_query, _GIGAPATH_FEAT_DIM),
    }
    out = model.sample(context, query)
    assert out["expression"].shape == (n_query, n_genes)
    assert torch.isfinite(out["expression"]).all()

    batch = {"context": context, "query": query, "target_expression": torch.rand(n_query, n_genes)}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.configure_optimizers() is not None
    print(f"[registry: {mode}] OK — end-to-end sample/training_step/backward, loss={loss.item():.4f}")


def _run_guardrail_case():
    try:
        build_model({"name": "context_transport_regressor", "params": {
            "n_genes": 10, "conditioning_mode": "simple_fusion", "context_encoder_type": "geometry",
        }})
        raise AssertionError("should have raised on mismatched context_encoder_type")
    except ValueError:
        pass

    try:
        build_model({"name": "context_transport_regressor", "params": {
            "n_genes": 10, "conditioning_mode": "nonsense",
        }})
        raise AssertionError("should have raised on unknown conditioning_mode")
    except ValueError:
        pass
    print("[guardrails] OK — mismatched/unknown conditioning_mode both raise ValueError")


if __name__ == "__main__":
    _run_encoder_case(SimpleFusionContextEncoder, "SimpleFusionContextEncoder")
    _run_encoder_case(SimpleCrossAttentionContextEncoder, "SimpleCrossAttentionContextEncoder")
    _run_registry_case("simple_fusion")
    _run_registry_case("simple_cross_attn")
    _run_guardrail_case()
    print("\nAll simple_fusion_encoder tests passed.")
