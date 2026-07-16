"""
Smoke test for StormLiteContextEncoder (src/models/storm_lite_encoder.py,
2026-07-16 STORM-recipe-at-pilot-scale investigation).

No external dependency needed to test the architecture itself — uses
precomputed [B, gigapath_dim]/[B, novae_dim] feature tensors (the fast
path GigapathPatchEncoder/NovaeGeneEncoder always use in real training),
never touching the real frozen GigaPath/Novae models.

Run with:
    python -m tests.test_storm_lite_encoder
"""
import torch

from src.models.storm_lite_encoder import StormLiteContextEncoder
from src.models.conditioning import _GIGAPATH_FEAT_DIM


def test_storm_lite_context_encoder():
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 10, 4, 20, 64, 16

    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)  # precomputed-features fast path
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    context_novae_features = torch.rand(n_context, novae_dim)

    for gene_encoder_type in ("mlp", "novae", "both"):
        encoder = StormLiteContextEncoder(
            n_genes=n_genes, novae_dim=novae_dim, hidden_dim=hidden_dim,
            n_transformer_layers=2, n_heads=4, gene_encoder_type=gene_encoder_type,
        )
        c = encoder(
            context_coords, context_expression, query_coords,
            context_images, query_images,
            context_novae_features=context_novae_features,
        )
        assert c.shape == (n_query, hidden_dim), (gene_encoder_type, c.shape)
        assert torch.isfinite(c).all()

        # real gradient-flow check: every trainable submodule should
        # receive a gradient from a backward pass through c, not just the
        # transformer (a real bug class this project has hit before —
        # see stpath_encoder.py's own docstring on a near-identical
        # silent-no-grad mistake)
        loss = c.sum()
        loss.backward()
        no_grad_params = [
            name for name, p in encoder.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
        ]
        assert not no_grad_params, (
            f"gene_encoder_type={gene_encoder_type!r}: params with no/invalid gradient: {no_grad_params}"
        )
        print(f"[StormLiteContextEncoder gene_encoder_type={gene_encoder_type!r}] "
              f"OK — output shape {tuple(c.shape)}, all params received gradient")


if __name__ == "__main__":
    test_storm_lite_context_encoder()
    print("\nStormLiteContextEncoder smoke test done.")
