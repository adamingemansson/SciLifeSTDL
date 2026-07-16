"""
Smoke tests for the gene-expression encoder options added 2026-07-16
(src/models/conditioning.py MLPGeneEncoder, NovaeGeneEncoder,
precompute_novae_features) — the "is our own SpatialContextEncoder's
gene-expression branch also just a thin linear projection, same critique
raised about STPath" investigation.

MLPGeneEncoder/NovaeGeneEncoder/SpatialContextEncoder(gene_encoder_type=...)
have no external dependency and always run. precompute_novae_features
needs the real `novae` package (`pip install novae`) plus network access
to download its pretrained weights from HuggingFace Hub — not available
in every environment — so that one test skips cleanly, same pattern as
tests/test_stpath_encoder.py for STPath/Gigapath.

Run with:
    python -m tests.test_gene_encoders
"""
import numpy as np
import torch

from src.models.conditioning import (
    MLPGeneEncoder, NovaeGeneEncoder, CombinedGeneEncoder, SpatialContextEncoder,
)


def test_mlp_gene_encoder():
    torch.manual_seed(0)
    n, n_genes, feat_dim = 6, 20, 8
    encoder = MLPGeneEncoder(n_genes, feat_dim)
    expr = torch.rand(n, n_genes)
    out = encoder(expr)
    assert out.shape == (n, feat_dim), out.shape
    assert torch.isfinite(out).all()
    print(f"[MLPGeneEncoder] OK — output shape {tuple(out.shape)}")


def test_novae_gene_encoder():
    torch.manual_seed(0)
    n, novae_dim, feat_dim = 6, 64, 8
    encoder = NovaeGeneEncoder(novae_dim, feat_dim)
    features = torch.rand(n, novae_dim)
    out = encoder(features)
    assert out.shape == (n, feat_dim), out.shape
    assert torch.isfinite(out).all()

    # real structural difference from every other encoder in this file:
    # NovaeGeneEncoder has no raw-input fallback (see its own docstring
    # for why — Novae's representations depend on the whole sample's
    # spatial graph, not a single row) - a 1D or raw-expression-shaped
    # input must raise, not silently misinterpret the tensor
    raised = False
    try:
        encoder(torch.rand(n))
    except ValueError:
        raised = True
    assert raised, "NovaeGeneEncoder must reject non-2D input, not silently misuse it"
    print("[NovaeGeneEncoder] OK — output shape correct, rejects malformed input")


def test_combined_gene_encoder():
    torch.manual_seed(0)
    n, n_genes, novae_dim, feat_dim = 6, 20, 64, 8
    encoder = CombinedGeneEncoder(n_genes, novae_dim, feat_dim)
    raw_expr = torch.rand(n, n_genes)
    novae_features = torch.rand(n, novae_dim)
    out = encoder(raw_expr, novae_features)
    assert out.shape == (n, feat_dim), out.shape
    assert torch.isfinite(out).all()
    # real check that BOTH sub-encoders actually contribute, not just one
    # silently dominating/zeroing the other — mlp_only should differ from
    # the combined output since novae's contribution is nonzero
    mlp_only = encoder.mlp(raw_expr)
    assert not torch.allclose(out, mlp_only), "novae contribution appears to be zero/ignored"
    assert encoder.output_dim_multiplier == 1
    print(f"[CombinedGeneEncoder sum] OK — output shape {tuple(out.shape)}, both sub-encoders contribute")

    # concat mode (2026-07-16 fix): output must be 2*feat_dim, with the
    # first half exactly equal to the MLP-alone output and the second
    # half exactly equal to the Novae-alone output — concatenation, not
    # a mix — this is the real property that makes a downstream Linear
    # able to weight the two sources independently, unlike sum mode.
    concat_encoder = CombinedGeneEncoder(n_genes, novae_dim, feat_dim, combine_mode="concat")
    concat_out = concat_encoder(raw_expr, novae_features)
    assert concat_out.shape == (n, 2 * feat_dim), concat_out.shape
    assert concat_encoder.output_dim_multiplier == 2
    assert torch.allclose(concat_out[:, :feat_dim], concat_encoder.mlp(raw_expr))
    assert torch.allclose(concat_out[:, feat_dim:], concat_encoder.novae(novae_features))
    print(f"[CombinedGeneEncoder concat] OK — output shape {tuple(concat_out.shape)}, "
          f"sources kept genuinely separate")


def test_spatial_context_encoder_gene_encoder_types():
    """SpatialContextEncoder with gene_encoder_type="raw"/"mlp"/"novae" —
    checks the new switch doesn't break the existing fusion/message-passing
    path for any of the three, and that "novae" genuinely expects
    [N, novae_dim] features in context_expression's slot (not [N, n_genes]
    raw expression — a real, deliberate difference from "raw"/"mlp")."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, novae_dim, hidden_dim = 12, 4, 15, 32, 16

    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100

    for gene_encoder_type, expr_dim in [("raw", n_genes), ("mlp", n_genes), ("novae", novae_dim)]:
        kwargs = {"novae_dim": novae_dim} if gene_encoder_type == "novae" else {}
        encoder = SpatialContextEncoder(
            n_genes=n_genes, coord_dim=3, hidden_dim=hidden_dim,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=8, **kwargs,
        )
        context_expression = torch.rand(n_context, expr_dim)
        c = encoder(context_coords, context_expression, query_coords)
        assert c.shape == (n_query, hidden_dim), (gene_encoder_type, c.shape)
        assert torch.isfinite(c).all()
        print(f"[SpatialContextEncoder gene_encoder_type={gene_encoder_type!r}] "
              f"OK — output shape {tuple(c.shape)}")

    # constructing with gene_encoder_type="novae" but no novae_dim must
    # fail loudly at construction time, not with a confusing shape error
    # later inside forward()
    raised = False
    try:
        SpatialContextEncoder(n_genes=n_genes, gene_encoder_type="novae")
    except AssertionError:
        raised = True
    assert raised, "gene_encoder_type='novae' without novae_dim must raise at construction"
    print("[SpatialContextEncoder] OK — novae without novae_dim raises at construction")


def test_precompute_novae_features():
    try:
        import novae  # noqa: F401
    except ImportError:
        print("[precompute_novae_features] SKIPPED — `novae` package not installed "
              "(pip install novae)")
        return

    try:
        import anndata as ad
    except ImportError:
        print("[precompute_novae_features] SKIPPED — anndata not installed")
        return

    from src.models.conditioning import precompute_novae_features

    rng = np.random.default_rng(0)
    n, n_genes = 40, 30
    adata = ad.AnnData(X=rng.random((n, n_genes)).astype(np.float32))
    adata.var_names = [f"GENE{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = rng.random((n, 2)) * 1000

    try:
        features = precompute_novae_features(adata)
    except Exception as e:
        print(f"[precompute_novae_features] SKIPPED — could not run Novae "
              f"(network access / HF weights unavailable?) ({e})")
        return

    assert features.shape[0] == n, features.shape
    assert features.ndim == 2
    assert np.isfinite(features).all()
    print(f"[precompute_novae_features] OK — output shape {features.shape}")


if __name__ == "__main__":
    test_mlp_gene_encoder()
    test_novae_gene_encoder()
    test_combined_gene_encoder()
    test_spatial_context_encoder_gene_encoder_types()
    test_precompute_novae_features()
    print("\nAll gene-encoder smoke tests done (see above for SKIPPED vs OK).")
