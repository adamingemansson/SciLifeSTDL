import torch

from gen2_architectures.models.arch1_gpt_baseline import Architecture1
from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder
from gen2_architectures.models.arch3_stage_b_latent_transformer import Architecture3StageB
from gen2_architectures.training.diagnostics import (
    attach_diagnostic_hooks, collect_diagnostics, compute_attention_entropy, format_diagnostics,
)


def _synthetic_context(n_context: int, feature_width: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "coords": torch.randn(n_context, 2, generator=g) * 500,
        "expression": torch.randn(n_context, feature_width, generator=g),
        "images": torch.randn(n_context, 1536, generator=g),
        "image_available": torch.ones(n_context, dtype=torch.bool),
    }


def test_architecture1_diagnostics_cover_gene_embedding_query_token_and_decoder():
    n_genes = 50
    model = Architecture1(n_genes=n_genes, feat_dim=32, coord_dim=16, conf_dim=8,
                           hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20, decoder_hidden_dim=128)
    stats = attach_diagnostic_hooks(model)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    model(context, query)

    diag = collect_diagnostics(model, stats)
    for key in ("gene_embedding_norm", "query_token_norm", "decoder_output_norm", "query_token_param_norm"):
        assert key in diag, f"missing {key}"
        assert diag[key] > 0.0
    # Architecture 1 has no true bottleneck latent -- must NOT report one
    assert "latent_norm" not in diag
    assert format_diagnostics(diag)  # non-empty, human-readable


def test_architecture1_attention_entropy_is_a_finite_positive_nats_value():
    n_genes = 50
    model = Architecture1(n_genes=n_genes, feat_dim=32, coord_dim=16, conf_dim=8,
                           hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20, decoder_hidden_dim=128)
    stats = attach_diagnostic_hooks(model)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    model(context, query)

    entropy = compute_attention_entropy(model, stats)
    assert entropy is not None
    assert entropy > 0.0


def test_attention_entropy_is_none_before_any_forward_pass():
    n_genes = 50
    model = Architecture1(n_genes=n_genes, feat_dim=32, coord_dim=16, conf_dim=8,
                           hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20, decoder_hidden_dim=128)
    stats = attach_diagnostic_hooks(model)
    assert compute_attention_entropy(model, stats) is None


def test_stage_a_autoencoder_diagnostics_report_true_latent_not_gene_embedding():
    n_genes = 100
    model = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    stats = attach_diagnostic_hooks(model)
    model(torch.randn(8, n_genes))

    diag = collect_diagnostics(model, stats)
    assert "latent_norm" in diag and diag["latent_norm"] > 0.0
    assert "decoder_output_norm" in diag and diag["decoder_output_norm"] > 0.0
    # no spatial component in Stage A -- these concepts genuinely don't apply
    assert "gene_embedding_norm" not in diag
    assert "query_token_norm" not in diag
    # attention entropy requires a `.transformer` attribute Stage A doesn't have
    assert compute_attention_entropy(model, stats) is None


def test_stage_b_diagnostics_cover_gene_embedding_and_query_token():
    n_genes = 50
    ae = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=16, hidden_dims=(64, 32))
    model = Architecture3StageB(ae, feat_dim=32, coord_dim=16, conf_dim=8, hidden_dim=64,
                                 n_layers=2, n_heads=4, max_neighbors=20)
    stats = attach_diagnostic_hooks(model)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    model(context, query)

    diag = collect_diagnostics(model, stats)
    assert diag["gene_embedding_norm"] > 0.0  # context spots' encoded latent (reused Stage-A encoder)
    assert diag["query_token_norm"] > 0.0     # latent_head's input, the post-transformer hidden state
    assert diag["decoder_output_norm"] > 0.0  # autoencoder.decoder's output == predicted_expression
