import torch

from src.models.registry import build_model


def test_honest_gene_conditioned_decoder_name_builds():
    model = build_model({
        "name": "fm_ot",
        "params": {
            "n_genes": 5,
            "coord_dim": 3,
            "cond_hidden_dim": 8,
            "latent_dim": 4,
            "ae_hidden_dim": 8,
            "hidden_dim": 16,
            "time_embed_dim": 4,
            "n_ode_steps": 2,
            "context_encoder_type": "builtin",
            "gene_encoder_type": "raw",
            "decoder_type": "gene_conditioned_vocabulary",
            "decoder_gene_names": [f"g{i}" for i in range(5)],
            "decoder_gene_embed_dim": 8,
            "decoder_hidden_dim": 8,
        },
    })
    h = torch.randn(3, 4 + 8)
    out = model._decode(h)
    assert out.shape == (3, 5)
