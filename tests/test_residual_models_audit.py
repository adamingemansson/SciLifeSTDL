from pathlib import Path

import torch
import torch.nn as nn

from src.models.conditioning import MLPGeneEncoder
from src.models.registry import build_model


def _item(n_context=12, n_query=4, n_genes=6):
    return {
        "context": {
            "coords": torch.randn(n_context, 3),
            "expression": torch.rand(n_context, n_genes),
        },
        "query": {"coords": torch.randn(n_query, 3)},
        "target_expression": torch.rand(n_query, n_genes),
    }


def test_mlp_gene_encoder_uses_lightweight_default():
    encoder = MLPGeneEncoder(16570, feat_dim=256)
    first = next(module for module in encoder.modules() if isinstance(module, nn.Linear))
    assert first.out_features == 512
    assert first.weight.numel() < 10_000_000


def test_harmonic_residual_starts_near_anchor_with_upstream_gradients():
    model = build_model({
        "name": "harmonic_residual",
        "params": {
            "n_genes": 6, "cond_hidden_dim": 16, "hidden_dim": 16,
            "gene_encoder_type": "mlp", "gene_feat_dim": 8,
        },
    })
    item = _item()
    out = model.sample(item["context"], item["query"])
    assert torch.count_nonzero(out["residual_expression"]) > 0
    assert out["residual_expression"].square().mean().sqrt() < 0.05
    loss = model.training_step(item, 0)
    loss.backward()
    assert model.residual[0].weight.grad is not None
    assert model.residual[0].weight.grad.norm() > 0
    context_grad = sum(
        p.grad.norm() for p in model.context_encoder.parameters() if p.grad is not None
    )
    assert context_grad > 0


def test_harmonic_residual_uses_configured_gene_scales():
    model = build_model({
        "name": "harmonic_residual",
        "params": {
            "n_genes": 3, "cond_hidden_dim": 8, "hidden_dim": 8,
            "gene_encoder_type": "mlp", "gene_feat_dim": 4,
            "residual_gene_scale": [0.01, 0.2, 2.0],
            "residual_scale_floor": 0.05,
        },
    })
    assert torch.allclose(model.residual_gene_scale, torch.tensor([0.05, 0.2, 2.0]))


def test_direct_context_regressor_has_no_harmonic_prediction_path():
    mean = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    scale = torch.tensor([0.01, 0.1, 0.2, 0.5, 1.0, 2.0])
    model = build_model({
        "name": "direct_context_regressor",
        "params": {
            "n_genes": 6, "cond_hidden_dim": 16, "hidden_dim": 16,
            "gene_encoder_type": "mlp", "gene_feat_dim": 8,
            "context_encoder_type": "storm_lite",
            "storm_lite_fusion_mode": "local_pool",
            "target_gene_mean": mean.tolist(),
            "target_gene_scale": scale.tolist(),
            "target_scale_floor": 0.05,
        },
    })
    item = _item()
    out = model.sample(item["context"], item["query"])
    assert out["expression"].shape == item["target_expression"].shape
    assert torch.allclose(out["anchor_expression"], mean.expand(4, -1))
    assert torch.allclose(model.target_gene_scale, torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 2.0]))

    # The learned prediction is not an interpolation of context expression:
    # its only reported baseline is the training-only global gene mean.
    shifted = _item()
    shifted["context"]["coords"] = item["context"]["coords"]
    shifted["query"]["coords"] = item["query"]["coords"]
    shifted["context"]["expression"] = item["context"]["expression"] + 2.0
    shifted_out = model.sample(shifted["context"], shifted["query"])
    assert torch.allclose(shifted_out["anchor_expression"], out["anchor_expression"])
    assert not torch.allclose(shifted_out["expression"], out["expression"])

    loss = model.training_step(item, 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.decoder[-1].weight.grad is not None
    assert model.decoder[-1].weight.grad.norm() > 0
    context_grad = sum(
        p.grad.norm() for p in model.context_encoder.parameters() if p.grad is not None
    )
    assert context_grad > 0


def test_residual_flow_loads_and_freezes_validated_autoencoder(tmp_path: Path):
    n_genes, latent, hidden = 6, 3, 7
    encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU(), nn.Linear(hidden, latent))
    decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_genes))
    ckpt = tmp_path / "ae.pt"
    torch.save({
        "n_genes": n_genes,
        "latent_dim": latent,
        "hidden_dim": hidden,
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "target_type": "harmonic_residual",
        "harmonic_k": 8,
        "harmonic_ridge": 1e-4,
        "gene_names": [f"g{i}" for i in range(n_genes)],
        "validation_rmse": 0.1,
        "validation_pcc": 0.9,
    }, ckpt)
    model = build_model({
        "name": "residual_fm_ot",
        "params": {
            "n_genes": n_genes, "cond_hidden_dim": 12, "latent_dim": latent,
            "ae_hidden_dim": hidden, "hidden_dim": 16, "time_embed_dim": 8,
            "n_ode_steps": 2, "gene_encoder_type": "mlp", "gene_feat_dim": 8,
            "pretrained_autoencoder_path": str(ckpt),
            "freeze_pretrained_autoencoder": True,
            "pretrained_autoencoder_gene_names": [f"g{i}" for i in range(n_genes)],
        },
    })
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.pretrained_ae_decoder.parameters())
    item = _item(n_genes=n_genes)
    out = model.sample(item["context"], item["query"])
    assert out["expression"].shape == item["target_expression"].shape
    loss = model.training_step(item, 0)
    assert torch.isfinite(loss)


def test_residual_flow_rejects_absolute_expression_autoencoder(tmp_path: Path):
    n_genes, latent, hidden = 4, 2, 5
    encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU(), nn.Linear(hidden, latent))
    decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_genes))
    ckpt = tmp_path / "old_absolute_ae.pt"
    torch.save({
        "n_genes": n_genes,
        "latent_dim": latent,
        "hidden_dim": hidden,
        "gene_names": [f"g{i}" for i in range(n_genes)],
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "validation_rmse": 0.1,
        "validation_pcc": 0.9,
    }, ckpt)
    import pytest
    with pytest.raises(ValueError, match="not validated on harmonic residuals"):
        build_model({
            "name": "residual_fm_ot",
            "params": {
                "n_genes": n_genes, "cond_hidden_dim": 8, "latent_dim": latent,
                "ae_hidden_dim": hidden, "hidden_dim": 8, "time_embed_dim": 4,
                "n_ode_steps": 2, "gene_encoder_type": "mlp", "gene_feat_dim": 4,
                "pretrained_autoencoder_path": str(ckpt),
                "pretrained_autoencoder_gene_names": [f"g{i}" for i in range(n_genes)],
            },
        })
