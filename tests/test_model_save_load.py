"""
Smoke test for save_trained_model / load_trained_model
(src/training/train.py, added 2026-07-15 after training runs had no way
to persist a trained model — see docs/architecture_plan.md). Proves a
round trip: train a tiny model, save it, reload it into a FRESH model
instance via its saved model_cfg, and check the reloaded model produces
the same output as the original for the same input — not just that
saving/loading doesn't crash, but that the weights actually transferred.

Also checks gene_names round-trips exactly, since that's the piece this
whole feature exists for (aligning a saved model's fixed-width output to
a DIFFERENT sample's genes by name later — see save_trained_model's
docstring).

Run with: python -m tests.test_model_save_load
"""
import tempfile
from pathlib import Path

import torch

from src.models.registry import build_model
from src.training.train import save_trained_model, load_trained_model


def test_save_load_round_trip():
    torch.manual_seed(0)
    n_genes = 30
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    model_cfg = {
        "name": "wae_gan",
        "params": {
            "n_genes": n_genes, "coord_dim": 3, "latent_dim": 4,
            "hidden_dim": 16, "cond_hidden_dim": 16, "disc_hidden_dim": 8,
        },
    }
    model = build_model(model_cfg)
    model.eval()

    n_context, n_query = 20, 5
    context = {
        "coords": torch.randn(n_context, 3),
        "expression": torch.rand(n_context, n_genes),
    }
    query = {"coords": torch.randn(n_query, 3)}

    torch.manual_seed(1)
    with torch.no_grad():
        original_out = model.sample(context, query)["expression"]

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpt"
        saved_path = save_trained_model(model, model_cfg, gene_names, str(checkpoint_dir))
        assert saved_path is not None, "wae_gan has trainable params, save should not no-op"
        assert saved_path.exists()
        assert (checkpoint_dir / "model_cfg.json").exists()
        assert (checkpoint_dir / "gene_names.json").exists()

        loaded_model, loaded_gene_names = load_trained_model(str(checkpoint_dir))
        assert loaded_gene_names == gene_names, "gene_names must round-trip exactly"

        torch.manual_seed(1)  # same seed as the original .sample() call, for any stochastic ops
        with torch.no_grad():
            loaded_out = loaded_model.sample(context, query)["expression"]

        assert torch.allclose(original_out, loaded_out, atol=1e-6), (
            "reloaded model produced different output than the original for the same "
            "input — weights did not actually transfer correctly"
        )
        print("[model_save_load] OK — reloaded model matches original output exactly, "
              "gene_names round-tripped correctly")


def test_random_fourier_features_buffer_round_trips():
    """Regression test for a real bug found 2026-07-19: save_trainable_state_dict
    used to filter purely on requires_grad (model.named_parameters()),
    which silently drops every BUFFER — including RandomFourierFeatures.B,
    the fixed random projection matrix that defines the whole model's
    coordinate encoding (src/models/conditioning.py). Since build_model()
    reinitializes it with a NEW random value on every call, a reloaded
    model previously got a completely different coordinate encoding than
    the one it was actually trained with — test_save_load_round_trip's own
    output-equality check would have already caught this (and did, before
    the fix), but this test isolates the exact buffer to make the
    regression unambiguous if it's ever reintroduced."""
    torch.manual_seed(0)
    model_cfg = {
        "name": "wae_gan",
        "params": {"n_genes": 10, "coord_dim": 3, "latent_dim": 4,
                    "hidden_dim": 16, "cond_hidden_dim": 16, "disc_hidden_dim": 8},
    }
    model = build_model(model_cfg)
    original_B = model.context_encoder.coord_encoder.B.clone()

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpt"
        save_trained_model(model, model_cfg, [f"G{i}" for i in range(10)], str(checkpoint_dir))
        loaded_model, _ = load_trained_model(str(checkpoint_dir))
        assert torch.equal(original_B, loaded_model.context_encoder.coord_encoder.B), (
            "RandomFourierFeatures.B did not survive save/load — the reloaded model's "
            "coordinate encoding basis differs from what it was actually trained with"
        )
    print("[model_save_load] OK — RandomFourierFeatures.B (fixed random buffer) round-trips exactly")


def test_vqvae_codebook_buffer_round_trips():
    """Regression test for the same class of bug as
    test_random_fourier_features_buffer_round_trips, for VectorQuantizer's
    EMA-updated codebook (src/models/vqvae.py) — embed/ema_cluster_size/
    ema_embed_sum are buffers UPDATED BY TRAINING (via EMA, not gradient
    descent, so never in named_parameters() at all), meaning a saved
    VQ-VAE+AR checkpoint previously lost its ENTIRE learned codebook on
    reload, resetting to random init with zero cluster usage."""
    torch.manual_seed(0)
    model_cfg = {
        "name": "vqvae_ar",
        "params": {"n_genes": 10, "coord_dim": 3, "cond_hidden_dim": 16, "latent_dim": 8,
                    "ae_hidden_dim": 16, "codebook_size": 16, "transformer_dim": 16,
                    "n_transformer_layers": 1, "n_heads": 2},
    }
    model = build_model(model_cfg)
    # simulate real EMA drift away from random init, the way actual training would
    with torch.no_grad():
        model.vq.embed.add_(1.0)
        model.vq.ema_cluster_size.add_(5.0)
    original_embed = model.vq.embed.clone()
    original_cluster_size = model.vq.ema_cluster_size.clone()

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpt"
        save_trained_model(model, model_cfg, [f"G{i}" for i in range(10)], str(checkpoint_dir))
        loaded_model, _ = load_trained_model(str(checkpoint_dir))
        assert torch.equal(original_embed, loaded_model.vq.embed), (
            "VectorQuantizer's learned codebook (embed) did not survive save/load"
        )
        assert torch.equal(original_cluster_size, loaded_model.vq.ema_cluster_size), (
            "VectorQuantizer's ema_cluster_size did not survive save/load"
        )
    print("[model_save_load] OK — VectorQuantizer's EMA codebook buffers round-trip exactly")


def test_gigapath_frozen_backbone_still_excluded():
    """Guard against the fix above being TOO broad: frozen pretrained
    backbones (GigapathPatchEncoder's tile_encoder, when actually loaded)
    must stay excluded from the saved checkpoint — that's the whole
    ~4.7GB-redundancy point save_trainable_state_dict exists for. Uses
    GigapathPatchEncoder directly (not a full model) so this doesn't
    require downloading real Gigapath weights: constructs a small stand-in
    frozen submodule with the exact same signature
    _is_frozen_backbone_module checks (all params requires_grad=False),
    verifying it's correctly identified as frozen."""
    from src.training.train import _is_frozen_backbone_module
    import torch.nn as nn

    frozen = nn.Linear(4, 4)
    for p in frozen.parameters():
        p.requires_grad_(False)
    assert _is_frozen_backbone_module(frozen), "a module with all params frozen must be classified as frozen"

    trainable = nn.Linear(4, 4)
    assert not _is_frozen_backbone_module(trainable), "a module with trainable params must NOT be classified as frozen"

    buffer_only = torch.nn.Module()
    buffer_only.register_buffer("B", torch.randn(4, 4))
    assert not _is_frozen_backbone_module(buffer_only), (
        "a parameter-free buffer-only module (e.g. RandomFourierFeatures) must NOT be "
        "classified as frozen, even though 'all zero of its params are trainable' is "
        "vacuously true — it requires >=1 owned parameter, see the function's own docstring"
    )
    print("[model_save_load] OK — _is_frozen_backbone_module correctly distinguishes "
          "frozen-backbone / trainable / buffer-only modules")


def test_interp_baseline_save_is_noop():
    """Parameter-free models (interp_baseline) have nothing to save —
    save_trained_model must return None cleanly, not crash or write an
    empty/meaningless checkpoint."""
    model = build_model({"name": "interp_baseline", "params": {}})
    with tempfile.TemporaryDirectory() as tmp:
        result = save_trained_model(model, {"name": "interp_baseline", "params": {}}, [], tmp)
        assert result is None
        assert not (Path(tmp) / "trainable_weights.pt").exists()
    print("[model_save_load] OK — parameter-free model save is a clean no-op")


if __name__ == "__main__":
    test_save_load_round_trip()
    test_random_fourier_features_buffer_round_trips()
    test_vqvae_codebook_buffer_round_trips()
    test_gigapath_frozen_backbone_still_excluded()
    test_interp_baseline_save_is_noop()
    print("\nAll model save/load smoke tests passed.")
