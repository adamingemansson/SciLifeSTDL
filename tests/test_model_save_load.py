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
    test_interp_baseline_save_is_noop()
    print("\nAll model save/load smoke tests passed.")
