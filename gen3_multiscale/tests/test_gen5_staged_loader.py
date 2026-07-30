"""Item 4 (six-launch-blocker audit), Gen5-only half: "Gen5 must also
load the exact shared autoencoder." gen4/staged_loader.py's
`load_shared_autoencoder` is generic (any nn.Module); this test exercises
it against a real ExpressionAutoencoder, which only exists in this (Gen5)
worktree."""
from __future__ import annotations

import torch

from gen3_multiscale.gen4.staged_loader import load_shared_autoencoder
from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder
from gen3_multiscale.training import checkpoint as checkpoint_module

N_GENES = 6
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


def test_load_shared_autoencoder_loads_real_weights(tmp_path):
    torch.manual_seed(0)
    trained = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=5, hidden_dim=16)
    checkpoint_dir = tmp_path / "gen5_autoencoder"
    checkpoint_module.save_checkpoint(trained, {"kind": "autoencoder"}, GENE_NAMES, checkpoint_dir, step=7)

    torch.manual_seed(1)
    fresh = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=5, hidden_dim=16)
    trained_state = trained.state_dict()
    fresh_state_before = {k: v.clone() for k, v in fresh.state_dict().items()}
    assert any(not torch.allclose(trained_state[k], fresh_state_before[k]) for k in trained_state)

    info = load_shared_autoencoder(fresh, str(checkpoint_dir), GENE_NAMES)
    assert info["loaded"] is True
    assert info["checkpoint_step"] == 7
    assert info["checkpoint_sha256"] is not None
    fresh_state_after = fresh.state_dict()
    assert all(torch.allclose(trained_state[k], fresh_state_after[k]) for k in trained_state)


def test_load_shared_autoencoder_rejects_gene_panel_mismatch(tmp_path):
    trained = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=5, hidden_dim=16)
    checkpoint_dir = tmp_path / "gen5_autoencoder"
    checkpoint_module.save_checkpoint(trained, {"kind": "autoencoder"}, GENE_NAMES, checkpoint_dir, step=1)

    fresh = ExpressionAutoencoder(n_genes=N_GENES, gene_names=GENE_NAMES, latent_dim=5, hidden_dim=16)
    import pytest
    with pytest.raises(ValueError, match="different gene panel"):
        load_shared_autoencoder(fresh, str(checkpoint_dir), list(reversed(GENE_NAMES)))
