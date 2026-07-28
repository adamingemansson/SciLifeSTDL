"""GPT-audit-flagged bug (2026-07-27, second-pass re-audit, confirmed and
fixed): Stage A/B's ordered-gene-identity check silently skipped itself
(fell back to the width-only check) whenever gene_names.json was missing,
instead of raising. checkpoint.save_checkpoint always writes
gene_names.json, so a missing file means a genuinely incomplete Stage A
checkpoint -- this must fail closed, not silently trust an unverifiable
gene panel."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder
from gen2_architectures.training import checkpoint
from gen2_architectures.training.train_arch3_stage_b import _load_stage_a
from gen2_architectures.training.run_held_out_evaluation import _load_stage_b_model


def _save_stage_a_checkpoint(checkpoint_dir, n_genes=10, gene_names=None):
    model = DenoisingTranscriptomeAutoencoder(n_genes=n_genes, latent_dim=8, hidden_dims=(16, 12))
    checkpoint.save_checkpoint(
        model, {"n_genes": n_genes, "params": {"latent_dim": 8, "hidden_dims": (16, 12)}},
        gene_names or [f"g{i}" for i in range(n_genes)], checkpoint_dir, step=0,
    )


def test_load_stage_a_raises_when_gene_names_json_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        _save_stage_a_checkpoint(tmp, n_genes=10)
        (Path(tmp) / "gene_names.json").unlink()  # simulate an incomplete checkpoint

        with pytest.raises(ValueError, match="gene_names.json"):
            _load_stage_a(tmp, gene_names=[f"g{i}" for i in range(10)])


def test_load_stage_a_succeeds_when_gene_names_match():
    gene_names = [f"g{i}" for i in range(10)]
    with tempfile.TemporaryDirectory() as tmp:
        _save_stage_a_checkpoint(tmp, n_genes=10, gene_names=gene_names)
        autoencoder = _load_stage_a(tmp, gene_names=gene_names)
        assert isinstance(autoencoder, DenoisingTranscriptomeAutoencoder)


def test_load_stage_a_raises_on_same_width_different_gene_order():
    with tempfile.TemporaryDirectory() as tmp:
        _save_stage_a_checkpoint(tmp, n_genes=3, gene_names=["a", "b", "c"])
        with pytest.raises(ValueError, match="ORDERED gene identities differ"):
            _load_stage_a(tmp, gene_names=["a", "c", "b"])


def test_load_stage_b_model_raises_when_gene_names_json_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        _save_stage_a_checkpoint(tmp, n_genes=10)
        (Path(tmp) / "gene_names.json").unlink()
        saved_model_config = {
            "stage_a_checkpoint_dir": tmp,
            "params": {"n_heads": 1, "n_layers": 1, "hidden_dim": 8, "max_neighbors": 4, "coord_scale": 100.0},
        }
        with pytest.raises(ValueError, match="gene_names.json"):
            _load_stage_b_model(saved_model_config, gene_names=[f"g{i}" for i in range(10)], device=torch.device("cpu"))
