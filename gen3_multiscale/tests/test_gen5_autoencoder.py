"""Expression autoencoder tests -- GEN5_CONTRACT.md section 7, gates 1/2/8."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gen3_multiscale.gen5.autoencoder import (
    ExpressionAutoencoder, evaluate_autoencoder_reconstruction, load_expression_autoencoder_checkpoint,
    save_expression_autoencoder_checkpoint, verify_expression_autoencoder_gene_names,
)
from gen3_multiscale.gen5.autoencoder_training import train_expression_autoencoder
from gen3_multiscale.tests._gen5_fixtures import synthetic_expression, tiny_autoencoder


def test_autoencoder_trains_and_reduces_loss():
    train_expression = synthetic_expression(n_rows=64, n_genes=6)
    gene_names = [f"g{i}" for i in range(6)]
    _autoencoder, report = train_expression_autoencoder(
        train_expression, gene_names, latent_dim=8, hidden_dim=16, n_epochs=20, batch_size=16, device="cpu",
    )
    assert report.loss_per_epoch[-1] < report.loss_per_epoch[0]


def test_autoencoder_overfits_a_single_batch():
    """Gate 2: a single small batch, trained long enough, should be
    reconstructed almost exactly -- a real capacity sanity check."""
    train_expression = synthetic_expression(n_rows=8, n_genes=6)
    gene_names = [f"g{i}" for i in range(6)]
    autoencoder, _report = train_expression_autoencoder(
        train_expression, gene_names, latent_dim=8, hidden_dim=64, n_epochs=300, batch_size=8, lr=1e-2, device="cpu",
    )
    result = evaluate_autoencoder_reconstruction(autoencoder, train_expression, gene_names)
    assert result.rmse < 0.1


def test_evaluate_autoencoder_reconstruction_reports_finite_metrics():
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    expression = synthetic_expression(n_rows=20, n_genes=6)
    gene_names = [f"g{i}" for i in range(6)]
    report = evaluate_autoencoder_reconstruction(autoencoder, expression, gene_names)
    assert report.n_rows == 20 and report.n_genes == 6
    assert np.isfinite(report.rmse)


def test_verify_gene_names_rejects_reordered_panel():
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    reordered = [f"g{i}" for i in range(6)][::-1]
    with pytest.raises(ValueError, match="does not match the panel"):
        verify_expression_autoencoder_gene_names(autoencoder, reordered)


def test_checkpoint_round_trip(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p",
        code_identity="commit-abc",
    )
    loaded, payload = load_expression_autoencoder_checkpoint(
        path, dataset_manifest_fingerprint="fp123", code_identity="commit-abc",
    )
    assert loaded.gene_names == autoencoder.gene_names
    for (name, p1), p2 in zip(autoencoder.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(p1, p2)
    assert payload["preprocessing_spec"] == "normalize_log1p"


def test_checkpoint_dataset_fingerprint_mismatch_fails_closed(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p", code_identity="commit-abc",
    )
    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        load_expression_autoencoder_checkpoint(path, dataset_manifest_fingerprint="a-different-fingerprint")


def test_checkpoint_code_identity_mismatch_fails_closed_unless_allowed(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p", code_identity="commit-abc",
    )
    with pytest.raises(ValueError, match="code_identity"):
        load_expression_autoencoder_checkpoint(path, code_identity="commit-different")
    # explicit override succeeds
    loaded, _payload = load_expression_autoencoder_checkpoint(path, code_identity="commit-different", allow_code_drift=True)
    assert loaded.gene_names == autoencoder.gene_names


def test_checkpoint_gene_names_hash_tamper_fails_closed(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p", code_identity="commit-abc",
    )
    payload = torch.load(path, map_location="cpu")
    payload["gene_names"] = list(payload["gene_names"])[::-1]  # tamper: reorder without updating the hash
    torch.save(payload, path)
    with pytest.raises(ValueError, match="does not match its own saved gene_names"):
        load_expression_autoencoder_checkpoint(path)


def test_checkpoint_state_dict_shape_mismatch_fails_closed(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p", code_identity="commit-abc",
    )
    payload = torch.load(path, map_location="cpu")
    payload["latent_dim"] = 999  # reconstructed model's layers won't match the saved tensors' real shapes
    torch.save(payload, path)
    with pytest.raises(ValueError, match="shape mismatches"):
        load_expression_autoencoder_checkpoint(path)


def test_checkpoint_gene_names_count_mismatch_fails_closed(tmp_path):
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=8)
    path = tmp_path / "ae.pt"
    save_expression_autoencoder_checkpoint(
        autoencoder, path, dataset_manifest_fingerprint="fp123", preprocessing_spec="normalize_log1p", code_identity="commit-abc",
    )
    payload = torch.load(path, map_location="cpu")
    payload["n_genes"] = 99  # inconsistent with the saved (unchanged) 6-entry gene_names list
    torch.save(payload, path)
    with pytest.raises(ValueError, match="gene_names has .* entries"):
        load_expression_autoencoder_checkpoint(path)


def test_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_expression_autoencoder_checkpoint(tmp_path / "does_not_exist.pt")


def test_no_top_hvg_restriction_full_gene_panel_is_used():
    """The encoder/decoder always operate over the complete gene panel --
    there is no code path that selects a subset."""
    n_genes = 30
    autoencoder = ExpressionAutoencoder(n_genes, [f"g{i}" for i in range(n_genes)], latent_dim=8, hidden_dim=16)
    assert autoencoder.encoder.net[0].in_features == n_genes
    assert autoencoder.decoder.net[-1].out_features == n_genes
