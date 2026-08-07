from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from gen3_multiscale.scripts.prepare_gen5_suite import (
    _pairs as prepare_pairs,
    _verified_autoencoder_report,
)
from gen3_multiscale.scripts.resolve_gen45_config import resolve_gen45_config


def test_resolver_can_bind_train_derived_evaluation_panels(tmp_path):
    base = Path(__file__).resolve().parents[1] / "configs" / "gen5" / "gen5a.yaml"
    output = tmp_path / "resolved.yaml"
    panel_path = tmp_path / "train_gene_panels.json"
    resolved = resolve_gen45_config(
        str(base),
        str(output),
        manifest=str(tmp_path / "manifest.json"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        fingerprints={
            "uni2_checkpoint": "uni2.pt",
            "uni2_revision": "a" * 40,
            "uni2_package_version": "1.0",
            "uni2_preprocessing_spec": "spec",
            "expression_autoencoder_checkpoint": "autoencoder.pt",
            "gen4_conditioner_checkpoint": "conditioner/best",
        },
        data_overrides={},
        evaluation_overrides={"train_gene_panel_artifact": str(panel_path)},
    )
    assert resolved["evaluation"]["train_gene_panel_artifact"] == str(panel_path)
    assert yaml.safe_load(output.read_text())["evaluation"]["train_gene_panel_artifact"] == str(panel_path)


def test_prepare_gen5_conditioner_pairs_require_every_primary_arm():
    values = [
        "gen5c=/checkpoints/c/best",
        "gen5b=/checkpoints/b/best",
        "gen5d=/checkpoints/d/best",
        "gen5e=/checkpoints/e/best",
    ]
    assert prepare_pairs(
        values, expected={"gen5c", "gen5b", "gen5d", "gen5e"}
    )["gen5e"] == "/checkpoints/e/best"

    with pytest.raises(ValueError, match="missing --conditioner"):
        prepare_pairs(values[:-1], expected={"gen5c", "gen5b", "gen5d", "gen5e"})


def test_prepare_gen5_binds_autoencoder_report_to_checkpoint_and_split(tmp_path):
    checkpoint = tmp_path / "shared_autoencoder.pt"
    checkpoint.write_bytes(b"weights")
    report = {
        "kind": "gen5_expression_autoencoder_training_report",
        "checkpoint_sha256": "checkpoint-hash",
        "dataset_manifest_fingerprint": "manifest-hash",
        "train_gene_panel_artifact": {"artifact_sha256": "panels-hash"},
        "validation_reconstruction": {
            "sample_ids": ["v2", "v1"],
            "rmse": 0.2,
            "pcc_mean": 0.8,
        },
    }
    Path(f"{checkpoint}.report.json").write_text(json.dumps(report))
    loaded = _verified_autoencoder_report(
        checkpoint,
        checkpoint_sha256="checkpoint-hash",
        manifest_fingerprint="manifest-hash",
        validation_sample_ids=["v1", "v2"],
        train_panel_sha256="panels-hash",
    )
    assert loaded["validation_reconstruction"]["pcc_mean"] == 0.8

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        _verified_autoencoder_report(
            checkpoint,
            checkpoint_sha256="different-hash",
            manifest_fingerprint="manifest-hash",
            validation_sample_ids=["v1", "v2"],
            train_panel_sha256="panels-hash",
        )
