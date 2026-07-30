from types import SimpleNamespace

import numpy as np
import pytest

from gen3_multiscale.evaluation import train_gene_panels as panels_module


def _manifest():
    genes = [f"g{i:03d}" for i in range(220)]
    return {
        "gene_panel": genes,
        "train_sample_ids": ["train_a", "train_b"],
        "validation_sample_ids": ["val"],
        "test_sample_ids": ["test"],
        "build_args": {"expression_transform": "log1p", "expression_target_sum": 10000},
        "samples": {},
    }


def test_train_panels_are_deterministic_nested_and_use_only_training_samples(monkeypatch):
    manifest = _manifest()
    calls = []

    def fake_load(_manifest, sample_id):
        calls.append(sample_id)
        offset = 0 if sample_id == "train_a" else 1
        x = np.arange(4 * 220, dtype=np.float64).reshape(4, 220) + offset
        return SimpleNamespace(X=x, var_names=manifest["gene_panel"]), None, None

    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", lambda m, s: fake_load(m, s)[0])
    first = panels_module.build_train_derived_gene_panels(manifest)
    second = panels_module.build_train_derived_gene_panels(manifest)
    assert calls == ["train_a", "train_b", "train_a", "train_b"]
    assert first == second
    top50 = first["panels"]["train_log1p_variance_top50"]
    top200 = first["panels"]["train_log1p_variance_top200"]
    assert top200[:50] == top50
    assert len(top50) == 50 and len(top200) == 200


def test_train_panel_artifact_validation_rejects_tampering(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        panels_module, "load_expression_for_model_target_space",
        lambda _manifest, _sid: (
            SimpleNamespace(X=np.arange(660, dtype=float).reshape(3, 220), var_names=manifest["gene_panel"])
        ),
    )
    artifact = panels_module.build_train_derived_gene_panels(manifest)
    artifact["panels"]["train_log1p_variance_top50"][0] = "tampered"
    with pytest.raises(ValueError, match="SHA256"):
        panels_module.validate_train_derived_gene_panels(artifact, manifest)


def test_train_panel_artifact_is_bound_to_manifest(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr(
        panels_module, "load_expression_for_model_target_space",
        lambda _manifest, _sid: (
            SimpleNamespace(X=np.arange(660, dtype=float).reshape(3, 220), var_names=manifest["gene_panel"])
        ),
    )
    artifact = panels_module.build_train_derived_gene_panels(manifest)
    changed = dict(manifest)
    changed["test_sample_ids"] = ["different_test"]
    with pytest.raises(ValueError, match="dataset_manifest_fingerprint"):
        panels_module.validate_train_derived_gene_panels(artifact, changed)
