import numpy as np
import pytest
import csv

from gen3_multiscale.scripts.analyze_mk_stain_domain_shift import (
    FEATURE_NAMES,
    _write_tsv,
    patch_stain_features,
    robust_reference,
    stain_distance,
)
from gen3_multiscale.scripts.audit_mk_external_validation import audit


def test_patch_stain_features_are_finite_and_color_sensitive():
    patches = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    patches[0, ..., 0] = 255
    patches[1, ..., 2] = 255
    features = patch_stain_features(patches, batch_size=1)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert not np.allclose(features[0], features[1])


def test_robust_stain_distance_is_zero_at_train_center():
    train = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    center, scale = robust_reference(train)
    assert stain_distance(center, center, scale) == 0.0
    assert stain_distance(center + scale, center, scale) == pytest.approx(1.0)


def test_stain_writer_supports_validation_only_columns(tmp_path):
    path = tmp_path / "rows.tsv"
    _write_tsv(path, [
        {"split": "train", "sample_id": "a"},
        {"split": "validation", "sample_id": "b", "all_gene_pcc": 0.2},
    ])
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["all_gene_pcc"] == ""
    assert rows[1]["all_gene_pcc"] == "0.2"


def test_external_audit_passes_only_complete_disjoint_cohort():
    contract = {
        "external_validation": {
            "accepted_target_spaces": ["normalize_total_then_log1p"],
            "accepted_spatial_technologies": ["Visium"],
            "require_gene_coverage_fraction": 0.95,
        }
    }
    candidate = {
        "dataset_id": "external-v1",
        "sample_ids": ["a", "b"],
        "patient_ids": ["p1", "p2"],
        "organs": ["Bowel"],
        "spatial_technology": "Visium",
        "target_space": "normalize_total_then_log1p",
        "gene_coverage_fraction": 0.98,
        "patient_disjoint_from_training": True,
        "used_for_model_selection": False,
        "pretraining_overlap_status": "audited_none",
        "data_provenance": "doi:example",
    }
    assert audit(candidate, contract)["ready"] is True
    candidate["used_for_model_selection"] = True
    result = audit(candidate, contract)
    assert result["ready"] is False
    assert "used for model selection" in " ".join(result["failures"])
