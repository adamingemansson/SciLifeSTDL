import numpy as np
import pytest
import csv
import json

from gen3_multiscale.scripts.analyze_mk_stain_domain_shift import (
    FEATURE_NAMES,
    _write_tsv,
    patch_stain_features,
    robust_reference,
    stain_distance,
)
from gen3_multiscale.scripts.audit_mk_external_validation import audit
from gen3_multiscale.scripts.analyze_mk_stain_confounders import (
    _organ_pair_summary,
    _residualize_categories,
)
from gen3_multiscale.scripts.summarize_mk_wae_best_individual_draws import summarize


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


def test_stain_confounder_helpers_preserve_within_organ_direction():
    values = np.asarray([1.0, 2.0, 10.0, 12.0])
    residual = _residualize_categories(values, ["a", "a", "b", "b"])
    assert np.allclose(residual, [-0.5, 0.5, -1.0, 1.0])
    pairs = _organ_pair_summary([
        {"organ": "a", "sample_id": "a1", "stain_distance": 1.0, "all_gene_pcc": 0.1},
        {"organ": "a", "sample_id": "a2", "stain_distance": 2.0, "all_gene_pcc": 0.2},
        {"organ": "b", "sample_id": "b1", "stain_distance": 1.0, "all_gene_pcc": 0.3},
        {"organ": "b", "sample_id": "b2", "stain_distance": 2.0, "all_gene_pcc": 0.1},
    ])
    assert pairs["n_concordant_pairs"] == 1
    assert pairs["n_comparable_organ_pairs"] == 2


def test_best_individual_draw_summary_compares_to_exact_deterministic_entry(tmp_path):
    source = tmp_path / "diversity"
    source.mkdir()
    (source / "report.json").write_text(json.dumps({
        "arm": "wae", "n_slides": 2,
    }))
    _write_tsv(source / "per_draw.tsv", [
        {"sample_id": "a", "organ": "x", "draw": 1, "draw_vs_target_pcc": 0.4, "draw_vs_target_rmse": 0.6},
        {"sample_id": "a", "organ": "x", "draw": 2, "draw_vs_target_pcc": 0.6, "draw_vs_target_rmse": 0.4},
        {"sample_id": "b", "organ": "y", "draw": 1, "draw_vs_target_pcc": 0.2, "draw_vs_target_rmse": 0.8},
        {"sample_id": "b", "organ": "y", "draw": 2, "draw_vs_target_pcc": 0.4, "draw_vs_target_rmse": 0.6},
    ])
    _write_tsv(source / "ensemble_convergence.tsv", [
        {"sample_id": "a", "n_draws": 2, "sampled_value_pcc_vs_target": 0.5, "sampled_value_rmse_vs_target": 0.5, "sampled_value_rmse_vs_deterministic": 0.0},
        {"sample_id": "b", "n_draws": 2, "sampled_value_pcc_vs_target": 0.3, "sampled_value_rmse_vs_target": 0.7, "sampled_value_rmse_vs_deterministic": 0.0},
    ])
    outputs = summarize(diversity_root=str(source), output_dir=str(tmp_path / "out"))
    result = json.loads(outputs["summary"].read_text())
    assert result["deterministic_macro_pcc"] == pytest.approx(0.4)
    assert result["best_single_global_draw_by_pcc"]["draw"] == 2
    assert result["best_global_draw_pcc_improvement"] == pytest.approx(0.1)
