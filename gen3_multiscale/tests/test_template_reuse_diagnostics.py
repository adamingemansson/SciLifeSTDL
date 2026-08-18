import numpy as np
import pytest

from gen3_multiscale.evaluation.template_reuse_diagnostics import (
    effective_rank,
    multiscale_pcc_per_gene,
    selected_template_diagnostics,
    template_retrieval,
    variance_diagnostics,
)


def _coords(n: int) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=float), np.zeros(n, dtype=float)])


def test_multiscale_pcc_is_exact_for_identical_fields():
    rng = np.random.default_rng(4)
    target = rng.normal(size=(20, 5)).astype(np.float32)
    result = multiscale_pcc_per_gene(target, target, _coords(20), k_neighbors=2)
    np.testing.assert_allclose(result["exact"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["blur1"], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["blur2"], 1.0, atol=1e-12)


def test_multiscale_pcc_exposes_local_displacement():
    target = np.zeros((31, 1), dtype=np.float32)
    predicted = np.zeros_like(target)
    target[14:17, 0] = 1.0
    predicted[16:19, 0] = 1.0
    result = multiscale_pcc_per_gene(
        predicted, target, _coords(31), k_neighbors=2, chunk_size=1,
    )
    assert result["blur2"][0] > result["exact"][0]


def test_variance_ratio_detects_amplitude_shrinkage():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(40, 4))
    result = variance_diagnostics(target * 0.2, target)
    np.testing.assert_allclose(result["predicted_to_target_std_ratio"], 0.2)


def test_template_retrieval_is_perfect_for_matching_maps():
    rng = np.random.default_rng(6)
    target = rng.normal(size=(80, 12))
    result = template_retrieval(target, target)
    assert result["same_gene_top1_fraction"] == pytest.approx(1.0)
    assert result["unique_top_target_fraction"] == pytest.approx(1.0)
    assert result["maximum_target_reuse_count"] == 1


def test_effective_rank_and_retrieval_detect_reused_templates():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(100, 20))
    predicted = np.repeat(target[:, :1], 20, axis=1)
    true_rank = effective_rank(target)["entropy_effective_rank"]
    pred_rank = effective_rank(predicted)["entropy_effective_rank"]
    retrieval = template_retrieval(predicted, target)
    assert pred_rank < true_rank * 0.2
    assert retrieval["unique_top_target_fraction"] == pytest.approx(1 / 20)
    assert retrieval["maximum_target_reuse_count"] == 20


def test_selected_template_diagnostics_returns_auditable_indices():
    rng = np.random.default_rng(8)
    target = rng.normal(size=(50, 10)) * np.arange(1, 11)[None, :]
    selected, result = selected_template_diagnostics(target, target, max_genes=4)
    expected = np.argsort(-target.var(axis=0), kind="mergesort")[:4]
    assert selected.tolist() == expected.tolist()
    assert result["n_selected_genes"] == 4
    assert result["same_gene_top1_fraction"] == pytest.approx(1.0)
    assert result["entropy_effective_rank_ratio"] == pytest.approx(1.0)
