import numpy as np
import pytest

from gen3_multiscale.evaluation.stpath_zero_shot_evaluator import (
    scatter_supported_genes,
    stpath_log1p_to_normalized_log1p,
)


def test_stpath_log1p_to_normalized_log1p_matches_normalize_total_contract():
    raw_counts = np.asarray([[1.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    transformed = stpath_log1p_to_normalized_log1p(np.log1p(raw_counts), target_sum=100.0)
    np.testing.assert_allclose(transformed[0], np.log1p([25.0, 75.0]), rtol=1e-6)
    np.testing.assert_array_equal(transformed[1], np.zeros(2, dtype=np.float32))


def test_stpath_log1p_to_normalized_log1p_clamps_impossible_negative_counts():
    transformed = stpath_log1p_to_normalized_log1p(
        np.asarray([[-3.0, np.log1p(2.0)]], dtype=np.float32), target_sum=10.0,
    )
    np.testing.assert_allclose(transformed, [[0.0, np.log1p(10.0)]], rtol=1e-6)


def test_scatter_supported_genes_preserves_manifest_order_and_zeros_unsupported():
    supported = np.asarray([[10.0, 20.0]], dtype=np.float32)
    full = scatter_supported_genes(supported, [2, 0], 4)
    np.testing.assert_array_equal(full, [[20.0, 0.0, 10.0, 0.0]])


def test_scatter_supported_genes_rejects_duplicate_positions():
    with pytest.raises(ValueError, match="duplicate"):
        scatter_supported_genes(np.ones((1, 2), dtype=np.float32), [1, 1], 3)
