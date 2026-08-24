import numpy as np
import pytest

from gen3_multiscale.scripts.analyze_conditional_wae_predictive_diversity import (
    _column_diagnostics,
    _draw_pair_records,
    _parse_positive_ints,
    _pearson_1d,
)


def test_parse_ensemble_sizes_filters_and_includes_final_draw():
    assert _parse_positive_ints("1,2,4,8,64", maximum=16) == [1, 2, 4, 8, 16]


def test_pairwise_draw_records_detect_identical_and_different_draws():
    draws = np.asarray([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
    ])
    records = _draw_pair_records(draws)
    assert len(records) == 3
    assert records[0]["sampled_value_pcc"] == pytest.approx(1.0)
    assert records[0]["sampled_value_rmse"] == pytest.approx(0.0)
    assert records[1]["sampled_value_pcc"] == pytest.approx(-1.0)


def test_column_diagnostics_identifies_helpful_mean_shift():
    target = np.asarray([
        [0.0, 2.0],
        [1.0, 1.0],
        [2.0, 0.0],
    ], dtype=np.float32)
    point = np.zeros_like(target)
    predictive_mean = target.copy()
    predictive_std = np.full_like(target, 0.25)
    result = _column_diagnostics(point, predictive_mean, predictive_std, target)
    assert np.allclose(result["predictive_mean_rmse"], 0.0)
    assert np.all(result["point_rmse"] > 0)
    assert np.allclose(result["predictive_std_mean"], 0.25)
    assert np.allclose(result["mean_shift_residual_pcc"], 1.0)


def test_pearson_rejects_mismatched_vectors():
    with pytest.raises(ValueError):
        _pearson_1d(np.zeros(2), np.zeros(3))
