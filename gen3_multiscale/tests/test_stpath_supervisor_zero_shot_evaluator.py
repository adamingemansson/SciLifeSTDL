import numpy as np
import pytest

from gen3_multiscale.evaluation.stpath_supervisor_zero_shot_evaluator import (
    _stpath_task_positions,
)


def test_he_to_st_masks_expression_for_every_slide_row_and_scores_fixed_query():
    context, query, score = _stpath_task_positions(
        "he_to_st",
        n_spots=7,
        context_pos=np.asarray([0, 2, 4, 6]),
        query_pos=np.asarray([1, 3, 5]),
    )
    np.testing.assert_array_equal(context, np.empty(0, dtype=np.int64))
    np.testing.assert_array_equal(query, np.arange(7))
    np.testing.assert_array_equal(score, [1, 3, 5])


def test_he_plus_st_to_st_exposes_only_context_expression():
    context, query, score = _stpath_task_positions(
        "he_plus_st_to_st",
        n_spots=7,
        context_pos=np.asarray([0, 2, 4, 6]),
        query_pos=np.asarray([1, 3, 5]),
    )
    np.testing.assert_array_equal(context, [0, 2, 4, 6])
    np.testing.assert_array_equal(query, [1, 3, 5])
    np.testing.assert_array_equal(score, np.arange(3))


def test_stpath_supervisor_task_rejects_unknown_mode():
    with pytest.raises(ValueError, match="task must be"):
        _stpath_task_positions(
            "unknown",
            n_spots=2,
            context_pos=np.asarray([0]),
            query_pos=np.asarray([1]),
        )
