import numpy as np

from src.evaluation.metrics import pearson_per_gene


def test_constant_prediction_is_scored_not_dropped():
    true = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    pred = np.asarray([[3.0, 0.0], [3.0, 2.0], [3.0, 4.0]])
    pcc = pearson_per_gene(pred, true)
    assert pcc[0] == 0.0
    assert np.isnan(pcc[1])
