import numpy as np
import torch

from scripts.pretrain_expression_autoencoder import _residual_for_mask
from src.models.spatial_baselines import harmonic_interpolate


def test_pretraining_target_is_harmonic_residual_not_absolute_expression():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    expression = np.array([[1, 2], [3, 4], [10, 20]], dtype=np.float32)
    context = np.array([True, True, False])
    query = ~context
    residual = _residual_for_mask(
        coords, expression, context, query, harmonic_k=2, harmonic_ridge=1e-4
    )
    anchor = harmonic_interpolate(
        torch.from_numpy(coords[context]), torch.from_numpy(expression[context]),
        torch.from_numpy(coords[query]), k=2, ridge=1e-4,
    ).numpy()
    assert np.allclose(residual, expression[query] - anchor)
    assert not np.allclose(residual, expression[query])
