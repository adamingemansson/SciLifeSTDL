import numpy as np
import pytest
import torch

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from src.training.train import _prepare_images_for_split


def test_image_availability_modes_are_explicit_and_zero_missing_rows():
    images = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    context = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    query = ~context

    c, q, ca, qa = _prepare_images_for_split(images, context, query, mode="target_zero")
    assert ca.all() and not qa.any()
    assert torch.count_nonzero(q) == 0
    assert torch.count_nonzero(c) > 0

    c, q, ca, qa = _prepare_images_for_split(images, context, query, mode="all_zero")
    assert not ca.any() and not qa.any()
    assert torch.count_nonzero(c) == 0 and torch.count_nonzero(q) == 0


def test_shuffled_query_images_are_deterministic_and_not_marked_missing():
    images = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    context = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    query = ~context
    one = _prepare_images_for_split(images, context, query, mode="shuffled", seed=3)
    two = _prepare_images_for_split(images, context, query, mode="shuffled", seed=3)
    assert torch.equal(one[1], two[1])
    assert one[3].all()
    assert sorted(map(tuple, one[1].numpy())) == sorted(map(tuple, images[query]))
