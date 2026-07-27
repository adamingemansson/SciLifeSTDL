import torch

from gen2_architectures.training.data_prep import is_finite_update


def test_finite_loss_and_grad_norm_is_true():
    assert is_finite_update(torch.tensor(0.5), torch.tensor(0.3)) is True


def test_nan_loss_is_false():
    assert is_finite_update(torch.tensor(float("nan")), torch.tensor(0.3)) is False


def test_nan_grad_norm_is_false():
    assert is_finite_update(torch.tensor(0.5), torch.tensor(float("nan"))) is False


def test_inf_loss_or_grad_norm_is_false():
    assert is_finite_update(torch.tensor(float("inf")), torch.tensor(0.3)) is False
    assert is_finite_update(torch.tensor(0.5), torch.tensor(float("inf"))) is False
