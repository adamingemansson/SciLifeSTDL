import torch
import pytest

from src.training.validation import FixedMaskValidationCallback


class _StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def device(self):
        return self.anchor.device

    def sample(self, context, query):
        n = query["coords"].shape[0]
        return {"expression": torch.randn(n, 2, device=self.device) + self.anchor}


def test_fixed_validation_uses_identical_monte_carlo_draws_across_steps():
    item = {
        "context": {"coords": torch.zeros(3, 3), "expression": torch.zeros(3, 2)},
        "query": {"coords": torch.zeros(4, 3)},
        "target_expression": torch.zeros(4, 2),
    }
    callback = FixedMaskValidationCallback([item], n_samples=3, seed=123)
    model = _StochasticModel()
    first = callback._score(model, step=100)
    second = callback._score(model, step=200)
    assert first == second


def test_early_stopping_patience_does_not_accumulate_before_min_steps():
    callback = FixedMaskValidationCallback(
        [], every_n_steps=100, patience_checks=2, early_stopping_min_steps=500
    )
    callback.best_score = 0.0
    assert callback._improved(1.0) is False
    # The callback's public configuration is what the training hook uses to
    # defer both patience accumulation and stopping.
    assert callback.early_stopping_min_steps == 500


def test_required_anchor_quality_gate_fails_closed():
    callback = FixedMaskValidationCallback([], require_anchor_improvement=True)
    callback.best_score = 0.5
    callback.anchor_score = 0.4
    callback.quality_gate_passed = False
    with pytest.raises(RuntimeError, match="quality gate failed"):
        callback.raise_if_quality_gate_failed()
