import torch

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
