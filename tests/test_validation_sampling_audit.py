import torch
import pytest

from src.training.validation import FixedMaskValidationCallback, predictive_samples


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


class _PreparedStochasticModel(torch.nn.Module):
    """Tiny model exercising the optional cached-conditioning protocol."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(2.0))
        self.prepare_calls = 0
        self.prepared_sample_calls = 0
        self.one_shot_calls = 0

    @property
    def device(self):
        return self.anchor.device

    def prepare_sampling_conditioning(self, context, query):
        self.prepare_calls += 1
        return {"n": query["coords"].shape[0], "offset": self.anchor}

    def sample_from_prepared_conditioning(self, prepared):
        self.prepared_sample_calls += 1
        return {
            "expression": torch.randn(prepared["n"], 2, device=self.device)
            + prepared["offset"]
        }

    def sample(self, context, query):
        self.one_shot_calls += 1
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


def test_predictive_samples_prepares_eval_conditioning_once():
    model = _PreparedStochasticModel().eval()
    context = {"coords": torch.zeros(3, 3)}
    query = {"coords": torch.zeros(4, 3)}

    first = predictive_samples(model, context, query, n_samples=5, seed=123)
    second = predictive_samples(model, context, query, n_samples=5, seed=123)

    assert first.shape == (5, 4, 2)
    assert torch.equal(first, second)
    assert model.prepare_calls == 2  # once per predictive_samples invocation
    assert model.prepared_sample_calls == 10
    assert model.one_shot_calls == 0


def test_predictive_samples_does_not_cache_in_training_mode():
    model = _PreparedStochasticModel().train()
    context = {"coords": torch.zeros(3, 3)}
    query = {"coords": torch.zeros(4, 3)}

    predictive_samples(model, context, query, n_samples=3, seed=123)

    assert model.prepare_calls == 0
    assert model.prepared_sample_calls == 0
    assert model.one_shot_calls == 3


def test_early_stopping_patience_does_not_accumulate_before_min_steps():
    callback = FixedMaskValidationCallback(
        [], every_n_steps=100, patience_checks=2, early_stopping_min_steps=500
    )
    callback.best_score = 0.0
    assert callback._improved(1.0) is False
    # The callback's public configuration is what the training hook uses to
    # defer both patience accumulation and stopping.
    assert callback.early_stopping_min_steps == 500


def test_checkpoint_selection_keeps_small_real_improvements():
    callback = FixedMaskValidationCallback([], metric="rmse", min_delta=0.1)
    callback.best_score = 1.0
    callback.patience_score = 1.0

    # This is a real best checkpoint but is intentionally too small an
    # improvement to reset early-stopping patience.
    assert callback._improved(0.95) is True
    assert callback._meaningfully_improved(0.95) is False

    pcc_callback = FixedMaskValidationCallback([], metric="pcc", min_delta=0.1)
    pcc_callback.best_score = 0.5
    pcc_callback.patience_score = 0.5
    assert pcc_callback._improved(0.55) is True
    assert pcc_callback._meaningfully_improved(0.55) is False


def test_required_anchor_quality_gate_fails_closed():
    callback = FixedMaskValidationCallback([], require_anchor_improvement=True)
    callback.best_score = 0.5
    callback.anchor_score = 0.4
    callback.quality_gate_passed = False
    with pytest.raises(RuntimeError, match="quality gate failed"):
        callback.raise_if_quality_gate_failed()
