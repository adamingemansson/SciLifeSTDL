import pytest

from gen3_multiscale.training.early_stopping import (
    EarlyStoppingConfig, early_stopping_status, resolve_early_stopping_config,
)


def test_resolve_returns_none_when_absent_or_falsy():
    assert resolve_early_stopping_config({}) is None
    assert resolve_early_stopping_config({"early_stopping": None}) is None
    assert resolve_early_stopping_config({"early_stopping": {}}) is None


def test_resolve_applies_documented_defaults():
    cfg = resolve_early_stopping_config({"early_stopping": {"patience_validations": 8}})
    assert cfg == EarlyStoppingConfig(
        monitor="validation_total", mode="min", patience_validations=8, min_delta=0.0,
    )


def test_resolve_reads_every_field():
    cfg = resolve_early_stopping_config({
        "early_stopping": {
            "monitor": "validation_total", "mode": "max",
            "patience_validations": 3, "min_delta": 0.001,
        },
    })
    assert cfg == EarlyStoppingConfig(
        monitor="validation_total", mode="max", patience_validations=3, min_delta=0.001,
    )


@pytest.mark.parametrize("bad", [
    {"early_stopping": "not_a_mapping"},
    {"early_stopping": {"monitor": "unknown_field"}},
    {"early_stopping": {"mode": "sideways"}},
    {"early_stopping": {"patience_validations": 0}},
    {"early_stopping": {"patience_validations": -1}},
    {"early_stopping": {"min_delta": -0.1}},
    {"early_stopping": {"min_delta": float("nan")}},
    {"early_stopping": {"min_delta": float("inf")}},
])
def test_resolve_rejects_invalid_config(bad):
    with pytest.raises(ValueError):
        resolve_early_stopping_config(bad)


_MIN_CFG = EarlyStoppingConfig(monitor="validation_total", mode="min", patience_validations=2, min_delta=0.0)
_MAX_CFG = EarlyStoppingConfig(monitor="validation_total", mode="max", patience_validations=2, min_delta=0.0)


def test_empty_history_never_stops():
    status = early_stopping_status([], _MIN_CFG)
    assert status == {
        "best_step": None, "best_value": None,
        "n_since_improvement": 0, "patience_remaining": 2, "should_stop": False,
    }


def test_min_mode_tracks_best_and_resets_patience_on_improvement():
    history = [
        {"step": 1, "total": 1.0},  # best
        {"step": 2, "total": 0.9},  # improves -> resets
        {"step": 3, "total": 0.95},  # worse -> n_since=1
    ]
    status = early_stopping_status(history, _MIN_CFG)
    assert status["best_step"] == 2
    assert status["best_value"] == 0.9
    assert status["n_since_improvement"] == 1
    assert status["patience_remaining"] == 1
    assert status["should_stop"] is False


def test_min_mode_stops_once_patience_is_exhausted():
    history = [
        {"step": 1, "total": 1.0},
        {"step": 2, "total": 1.0},  # not strictly less -> n_since=1
        {"step": 3, "total": 1.0},  # n_since=2 -> patience_validations=2 -> stop
    ]
    status = early_stopping_status(history, _MIN_CFG)
    assert status["best_step"] == 1
    assert status["n_since_improvement"] == 2
    assert status["patience_remaining"] == 0
    assert status["should_stop"] is True


def test_min_delta_requires_a_real_margin_not_just_any_decrease():
    cfg = EarlyStoppingConfig(monitor="validation_total", mode="min", patience_validations=1, min_delta=0.01)
    history = [
        {"step": 1, "total": 1.0},
        {"step": 2, "total": 0.995},  # decreased, but by less than min_delta -> not an improvement
    ]
    status = early_stopping_status(history, cfg)
    assert status["best_step"] == 1
    assert status["n_since_improvement"] == 1
    assert status["should_stop"] is True


def test_max_mode_is_the_mirror_image_of_min_mode():
    history = [
        {"step": 1, "total": 0.1},
        {"step": 2, "total": 0.2},  # higher is better in max mode -> improves
        {"step": 3, "total": 0.15},  # worse -> n_since=1
    ]
    status = early_stopping_status(history, _MAX_CFG)
    assert status["best_step"] == 2
    assert status["best_value"] == 0.2
    assert status["n_since_improvement"] == 1


def test_unknown_monitor_field_raises_with_the_offending_entry():
    with pytest.raises(ValueError, match="missing monitored field"):
        early_stopping_status([{"step": 1}], _MIN_CFG)


def test_status_is_a_pure_function_of_the_full_history_recomputation_is_resume_safe():
    """The core design guarantee: calling this twice on the SAME history
    (as a resumed run does after reloading validation_history.json) gives
    an identical result -- no separate mutable counter to desynchronize."""
    history = [{"step": s, "total": 1.0 / s} for s in range(1, 6)]
    first = early_stopping_status(history, _MIN_CFG)
    second = early_stopping_status(list(history), _MIN_CFG)
    assert first == second
