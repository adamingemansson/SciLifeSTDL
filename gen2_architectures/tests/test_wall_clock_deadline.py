import time

import pytest
from omegaconf import OmegaConf

from gen2_architectures.training.data_prep import resolve_wall_clock_deadline


def test_returns_none_when_unset():
    cfg = OmegaConf.create({"training": {}})
    assert resolve_wall_clock_deadline(cfg) is None


def test_returns_a_future_deadline_when_set():
    cfg = OmegaConf.create({"training": {"max_wall_clock_hours": 1.0}})
    before = time.monotonic()
    deadline = resolve_wall_clock_deadline(cfg)
    after = time.monotonic()
    assert before + 3600 <= deadline <= after + 3600


def test_rejects_non_positive_hours():
    cfg = OmegaConf.create({"training": {"max_wall_clock_hours": 0}})
    with pytest.raises(ValueError, match="positive"):
        resolve_wall_clock_deadline(cfg)

    cfg = OmegaConf.create({"training": {"max_wall_clock_hours": -2}})
    with pytest.raises(ValueError, match="positive"):
        resolve_wall_clock_deadline(cfg)


def test_already_elapsed_seconds_shrinks_the_remaining_deadline():
    """GPT-audit-flagged bug (2026-07-27, second-pass re-audit): every
    resume used to grant itself a FRESH max_wall_clock_hours budget
    instead of continuing the same overall budget an uninterrupted run
    would have had."""
    cfg = OmegaConf.create({"training": {"max_wall_clock_hours": 1.0}})  # 3600s budget
    before = time.monotonic()
    deadline = resolve_wall_clock_deadline(cfg, already_elapsed_seconds=3000.0)
    after = time.monotonic()
    # only 600s of budget should remain
    assert before + 600 <= deadline <= after + 600


def test_already_elapsed_seconds_beyond_the_budget_gives_an_immediate_deadline():
    cfg = OmegaConf.create({"training": {"max_wall_clock_hours": 1.0}})
    before = time.monotonic()
    deadline = resolve_wall_clock_deadline(cfg, already_elapsed_seconds=999999.0)
    assert deadline >= before  # never negative/past -- clamped to "right now"
