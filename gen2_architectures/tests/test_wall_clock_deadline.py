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
