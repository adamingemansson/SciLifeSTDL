"""Resumable early stopping for the shared Gen3/Gen4/Gen5/Gen6 trainer.

Design goal: no new mutable state needs to be threaded through
checkpoint/resume plumbing for CORRECTNESS. `validation_history.json` is
already the trainer's durable, ordered record of every validation this
run has ever produced, and is already loaded back into memory before the
training loop starts on resume (`training/train.py::run_training`).
`early_stopping_status` is a PURE function of that list plus the
configured monitor/mode/patience/min_delta -- recomputing it from
scratch after every validation therefore gives an early-stopping
decision that is automatically correct across any number of resumes,
with no separate "patience counter" to keep in sync or accidentally
desynchronize from the history it is supposed to describe.

`extra_metadata` recorded into `training_state.json` at each checkpoint
save (see `train.py`'s call sites) is diagnostic/inspectable state only
-- never read back to make the stop/continue decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_SUPPORTED_MONITORS = {"validation_total": "total"}
_SUPPORTED_MODES = ("min", "max")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str
    mode: str
    patience_validations: int
    min_delta: float


def resolve_early_stopping_config(training_cfg: dict) -> EarlyStoppingConfig | None:
    """Return the validated config, or None if early stopping is disabled.

    Disabled (returns None) whenever `training.early_stopping` is absent
    or falsy -- existing configs that never mention this section keep
    running exactly as before (wall-clock limit and total_steps are the
    only ways such a run ever stops), so this is purely additive."""
    raw = (training_cfg or {}).get("early_stopping")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("training.early_stopping must be a mapping when present")
    monitor = str(raw.get("monitor", "validation_total"))
    if monitor not in _SUPPORTED_MONITORS:
        raise ValueError(
            f"training.early_stopping.monitor must be one of {sorted(_SUPPORTED_MONITORS)}, got {monitor!r}"
        )
    mode = str(raw.get("mode", "min"))
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"training.early_stopping.mode must be one of {_SUPPORTED_MODES}, got {mode!r}")
    patience_validations = int(raw.get("patience_validations", 8))
    if patience_validations < 1:
        raise ValueError("training.early_stopping.patience_validations must be a positive integer")
    min_delta = float(raw.get("min_delta", 0.0))
    if math.isnan(min_delta) or math.isinf(min_delta) or min_delta < 0:
        raise ValueError("training.early_stopping.min_delta must be a finite, non-negative number")
    return EarlyStoppingConfig(
        monitor=monitor, mode=mode, patience_validations=patience_validations, min_delta=min_delta,
    )


def _monitored_value(entry: dict, monitor: str) -> float:
    key = _SUPPORTED_MONITORS[monitor]
    if key not in entry:
        raise ValueError(f"validation history entry is missing monitored field {key!r}: {entry!r}")
    return float(entry[key])


def early_stopping_status(validation_history: list[dict], config: EarlyStoppingConfig) -> dict:
    """Recompute the full early-stopping state from the ordered history.

    `is_improvement(value, best)`: for mode="min", a new value must be
    strictly less than `best - min_delta` to reset patience (mode="max"
    is the mirror image, `best + min_delta`). This intentionally matches
    the common Keras/PyTorch-Lightning ``EarlyStopping`` semantics: a
    change smaller than `min_delta` does not count as progress, so a
    long, noisy plateau eventually triggers a stop instead of resetting
    patience on floating-point noise forever.

    Returns a dict: `best_step`, `best_value` (None if history is empty),
    `n_since_improvement` (validations since the best value, 0 if the
    LAST entry is itself the best so far), `patience_remaining`, and
    `should_stop` (True once `n_since_improvement >= patience_validations`,
    which only ever happens with at least one validation on record).
    """
    if not validation_history:
        return {
            "best_step": None, "best_value": None,
            "n_since_improvement": 0,
            "patience_remaining": config.patience_validations,
            "should_stop": False,
        }
    sign = 1.0 if config.mode == "min" else -1.0
    best_step: int | None = None
    best_value: float | None = None
    n_since_improvement = 0
    for entry in validation_history:
        value = _monitored_value(entry, config.monitor)
        step = int(entry["step"])
        is_improvement = best_value is None or (sign * value) < (sign * best_value - config.min_delta)
        if is_improvement:
            best_step, best_value = step, value
            n_since_improvement = 0
        else:
            n_since_improvement += 1
    patience_remaining = max(0, config.patience_validations - n_since_improvement)
    return {
        "best_step": best_step, "best_value": best_value,
        "n_since_improvement": n_since_improvement,
        "patience_remaining": patience_remaining,
        "should_stop": n_since_improvement >= config.patience_validations,
    }
