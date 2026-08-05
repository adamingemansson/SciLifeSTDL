"""Opt-in TensorBoard scalar logging for the shared Gen3/4/5/6 trainer.

Purely additive: `training.tensorboard` absent or falsy means the
trainer behaves exactly as before this module existed -- no TensorBoard
import, no log_dir created, no behavior change for any run that does not
ask for it. Reuses the same `purge_step`-on-resume discipline already
validated for the supervisor (MK) trainers in
`conditional_wae/tensorboard.py`, so a resumed run does not leave stale
post-crash scalar points visible in the same run's TensorBoard tab.

Scalar-only by design: unlike `conditional_wae/tensorboard.py`'s
Projector/thumbnail snapshot machinery (built for the supervisor tasks'
own diagnostic needs), the shared Gen3/4/5/6 trainer here only needs the
plain per-step/per-validation numbers a stability audit reads -- there is
no query-region H&E/embedding projector requirement for this trainer.
"""
from __future__ import annotations

from pathlib import Path


def resolve_core_tensorboard_config(training_cfg: dict) -> dict | None:
    """Return the validated config, or None if TensorBoard is disabled."""
    raw = (training_cfg or {}).get("tensorboard")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("training.tensorboard must be a mapping when present")
    log_dir = raw.get("log_dir")
    if not log_dir:
        raise ValueError("training.tensorboard.log_dir is required when training.tensorboard is set")
    return {"log_dir": str(log_dir)}


class CoreTrainerTensorBoardLogger:
    """Minimal scalar-only logger for the shared trainer's own loop."""

    def __init__(self, log_dir: str | Path, *, writer=None, purge_step: int | None = None):
        self.log_dir = Path(log_dir)
        if writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is not installed"
                ) from exc
            writer = SummaryWriter(
                log_dir=str(self.log_dir),
                purge_step=(int(purge_step) if purge_step is not None else None),
            )
        self.writer = writer

    @staticmethod
    def _scalar(value) -> float:
        return float(value.detach()) if hasattr(value, "detach") else float(value)

    def add_train_scalars(self, step: int, losses: dict, *, grad_norm, learning_rate: float) -> None:
        for name, value in losses.items():
            self.writer.add_scalar(f"train/{name}", self._scalar(value), step)
        self.writer.add_scalar("train/grad_norm", self._scalar(grad_norm), step)
        self.writer.add_scalar("train/learning_rate", float(learning_rate), step)

    def add_validation_scalars(self, step: int, entry: dict) -> None:
        for name, value in entry.items():
            if name == "step":
                continue
            self.writer.add_scalar(f"validation/{name}", self._scalar(value), step)

    def add_early_stopping_status(self, step: int, status: dict) -> None:
        if status.get("best_value") is not None:
            self.writer.add_scalar("early_stopping/best_value", self._scalar(status["best_value"]), step)
        if status.get("best_step") is not None:
            self.writer.add_scalar("early_stopping/best_step", float(status["best_step"]), step)
        self.writer.add_scalar(
            "early_stopping/patience_remaining", float(status["patience_remaining"]), step,
        )
        self.writer.add_scalar(
            "early_stopping/n_since_improvement", float(status["n_since_improvement"]), step,
        )

    def close(self) -> None:
        self.writer.close()
