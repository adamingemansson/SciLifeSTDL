import pytest
import torch

from gen3_multiscale.training.core_tensorboard import (
    CoreTrainerTensorBoardLogger, resolve_core_tensorboard_config,
)


class _FakeWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def close(self):
        self.closed = True


def test_resolve_returns_none_when_absent_or_falsy():
    assert resolve_core_tensorboard_config({}) is None
    assert resolve_core_tensorboard_config({"tensorboard": None}) is None
    assert resolve_core_tensorboard_config({"tensorboard": {}}) is None


def test_resolve_requires_log_dir():
    # An empty mapping is falsy and means "disabled" (see the None test
    # above); a non-empty mapping missing log_dir is a real config error.
    with pytest.raises(ValueError, match="log_dir"):
        resolve_core_tensorboard_config({"tensorboard": {"max_spatial_samples": 4}})
    assert resolve_core_tensorboard_config({"tensorboard": {"log_dir": "/tmp/x"}}) == {"log_dir": "/tmp/x"}


def test_resolve_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        resolve_core_tensorboard_config({"tensorboard": "not_a_mapping"})


def test_add_train_scalars_writes_one_tag_per_loss_component_plus_grad_norm_and_lr():
    writer = _FakeWriter()
    logger = CoreTrainerTensorBoardLogger("/tmp/unused", writer=writer)
    losses = {
        "total": torch.tensor(1.5), "primary": torch.tensor(1.0),
        "gradient": torch.tensor(0.5), "rmse_loss": torch.tensor(0.9),
        "pcc_loss": torch.tensor(0.1),
    }
    logger.add_train_scalars(42, losses, grad_norm=torch.tensor(2.0), learning_rate=1e-4)
    tags = dict((tag, (value, step)) for tag, value, step in writer.scalars)
    # float32 tensor -> python float widening means these are only exact
    # to float32 precision, not float64 -- compare with approx throughout.
    assert tags["train/total"][0] == pytest.approx(1.5) and tags["train/total"][1] == 42
    assert tags["train/primary"][0] == pytest.approx(1.0) and tags["train/primary"][1] == 42
    assert tags["train/gradient"][0] == pytest.approx(0.5) and tags["train/gradient"][1] == 42
    assert tags["train/rmse_loss"][0] == pytest.approx(0.9) and tags["train/rmse_loss"][1] == 42
    assert tags["train/pcc_loss"][0] == pytest.approx(0.1) and tags["train/pcc_loss"][1] == 42
    assert tags["train/grad_norm"][0] == pytest.approx(2.0) and tags["train/grad_norm"][1] == 42
    assert tags["train/learning_rate"] == pytest.approx((1e-4, 42))


def test_add_validation_scalars_prefixes_every_field_except_step():
    writer = _FakeWriter()
    logger = CoreTrainerTensorBoardLogger("/tmp/unused", writer=writer)
    logger.add_validation_scalars(10, {"step": 10, "total": 0.4, "flow_loss_mean_fixed_generator": 0.2})
    tags = {tag for tag, _value, _step in writer.scalars}
    assert tags == {"validation/total", "validation/flow_loss_mean_fixed_generator"}


def test_add_early_stopping_status_handles_no_best_yet():
    writer = _FakeWriter()
    logger = CoreTrainerTensorBoardLogger("/tmp/unused", writer=writer)
    logger.add_early_stopping_status(5, {
        "best_step": None, "best_value": None,
        "n_since_improvement": 0, "patience_remaining": 8,
    })
    tags = {tag for tag, _value, _step in writer.scalars}
    assert tags == {"early_stopping/patience_remaining", "early_stopping/n_since_improvement"}


def test_add_early_stopping_status_logs_best_value_and_step_once_known():
    writer = _FakeWriter()
    logger = CoreTrainerTensorBoardLogger("/tmp/unused", writer=writer)
    logger.add_early_stopping_status(7, {
        "best_step": 3, "best_value": 0.25,
        "n_since_improvement": 4, "patience_remaining": 4,
    })
    tags = dict((tag, (value, step)) for tag, value, step in writer.scalars)
    assert tags["early_stopping/best_step"] == (3.0, 7)
    assert tags["early_stopping/best_value"] == (0.25, 7)
    assert tags["early_stopping/n_since_improvement"] == (4.0, 7)
    assert tags["early_stopping/patience_remaining"] == (4.0, 7)


def test_close_closes_the_underlying_writer():
    writer = _FakeWriter()
    logger = CoreTrainerTensorBoardLogger("/tmp/unused", writer=writer)
    logger.close()
    assert writer.closed is True


def test_missing_tensorboard_package_raises_a_clear_runtime_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard" or name.startswith("tensorboard"):
            raise ImportError("simulated missing tensorboard package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(RuntimeError, match="tensorboard is not installed"):
        CoreTrainerTensorBoardLogger("/tmp/unused")
