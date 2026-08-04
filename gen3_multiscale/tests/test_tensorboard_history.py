import json
from pathlib import Path

from gen3_multiscale.scripts.tensorboard_history import (
    TensorBoardHistoryBridge,
    parse_training_line,
    run_name_for_path,
)


class _FakeWriter:
    instances = {}

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.instances[log_dir] = self

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


def test_parse_training_and_autoencoder_lines():
    assert parse_training_line(
        "[step 50] train: total=0.500000, primary=0.400000, grad_norm=1.25e-2"
    ) == [
        ("train/total", 0.5, 50),
        ("train/primary", 0.4, 50),
        ("train/grad_norm", 0.0125, 50),
    ]
    assert parse_training_line("[autoencoder epoch 4/50] train_mse=0.12345678") == [
        ("train/mse", 0.12345678, 4),
    ]


def test_run_name_unifies_log_and_checkpoint_history(tmp_path):
    root = tmp_path / "results"
    log = root / "gen6_screen_stamp" / "logs" / "gen6c.log"
    history = root / "gen6_screen_stamp" / "checkpoints" / "gen6c" / "validation_history.json"
    assert run_name_for_path(log, root) == "gen6_screen_stamp/gen6c"
    assert run_name_for_path(history, root) == "gen6_screen_stamp/gen6c"


def test_bridge_backfills_and_then_imports_only_new_values(tmp_path):
    _FakeWriter.instances = {}
    results = tmp_path / "results"
    output = tmp_path / "tensorboard"
    log = results / "gen3_full" / "logs" / "arch1.log"
    history = results / "gen3_full" / "checkpoints_arch1" / "validation_history.json"
    log.parent.mkdir(parents=True)
    history.parent.mkdir(parents=True)
    log.write_text(
        "[step 50] train: total=0.5, primary=0.4\n"
        "[step 50] validation: total=0.3\n"
    )
    history.write_text(json.dumps([{"step": 50, "total": 0.3}]))

    bridge = TensorBoardHistoryBridge(results, output, writer_factory=_FakeWriter)
    first = bridge.sync_once()
    assert first["new_scalars"] == 3
    assert bridge.sync_once()["new_scalars"] == 0

    with log.open("a") as handle:
        handle.write("[step 100] train: total=0.25\n")
    history.write_text(json.dumps([
        {"step": 50, "total": 0.3},
        {"step": 100, "total": 0.2, "flow_loss_mean_fixed_generator": 0.1},
    ]))
    third = bridge.sync_once()
    assert third["new_scalars"] == 3
    bridge.close()

    writer = _FakeWriter.instances[str((output / "gen3_full" / "arch1").resolve())]
    assert ("train/total", 0.25, 100) in writer.scalars
    assert ("validation/flow_loss_mean_fixed_generator", 0.1, 100) in writer.scalars
