import os
import tempfile

import torch
import torch.nn as nn

from gen2_architectures.training.checkpoint import (
    save_checkpoint, load_trainable_state, load_training_state,
    list_checkpoint_history, rollback_checkpoint,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        frozen = nn.Linear(4, 4)
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.frozen = frozen


class _AllFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad_(False)


def test_checkpoint_round_trip_preserves_trainable_weights_only():
    m = _Tiny()
    with torch.no_grad():
        m.lin.weight.fill_(3.14)
        m.frozen.weight.fill_(2.71)
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1", "g2"], tmp, step=100)
        m2 = _Tiny()
        assert not torch.allclose(m2.lin.weight, m.lin.weight)
        load_trainable_state(m2, tmp)
        assert torch.allclose(m2.lin.weight, m.lin.weight)
        # frozen weights must NOT be saved/restored -- a fresh instance's
        # frozen submodule stays at its own fresh random init
        assert not torch.allclose(m2.frozen.weight, m.frozen.weight)
        assert load_training_state(tmp)["step"] == 100


def test_checkpoint_zero_trainable_params_still_saves_metadata():
    """Real bug this session (original codebase): skipping ALL checkpoint
    files whenever there was nothing trainable made a fully-frozen model
    impossible to reconstruct later. Metadata must survive even then."""
    import os

    m = _AllFrozen()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "frozen"}, [], tmp, step=0)
        assert not os.path.exists(os.path.join(tmp, "trainable_weights.pt"))
        assert os.path.exists(os.path.join(tmp, "model_config.json"))
        assert os.path.exists(os.path.join(tmp, "gene_names.json"))
        m2 = _AllFrozen()
        load_trainable_state(m2, tmp)  # must be a clean no-op, not crash


def test_load_trainable_state_raises_on_genuine_mismatch():
    """If a model architecture DOES have trainable params but the weights
    file is missing, that's a real incomplete-checkpoint bug, not a
    frozen-by-design case -- must raise loudly, not silently proceed with
    random weights."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        # write only the metadata a save_checkpoint call for an ALL-FROZEN
        # model would produce, but load it onto a model that DOES have
        # trainable params
        import json
        import os

        with open(os.path.join(tmp, "model_config.json"), "w") as f:
            json.dump({"name": "tiny"}, f)
        with open(os.path.join(tmp, "gene_names.json"), "w") as f:
            json.dump([], f)
        try:
            load_trainable_state(m, tmp)
            assert False, "expected an AssertionError for a genuinely incomplete checkpoint"
        except AssertionError:
            pass


def test_checkpoint_history_is_pruned_to_keep_last():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        for step in (0, 100, 200, 300):
            with torch.no_grad():
                m.lin.weight.fill_(float(step))
            save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=step, keep_last=2)
        # only the two most recent snapshots survive
        assert list_checkpoint_history(tmp) == [200, 300]
        # root ("latest") always reflects the most recent save regardless of pruning
        assert load_training_state(tmp)["step"] == 300


def test_checkpoint_history_disabled_when_keep_last_zero():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=0, keep_last=0)
        assert list_checkpoint_history(tmp) == []
        assert os.path.exists(os.path.join(tmp, "model_config.json"))


def test_rollback_checkpoint_restores_an_earlier_known_good_snapshot():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=3)
        with torch.no_grad():
            m.lin.weight.fill_(999.0)  # simulate a later step that diverged/corrupted
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=200, keep_last=3)

        rollback_checkpoint(tmp, step=100)
        assert load_training_state(tmp)["step"] == 100
        restored = _Tiny()
        load_trainable_state(restored, tmp)
        assert torch.allclose(restored.lin.weight, torch.full((4, 4), 1.0))


def test_rollback_checkpoint_raises_on_unknown_step():
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=3)
        try:
            rollback_checkpoint(tmp, step=999)
            assert False, "expected a ValueError listing the real available steps"
        except ValueError as e:
            assert "999" in str(e) and "100" in str(e)


def test_history_snapshots_survive_root_being_overwritten():
    """Hard-linking means the root file at a given step gets a NEW inode on
    the next save (via os.replace) -- the history snapshot's hard link must
    still point at the OLD data, not silently become the new step's data."""
    m = _Tiny()
    with tempfile.TemporaryDirectory() as tmp:
        with torch.no_grad():
            m.lin.weight.fill_(1.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=100, keep_last=5)
        with torch.no_grad():
            m.lin.weight.fill_(2.0)
        save_checkpoint(m, {"name": "tiny"}, ["g1"], tmp, step=200, keep_last=5)

        snapshot_100 = _Tiny()
        load_trainable_state(snapshot_100, os.path.join(tmp, "history", "step_00000100"))
        assert torch.allclose(snapshot_100.lin.weight, torch.full((4, 4), 1.0))

        latest = _Tiny()
        load_trainable_state(latest, tmp)
        assert torch.allclose(latest.lin.weight, torch.full((4, 4), 2.0))
