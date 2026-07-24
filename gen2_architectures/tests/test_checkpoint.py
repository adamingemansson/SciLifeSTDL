import tempfile

import torch
import torch.nn as nn

from gen2_architectures.training.checkpoint import (
    save_checkpoint, load_trainable_state, load_training_state,
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
