"""GPT-audit-flagged bug (2026-07-27, confirmed and fixed): gen2 never
pinned its resolved train/validation/test sample-ID split anywhere, so a
resumed run (or a later standalone evaluation run) had no way to detect if
the "held-out" test samples had silently changed underneath it."""
import json

import pytest
from omegaconf import OmegaConf

from gen2_architectures.training.data_prep import save_or_verify_split_manifest


def _cfg(train, validation, test):
    return OmegaConf.create({
        "data": {
            "train_sample_ids": train,
            "validation_sample_ids": validation,
            "test_sample_ids": test,
        }
    })


def test_first_call_pins_the_manifest_to_disk(tmp_path):
    cfg = _cfg(["a", "b"], ["c"], ["d"])

    save_or_verify_split_manifest(tmp_path, cfg)

    manifest = json.loads((tmp_path / "sample_split.json").read_text())
    assert manifest == {
        "train_sample_ids": ["a", "b"],
        "validation_sample_ids": ["c"],
        "test_sample_ids": ["d"],
    }


def test_later_call_with_the_same_split_does_not_raise(tmp_path):
    cfg = _cfg(["a", "b"], ["c"], ["d"])
    save_or_verify_split_manifest(tmp_path, cfg)

    save_or_verify_split_manifest(tmp_path, _cfg(["a", "b"], ["c"], ["d"]))


def test_later_call_with_a_different_test_split_raises(tmp_path):
    save_or_verify_split_manifest(tmp_path, _cfg(["a", "b"], ["c"], ["d"]))

    with pytest.raises(ValueError, match="test_sample_ids"):
        save_or_verify_split_manifest(tmp_path, _cfg(["a", "b"], ["c"], ["e"]))


def test_later_call_with_a_different_train_split_raises(tmp_path):
    save_or_verify_split_manifest(tmp_path, _cfg(["a", "b"], ["c"], ["d"]))

    with pytest.raises(ValueError, match="train_sample_ids"):
        save_or_verify_split_manifest(tmp_path, _cfg(["a"], ["c"], ["d"]))
