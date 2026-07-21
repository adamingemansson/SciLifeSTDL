"""Fail-closed fingerprints for resumable audit evaluation."""

import torch

from src.evaluation.audit_evaluation import _resume_signature


def _config(checkpoint_dir):
    return {
        "data": {"sample_id": "INT1"},
        "masking": {"strategy": "random_dropout_patches"},
        "model": {"name": "fm_ot", "params": {"n_ode_steps": 50}},
        "training": {"checkpoint_dir": str(checkpoint_dir)},
        "evaluation": {"n_samples": 20, "image_modes": ["full", "all_zero"]},
    }


def test_resume_signature_requires_saved_weights(tmp_path):
    cfg = _config(tmp_path)
    records = [{"index": 0, "seed": 10, "context_obs_names": ["a"]}]
    assert _resume_signature(cfg, records, tmp_path / "metrics.json") is None


def test_resume_signature_changes_with_weights_masks_and_config(tmp_path):
    weights = tmp_path / "trainable_weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, weights)
    output = tmp_path / "audit_test_metrics.json"
    records = [{"index": 0, "seed": 10, "context_obs_names": ["a"]}]
    cfg = _config(tmp_path)

    original = _resume_signature(cfg, records, output)
    assert original is not None

    changed_masks = _resume_signature(
        cfg,
        [{"index": 0, "seed": 11, "context_obs_names": ["a"]}],
        output,
    )
    assert changed_masks != original

    changed_cfg = _config(tmp_path)
    changed_cfg["evaluation"]["n_samples"] = 4
    assert _resume_signature(changed_cfg, records, output) != original

    torch.save({"weight": torch.tensor([2.0])}, weights)
    assert _resume_signature(cfg, records, output) != original
