import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gen3_multiscale.scripts.gen6b_stability_audit import audit_gen6b_training_stability
from gen3_multiscale.training.checkpoint import save_checkpoint


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


def _write_validation_history(checkpoint_dir: Path, entries: list[dict]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "validation_history.json").write_text(json.dumps(entries))


def _save_bundle(checkpoint_dir: Path, step: int) -> None:
    save_checkpoint(_Tiny(), {"name": "tiny"}, ["g1"], checkpoint_dir, step=step, keep_last=100)


def test_raises_when_validation_history_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="validation_history.json"):
        audit_gen6b_training_stability(tmp_path / "no_history")


def test_raises_when_validation_history_is_empty(tmp_path):
    checkpoint_dir = tmp_path / "empty_history"
    _write_validation_history(checkpoint_dir, [])
    with pytest.raises(ValueError, match="zero validations"):
        audit_gen6b_training_stability(checkpoint_dir)


def test_raises_on_non_ascending_steps(tmp_path):
    checkpoint_dir = tmp_path / "bad_order"
    _write_validation_history(checkpoint_dir, [
        {"step": 200, "total": 0.5}, {"step": 100, "total": 0.4},
    ])
    with pytest.raises(ValueError, match="not monotonically ascending"):
        audit_gen6b_training_stability(checkpoint_dir)


def test_stable_run_still_improving(tmp_path):
    checkpoint_dir = tmp_path / "stable_run"
    entries = [{"step": s, "total": 1.0 - 0.05 * (s // 100)} for s in range(100, 1001, 100)]
    _write_validation_history(checkpoint_dir, entries)
    for step in (900, 1000):
        _save_bundle(checkpoint_dir, step)

    report = audit_gen6b_training_stability(checkpoint_dir)
    assert report["verdict"] == "stable"
    assert report["best_step"] == 1000
    assert report["final_step"] == 1000
    assert report["relative_degradation_final_vs_best"] == pytest.approx(0.0)
    assert report["recent_validation_slope"] is not None and report["recent_validation_slope"] < 0
    assert report["retained_checkpoint_steps"] == [900, 1000]
    assert report["has_best_bundle"] is False


def test_plateaued_run_flat_validation_never_worsens(tmp_path):
    checkpoint_dir = tmp_path / "plateaued_run"
    # Ten validations all at the same value -- an exact plateau: never
    # improves by min_delta, but also never gets meaningfully worse.
    entries = [{"step": s, "total": 0.30} for s in range(100, 1101, 100)]
    _write_validation_history(checkpoint_dir, entries)
    report = audit_gen6b_training_stability(
        checkpoint_dir, early_stopping_patience_validations=8, early_stopping_min_delta=0.0001,
    )
    assert report["retrospective_early_stopping"]["should_stop"] is True
    assert report["verdict"] == "plateaued"
    assert report["relative_degradation_final_vs_best"] == pytest.approx(0.0)


def test_mildly_overfit_run_degrades_after_the_best_step(tmp_path):
    checkpoint_dir = tmp_path / "overfit_run"
    improving = [{"step": s, "total": 1.0 - 0.01 * (s // 100)} for s in range(100, 501, 100)]
    best_value = improving[-1]["total"]
    # Ten further validations, each strictly worse than the best, well
    # beyond min_delta and beyond the 1% relative-degradation threshold.
    degrading = [
        {"step": s, "total": best_value * 1.05 + 0.001 * (i)}
        for i, s in enumerate(range(600, 1601, 100))
    ]
    entries = improving + degrading
    _write_validation_history(checkpoint_dir, entries)
    report = audit_gen6b_training_stability(
        checkpoint_dir, early_stopping_patience_validations=8, early_stopping_min_delta=0.0001,
    )
    assert report["retrospective_early_stopping"]["should_stop"] is True
    assert report["relative_degradation_final_vs_best"] > 0.01
    assert report["verdict"] == "mildly_overfit"


def test_unstable_run_detects_gradient_spikes_from_the_log(tmp_path):
    checkpoint_dir = tmp_path / "unstable_run"
    entries = [{"step": s, "total": 0.5} for s in range(100, 501, 100)]
    _write_validation_history(checkpoint_dir, entries)
    log_lines = []
    for step in range(1, 101):
        grad_norm = 100.0 if step in (37, 62) else 1.0  # 2/100 = 2% ... below 5%, bump the count
        log_lines.append(f"[step {step}] train: total=0.500000, primary=0.400000, gradient=0.100000, grad_norm={grad_norm:.4f}")
    # Push the spike fraction above the 5% default threshold explicitly.
    for step in range(101, 111):
        log_lines.append(f"[step {step}] train: total=0.500000, primary=0.400000, gradient=0.100000, grad_norm=999.0000")
    log_path = tmp_path / "train.log"
    log_path.write_text("\n".join(log_lines) + "\n")

    report = audit_gen6b_training_stability(checkpoint_dir, train_log_path=log_path)
    assert report["gradient_norm_summary"]["n"] == 110
    assert report["gradient_norm_summary"]["n_spikes"] >= 10
    assert report["verdict"] == "unstable"


def test_unstable_run_detects_a_real_non_finite_crash(tmp_path):
    checkpoint_dir = tmp_path / "crashed_run"
    entries = [{"step": 100, "total": 0.5}]
    _write_validation_history(checkpoint_dir, entries)
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[step 100] train: total=0.500000, primary=0.400000, gradient=0.100000, grad_norm=1.2345\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: [step 101] non-finite total loss (nan) -- failing the run rather than "
        "skipping this step\n"
    )
    report = audit_gen6b_training_stability(checkpoint_dir, train_log_path=log_path)
    assert report["verdict"] == "unstable"
    assert len(report["non_finite_events"]) == 1
    assert report["non_finite_events"][0]["step"] == 101
    assert report["non_finite_events"][0]["kind"] == "non-finite total loss"


def test_train_validation_gap_is_computed_when_a_log_is_given(tmp_path):
    checkpoint_dir = tmp_path / "gap_run"
    entries = [{"step": 100, "total": 0.6}, {"step": 200, "total": 0.5}]
    _write_validation_history(checkpoint_dir, entries)
    log_lines = [
        f"[step {s}] train: total={0.3:.6f}, primary=0.2, gradient=0.05, grad_norm=1.0"
        for s in range(1, 201)
    ]
    log_path = tmp_path / "train.log"
    log_path.write_text("\n".join(log_lines) + "\n")
    report = audit_gen6b_training_stability(checkpoint_dir, train_log_path=log_path)
    assert len(report["train_validation_gap"]) == 2
    first, second = report["train_validation_gap"]
    assert first["validation_step"] == 100
    assert first["train_moving_average"] == pytest.approx(0.3)
    assert first["gap"] == pytest.approx(0.3 - 0.6)
    assert second["gap"] == pytest.approx(0.3 - 0.5)


def test_best_checkpoint_availability_reflects_the_real_best_bundle(tmp_path):
    checkpoint_dir = tmp_path / "best_bundle_run"
    entries = [{"step": 100, "total": 0.6}, {"step": 200, "total": 0.4}]
    _write_validation_history(checkpoint_dir, entries)
    save_checkpoint(
        _Tiny(), {"name": "tiny"}, ["g1"], checkpoint_dir / "best", step=200, keep_last=1,
    )
    report = audit_gen6b_training_stability(checkpoint_dir)
    assert report["has_best_bundle"] is True
    assert report["best_checkpoint_availability"] == "best_bundle_present"


def test_pruned_checkpoints_near_the_minimum_are_reported_as_not_retained(tmp_path):
    checkpoint_dir = tmp_path / "pruned_run"
    entries = [{"step": s, "total": 1.0 - 0.001 * s} for s in range(100, 1001, 100)]
    _write_validation_history(checkpoint_dir, entries)
    # Only the FINAL step's bundle is retained -- as a real keep_last=1 run would leave.
    _save_bundle(checkpoint_dir, 1000)
    report = audit_gen6b_training_stability(checkpoint_dir)
    assert report["best_step"] == 1000
    assert report["retained_checkpoint_steps"] == [1000]
    assert report["best_checkpoint_availability"] == "present_in_history_only"
