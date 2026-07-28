import time

from gen2_architectures.training.data_prep import make_progress_fn


def test_falls_back_to_step_fraction_when_no_wall_clock_deadline():
    progress_fn = make_progress_fn(None, total_steps=1000)
    assert progress_fn(0) == 0.0
    assert progress_fn(500) == 0.5
    assert progress_fn(999) == 0.999


def test_uses_wall_clock_fraction_when_deadline_is_set():
    # a huge total_steps (the real safety-cap scenario) must NOT make
    # progress stay near 0 for the whole run once a wall-clock deadline
    # is set -- that was the actual 2026-07-25 bug.
    deadline = time.monotonic() + 0.2
    progress_fn = make_progress_fn(deadline, total_steps=100_000_000)
    early = progress_fn(step=1)
    time.sleep(0.1)
    later = progress_fn(step=2)
    assert 0.0 <= early < later <= 1.0
    assert later > 0.3, "progress should reflect elapsed wall-clock time, not step / total_steps"


def test_wall_clock_progress_is_capped_at_one():
    deadline = time.monotonic() + 0.05
    progress_fn = make_progress_fn(deadline, total_steps=1000)
    time.sleep(0.15)
    assert progress_fn(step=1) == 1.0


def test_already_elapsed_seconds_continues_the_curriculum_instead_of_resetting_it():
    """GPT-audit-flagged bug (2026-07-27, second-pass re-audit): every
    resume reset the loss-stage curriculum back to stage 1 (pure MSE)
    because progress was always computed relative to THIS launch's own
    training_start, even after the model had trained long enough to reach
    a later curriculum stage before the restart."""
    # simulate a resume 90 seconds into a 100-second total budget: only 10s
    # of wall-clock deadline remains, but progress should read ~0.9, not ~0.
    deadline = time.monotonic() + 10.0
    progress_fn = make_progress_fn(deadline, total_steps=1000, already_elapsed_seconds=90.0)
    assert progress_fn(step=1) >= 0.85


def test_zero_already_elapsed_seconds_matches_the_original_behavior():
    deadline = time.monotonic() + 0.2
    progress_fn = make_progress_fn(deadline, total_steps=1000, already_elapsed_seconds=0.0)
    assert progress_fn(step=1) < 0.1
