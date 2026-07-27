import pytest

from gen2_architectures.training import train_arch3_combined


def test_splits_total_hours_and_calls_both_stages_in_order(monkeypatch):
    calls = []

    def fake_stage_a_main(config_path, smoke_steps=None, max_wall_clock_hours_override=None, **_kwargs):
        calls.append(("stage_a", config_path, smoke_steps, max_wall_clock_hours_override))

    def fake_stage_b_main(config_path, smoke_steps=None, max_wall_clock_hours_override=None, **_kwargs):
        calls.append(("stage_b", config_path, smoke_steps, max_wall_clock_hours_override))

    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_a, "main", fake_stage_a_main)
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_b, "main", fake_stage_b_main)

    train_arch3_combined.main(
        "stage_a.yaml", "stage_b.yaml", total_hours=36.0, stage_a_fraction=0.2,
    )

    assert len(calls) == 2
    assert calls[0][0] == "stage_a"
    assert calls[0][1] == "stage_a.yaml"
    assert calls[0][3] == pytest.approx(7.2)  # 20% of 36h
    assert calls[1][0] == "stage_b"
    assert calls[1][1] == "stage_b.yaml"
    assert calls[1][3] == pytest.approx(28.8)  # 80% of 36h


def test_stage_a_runs_before_stage_b(monkeypatch):
    order = []
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_a, "main",
                         lambda *a, **k: order.append("a"))
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_b, "main",
                         lambda *a, **k: order.append("b"))

    train_arch3_combined.main("a.yaml", "b.yaml", total_hours=10.0)
    assert order == ["a", "b"]


def test_smoke_steps_passed_through_to_both_stages(monkeypatch):
    calls = []
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_a, "main",
                         lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None, **_kwargs:
                         calls.append(("a", smoke_steps)))
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_b, "main",
                         lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None, **_kwargs:
                         calls.append(("b", smoke_steps)))

    train_arch3_combined.main("a.yaml", "b.yaml", total_hours=10.0, smoke_steps=25)
    assert calls == [("a", 25), ("b", 25)]


def test_invalid_stage_a_fraction_raises():
    with pytest.raises(ValueError, match="stage_a_fraction"):
        train_arch3_combined.main("a.yaml", "b.yaml", total_hours=10.0, stage_a_fraction=1.5)
    with pytest.raises(ValueError, match="stage_a_fraction"):
        train_arch3_combined.main("a.yaml", "b.yaml", total_hours=10.0, stage_a_fraction=0.0)


def test_skip_final_eval_passed_through_to_stage_b_only(monkeypatch):
    calls = []
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_a, "main",
                         lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None, **kwargs:
                         calls.append(("a", kwargs)))
    monkeypatch.setattr(train_arch3_combined.train_arch3_stage_b, "main",
                         lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None, **kwargs:
                         calls.append(("b", kwargs)))

    train_arch3_combined.main("a.yaml", "b.yaml", total_hours=10.0, skip_final_eval=True)

    assert calls[0] == ("a", {})
    assert calls[1] == ("b", {"skip_final_eval": True})
