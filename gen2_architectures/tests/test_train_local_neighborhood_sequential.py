import pytest

from gen2_architectures.training import train_local_neighborhood_sequential


def test_runs_every_config_in_order(monkeypatch):
    calls = []

    def fake_main(config_path, smoke_steps=None, max_wall_clock_hours_override=None):
        calls.append((config_path, smoke_steps, max_wall_clock_hours_override))

    monkeypatch.setattr(train_local_neighborhood_sequential.train_local_neighborhood, "main", fake_main)

    train_local_neighborhood_sequential.main(["a.yaml", "b.yaml", "c.yaml"])

    assert [c[0] for c in calls] == ["a.yaml", "b.yaml", "c.yaml"]


def test_smoke_steps_and_hours_override_passed_to_every_config(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_local_neighborhood_sequential.train_local_neighborhood, "main",
        lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None:
        calls.append((config_path, smoke_steps, max_wall_clock_hours_override)),
    )

    train_local_neighborhood_sequential.main(["a.yaml", "b.yaml"], smoke_steps=25, max_wall_clock_hours_override=12.0)

    assert calls == [("a.yaml", 25, 12.0), ("b.yaml", 25, 12.0)]


def test_a_failure_in_one_config_stops_the_chain(monkeypatch):
    calls = []

    def fake_main(config_path, smoke_steps=None, max_wall_clock_hours_override=None):
        calls.append(config_path)
        if config_path == "b.yaml":
            raise RuntimeError("real training failure")

    monkeypatch.setattr(train_local_neighborhood_sequential.train_local_neighborhood, "main", fake_main)

    with pytest.raises(RuntimeError, match="real training failure"):
        train_local_neighborhood_sequential.main(["a.yaml", "b.yaml", "c.yaml"])

    assert calls == ["a.yaml", "b.yaml"], "must not silently continue past a real failure"


def test_per_config_hours_overrides_give_each_config_a_different_budget(monkeypatch):
    """Real scenario this was built for: Architecture 1 + its 1b/1c
    ablations chained on one GPU should fit in roughly the SAME total
    wall-clock window as every other GPU's single-config job (~36h), not
    each config using its own full standalone budget (which would make
    the chain ~60h+, well past the others)."""
    calls = []
    monkeypatch.setattr(
        train_local_neighborhood_sequential.train_local_neighborhood, "main",
        lambda config_path, smoke_steps=None, max_wall_clock_hours_override=None:
        calls.append((config_path, max_wall_clock_hours_override)),
    )

    train_local_neighborhood_sequential.main(
        ["arch1.yaml", "arch1b.yaml", "arch1c.yaml"],
        max_wall_clock_hours_overrides=[26.0, 5.0, 5.0],
    )

    assert calls == [("arch1.yaml", 26.0), ("arch1b.yaml", 5.0), ("arch1c.yaml", 5.0)]


def test_per_config_overrides_length_must_match_config_count():
    with pytest.raises(ValueError, match="entries"):
        train_local_neighborhood_sequential.main(
            ["a.yaml", "b.yaml", "c.yaml"], max_wall_clock_hours_overrides=[26.0, 5.0],
        )


def test_uniform_and_per_config_overrides_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        train_local_neighborhood_sequential.main(
            ["a.yaml", "b.yaml"],
            max_wall_clock_hours_override=10.0,
            max_wall_clock_hours_overrides=[5.0, 5.0],
        )
