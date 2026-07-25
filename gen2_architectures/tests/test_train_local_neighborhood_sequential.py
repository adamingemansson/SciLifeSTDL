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

    import pytest
    with pytest.raises(RuntimeError, match="real training failure"):
        train_local_neighborhood_sequential.main(["a.yaml", "b.yaml", "c.yaml"])

    assert calls == ["a.yaml", "b.yaml"], "must not silently continue past a real failure"
