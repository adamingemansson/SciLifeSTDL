import tempfile
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from gen2_architectures.training import data_prep


def _fake_hest1k(tmp_path: Path) -> tuple[Path, Path]:
    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True)
    (hest_dir / "patches").mkdir(parents=True)
    rows = []
    for organ, ids in {"Lung": [f"L{i}" for i in range(10)], "Kidney": [f"K{i}" for i in range(8)]}.items():
        for sid in ids:
            rows.append({"id": sid, "organ": organ, "st_technology": "Visium"})
            (hest_dir / "st" / f"{sid}.h5ad").touch()
            (hest_dir / "patches" / f"{sid}.h5").touch()
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return hest_dir, meta_path


def test_noop_when_sample_selection_absent():
    cfg = OmegaConf.create({
        "data": {"train_sample_ids": ["A", "B"], "hest_data_dir": "x"},
        "model": {"architecture": "1", "params": {}},
    })
    data_prep.apply_sample_selection(cfg)
    assert cfg.data.train_sample_ids == ["A", "B"]
    assert "organ_vocab" not in cfg.model.params


def test_architecture1_gets_organ_vocab_injected():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _fake_hest1k(Path(tmp))
        cfg = OmegaConf.create({
            "data": {"hest_data_dir": str(hest_dir), "sample_selection": {
                "organs": "all", "metadata_csv": str(meta_path), "min_samples_per_organ": 3,
                "n_validation_per_organ": 1, "n_test_per_organ": 1,
            }},
            "model": {"architecture": "1", "params": {}},
        })
        data_prep.apply_sample_selection(cfg)
        assert cfg.model.params.organ_vocab == ["Kidney", "Lung"]
        assert cfg.model.params.tech_vocab == ["Visium"]
        assert len(cfg.data.train_sample_ids) > 0
        assert not (set(cfg.data.train_sample_ids) & set(cfg.data.validation_sample_ids))
        assert not (set(cfg.data.train_sample_ids) & set(cfg.data.test_sample_ids))


def test_architecture4_does_not_get_organ_vocab_injected():
    """Real bug caught during development: Architecture4's constructor has
    no organ_vocab/tech_vocab param (fixed organ_type/tech_type strings
    instead) -- injecting it would crash the constructor with an
    unexpected keyword argument."""
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _fake_hest1k(Path(tmp))
        cfg = OmegaConf.create({
            "data": {"hest_data_dir": str(hest_dir), "sample_selection": {
                "organs": ["Lung"], "metadata_csv": str(meta_path),
            }},
            "model": {"architecture": "4", "params": {"organ_type": "Lung", "tech_type": "Visium"}},
        })
        data_prep.apply_sample_selection(cfg)
        assert "organ_vocab" not in cfg.model.params
        assert "tech_vocab" not in cfg.model.params
        assert cfg.model.params.organ_type == "Lung"  # untouched


def test_stage_a_style_config_does_not_get_organ_vocab_injected():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _fake_hest1k(Path(tmp))
        cfg = OmegaConf.create({
            "data": {"hest_data_dir": str(hest_dir), "sample_selection": {
                "organs": "all", "metadata_csv": str(meta_path),
            }},
            "model": {"params": {}},  # no "architecture", no "stage_a_checkpoint_dir" -> Stage A
        })
        data_prep.apply_sample_selection(cfg)
        assert "organ_vocab" not in cfg.model.params


def test_stage_b_style_config_gets_organ_vocab_injected():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _fake_hest1k(Path(tmp))
        cfg = OmegaConf.create({
            "data": {"hest_data_dir": str(hest_dir), "sample_selection": {
                "organs": "all", "metadata_csv": str(meta_path),
            }},
            "model": {"stage_a_checkpoint_dir": "/fake/path", "params": {}},
        })
        data_prep.apply_sample_selection(cfg)
        assert cfg.model.params.organ_vocab == ["Kidney", "Lung"]


def test_explicit_organ_vocab_in_config_is_not_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _fake_hest1k(Path(tmp))
        cfg = OmegaConf.create({
            "data": {"hest_data_dir": str(hest_dir), "sample_selection": {
                "organs": "all", "metadata_csv": str(meta_path),
            }},
            "model": {"architecture": "1", "params": {"organ_vocab": ["Custom"], "tech_vocab": ["Custom"]}},
        })
        data_prep.apply_sample_selection(cfg)
        assert cfg.model.params.organ_vocab == ["Custom"], "an explicitly-set organ_vocab must not be overwritten"
