import json
from pathlib import Path

import numpy as np
import pytest

from src.data.mask_bank import (
    build_mask_bank,
    load_mask_bank,
    record_masks,
    save_mask_bank,
    split_records,
)


def test_mask_bank_is_deterministic_and_barcode_keyed(tmp_path: Path):
    rng = np.random.default_rng(0)
    coords = np.c_[rng.normal(size=(80, 2)), np.zeros(80)]
    slices = np.array(["s"] * 80)
    names = np.array([f"spot-{i}" for i in range(80)])
    cfg = {"strategy": "sparse_spot_dropout", "params": {"fraction": 0.2}}
    counts = {"validation": 2, "test": 3}
    seeds = {"validation": 100, "test": 200}

    one = build_mask_bank(coords, slices, names, cfg, counts, seeds)
    two = build_mask_bank(coords, slices, names, cfg, counts, seeds)
    assert one == two
    assert len(split_records(one, "validation")) == 2
    assert len(split_records(one, "test")) == 3

    path = tmp_path / "bank.json"
    save_mask_bank(one, path)
    loaded = load_mask_bank(path, names)
    context, query = record_masks(split_records(loaded, "test")[0], names)
    assert context.any() and query.any()
    assert not np.any(context & query)

    reordered = names[::-1]
    with pytest.raises(ValueError, match="different observation set"):
        load_mask_bank(path, reordered)


def test_training_seed_bank_is_persisted_and_rejects_changed_run(tmp_path: Path):
    from src.data.mask_bank import ensure_training_seed_bank

    names = ["a", "b", "c"]
    path = tmp_path / "training.json"
    bank, returned = ensure_training_seed_bank(path, names, n_items=5, base_seed=10)
    assert returned == path
    assert bank["seeds"] == [10, 11, 12, 13, 14]
    loaded, _ = ensure_training_seed_bank(path, names, n_items=5, base_seed=10)
    assert loaded == bank
    with pytest.raises(ValueError, match="does not match"):
        ensure_training_seed_bank(path, names, n_items=6, base_seed=10)


def test_mask_bank_rejects_stale_masking_or_coordinates(tmp_path: Path):
    from src.data.mask_bank import ensure_mask_bank

    rng = np.random.default_rng(7)
    coords = np.c_[rng.normal(size=(60, 2)), np.zeros(60)]
    slices = np.array(["s"] * 60)
    names = np.array([f"spot-{i}" for i in range(60)])
    cfg = {"strategy": "sparse_spot_dropout", "params": {"fraction": 0.2}}
    path = tmp_path / "bank.json"
    ensure_mask_bank(path, coords, slices, names, cfg,
                     {"validation": 1, "test": 1}, {"validation": 10, "test": 20})

    changed_cfg = {"strategy": "sparse_spot_dropout", "params": {"fraction": 0.3}}
    with pytest.raises(ValueError, match="different masking parameters"):
        ensure_mask_bank(path, coords, slices, names, changed_cfg,
                         {"validation": 1, "test": 1}, {"validation": 10, "test": 20})

    shifted = coords.copy()
    shifted[0, 0] += 1.0
    with pytest.raises(ValueError, match="different coordinates"):
        ensure_mask_bank(path, shifted, slices, names, cfg,
                         {"validation": 1, "test": 1}, {"validation": 10, "test": 20})
