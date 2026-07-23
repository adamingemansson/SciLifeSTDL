"""Tests for train.py's inject_stpath_frozen_gene_table -- the data-loading
glue for gene_encoder_type='stpath_frozen_table' (see
src/models/stpath_gene_table.py for the extraction itself). No scanpy
needed: this function only reads a JSON vocab file + a torch checkpoint,
never touches AnnData beyond .var_names.
"""
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training.train import inject_stpath_frozen_gene_table


class _FakeAdata:
    def __init__(self, gene_names):
        self.var_names = np.asarray(gene_names)


def _write_fixtures(tmp_path, d_model=4, n_tokens=10):
    mapping = {"GENE_A": "ENSG001", "GENE_B": "ENSG002", "GENE_C": "ENSG003"}
    voc_path = tmp_path / "symbol2ensembl.json"
    voc_path.write_text(json.dumps(mapping))
    weight = torch.arange(d_model * n_tokens, dtype=torch.float32).reshape(d_model, n_tokens)
    ckpt_path = tmp_path / "stpath_weights.pt"
    torch.save({"input_encoder.gene_embed.weight": weight}, ckpt_path)
    return str(voc_path), str(ckpt_path)


def _make_cfg(tmp_path, experiment_name="test_stpath_frozen"):
    return OmegaConf.create({
        "experiment_name": experiment_name,
        "data": {"hest_cache_dir": str(tmp_path / "cache")},
    })


def test_noop_when_gene_encoder_type_is_not_stpath_frozen_table(tmp_path):
    model_cfg = {"params": {"gene_encoder_type": "weighted_linear"}}
    inject_stpath_frozen_gene_table(model_cfg, _FakeAdata(["GENE_A"]), _make_cfg(tmp_path))
    assert "stpath_frozen_gene_table" not in model_cfg["params"]
    print("[inject_stpath_frozen_gene_table] OK — no-op for every other gene_encoder_type")


def test_raises_without_voc_and_weight_paths(tmp_path):
    model_cfg = {"params": {"gene_encoder_type": "stpath_frozen_table"}}
    try:
        inject_stpath_frozen_gene_table(model_cfg, _FakeAdata(["GENE_A"]), _make_cfg(tmp_path))
        raise AssertionError("expected a ValueError for missing voc/weight paths")
    except ValueError as exc:
        assert "stpath_gene_voc_path" in str(exc)
    print("[inject_stpath_frozen_gene_table] OK — fails closed without the STPath checkpoint paths")


def test_extracts_and_caches_the_real_table(tmp_path):
    voc_path, ckpt_path = _write_fixtures(tmp_path)
    cfg = _make_cfg(tmp_path)
    model_cfg = {"params": {
        "gene_encoder_type": "stpath_frozen_table",
        "stpath_gene_voc_path": voc_path,
        "stpath_model_weight_path": ckpt_path,
        "stpath_d_model": 4,
    }}
    inject_stpath_frozen_gene_table(model_cfg, _FakeAdata(["GENE_B", "GENE_A"]), cfg)
    table = model_cfg["params"]["stpath_frozen_gene_table"]
    assert np.asarray(table).shape == (4, 2)

    # stpath_gene_voc_path/stpath_model_weight_path/stpath_d_model are
    # config-only INPUTS to this extraction, not accepted by
    # HierarchicalGeneTransportRegressor's own constructor -- build_model
    # does a plain **params unpack with no filtering, so leaving these
    # behind would crash with an unexpected-keyword TypeError at model
    # construction. Real bug caught and fixed while building this.
    for leaked_key in ("stpath_gene_voc_path", "stpath_model_weight_path", "stpath_d_model"):
        assert leaked_key not in model_cfg["params"], (
            f"{leaked_key} must be popped from params after use, not left for build_model"
        )

    cache_files = list((tmp_path / "cache" / "stpath_frozen_gene_table_cache").glob("*.npz"))
    assert len(cache_files) == 1, "expected exactly one cache file to be written"

    # Second call, fresh model_cfg (no pre-set table) -- must hit the cache
    # rather than requiring the checkpoint file to still exist.
    (tmp_path / "stpath_weights.pt").unlink()
    model_cfg_2 = {"params": {
        "gene_encoder_type": "stpath_frozen_table",
        "stpath_gene_voc_path": voc_path,
        "stpath_model_weight_path": ckpt_path,
        "stpath_d_model": 4,
    }}
    inject_stpath_frozen_gene_table(model_cfg_2, _FakeAdata(["GENE_B", "GENE_A"]), cfg)
    assert np.allclose(model_cfg_2["params"]["stpath_frozen_gene_table"], table)
    print("[inject_stpath_frozen_gene_table] OK — extracts the real table, caches it to disk, "
          "and a second call hits the cache without needing the checkpoint again")


def test_does_not_overwrite_an_explicit_table(tmp_path):
    model_cfg = {"params": {
        "gene_encoder_type": "stpath_frozen_table",
        "stpath_frozen_gene_table": [[1.0, 2.0]],
    }}
    inject_stpath_frozen_gene_table(model_cfg, _FakeAdata(["GENE_A", "GENE_B"]), _make_cfg(tmp_path))
    assert model_cfg["params"]["stpath_frozen_gene_table"] == [[1.0, 2.0]]
    print("[inject_stpath_frozen_gene_table] OK — never overrides an explicitly-set table")


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_noop_when_gene_encoder_type_is_not_stpath_frozen_table,
        test_raises_without_voc_and_weight_paths,
        test_extracts_and_caches_the_real_table,
        test_does_not_overwrite_an_explicit_table,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    print("\nAll inject_stpath_frozen_gene_table tests passed.")
