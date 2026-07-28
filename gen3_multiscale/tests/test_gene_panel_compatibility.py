import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gen3_multiscale.data.hest1k_catalog import resolve_compatible_sample_ids, resolve_sample_selection


def _write_real_sample(hest_dir: Path, sample_id: str, gene_names: list[str], n_spots: int = 5) -> None:
    (hest_dir / "st").mkdir(parents=True, exist_ok=True)
    X = np.random.default_rng(0).integers(1, 20, size=(n_spots, len(gene_names))).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"{sample_id}_spot{i}" for i in range(n_spots)]
    adata.write_h5ad(hest_dir / "st" / f"{sample_id}.h5ad")


def test_excludes_a_continuously_varying_small_panel_outlier():
    """Real bug found 2026-07-25 on the actual training server, AFTER the
    min_nb_genes fix: a size-based metadata threshold was not enough,
    because real HEST-1k nb_genes values vary CONTINUOUSLY within every
    source-study group rather than being cleanly bimodal (real confirmed
    TENX values: 538, 541, 1056, ..., 5001, 10006, 10017 -- some clear
    the 5000 threshold despite still being incompatible). This exercises
    the real measured-overlap check directly: 9 samples share a 20-gene
    core panel; a 10th sample shares only 2 of those genes (a targeted
    panel that happens to have enough TOTAL genes to clear a naive size
    threshold, but doesn't overlap the real cohort) -- it must still be
    excluded."""
    core_genes = [f"CORE{i}" for i in range(20)]
    outlier_genes = ["CORE0", "CORE1"] + [f"OTHER{i}" for i in range(30)]  # 32 total genes -- clears a 5000 threshold trivially in spirit, but not in real overlap
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        compatible_ids = [f"S{i}" for i in range(9)]
        for sid in compatible_ids:
            _write_real_sample(hest_dir, sid, core_genes)
        _write_real_sample(hest_dir, "OUTLIER", outlier_genes)

        kept, shared_genes = resolve_compatible_sample_ids(
            hest_dir, compatible_ids + ["OUTLIER"],
            min_gene_coverage=0.9, min_sample_coverage=0.9, min_panel_size=5,
        )
        assert "OUTLIER" not in kept
        assert set(kept) == set(compatible_ids)
        assert set(shared_genes) == set(core_genes)


def test_raises_when_the_whole_cohort_is_incompatible():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        _write_real_sample(hest_dir, "A", [f"AGENE{i}" for i in range(10)])
        _write_real_sample(hest_dir, "B", [f"BGENE{i}" for i in range(10)])
        with pytest.raises(ValueError, match="not compatible enough"):
            resolve_compatible_sample_ids(hest_dir, ["A", "B"], min_panel_size=5)


def test_all_compatible_keeps_everyone():
    genes = [f"GENE{i}" for i in range(30)]
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        ids = [f"S{i}" for i in range(5)]
        for sid in ids:
            _write_real_sample(hest_dir, sid, genes)
        kept, shared_genes = resolve_compatible_sample_ids(hest_dir, ids, min_panel_size=5)
        assert set(kept) == set(ids)
        assert set(shared_genes) == set(genes)


def _fake_metadata(tmp_path: Path, ids: list[str], organ: str = "Lung") -> Path:
    rows = [{"id": sid, "organ": organ, "st_technology": "Visium", "species": "Homo sapiens", "nb_genes": 5000} for sid in ids]
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return meta_path


def test_resolve_sample_selection_end_to_end_drops_incompatible_organ_member():
    """The full integration this whole investigation was chasing: a
    resolve_sample_selection call whose per-organ candidate pool includes
    one real-panel-incompatible sample must still succeed, silently
    dropping only that one sample rather than crashing the whole run."""
    core_genes = [f"CORE{i}" for i in range(20)]
    outlier_genes = [f"OTHER{i}" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        (hest_dir / "patches").mkdir(parents=True)
        compatible_ids = [f"S{i}" for i in range(9)]
        for sid in compatible_ids:
            _write_real_sample(hest_dir, sid, core_genes)
            (hest_dir / "patches" / f"{sid}.h5").touch()
        _write_real_sample(hest_dir, "OUTLIER", outlier_genes)
        (hest_dir / "patches" / "OUTLIER.h5").touch()

        meta_path = _fake_metadata(Path(tmp), compatible_ids + ["OUTLIER"])
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            min_gene_coverage=0.9, min_sample_coverage=0.9, min_panel_size=5,
        )
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert "OUTLIER" not in all_selected
        assert set(all_selected) == set(compatible_ids)
