import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from gen2_architectures.data.loaders import load_multi_sample


def _write_fake_sample(hest_dir: Path, sample_id: str, gene_names: list[str], X: np.ndarray) -> None:
    n_spots = X.shape[0]
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = gene_names
    adata.obs_names = [f"{sample_id}_spot{i}" for i in range(n_spots)]
    adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 1000, size=(n_spots, 2))
    (hest_dir / "st").mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(hest_dir / "st" / f"{sample_id}.h5ad")


def test_gene_panel_survives_per_sample_qc_noise_across_many_samples():
    """Real bug found 2026-07-25 on the actual training server: the OLD
    code applied sc.pp.filter_genes(min_cells=3) INDEPENDENTLY to each
    sample before intersecting -- with enough samples, a gene need only
    fail that per-sample threshold in ONE of them to be dropped from the
    intersection entirely. This reproduces exactly that: 30 samples, each
    with a genuinely different tiny subset of genes under-detected (<3
    spots), covering every "real" gene at least once across the cohort --
    the old code's intersection would have been empty. The new pooled
    min_cells filter must still return a healthy panel, correctly
    excluding only the ONE gene that is truly dead (zero signal) in
    EVERY sample."""
    n_genes = 20
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    n_spots = 50
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        sample_ids = [f"S{i}" for i in range(30)]
        for i, sid in enumerate(sample_ids):
            X = rng.integers(5, 50, size=(n_spots, n_genes)).astype(np.float32)
            # gene index (i % (n_genes - 1)) is under-detected in THIS sample only
            # (real signal in <3 spots) -- a different gene for every sample, so
            # every "real" gene fails the per-sample min_cells=3 bar somewhere.
            under_detected_gene = i % (n_genes - 1)
            X[3:, under_detected_gene] = 0.0  # only the first 3 spots keep signal
            # GENE19 (index n_genes - 1) is truly dead everywhere -- must stay excluded.
            X[:, n_genes - 1] = 0.0
            _write_fake_sample(hest_dir, sid, gene_names, X)

        adatas = load_multi_sample(hest_dir, sample_ids, min_genes=0, min_cells=3)
        shared = set(adatas[0].var_names)
        assert shared, "gene panel must not collapse to empty across many diverse samples"
        assert f"GENE{n_genes - 1}" not in shared, "a gene with zero signal in every sample must still be excluded"
        assert len(shared) == n_genes - 1, "every other gene should survive the pooled (not per-sample) min_cells filter"


def test_raw_panel_intersection_across_samples_with_genuinely_different_panels():
    """Mirrors the real INT1 (36,601 genes) vs MEND139 (33,538 genes)
    finding: different HEST-1k source studies/reference versions produce
    genuinely different but heavily-overlapping raw gene panels. The
    shared panel must be the real overlap, not empty and not silently
    zero-filled for genes only one group actually measured."""
    shared_core = [f"CORE{i}" for i in range(10)]
    group_a_only = [f"A_ONLY{i}" for i in range(5)]
    group_b_only = [f"B_ONLY{i}" for i in range(3)]
    rng = np.random.default_rng(1)

    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        group_a_ids = ["A0", "A1"]
        group_b_ids = ["B0", "B1"]
        for sid in group_a_ids:
            genes = shared_core + group_a_only
            X = rng.integers(5, 50, size=(20, len(genes))).astype(np.float32)
            _write_fake_sample(hest_dir, sid, genes, X)
        for sid in group_b_ids:
            genes = shared_core + group_b_only
            X = rng.integers(5, 50, size=(20, len(genes))).astype(np.float32)
            _write_fake_sample(hest_dir, sid, genes, X)

        adatas = load_multi_sample(hest_dir, group_a_ids + group_b_ids, min_genes=0, min_cells=0)
        shared = set(adatas[0].var_names)
        assert shared == set(shared_core)
        assert not (shared & set(group_a_only))
        assert not (shared & set(group_b_only))


def test_zero_raw_overlap_still_raises():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        rng = np.random.default_rng(2)
        _write_fake_sample(hest_dir, "H0", ["HUMAN_GENE0", "HUMAN_GENE1"], rng.integers(5, 50, size=(10, 2)).astype(np.float32))
        _write_fake_sample(hest_dir, "M0", ["Mouse_gene0", "Mouse_gene1"], rng.integers(5, 50, size=(10, 2)).astype(np.float32))
        with pytest.raises(ValueError, match="zero overlap"):
            load_multi_sample(hest_dir, ["H0", "M0"], min_genes=0, min_cells=0)
