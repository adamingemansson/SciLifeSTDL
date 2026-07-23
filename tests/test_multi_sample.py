"""
Smoke test for multi-sample data loading/dataset scaffolding
(src/data/loaders.py load_multi_sample, src/training/train.py
MultiSampleMaskedContextQueryDataset — added 2026-07-15 as the first step
toward real multi-sample training, task #19 follow-up). Checks:
  - gene panels get correctly intersected across samples with PARTIAL
    overlap (not just identical panels, the easy case)
  - zero gene overlap across samples raises, not silently produces garbage
  - each drawn item's context+query never mixes spots from more than one
    sample — verified via disjoint synthetic coordinate ranges, since
    that's the real risk this design exists to avoid (see
    MultiSampleMaskedContextQueryDataset's docstring: independent samples
    have no real spatial relationship, so a k-NN context encoder must
    never blend them)

Run with: python -m tests.test_multi_sample
"""
import tempfile
from pathlib import Path

import numpy as np
import anndata as ad
from omegaconf import OmegaConf

from src.data.loaders import load_multi_sample
from src.training.train import MultiSampleMaskedContextQueryDataset, inject_multi_sample_n_genes


def _make_synthetic_sample(hest_dir: Path, sample_id: str, gene_names: list[str],
                            n_points: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    adata = ad.AnnData(X=rng.poisson(2.0, size=(n_points, n_genes)).astype(np.float32))
    adata.var_names = gene_names
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_points, 2))
    adata.obs["z"] = 0.0
    adata.write_h5ad(hest_dir / f"{sample_id}.h5ad")


def test_load_multi_sample_gene_intersection():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp)
        # sample A: genes 0-19, sample B: genes 10-29 -> overlap = genes 10-19 (10 genes)
        genes_a = [f"G{i}" for i in range(20)]
        genes_b = [f"G{i}" for i in range(10, 30)]
        _make_synthetic_sample(hest_dir, "SAMPA", gene_names=genes_a, seed=0)
        _make_synthetic_sample(hest_dir, "SAMPB", gene_names=genes_b, seed=1)

        adatas = load_multi_sample(hest_dir, ["SAMPA", "SAMPB"], min_genes=1, min_cells=1)
        assert len(adatas) == 2
        expected_shared = sorted(set(genes_a) & set(genes_b))
        assert list(adatas[0].var_names) == expected_shared
        assert list(adatas[1].var_names) == expected_shared
        assert adatas[0].n_vars == 10
        print(f"[load_multi_sample] OK — correctly intersected to {adatas[0].n_vars} shared genes")


def test_load_multi_sample_zero_overlap_raises():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp)
        _make_synthetic_sample(hest_dir, "SAMPC", gene_names=[f"C{i}" for i in range(10)], seed=0)
        _make_synthetic_sample(hest_dir, "SAMPD", gene_names=[f"D{i}" for i in range(10)], seed=1)
        try:
            load_multi_sample(hest_dir, ["SAMPC", "SAMPD"], min_genes=1, min_cells=1)
            raised = False
        except ValueError:
            raised = True
        assert raised, "zero gene overlap across samples should raise, not silently proceed"
        print("[load_multi_sample] OK — raises on zero gene overlap across samples")


def test_multi_sample_dataset_never_mixes_samples():
    n_genes, n_points = 15, 40
    rng = np.random.default_rng(0)

    # two samples with DISJOINT coordinate ranges, so any cross-sample
    # mixing within one drawn item is directly detectable
    coords_a = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)),
                                np.zeros((n_points, 1))], axis=1)
    expr_a = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_a = np.array(["SAMPA"] * n_points)

    coords_b = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)) + 100_000.0,
                                np.zeros((n_points, 1))], axis=1)
    expr_b = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_b = np.array(["SAMPB"] * n_points)

    samples = [
        (coords_a, expr_a, slice_ids_a, None, "Kidney", "Visium", None, None),
        (coords_b, expr_b, slice_ids_b, None, "Lung", "Visium", None, None),
    ]
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 2, "radius_range": [10, 30]},
    })
    dataset = MultiSampleMaskedContextQueryDataset(samples, masking_cfg, n_items=20, base_seed=0)

    both_samples_seen = set()
    for i in range(len(dataset)):
        item = dataset[i]
        context_xy = item["context"]["coords"][:, :2].numpy()
        query_xy = item["query"]["coords"][:, :2].numpy()
        all_xy = np.concatenate([context_xy, query_xy], axis=0)
        # every point in this ONE item must be in the SAME sample's disjoint range
        in_a = bool((all_xy < 50_000.0).all())
        in_b = bool((all_xy >= 50_000.0).all())
        assert in_a or in_b, f"item {i} mixed coordinates from both samples — cross-sample leakage"
        # organ/tech (2026-07-16) must match whichever sample the coords
        # came from — same cross-sample-leakage concern, applied to the
        # new per-sample metadata channel
        expected_organ = "Kidney" if in_a else "Lung"
        assert item["context"]["organ"] == expected_organ == item["query"]["organ"], (
            f"item {i}: organ label doesn't match its own sample's coordinates"
        )
        both_samples_seen.add("A" if in_a else "B")

    assert both_samples_seen == {"A", "B"}, (
        f"expected to draw from both samples across {len(dataset)} items, only saw {both_samples_seen}"
    )
    print(f"[MultiSampleMaskedContextQueryDataset] OK — {len(dataset)} items, "
          f"never mixed samples within one draw, drew from both samples across the dataset")


def test_load_multi_sample_organs_techs():
    """organs/techs (2026-07-16, OrganTechEmbedding follow-up) — optional,
    parallel to sample_ids, stored into each returned adata's .obs and
    picked up by load_multi_sample_data (see that function's docstring)."""
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp)
        genes = [f"G{i}" for i in range(15)]
        _make_synthetic_sample(hest_dir, "SAMPE", gene_names=genes, seed=0)
        _make_synthetic_sample(hest_dir, "SAMPF", gene_names=genes, seed=1)

        adatas = load_multi_sample(
            hest_dir, ["SAMPE", "SAMPF"], min_genes=1, min_cells=1,
            organs=["Kidney", "Lung"], techs=["Visium", "Visium"],
        )
        assert list(adatas[0].obs["organ"].unique()) == ["Kidney"]
        assert list(adatas[1].obs["organ"].unique()) == ["Lung"]
        assert set(adatas[0].obs["tech"].unique()) == {"Visium"}

        # mismatched-length organs/techs must raise, not silently misalign
        raised = False
        try:
            load_multi_sample(hest_dir, ["SAMPE", "SAMPF"], min_genes=1, min_cells=1,
                               organs=["Kidney"])
        except ValueError:
            raised = True
        assert raised, "organs with wrong length must raise, not silently misalign to sample_ids"

        # omitted organs/techs default every sample to "unknown"
        adatas_default = load_multi_sample(hest_dir, ["SAMPE", "SAMPF"], min_genes=1, min_cells=1)
        assert list(adatas_default[0].obs["organ"].unique()) == ["unknown"]
        print("[load_multi_sample organs/techs] OK — stored correctly, length-checked, "
              "defaults to 'unknown'")


def test_multi_sample_dataset_carries_gene_features_per_sample():
    """2026-07-17 real gap closed — MultiSampleMaskedContextQueryDataset's
    sample tuples used to carry ONLY (coords, expr, slice_ids, images,
    organ, tech); context_gene_features/context_novae_features were never
    threaded through at all, meaning gene_encoder_type="novae"/"both" and
    STPath's Route-B residual literally could not be exercised via
    multi-sample training. Verifies the extra two tuple slots actually
    reach _build_masked_item's output, per-sample (sample A's own
    features, not sample B's, and vice versa) — not just that the
    class accepts 8-tuples without crashing."""
    n_genes, n_points, novae_dim = 12, 30, 5
    rng = np.random.default_rng(0)

    coords_a = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)),
                                np.zeros((n_points, 1))], axis=1)
    expr_a = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_a = np.array(["SAMPA"] * n_points)
    novae_a = np.full((n_points, novae_dim), 1.0, dtype=np.float32)  # marker value: all 1s

    coords_b = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)) + 100_000.0,
                                np.zeros((n_points, 1))], axis=1)
    expr_b = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_b = np.array(["SAMPB"] * n_points)
    novae_b = np.full((n_points, novae_dim), 2.0, dtype=np.float32)  # marker value: all 2s

    samples = [
        (coords_a, expr_a, slice_ids_a, None, "Kidney", "Visium", None, novae_a),
        (coords_b, expr_b, slice_ids_b, None, "Lung", "Visium", None, novae_b),
    ]
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 2, "radius_range": [10, 30]},
    })
    dataset = MultiSampleMaskedContextQueryDataset(samples, masking_cfg, n_items=20, base_seed=0)

    for i in range(len(dataset)):
        item = dataset[i]
        context_xy = item["context"]["coords"][:, :2].numpy()
        in_a = bool((context_xy < 50_000.0).all())
        novae_feat = item["context"]["novae_features"].numpy()
        expected_marker = 1.0 if in_a else 2.0
        assert np.allclose(novae_feat, expected_marker), (
            f"item {i}: context_novae_features didn't match its own sample "
            f"(expected all {expected_marker}, got values {novae_feat[:3]})"
        )
    print("[MultiSampleMaskedContextQueryDataset] OK — context_novae_features "
          "correctly carried through per-sample, never cross-contaminated")


def test_multi_sample_dataset_carries_niche_labels_per_sample():
    """12-entry tuple form (context_niche_feature_provider added as the
    12th slot, 2026-07-23) -- verifies the provider is invoked with THIS
    item's own sample context-only, and its output lands in
    context['niche_labels'] correctly per-sample, mirroring
    test_multi_sample_dataset_carries_gene_features_per_sample above."""
    n_genes, n_points = 12, 30
    rng = np.random.default_rng(0)

    coords_a = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)),
                                np.zeros((n_points, 1))], axis=1)
    expr_a = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_a = np.array(["SAMPA"] * n_points)

    coords_b = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)) + 100_000.0,
                                np.zeros((n_points, 1))], axis=1)
    expr_b = rng.random((n_points, n_genes)).astype(np.float32)
    slice_ids_b = np.array(["SAMPB"] * n_points)

    def provider_a(context_mask):
        return np.full((int(np.asarray(context_mask).sum()), 1), 1.0, dtype=np.float32)

    def provider_b(context_mask):
        return np.full((int(np.asarray(context_mask).sum()), 1), 2.0, dtype=np.float32)

    samples = [
        (coords_a, expr_a, slice_ids_a, None, "Kidney", "Visium",
         None, None, None, None, None, provider_a),
        (coords_b, expr_b, slice_ids_b, None, "Lung", "Visium",
         None, None, None, None, None, provider_b),
    ]
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 2, "radius_range": [10, 30]},
    })
    dataset = MultiSampleMaskedContextQueryDataset(samples, masking_cfg, n_items=20, base_seed=0)

    for i in range(len(dataset)):
        item = dataset[i]
        context_xy = item["context"]["coords"][:, :2].numpy()
        in_a = bool((context_xy < 50_000.0).all())
        niche = item["context"]["niche_labels"].numpy()
        expected_marker = 1.0 if in_a else 2.0
        assert np.allclose(niche, expected_marker), (
            f"item {i}: niche_labels didn't match its own sample's provider "
            f"(expected all {expected_marker}, got {niche[:3]})"
        )
    print("[MultiSampleMaskedContextQueryDataset] OK — context_niche_feature_provider "
          "correctly carried through per-sample via the 12-entry tuple form")


def test_multi_sample_dataset_backwards_compatible_tuple_lengths():
    """8/10/11-entry tuples (older configs/tests, providers absent) must
    still work unchanged after the niche provider slot was added as a 12th
    entry."""
    n_points = 10
    rng = np.random.default_rng(0)
    coords = np.concatenate([rng.uniform(0, 100, size=(n_points, 2)),
                              np.zeros((n_points, 1))], axis=1)
    expr = rng.random((n_points, 5)).astype(np.float32)
    slice_ids = np.array(["SAMPA"] * n_points)
    masking_cfg = OmegaConf.create({
        "strategy": "random_dropout_patches",
        "params": {"n_patches": 1, "radius_range": [5, 15]},
    })
    base = (coords, expr, slice_ids, None, "Kidney", "Visium")
    for tail in [(None, None), (None, None, None, None), (None, None, None, None, None)]:
        samples = [base + tail]
        dataset = MultiSampleMaskedContextQueryDataset(samples, masking_cfg, n_items=3, base_seed=0)
        for i in range(len(dataset)):
            item = dataset[i]
            assert "niche_labels" not in item["context"]
    print("[MultiSampleMaskedContextQueryDataset] OK — 8/10/11-entry tuples "
          "(no niche provider) still work unchanged")


def test_inject_multi_sample_n_genes():
    """2026-07-17 real gap closed — multi-sample n_genes previously had no
    auto-injector at all (see exp_hest1k_fm_ot_multisample.yaml's own
    header, which required manually checking real console output and
    hand-correcting a placeholder value). A duck-typed stand-in is enough
    here since inject_multi_sample_n_genes only ever reads .n_vars."""
    class _FakeAdata:
        def __init__(self, n_vars):
            self.n_vars = n_vars

    model_cfg = {"params": {}}
    inject_multi_sample_n_genes(model_cfg, [_FakeAdata(37), _FakeAdata(37)])
    assert model_cfg["params"]["n_genes"] == 37

    # explicit config value always wins, never silently overridden
    model_cfg_explicit = {"params": {"n_genes": 999}}
    inject_multi_sample_n_genes(model_cfg_explicit, [_FakeAdata(37)])
    assert model_cfg_explicit["params"]["n_genes"] == 999
    print("[inject_multi_sample_n_genes] OK — auto-derives from real data, "
          "never overrides an explicit config value")


if __name__ == "__main__":
    test_load_multi_sample_gene_intersection()
    test_load_multi_sample_zero_overlap_raises()
    test_multi_sample_dataset_never_mixes_samples()
    test_load_multi_sample_organs_techs()
    test_multi_sample_dataset_carries_gene_features_per_sample()
    test_multi_sample_dataset_carries_niche_labels_per_sample()
    test_multi_sample_dataset_backwards_compatible_tuple_lengths()
    test_inject_multi_sample_n_genes()
    print("\nAll multi-sample loader/dataset smoke tests passed.")


def test_load_multi_sample_reference_panel_does_not_intersect_with_test():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp)
        reference = ["G2", "G1"]
        _make_synthetic_sample(hest_dir, "TRAIN", ["G0", "G1", "G2"], seed=0)
        _make_synthetic_sample(hest_dir, "TEST", ["G3", "G2", "G1"], seed=1)
        heldout = load_multi_sample(
            hest_dir, ["TEST"], min_genes=1, min_cells=1,
            reference_genes=reference,
        )
        assert list(heldout[0].var_names) == reference

        _make_synthetic_sample(hest_dir, "BADTEST", ["G1", "G3"], seed=2)
        import pytest
        with pytest.raises(ValueError, match="Refusing to intersect with test data"):
            load_multi_sample(
                hest_dir, ["BADTEST"], min_genes=1, min_cells=1,
                reference_genes=reference,
            )
