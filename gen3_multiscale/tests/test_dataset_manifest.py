"""Tests for the real Gen3 dataset manifest (Step 1 of the real data
builder/trainer, CONTRACT.md section 30). Uses REAL small synthetic
AnnData files written to disk (not just .touch()'d placeholders like
test_hest1k_catalog.py's fixtures) -- this module reads real barcodes,
real coordinates, and real gene panels, so the fixture must provide
them."""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gen3_multiscale.data.dataset_manifest import (
    all_composite_spot_ids, build_dataset_manifest, composite_spot_id, ensure_dataset_manifest,
    gene_panel_hash, load_dataset_manifest, save_dataset_manifest,
)

_SMALL_BUILD_KWARGS = dict(
    min_nb_genes=None, check_gene_panel_compatibility=True, min_panel_size=1,
    min_gene_coverage=0.5, min_sample_coverage=0.5,
    gene_min_genes_per_spot=1, gene_min_cells=0,
)


def _make_synthetic_hest1k(
    tmp_path: Path,
    organ_sample_ids: dict[str, list[str]],
    n_spots_per_sample: int = 6,
    gene_names: list[str] | None = None,
    patient_by_id: dict[str, str] | None = None,
    barcodes_by_id: dict[str, list[str]] | None = None,
    seed: int = 0,
) -> tuple[Path, Path, list[str]]:
    """Real local HEST-1k-shaped directory: real .h5ad files with
    obsm['spatial'] and var_names/obs_names, plus a matching metadata
    CSV. barcodes_by_id lets a test simulate two DIFFERENT samples
    reusing the exact same literal barcode string (the real,
    not-globally-unique-barcode scenario composite_spot_id exists to
    handle)."""
    gene_names = gene_names or [f"GENE{i}" for i in range(8)]
    barcodes_by_id = barcodes_by_id or {}
    rng = np.random.default_rng(seed)
    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True, exist_ok=True)
    (hest_dir / "patches").mkdir(parents=True, exist_ok=True)

    rows = []
    for organ, ids in organ_sample_ids.items():
        for sid in ids:
            barcodes = barcodes_by_id.get(sid, [f"{sid}-SPOT{i}-1" for i in range(n_spots_per_sample)])
            n = len(barcodes)
            counts = rng.poisson(5, size=(n, len(gene_names))).astype(np.float32)
            coords = rng.uniform(0, 1000, size=(n, 2))
            adata = ad.AnnData(
                X=counts,
                obs=pd.DataFrame(index=pd.Index(barcodes, name="barcode")),
                var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
            )
            adata.obsm["spatial"] = coords
            adata.write_h5ad(hest_dir / "st" / f"{sid}.h5ad")
            (hest_dir / "patches" / f"{sid}.h5").touch()
            row = {
                "id": sid, "organ": organ, "st_technology": "Visium",
                "species": "Homo sapiens", "nb_genes": len(gene_names),
            }
            if patient_by_id is not None:
                row["patient"] = patient_by_id.get(sid, sid)
            rows.append(row)
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return hest_dir, meta_path, gene_names


def test_composite_spot_id_format_and_delimiter_rejection():
    assert composite_spot_id("L0", "AAACAAGTATCTCCCA-1") == "L0::AAACAAGTATCTCCCA-1"
    with pytest.raises(ValueError, match="delimiter"):
        composite_spot_id("L::0", "barcode")
    with pytest.raises(ValueError, match="delimiter"):
        composite_spot_id("L0", "bar::code")


def test_gene_panel_hash_is_order_sensitive():
    assert gene_panel_hash(["A", "B"]) != gene_panel_hash(["B", "A"])
    assert gene_panel_hash(["A", "B"]) == gene_panel_hash(["A", "B"])


def test_build_dataset_manifest_basic_fields(tmp_path):
    hest_dir, meta_path, gene_names = _make_synthetic_hest1k(
        tmp_path, {"Lung": [f"L{i}" for i in range(6)], "Kidney": [f"K{i}" for i in range(6)]},
    )
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=1, n_test_per_organ=1, split_seed=0,
        **_SMALL_BUILD_KWARGS,
    )
    train, val, test = manifest["train_sample_ids"], manifest["validation_sample_ids"], manifest["test_sample_ids"]
    all_ids = set(train) | set(val) | set(test)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert manifest["organ_vocab"] == ["Kidney", "Lung"]
    assert set(manifest["gene_panel"]) == set(gene_names)
    assert manifest["n_genes"] == len(manifest["gene_panel"])
    assert manifest["gene_panel_hash"] == gene_panel_hash(manifest["gene_panel"])
    assert set(manifest["samples"].keys()) == all_ids
    for sid in all_ids:
        record = manifest["samples"][sid]
        assert record["n_spots"] == 6
        assert len(record["barcodes"]) == 6
        assert len(record["composite_spot_ids"]) == 6
        assert record["composite_spot_ids"] == [composite_spot_id(sid, b) for b in record["barcodes"]]
        assert np.asarray(record["coords"]).shape == (6, 2)
        assert record["split"] in {"train", "validation", "test"}
        assert record["wsi_cache"] is None  # no cache file exists in this fixture


def test_composite_spot_ids_are_unique_even_when_raw_barcodes_collide_across_samples(tmp_path):
    """The exact real-world scenario this whole module exists to guard
    against: two DIFFERENT samples reusing the identical literal Visium
    barcode string (real HEST-1k barcodes are not globally unique).
    composite_spot_id must still produce two distinct identities."""
    shared_barcodes = [f"SPOT{i}-1" for i in range(6)]  # identical raw barcodes in both samples
    hest_dir, meta_path, _ = _make_synthetic_hest1k(
        tmp_path, {"Lung": ["L0", "L1"]},
        barcodes_by_id={"L0": shared_barcodes, "L1": shared_barcodes},
    )
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=2,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        **_SMALL_BUILD_KWARGS,
    )
    l0_ids = set(manifest["samples"]["L0"]["composite_spot_ids"])
    l1_ids = set(manifest["samples"]["L1"]["composite_spot_ids"])
    assert manifest["samples"]["L0"]["barcodes"] == manifest["samples"]["L1"]["barcodes"]  # raw barcodes DO collide
    assert l0_ids.isdisjoint(l1_ids)  # but composite identities never do
    assert all_composite_spot_ids(manifest) == l0_ids | l1_ids


def test_gene_panel_is_derived_from_training_samples_only(tmp_path):
    """Regression test for the explicit instruction: 'Derive the ordered
    gene panel from training samples only... never let held-out samples
    shrink or choose the vocabulary.' Validation/test samples here carry
    EXTRA genes train samples don't have; the declared gene_panel must
    reflect only what training samples actually have."""
    train_genes = [f"GENE{i}" for i in range(6)]
    extra_val_test_genes = train_genes + [f"HELDOUT_ONLY_GENE{i}" for i in range(3)]
    sample_ids = ["L0", "L1", "L2", "L3"]

    # Determine which sample the split will actually hold out FIRST (a dry
    # run against placeholder files), so the extra-gene assignment below is
    # deterministic and doesn't depend on guessing resolve_sample_selection's
    # internal shuffle -- the real, meaningful assertion is "gene_panel only
    # reflects the reported train_sample_ids' real content", which this
    # still tests either way, but pinning WHICH sample carries the extra
    # genes to the actual held-out one makes the test's intent legible.
    dry_run_dir, dry_run_meta, _ = _make_synthetic_hest1k(
        tmp_path / "dry_run", {"Lung": sample_ids}, gene_names=train_genes, n_spots_per_sample=1,
    )
    dry_run = build_dataset_manifest(
        dry_run_dir, str(dry_run_meta), organs="all", min_samples_per_organ=4,
        n_validation_per_organ=1, n_test_per_organ=0, split_seed=0,
        check_gene_panel_compatibility=False, min_nb_genes=None, gene_min_genes_per_spot=1, gene_min_cells=0,
    )
    held_out_id = dry_run["validation_sample_ids"][0]

    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True)
    (hest_dir / "patches").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    sample_genes = {sid: (extra_val_test_genes if sid == held_out_id else train_genes) for sid in sample_ids}
    for sid, genes in sample_genes.items():
        n = 5
        barcodes = [f"{sid}-SPOT{i}-1" for i in range(n)]
        counts = rng.poisson(5, size=(n, len(genes))).astype(np.float32)
        coords = rng.uniform(0, 1000, size=(n, 2))
        adata = ad.AnnData(
            X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(genes)),
        )
        adata.obsm["spatial"] = coords
        adata.write_h5ad(hest_dir / "st" / f"{sid}.h5ad")
        (hest_dir / "patches" / f"{sid}.h5").touch()
        rows.append({"id": sid, "organ": "Lung", "st_technology": "Visium", "species": "Homo sapiens", "nb_genes": len(genes)})
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=4,
        n_validation_per_organ=1, n_test_per_organ=0, split_seed=0,
        check_gene_panel_compatibility=False,  # isolate the train-only-derivation claim from the separate compatibility filter
        min_nb_genes=None, gene_min_genes_per_spot=1, gene_min_cells=0,
    )
    assert manifest["validation_sample_ids"] == [held_out_id]
    assert set(manifest["gene_panel"]) == set(train_genes)
    assert "HELDOUT_ONLY_GENE0" not in manifest["gene_panel"]


def test_wsi_cache_provenance_recorded_when_cache_file_exists(tmp_path):
    hest_dir, meta_path, _ = _make_synthetic_hest1k(tmp_path, {"Lung": ["L0", "L1"]})
    cache_dir = hest_dir / "gigapath_slide_cache"
    cache_dir.mkdir()
    (cache_dir / "L0.npz").write_bytes(b"fake cache content")

    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=2,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        **_SMALL_BUILD_KWARGS,
    )
    assert manifest["samples"]["L0"]["wsi_cache"] is not None
    assert manifest["samples"]["L0"]["wsi_cache"]["size_bytes"] == len(b"fake cache content")
    assert manifest["samples"]["L1"]["wsi_cache"] is None


def test_save_and_load_dataset_manifest_round_trips(tmp_path):
    hest_dir, meta_path, _ = _make_synthetic_hest1k(tmp_path, {"Lung": [f"L{i}" for i in range(3)]})
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        **_SMALL_BUILD_KWARGS,
    )
    path = save_dataset_manifest(manifest, tmp_path / "manifest.json")
    assert path.is_file()
    reloaded = load_dataset_manifest(path)
    assert reloaded == manifest


def test_ensure_dataset_manifest_builds_then_reuses(tmp_path):
    hest_dir, meta_path, _ = _make_synthetic_hest1k(tmp_path, {"Lung": [f"L{i}" for i in range(3)]})
    kwargs = dict(
        hest_data_dir=hest_dir, metadata_csv=str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0, **_SMALL_BUILD_KWARGS,
    )
    path = tmp_path / "manifest.json"
    first, _ = ensure_dataset_manifest(path, **kwargs)
    assert path.is_file()
    second, _ = ensure_dataset_manifest(path, **kwargs)
    assert first == second


def test_ensure_dataset_manifest_rejects_a_manifest_that_no_longer_matches_a_fresh_rebuild(tmp_path):
    hest_dir, meta_path, _ = _make_synthetic_hest1k(tmp_path, {"Lung": [f"L{i}" for i in range(6)]})
    path = tmp_path / "manifest.json"
    ensure_dataset_manifest(
        path, hest_data_dir=hest_dir, metadata_csv=str(meta_path), organs="all", min_samples_per_organ=3,
        n_validation_per_organ=1, n_test_per_organ=1, split_seed=0, **_SMALL_BUILD_KWARGS,
    )
    with pytest.raises(ValueError, match="does not match a fresh rebuild"):
        ensure_dataset_manifest(
            path, hest_data_dir=hest_dir, metadata_csv=str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=1, n_test_per_organ=1, split_seed=1,  # different seed -> different split
            **_SMALL_BUILD_KWARGS,
        )


def test_all_composite_spot_ids_can_be_restricted_to_a_subset_of_samples(tmp_path):
    hest_dir, meta_path, _ = _make_synthetic_hest1k(tmp_path, {"Lung": ["L0", "L1"]})
    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all", min_samples_per_organ=2,
        n_validation_per_organ=0, n_test_per_organ=0, split_seed=0,
        **_SMALL_BUILD_KWARGS,
    )
    only_l0 = all_composite_spot_ids(manifest, sample_ids=["L0"])
    everyone = all_composite_spot_ids(manifest)
    assert only_l0 == set(manifest["samples"]["L0"]["composite_spot_ids"])
    assert everyone == only_l0 | set(manifest["samples"]["L1"]["composite_spot_ids"])
