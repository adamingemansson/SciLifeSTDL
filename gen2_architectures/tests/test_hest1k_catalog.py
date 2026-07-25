import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gen2_architectures.data.hest1k_catalog import resolve_sample_selection


def _make_fake_hest1k(tmp_path: Path, organ_sample_ids: dict[str, list[str]], technology: str = "Visium",
                       missing_patches: set[str] | None = None,
                       species_by_id: dict[str, str] | None = None,
                       nb_genes_by_id: dict[str, int] | None = None) -> tuple[Path, Path]:
    """Build a fake local hest1k dir + matching metadata CSV. missing_patches
    lets a test simulate a sample with expression but no image patches.
    species_by_id (default: every sample "Homo sapiens") lets a test
    simulate HEST-1k's real human/mouse mix (see
    test_species_filter_excludes_mouse_by_default). nb_genes_by_id
    (default: every sample 20000, comfortably whole-transcriptome-scale)
    lets a test simulate HEST-1k's real small-targeted-panel-mislabeled-
    Visium finding (see test_min_nb_genes_excludes_small_panel_samples)."""
    missing_patches = missing_patches or set()
    species_by_id = species_by_id or {}
    nb_genes_by_id = nb_genes_by_id or {}
    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True)
    (hest_dir / "patches").mkdir(parents=True)

    rows = []
    for organ, ids in organ_sample_ids.items():
        for sid in ids:
            rows.append({
                "id": sid, "organ": organ, "st_technology": technology,
                "species": species_by_id.get(sid, "Homo sapiens"),
                "nb_genes": nb_genes_by_id.get(sid, 20000),
            })
            (hest_dir / "st" / f"{sid}.h5ad").touch()
            if sid not in missing_patches:
                (hest_dir / "patches" / f"{sid}.h5").touch()
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return hest_dir, meta_path


def test_basic_split_respects_counts_and_is_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {
            "Lung": [f"L{i}" for i in range(10)],
            "Kidney": [f"K{i}" for i in range(8)],
        })
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all",
            min_samples_per_organ=3, n_validation_per_organ=2, n_test_per_organ=2, split_seed=0,
            check_gene_panel_compatibility=False,
        )
        train, val, test = set(result["train_sample_ids"]), set(result["validation_sample_ids"]), set(result["test_sample_ids"])
        assert not (train & val)
        assert not (train & test)
        assert not (val & test)
        # Lung: 10 total -> 2 val + 2 test + 6 train. Kidney: 8 -> 2+2+4.
        assert len([s for s in train if s.startswith("L")]) == 6
        assert len([s for s in val if s.startswith("L")]) == 2
        assert len([s for s in test if s.startswith("L")]) == 2
        assert len([s for s in train if s.startswith("K")]) == 4
        assert result["organ_vocab"] == ["Kidney", "Lung"]
        assert result["tech_vocab"] == ["Visium"]


def test_min_samples_per_organ_excludes_small_organs():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {
            "Lung": [f"L{i}" for i in range(10)],
            "Embryo": ["E0"],  # single sample -- must be excluded
        })
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", min_samples_per_organ=3, check_gene_panel_compatibility=False)
        assert "Embryo" not in result["organ_vocab"]
        assert not any(s.startswith("E") for s in result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"])


def test_max_samples_per_organ_caps_deterministically():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {"Brain": [f"B{i}" for i in range(50)]})
        result_a = resolve_sample_selection(hest_dir, str(meta_path), organs="all", max_samples_per_organ=10, split_seed=42, check_gene_panel_compatibility=False)
        result_b = resolve_sample_selection(hest_dir, str(meta_path), organs="all", max_samples_per_organ=10, split_seed=42, check_gene_panel_compatibility=False)
        total_a = len(result_a["train_sample_ids"]) + len(result_a["validation_sample_ids"]) + len(result_a["test_sample_ids"])
        assert total_a == 10
        assert result_a["train_sample_ids"] == result_b["train_sample_ids"], "same seed must reproduce the same split"


def test_different_seeds_give_different_splits():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {"Brain": [f"B{i}" for i in range(50)]})
        result_a = resolve_sample_selection(hest_dir, str(meta_path), organs="all", max_samples_per_organ=10, split_seed=1, check_gene_panel_compatibility=False)
        result_b = resolve_sample_selection(hest_dir, str(meta_path), organs="all", max_samples_per_organ=10, split_seed=2, check_gene_panel_compatibility=False)
        assert result_a["train_sample_ids"] != result_b["train_sample_ids"]


def test_explicit_organ_list_restricts_selection():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {
            "Lung": [f"L{i}" for i in range(10)],
            "Kidney": [f"K{i}" for i in range(10)],
        })
        result = resolve_sample_selection(hest_dir, str(meta_path), organs=["Lung"], check_gene_panel_compatibility=False)
        assert result["organ_vocab"] == ["Lung"]
        assert all(s.startswith("L") for s in result["train_sample_ids"])


def test_requesting_an_organ_with_zero_local_coverage_raises():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {"Lung": [f"L{i}" for i in range(10)]})
        with pytest.raises(ValueError, match="Kidney"):
            resolve_sample_selection(hest_dir, str(meta_path), organs=["Kidney"], check_gene_panel_compatibility=False)


def test_samples_missing_patches_are_excluded_even_if_expression_exists():
    with tempfile.TemporaryDirectory() as tmp:
        ids = [f"L{i}" for i in range(10)]
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": ids}, missing_patches={"L0", "L1", "L2"},
        )
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", min_samples_per_organ=3, check_gene_panel_compatibility=False)
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert "L0" not in all_selected and "L1" not in all_selected and "L2" not in all_selected
        assert len(all_selected) == 7


def test_non_visium_samples_are_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Breast": [f"X{i}" for i in range(10)]}, technology="Xenium",
        )
        with pytest.raises(ValueError):
            resolve_sample_selection(hest_dir, str(meta_path), organs="all", min_samples_per_organ=3, check_gene_panel_compatibility=False)


def test_organ_by_sample_and_tech_by_sample_cover_every_selected_id():
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {"Lung": [f"L{i}" for i in range(10)]})
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", check_gene_panel_compatibility=False)
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        for sid in all_selected:
            assert result["organ_by_sample"][sid] == "Lung"
            assert result["tech_by_sample"][sid] == "Visium"


def test_species_filter_excludes_mouse_by_default():
    """Real bug found 2026-07-25 on the actual training server: HEST-1k
    genuinely mixes 421 human + 181 mouse Visium samples, and nothing
    filtered on species before this fix -- an "organs: all" run silently
    combined both, and human/mouse gene symbols essentially never match,
    so load_multi_sample's cross-sample gene intersection collapsed to
    exactly zero shared genes. species defaults to human-only now."""
    with tempfile.TemporaryDirectory() as tmp:
        mouse_ids = {f"L{i}": "Mus musculus" for i in range(5)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(10)]}, species_by_id=mouse_ids,
        )
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", min_samples_per_organ=3, check_gene_panel_compatibility=False)
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert set(all_selected) == {f"L{i}" for i in range(5, 10)}, "only the 5 human samples should survive"


def test_min_nb_genes_excludes_small_panel_samples():
    """Real bug found 2026-07-25 on the actual training server, AFTER the
    species fix: HEST-1k's real 'TENX'-prefixed sample ids carry the
    st_technology='Visium' label but really have a small ~541-gene
    targeted panel -- confirmed against real data, this alone collapsed
    a multi-organ cross-sample gene intersection to zero. min_nb_genes
    (default 5000) excludes them using the real nb_genes metadata
    column, without hardcoding the "TENX" prefix."""
    with tempfile.TemporaryDirectory() as tmp:
        small_panel_ids = {f"L{i}": 541 for i in range(5)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(10)]}, nb_genes_by_id=small_panel_ids,
        )
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", min_samples_per_organ=3, check_gene_panel_compatibility=False)
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert set(all_selected) == {f"L{i}" for i in range(5, 10)}, "only the 5 whole-transcriptome samples should survive"


def test_min_nb_genes_none_disables_the_filter():
    with tempfile.TemporaryDirectory() as tmp:
        small_panel_ids = {f"L{i}": 541 for i in range(5)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(10)]}, nb_genes_by_id=small_panel_ids,
        )
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_nb_genes=None, min_samples_per_organ=3,
            check_gene_panel_compatibility=False,
        )
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert len(all_selected) == 10, "min_nb_genes=None must keep every sample regardless of panel size"


def test_species_all_keeps_every_species():
    with tempfile.TemporaryDirectory() as tmp:
        mouse_ids = {f"L{i}": "Mus musculus" for i in range(5)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(10)]}, species_by_id=mouse_ids,
        )
        result = resolve_sample_selection(hest_dir, str(meta_path), organs="all", species=None, min_samples_per_organ=3, check_gene_panel_compatibility=False)
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert len(all_selected) == 10, "species=None must keep both human and mouse samples"
