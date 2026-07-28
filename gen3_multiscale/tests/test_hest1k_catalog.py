import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gen3_multiscale.data.hest1k_catalog import resolve_sample_selection, _resolve_cross_organ_patient_conflicts


def _make_fake_hest1k(tmp_path: Path, organ_sample_ids: dict[str, list[str]], technology: str = "Visium",
                       missing_patches: set[str] | None = None,
                       species_by_id: dict[str, str] | None = None,
                       nb_genes_by_id: dict[str, int] | None = None,
                       patient_by_id: dict[str, str] | None = None) -> tuple[Path, Path]:
    """Build a fake local hest1k dir + matching metadata CSV. missing_patches
    lets a test simulate a sample with expression but no image patches.
    species_by_id (default: every sample "Homo sapiens") lets a test
    simulate HEST-1k's real human/mouse mix (see
    test_species_filter_excludes_mouse_by_default). nb_genes_by_id
    (default: every sample 20000, comfortably whole-transcriptome-scale)
    lets a test simulate HEST-1k's real small-targeted-panel-mislabeled-
    Visium finding (see test_min_nb_genes_excludes_small_panel_samples).
    patient_by_id (default: no `patient` column at all, matching most of
    this file's other fixtures) lets a test simulate HEST-1k's real
    `patient` metadata column (see test_split_by_patient_*)."""
    missing_patches = missing_patches or set()
    species_by_id = species_by_id or {}
    nb_genes_by_id = nb_genes_by_id or {}
    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True)
    (hest_dir / "patches").mkdir(parents=True)

    rows = []
    for organ, ids in organ_sample_ids.items():
        for sid in ids:
            row = {
                "id": sid, "organ": organ, "st_technology": technology,
                "species": species_by_id.get(sid, "Homo sapiens"),
                "nb_genes": nb_genes_by_id.get(sid, 20000),
            }
            if patient_by_id is not None:
                row["patient"] = patient_by_id.get(sid, sid)
            rows.append(row)
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


def test_split_by_patient_keeps_every_sample_from_a_held_out_patient_together():
    """GPT-audit-flagged bug (2026-07-27, confirmed and fixed): HEST-1k's
    real metadata has a `patient` column -- multiple Visium samples can
    come from the same donor. Splitting at the sample level could put two
    slides from the same patient on opposite sides of train/test, leaking
    patient identity into what's supposed to be a clean held-out eval."""
    with tempfile.TemporaryDirectory() as tmp:
        # 3 patients x 3 slides each = 9 samples, one organ.
        patient_by_id = {f"L{i}": f"P{i // 3}" for i in range(9)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(9)]}, patient_by_id=patient_by_id,
        )
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=1, n_test_per_organ=1, split_seed=0,
            check_gene_panel_compatibility=False, split_by_patient=True,
        )
        train, val, test = result["train_sample_ids"], result["validation_sample_ids"], result["test_sample_ids"]
        assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))

        def patient_of(sid):
            return patient_by_id[sid]

        # val/test each pull one whole 3-slide patient (patients move
        # together, never split across the train/val/test boundary)
        assert len({patient_of(sid) for sid in val}) == 1
        assert len({patient_of(sid) for sid in test}) == 1
        assert len(val) == 3 and len(test) == 3  # the whole 3-slide patient moved together
        # the held-out patients' samples never appear in train
        assert not ({patient_of(sid) for sid in val} & {patient_of(sid) for sid in train})
        assert not ({patient_of(sid) for sid in test} & {patient_of(sid) for sid in train})


def test_split_by_patient_false_reproduces_the_old_sample_level_split():
    with tempfile.TemporaryDirectory() as tmp:
        patient_by_id = {f"L{i}": f"P{i // 3}" for i in range(9)}
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": [f"L{i}" for i in range(9)]}, patient_by_id=patient_by_id,
        )
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=1, n_test_per_organ=1, split_seed=0,
            check_gene_panel_compatibility=False, split_by_patient=False,
        )
        # sample-level split: exactly 1 sample in validation/test each,
        # regardless of which patient it happens to belong to
        assert len(result["validation_sample_ids"]) == 1
        assert len(result["test_sample_ids"]) == 1


def test_resolve_cross_organ_patient_conflicts_removes_the_train_side_of_a_conflict():
    """GPT-audit-flagged bug (2026-07-27, second-pass re-audit, confirmed
    and fixed): the per-organ patient split doesn't see a patient with
    samples in more than one organ. Patient P1 here has a Lung sample in
    train and a Kidney sample in test -- must be removed from train."""
    patient_by_sample = {"L0": "P0", "L1": "P1", "K0": "P1", "K1": "P2"}
    train_ids, validation_ids, test_ids = _resolve_cross_organ_patient_conflicts(
        train_ids=["L0", "L1"], validation_ids=[], test_ids=["K0", "K1"],
        patient_by_sample=patient_by_sample,
    )
    assert train_ids == ["L0"]  # L1 (patient P1) removed -- P1 is in test via K0
    assert test_ids == ["K0", "K1"]  # test is authoritative, never modified


def test_resolve_cross_organ_patient_conflicts_removes_the_validation_side_of_a_conflict():
    patient_by_sample = {"L0": "P0", "K0": "P0"}
    train_ids, validation_ids, test_ids = _resolve_cross_organ_patient_conflicts(
        train_ids=[], validation_ids=["L0"], test_ids=["K0"],
        patient_by_sample=patient_by_sample,
    )
    assert validation_ids == []
    assert test_ids == ["K0"]


def test_resolve_cross_organ_patient_conflicts_is_a_no_op_when_already_disjoint():
    patient_by_sample = {"L0": "P0", "L1": "P1", "K0": "P2", "K1": "P3"}
    train_ids, validation_ids, test_ids = _resolve_cross_organ_patient_conflicts(
        train_ids=["L0"], validation_ids=["L1"], test_ids=["K0", "K1"],
        patient_by_sample=patient_by_sample,
    )
    assert train_ids == ["L0"] and validation_ids == ["L1"] and test_ids == ["K0", "K1"]


def test_resolve_sample_selection_end_to_end_has_no_patient_spanning_multiple_splits():
    """Integration-level check through the real per-organ split logic:
    patient "SHARED" has samples in both Lung and Kidney. Regardless of
    which organ's rng draw picks it for train/val/test first, the final
    result must never place the same patient in two different splits."""
    with tempfile.TemporaryDirectory() as tmp:
        patient_by_id = {
            "L0": "SHARED", "L1": "PL1", "L2": "PL2", "L3": "PL3",
            "K0": "SHARED", "K1": "PK1", "K2": "PK2", "K3": "PK3",
        }
        hest_dir, meta_path = _make_fake_hest1k(
            Path(tmp), {"Lung": ["L0", "L1", "L2", "L3"], "Kidney": ["K0", "K1", "K2", "K3"]},
            patient_by_id=patient_by_id,
        )
        for seed in range(10):  # try several seeds -- the bug is seed-dependent
            result = resolve_sample_selection(
                hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
                n_validation_per_organ=1, n_test_per_organ=1, split_seed=seed,
                check_gene_panel_compatibility=False, split_by_patient=True,
            )
            patient_of_split = {}
            for split_name in ("train_sample_ids", "validation_sample_ids", "test_sample_ids"):
                for sid in result[split_name]:
                    patient = patient_by_id[sid]
                    assert patient not in patient_of_split or patient_of_split[patient] == split_name, (
                        f"seed={seed}: patient {patient!r} appears in both "
                        f"{patient_of_split.get(patient)!r} and {split_name!r}"
                    )
                    patient_of_split[patient] = split_name


def test_split_by_patient_falls_back_to_sample_level_when_no_patient_column():
    """No `patient` column in the metadata at all (most of this file's
    fixtures) -- split_by_patient=True must not crash, and must behave
    exactly like the old sample-level split (this file's other tests all
    already assert exact sample-level counts with split_by_patient
    defaulted to True, so this just documents the fallback explicitly)."""
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir, meta_path = _make_fake_hest1k(Path(tmp), {"Lung": [f"L{i}" for i in range(10)]})
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=2, n_test_per_organ=2, split_seed=0,
            check_gene_panel_compatibility=False, split_by_patient=True,
        )
        assert len(result["validation_sample_ids"]) == 2
        assert len(result["test_sample_ids"]) == 2
        assert len(result["train_sample_ids"]) == 6
