import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gen2_architectures.data.hest1k_catalog import resolve_compatible_sample_ids, resolve_sample_selection


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


def test_resolve_sample_selection_derives_panel_from_train_only_not_pooled_with_held_out():
    """9th Codex re-audit of commit 9592d9e, finding #1 (CRITICAL,
    confirmed): resolve_sample_selection used to compute the "core
    genes" / final shared panel from train+validation+test IDs POOLED
    together, so a validation sample's incomplete real panel could
    shrink the final panel used for TRAINING even though every training
    sample was, on its own, fully 30-gene-compatible. Regression: 11
    same-organ samples all start with an identical, fully compatible
    30-gene panel; whichever land in validation (determined by a dry
    run, not hardcoded -- the split RNG's exact assignment isn't this
    test's concern) are then rewritten to a real panel missing 5 of
    those 30 genes (25/30 = 83% coverage of the training panel, i.e.
    below the 90% min_sample_coverage threshold against the TRUE
    training reference, but -- under the old buggy pooled logic --
    enough to silently redefine "core" down to 25 genes and get
    accepted anyway, silently truncating the panel actually used for
    training). Under the fixed train-only logic: (a) no training sample
    is dropped and the frozen shared panel derived from train_ids alone
    is still all 30 genes, and (b) the validation sample(s) are
    correctly EXCLUDED from validation_sample_ids for failing to cover
    90% of that frozen training panel, without ever touching train_ids
    or the panel itself."""
    core_genes = [f"CORE{i}" for i in range(30)]
    all_ids = [f"S{i}" for i in range(13)]
    with tempfile.TemporaryDirectory() as tmp:
        hest_dir = Path(tmp) / "hest1k"
        (hest_dir / "patches").mkdir(parents=True)
        for sid in all_ids:
            _write_real_sample(hest_dir, sid, core_genes)
            (hest_dir / "patches" / f"{sid}.h5").touch()
        meta_path = _fake_metadata(Path(tmp), all_ids)

        # Dry run (no compatibility check) purely to learn which IDs the
        # seeded split assigns to validation -- never hardcode a guess
        # about the split RNG's internal behavior.
        dry_run = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=2, n_test_per_organ=0, split_seed=0,
            check_gene_panel_compatibility=False,
        )
        train_ids = dry_run["train_sample_ids"]
        validation_ids = dry_run["validation_sample_ids"]
        assert len(validation_ids) == 2

        # Now degrade ONLY the validation-designated samples' real panels.
        degraded_genes = core_genes[:25]
        for sid in validation_ids:
            _write_real_sample(hest_dir, sid, degraded_genes)

        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            n_validation_per_organ=2, n_test_per_organ=0, split_seed=0,
            check_gene_panel_compatibility=True,
            min_gene_coverage=0.9, min_sample_coverage=0.9, min_panel_size=5,
        )
        # (a) train_ids are completely unaffected by the degraded held-out panels.
        assert set(result["train_sample_ids"]) == set(train_ids)
        _kept_train, shared_genes = resolve_compatible_sample_ids(
            hest_dir, result["train_sample_ids"],
            min_gene_coverage=0.9, min_sample_coverage=0.9, min_panel_size=5,
        )
        assert set(shared_genes) == set(core_genes)  # still all 30, not truncated to 25

        # (b) the degraded validation samples are excluded for failing to
        # cover 90% of the FROZEN training panel (25/30 = 83% < 90%).
        assert set(result["validation_sample_ids"]).isdisjoint(set(validation_ids))


def test_resolve_sample_selection_end_to_end_drops_incompatible_organ_member():
    """The full integration this whole investigation was chasing: a
    resolve_sample_selection call whose per-organ candidate pool includes
    one real-panel-incompatible sample must still succeed, silently
    dropping only that one sample rather than crashing the whole run.

    Compatibility is now train-only (see the train-only regression test
    above), so with only 9 compatible samples total the exact threshold
    math depends on whether the split happens to place OUTLIER inside
    train_ids itself (a real, separate small-N boundary effect: with
    OUTLIER counted as one of only 9 train candidates, genes it doesn't
    share fall to 8/9=0.889 coverage, just under the 0.9 threshold,
    which can legitimately exclude the whole cohort -- not a bug, just a
    different scenario than "one held-out-quality outlier gets dropped
    from an otherwise-untouched training set"). This test's intent is
    the latter, so it picks (via a cheap dry run, never a hardcoded
    guess) a split_seed where OUTLIER lands outside train_ids."""
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
        split_seed = next(
            seed for seed in range(50)
            if "OUTLIER" not in resolve_sample_selection(
                hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
                split_seed=seed, check_gene_panel_compatibility=False,
            )["train_sample_ids"]
        )
        result = resolve_sample_selection(
            hest_dir, str(meta_path), organs="all", min_samples_per_organ=3,
            split_seed=split_seed,
            min_gene_coverage=0.9, min_sample_coverage=0.9, min_panel_size=5,
        )
        all_selected = result["train_sample_ids"] + result["validation_sample_ids"] + result["test_sample_ids"]
        assert "OUTLIER" not in all_selected
        assert set(all_selected) == set(compatible_ids)
