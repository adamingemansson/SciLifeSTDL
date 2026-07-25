"""HEST-1k metadata + local-availability catalog: the shared logic behind
both scripts/inventory_hest1k.py (human-readable report) and
resolve_sample_selection (config-driven train/validation/test split for
the training entrypoints). One implementation of "what's actually usable"
so the two never disagree.

Real column name confirmed against the live CSV (2026-07-25, after a real
KeyError on the training server): `st_technology`, not `technology` --
docs/dataset_notes.md had the same wrong name and was fixed at the same
time.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def locally_downloaded_ids(hest_data_dir: str | Path) -> tuple[set[str], set[str]]:
    """IDs with an expression file (st/*.h5ad) and IDs with a patch file
    (patches/*.h5), reported separately -- a sample downloaded for
    expression-only use may be missing patches, and every gen2 architecture
    requires both."""
    hest_data_dir = Path(hest_data_dir)
    st_dir = hest_data_dir / "st"
    patch_dir = hest_data_dir / "patches"
    st_ids = {p.stem for p in st_dir.glob("*.h5ad")} if st_dir.is_dir() else set()
    patch_ids = {p.stem for p in patch_dir.glob("*.h5")} if patch_dir.is_dir() else set()
    return st_ids, patch_ids


def usable_local_ids(hest_data_dir: str | Path) -> set[str]:
    """IDs with BOTH expression and patches present -- the only ones any
    gen2 architecture can actually use."""
    st_ids, patch_ids = locally_downloaded_ids(hest_data_dir)
    return st_ids & patch_ids


def load_visium_metadata(
    metadata_csv: str, species: str | None = "Homo sapiens", min_nb_genes: int | None = 5000,
) -> pd.DataFrame:
    """Real HEST-1k metadata CSV, filtered to Visium (st_technology).
    metadata_csv is normally the public `hf://datasets/MahmoodLab/hest/
    HEST_v1_3_0.csv` URI (docs/dataset_notes.md), but any local path works
    identically -- pandas dispatches on the string itself.

    species (real bug found 2026-07-25, first real multi-organ training
    run on the actual server): HEST-1k genuinely contains two species --
    421 "Homo sapiens" + 181 "Mus musculus" Visium samples, confirmed
    against the live CSV's real `species` column values -- and nothing
    here filtered on it before. A sample_selection spanning "all" organs
    silently mixed both, and human/mouse gene symbols essentially never
    match (`ACTB` vs `Actb`), so load_multi_sample's cross-sample gene
    intersection collapsed to exactly zero shared genes across ~240
    samples -- not a partial loss, the literal symptom this fix targets.
    Defaults to human-only (every gene-vocabulary assumption downstream --
    STPath, scFoundation -- is human-gene-symbol-based); pass
    species=None to keep every species (e.g. for a deliberate
    mouse-vs-human comparison run), or an explicit string/list to select
    otherwise.

    min_nb_genes (real SECOND bug found 2026-07-25, same server run,
    AFTER the species fix): even human-only, a multi-organ selection
    still hit a zero-gene-panel crash. Confirmed against the live data:
    HEST-1k's `st_technology == "Visium"` label is not a reliable proxy
    for "whole-transcriptome" -- samples with an id starting "TENX" carry
    that label but really have a small ~541-gene TARGETED panel (likely
    CytAssist or a specific Xenium-adjacent 10x product bundled under the
    same technology tag), exactly the Visium-vs-Xenium scale mismatch
    this project already knew to avoid, just hiding inside "Visium." Every
    other real source-study group (INT/MEND/MISC/NCBI/SPA/ZEN prefixes)
    is whole-transcriptome scale (14,808-36,601 raw genes) and mutually
    compatible (pairwise overlap >=79%, full 6-way intersection 13,234
    genes with TENX excluded, confirmed against real data). Rather than
    hardcode the "TENX" prefix (brittle -- HEST-1k could add other
    small-panel studies later), this filters on the real `nb_genes`
    metadata column HEST-1k already provides for exactly this purpose,
    with a threshold (5000) chosen with a large safety margin below
    every legitimate whole-transcriptome group's real count and far
    above TENX's real ~541. Rows with a missing/NaN nb_genes value are
    EXCLUDED (unknown panel compatibility, not assumed safe) -- pass
    min_nb_genes=None to disable this filter entirely."""
    meta = pd.read_csv(metadata_csv)
    visium = meta[meta["st_technology"] == "Visium"].copy()
    if species is not None:
        allowed = {species} if isinstance(species, str) else set(species)
        visium = visium[visium["species"].isin(allowed)]
    if min_nb_genes is not None:
        visium = visium[visium["nb_genes"] >= min_nb_genes]
    visium["organ"] = visium["organ"].fillna("(unlabeled)")
    return visium


def _real_var_names(hest_data_dir: str | Path, sample_id: str) -> set[str]:
    """The real gene panel (var_names) of one local sample, read WITHOUT
    loading its expression matrix (anndata's backed='r' mode still loads
    var/obs eagerly, only X stays lazy -- cheap enough to call for
    hundreds of samples during sample-selection resolution)."""
    import anndata as ad
    path = Path(hest_data_dir) / "st" / f"{sample_id}.h5ad"
    return set(ad.read_h5ad(path, backed="r").var_names)


def resolve_compatible_sample_ids(
    hest_data_dir: str | Path, candidate_ids: list[str],
    min_gene_coverage: float = 0.9, min_sample_coverage: float = 0.9, min_panel_size: int = 5000,
) -> tuple[list[str], list[str]]:
    """Real measured gene-panel compatibility filter (third real bug on
    the same server run, found 2026-07-25, AFTER the min_nb_genes fix --
    see load_visium_metadata's own docstring for why: HEST-1k's `nb_genes`
    metadata column turned out NOT to be a reliable size-based proxy for
    panel compatibility -- real confirmed values span CONTINUOUSLY within
    every single source-study prefix group, e.g. real TENX values include
    538, 541, 1056, ..., 5001, 10006, 10017, and real NCBI -- otherwise a
    clean whole-transcriptome group -- includes at least one 541-gene
    outlier too. A numeric size threshold cannot cleanly separate
    compatible from incompatible samples here; only real measured gene
    IDENTITY overlap can).

    Two-pass, symmetric in genes and samples:
    1. A gene is "core" if at least min_gene_coverage of candidate_ids'
       real panels contain it.
    2. A sample is "compatible" if it covers at least min_sample_coverage
       of the core genes.
    The final shared panel is the EXACT intersection of every kept
    sample's real panel (not an approximation) -- guaranteed correct,
    never silently zero-fills a gene a kept sample didn't actually
    measure. Prints which candidate_ids get excluded and why (visible,
    debuggable, not a silent shrink).

    Returns (kept_ids, shared_genes) -- kept_ids is a SUBSET of
    candidate_ids in the same relative order; callers must use kept_ids,
    not candidate_ids, for everything downstream. Raises if the final
    panel is smaller than min_panel_size (a real incompatible cohort,
    not just "a few outliers")."""
    panels = {sid: _real_var_names(hest_data_dir, sid) for sid in candidate_ids}
    n = len(candidate_ids)
    gene_counts: Counter = Counter()
    for panel in panels.values():
        gene_counts.update(panel)
    core_genes = {g for g, c in gene_counts.items() if c / n >= min_gene_coverage}

    kept, dropped = [], []
    for sid in candidate_ids:
        coverage = len(panels[sid] & core_genes) / len(core_genes) if core_genes else 0.0
        (kept if coverage >= min_sample_coverage else dropped).append(sid)

    if dropped:
        preview = dropped[:20]
        print(
            f"resolve_compatible_sample_ids: excluded {len(dropped)}/{n} sample(s) with "
            f"incompatible real gene panels (covered <{min_sample_coverage:.0%} of the "
            f"{len(core_genes)}-gene core panel): {preview}"
            f"{', ...' if len(dropped) > len(preview) else ''}"
        )

    shared_genes = sorted(set.intersection(*(panels[sid] for sid in kept))) if kept else []
    if len(shared_genes) < min_panel_size:
        raise ValueError(
            f"real gene-panel compatibility check left only {len(shared_genes)} shared genes "
            f"across {len(kept)}/{n} kept samples (min_panel_size={min_panel_size}) -- "
            f"this candidate cohort is not compatible enough for multi-sample training; "
            "try excluding some organs/source studies explicitly, or lower min_panel_size "
            "if a smaller shared panel is acceptable"
        )
    return kept, shared_genes


def resolve_sample_selection(
    hest_data_dir: str | Path, metadata_csv: str,
    organs: list[str] | str = "all",
    species: str | list[str] | None = "Homo sapiens",
    min_nb_genes: int | None = 5000,
    min_samples_per_organ: int = 3,
    max_samples_per_organ: int | None = None,
    n_validation_per_organ: int = 1,
    n_test_per_organ: int = 1,
    split_seed: int = 0,
    check_gene_panel_compatibility: bool = True,
    min_gene_coverage: float = 0.9,
    min_sample_coverage: float = 0.9,
    min_panel_size: int = 5000,
) -> dict:
    """Deterministically resolve train/validation/test sample IDs from the
    real HEST-1k Visium catalog, restricted to what's ACTUALLY downloaded
    and usable locally (never a sample the training run couldn't load).

    organs: "all" (every organ with enough local samples) or an explicit
    list of organ names to restrict to (e.g. ["Lung"] for Architecture 4's
    single-organ constraint).

    species: defaults to human-only (see load_visium_metadata's own
    docstring for the real zero-gene-intersection bug this guards
    against -- HEST-1k genuinely mixes human and mouse Visium samples).

    min_nb_genes: defaults to 5000, excluding small-targeted-panel
    samples mislabeled "Visium" (see load_visium_metadata's own
    docstring for the real TENX-prefix finding this guards against) --
    a cheap first pass on metadata alone, kept even though it isn't
    fully reliable on its own (see check_gene_panel_compatibility below).

    check_gene_panel_compatibility: defaults to True. Real THIRD bug
    found 2026-07-25 on the same server run: min_nb_genes alone was not
    enough -- HEST-1k's nb_genes values are NOT cleanly bimodal, they
    vary continuously within every source-study group, so a size
    threshold cannot reliably separate compatible from incompatible
    samples. When True, resolve_compatible_sample_ids (real measured
    gene-identity overlap, see its own docstring) runs on the union of
    the resolved train/validation/test ids AFTER the per-organ split
    below, dropping any sample whose real panel doesn't sufficiently
    match the cohort's consensus core panel. Organs that lose every
    train sample this way are dropped from organ_vocab entirely; organs
    that only lose their validation/test sample(s) keep their remaining
    train samples (a real but minor degradation, preferred over raising
    for one held-out sample). min_gene_coverage/min_sample_coverage/
    min_panel_size are passed straight through -- see that function's
    docstring for what they mean.

    min_samples_per_organ: organs with fewer usable local samples than
    this are excluded entirely -- below n_validation_per_organ +
    n_test_per_organ + 1 there's no meaningful train/val/test split for
    that organ (e.g. the catalog's several 1-sample organs).

    max_samples_per_organ: if set, caps each organ's sample count via a
    deterministic seeded subset (not just the first N alphabetically/by
    ID, which could introduce an unintended ordering bias) -- keeps data
    loading/caching time bounded for very large organs (Brain: 121
    samples) without silently favoring whichever samples happen to sort
    first.

    Returns a dict: train_sample_ids, validation_sample_ids, test_sample_ids
    (all list[str]), organ_by_sample, tech_by_sample (both dict[str, str]),
    organ_vocab, tech_vocab (both list[str], sorted-unique -- ready to pass
    straight into OrganTechEmbedding via build_organ_tech_vocab-equivalent
    ordering).
    """
    import random

    visium = load_visium_metadata(metadata_csv, species=species, min_nb_genes=min_nb_genes)
    usable = usable_local_ids(hest_data_dir)
    visium = visium[visium["id"].isin(usable)]

    if organs != "all":
        requested = set(organs)
        missing = requested - set(visium["organ"])
        if missing:
            raise ValueError(
                f"requested organs {sorted(missing)} have zero usable local Visium samples "
                f"(cross-checked against {hest_data_dir}) -- run scripts/inventory_hest1k.py "
                "to see real local coverage before requesting an organ"
            )
        visium = visium[visium["organ"].isin(requested)]

    rng = random.Random(split_seed)
    train_ids, validation_ids, test_ids = [], [], []
    organ_by_sample, tech_by_sample = {}, {}
    kept_organs = []
    for organ, group in visium.groupby("organ"):
        ids = sorted(group["id"].tolist())
        if len(ids) < min_samples_per_organ:
            continue
        if len(ids) < n_validation_per_organ + n_test_per_organ + 1:
            continue  # not enough left over for a real training set after holding out val/test
        shuffled = list(ids)
        rng.shuffle(shuffled)
        if max_samples_per_organ is not None:
            shuffled = shuffled[:max_samples_per_organ]
            if len(shuffled) < n_validation_per_organ + n_test_per_organ + 1:
                continue  # the cap itself left too few samples for this organ
        organ_val = shuffled[:n_validation_per_organ]
        organ_test = shuffled[n_validation_per_organ:n_validation_per_organ + n_test_per_organ]
        organ_train = shuffled[n_validation_per_organ + n_test_per_organ:]
        validation_ids.extend(organ_val)
        test_ids.extend(organ_test)
        train_ids.extend(organ_train)
        kept_organs.append(organ)
        for sid in shuffled:
            organ_by_sample[sid] = organ
            tech_by_sample[sid] = "Visium"

    if not train_ids:
        raise ValueError(
            f"no organs had enough usable local samples (min_samples_per_organ={min_samples_per_organ}, "
            f"requested organs={organs}) -- run scripts/inventory_hest1k.py against {hest_data_dir} "
            "to see what's actually available"
        )

    if check_gene_panel_compatibility:
        all_ids = sorted(set(train_ids) | set(validation_ids) | set(test_ids))
        kept_ids, _shared_genes = resolve_compatible_sample_ids(
            hest_data_dir, all_ids, min_gene_coverage=min_gene_coverage,
            min_sample_coverage=min_sample_coverage, min_panel_size=min_panel_size,
        )
        kept_set = set(kept_ids)
        train_ids = [sid for sid in train_ids if sid in kept_set]
        validation_ids = [sid for sid in validation_ids if sid in kept_set]
        test_ids = [sid for sid in test_ids if sid in kept_set]
        organ_by_sample = {sid: organ for sid, organ in organ_by_sample.items() if sid in kept_set}
        tech_by_sample = {sid: tech for sid, tech in tech_by_sample.items() if sid in kept_set}
        # An organ only stays in the vocabulary if it still has real
        # TRAINING samples -- losing just its validation/test sample(s)
        # to the compatibility check is a real but minor degradation
        # (that organ simply has no held-out eval this run), not a
        # reason to drop the organ (and its training signal) entirely.
        kept_organs = sorted({organ_by_sample[sid] for sid in train_ids})
        if not train_ids:
            raise ValueError(
                "every training sample was excluded by the real gene-panel compatibility "
                "check -- this organ/sample_selection combination has no mutually "
                "compatible cohort; try a different organs list or split_seed"
            )

    return {
        "train_sample_ids": sorted(train_ids),
        "validation_sample_ids": sorted(validation_ids),
        "test_sample_ids": sorted(test_ids),
        "organ_by_sample": organ_by_sample,
        "tech_by_sample": tech_by_sample,
        "organ_vocab": sorted(kept_organs),
        "tech_vocab": ["Visium"],
    }
