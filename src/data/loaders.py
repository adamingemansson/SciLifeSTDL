"""
Data loading built around AnnData / SpatialData, the standard ST data model.

An AnnData object `adata` is expected to have, at minimum:
    adata.X                    -> [n_cells/spots, n_genes] expression matrix
    adata.obsm['spatial']      -> [n_cells/spots, 2] in-plane coordinates
    adata.obs['slice_id']      -> which physical section a point came from
    adata.obs['z']             -> z-axis / depth coordinate for that slice
                                   (fill in from known section spacing if not
                                   provided by the source dataset)
    adata.obs['cell_type']     -> optional, for cell-type-aware models
"""
from __future__ import annotations
from pathlib import Path
import re

import anndata as ad
import numpy as np


def _resolve_hest_sample_file(
    hest_data_dir: str | Path,
    sample_id: str,
    suffix: str,
    required_path_part: str | None = None,
) -> Path:
    """Resolve one HEST file without prefix-colliding sample identifiers.

    The historical ``rglob(f"*{sample_id}*")`` lookup allowed ``INT1`` to
    select ``INT10``--``INT19``. Which file appeared first depended on each
    server's directory traversal order, so two machines could silently train
    on different samples under the same configured ID. Prefer the canonical
    exact filename and otherwise allow a single delimiter-bounded legacy name
    such as ``TEST_INT1.h5``. Any ambiguity fails loudly.
    """
    root = Path(hest_data_dir)
    candidates = sorted(
        path for path in root.rglob(f"*{suffix}")
        if required_path_part is None or required_path_part in path.parts
    )
    exact = [path for path in candidates if path.name == f"{sample_id}{suffix}"]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"Multiple exact {sample_id!r} {suffix} files found under {root}: "
            + ", ".join(map(str, exact))
        )

    token = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(str(sample_id))}(?![A-Za-z0-9])"
    )
    bounded = [path for path in candidates if token.search(path.stem)]
    if len(bounded) == 1:
        return bounded[0]
    if len(bounded) > 1:
        raise ValueError(
            f"Ambiguous delimiter-bounded files for sample_id={sample_id!r} "
            f"under {root}: " + ", ".join(map(str, bounded))
        )

    near = [path for path in candidates if str(sample_id) in path.stem]
    detail = f" Prefix-only near matches: {', '.join(map(str, near[:10]))}" if near else ""
    raise FileNotFoundError(
        f"No unambiguous {suffix} file found for sample_id={sample_id!r} under {root}."
        + detail
    )


def load_multi_slice(paths: list[str | Path], z_positions: list[float] | None = None
                      ) -> ad.AnnData:
    """
    Load and concatenate several single-slice files (h5ad/Visium/etc.) into
    one AnnData with a 'z' column, so downstream code always sees one 3D
    point cloud regardless of the source format.

    z_positions: physical depth of each slice (e.g. mm from a reference
    section). If None, uses integer slice index as a placeholder — REPLACE
    with real spacing before doing anything quantitative with the z-axis.
    """
    adatas = []
    for i, p in enumerate(paths):
        a = ad.read_h5ad(p) if str(p).endswith(".h5ad") else ad.read(p)
        z = z_positions[i] if z_positions is not None else float(i)
        a.obs["slice_id"] = Path(p).stem
        a.obs["z"] = z
        adatas.append(a)
    combined = ad.concat(adatas, join="outer", label="slice_id_batch")
    return combined


EXPRESSION_STATE_KEY = "_scilifestdl_expression_state"


def basic_qc_and_normalize(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    transform: str = "normalize_log1p",
    target_sum: float = 1e4,
    force: bool = False,
) -> ad.AnnData:
    """Apply the repository's single, explicit expression preprocessing contract.

    ``transform`` is one of:

    - ``"normalize_log1p"``: library-size normalize to ``target_sum`` and
      apply exactly one ``log1p`` (the recommended/default contract).
    - ``"normalize"``: library-size normalize only.
    - ``"none"``: QC-filter only; leave expression values untouched.

    The applied state is recorded in ``adata.uns[EXPRESSION_STATE_KEY]``. A
    second call with the same contract is a no-op instead of applying another
    normalization/log transform. A conflicting second call raises unless
    ``force=True``. This prevents the accidental double-``log1p`` that used to
    occur when the loader and a context encoder both transformed ``adata.X``.
    """
    import scanpy as sc

    allowed = {"normalize_log1p", "normalize", "none"}
    if transform not in allowed:
        raise ValueError(f"unknown expression transform {transform!r}; expected one of {sorted(allowed)}")

    requested = {
        "transform": transform,
        "target_sum": float(target_sum),
        "min_genes": int(min_genes),
        "min_cells": int(min_cells),
    }
    existing = adata.uns.get(EXPRESSION_STATE_KEY)
    if existing is not None and not force:
        existing_dict = dict(existing)
        if existing_dict == requested:
            return adata
        raise ValueError(
            "AnnData already has expression preprocessing with a different SciLifeSTDL "
            f"state: existing={existing_dict}, requested={requested}. Pass force=True "
            "only when intentionally rebuilding from untransformed values."
        )

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # Stash raw (post-QC, pre-normalization) counts and each spot's true
    # total count BEFORE normalize_total rescales adata.X in place. This is
    # for notebook-comparable raw-log1p evaluation only (STPath's own
    # preprocessing/the reference notebook never library-size-normalize —
    # see src/models/stpath_encoder.py's input_already_log1p docstring) and
    # never feeds training/model input, which stays on the transform above.
    # The library size must be captured HERE, before any later shared-gene-
    # panel subsetting (loaders.load_multi_sample's a[:, shared_genes]),
    # since that subsetting would otherwise silently shrink the sum.
    raw_x = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    adata.obs["_scilifestdl_raw_library_size"] = np.asarray(raw_x.sum(axis=1)).ravel()
    adata.layers["raw_counts"] = adata.X.copy()
    if transform in {"normalize", "normalize_log1p"}:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if transform == "normalize_log1p":
        sc.pp.log1p(adata)
    adata.uns[EXPRESSION_STATE_KEY] = requested
    return adata


def expression_is_log1p(adata: ad.AnnData) -> bool:
    """Return whether ``adata.X`` follows the recorded single-log contract."""
    state = adata.uns.get(EXPRESSION_STATE_KEY, {})
    return dict(state).get("transform") == "normalize_log1p"


def get_coords_3d(adata: ad.AnnData) -> np.ndarray:
    """[N, 3] array of (x, y, z) for every point — the shared spatial index
    that both the inter-slice and intra-slice tasks are built on."""
    xy = adata.obsm["spatial"]
    z = adata.obs["z"].to_numpy()[:, None]
    return np.concatenate([xy, z], axis=1)


def load_hest_patches(hest_data_dir: str | Path, sample_id: str
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    H&E patches for one HEST-1k sample (task #17, docs/architecture_plan.md
    "Known gaps" — the H&E branch). HEST-1k pre-extracts per-spot patches
    into patches/{sample_id}.h5. Real h5py keys, confirmed 2026-07-15 by
    inspecting an actual downloaded file (corrects the earlier
    2026-07-14 note, which was based on HESTData.dump_patches() source
    reading alone and got two details wrong): 'img' [N,224,224,3] uint8
    (224x224, NOT 256x256), 'coords' [N,2], 'barcode' [N,1] object
    (singular key name, NOT 'barcodes', and 2D not 1D). No raw
    WSI/openslide handling needed.

    Same download command as load_hest_sample already pulls this file
    (docs/dataset_notes.md's `allow_patterns=["*INT1[_.]**"]` matches
    "patches/INT1.h5" the same way it matches "st/INT1.h5ad") — if this
    raises FileNotFoundError, re-run that download command rather than
    assuming a separate one is needed.

    Returns (patches [N,224,224,3] uint8, barcodes [N] str) in whatever
    order the .h5 file stores them — NOT necessarily aligned to any
    AnnData's obs order. Use align_patches_to_adata() for that.
    """
    import h5py
    hest_data_dir = Path(hest_data_dir)
    patch_path = _resolve_hest_sample_file(
        hest_data_dir, sample_id, ".h5", required_path_part="patches"
    )
    print(f"load_hest_patches: resolved {sample_id} -> {patch_path}")
    with h5py.File(patch_path, "r") as f:
        patches = f["img"][:]
        raw_barcodes = f["barcode"][:, 0]  # [N, 1] object array -> [N]
        barcodes = np.array([b.decode() if isinstance(b, bytes) else b for b in raw_barcodes])
    return patches, barcodes



def load_hest_patch_barcodes(hest_data_dir: str | Path, sample_id: str) -> np.ndarray:
    """Read only patch barcodes without materializing the large image array."""
    import h5py
    hest_data_dir = Path(hest_data_dir)
    patch_path = _resolve_hest_sample_file(
        hest_data_dir, sample_id, ".h5", required_path_part="patches"
    )
    print(f"load_hest_patch_barcodes: resolved {sample_id} -> {patch_path}")
    with h5py.File(patch_path, "r") as f:
        raw = f["barcode"][:, 0]
        return np.array([b.decode() if isinstance(b, bytes) else str(b) for b in raw])


def align_adata_to_patch_barcodes(adata: ad.AnnData, barcodes: np.ndarray) -> ad.AnnData:
    """Apply the same H&E-coverage cohort filter without loading pixels."""
    available = set(map(str, barcodes))
    has_patch = np.asarray([str(name) in available for name in adata.obs_names], dtype=bool)
    if not has_patch.any():
        raise ValueError("None of the adata spots matched any H&E patch barcode")
    n_dropped = int((~has_patch).sum())
    if n_dropped:
        print(f"align_adata_to_patch_barcodes: dropping {n_dropped}/{adata.n_obs} spots without H&E coverage")
    return adata[has_patch].copy()

def align_patches_to_adata(adata: ad.AnnData, patches: np.ndarray, barcodes: np.ndarray
                            ) -> tuple[ad.AnnData, np.ndarray]:
    """Subset adata to only the spots with a matching H&E patch, and
    return (filtered_adata, aligned_patches) in that same (filtered)
    order. HEST-1k's own patch extraction naturally drops some spots
    (tissue-mask/WSI-border edge cases) — confirmed 2026-07-15 on real
    INT1 data: 49/1080 spots had no matching patch, a normal ~4.5% gap in
    HEST-1k's own pipeline, not a data-mismatch bug (an earlier version of
    this function raised on ANY gap, which was too strict for real data).
    Only raises if NONE of the spots match at all — that would indicate a
    genuine version/sample mismatch, not normal partial coverage."""
    barcode_to_idx = {b: i for i, b in enumerate(barcodes)}
    has_patch = np.array([name in barcode_to_idx for name in adata.obs_names])
    if not has_patch.any():
        raise ValueError(
            "None of the adata spots matched any H&E patch barcode — patches "
            "and expression data are likely from different downloads/versions "
            "of this sample."
        )
    n_dropped = int((~has_patch).sum())
    if n_dropped:
        print(f"align_patches_to_adata: dropping {n_dropped}/{adata.n_obs} spots "
              f"with no matching H&E patch (HEST-1k's own patch extraction misses "
              f"some spots at tissue/WSI edges — this is normal)")
    filtered_adata = adata[has_patch].copy()
    order = [barcode_to_idx[name] for name in filtered_adata.obs_names]
    return filtered_adata, patches[order]


def load_hest_sample(hest_data_dir: str | Path, sample_id: str,
                      organ: str | None = None, tech: str | None = None) -> ad.AnnData:
    """
    Load one HEST-1k sample (docs/dataset_notes.md Track A primary pick) —
    a single 2D section, not part of a serial z-series, so `z` is a constant
    placeholder. Track A only; Track B needs load_multi_slice with real
    z-spacing instead.

    Expects hest_data_dir already populated via HEST-1k's own download_hest
    (huggingface_hub-based, needs a free HF account + auth token — see
    docs/dataset_notes.md). Searches for the sample's .h5ad file by pattern
    rather than assuming an exact folder nesting depth, since that layout
    wasn't independently confirmed byte-for-byte from documentation alone.

    organ/tech (2026-07-16, multi-sample training + OrganTechEmbedding
    follow-up, src/models/conditioning.py): supplied BY THE CALLER, not
    auto-parsed from HEST-1k's metadata CSV — that CSV's exact
    organ/technology column format was never independently verified in
    this project (see load_multi_sample's own docstring on the same
    caution), and a wrong silent auto-parse would be far worse than
    requiring the caller to state it explicitly. Default "unknown" when
    not given, so every sample always has SOME value (OrganTechEmbedding's
    vocabulary just needs to be built to include "unknown" too, via
    build_organ_tech_vocab, if any sample omits these)."""
    hest_data_dir = Path(hest_data_dir)
    sample_path = _resolve_hest_sample_file(hest_data_dir, sample_id, ".h5ad")
    print(f"load_hest_sample: resolved {sample_id} -> {sample_path}")
    adata = ad.read_h5ad(sample_path)
    # HEST-1k already uses the standard scanpy spatial convention
    # (adata.obsm['spatial']), so no coordinate remapping needed here.
    adata.obs["slice_id"] = sample_id
    adata.obs["z"] = 0.0
    adata.obs["organ"] = organ if organ is not None else "unknown"
    adata.obs["tech"] = tech if tech is not None else "unknown"
    return adata


def load_multi_sample(hest_data_dir: str | Path, sample_ids: list[str],
                       min_genes: int = 200, min_cells: int = 3,
                       organs: list[str] | None = None,
                       techs: list[str] | None = None,
                       expression_transform: str = "normalize_log1p",
                       expression_target_sum: float = 1e4,
                       reference_genes: list[str] | None = None) -> list[ad.AnnData]:
    """Multiple INDEPENDENT HEST-1k samples (different patients/sections,
    not a serial z-series of the same tissue block — for that, use
    load_multi_slice instead) for multi-sample training (scaffolding,
    2026-07-15, ahead of the eventual "real" multi-sample training run
    discussed after task #19). Reuses load_hest_sample +
    basic_qc_and_normalize per sample (no new per-sample loading logic)
    rather than reinventing what already works for the single-sample
    case.

    Returns a LIST of AnnData, one per sample — deliberately NOT
    ad.concat()'d into one pooled object the way load_multi_slice does
    for Track B's slices. Track B's slices are legitimately part of one
    spatial 3D volume (cross-slice attention in the context encoder is
    the whole point); independent HEST-1k samples are not — spot (100,
    200) in one patient's section has no real spatial relationship to
    spot (100, 200) in another's, so a k-NN/attention context encoder
    must never be allowed to mix them. Keeping samples separate here
    means callers (see MultiSampleMaskedContextQueryDataset in
    src/training/train.py) draw each masking split from exactly ONE
    sample's own coordinate system, never blending across samples.

    All samples ARE aligned to a SHARED gene panel. By default this is the
    intersection of every supplied sample's post-QC var_names in a fixed
    sorted order. When ``reference_genes`` is supplied (the strict held-out
    path), each sample is only reordered/subset to that already fit-derived
    panel and missing genes raise; held-out test samples therefore cannot
    participate in choosing the vocabulary. The
    generative models here use a dense fixed-width decoder (n_genes is
    baked into the architecture at construction time), so every sample
    fed through the same model instance must present identically-shaped,
    identically-ordered expression vectors. This is why, unlike
    load_multi_slice's join="outer" (safe there since serial sections of
    one tissue block typically share the same sequencing run/gene
    panel), an outer join with zero-filling would be scientifically
    misleading here: it would conflate "this gene wasn't measured in
    this sample" with "this gene measured as zero expression," and could
    easily happen across genuinely different gene panels (HEST-1k spans
    Visium ~20k-gene whole-transcriptome AND Xenium ~few-hundred-gene
    targeted panels, docs/dataset_notes.md — confirm sample_ids share a
    platform before pooling; INT1-INT24 are documented as all-Visium,
    same ccRCC cohort, a well-grounded first choice).

    organs/techs (2026-07-16, OrganTechEmbedding follow-up): optional,
    parallel to sample_ids — organs[i]/techs[i] is sample_ids[i]'s
    organ/technology, passed straight through to load_hest_sample (see
    that function's docstring on why this is caller-supplied rather than
    auto-parsed). None (default) for either leaves every sample at
    "unknown" for that field — harmless, since build_organ_tech_vocab
    just builds whatever vocabulary the values actually given produce."""
    if organs is not None and len(organs) != len(sample_ids):
        raise ValueError(f"organs has {len(organs)} entries but sample_ids has {len(sample_ids)}")
    if techs is not None and len(techs) != len(sample_ids):
        raise ValueError(f"techs has {len(techs)} entries but sample_ids has {len(sample_ids)}")
    adatas = [
        basic_qc_and_normalize(
            load_hest_sample(hest_data_dir, sid,
                              organ=organs[i] if organs is not None else None,
                              tech=techs[i] if techs is not None else None),
            min_genes=min_genes,
            # A held-out sample must not select the evaluation vocabulary by
            # expression prevalence. Keep every measured gene, then align to
            # the training-derived reference panel below. Missing reference
            # names still raise and therefore distinguish an unmeasured gene
            # from a measured all-zero/rare gene.
            min_cells=0 if reference_genes is not None else min_cells,
            transform=expression_transform, target_sum=expression_target_sum)
        for i, sid in enumerate(sample_ids)
    ]
    if reference_genes is None:
        shared_genes = sorted(set.intersection(*(set(a.var_names) for a in adatas)))
        if not shared_genes:
            raise ValueError(
                f"No genes shared across all samples {sample_ids!r} after per-sample QC — "
                "likely mixing different gene panels/platforms (see this function's "
                "docstring); check the `technology` column in HEST-1k's metadata CSV."
            )
        panel_source = "fit-sample intersection"
    else:
        # Strict held-out evaluation: the target vocabulary is established
        # from train/validation samples before any test slide is loaded.  Test
        # slides may be reordered to that vocabulary, but they are never
        # allowed to shrink or otherwise choose it.
        shared_genes = [str(g) for g in reference_genes]
        if len(shared_genes) != len(set(shared_genes)):
            raise ValueError("reference_genes contains duplicates")
        for sid, a in zip(sample_ids, adatas):
            missing = [g for g in shared_genes if g not in a.var_names]
            if missing:
                raise ValueError(
                    f"held-out sample {sid!r} is missing {len(missing)} genes from the "
                    f"fit-derived reference panel (examples: {missing[:5]}). Refusing "
                    "to intersect with test data because that would make the test set "
                    "participate in model-vocabulary selection."
                )
        panel_source = "predeclared fit-derived reference"
    for sid, a in zip(sample_ids, adatas):
        n_before = a.n_vars
        print(f"load_multi_sample: {sid} keeps {len(shared_genes)}/{n_before} genes "
              f"({len(shared_genes) / n_before:.0%}) using {panel_source}")
    return [a[:, shared_genes].copy() for a in adatas]
