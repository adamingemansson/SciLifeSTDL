"""Immutable Gen3 dataset manifest -- Step 1 of the real Gen3 data
builder/trainer (Adam's explicit instruction and 9-step implementation
order, gen3_multiscale/CONTRACT.md section 30: "Build an immutable
dataset manifest with sample IDs, composite (sample_id, spot_id)
identities, gene order/hash, coordinates, WSI/cache provenance and
patient-level splits.").

Builds and persists ONE authoritative, atomically-written artifact
describing exactly what data a Gen3 training run will use:
  - patient-disjoint train/validation/test sample splits (reusing
    hest1k_catalog.resolve_sample_selection's already-audited logic,
    never re-derived independently here)
  - the ordered gene panel, derived from TRAINING samples only via
    loaders.load_multi_sample's real QC/normalization pipeline (never
    letting held-out samples shrink or choose the vocabulary -- the
    exact discipline the 5th Codex re-audit's implementation order
    item 5 specified)
  - every kept sample's real per-spot barcodes and coordinates, and the
    composite (sample_id, spot_id) identities every later leakage check
    (Step 3) is built on -- plain Visium barcodes are NOT globally
    unique and are reused across different samples/slides, a gap
    repeatedly confirmed across this project's audit rounds (most
    recently CONTRACT.md section 26 finding #6)
  - real SHA256 content provenance for the metadata CSV and every kept
    sample's h5ad/patch-h5 files (and its WSI tile cache, if present) --
    NOT the cheap path/size/mtime "identity" slide_context.py's own
    runtime cache key uses, which the 10th Codex re-audit of commit
    9592d9e (finding #2, confirmed) correctly flagged as insufficient
    for the MANIFEST's own immutability claim (a runtime cache key only
    needs to detect *most* changes cheaply; an immutability record needs
    to actually identify the bytes it describes). Hashing full file
    content at manifest-build time is a real, non-trivial cost for large
    per-sample files, so it is memoized in an on-disk "digest database"
    (see _cached_file_content_hash/load_digest_cache/save_digest_cache
    below) keyed by each file's own path+size+mtime -- exactly the
    mitigation the audit itself proposed ("hashing can be cached via a
    separately-verified digest database for the one-time cost"): a cache
    hit still re-verifies size/mtime against the real file before ever
    trusting a memoized digest.

Every later step in the real data builder (example construction, mask-
realization fingerprinting, Novae graphs, WSI wiring, the trainer and
evaluator) reads THIS manifest rather than re-deriving splits/panels/
identities independently -- one authoritative source of truth, not
several independently-computed ones that could silently disagree.

Deliberately NOT built here: loading full expression matrices for
validation/test samples (only train samples are loaded, to derive the
gene panel; validation/test samples are only read cheaply for
barcodes/coordinates, via anndata's backed='r' mode -- mirroring
hest1k_catalog._real_var_names's own established cheap-read pattern),
mask realization (Step 2/3), or WSI feature loading (Step 5). This
module's job is provenance and identity bookkeeping, not data loading
for training itself.
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np

from gen3_multiscale.data import loaders
from gen3_multiscale.data.hest1k_catalog import resolve_sample_selection

_MANIFEST_VERSION = 1


def composite_spot_id(sample_id: str, barcode: str) -> str:
    """The TRUE globally-unique spot identity used throughout this
    module and every later leakage check -- plain Visium barcodes are
    NOT globally unique and are reused across different samples/slides
    (repeatedly confirmed across this project's audit rounds, most
    recently CONTRACT.md section 26 finding #6: "Plain Visium barcodes
    are not globally unique and are reused between slides"). Uses a
    delimiter ("::") that cannot appear in a real HEST-1k sample_id (a
    filesystem-safe identifier) or a real 10x/Visium barcode (e.g.
    "AAACAAGTATCTCCCA-1")."""
    sample_id = str(sample_id)
    barcode = str(barcode)
    if "::" in sample_id or "::" in barcode:
        raise ValueError(
            f"sample_id {sample_id!r} or barcode {barcode!r} contains the '::' composite-id "
            "delimiter -- this would make composite_spot_id ambiguous"
        )
    return f"{sample_id}::{barcode}"


def _read_sample_barcodes_and_coords(
    hest_data_dir: str | Path, sample_id: str, min_genes_per_spot: int,
) -> tuple[list[str], np.ndarray]:
    """Real per-spot barcodes and spatial coordinates for one sample,
    AFTER the same per-spot QC filter (`sc.pp.filter_cells(min_genes=...)`)
    `loaders.basic_qc_and_normalize` applies everywhere else.

    Real, confirmed gap fixed here (found while designing Step 2, before
    any Step-2 code shipped): an earlier version of this function used a
    cheap `backed='r'` read with NO QC filtering at all, so a sample's
    manifest-recorded spot set (barcodes/coords/composite_spot_ids)
    could silently disagree with the spot set `loaders.load_multi_sample`
    (used for train samples' gene-panel derivation, and reused again by
    the real example builder for every sample) actually keeps after
    `min_genes` filtering -- a spot the manifest declares to exist could
    turn out to not actually be usable, or vice versa. Applying the
    IDENTICAL filter here, for every sample (train AND held-out), keeps
    the manifest's declared spot set authoritative and consistent with
    what every later consumer actually sees. This costs a real (not
    backed) per-sample load -- a one-time manifest-build cost, not a
    per-training-step one."""
    import anndata as ad
    import scanpy as sc
    path = loaders._resolve_hest_sample_file(hest_data_dir, sample_id, ".h5ad")
    adata = ad.read_h5ad(path)
    if min_genes_per_spot > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes_per_spot)
    barcodes = [str(x) for x in adata.obs_names]
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if coords.shape[0] != len(barcodes):
        raise ValueError(
            f"{sample_id}: obsm['spatial'] has {coords.shape[0]} rows but obs_names has "
            f"{len(barcodes)} entries -- misaligned real data"
        )
    if not barcodes:
        raise ValueError(
            f"{sample_id}: no spots survive the min_genes={min_genes_per_spot} per-spot QC filter"
        )
    return barcodes, coords


def gene_panel_hash(gene_names: list[str]) -> str:
    """Ordered-panel hash -- same discipline as
    gene_basis.fit_gene_residual_basis's gene_names_hash and
    checkpoint.verify_gene_names: the exact ORDER is hashed, not just
    set membership, since every architecture's dense gene-indexed
    tensors depend on a fixed, agreed-upon ordering."""
    return sha256("\n".join(str(g) for g in gene_names).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file's real content through SHA256 without loading it
    entirely into memory -- safe for large h5ad/patch/WSI-cache files."""
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_file_content_hash(path: Path, digest_cache: dict) -> str:
    """SHA256 content hash of `path`, memoized in `digest_cache` (an
    in-memory dict backed by load_digest_cache/save_digest_cache below)
    keyed by the file's own path+size+mtime identity. A cache hit still
    re-verifies size/mtime against the real file on disk before ever
    trusting a memoized digest -- never trusts a stale key alone."""
    stat = path.stat()
    key = str(path.resolve())
    cached = digest_cache.get(key)
    if (
        cached is not None
        and cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
    ):
        return cached["sha256"]
    digest = _sha256_file(path)
    digest_cache[key] = {
        "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest,
    }
    return digest


def load_digest_cache(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save_digest_cache(digest_cache: dict, path: str | Path) -> Path:
    """Atomic write, mirroring save_dataset_manifest -- multiple
    concurrent manifest builds may share and update the same digest
    database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(digest_cache, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def _wsi_cache_provenance(hest_cache_dir: str | Path, sample_id: str, digest_cache: dict) -> dict | None:
    """Real content provenance for one sample's dense GigaPath WSI tile
    cache: path/size/mtime plus a real SHA256 content hash (memoized via
    digest_cache -- see module docstring). Returns None (not an error)
    when the cache file doesn't exist yet -- WSI conditioning may not be
    enabled for every organ/run, and Step 8's preflight gates (not this
    module) are where "required but missing" becomes fail-closed."""
    path = Path(hest_cache_dir) / "gigapath_slide_cache" / f"{sample_id}.npz"
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _cached_file_content_hash(path, digest_cache),
    }


def _sample_content_provenance(hest_data_dir: str | Path, sample_id: str, digest_cache: dict) -> dict:
    """Real SHA256 content provenance for one sample's h5ad expression
    file and its patch h5 file. Both are required to exist here:
    resolve_sample_selection only ever returns sample_ids that passed
    hest1k_catalog.usable_local_ids's BOTH-expression-AND-patches
    filter, so a kept sample missing either real file on disk is a real
    invariant violation, not an expected scope boundary -- it fails
    loudly (via _resolve_hest_sample_file's own FileNotFoundError)
    rather than silently recording None. (The documented ~4.5% HEST-1k
    gap is about individual SPOTS within an existing patch file lacking
    a matching tile -- loaders.align_patches_to_adata's concern, not
    this function's.)"""
    h5ad_path = loaders._resolve_hest_sample_file(hest_data_dir, sample_id, ".h5ad")
    patch_path = loaders._resolve_hest_sample_file(
        hest_data_dir, sample_id, ".h5", required_path_part="patches",
    )
    return {
        "h5ad": {
            "path": str(h5ad_path.resolve()),
            "sha256": _cached_file_content_hash(h5ad_path, digest_cache),
        },
        "patch_h5": {
            "path": str(patch_path.resolve()),
            "sha256": _cached_file_content_hash(patch_path, digest_cache),
        },
    }


def build_dataset_manifest(
    hest_data_dir: str | Path,
    metadata_csv: str,
    *,
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
    split_by_patient: bool = True,
    gene_min_genes_per_spot: int = 200,
    gene_min_cells: int = 3,
    expression_transform: str = "normalize_log1p",
    expression_target_sum: float = 1e4,
    hest_cache_dir: str | Path | None = None,
    digest_cache_path: str | Path | None = None,
) -> dict:
    """Build the complete, in-memory dataset manifest. A pure function of
    its inputs and the real files on disk EXCEPT for one deliberate side
    effect: it reads and updates the on-disk content-digest cache (see
    module docstring) at `digest_cache_path` (default:
    `hest_cache_dir/content_digest_cache.json`) so repeated manifest
    builds don't re-hash unchanged large files. No other persistence
    happens here (see save_dataset_manifest/ensure_dataset_manifest
    below for the manifest's own atomic-write/fail-closed-reuse pair)."""
    hest_data_dir = Path(hest_data_dir)
    hest_cache_dir = Path(hest_cache_dir) if hest_cache_dir is not None else hest_data_dir
    digest_cache_path = Path(digest_cache_path) if digest_cache_path is not None else hest_cache_dir / "content_digest_cache.json"
    digest_cache = load_digest_cache(digest_cache_path)

    split = resolve_sample_selection(
        hest_data_dir, metadata_csv, organs=organs, species=species, min_nb_genes=min_nb_genes,
        min_samples_per_organ=min_samples_per_organ, max_samples_per_organ=max_samples_per_organ,
        n_validation_per_organ=n_validation_per_organ, n_test_per_organ=n_test_per_organ,
        split_seed=split_seed, check_gene_panel_compatibility=check_gene_panel_compatibility,
        min_gene_coverage=min_gene_coverage, min_sample_coverage=min_sample_coverage,
        min_panel_size=min_panel_size, split_by_patient=split_by_patient,
    )
    train_ids = split["train_sample_ids"]
    validation_ids = split["validation_sample_ids"]
    test_ids = split["test_sample_ids"]
    all_ids = sorted(set(train_ids) | set(validation_ids) | set(test_ids))

    # The gene panel is derived from TRAINING samples ONLY, via the real
    # QC/normalization pipeline every sample actually goes through
    # (loaders.load_multi_sample) -- "Derive the ordered gene panel from
    # training samples only... never let held-out samples shrink or
    # choose the vocabulary" (5th Codex re-audit implementation order,
    # item 5). The loaded AnnData objects themselves are discarded
    # immediately after extracting the panel; this function only ever
    # needs to know WHAT the panel is, not hold every training sample's
    # full expression matrix in memory afterward.
    train_organs = [split["organ_by_sample"][sid] for sid in train_ids]
    train_techs = [split["tech_by_sample"][sid] for sid in train_ids]
    train_adatas = loaders.load_multi_sample(
        hest_data_dir, train_ids, min_genes=gene_min_genes_per_spot, min_cells=gene_min_cells,
        organs=train_organs, techs=train_techs, expression_transform=expression_transform,
        expression_target_sum=expression_target_sum,
    )
    gene_panel = list(train_adatas[0].var_names) if train_adatas else []
    del train_adatas  # discard the loaded expression matrices -- only the panel is kept

    samples: dict[str, dict] = {}
    for sample_id in all_ids:
        barcodes, coords = _read_sample_barcodes_and_coords(hest_data_dir, sample_id, gene_min_genes_per_spot)
        samples[sample_id] = {
            "organ": split["organ_by_sample"][sample_id],
            "tech": split["tech_by_sample"][sample_id],
            "patient_id": split["patient_by_sample"][sample_id],
            "split": (
                "train" if sample_id in train_ids
                else "validation" if sample_id in validation_ids
                else "test"
            ),
            "n_spots": len(barcodes),
            "barcodes": barcodes,
            "composite_spot_ids": [composite_spot_id(sample_id, b) for b in barcodes],
            "coords": coords.tolist(),
            "content_provenance": _sample_content_provenance(hest_data_dir, sample_id, digest_cache),
            "wsi_cache": _wsi_cache_provenance(hest_cache_dir, sample_id, digest_cache),
        }

    metadata_csv_provenance = {
        "path": str(Path(metadata_csv).resolve()),
        "sha256": _cached_file_content_hash(Path(metadata_csv), digest_cache),
    }
    save_digest_cache(digest_cache, digest_cache_path)

    return {
        "version": _MANIFEST_VERSION,
        "hest_data_dir": str(hest_data_dir.resolve()),
        "metadata_csv": str(metadata_csv),
        "metadata_csv_provenance": metadata_csv_provenance,
        "build_args": {
            "organs": organs, "species": species, "min_nb_genes": min_nb_genes,
            "min_samples_per_organ": min_samples_per_organ, "max_samples_per_organ": max_samples_per_organ,
            "n_validation_per_organ": n_validation_per_organ, "n_test_per_organ": n_test_per_organ,
            "split_seed": split_seed, "check_gene_panel_compatibility": check_gene_panel_compatibility,
            "min_gene_coverage": min_gene_coverage, "min_sample_coverage": min_sample_coverage,
            "min_panel_size": min_panel_size, "split_by_patient": split_by_patient,
            "gene_min_genes_per_spot": gene_min_genes_per_spot, "gene_min_cells": gene_min_cells,
            "expression_transform": expression_transform, "expression_target_sum": expression_target_sum,
        },
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "test_sample_ids": test_ids,
        "organ_vocab": split["organ_vocab"],
        "tech_vocab": split["tech_vocab"],
        "gene_panel": gene_panel,
        "n_genes": len(gene_panel),
        "gene_panel_hash": gene_panel_hash(gene_panel),
        "samples": samples,
    }


def save_dataset_manifest(manifest: dict, path: str | Path) -> Path:
    """Atomic write, mirroring mask_bank.save_mask_bank/
    mask_schedule.save_stratified_mask_bank exactly (process-specific
    temp file then os.replace) -- multiple concurrent jobs may all
    request the same immutable manifest at once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_dataset_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def ensure_dataset_manifest(path: str | Path, **build_kwargs) -> tuple[dict, Path]:
    """Load-or-atomically-create, mirroring every other ensure_* function
    in this package: reuse an existing on-disk manifest if it is BYTE-
    FOR-BYTE identical to a fresh rebuild from the same inputs, otherwise
    build and persist a fresh one. Fails closed (raises) if an existing
    file differs from what these exact build_kwargs would produce --
    real HEST-1k data on disk can change (a new download, a corrected
    sample) between runs, and an immutable manifest that silently kept
    describing stale data would defeat the entire point of building one."""
    path = Path(path)
    expected = build_dataset_manifest(**build_kwargs)
    if path.exists():
        existing = load_dataset_manifest(path)
        if existing != expected:
            raise ValueError(
                f"dataset manifest {path} does not match a fresh rebuild from the current "
                "hest_data_dir/metadata_csv/build_args -- the underlying data or build "
                "arguments changed since this manifest was built; use a new path or remove "
                "the stale manifest"
            )
        return existing, path
    save_dataset_manifest(expected, path)
    return expected, path


def all_composite_spot_ids(manifest: dict, sample_ids: Iterable[str] | None = None) -> set[str]:
    """Every composite (sample_id, spot_id) identity across the given
    samples (default: every sample in the manifest) -- the base set
    Step 3's leakage/novelty checks are built on."""
    ids = sample_ids if sample_ids is not None else manifest["samples"].keys()
    out: set[str] = set()
    for sample_id in ids:
        out.update(manifest["samples"][sample_id]["composite_spot_ids"])
    return out
