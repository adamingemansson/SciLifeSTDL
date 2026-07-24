"""Data loading, GigaPath feature caching, and scFoundation feature-provider
wiring shared by every gen2 architecture's training script.

The GigaPath caching logic (_gigapath_cache_path/get_gigapath_features/
_atomic_savez) is a FAITHFUL port of src/training/train.py's own
already-fixed version — this project hit a real, costly bug this session
(2026-07-24) where a preprocessing fix was silently masked because the
on-disk cache fingerprint depended only on raw patch bytes, never the
preprocessing CODE. That fix (folding _GIGAPATH_PREPROCESS_VERSION into the
fingerprint) is preserved here exactly, not re-derived, specifically to
avoid reintroducing the same class of bug in a "fresh" reimplementation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from gen2_architectures.data import loaders
from gen2_architectures.data.context_features import ContextOnlyFeatureProvider
from gen2_architectures.data.masked_item import make_context_query_split
from gen2_architectures.data.mask_bank import cap_context_mask


def _atomic_savez(cache_path: Path, **arrays) -> None:
    """np.savez, but crash-safe under concurrent writers (e.g. multiple
    GPU processes racing to populate the same cache file the first time).
    Ported unchanged from src/training/train.py."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp{os.getpid()}")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, cache_path)


def cache_root(cfg) -> Path:
    """Where cache subfolders (gigapath_cache/, scfoundation_context_cache/)
    get created. cfg.data.hest_cache_dir overrides cfg.data.hest_data_dir
    when the raw data directory is read-only shared storage."""
    cache_dir = cfg.data.get("hest_cache_dir") if hasattr(cfg.data, "get") else None
    return Path(cache_dir) if cache_dir else Path(cfg.data.hest_data_dir)


def _gigapath_cache_path(cfg, sample_id: str) -> Path:
    return cache_root(cfg) / "gigapath_cache" / f"{sample_id}.npz"


def get_gigapath_features(cfg, patches: np.ndarray, barcodes: np.ndarray, sample_id: str) -> np.ndarray:
    """Load cached GigaPath features for these patches if available (and
    genuinely still valid — see the fingerprint discussion below), else
    compute and cache them."""
    from gen2_architectures.models.conditioning import _GIGAPATH_PREPROCESS_VERSION

    cache_path = _gigapath_cache_path(cfg, sample_id)
    # Barcode equality alone cannot prove a cache belongs to the selected
    # H&E file (HEST Visium samples may reuse the same capture-array
    # barcode strings) -- hash the actual patch tensor, AND fold in the
    # preprocessing version, so a future preprocessing change (or a
    # barcode collision across samples) can never be silently served
    # stale. This is the exact fingerprint construction that fixed the
    # real cache-staleness bug found this session -- see this module's
    # own top-of-file docstring.
    patch_array = np.ascontiguousarray(patches)
    digest = hashlib.sha256()
    digest.update(str(patch_array.shape).encode("ascii"))
    digest.update(str(patch_array.dtype).encode("ascii"))
    digest.update(memoryview(patch_array).cast("B"))
    digest.update(_GIGAPATH_PREPROCESS_VERSION.encode("ascii"))
    patch_fingerprint = digest.hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_fingerprint = (
            str(cached["patch_fingerprint"].item()) if "patch_fingerprint" in cached.files else None
        )
        if np.array_equal(cached["barcodes"], barcodes) and cached_fingerprint == patch_fingerprint:
            print(
                f"get_gigapath_features: loaded cached features for {cached['features'].shape[0]} "
                f"spots from {cache_path} (delete this file to force a recompute)."
            )
            return cached["features"]
        print(
            f"get_gigapath_features: cache at {cache_path} does not match the current patch "
            "tensor, barcode set, or preprocessing version — recomputing."
        )
    from gen2_architectures.models.conditioning import precompute_gigapath_features, _default_device

    print(
        f"Precomputing GigaPath features for {patches.shape[0]} spots on {_default_device()} "
        f"(one-time cost, cached to {cache_path} so future runs skip this step)..."
    )
    features = precompute_gigapath_features(patches)
    _atomic_savez(
        cache_path, features=features, barcodes=barcodes, patch_fingerprint=np.asarray(patch_fingerprint),
    )
    return features


def load_multi_sample_with_images(cfg, sample_ids: list[str], reference_genes: list[str] | None = None):
    """Multi-sample loading with a shared gene panel (loaders.py::
    load_multi_sample's real contract, reused directly — see that
    function's own docstring for why a strict, non-zero-filling
    intersection matters for the missing_tissue task), THEN per-sample
    GigaPath image loading/caching, mirroring the original codebase's
    load_multi_sample_data two-stage structure. Per-sample image loading
    can drop spots with no matching H&E patch (a normal partial gap, not
    an error — see loaders.py::align_patches_to_adata's own docstring);
    the returned adata for each sample reflects that possibly-subsetted
    version, never the pre-image-loading one.

    Returns (adatas, images_list) — parallel lists, same order as
    sample_ids."""
    adatas = loaders.load_multi_sample(
        cfg.data.hest_data_dir, sample_ids,
        min_genes=cfg.data.get("min_genes", 200), min_cells=cfg.data.get("min_cells", 3),
        organs=[cfg.data.get("organ_by_sample", {}).get(str(sid)) for sid in sample_ids] or None,
        techs=[cfg.data.get("tech_by_sample", {}).get(str(sid)) for sid in sample_ids] or None,
        expression_transform=cfg.data.get("expression_transform", "normalize_log1p"),
        expression_target_sum=float(cfg.data.get("expression_target_sum", 1e4)),
        reference_genes=reference_genes,
    )
    updated_adatas, images_list = [], []
    for sample_id, adata in zip(sample_ids, adatas):
        patches, barcodes = loaders.load_hest_patches(cfg.data.hest_data_dir, sample_id)
        features = get_gigapath_features(cfg, patches, barcodes, sample_id=str(sample_id))
        adata, images = loaders.align_patches_to_adata(adata, features, barcodes)
        updated_adatas.append(adata)
        images_list.append(images)
    return updated_adatas, images_list


def _probe_context_feature_dim(cfg, adata, provider: ContextOnlyFeatureProvider) -> int:
    """Deterministic probe context mask (fixed seed, never used for real
    training/eval) so a precomputed feature provider's real output width
    can be read before model construction -- same reasoning as the
    original prepare_scfoundation_inputs (src/training/train.py)."""
    coords3d = loaders.get_coords_3d(adata)
    probe_context, probe_query = make_context_query_split(
        coords3d, adata.obs["slice_id"].to_numpy(), cfg.masking, seed=606_061,
    )
    probe_context = cap_context_mask(
        probe_context, cfg.masking.get("max_context_points"), 606_061,
        coords3d=coords3d, query_mask=probe_query,
        selection=str(cfg.masking.get("context_selection", "random")),
    )
    probe = provider(probe_context)
    return int(probe.shape[1])


def build_scfoundation_provider(cfg, adata, sample_id: str) -> ContextOnlyFeatureProvider:
    """One ContextOnlyFeatureProvider computing scFoundation cell
    embeddings on the observed context subgraph only (never the full
    slide -- see that class's own docstring on why leaking hidden query
    expression through a full-slide computation would be invalid for the
    missing_tissue task; scFoundation itself has no such leakage risk
    since it's computed per-spot from that spot's own expression alone,
    but this project's OWN established discipline is to always route
    precomputed context features through this same context-only engine,
    both for cache-key consistency and so every gene-feature pathway in
    this codebase follows one uniform, auditable rule).

    already_normalized_log1p=True: this project's own loaders already
    apply the exact library-size-to-1e4 + log1p transform scFoundation's
    own real preprocessing expects (see
    gen2_architectures/models/conditioning.py::precompute_scfoundation_
    features's own docstring) -- applying it again here would silently
    double-normalize every value. Used identically by both Architecture 2
    (replacement channel) and Architecture 4 (additive residual channel);
    only which context dict key the caller stores the provider under
    differs.
    """
    repo_path = cfg.data.get("scfoundation_repo_path")
    model_path = cfg.data.get("scfoundation_model_path")
    if not repo_path or not model_path:
        raise ValueError(
            "scFoundation features were requested, but data.scfoundation_repo_path "
            "and/or data.scfoundation_model_path are not set."
        )

    def _feature_fn(spot_adata):
        from gen2_architectures.models.conditioning import precompute_scfoundation_features

        expr = spot_adata.X if isinstance(spot_adata.X, np.ndarray) else spot_adata.X.toarray()
        return precompute_scfoundation_features(
            expr, list(spot_adata.var_names), str(repo_path), str(model_path),
            already_normalized_log1p=True,
        )

    return ContextOnlyFeatureProvider(
        adata, cache_dir=cache_root(cfg) / "scfoundation_context_cache",
        sample_id=str(sample_id), feature_fn=_feature_fn,
    )
