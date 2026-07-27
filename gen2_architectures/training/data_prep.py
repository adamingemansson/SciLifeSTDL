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
import time
from pathlib import Path

import numpy as np

from gen2_architectures.data import loaders
from gen2_architectures.data.context_features import ContextOnlyFeatureProvider, PrecomputedSpotFeatureProvider
from gen2_architectures.data.hest1k_catalog import resolve_sample_selection
from gen2_architectures.data.masked_item import make_context_query_split
from gen2_architectures.data.mask_bank import cap_context_mask


def apply_sample_selection(cfg) -> None:
    """If cfg.data.sample_selection is present, resolve it against the real
    HEST-1k metadata + local inventory and OVERWRITE cfg.data's
    train_sample_ids/validation_sample_ids/test_sample_ids/organ_by_sample/
    tech_by_sample in place, plus cfg.model.params.organ_vocab/tech_vocab
    if those aren't already explicitly set. A no-op (does nothing) when
    sample_selection isn't present, so existing literal-sample-ID configs
    (e.g. the 8-sample Lung placeholder configs) keep working completely
    unchanged -- this is purely additive.

    Mutates cfg in place (OmegaConf configs are mutable containers) rather
    than returning a new one, so every call site just calls this once,
    early, and continues reading cfg.data.* normally afterward -- no
    caller needs to know whether the values came from a literal list or a
    resolved selection.

    species/min_nb_genes/check_gene_panel_compatibility/min_gene_coverage/
    min_sample_coverage/min_panel_size (all optional keys under
    sample_selection) pass straight through to resolve_sample_selection,
    which defaults every one of them safely on its own -- only set these
    in a config to deliberately override (e.g. species: "all" for a
    cross-species comparison run). See resolve_sample_selection's own
    docstring for what each really does and the real HEST-1k bugs they
    guard against.
    """
    selection_cfg = cfg.data.get("sample_selection")
    if selection_cfg is None:
        return
    organs = selection_cfg.get("organs", "all")
    if organs != "all":
        organs = list(organs)
    species = selection_cfg.get("species", "Homo sapiens")
    min_nb_genes = selection_cfg.get("min_nb_genes", 5000)
    result = resolve_sample_selection(
        hest_data_dir=cfg.data.hest_data_dir,
        metadata_csv=selection_cfg.get("metadata_csv", "hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv"),
        organs=organs,
        species=None if species == "all" else species,
        min_nb_genes=None if min_nb_genes in (None, 0) else int(min_nb_genes),
        min_samples_per_organ=int(selection_cfg.get("min_samples_per_organ", 3)),
        max_samples_per_organ=selection_cfg.get("max_samples_per_organ"),
        n_validation_per_organ=int(selection_cfg.get("n_validation_per_organ", 1)),
        n_test_per_organ=int(selection_cfg.get("n_test_per_organ", 1)),
        split_seed=int(selection_cfg.get("split_seed", 0)),
        check_gene_panel_compatibility=bool(selection_cfg.get("check_gene_panel_compatibility", True)),
        min_gene_coverage=float(selection_cfg.get("min_gene_coverage", 0.9)),
        min_sample_coverage=float(selection_cfg.get("min_sample_coverage", 0.9)),
        min_panel_size=int(selection_cfg.get("min_panel_size", 5000)),
    )
    cfg.data.train_sample_ids = result["train_sample_ids"]
    cfg.data.validation_sample_ids = result["validation_sample_ids"]
    cfg.data.test_sample_ids = result["test_sample_ids"]
    cfg.data.organ_by_sample = result["organ_by_sample"]
    cfg.data.tech_by_sample = result["tech_by_sample"]
    n_train, n_val, n_test = len(result["train_sample_ids"]), len(result["validation_sample_ids"]), len(result["test_sample_ids"])
    print(
        f"apply_sample_selection: resolved {n_train} train / {n_val} validation / {n_test} test "
        f"samples across organs {result['organ_vocab']}"
    )
    # organ_vocab/tech_vocab are only real constructor params for
    # Architectures 1/2 (LocalNeighborhoodTransformer) and Architecture 3
    # Stage B -- NOT Architecture 4 (fixed organ_type/tech_type strings,
    # a single value not a vocabulary) or Stage A (no organ conditioning
    # at all). Injecting them unconditionally would pass an unexpected
    # kwarg into those constructors and crash. Detect which kind of config
    # this is the same way each training entrypoint already does: an
    # "architecture" key means 1/2/4 (train_local_neighborhood.py), a
    # "stage_a_checkpoint_dir" key means Stage B, neither means Stage A.
    accepts_vocab = (
        str(cfg.get("model", {}).get("architecture", "")) in ("1", "2")
        or "stage_a_checkpoint_dir" in cfg.get("model", {})
    )
    if accepts_vocab and "params" in cfg.model:
        if "organ_vocab" not in cfg.model.params or cfg.model.params.organ_vocab is None:
            cfg.model.params.organ_vocab = result["organ_vocab"]
        if "tech_vocab" not in cfg.model.params or cfg.model.params.tech_vocab is None:
            cfg.model.params.tech_vocab = result["tech_vocab"]


def apply_smoke_override(cfg, smoke_steps: int | None) -> None:
    """If smoke_steps is set (from a training script's --smoke_steps CLI
    flag), override cfg.training.total_steps down to it in place, and
    scale checkpoint_every_n_steps/eval_every_n_steps/log_every_n_steps
    down proportionally so a short smoke run actually exercises a
    checkpoint save and a validation eval before it ends, instead of
    running out before either ever fires. checkpoint_keep_last is left
    alone (smoke runs use the same tiny footprint as a real run). A no-op
    when smoke_steps is None, so real runs are completely unaffected --
    this exists specifically so a smoke run needs no config file edits
    (and therefore nothing to remember to revert before the real run)."""
    if smoke_steps is None:
        return
    smoke_steps = int(smoke_steps)
    cfg.training.total_steps = smoke_steps
    cfg.training.checkpoint_every_n_steps = max(1, smoke_steps // 4)
    cfg.training.eval_every_n_steps = max(1, smoke_steps // 2)
    cfg.training.log_every_n_steps = max(1, min(int(cfg.training.get("log_every_n_steps", 50)), smoke_steps // 10))
    print(f"--smoke_steps {smoke_steps}: total_steps/checkpoint/eval/log intervals overridden for this run only")


def resolve_wall_clock_deadline(cfg) -> float | None:
    """training.max_wall_clock_hours (2026-07-25), if set, gives training
    scripts a REAL alternative to total_steps for "run for about N hours"
    -- deliberately preferred over back-deriving total_steps from a short
    smoke run's measured steps/sec, since a smoke run's early steps
    include one-time GigaPath/scFoundation cache-population overhead
    that is not representative of steady-state throughput, and per-step
    cost can drift over a genuinely long run (disk I/O contention,
    thermal throttling, etc. -- especially relevant with multiple
    architectures training concurrently on the same server).
    total_steps remains a hard safety cap regardless -- whichever limit
    (step count or wall clock) is hit first stops training; the final
    checkpoint save and held-out test evaluation still run normally
    either way, since callers just `break` out of the training loop on
    deadline, not return/exit early.

    Returns an absolute time.monotonic() deadline, or None if
    max_wall_clock_hours isn't set (matches every existing config,
    which relies on total_steps alone -- this is purely additive)."""
    hours = cfg.training.get("max_wall_clock_hours")
    if hours is None:
        return None
    hours = float(hours)
    if hours <= 0:
        raise ValueError(f"training.max_wall_clock_hours must be positive, got {hours}")
    return time.monotonic() + hours * 3600.0


def make_progress_fn(wall_clock_deadline: float | None, total_steps: int):
    """Returns a step -> progress-in-[0, 1] callable for StagedGeneLoss's
    stage curriculum (models/components.py).

    2026-07-25 bugfix: progress used to be `step / total_steps` unconditionally
    in every training script. Now that max_wall_clock_hours is the real
    stopping mechanism and total_steps is a rarely-hit safety cap (set far
    larger than any real run should reach -- see the configs), step /
    total_steps barely moves over an entire real run, so the loss curriculum
    would never leave stage 1 (pure MSE, no Pearson term). When a wall-clock
    deadline is set, progress is instead the fraction of the wall-clock
    budget elapsed -- what's actually governing how far into the run we are.
    Falls back to step / total_steps when no wall-clock deadline is set."""
    training_start = time.monotonic()
    total_wall_clock_seconds = (
        wall_clock_deadline - training_start if wall_clock_deadline is not None else None
    )

    def progress_fn(step: int) -> float:
        if total_wall_clock_seconds is not None and total_wall_clock_seconds > 0:
            return min(1.0, (time.monotonic() - training_start) / total_wall_clock_seconds)
        return step / max(1, total_steps)

    return progress_fn


def is_finite_update(loss, grad_norm) -> bool:
    """True iff both the loss and the (post-clipping) gradient norm are
    finite -- i.e. it's safe to call optimizer.step().

    2026-07-27 bugfix: torch.nn.utils.clip_grad_norm_ compares the
    gradient norm against a threshold to decide whether to rescale; a NaN
    norm fails every such comparison, so a NaN gradient sails straight
    through UNCLIPPED and optimizer.step() then permanently corrupts
    every parameter with NaN -- observed directly on a real 24h run
    (every diagnostic, including a weight's own norm, went NaN and never
    recovered, since NaN has no way to self-correct). Callers should skip
    the optimizer.step() entirely (not just clip harder) whenever this
    returns False."""
    return bool(loss.isfinite()) and bool(grad_norm.isfinite())


def derive_coord_scale(adatas: list) -> float:
    """Auto-derive coord_scale for RandomFourierFeatures-based
    CoordEmbedding from these samples' REAL coordinate spread — ports the
    exact formula already established (and real-bug-fixed, 2026-07-17) in
    src/training/train.py::inject_coord_scale's own call site: mean, across
    samples, of each sample's per-point (x, y) std. Real HEST-1k
    coordinates are pixel-scale and different samples/organs can have
    different physical pixel resolutions (see
    models/conditioning.py::RandomFourierFeatures' own docstring for the
    aliasing failure mode this exists to avoid) — this must be computed
    from the real training data actually being used, not guessed once and
    reused across every organ."""
    stds = [float(np.asarray(adata.obsm["spatial"][:, :2]).std()) for adata in adatas]
    return float(np.mean(stds))


def apply_coord_scale(cfg, train_adatas: list) -> None:
    """Auto-inject cfg.model.params.coord_scale from the real training
    data (derive_coord_scale) unless a config already sets it explicitly
    -- same override discipline, and the same architecture-eligibility
    condition (only architectures whose constructor actually accepts
    coord_scale), as apply_sample_selection's organ_vocab/tech_vocab
    injection: Architectures 1/2 and Stage B build a CoordEmbedding and
    accept it; Architecture 4 (STPathContextEncoder conditions on
    organ/tech directly) and Stage A (no spatial component at all) do
    not, and injecting into either would crash with an unexpected
    keyword argument -- same real bug class apply_sample_selection's own
    accepts_vocab guard was written to avoid, see that function's
    docstring."""
    accepts_coord_scale = (
        str(cfg.get("model", {}).get("architecture", "")) in ("1", "2")
        or "stage_a_checkpoint_dir" in cfg.get("model", {})
    )
    if not accepts_coord_scale or "params" not in cfg.model:
        return
    if "coord_scale" in cfg.model.params and cfg.model.params.coord_scale is not None:
        return
    coord_scale = derive_coord_scale(train_adatas)
    cfg.model.params.coord_scale = coord_scale
    print(f"coord_scale auto-derived from real training data: {coord_scale:.2f}")


def _atomic_savez(cache_path: Path, **arrays) -> None:
    """np.savez, but crash-safe under concurrent writers (e.g. multiple
    GPU processes racing to populate the same cache file the first time).

    REGRESSION fixed 2026-07-25 (first real training-server run to
    actually exercise a cold/uncached GigaPath computation through this
    copy): this function's docstring used to claim "ported unchanged
    from src/training/train.py", but it had actually reverted to that
    file's OWN pre-2026-07-17 buggy version --
    cache_path.with_suffix(cache_path.suffix + f".tmp{pid}") produces a
    tmp filename like "INT1.npz.tmp12345", which does NOT end in
    ".npz" -- np.savez SILENTLY APPENDS ".npz" to any string/Path target
    that doesn't already end in ".npz" (a well-known numpy gotcha), so it
    actually wrote "INT1.npz.tmp12345.npz", and the os.replace() below
    then raised FileNotFoundError looking for the path numpy never
    created. src/training/train.py's own _atomic_savez already found and
    fixed this exact bug on 2026-07-17 -- restored that exact fix here
    (keep ".npz" as the tmp path's real suffix) rather than re-deriving
    it, per this project's own discipline of reusing already-debugged
    solutions instead of risking a fresh, differently-wrong one."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, cache_path)  # atomic on POSIX -- no reader ever sees a partial file


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


def load_held_out_samples_with_images(cfg, sample_ids: list[str], reference_genes: list[str]):
    """Like load_multi_sample_with_images(..., reference_genes=...), but
    loads each held-out sample INDIVIDUALLY and skips (with a clear
    warning) any one that doesn't fully cover the train-derived
    reference panel, instead of raising and losing the whole
    validation/test batch to one incompatible sample.

    Real bug found 2026-07-25 on the actual training server: a held-out
    sample can pass resolve_compatible_sample_ids' coarser ~90%-coverage
    compatibility check (hest1k_catalog.py) yet still be missing a
    handful of genes from the SPECIFIC exact panel load_multi_sample
    ultimately derives from the train samples ALONE -- those are
    genuinely different bars (a statistical "close enough" cohort-level
    check vs. an exact zero-tolerance per-sample check), and
    load_multi_sample's reference_genes path correctly raises rather
    than silently zero-filling or letting held-out data influence the
    vocabulary (see that function's own docstring on why). Previously
    that raise propagated all the way up and crashed the whole training
    run over ONE held-out sample; this function contains it instead.

    Returns (kept_ids, adatas, images_list) -- kept_ids is a SUBSET of
    sample_ids in the same relative order; callers must use kept_ids,
    not sample_ids, for everything downstream."""
    kept_ids, adatas, images_list = [], [], []
    for sample_id in sample_ids:
        try:
            [adata], [images] = load_multi_sample_with_images(cfg, [sample_id], reference_genes=reference_genes)
        except ValueError as exc:
            if "fit-derived reference panel" not in str(exc):
                raise
            print(f"load_held_out_samples_with_images: skipping {sample_id!r} ({exc})")
            continue
        kept_ids.append(sample_id)
        adatas.append(adata)
        images_list.append(images)
    return kept_ids, adatas, images_list


def _probe_context_feature_dim(
    cfg, adata, provider: ContextOnlyFeatureProvider | PrecomputedSpotFeatureProvider,
) -> int:
    """Deterministic probe context mask (fixed seed, never used for real
    training/eval) so a precomputed feature provider's real output width
    can be read before model construction -- same reasoning as the
    original prepare_scfoundation_inputs (src/training/train.py). Works
    for either provider type -- both share the same
    provider(context_mask) -> [n_context, feature_dim] call contract."""
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


def build_scfoundation_provider(cfg, adata, sample_id: str) -> PrecomputedSpotFeatureProvider:
    """One PrecomputedSpotFeatureProvider computing scFoundation cell
    embeddings for every spot in the sample ONCE (never per masking
    draw), disk-cached one file per sample -- exactly like GigaPath's own
    cache (_gigapath_cache_path). scFoundation itself has no query-
    leakage risk (computed per-spot from that spot's own expression
    alone, not a spatial-neighbor graph -- see
    data/context_features.py::model_uses_scfoundation's own docstring),
    so it does NOT need the context-only-per-mask recomputation
    ContextOnlyNovaeProvider exists for.

    REGRESSION fixed 2026-07-25 (found by directly checking storage
    behavior after a question about GigaPath's disk footprint): this
    function used to route scFoundation through ContextOnlyNovaeProvider,
    whose cache is keyed by a digest of the CURRENT masking draw's
    context set. Training masking uses seed=step, so a fresh,
    essentially never-repeated context mask gets drawn on almost every
    step -- meaning a NEW small .npz cache file got written on nearly
    every training step, for the life of the run, with near-zero real
    cache-hit rate. For Architecture 4 (max_context_points=3000,
    50,000-step budget) this could have accumulated into hundreds of GB
    against a real 50GB disk budget. See
    PrecomputedSpotFeatureProvider's own docstring for the full story.

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

    return PrecomputedSpotFeatureProvider(
        adata, cache_dir=cache_root(cfg) / "scfoundation_cache",
        sample_id=str(sample_id), feature_fn=_feature_fn,
    )
