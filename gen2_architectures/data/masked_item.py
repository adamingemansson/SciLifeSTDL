"""Build one {context, query, target_expression} training/eval item.

Ported and trimmed from src/training/train.py's _build_masked_item +
make_context_query_split + _prepare_images_for_split + _images_tensor
(2026-07-25), for the gen2_architectures self-contained folder. Two things
were deliberately dropped relative to the original, both because none of
the gen2 architectures use them:

- The WSI-dense-tile "slide_context" mechanism (src/data/slide_context.py's
  load_slide_context/visible_slide_context) — that's specific to the
  hierarchical_slide model family. The one thing from that file which IS a
  real, general leakage-prevention check (nonoverlapping_context_patch_mask)
  is kept, via gen2_architectures/data/patch_overlap.py.
- Novae/BANKSY-niche context features — not part of any of the 4 gen2
  architectures. The additive residual channel Architecture 4 needs (for
  its scFoundation residual into STPath's frozen backbone) is implemented
  here as a generically-named "extra" channel instead, since it has nothing
  to do with Novae.

Everything else (masking strategy dispatch, context/query split, image mode
handling, context_gex_mode ablation, organ/tech stashing, coordinate
augmentation) is preserved with the same semantics and the same real bugs
already fixed in the original (e.g. context_gex_mode="zero"/"shuffled"
covering every GEX-derived channel, not just raw expression).
"""
from __future__ import annotations

import numpy as np
import torch

from gen2_architectures.data import masking
from gen2_architectures.data.augmentation import augment_coords_xy
from gen2_architectures.data.mask_bank import cap_context_mask
from gen2_architectures.data.patch_overlap import nonoverlapping_context_patch_mask


def make_context_query_split(coords3d: np.ndarray, slice_ids: np.ndarray, masking_cfg, seed: int):
    """One random context/query mask draw. Mirrors the original's strategy
    dispatch exactly (src/training/train.py::make_context_query_split)."""
    strategy = masking_cfg.strategy
    if strategy == "hold_out_slice":
        rng = np.random.default_rng(seed)
        held_out = rng.choice(np.unique(slice_ids))
        context_mask, query_mask = masking.hold_out_slice(coords3d[:, 2], held_out, slice_ids)
    elif strategy == "random_dropout_patches":
        context_mask, query_mask = masking.random_dropout_patches(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    elif strategy == "sparse_spot_dropout":
        context_mask, query_mask = masking.sparse_spot_dropout(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    elif strategy == "mixed_dropout":
        context_mask, query_mask = masking.mixed_dropout(
            coords3d[:, :2], slice_ids, seed=seed, **masking_cfg.params
        )
    else:
        raise ValueError(f"Unknown masking strategy {strategy}")
    return context_mask, query_mask


def make_training_context_query_split(
    coords3d: np.ndarray,
    slice_ids: np.ndarray,
    masking_cfg,
    seed: int,
    excluded_training_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one training split, optionally on evaluation-safe observations
    only. Mirrors the original exactly (see its own docstring in
    src/training/train.py for the full reasoning on the rejection-sampling
    loop below)."""
    if excluded_training_mask is None:
        context_mask, query_mask = make_context_query_split(coords3d, slice_ids, masking_cfg, seed)
        candidate_seed = seed
    else:
        excluded = np.asarray(excluded_training_mask, dtype=bool)
        if excluded.shape != (coords3d.shape[0],):
            raise ValueError(
                f"excluded_training_mask shape {excluded.shape} does not match "
                f"{coords3d.shape[0]} observations"
            )
        eligible = ~excluded
        if int(eligible.sum()) < 2:
            raise ValueError("evaluation-spot exclusion leaves fewer than two training spots")
        context_mask = query_mask = None
        candidate_seed = int(seed)
        for attempt in range(10_000):
            candidate_seed = int(seed) + attempt * 1_000_003
            candidate_context, candidate_query = make_context_query_split(
                coords3d, slice_ids, masking_cfg, candidate_seed
            )
            if np.any(candidate_query & excluded):
                continue
            candidate_context = np.asarray(candidate_context, dtype=bool) & eligible
            if candidate_context.any() and candidate_query.any():
                context_mask = candidate_context
                query_mask = np.asarray(candidate_query, dtype=bool)
                break
        if context_mask is None or query_mask is None:
            raise ValueError(
                "could not draw a training patch disjoint from immutable evaluation "
                "queries after 10000 deterministic attempts; reduce the reserved mask bank"
            )
    context_mask = cap_context_mask(
        context_mask,
        getattr(masking_cfg, "max_context_points", None),
        candidate_seed,
        coords3d=coords3d,
        query_mask=query_mask,
        selection=str(getattr(masking_cfg, "context_selection", "random")),
    )
    if not context_mask.any() or not query_mask.any():
        raise ValueError(f"training mask seed {seed} produced an empty context or query")
    return context_mask, query_mask


def _images_tensor(images: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Slice + tensor-ify per-spot image data for one masking draw. `images`
    is either raw uint8 patches [N, H, W, 3] or precomputed feature vectors
    [N, feat_dim] float32 — dispatches on ndim."""
    selected = images[mask]
    if selected.ndim == 4:
        return torch.tensor(selected, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return torch.tensor(selected, dtype=torch.float32)


def _prepare_images_for_split(
    images: np.ndarray | None,
    context_mask: np.ndarray,
    query_mask: np.ndarray,
    mode: str = "full",
    seed: int = 0,
    query_dropout_p: float = 0.0,
    all_dropout_p: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Build image tensors plus explicit availability masks for one split.
    `mode`: full (all images kept) / target_zero (query images removed --
    the real missing_tissue task) / all_zero (both removed) / shuffled
    (query images permuted, a diagnostic)."""
    if images is None:
        return None, None, None, None
    if mode not in {"full", "target_zero", "all_zero", "shuffled"}:
        raise ValueError(f"unknown image mode {mode!r}")

    context_images = _images_tensor(images, context_mask)
    query_images = _images_tensor(images, query_mask)
    context_available = torch.ones(context_images.shape[0], dtype=torch.bool)
    query_available = torch.ones(query_images.shape[0], dtype=torch.bool)
    rng = np.random.default_rng(seed + 17)

    if mode == "target_zero":
        query_available[:] = False
    elif mode == "all_zero":
        context_available[:] = False
        query_available[:] = False
    elif mode == "shuffled" and query_images.shape[0] > 1:
        perm = torch.as_tensor(rng.permutation(query_images.shape[0]), dtype=torch.long)
        query_images = query_images[perm]

    if all_dropout_p > 0 and rng.random() < all_dropout_p:
        context_available[:] = False
        query_available[:] = False
    elif query_dropout_p > 0:
        drop = torch.as_tensor(rng.random(query_images.shape[0]) < query_dropout_p)
        query_available &= ~drop

    def _zero_unavailable(x: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        if bool(available.all()):
            return x
        x = x.clone()
        x[~available] = 0
        return x

    return (
        _zero_unavailable(context_images, context_available),
        _zero_unavailable(query_images, query_available),
        context_available,
        query_available,
    )


def build_masked_item(
    coords3d: np.ndarray, expr: np.ndarray, slice_ids: np.ndarray,
    masking_cfg, images: np.ndarray | None, seed: int,
    context_gene_features: np.ndarray | None = None,
    context_gene_feature_provider=None,
    context_extra_features: np.ndarray | None = None,
    context_extra_feature_provider=None,
    organ: str | None = None, tech: str | None = None,
    augment: bool = False,
    image_mode: str = "full",
    query_image_dropout_p: float = 0.0,
    all_image_dropout_p: float = 0.0,
    context_gex_mode: str = "full",
    context_gex_dropout_p: float = 0.0,
    fixed_context_mask: np.ndarray | None = None,
    fixed_query_mask: np.ndarray | None = None,
    excluded_training_mask: np.ndarray | None = None,
    strict_broken_region: bool = False,
    query_patch_size: float = 224.0,
) -> dict:
    """One {context, query, target_expression} item for a single sample.

    context_gene_features / context_gene_feature_provider (mirrors the
    original's builtin gene_encoder_type="novae"/"scfoundation" REPLACEMENT
    channel): optional [N, feat_dim] array or a
    ContextOnlyFeatureProvider-style callable(context_mask)->array, used
    INSTEAD of raw expr for context["expression"] only. target_expression
    always comes from the real raw expr regardless — the task is predicting
    real gene expression, not an embedding of it. Used by Architecture 2
    (scFoundation as the primary gene encoder).

    context_extra_features / context_extra_feature_provider (Architecture 4's
    scFoundation-residual-into-frozen-STPath channel): a SEPARATE, ADDITIVE
    channel, stashed under context["extra_features"] — never replaces
    context["expression"]. Generalizes the original's context_novae_features
    slot; renamed since nothing in gen2_architectures is actually Novae.

    context_gex_mode: the explicit modality intervention for the image-only
    baseline and any missing-tissue ablation. "zero" removes BOTH raw
    context expression and the extra-features channel. Zeroing only one
    would leave the same biological signal available through the other.
    context_gex_dropout_p drops the complete context-GEX modality for a
    training item, independent of image dropout.
    """
    if fixed_context_mask is not None or fixed_query_mask is not None:
        if fixed_context_mask is None or fixed_query_mask is None:
            raise ValueError("fixed_context_mask and fixed_query_mask must be provided together")
        if augment:
            raise ValueError("fixed mask-bank items must not use coordinate augmentation")
        if excluded_training_mask is not None:
            raise ValueError("excluded_training_mask is only valid for random training draws")
        context_mask = np.asarray(fixed_context_mask, dtype=bool)
        query_mask = np.asarray(fixed_query_mask, dtype=bool)
    else:
        if augment:
            coords3d = augment_coords_xy(coords3d, seed=seed + 2)
        context_mask, query_mask = make_training_context_query_split(
            coords3d, slice_ids, masking_cfg, seed,
            excluded_training_mask=excluded_training_mask,
        )
    if context_gene_feature_provider is not None and context_gene_features is not None:
        raise ValueError("provide either context_gene_features or context_gene_feature_provider, not both")
    if context_extra_feature_provider is not None and context_extra_features is not None:
        raise ValueError("provide either context_extra_features or context_extra_feature_provider, not both")

    if context_gene_feature_provider is not None:
        context_expr = np.asarray(context_gene_feature_provider(context_mask), dtype=np.float32)
    else:
        context_expr_source = expr if context_gene_features is None else context_gene_features
        context_expr = np.asarray(context_expr_source[context_mask], dtype=np.float32)
    context = {
        "coords": torch.tensor(coords3d[context_mask], dtype=torch.float32),
        "expression": torch.tensor(context_expr, dtype=torch.float32),
    }
    if context_extra_feature_provider is not None:
        extra_context = np.asarray(context_extra_feature_provider(context_mask), dtype=np.float32)
        context["extra_features"] = torch.tensor(extra_context, dtype=torch.float32)
    elif context_extra_features is not None:
        context["extra_features"] = torch.tensor(
            context_extra_features[context_mask], dtype=torch.float32
        )

    context_gex_mode = str(context_gex_mode).lower()
    if context_gex_mode not in {"full", "zero", "shuffled"}:
        raise ValueError(f"context_gex_mode must be 'full', 'zero', or 'shuffled', got {context_gex_mode!r}")
    context_gex_dropout_p = float(context_gex_dropout_p)
    if not 0.0 <= context_gex_dropout_p <= 1.0:
        raise ValueError(f"context_gex_dropout_p must be in [0, 1], got {context_gex_dropout_p}")
    drop_context_gex = (
        context_gex_mode == "zero"
        or (context_gex_dropout_p > 0.0 and np.random.default_rng(seed + 3).random() < context_gex_dropout_p)
    )
    gex_keys = [key for key in ("expression", "extra_features") if key in context]
    if drop_context_gex:
        for key in gex_keys:
            context[key] = torch.zeros_like(context[key])
    elif context_gex_mode == "shuffled" and context["expression"].shape[0] > 1:
        permutation = torch.as_tensor(
            np.random.default_rng(seed + 4).permutation(context["expression"].shape[0]), dtype=torch.long,
        )
        for key in gex_keys:
            if context[key].shape[0] != permutation.numel():
                raise ValueError(f"context {key} row count does not match expression for shared GEX shuffle")
            context[key] = context[key][permutation]

    query = {"coords": torch.tensor(coords3d[query_mask], dtype=torch.float32)}
    if images is not None:
        c_img, q_img, c_available, q_available = _prepare_images_for_split(
            images, context_mask, query_mask, mode=image_mode, seed=seed,
            query_dropout_p=query_image_dropout_p, all_dropout_p=all_image_dropout_p,
        )
        context["images"], query["images"] = c_img, q_img
        context["image_available"], query["image_available"] = c_available, q_available
        if strict_broken_region and image_mode == "target_zero":
            safe = torch.as_tensor(
                nonoverlapping_context_patch_mask(
                    coords3d[context_mask], coords3d[query_mask], query_patch_size
                ),
                dtype=torch.bool,
            )
            context["image_available"] &= safe
            context["images"] = context["images"].clone()
            context["images"][~context["image_available"]] = 0

    if organ is not None:
        context["organ"] = organ
        query["organ"] = organ
    if tech is not None:
        context["tech"] = tech
        query["tech"] = tech
    target_expression = torch.tensor(expr[query_mask], dtype=torch.float32)
    return {"context": context, "query": query, "target_expression": target_expression}
