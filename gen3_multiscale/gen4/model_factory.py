"""Gen4 arm dispatch -- config dict -> real Gen4Conditioner/Gen4ResidualFlowModel
instance. Mirrors the shape of `models/model_factory.py::build_architecture`
(fail closed on unrecognized config fields, seed applied immediately before
construction) but is a fully separate module/dispatch table -- Gen3's own
`model_factory.py` is never imported or modified by this file.
"""
from __future__ import annotations

import inspect

import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.flow import Gen4ResidualFlowModel
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.models.gene_basis import GeneResidualBasis

# arm -> (gex_feature_source, global_context_source, image_feature_source) --
# GEN4_CONTRACT.md section 2.
#
# Intended-design arm mapping (Adam's original four-arm design; internal
# gen4a-e keys below are unchanged for backward compatibility with existing
# configs/tests, since a blanket key rename would touch every config/test
# file for zero behavioral gain -- this table IS the canonical mapping):
#   gen4c -> intended Arm 1 (UNI2 + scFoundation)
#   gen4b -> intended Arm 2 (GigaPath + scFoundation)
#   gen4d -> intended Arm 3 (STPath joint conditioner)
#   gen4e -> intended Arm 4 (STPath + UNI2 + scFoundation hybrid -- NEW)
#   gen4a -> optional baseline (UNI2 + trainable weighted-linear GEX), not
#            one of the four primary long-run arms
INTENDED_ARM_NAMES = {
    "gen4c": "arm1_uni2_scfoundation",
    "gen4b": "arm2_gigapath_scfoundation",
    "gen4d": "arm3_stpath",
    "gen4e": "arm4_hybrid_stpath_uni2_scfoundation",
    "gen4a": "baseline_uni2_weighted_linear",
}
ARM_TABLE = {
    "gen4a": {"gex_feature_source": "weighted_linear", "global_context_source": "uni2_pool", "image_feature_source": "precomputed"},
    "gen4b": {"gex_feature_source": "frozen_context", "global_context_source": "gigapath", "image_feature_source": "precomputed"},
    "gen4c": {"gex_feature_source": "frozen_context", "global_context_source": "uni2_pool", "image_feature_source": "precomputed"},
    "gen4d": {"gex_feature_source": "stpath_joint", "global_context_source": "none", "image_feature_source": "stpath_context"},
    "gen4e": {"gex_feature_source": "hybrid_context", "global_context_source": "none", "image_feature_source": "hybrid_context"},
}

_KNOWN_NON_CONSTRUCTOR_FIELDS = frozenset({"init_seed"})
_FLOW_EXTRA_NON_CONSTRUCTOR_FIELDS = frozenset({"gene_basis_rank"})


def _constructor_param_names(cls: type) -> frozenset[str]:
    return frozenset(inspect.signature(cls.__init__).parameters) - {"self"}


def _resolve_seed(model_cfg: dict, seed: int | None) -> int | None:
    return seed if seed is not None else (model_cfg.get("params") or {}).get("init_seed")


def _build_uni2_pool_if_needed(arm: str, model_cfg: dict, image_feature_dim: int, global_slide_dim: int):
    if ARM_TABLE[arm]["global_context_source"] != "uni2_pool":
        return None
    pool_params = dict((model_cfg.get("uni2_global_pool") or {}))
    return MaskAwareCoordinateAttentionPool(
        tile_feature_dim=image_feature_dim, output_dim=global_slide_dim,
        coord_dim=pool_params.get("coord_dim", 64), hidden_dim=pool_params.get("hidden_dim", 256),
        n_heads=pool_params.get("n_heads", 4),
    )


def _resolve_kwargs(
    config: dict, cls: type, *, extra_metadata: frozenset[str], n_genes: int, gex_feature_dim: int,
    image_feature_dim: int, gex_context_embedding_dim: int | None, slide_encoder, gigapath_checkpoint_sha256,
    uni2_global_pool, stpath_encoder, gene_basis: GeneResidualBasis | None, gene_names: list[str] | None,
) -> dict:
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    if arm not in ARM_TABLE:
        raise ValueError(f"unknown model.arm {arm!r} -- expected one of {sorted(ARM_TABLE)}")
    accepted = _constructor_param_names(cls)
    known_metadata = _KNOWN_NON_CONSTRUCTOR_FIELDS | extra_metadata

    raw_params = dict(model_cfg.get("params") or {})
    kwargs: dict = {}
    for key, value in raw_params.items():
        if key in {"n_genes", "gex_feature_dim", "image_feature_dim", "gex_context_embedding_dim"}:
            continue  # always supplied explicitly below, from real data/cache identity, never from static config
        if key in accepted:
            kwargs[key] = value
        elif key in known_metadata:
            continue
        else:
            raise ValueError(
                f"config field model.params.{key!r} is neither a constructor parameter of {cls.__name__} "
                f"nor a known non-constructor field ({sorted(known_metadata)}) -- refusing to silently drop it"
            )

    kwargs.update(ARM_TABLE[arm])
    kwargs["n_genes"] = n_genes
    kwargs["gex_feature_dim"] = gex_feature_dim
    kwargs["image_feature_dim"] = image_feature_dim
    if kwargs["gex_feature_source"] in ("frozen_context", "hybrid_context"):
        if not gex_context_embedding_dim:
            raise ValueError(f"arm {arm!r} requires gex_context_embedding_dim (frozen scFoundation cache width)")
        kwargs["gex_context_embedding_dim"] = gex_context_embedding_dim
    if kwargs["global_context_source"] == "gigapath":
        if slide_encoder is None or not gigapath_checkpoint_sha256:
            raise ValueError(f"arm {arm!r} requires a real slide_encoder and gigapath_checkpoint_sha256")
        kwargs["slide_encoder"] = slide_encoder
        kwargs["gigapath_checkpoint_sha256"] = gigapath_checkpoint_sha256
    elif kwargs["global_context_source"] == "uni2_pool":
        if uni2_global_pool is None:
            raise ValueError(f"arm {arm!r} requires a real uni2_global_pool module")
        kwargs["uni2_global_pool"] = uni2_global_pool
    if kwargs["image_feature_source"] in ("stpath_context", "hybrid_context"):
        if stpath_encoder is None:
            raise ValueError(f"arm {arm!r} requires a real stpath_encoder module")
        kwargs["stpath_encoder"] = stpath_encoder

    if cls is Gen4ResidualFlowModel:
        if gene_basis is None or gene_names is None:
            raise ValueError("the flow model requires gene_basis and gene_names (fit offline)")
        kwargs["gene_basis"] = gene_basis
        kwargs["gene_names"] = gene_names

    required = {
        name for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    }
    missing = required - set(kwargs)
    if missing:
        raise ValueError(f"config is missing required constructor arguments for {cls.__name__}: {sorted(missing)}")
    return kwargs


def build_gen4_conditioner(
    config: dict, *, n_genes: int, gex_feature_dim: int, image_feature_dim: int,
    gex_context_embedding_dim: int | None = None, slide_encoder=None, gigapath_checkpoint_sha256: str | None = None,
    uni2_global_pool=None, stpath_encoder=None, seed: int | None = None,
) -> Gen4Conditioner:
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    # Codex audit (first Gen4 push), confirmed real: seeding MUST happen
    # before any trainable submodule is constructed, including the
    # uni2_global_pool this function builds on the caller's behalf below --
    # a prior version seeded only right before the top-level model's own
    # construction, after MaskAwareCoordinateAttentionPool's own random
    # init had already consumed RNG state from an UNSEEDED stream, so two
    # "same seed" calls produced different pool weights.
    effective_seed = _resolve_seed(model_cfg, seed)
    if effective_seed is not None:
        torch.manual_seed(int(effective_seed))
    if uni2_global_pool is None and arm in ARM_TABLE and ARM_TABLE[arm]["global_context_source"] == "uni2_pool":
        global_slide_dim = int((model_cfg.get("params") or {}).get("global_slide_dim", 768))
        uni2_global_pool = _build_uni2_pool_if_needed(arm, model_cfg, image_feature_dim, global_slide_dim)
    kwargs = _resolve_kwargs(
        config, Gen4Conditioner, extra_metadata=frozenset(), n_genes=n_genes, gex_feature_dim=gex_feature_dim,
        image_feature_dim=image_feature_dim, gex_context_embedding_dim=gex_context_embedding_dim,
        slide_encoder=slide_encoder, gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
        uni2_global_pool=uni2_global_pool, stpath_encoder=stpath_encoder, gene_basis=None, gene_names=None,
    )
    return Gen4Conditioner(**kwargs)


def build_gen4_flow(
    config: dict, *, n_genes: int, gex_feature_dim: int, image_feature_dim: int, gene_basis: GeneResidualBasis,
    gene_names: list[str], gex_context_embedding_dim: int | None = None, slide_encoder=None,
    gigapath_checkpoint_sha256: str | None = None, uni2_global_pool=None, stpath_encoder=None,
    seed: int | None = None,
) -> Gen4ResidualFlowModel:
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    # See build_gen4_conditioner's identical comment: seed BEFORE
    # constructing any trainable submodule, including uni2_global_pool.
    effective_seed = _resolve_seed(model_cfg, seed)
    if effective_seed is not None:
        torch.manual_seed(int(effective_seed))
    if uni2_global_pool is None and arm in ARM_TABLE and ARM_TABLE[arm]["global_context_source"] == "uni2_pool":
        global_slide_dim = int((model_cfg.get("params") or {}).get("global_slide_dim", 768))
        uni2_global_pool = _build_uni2_pool_if_needed(arm, model_cfg, image_feature_dim, global_slide_dim)
    kwargs = _resolve_kwargs(
        config, Gen4ResidualFlowModel, extra_metadata=_FLOW_EXTRA_NON_CONSTRUCTOR_FIELDS,
        n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
        gex_context_embedding_dim=gex_context_embedding_dim, slide_encoder=slide_encoder,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256, uni2_global_pool=uni2_global_pool,
        stpath_encoder=stpath_encoder, gene_basis=gene_basis, gene_names=gene_names,
    )
    return Gen4ResidualFlowModel(**kwargs)
