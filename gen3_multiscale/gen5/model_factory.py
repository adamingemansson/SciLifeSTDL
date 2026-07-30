"""Gen5 arm dispatch -- config dict -> real Gen5LatentFlowModel instance.
Reuses `gen4.model_factory.ARM_TABLE` directly (imported, not copied) so
Gen5's arm->(gex_feature_source, global_context_source, image_feature_source)
mapping can never silently drift from Gen4's own -- GEN5_CONTRACT.md
section 2's "matched conditioning systems" requirement.
"""
from __future__ import annotations

import inspect

import torch

from gen3_multiscale.gen4.model_factory import ARM_TABLE, _build_uni2_pool_if_needed
from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder
from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel

_KNOWN_NON_CONSTRUCTOR_FIELDS = frozenset({"init_seed"})

# Gen5 arm ids (gen5a-gen5d) mirror Gen4's own (gen4a-gen4d) one-to-one --
# GEN5_CONTRACT.md section 2's table. `ARM_TABLE` (gen4/model_factory.py)
# is keyed by the gen4 ids; this mapping is the single place that
# translates between the two naming schemes so it can never silently
# drift out of sync arm-by-arm.
GEN5_TO_GEN4_ARM = {"gen5a": "gen4a", "gen5b": "gen4b", "gen5c": "gen4c", "gen5d": "gen4d"}


def _constructor_param_names(cls: type) -> frozenset[str]:
    return frozenset(inspect.signature(cls.__init__).parameters) - {"self"}


def build_gen5_model(
    config: dict, *,
    n_genes: int, gene_names: list[str], gex_feature_dim: int, image_feature_dim: int,
    autoencoder: ExpressionAutoencoder, gex_context_embedding_dim: int | None = None,
    slide_encoder=None, gigapath_checkpoint_sha256: str | None = None, uni2_global_pool=None,
    stpath_encoder=None, seed: int | None = None,
) -> Gen5LatentFlowModel:
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    if arm not in GEN5_TO_GEN4_ARM:
        raise ValueError(f"unknown model.arm {arm!r} -- expected one of {sorted(GEN5_TO_GEN4_ARM)}")
    gen4_arm = GEN5_TO_GEN4_ARM[arm]

    # Codex audit (Gen4 finding, applies identically here), confirmed
    # real: seed BEFORE constructing any trainable submodule, including
    # uni2_global_pool below -- a prior version seeded only right before
    # Gen5LatentFlowModel's own construction, after the pool's random init
    # had already consumed RNG state from an unseeded stream.
    effective_seed = seed if seed is not None else (model_cfg.get("params") or {}).get("init_seed")
    if effective_seed is not None:
        torch.manual_seed(int(effective_seed))

    if uni2_global_pool is None and ARM_TABLE[gen4_arm]["global_context_source"] == "uni2_pool":
        global_slide_dim = int((model_cfg.get("params") or {}).get("global_slide_dim", 768))
        uni2_global_pool = _build_uni2_pool_if_needed(gen4_arm, model_cfg, image_feature_dim, global_slide_dim)

    accepted = _constructor_param_names(Gen5LatentFlowModel)
    raw_params = dict(model_cfg.get("params") or {})
    declared_latent_dim = raw_params.get("latent_dim")
    if declared_latent_dim is not None and int(declared_latent_dim) != autoencoder.latent_dim:
        raise ValueError(
            f"config declares model.params.latent_dim={declared_latent_dim}, but the real autoencoder's "
            f"latent_dim is {autoencoder.latent_dim} -- refusing to silently ignore the mismatch"
        )
    kwargs: dict = {}
    for key, value in raw_params.items():
        if key in {"n_genes", "gex_feature_dim", "image_feature_dim", "gex_context_embedding_dim", "gene_basis_rank", "latent_dim"}:
            # gene_basis_rank has no Gen5 meaning (documented metadata only);
            # latent_dim is determined by the real `autoencoder` argument,
            # never re-specified as a separate constructor kwarg -- checked
            # for consistency above instead.
            continue
        if key in accepted:
            kwargs[key] = value
        elif key in _KNOWN_NON_CONSTRUCTOR_FIELDS:
            continue
        else:
            raise ValueError(
                f"config field model.params.{key!r} is neither a constructor parameter of "
                f"Gen5LatentFlowModel nor a known non-constructor field "
                f"({sorted(_KNOWN_NON_CONSTRUCTOR_FIELDS)}) -- refusing to silently drop it"
            )

    kwargs.update(ARM_TABLE[gen4_arm])
    kwargs["n_genes"] = n_genes
    kwargs["gene_names"] = gene_names
    kwargs["gex_feature_dim"] = gex_feature_dim
    kwargs["image_feature_dim"] = image_feature_dim
    kwargs["autoencoder"] = autoencoder
    if kwargs["gex_feature_source"] == "frozen_context":
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
    if kwargs["image_feature_source"] == "stpath_context":
        if stpath_encoder is None:
            raise ValueError(f"arm {arm!r} requires a real stpath_encoder module")
        kwargs["stpath_encoder"] = stpath_encoder

    required = {
        name for name, param in inspect.signature(Gen5LatentFlowModel.__init__).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    }
    missing = required - set(kwargs)
    if missing:
        raise ValueError(f"config is missing required constructor arguments for Gen5LatentFlowModel: {sorted(missing)}")

    return Gen5LatentFlowModel(**kwargs)
