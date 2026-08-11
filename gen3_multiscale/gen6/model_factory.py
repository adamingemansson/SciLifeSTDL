"""Gen6 config-to-model construction with fail-closed arm dispatch."""
from __future__ import annotations

import inspect

import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.gen6.conditioner import Gen6Conditioner
from gen3_multiscale.gen6.contract import get_gen6_arm_spec
from gen3_multiscale.gen6.stpath_model import FineTunedSTPathBenchmark


def _constructor_kwargs(params: dict) -> dict:
    accepted = set(inspect.signature(Gen4Conditioner.__init__).parameters) - {"self"}
    metadata = {"init_seed", "fusion_heads", "coord_scale"}
    unknown = set(params) - accepted - metadata - {
        "n_genes", "gex_feature_dim", "gex_context_embedding_dim", "image_feature_dim",
    }
    if unknown:
        raise ValueError(f"unknown Gen6 model.params fields: {sorted(unknown)}")
    return {key: value for key, value in params.items() if key in accepted}


def build_gen6_model(
    config: dict, *, gene_names: list[str], gex_feature_dim: int,
    image_feature_dim: int, gex_context_embedding_dim: int | None = None,
    slide_encoder=None, gigapath_checkpoint_sha256: str | None = None,
    seed: int = 0,
):
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    spec = get_gen6_arm_spec(arm)
    if str(model_cfg.get("kind", "")) != "conditioner":
        raise ValueError(
            f"{arm} construction currently expects model.kind='conditioner'; "
            "generative arms use their dedicated staged builders"
        )
    fingerprints = config.get("required_fingerprints") or {}
    torch.manual_seed(int(seed))
    if arm == "gen6a":
        checkpoint = fingerprints.get("stpath_checkpoint")
        vocab = fingerprints.get("stpath_gene_vocab")
        if not checkpoint or not vocab:
            raise ValueError("gen6a requires stpath_checkpoint and stpath_gene_vocab fingerprints")
        return FineTunedSTPathBenchmark(
            gene_names=gene_names, gene_vocab_path=str(vocab), checkpoint_path=str(checkpoint),
        )
    if spec.staged_conditioner:
        raise ValueError(f"{arm} must be built through its staged generative builder")

    params = dict(model_cfg.get("params") or {})
    kwargs = _constructor_kwargs(params)
    # Refinement is part of an arm's identity, so it is required where the
    # contract declares it and refused everywhere else. Without the second
    # branch a stray model.params entry would silently turn a control arm into
    # a refinement arm, and the screen would then be comparing nothing.
    refinement_steps = int(params.get("n_refinement_steps") or 0)
    if spec.spatial_model == "refined_spatial_field":
        if refinement_steps < 1:
            raise ValueError(f"{arm} requires positive model.params.n_refinement_steps")
    elif refinement_steps:
        raise ValueError(
            f"{arm} declares spatial_model={spec.spatial_model!r} and must not set "
            "model.params.n_refinement_steps"
        )
    kwargs.update({
        "n_genes": len(gene_names), "gex_feature_dim": int(gex_feature_dim),
        "image_feature_dim": int(image_feature_dim),
        "model_architecture_version": f"gen6-{arm}-v1",
    })
    if spec.gene_encoder == "scfoundation":
        if not gex_context_embedding_dim:
            raise ValueError(f"{arm} requires gex_context_embedding_dim")
        kwargs.update({
            "gex_feature_source": "frozen_context",
            "gex_context_embedding_dim": int(gex_context_embedding_dim),
        })
    else:
        kwargs["gex_feature_source"] = "weighted_linear"
    kwargs["image_feature_source"] = "precomputed"
    kwargs["use_global_gex"] = spec.spatial_model == "full_spatial_field"
    kwargs["use_regional_he"] = spec.spatial_model == "full_spatial_field"
    uni2_pool = None
    if spec.image_encoder == "gigapath_longnet":
        if slide_encoder is None or not gigapath_checkpoint_sha256:
            raise ValueError(f"{arm} requires a real GigaPath slide encoder and checkpoint SHA256")
        kwargs.update({
            "global_context_source": "gigapath", "slide_encoder": slide_encoder,
            "gigapath_checkpoint_sha256": gigapath_checkpoint_sha256,
        })
    elif spec.spatial_model == "full_spatial_field":
        global_dim = int(kwargs.get("global_slide_dim", 768))
        uni2_pool = MaskAwareCoordinateAttentionPool(
            tile_feature_dim=image_feature_dim, output_dim=global_dim,
        )
        kwargs.update({"global_context_source": "uni2_pool", "uni2_global_pool": uni2_pool})
    else:
        kwargs["global_context_source"] = "none"
    return Gen6Conditioner(
        arm_spec=spec, fusion_heads=int(params.get("fusion_heads", 4)),
        coord_scale=float(params.get("coord_scale", 0.0)), **kwargs,
    )
