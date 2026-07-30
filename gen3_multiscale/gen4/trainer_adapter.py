"""Item 5 (six-launch-blocker audit): "Add minimal adapters to the
existing Gen3 trainer/evaluator. Reuse them; do not create another
trainer/evaluator."

`training/train.py::build_model_for_inference` is already the ONE real
model-reconstruction pipeline every Gen3 caller (the trainer loop, the
evaluator, the overfit gate, the residual-basis fitter) shares -- see its
own docstring. `evaluation/gen3_evaluator.py::evaluate_gen3_checkpoint`
already implements full/top50/top200 metrics, strata, patient
aggregation, baselines, split lock, query fingerprints, and generative
uncertainty (see CONTRACT.md's audit-response history) against whatever
model that ONE function hands it. This module therefore does not touch
either of those -- it is the missing piece `build_model_for_inference`
needs to also recognize a Gen4 config (`model.arm` present) and dispatch
to `gen4.model_factory`'s real builders + `gen4.staged_loader`'s real
staged-conditioner discipline instead of Gen3's own
`model_factory.build_architecture`/`maybe_load_pretrained_conditioner_
for_architecture4`. Once wired (see `training/train.py`'s own dispatch
at the top of `build_model_for_inference`), every existing caller gets a
real Gen4 model for free, with zero changes to their own code.

Scope of this pass (documented, not silently omitted): `model.kind ==
"conditioner"` and `"flow"` (Gen4) are wired here. `"latent_flow"`
(Gen5) is NOT -- Gen5's own `ExpressionAutoencoder` needs its exact
training-time `hidden_dim` to reconstruct an architecturally-identical
module before `gen4.staged_loader.load_shared_autoencoder` can load real
weights onto it, and no Gen5 flow config currently records that
dimension (`model.params.hidden_dim` in configs/gen5/*.yaml is the FLOW
model's own hidden_dim, a different, independently-configured value).
Wiring Gen5 requires that config-schema addition first; deferred rather
than guessing a dimension that could silently produce a wrong-shape (or
worse, coincidentally-same-shape-but-wrong-semantics) load.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from gen3_multiscale.gen4 import model_factory as gen4_model_factory
from gen3_multiscale.gen4 import staged_loader as gen4_staged_loader
from gen3_multiscale.gen4.model_factory import ARM_TABLE
from gen3_multiscale.models.gene_basis import load_gene_residual_basis
from gen3_multiscale.training import checkpoint as checkpoint_module

_UNLOADED_CONDITIONER_INFO = {
    "loaded": False, "checkpoint_dir": None, "checkpoint_sha256": None, "checkpoint_step": None,
    "checkpoint_bundle_id": None, "checkpoint_manifest_sha256": None,
}


def is_gen4_config(config: dict) -> bool:
    """The one predicate `build_model_for_inference` uses to decide
    whether to dispatch here instead of Gen3's own
    `model_factory.build_architecture` path -- Gen4/Gen5 configs declare
    `model.arm`, Gen3 configs declare `model.architecture` (a numeric
    string id); the two schemas never overlap."""
    return "arm" in (config.get("model") or {})


def _maybe_build_stpath_encoder(config: dict, gene_names: list[str], image_feature_dim: int):
    """Mirrors `training/train.py::maybe_build_slide_encoder`'s own
    "only construct when the arm actually needs it, and only when a real
    checkpoint is configured" discipline, applied to
    `gen4.stpath_context.Gen4STPathContextEncoder` (arm D/3, arm 4's
    STPath-consuming path). No real STPath package/weights are available
    in this environment (GEN4_CONTRACT.md section 13's documented gap) --
    this function is structurally complete and exercised in tests only
    via a stub `stpath_encoder=` passed directly to `build_gen4_
    conditioner`/`build_gen4_flow`; real-weight construction is listed as
    an explicit gap in the runbook (Item 6)."""
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    if arm not in ARM_TABLE or ARM_TABLE[arm]["image_feature_source"] not in ("stpath_context", "hybrid_context"):
        return None
    fingerprints = config.get("required_fingerprints") or {}
    checkpoint_path = fingerprints.get("stpath_checkpoint")
    gene_vocab_path = fingerprints.get("stpath_gene_vocab")
    if not checkpoint_path or not gene_vocab_path:
        return None
    from gen3_multiscale.gen4.stpath_context import Gen4STPathContextEncoder

    return Gen4STPathContextEncoder(
        gene_names=gene_names, gene_voc_path=str(gene_vocab_path), model_weight_path=str(checkpoint_path),
        hidden_dim=image_feature_dim,
    )


def build_gen4_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False, staged_smoke: bool = False,
) -> tuple[nn.Module, dict]:
    """Gen4 equivalent of `train.py::build_model_for_inference` (called
    from that same function once it detects `is_gen4_config(config)`) --
    construct (conditioner or flow), stage-load+freeze the flow's
    conditioner from `required_fingerprints.gen4_conditioner_checkpoint`
    when `kind == "flow"`, then optionally load `checkpoint_dir`'s
    trainable weights on top, exactly mirroring Gen3's own
    checkpoint_dir=None-means-freshly-initialized / real-path-means-
    verify-and-load contract."""
    model_cfg = config.get("model") or {}
    kind = str(model_cfg.get("kind", ""))
    if kind not in ("conditioner", "flow"):
        raise ValueError(f"build_gen4_model_for_inference: unsupported model.kind {kind!r} (expected 'conditioner' or 'flow')")
    data_cfg = config.get("data") or {}
    training_cfg = config.get("training") or {}
    params = model_cfg.get("params") or {}
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    image_feature_dim = int(params.get("image_feature_dim", 1536))
    seed = int(training_cfg.get("seed", 0))

    stpath_encoder = _maybe_build_stpath_encoder(config, gene_names, image_feature_dim)

    torch.manual_seed(seed)
    if kind == "conditioner":
        model = gen4_model_factory.build_gen4_conditioner(
            config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            stpath_encoder=stpath_encoder, seed=seed,
        ).to(device)
        conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)
    else:
        fingerprints = config.get("required_fingerprints") or {}
        basis_path = fingerprints.get("gene_residual_basis")
        if not basis_path:
            if not smoke:
                raise ValueError(
                    "a Gen4 flow config requires required_fingerprints.gene_residual_basis -- fit one "
                    "offline (gen4/basis_fit.py) from the matching, already-trained gen4 conditioner first"
                )
            gene_basis = None
        else:
            gene_basis = load_gene_residual_basis(basis_path)
        if gene_basis is None:
            # Construction-only smoke with no real basis configured yet --
            # never reachable for a real (non-smoke) run per the check above.
            from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
            import numpy as np

            rank = int(params.get("gene_basis_rank", 8))
            gene_basis = fit_gene_residual_basis(
                np.zeros((max(rank + 1, 2), n_genes), dtype=np.float32), gene_names, rank=rank,
            )
        model = gen4_model_factory.build_gen4_flow(
            config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            gene_basis=gene_basis, gene_names=gene_names, stpath_encoder=stpath_encoder, seed=seed,
        ).to(device)

        conditioner_checkpoint_dir = (config.get("required_fingerprints") or {}).get("gen4_conditioner_checkpoint")
        needs_conditioner = not (smoke and not staged_smoke)
        if conditioner_checkpoint_dir and needs_conditioner:
            conditioner_info = gen4_staged_loader.load_and_freeze_deterministic_conditioner(
                model, str(conditioner_checkpoint_dir), gene_names,
            )
        elif needs_conditioner:
            raise ValueError(
                "a Gen4 flow config requires required_fingerprints.gen4_conditioner_checkpoint -- a real, "
                "already-trained, validation-selected gen4 conditioner checkpoint_dir. Train the matching "
                "conditioner arm to completion first; a Gen4 flow model must never start training from a "
                "random or unstaged conditioner"
            )
        else:
            conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)

    resolved_checkpoint_identity = None
    if checkpoint_dir is not None:
        resolved_checkpoint_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
        checkpoint_module.verify_gene_names(resolved_checkpoint_identity.resolved_dir, gene_names)
        checkpoint_module.load_trainable_state(model, resolved_checkpoint_identity.resolved_dir)

    return model, {
        "conditioner_info": conditioner_info,
        "checkpoint_bundle_id": resolved_checkpoint_identity.bundle_dir if resolved_checkpoint_identity else None,
        "checkpoint_manifest_sha256": resolved_checkpoint_identity.manifest_sha256 if resolved_checkpoint_identity else None,
        "trainable_weights_sha256": resolved_checkpoint_identity.weights_sha256 if resolved_checkpoint_identity else None,
    }
