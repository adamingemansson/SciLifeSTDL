"""Item 4 (six-launch-blocker audit): "Add one shared staged loader that
loads and verifies the exact selected deterministic-conditioner
checkpoint, then freezes/evals it before Gen4 or Gen5 flow training.
Gen5 must also load the exact shared autoencoder."

Mirrors `training/train.py::maybe_load_pretrained_conditioner_for_
architecture4` -- Architecture 4's own real, already-audited staged-
conditioner discipline (load -> verify gene identity -> load trainable
state -> freeze -> eval, with an identity dict a run manifest can bind
resume/evaluation to) -- generalized so ONE function serves BOTH
`Gen4ResidualFlowModel.conditioner` and `Gen5LatentFlowModel.conditioner`,
since both are real `Gen4Conditioner` instances (Gen5 reuses
Gen4Conditioner unmodified via gen5/model_factory.py's Gen5_TO_GEN4_ARM
mapping) and the identical load/verify/freeze/eval sequence applies to
both without any new architecture-specific logic. Not a new trainer:
this module owns nothing about masks/optimizers/loops -- it only
performs the one-time staged-checkpoint load a real Gen4/Gen5 flow
training entrypoint must run before constructing an optimizer over the
flow's parameters (Item 5's job is to actually build that entrypoint by
adapting `training/train.py`).
"""
from __future__ import annotations

import torch.nn as nn

from gen3_multiscale.training import checkpoint as checkpoint_module


def load_and_freeze_deterministic_conditioner(
    model: nn.Module, checkpoint_dir: str, gene_names: list[str], *, freeze: bool = True,
) -> dict:
    """`model` must have a real `.conditioner` submodule
    (Gen4ResidualFlowModel or Gen5LatentFlowModel). Verifies the
    checkpoint's saved gene panel matches `gene_names` EXACTLY before
    loading (mirrors `verify_gene_names`'s own audited rationale:
    trainable_weights.pt's tensors are positional, not gene-name-keyed,
    so a same-shaped but reordered/different panel would otherwise load
    silently and corrupt the run without any visible symptom). Resolves
    `checkpoint_dir` exactly ONCE (Codex re-audit of commit 7a2d819's
    "resolve-once" principle) and reuses that same resolved bundle for
    both the gene-name check and the weight load.

    `freeze=True` (default): every conditioner parameter has
    `requires_grad_(False)` and the conditioner is pinned to `eval()` --
    via `model.freeze_conditioner()` when present (Gen5LatentFlowModel;
    its own `train()` override then keeps the conditioner in eval() even
    across the rest of the model's future `.train()` calls), or directly
    on `model.conditioner` otherwise (Gen4ResidualFlowModel, which has no
    separate freeze method of its own -- the conditioner simply never
    receives an optimizer for its parameters in that case, matching
    Architecture4's own `freeze_conditioner_initially` discipline).

    Returns the same identity-binding dict shape
    `maybe_load_pretrained_conditioner_for_architecture4` returns
    (checkpoint_sha256/step/bundle_id/manifest_sha256) so a Gen4/Gen5 run
    manifest can bind resume/evaluation to the exact conditioner
    checkpoint a run started from."""
    if not hasattr(model, "conditioner"):
        raise ValueError(f"{type(model).__name__} has no .conditioner submodule to load a staged checkpoint onto")
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    checkpoint_module.verify_gene_names(identity.resolved_dir, gene_names)
    checkpoint_module.load_trainable_state(model.conditioner, identity.resolved_dir)
    if freeze:
        if hasattr(model, "freeze_conditioner"):
            model.freeze_conditioner()
        else:
            for parameter in model.conditioner.parameters():
                parameter.requires_grad_(False)
            model.conditioner.eval()
    return {
        "loaded": True, "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": identity.weights_sha256, "checkpoint_step": identity.step,
        "checkpoint_bundle_id": identity.bundle_dir, "checkpoint_manifest_sha256": identity.manifest_sha256,
    }


def load_shared_autoencoder(autoencoder: nn.Module, checkpoint_dir: str, gene_names: list[str]) -> dict:
    """Gen5-only half of Item 4: "Gen5 must also load the exact shared
    autoencoder." `autoencoder` must be a freshly-constructed
    `ExpressionAutoencoder`, architecturally identical (same n_genes/
    latent_dim/hidden_dim) to the one the checkpoint was saved from --
    pass the loaded result into `Gen5LatentFlowModel(autoencoder=...)`,
    which already freezes/evals it unconditionally at construction
    (gen5/latent_flow.py's own `__init__`), so this function does not
    duplicate that freeze step. Same resolve-once + gene-identity
    discipline as `load_and_freeze_deterministic_conditioner`."""
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    checkpoint_module.verify_gene_names(identity.resolved_dir, gene_names)
    checkpoint_module.load_trainable_state(autoencoder, identity.resolved_dir)
    return {
        "loaded": True, "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": identity.weights_sha256, "checkpoint_step": identity.step,
        "checkpoint_bundle_id": identity.bundle_dir, "checkpoint_manifest_sha256": identity.manifest_sha256,
    }
