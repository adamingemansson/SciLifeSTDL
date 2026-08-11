"""Item 5 (six-launch-blocker audit) + Integration audit follow-up:
"Add minimal adapters to the existing Gen3 trainer/evaluator. Reuse
them; do not create another trainer/evaluator."

`training/train.py::build_model_for_inference` is already the ONE real
model-reconstruction pipeline every Gen3 caller (the trainer loop, the
evaluator, the overfit gate, the residual-basis fitter) shares -- see its
own docstring. `evaluation/gen3_evaluator.py::evaluate_gen3_checkpoint`
already implements full/top50/top200 metrics, strata, patient
aggregation, baselines, split lock, query fingerprints, and generative
uncertainty (see CONTRACT.md's audit-response history) against whatever
model that ONE function hands it. This module therefore does not touch
either of those -- it is the missing piece `build_model_for_inference`
needs to also recognize a Gen4/Gen5 config (`model.arm` present) and
dispatch to `gen4.model_factory`/`gen5.model_factory`'s real builders +
`gen4.staged_loader`'s real staged-conditioner discipline instead of
Gen3's own `model_factory.build_architecture`/`maybe_load_pretrained_
conditioner_for_architecture4`. Once wired (see `training/train.py`'s
own dispatch at the top of `build_model_for_inference`), every existing
caller gets a real Gen4/Gen5 model for free, with zero changes of their
own.

`model.kind == "conditioner"`/`"flow"` (Gen4) and `"latent_flow"` (Gen5)
are all wired here. Gen5's shared `ExpressionAutoencoder` is loaded via
`gen5.autoencoder.load_expression_autoencoder_checkpoint` -- that
function ALREADY reconstructs the exact architecture from the
checkpoint's own saved n_genes/latent_dim/hidden_dim (see
`gen4/staged_loader.py`'s own docstring for why this module no longer
defines a second, incompatible autoencoder loader of its own), so no
extra config field is needed to know its dimensions ahead of time.
"""
from __future__ import annotations

import hashlib
import json
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


def _numeric_basis_sha256(gene_basis) -> str:
    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(gene_basis.basis.detach().cpu().numpy()).tobytes()
    ).hexdigest()


def _pin_and_verify_conditioner(
    checkpoint_dir: str | Path,
    *,
    expected_arm: str,
    dataset_manifest: dict,
    gene_names: list[str],
    cache_content_by_sample: dict[str, dict] | None,
    allow_code_drift: bool = False,
):
    """Resolve once and verify the staged conditioner in its own config.

    A flow config is intentionally not config-identical to its
    conditioner config, so checkpoint verification must use the
    conditioner bundle's own model_config.json while still requiring the
    current dataset, gene panel, and per-sample cache content to match.
    """
    identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
    bundled_config_path = identity.resolved_dir / "model_config.json"
    if not bundled_config_path.is_file():
        raise ValueError(f"{identity.resolved_dir}: staged conditioner has no model_config.json")
    conditioner_config = json.loads(bundled_config_path.read_text())
    if str((conditioner_config.get("model") or {}).get("kind")) != "conditioner":
        raise ValueError(
            f"{identity.resolved_dir}: staged checkpoint is not a deterministic conditioner"
        )
    actual_arm = _resolve_gen4_arm(conditioner_config)
    if actual_arm != expected_arm:
        raise ValueError(
            f"staged conditioner arm {actual_arm!r} does not match requested flow arm {expected_arm!r}"
        )
    from gen3_multiscale.training.train import verify_full_checkpoint_identity

    verify_full_checkpoint_identity(
        identity.resolved_dir,
        config=conditioner_config,
        dataset_manifest=dataset_manifest,
        gene_names=gene_names,
        cache_content_by_sample=cache_content_by_sample,
        allow_code_drift=allow_code_drift,
    )
    return identity


def _verify_gen4_basis_provenance(
    basis_path: str | Path,
    gene_basis,
    *,
    dataset_manifest: dict,
    gene_names: list[str],
    conditioner_identity,
    cache_content_by_sample: dict[str, dict] | None,
) -> dict:
    """Require the residual basis to describe this exact staged run."""
    from gen3_multiscale.data.dataset_manifest import gene_panel_hash
    from gen3_multiscale.training.train import dataset_manifest_fingerprint

    conditioner_run_manifest = checkpoint_module.load_checkpoint_run_manifest(
        conditioner_identity.resolved_dir
    )
    if not conditioner_run_manifest:
        raise ValueError(
            f"{conditioner_identity.resolved_dir}: staged conditioner has no bound run manifest"
        )
    provenance_path = Path(f"{basis_path}.provenance.json")
    if not provenance_path.is_file():
        raise ValueError(
            f"Gen4 residual basis provenance is missing at {provenance_path}; "
            "fit it with scripts.fit_gen4_residual_basis"
        )
    provenance = json.loads(provenance_path.read_text())
    expected = {
        "kind": "gen4_residual_basis_provenance",
        "conditioner_config_identity_fingerprint": conditioner_run_manifest.get(
            "config_identity_fingerprint"
        ),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "conditioner_checkpoint_sha256": conditioner_identity.weights_sha256,
        "conditioner_checkpoint_step": conditioner_identity.step,
        "conditioner_checkpoint_bundle_id": conditioner_identity.bundle_dir,
        "conditioner_checkpoint_manifest_sha256": conditioner_identity.manifest_sha256,
        "gene_residual_basis_sha256": _numeric_basis_sha256(gene_basis),
    }
    mismatches = {
        field: (provenance.get(field), value)
        for field, value in expected.items()
        if provenance.get(field) != value
    }
    if mismatches:
        raise ValueError(
            f"Gen4 residual basis {basis_path} provenance does not match this run: {mismatches}"
        )
    train_ids = sorted(dataset_manifest.get("train_sample_ids") or [])
    if sorted(provenance.get("train_sample_ids") or []) != train_ids:
        raise ValueError(
            f"Gen4 residual basis {basis_path} was not fit on the manifest's exact training sample set"
        )
    reports = provenance.get("mask_schedule_reports")
    recorded_schedule_fingerprint = provenance.get("training_mask_schedule_fingerprint")
    if not reports or not recorded_schedule_fingerprint:
        raise ValueError(
            f"Gen4 residual basis {basis_path} does not bind its realized training-mask schedule"
        )
    recomputed_schedule_fingerprint = hashlib.sha256(
        json.dumps(reports, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    if recorded_schedule_fingerprint != recomputed_schedule_fingerprint:
        raise ValueError(
            f"Gen4 residual basis {basis_path} has an inconsistent training-mask schedule fingerprint"
        )
    recorded_cache = provenance.get("cache_content_by_sample")
    if not isinstance(recorded_cache, dict):
        raise ValueError(f"Gen4 residual basis {basis_path} has no per-sample cache provenance")
    missing_recorded = sorted(set(train_ids) - set(recorded_cache))
    if missing_recorded:
        raise ValueError(
            f"Gen4 residual basis {basis_path} omitted training cache identities for {missing_recorded}"
        )
    live_cache = cache_content_by_sample or {}
    for sample_id in sorted(set(train_ids) & set(live_cache)):
        if recorded_cache[sample_id] != live_cache[sample_id]:
            raise ValueError(
                f"Gen4 residual basis {basis_path} cache identity for {sample_id!r} "
                "does not match the live preflight"
            )
    return provenance


def is_gen4_config(config: dict) -> bool:
    """The one predicate `build_model_for_inference` uses to decide
    whether to dispatch here instead of Gen3's own
    `model_factory.build_architecture` path -- Gen4/Gen5 configs declare
    `model.arm`, Gen3 configs declare `model.architecture` (a numeric
    string id); the two schemas never overlap."""
    return "arm" in (config.get("model") or {})


def _resolve_gen4_arm(config: dict) -> str:
    """A Gen4 config's `model.arm` already IS a gen4-style key (gen4a-e).
    A Gen5 config's `model.arm` is a gen5-style key (gen5a-e) -- translate
    it through `GEN5_TO_GEN4_ARM` so the encoder-construction helpers
    below (both keyed by `gen4.model_factory.ARM_TABLE`) work identically
    for either caller without needing two copies of the same logic."""
    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    if arm in ARM_TABLE:
        return arm
    from gen3_multiscale.gen5.model_factory import GEN5_TO_GEN4_ARM

    return GEN5_TO_GEN4_ARM.get(arm, arm)


def _maybe_build_gigapath_slide_encoder(config: dict, gen4_arm: str):
    """Gen4-arm-table equivalent of `train.py::maybe_build_slide_encoder`
    -- that function gates on Gen3's own `model.params.use_global_slide`
    flag, which no Gen4/Gen5 config ever sets (arm needs are derived from
    `ARM_TABLE[arm]["global_context_source"]` instead), so it can't be
    called as-is; this reuses the SAME `FrozenGigaPathSlideEncoder`
    construction + `required_fingerprints.gigapath_checkpoint` sourcing,
    just gated on the real Gen4/Gen5 condition. Returns `(None, None)`
    when the arm doesn't need one."""
    if gen4_arm not in ARM_TABLE or ARM_TABLE[gen4_arm]["global_context_source"] != "gigapath":
        return None, None
    checkpoint_path = (config.get("required_fingerprints") or {}).get("gigapath_checkpoint")
    if not checkpoint_path:
        return None, None
    from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder

    encoder = FrozenGigaPathSlideEncoder(str(checkpoint_path))
    return encoder, encoder.checkpoint_sha256


def _maybe_build_stpath_encoder(config: dict, gen4_arm: str, gene_names: list[str], image_feature_dim: int):
    """Mirrors `training/train.py::maybe_build_slide_encoder`'s own
    "only construct when the arm actually needs it, and only when a real
    checkpoint is configured" discipline, applied to
    `gen4.stpath_context.Gen4STPathContextEncoder` (arm D/3, arm 4's
    STPath-consuming path -- for BOTH Gen4 and Gen5, via `gen4_arm`).
    No real STPath package/weights are available in this environment
    (GEN4_CONTRACT.md section 13's documented gap) -- this function is
    structurally complete and exercised in tests only via a stub
    `stpath_encoder=` passed directly to `build_gen4_conditioner`/
    `build_gen4_flow`/`build_gen5_model`; real-weight construction is
    listed as an explicit gap in the runbook (Item 6)."""
    if gen4_arm not in ARM_TABLE or ARM_TABLE[gen4_arm]["image_feature_source"] not in ("stpath_context", "hybrid_context"):
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
    dataset_manifest: dict | None = None,
    cache_content_by_sample: dict[str, dict] | None = None,
    allow_code_drift: bool = False,
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
    # Integration audit finding #3 (CONFIRMED real): this was never read
    # from config at all, so every scFoundation-consuming arm (frozen_context/
    # hybrid_context -- gen4b/c/e) unconditionally hit
    # gen4_model_factory's own "requires gex_context_embedding_dim" guard.
    gex_context_embedding_dim = params.get("gex_context_embedding_dim")
    gex_context_embedding_dim = int(gex_context_embedding_dim) if gex_context_embedding_dim else None

    gen4_arm = _resolve_gen4_arm(config)
    stpath_encoder = _maybe_build_stpath_encoder(config, gen4_arm, gene_names, image_feature_dim)
    # Integration audit finding #3 (CONFIRMED real): no GigaPath slide
    # encoder was ever constructed here, so gen4b (global_context_source
    # == "gigapath") would fail on that guard right after the
    # gex_context_embedding_dim one above was fixed.
    slide_encoder, gigapath_checkpoint_sha256 = _maybe_build_gigapath_slide_encoder(config, gen4_arm)

    torch.manual_seed(seed)
    basis_info = None
    if kind == "conditioner":
        model = gen4_model_factory.build_gen4_conditioner(
            config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            gex_context_embedding_dim=gex_context_embedding_dim, slide_encoder=slide_encoder,
            gigapath_checkpoint_sha256=gigapath_checkpoint_sha256, stpath_encoder=stpath_encoder, seed=seed,
        ).to(device)
        conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)
    else:
        fingerprints = config.get("required_fingerprints") or {}
        basis_path = fingerprints.get("gene_residual_basis")
        if not basis_path:
            if not smoke:
                raise ValueError(
                    "a Gen4 flow config requires required_fingerprints.gene_residual_basis -- fit one "
                    "offline from the matching, already-trained Gen4 conditioner first"
                )
            gene_basis = None
        else:
            gene_basis = load_gene_residual_basis(basis_path)

        conditioner_checkpoint_dir = fingerprints.get("gen4_conditioner_checkpoint")
        needs_conditioner = not (smoke and not staged_smoke)
        pinned_conditioner_identity = None
        if needs_conditioner:
            if not conditioner_checkpoint_dir:
                raise ValueError(
                    "a Gen4 flow config requires required_fingerprints.gen4_conditioner_checkpoint -- "
                    "train and validation-select the matching conditioner first"
                )
            if dataset_manifest is not None:
                pinned_conditioner_identity = _pin_and_verify_conditioner(
                    conditioner_checkpoint_dir,
                    expected_arm=gen4_arm,
                    dataset_manifest=dataset_manifest,
                    gene_names=gene_names,
                    cache_content_by_sample=cache_content_by_sample,
                    allow_code_drift=allow_code_drift,
                )
            else:
                # Backward-compatible low-level construction path used
                # by isolated model tests. The real trainer/evaluator/
                # overfit adapters always supply a manifest and take the
                # complete identity-verification branch above.
                pinned_conditioner_identity = checkpoint_module.resolve_checkpoint_identity(
                    conditioner_checkpoint_dir
                )
                checkpoint_module.verify_gene_names(
                    pinned_conditioner_identity.resolved_dir, gene_names,
                )
        if basis_path and pinned_conditioner_identity is not None and dataset_manifest is not None:
            basis_info = _verify_gen4_basis_provenance(
                basis_path,
                gene_basis,
                dataset_manifest=dataset_manifest,
                gene_names=gene_names,
                conditioner_identity=pinned_conditioner_identity,
                cache_content_by_sample=cache_content_by_sample,
            )
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
            gex_context_embedding_dim=gex_context_embedding_dim, slide_encoder=slide_encoder,
            gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
            gene_basis=gene_basis, gene_names=gene_names, stpath_encoder=stpath_encoder, seed=seed,
        ).to(device)

        if pinned_conditioner_identity is not None:
            conditioner_info = gen4_staged_loader.load_and_freeze_deterministic_conditioner(
                model, str(pinned_conditioner_identity.resolved_dir), gene_names,
            )
        else:
            conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)

    resolved_checkpoint_identity = None
    if checkpoint_dir is not None:
        resolved_checkpoint_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
        checkpoint_module.verify_gene_names(resolved_checkpoint_identity.resolved_dir, gene_names)
        checkpoint_module.load_trainable_state(model, resolved_checkpoint_identity.resolved_dir)

    return model, {
        "kind": kind,
        "conditioner_info": conditioner_info,
        "gene_basis_info": basis_info,
        "autoencoder_info": None,
        "checkpoint_bundle_id": resolved_checkpoint_identity.bundle_dir if resolved_checkpoint_identity else None,
        "checkpoint_manifest_sha256": resolved_checkpoint_identity.manifest_sha256 if resolved_checkpoint_identity else None,
        "trainable_weights_sha256": resolved_checkpoint_identity.weights_sha256 if resolved_checkpoint_identity else None,
    }


def build_gen5_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False, staged_smoke: bool = False,
    dataset_manifest: dict | None = None,
    cache_content_by_sample: dict[str, dict] | None = None,
    allow_code_drift: bool = False,
) -> tuple[nn.Module, dict]:
    """Gen5 equivalent of `build_gen4_model_for_inference` -- construct a
    `Gen5LatentFlowModel` (kind == "latent_flow"): load the exact shared
    `ExpressionAutoencoder` (Integration audit finding #8: via
    `gen5.autoencoder.load_expression_autoencoder_checkpoint`, the real,
    already-complete standalone-checkpoint loader -- see
    `gen4/staged_loader.py`'s own docstring for why this module does not
    stage-load+freeze the flow's conditioner from `required_fingerprints.
    gen4_conditioner_checkpoint`, then optionally load `checkpoint_dir`'s
    trainable weights on top -- the same checkpoint_dir=None/real-path
    contract every other `build_*_model_for_inference` function uses."""
    from gen3_multiscale.gen5 import model_factory as gen5_model_factory
    from gen3_multiscale.gen5.autoencoder import (
        ExpressionAutoencoder, load_expression_autoencoder_checkpoint, verify_expression_autoencoder_gene_names,
    )

    model_cfg = config.get("model") or {}
    kind = str(model_cfg.get("kind", ""))
    if kind != "latent_flow":
        raise ValueError(f"build_gen5_model_for_inference: unsupported model.kind {kind!r} (expected 'latent_flow')")
    data_cfg = config.get("data") or {}
    training_cfg = config.get("training") or {}
    params = model_cfg.get("params") or {}
    n_genes = len(gene_names)
    gex_feature_dim = int(data_cfg.get("gex_feature_dim", 128))
    image_feature_dim = int(params.get("image_feature_dim", 1536))
    seed = int(training_cfg.get("seed", 0))
    gex_context_embedding_dim = params.get("gex_context_embedding_dim")
    gex_context_embedding_dim = int(gex_context_embedding_dim) if gex_context_embedding_dim else None

    fingerprints = config.get("required_fingerprints") or {}
    needs_staged = not (smoke and not staged_smoke)
    autoencoder_checkpoint_path = fingerprints.get("expression_autoencoder_checkpoint")
    if autoencoder_checkpoint_path and needs_staged:
        from gen3_multiscale.training.train import dataset_manifest_fingerprint

        manifest_fp = dataset_manifest_fingerprint(dataset_manifest) if dataset_manifest is not None else None
        autoencoder, ae_payload = load_expression_autoencoder_checkpoint(
            str(autoencoder_checkpoint_path), dataset_manifest_fingerprint=manifest_fp,
        )
        verify_expression_autoencoder_gene_names(autoencoder, gene_names)
        autoencoder_info = {
            "loaded": True, "checkpoint_path": str(autoencoder_checkpoint_path),
            "latent_dim": ae_payload["latent_dim"], "hidden_dim": ae_payload["hidden_dim"],
            "code_identity": ae_payload.get("code_identity"),
        }
    elif needs_staged:
        raise ValueError(
            "a Gen5 latent_flow config requires required_fingerprints.expression_autoencoder_checkpoint -- "
            "a real, already-fit shared ExpressionAutoencoder checkpoint (gen5.autoencoder."
            "save_expression_autoencoder_checkpoint). Gen5 must never train from a random or unstaged "
            "autoencoder"
        )
    else:
        # Construction-only smoke with no real autoencoder configured yet.
        latent_dim = int(params.get("latent_dim") or 0)
        if not latent_dim:
            raise ValueError("model.params.latent_dim must be set (even for a construction-only smoke)")
        autoencoder = ExpressionAutoencoder(n_genes, gene_names, latent_dim, hidden_dim=params.get("autoencoder_hidden_dim", 1024))
        autoencoder_info = {"loaded": False, "checkpoint_path": None, "latent_dim": latent_dim, "hidden_dim": None, "code_identity": None}

    gen4_arm = _resolve_gen4_arm(config)
    stpath_encoder = _maybe_build_stpath_encoder(config, gen4_arm, gene_names, image_feature_dim)
    slide_encoder, gigapath_checkpoint_sha256 = _maybe_build_gigapath_slide_encoder(config, gen4_arm)

    torch.manual_seed(seed)
    model = gen5_model_factory.build_gen5_model(
        config, n_genes=n_genes, gene_names=gene_names, gex_feature_dim=gex_feature_dim,
        image_feature_dim=image_feature_dim, autoencoder=autoencoder,
        gex_context_embedding_dim=gex_context_embedding_dim, slide_encoder=slide_encoder,
        gigapath_checkpoint_sha256=gigapath_checkpoint_sha256, stpath_encoder=stpath_encoder, seed=seed,
    ).to(device)

    conditioner_checkpoint_dir = fingerprints.get("gen4_conditioner_checkpoint")
    if conditioner_checkpoint_dir and needs_staged:
        if dataset_manifest is not None:
            pinned_conditioner_identity = _pin_and_verify_conditioner(
                conditioner_checkpoint_dir,
                expected_arm=gen4_arm,
                dataset_manifest=dataset_manifest,
                gene_names=gene_names,
                cache_content_by_sample=cache_content_by_sample,
                allow_code_drift=allow_code_drift,
            )
        else:
            pinned_conditioner_identity = checkpoint_module.resolve_checkpoint_identity(
                conditioner_checkpoint_dir
            )
            checkpoint_module.verify_gene_names(
                pinned_conditioner_identity.resolved_dir, gene_names,
            )
        conditioner_info = gen4_staged_loader.load_and_freeze_deterministic_conditioner(
            model, str(pinned_conditioner_identity.resolved_dir), gene_names,
        )
    elif needs_staged:
        raise ValueError(
            "a Gen5 latent_flow config requires required_fingerprints.gen4_conditioner_checkpoint -- a "
            "real, already-trained, validation-selected matching gen4 conditioner checkpoint_dir. A Gen5 "
            "flow model must never start training from a random or unstaged conditioner"
        )
    else:
        conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)

    resolved_checkpoint_identity = None
    if checkpoint_dir is not None:
        resolved_checkpoint_identity = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
        checkpoint_module.verify_gene_names(resolved_checkpoint_identity.resolved_dir, gene_names)
        checkpoint_module.load_trainable_state(model, resolved_checkpoint_identity.resolved_dir)

    return model, {
        "kind": kind,
        "conditioner_info": conditioner_info,
        "autoencoder_info": autoencoder_info,
        "checkpoint_bundle_id": resolved_checkpoint_identity.bundle_dir if resolved_checkpoint_identity else None,
        "checkpoint_manifest_sha256": resolved_checkpoint_identity.manifest_sha256 if resolved_checkpoint_identity else None,
        "trainable_weights_sha256": resolved_checkpoint_identity.weights_sha256 if resolved_checkpoint_identity else None,
    }


def build_gen6_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False,
    staged_smoke: bool = False, dataset_manifest: dict | None = None,
    cache_content_by_sample: dict[str, dict] | None = None,
    allow_code_drift: bool = False,
) -> tuple[nn.Module, dict]:
    from gen3_multiscale.gen6.contract import get_gen6_arm_spec
    from gen3_multiscale.gen6.model_factory import build_gen6_model

    model_cfg = config.get("model") or {}
    arm = str(model_cfg.get("arm", ""))
    spec = get_gen6_arm_spec(arm)
    data_cfg = config.get("data") or {}
    params = model_cfg.get("params") or {}
    seed = int((config.get("training") or {}).get("seed", 0))
    image_dim = int(params.get("image_feature_dim", 1536))
    gex_dim = int(data_cfg.get("gex_feature_dim", 128))
    context_dim = params.get("gex_context_embedding_dim")
    context_dim = int(context_dim) if context_dim else None
    slide_encoder = None
    slide_sha = None
    if spec.uses_gigapath_dense:
        checkpoint = (config.get("required_fingerprints") or {}).get("gigapath_checkpoint")
        if checkpoint:
            from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder

            slide_encoder = FrozenGigaPathSlideEncoder(str(checkpoint))
            slide_sha = slide_encoder.checkpoint_sha256
    if spec.staged_conditioner:
        fingerprints = config.get("required_fingerprints") or {}
        conditioner_path = fingerprints.get("gen6_conditioner_checkpoint")
        if not conditioner_path and not (smoke and not staged_smoke):
            raise ValueError(f"{arm} requires required_fingerprints.gen6_conditioner_checkpoint")
        if conditioner_path:
            conditioner_identity = checkpoint_module.resolve_checkpoint_identity(conditioner_path)
            conditioner_config_path = conditioner_identity.resolved_dir / "model_config.json"
            if not conditioner_config_path.is_file():
                raise ValueError(f"{conditioner_identity.resolved_dir}: missing model_config.json")
            conditioner_config = json.loads(conditioner_config_path.read_text())
            expected_arm = str(params.get("conditioner_arm", ""))
            actual_arm = str((conditioner_config.get("model") or {}).get("arm", ""))
            if actual_arm != expected_arm:
                raise ValueError(f"selected conditioner arm {actual_arm!r} != configured {expected_arm!r}")
            if dataset_manifest is not None:
                from gen3_multiscale.training.train import verify_full_checkpoint_identity

                verify_full_checkpoint_identity(
                    conditioner_identity.resolved_dir, config=conditioner_config,
                    dataset_manifest=dataset_manifest, gene_names=gene_names,
                    cache_content_by_sample=cache_content_by_sample,
                    allow_code_drift=allow_code_drift,
                )
            conditioner, _ = build_gen6_model_for_inference(
                conditioner_config, gene_names=gene_names, device=device,
                checkpoint_dir=str(conditioner_identity.resolved_dir), smoke=smoke,
                staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
                cache_content_by_sample=cache_content_by_sample,
                allow_code_drift=allow_code_drift,
            )
            conditioner_info = {
                "loaded": True, "checkpoint_dir": str(conditioner_identity.resolved_dir),
                "checkpoint_sha256": conditioner_identity.weights_sha256,
                "checkpoint_step": conditioner_identity.step,
                "checkpoint_bundle_id": conditioner_identity.bundle_dir,
                "checkpoint_manifest_sha256": conditioner_identity.manifest_sha256,
            }
        else:
            # Package-only smoke: use the explicitly named deterministic arm.
            smoke_config = json.loads(json.dumps(config))
            smoke_config["model"]["arm"] = str(params.get("conditioner_arm", "gen6c"))
            smoke_config["model"]["kind"] = "conditioner"
            for key in (
                "conditioner_arm", "gene_basis_rank", "n_flow_blocks", "n_flow_samples",
                "n_ode_steps", "ot_epsilon", "ot_sinkhorn_iters", "latent_dim",
                "autoencoder_hidden_dim", "ot_assignment",
                "wae_hidden_dim", "discriminator_hidden_dim", "adversarial_weight",
                "discriminator_weight", "conditional_mean_weight",
                "latent_spatial_correlation",
            ):
                smoke_config["model"]["params"].pop(key, None)
            conditioner = build_gen6_model(
                smoke_config, gene_names=gene_names, gex_feature_dim=gex_dim,
                image_feature_dim=image_dim, gex_context_embedding_dim=context_dim,
                slide_encoder=slide_encoder, gigapath_checkpoint_sha256=slide_sha, seed=seed,
            ).to(device)
            conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)
        from gen3_multiscale.gen6.generative import Gen6LatentOTFlowModel, Gen6WAEGANModel

        autoencoder_info = None
        if spec.generator == "latent_ot_flow":
            from gen3_multiscale.gen5.autoencoder import (
                ExpressionAutoencoder,
                load_expression_autoencoder_checkpoint,
                verify_expression_autoencoder_gene_names,
            )

            autoencoder_path = fingerprints.get("expression_autoencoder_checkpoint")
            needs_staged = not (smoke and not staged_smoke)
            if autoencoder_path and needs_staged:
                from gen3_multiscale.training.train import dataset_manifest_fingerprint

                manifest_fp = (
                    dataset_manifest_fingerprint(dataset_manifest)
                    if dataset_manifest is not None else None
                )
                autoencoder, payload = load_expression_autoencoder_checkpoint(
                    autoencoder_path, dataset_manifest_fingerprint=manifest_fp,
                )
                verify_expression_autoencoder_gene_names(autoencoder, gene_names)
                configured_latent_dim = int(params.get("latent_dim", payload["latent_dim"]))
                if configured_latent_dim != int(payload["latent_dim"]):
                    raise ValueError(
                        f"{arm} model.params.latent_dim does not match the staged "
                        f"autoencoder ({configured_latent_dim} != {payload['latent_dim']})"
                    )
                autoencoder_info = {
                    "loaded": True, "checkpoint_path": str(autoencoder_path),
                    "latent_dim": payload["latent_dim"], "hidden_dim": payload["hidden_dim"],
                    "code_identity": payload.get("code_identity"),
                }
            elif needs_staged:
                raise ValueError(
                    f"{arm} requires required_fingerprints.expression_autoencoder_checkpoint"
                )
            else:
                latent_dim = int(params.get("latent_dim", 8))
                autoencoder = ExpressionAutoencoder(
                    len(gene_names), gene_names, latent_dim=latent_dim,
                    hidden_dim=int(params.get("autoencoder_hidden_dim", 64)),
                )
                autoencoder_info = {
                    "loaded": False, "checkpoint_path": None,
                    "latent_dim": latent_dim, "hidden_dim": None, "code_identity": None,
                }
            model = Gen6LatentOTFlowModel(
                conditioner, autoencoder, gene_names,
                hidden_dim=int(params.get("hidden_dim", 512)), n_heads=int(params.get("n_heads", 8)),
                n_flow_blocks=int(params.get("n_flow_blocks", 2)),
                dense_threshold=int(params.get("dense_threshold", 256)),
                sparse_k=int(params.get("sparse_k", 10)), chunk_size=int(params.get("chunk_size", 1024)),
                n_flow_samples=int(params.get("n_flow_samples", 8)),
                n_ode_steps=int(params.get("n_ode_steps", 20)),
                ot_epsilon=float(params.get("ot_epsilon", 0.1)),
                ot_sinkhorn_iters=int(params.get("ot_sinkhorn_iters", 20)),
                ot_assignment=str(params.get("ot_assignment", "hard")),
            ).to(device)
        else:
            model = Gen6WAEGANModel(
                conditioner, len(gene_names), latent_dim=int(params.get("latent_dim", 256)),
                hidden_dim=int(params.get("wae_hidden_dim", 1024)),
                conditioner_hidden_dim=int(params.get("hidden_dim", 512)),
                discriminator_hidden_dim=int(params.get("discriminator_hidden_dim", 256)),
                adversarial_weight=float(params.get("adversarial_weight", 0.1)),
                discriminator_weight=float(params.get("discriminator_weight", 1.0)),
                pcc_weight=float((config.get("loss") or {}).get("pcc_weight", 0.1)),
                # 0.0 reproduces the original Gen6-L exactly: the head exists
                # but receives no gradient, so its presence cannot change a
                # historical arm's result.
                conditional_mean_weight=float(params.get("conditional_mean_weight", 0.0)),
                latent_spatial_correlation=float(params.get("latent_spatial_correlation", 0.0)),
                n_samples=int(params.get("n_flow_samples", 8)),
            ).to(device)
    else:
        model = build_gen6_model(
            config, gene_names=gene_names, gex_feature_dim=gex_dim,
            image_feature_dim=image_dim, gex_context_embedding_dim=context_dim,
            slide_encoder=slide_encoder, gigapath_checkpoint_sha256=slide_sha, seed=seed,
        ).to(device)
        conditioner_info = dict(_UNLOADED_CONDITIONER_INFO)
    resolved = None
    if checkpoint_dir is not None:
        resolved = checkpoint_module.resolve_checkpoint_identity(checkpoint_dir)
        checkpoint_module.verify_gene_names(resolved.resolved_dir, gene_names)
        checkpoint_module.load_trainable_state(model, resolved.resolved_dir)
    return model, {
        "kind": str(model_cfg.get("kind", "")), "conditioner_info": conditioner_info,
        "gene_basis_info": None, "autoencoder_info": autoencoder_info if spec.staged_conditioner else None,
        "checkpoint_bundle_id": resolved.bundle_dir if resolved else None,
        "checkpoint_manifest_sha256": resolved.manifest_sha256 if resolved else None,
        "trainable_weights_sha256": resolved.weights_sha256 if resolved else None,
    }


def build_gen4_or_gen5_model_for_inference(
    config: dict, *, gene_names: list[str], device: torch.device,
    checkpoint_dir: str | Path | None = None, smoke: bool = False, staged_smoke: bool = False,
    dataset_manifest: dict | None = None,
    cache_content_by_sample: dict[str, dict] | None = None,
    allow_code_drift: bool = False,
) -> tuple[nn.Module, dict]:
    """The single dispatch point `training/train.py::build_model_for_
    inference` calls whenever `is_gen4_config(config)` is true --
    Integration-1's "normalized model_info schema" requirement: every
    branch below returns the SAME dict shape (kind/conditioner_info/
    autoencoder_info/checkpoint_*), so a caller never needs to know
    whether it got a Gen4 or Gen5 model back to read identity-binding
    fields for a run manifest."""
    from gen3_multiscale.gen6.contract import is_gen6_config

    if is_gen6_config(config):
        return build_gen6_model_for_inference(
            config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir,
            smoke=smoke, staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
            cache_content_by_sample=cache_content_by_sample,
            allow_code_drift=allow_code_drift,
        )
    kind = str((config.get("model") or {}).get("kind", ""))
    if kind == "latent_flow":
        return build_gen5_model_for_inference(
            config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir,
            smoke=smoke, staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
            cache_content_by_sample=cache_content_by_sample,
            allow_code_drift=allow_code_drift,
        )
    return build_gen4_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir,
        smoke=smoke, staged_smoke=staged_smoke, dataset_manifest=dataset_manifest,
        cache_content_by_sample=cache_content_by_sample,
        allow_code_drift=allow_code_drift,
    )
