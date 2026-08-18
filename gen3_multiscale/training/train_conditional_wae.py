#!/usr/bin/env python3
"""Real manifest/cache trainer for the supervisor conditional-WAE suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAEMaskedGEXDataset,
    ConditionalWAE,
    DeterministicSpatialPredictor,
    LocalImageConditioner,
)
from gen3_multiscale.conditional_wae.coexpression import load_conditional_wae_gene_coexpression_basis
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.histology_context import HistologyContextInjector
from gen3_multiscale.conditional_wae.histology_features import FEATURE_DIM as HISTOLOGY_FEATURE_DIM
from gen3_multiscale.conditional_wae.film_diagnostics import compute_film_diagnostics
from gen3_multiscale.conditional_wae.reference_projection import ensure_reference_gex_projection
from gen3_multiscale.conditional_wae.spatial_prior import load_spatial_prior_into
from gen3_multiscale.conditional_wae.structured_field import (
    load_centered_gene_structure_artifact,
)
from gen3_multiscale.conditional_wae.tensorboard import (
    ConditionalWAESnapshotAccumulator,
    ConditionalWAETensorBoardLogger,
)
from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide, whole_slide_metrics
from gen3_multiscale.config_identity import config_identity_fingerprint, resolved_config
from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.evaluation.gen3_evaluator import load_configured_gene_panels
from gen3_multiscale.evaluation.train_gene_panels import load_train_derived_gene_panels
from gen3_multiscale.models.losses import rmse_pcc_reconstruction_loss
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples, save_gen3_preflight_report
from gen3_multiscale.training.train import (
    _code_commit_hash,
    _worktree_diff_hash,
    dataset_manifest_fingerprint,
    deterministic_train_index_for_step,
    expected_tile_encoder_provenance,
    save_best_checkpoint_bundle,
)


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(temporary, path)


def _configure_cpu_threads(training_cfg: dict) -> int:
    """Apply the explicit per-trainer CPU cap before data/model work starts."""
    cpu_threads = int(
        os.environ.get(
            "SCILIFESTDL_CPU_THREADS",
            training_cfg.get("cpu_threads", 8),
        )
    )
    if cpu_threads < 1:
        raise ValueError("training.cpu_threads must be at least 1")
    torch.set_num_threads(cpu_threads)
    return cpu_threads


def _select_train_item_with_minimum_queries(
    dataset,
    *,
    step: int,
    seed: int,
    minimum_query_spots: int = 2,
):
    """Deterministically bypass masks too small for latent distribution losses."""
    n_items = len(dataset)
    if n_items < 1:
        raise ValueError("conditional latent training dataset is empty")
    start = deterministic_train_index_for_step(step, n_items, seed)
    for offset in range(n_items):
        index = (start + offset) % n_items
        inputs, target, identity = dataset[index]
        if np.asarray(target).shape[0] >= minimum_query_spots:
            return inputs, target, identity, offset
    raise ValueError(
        "conditional latent training requires at least two query spots, but no "
        "eligible mask exists in the dataset"
    )


def _load_frozen_deterministic_backbone(
    model: ConditionalWAE, params: dict, gene_names: list[str] | None,
) -> None:
    """Reconstruct a pinned deterministic predictor inside a residual WAE."""
    source = params.get("deterministic_backbone_checkpoint")
    if not source:
        if bool(params.get("freeze_deterministic_backbone", False)):
            raise ValueError(
                "freeze_deterministic_backbone requires "
                "deterministic_backbone_checkpoint"
            )
        return
    if gene_names is None:
        raise ValueError("deterministic backbone warm-start requires gene_names")
    identity = checkpoint_module.resolve_checkpoint_identity(source)
    expected_sha256 = str(params.get("deterministic_backbone_weights_sha256", ""))
    if not expected_sha256 or identity.weights_sha256 != expected_sha256:
        raise ValueError(
            "deterministic backbone checkpoint does not match its pinned "
            f"weights hash ({identity.weights_sha256!r} != {expected_sha256!r})"
        )
    checkpoint_module.verify_gene_names(identity.resolved_dir, gene_names)
    weights_path = identity.resolved_dir / "trainable_weights.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("deterministic backbone weights must be a state dictionary")

    transfers = (
        ("image_conditioner", model.image_conditioner),
        ("conditional_mean_head", model.conditional_mean_head),
        ("coexpression_refinement", model.centered_gene_structure_refinement),
        ("spatial_refiner", model.spatial_refiner),
    )
    consumed: set[str] = set()
    for source_prefix, destination in transfers:
        if destination is None:
            raise ValueError(
                f"warm-start destination has no module for {source_prefix!r}"
            )
        prefix = source_prefix + "."
        module_state = {
            key[len(prefix):]: value for key, value in state.items()
            if key.startswith(prefix)
        }
        expected = set(destination.state_dict())
        if set(module_state) != expected:
            raise ValueError(
                f"deterministic backbone {source_prefix!r} keys do not match "
                f"the residual-WAE destination (missing={sorted(expected - set(module_state))}, "
                f"unexpected={sorted(set(module_state) - expected)})"
            )
        destination.load_state_dict(module_state, strict=True)
        consumed.update(prefix + key for key in module_state)

    for root_name in ("composition_gate_logits", "per_gene_scale"):
        if root_name not in state:
            raise ValueError(f"deterministic backbone is missing {root_name!r}")
        destination_value = getattr(model, root_name)
        if destination_value.shape != state[root_name].shape:
            raise ValueError(f"deterministic backbone {root_name!r} shape mismatch")
        with torch.no_grad():
            destination_value.copy_(state[root_name])
        consumed.add(root_name)
    unexpected = set(state) - consumed
    if unexpected:
        raise ValueError(
            "deterministic source contains unrecognized state keys: "
            f"{sorted(unexpected)}"
        )
    if not bool(params.get("freeze_deterministic_backbone", False)):
        raise ValueError(
            "a deterministic backbone checkpoint may only be used with "
            "freeze_deterministic_backbone=true in this controlled screen"
        )
    model.freeze_deterministic_backbone()


def accumulate_and_step_discriminator(
    model: ConditionalWAE, optimizer: torch.optim.Optimizer,
    micro_batches: list[tuple[int, object, torch.Tensor]], *,
    gradient_accumulation_steps: int, clip_value: float, seed: int,
) -> tuple[float, float]:
    """One discriminator optimizer update accumulated over every micro-
    batch in `micro_batches` (each `(mask_index, inputs, target_tensor)`).

    Dividing each micro-batch's loss by `gradient_accumulation_steps`
    BEFORE `.backward()` makes the SUM of those N accumulated backward
    calls equal the MEAN gradient over the N masks, not their sum --
    `optimizer.step()` therefore applies a mean-magnitude update
    regardless of how many masks were accumulated, matching the
    fixed-`lr` semantics every other arm already assumes. Caller
    guarantees `model.discriminator is not None`. Returns
    `(mean_loss, grad_norm)`; raises RuntimeError on any non-finite
    value, exactly like the pre-accumulation single-mask code did."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if not micro_batches:
        raise ValueError("micro_batches must be non-empty")
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for mask_index, micro_inputs, target_tensor in micro_batches:
        micro_loss = model.compute_discriminator_loss(
            target_tensor, inputs=micro_inputs,
            generator=torch.Generator(device=target_tensor.device).manual_seed(seed + 2 * mask_index),
        )
        if not torch.isfinite(micro_loss):
            raise RuntimeError(f"non-finite discriminator loss at mask_index={mask_index}")
        (micro_loss / gradient_accumulation_steps).backward()
        loss_accum += float(micro_loss.detach()) / gradient_accumulation_steps
    grad_norm = torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), clip_value)
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite discriminator gradient")
    optimizer.step()
    return loss_accum, float(grad_norm)


def accumulate_and_step_generator(
    model: ConditionalWAE, optimizer: torch.optim.Optimizer, generator_parameters: list,
    micro_batches: list[tuple[int, object, torch.Tensor]], *,
    gradient_accumulation_steps: int, clip_value: float, seed: int,
) -> tuple[dict[str, float], float]:
    """The generator-side counterpart of `accumulate_and_step_discriminator`
    -- same mean-not-sum normalization, same one-optimizer-update-per-call
    contract. Returns `(accumulated_scalar_losses, grad_norm)`, where
    `accumulated_scalar_losses` is every scalar entry `compute_generator_
    losses` returns (its non-scalar `expression`/`conditional_mean_
    expression`/`latent` tensors are dropped, not loggable loss
    components), each the mean over `micro_batches`."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if not micro_batches:
        raise ValueError("micro_batches must be non-empty")
    optimizer.zero_grad(set_to_none=True)
    accumulated_losses: dict[str, float] = {}
    for mask_index, inputs, target_tensor in micro_batches:
        losses = model.compute_generator_losses(
            inputs, target_tensor,
            generator=torch.Generator(device=target_tensor.device).manual_seed(seed + 2 * mask_index + 1),
        )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"non-finite generator loss at mask_index={mask_index}")
        (losses["total"] / gradient_accumulation_steps).backward()
        for key, value in losses.items():
            if key in ("expression", "conditional_mean_expression", "latent"):
                continue
            accumulated_losses[key] = (
                accumulated_losses.get(key, 0.0) + float(value.detach()) / gradient_accumulation_steps
            )
    grad_norm = torch.nn.utils.clip_grad_norm_(generator_parameters, clip_value)
    if not torch.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError(f"invalid generator gradient norm {grad_norm}")
    optimizer.step()
    return accumulated_losses, float(grad_norm)


def _build_model(config: dict, n_genes: int, *, gene_names: list[str] | None = None) -> torch.nn.Module:
    model_cfg = config["model"]
    params = model_cfg["params"]
    conditioner_mode = str(params.get("conditioner_mode", "spatial"))
    if conditioner_mode == "local":
        conditioner = LocalImageConditioner(
            n_genes=n_genes,
            image_feature_dim=int(params["image_feature_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            image_proj_dim=int(params.get("image_proj_dim", 256)),
            modality_flag_dim=int(params.get("modality_flag_dim", 16)),
            dropout=float(params.get("dropout", 0.1)),
        )
    elif conditioner_mode == "spatial":
        conditioner = Architecture1ImageConditioner(
            n_genes=n_genes,
            image_feature_dim=int(params["image_feature_dim"]),
            gex_feature_dim=int(params["gex_feature_dim"]),
            hidden_dim=int(params["hidden_dim"]),
            n_heads=int(params["n_heads"]),
            n_blocks=int(params["n_blocks"]),
            dense_threshold=int(params["dense_threshold"]),
            sparse_k=int(params["sparse_k"]),
            dropout=float(params.get("dropout", 0.1)),
        )
    else:
        raise ValueError("model.params.conditioner_mode must be 'local' or 'spatial'")
    if bool(params.get("use_histology_context", False)):
        conditioner = HistologyContextInjector(conditioner, histology_feature_dim=HISTOLOGY_FEATURE_DIM)
    loss = config["loss"]
    gene_coexpression_basis = None
    if bool(params.get("use_gene_coexpression_refinement", False)):
        if gene_names is None:
            raise ValueError(
                "model.params.use_gene_coexpression_refinement requires _build_model's gene_names argument"
            )
        basis_path = config["data"].get("gene_coexpression_basis_path")
        if not basis_path:
            raise ValueError(
                "model.params.use_gene_coexpression_refinement requires data.gene_coexpression_basis_path"
            )
        gene_coexpression_basis, _basis_metadata = load_conditional_wae_gene_coexpression_basis(
            basis_path, gene_names,
        )
    gene_encoder_table = None
    if str(params.get("gene_encoder_source", "linear")) == "frozen_table":
        if gene_names is None:
            raise ValueError(
                "model.params.gene_encoder_source='frozen_table' requires _build_model's gene_names argument"
            )
        table_path = config["data"].get("gene_encoder_table_path")
        if not table_path:
            raise ValueError(
                "model.params.gene_encoder_source='frozen_table' requires data.gene_encoder_table_path"
            )
        table_basis, _table_metadata = load_conditional_wae_gene_coexpression_basis(
            table_path, gene_names,
        )
        gene_encoder_table = table_basis.basis
    structure_artifact = None
    structure_path = (config.get("data") or {}).get("centered_gene_structure_path")
    if structure_path:
        if gene_names is None:
            raise ValueError("centered gene structure requires the current gene_names")
        structure_artifact = load_centered_gene_structure_artifact(
            structure_path, gene_names,
        )
        configured_hash = (config.get("data") or {}).get(
            "centered_gene_structure_basis_sha256"
        )
        if configured_hash != structure_artifact.metadata.get("basis_sha256"):
            raise ValueError(
                "centered gene-structure basis hash does not match the immutable config"
            )
    use_structure = bool(params.get("use_centered_gene_structure", False))
    if use_structure and structure_artifact is None:
        raise ValueError(
            "use_centered_gene_structure requires data.centered_gene_structure_path"
        )
    if bool(params.get("deterministic_only", False)):
        return DeterministicSpatialPredictor(
            n_genes, conditioner,
            hidden_dim=int(params["autoencoder_hidden_dim"]),
            pcc_weight=float(loss["pcc_weight"]),
            n_inference_samples=int(params.get("n_inference_samples", 1)),
            gene_structure_artifact=(structure_artifact if use_structure else None),
            gene_structure_hidden_dim=int(params.get("gene_structure_hidden_dim", 64)),
            n_refinement_steps=int(params.get("n_refinement_steps", 0)),
            structured_composition=str(
                params.get("structured_composition", "within_then_between")
            ),
            refinement_k_neighbors=int(params.get("refinement_k_neighbors", 6)),
            refinement_hidden_dim=int(params.get("refinement_hidden_dim", 256)),
            refinement_gex_feature_dim=int(params.get("refinement_gex_feature_dim", 256)),
            per_gene_scale=(
                structure_artifact.per_gene_scale if structure_artifact is not None else None
            ),
            local_gradient_weight=float(loss.get("local_gradient_weight", 0.0)),
            wide_gradient_weight=float(loss.get("wide_gradient_weight", 0.0)),
            local_gradient_k=int(params.get("local_gradient_k", 6)),
            wide_gradient_k=int(params.get("wide_gradient_k", 18)),
        )
    model = ConditionalWAE(
        n_genes,
        conditioner,
        regularizer=model_cfg["regularizer"],
        latent_dim=int(params["latent_dim"]),
        autoencoder_hidden_dim=int(params["autoencoder_hidden_dim"]),
        discriminator_hidden_dim=int(params["discriminator_hidden_dim"]),
        regularizer_weight=float(loss["regularizer_weight"]),
        conditional_mean_weight=float(loss["conditional_mean_weight"]),
        pcc_weight=float(loss["pcc_weight"]),
        n_inference_samples=int(params["n_inference_samples"]),
        encoder_conditioning=str(params.get("encoder_conditioning", "none")),
        film_layers=tuple(params.get("film_layers", ("first", "second"))),
        film_shared_generator=bool(params.get("film_shared_generator", False)),
        gene_coexpression_basis=gene_coexpression_basis,
        gene_structure_artifact=(structure_artifact if use_structure else None),
        gene_structure_hidden_dim=int(params.get("gene_structure_hidden_dim", 64)),
        gene_encoder_table=gene_encoder_table,
        z_noise_std=float(params.get("z_noise_std", 0.0)),
        n_refinement_steps=int(params.get("n_refinement_steps", 0)),
        structured_composition=str(
            params.get("structured_composition", "within_then_between")
        ),
        refinement_k_neighbors=int(params.get("refinement_k_neighbors", 6)),
        refinement_hidden_dim=int(params.get("refinement_hidden_dim", 256)),
        refinement_gex_feature_dim=int(params.get("refinement_gex_feature_dim", 256)),
        likelihood=str(params.get("likelihood", "gaussian_mse")),
        distributional_weight=float(params.get("distributional_weight", 1.0)),
        distributional_hidden_dim=int(params.get("distributional_hidden_dim", 1024)),
        prior_mode=str(params.get("prior_mode", "standard")),
        latent_residual_mode=str(params.get("latent_residual_mode", "free")),
        conditional_prior_hidden_dim=int(params.get("conditional_prior_hidden_dim", 256)),
        conditional_prior_context_weight=float(
            params.get("conditional_prior_context_weight", 1.0)
        ),
        conditional_prior_anchor_weight=float(
            params.get("conditional_prior_anchor_weight", 0.1)
        ),
        per_gene_scale=(
            structure_artifact.per_gene_scale if structure_artifact is not None else None
        ),
        local_gradient_weight=float(loss.get("local_gradient_weight", 0.0)),
        wide_gradient_weight=float(loss.get("wide_gradient_weight", 0.0)),
        local_gradient_k=int(params.get("local_gradient_k", 6)),
        wide_gradient_k=int(params.get("wide_gradient_k", 18)),
    )
    spatial_prior_path = (config.get("data") or {}).get("spatial_prior_path")
    if spatial_prior_path:
        # INITIALISATION ONLY. A resume or an evaluation immediately overwrites
        # these weights with the checkpoint's, which is correct: the prior is
        # where training starts, never what it ends at.
        if model.spatial_refiner is None:
            raise ValueError(
                "data.spatial_prior_path requires model.params.n_refinement_steps > 0"
            )
        if gene_names is None:
            raise ValueError(
                "data.spatial_prior_path requires _build_model's gene_names argument so the "
                "prior's gene panel can be checked against this run's"
            )
        load_spatial_prior_into(model.spatial_refiner, spatial_prior_path, gene_names=gene_names)
    _load_frozen_deterministic_backbone(model, params, gene_names)
    return model


def _manifest(config: dict, dataset_manifest: dict, preflight_report: dict) -> dict:
    return {
        "version": 1,
        "kind": "conditional_wae_supervisor_run",
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "regularizer": config["model"]["regularizer"],
        "conditioner_mode": str(config["model"]["params"].get("conditioner_mode", "spatial")),
        "prior_mode": str(config["model"]["params"].get("prior_mode", "standard")),
        "deterministic_only": bool(config["model"]["params"].get("deterministic_only", False)),
        "image_mode": "full_visible",
        "query_gex_visible": False,
        "surrounding_gex_visible": bool(config["model"]["include_observed_gex"]),
        "config_identity_fingerprint": config_identity_fingerprint(config),
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(dataset_manifest["gene_panel"]),
        "cache_content_fingerprint": preflight_report["cache_content_fingerprint"],
        "cache_content_by_sample": preflight_report["cache_content_by_sample"],
        "code_commit_hash": _code_commit_hash(),
        "code_worktree_diff_hash": _worktree_diff_hash(),
    }


def _verify_resume(old: dict, new: dict, *, allow_code_drift: bool) -> None:
    identity_fields = (
        "kind", "arm", "task", "regularizer", "image_mode",
        "query_gex_visible", "surrounding_gex_visible",
        "config_identity_fingerprint", "dataset_manifest_fingerprint",
        "gene_panel_hash", "cache_content_fingerprint", "cache_content_by_sample",
    )
    changed = [field for field in identity_fields if old.get(field) != new.get(field)]
    if changed:
        raise ValueError(f"conditional-WAE resume refused: changed identity fields {changed}")
    code_changed = (
        old.get("code_commit_hash") is None
        or new.get("code_commit_hash") is None
        or old.get("code_commit_hash") != new.get("code_commit_hash")
        or old.get("code_worktree_diff_hash") != new.get("code_worktree_diff_hash")
    )
    if code_changed and not allow_code_drift:
        raise ValueError(
            "conditional-WAE resume refused: code state changed; pass --allow-code-drift "
            "only after reviewing the difference"
        )
    new["code_drift_acknowledged"] = bool(code_changed and allow_code_drift)


def _stable_seed(seed: int, identity: dict) -> int:
    stable = ":".join(str(identity.get(key)) for key in (
        "sample_id", "stratum", "query_fingerprint",
    ))
    digest = hashlib.sha256(f"conditional-wae:{seed}:{stable}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63)


def _checkpoint_resume_state(checkpoint_dir: Path) -> tuple[bool, dict | None]:
    """Distinguish a real bundle from preflight-only root metadata."""
    has_checkpoint = (checkpoint_dir / "latest_bundle.json").is_file()
    old_manifest = checkpoint_module.load_checkpoint_run_manifest(checkpoint_dir)
    if has_checkpoint and old_manifest is None:
        raise ValueError("conditional-WAE checkpoint has no bundle-bound run manifest")
    return has_checkpoint, old_manifest


@torch.no_grad()
def _validate(model: torch.nn.Module, dataset, *, device: torch.device, seed: int,
              snapshot: ConditionalWAESnapshotAccumulator | None = None,
              samples: dict | None = None, collect_film_diagnostics: bool = False) -> dict:
    model.eval()
    totals, rmses, pcc_losses = [], [], []
    wae_prior_totals, wae_prior_rmses, wae_prior_pcc_losses = [], [], []
    generator_totals, posterior_rmses, prior_losses = [], [], []
    has_latent_model = bool(getattr(model, "has_latent_model", True))
    collect_film = (
        collect_film_diagnostics and has_latent_model
        and model.encoder_conditioning == "film"
    )
    film_posterior_z, film_context = [], []
    film_predictive_std, film_predictive_mean, film_conditional_mean = [], [], []
    for index in range(len(dataset)):
        inputs, target, identity = dataset[index]
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        generator = torch.Generator(device=device).manual_seed(_stable_seed(seed, identity))
        prediction = model.sample_predictive_distribution(inputs, generator=generator)
        # Primary point metrics use the deterministic H&E prediction.  The
        # WAE prior mean remains a separate generative diagnostic; mixing the
        # two previously made early stopping and deployment report different
        # notions of "the model output".
        total, rmse, pcc_loss = rmse_pcc_reconstruction_loss(
            prediction["point_prediction"], target_tensor, pcc_weight=model.pcc_weight,
        )
        wae_total, wae_rmse, wae_pcc_loss = rmse_pcc_reconstruction_loss(
            prediction["wae_predictive_mean"], target_tensor, pcc_weight=model.pcc_weight,
        )
        totals.append(float(total))
        rmses.append(float(rmse))
        pcc_losses.append(float(pcc_loss))
        wae_prior_totals.append(float(wae_total))
        wae_prior_rmses.append(float(wae_rmse))
        wae_prior_pcc_losses.append(float(wae_pcc_loss))
        if has_latent_model:
            validation_losses = model.compute_generator_losses(
                inputs, target_tensor,
                generator=torch.Generator(device=device).manual_seed(
                    _stable_seed(seed, identity) + 1
                ),
            )
            generator_totals.append(float(validation_losses["total"]))
            posterior_rmses.append(float(validation_losses["reconstruction_rmse"]))
            prior_losses.append(float(validation_losses["prior_loss"]))
        posterior_z = None
        if has_latent_model and ((snapshot is not None and not snapshot.full) or collect_film):
            # Held-out target GEX is encoded only for a diagnostic Projector/
            # collapse view. It is never fed to sample_predictive_distribution
            # or model selection.
            posterior_z = model.encode_posterior(target_tensor, inputs=inputs)
        if snapshot is not None and not snapshot.full:
            if not has_latent_model:
                raise ValueError("latent TensorBoard snapshots are unavailable for deterministic models")
            if samples is None or inputs.sample_id not in samples:
                raise ValueError("TensorBoard snapshot requires the aligned validation sample")
            snapshot.add(
                inputs=inputs, target=target_tensor, identity=identity,
                prediction=prediction, posterior_z=posterior_z,
                sample=samples[inputs.sample_id],
            )
        if collect_film:
            film_posterior_z.append(posterior_z.detach())
            film_context.append(prediction["image_context"].detach())
            film_predictive_std.append(prediction["predictive_std"].detach())
            film_predictive_mean.append(prediction["predictive_mean"].detach())
            film_conditional_mean.append(prediction["conditional_mean_expression"].detach())
    model.train()
    result = {
        "total": float(np.mean(totals)),
        "rmse": float(np.mean(rmses)),
        "pcc_loss": float(np.mean(pcc_losses)),
        "conditional_mean_rmse": float(np.mean(rmses)),
        "wae_prior_total": float(np.mean(wae_prior_totals)),
        "wae_prior_rmse": float(np.mean(wae_prior_rmses)),
        "wae_prior_pcc_loss": float(np.mean(wae_prior_pcc_losses)),
        "generator_total": (
            float(np.mean(generator_totals)) if generator_totals else float(np.mean(totals))
        ),
        "posterior_rmse": (
            float(np.mean(posterior_rmses)) if posterior_rmses else float(np.mean(rmses))
        ),
        "prior_loss": float(np.mean(prior_losses)) if prior_losses else 0.0,
        "n_items": len(dataset),
    }
    if collect_film and film_posterior_z:
        result["film_diagnostics"] = compute_film_diagnostics(
            model, torch.cat(film_posterior_z, dim=0), torch.cat(film_context, dim=0),
            torch.cat(film_predictive_std, dim=0), torch.cat(film_predictive_mean, dim=0),
            torch.cat(film_conditional_mean, dim=0),
        )
    return result


def _select_whole_slide_sample_ids(
    validation_ids: list[str], max_slides: int, *, organ_by_sample: dict[str, str] | None = None,
) -> list[str]:
    """Deterministic, memory-bounded slide selection for whole-slide
    validation/spatial-map logging -- never more than `max_slides`
    regardless of how many validation slides exist.

    When `organ_by_sample` is given, selection round-robins across organs
    (alphabetically within each) so every organ gets representation before
    any organ gets a second slide -- a small `max_slides` still covers
    every organ present in validation, rather than picking whichever
    slides happen to sort first alphabetically overall."""
    if max_slides < 1:
        raise ValueError("max_slides must be positive")
    ordered = sorted(str(sample_id) for sample_id in validation_ids)
    if not organ_by_sample:
        return ordered[:max_slides]
    by_organ: dict[str, list[str]] = {}
    for sample_id in ordered:
        by_organ.setdefault(str(organ_by_sample[sample_id]), []).append(sample_id)
    selected: list[str] = []
    round_index = 0
    while len(selected) < max_slides and any(round_index < len(ids) for ids in by_organ.values()):
        for organ in sorted(by_organ):
            if len(selected) >= max_slides:
                break
            ids = by_organ[organ]
            if round_index < len(ids):
                selected.append(ids[round_index])
        round_index += 1
    return selected


def _aggregate_whole_slide_metrics(per_slide_metrics: list[dict]) -> dict:
    """Mean-over-slides {arm: {panel: {pcc, rmse, auc}}}. A slide/panel
    combination with an undefined AUC (all-zero or all-nonzero true
    expression) is excluded from that one average rather than zeroed."""
    if not per_slide_metrics:
        raise ValueError("per_slide_metrics must be non-empty")
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for arm in per_slide_metrics[0]["per_arm"]:
        aggregated[arm] = {}
        for panel in per_slide_metrics[0]["per_arm"][arm]:
            aggregated[arm][panel] = {}
            for metric_name in ("pcc", "rmse", "auc"):
                values = [
                    slide["per_arm"][arm][panel][metric_name] for slide in per_slide_metrics
                ]
                finite = [value for value in values if value is not None and np.isfinite(value)]
                aggregated[arm][panel][metric_name] = (
                    float(np.mean(finite)) if finite else float("nan")
                )
    return aggregated


def _run_whole_slide_validation(
    model: ConditionalWAE, samples: dict, sample_ids: list[str], gene_names: list[str],
    panels: dict, *, chunk_size: int, n_samples: int | None, seed: int, device: torch.device,
) -> tuple[dict, float, float, float, float, list[dict]]:
    """Returns (aggregated_metrics, whole_slide_total, whole_slide_rmse,
    whole_slide_pcc_loss, whole_slide_hvg50_pcc_loss, per_slide_predictions).
    The total/rmse/pcc_loss use the deterministic H&E point prediction and
    the SAME rmse_pcc_reconstruction_loss shape as masked validation's own
    primary metrics, pooled over every
    predicted spot across `sample_ids`, purely so they are directly
    comparable in scale -- they are diagnostic-only and never read by
    checkpoint selection beyond the separate best_whole_slide/.
    `hvg50_pcc_loss` is the SAME pooled pcc_loss computation restricted to
    just the `train_log1p_variance_top50` panel's gene columns -- a single
    summary number for "how well are we doing on the genes that actually
    vary," complementing the full-panel pcc_loss above (which a huge
    low-variance gene majority can otherwise dominate/dilute).
    `aggregated_metrics` (per-panel PCC/RMSE/AUC) is computed for
    completeness but not logged to TensorBoard by default -- too many arm/
    panel/metric combinations to browse usefully there. `per_slide_
    predictions` is the raw predict_whole_slide() output per sample, reused
    for spatial-map logging so the model is never re-run just to plot what
    was already computed."""
    per_slide_metrics = []
    per_slide_predictions = []
    pooled_predictions, pooled_targets = [], []
    for sample_id in sample_ids:
        sample = samples[sample_id]
        prediction = predict_whole_slide(
            model, sample, chunk_size=chunk_size, n_samples=n_samples, seed=seed,
        )
        per_slide_metrics.append(whole_slide_metrics(prediction, gene_names, panels))
        per_slide_predictions.append(prediction)
        pooled_predictions.append(prediction["point_prediction"])
        pooled_targets.append(
            torch.as_tensor(prediction["target"], dtype=torch.float32, device=device)
        )
    aggregated = _aggregate_whole_slide_metrics(per_slide_metrics)
    all_predictions = torch.cat(pooled_predictions, dim=0)
    all_targets = torch.cat(pooled_targets, dim=0)
    total, rmse, pcc_loss = rmse_pcc_reconstruction_loss(
        all_predictions, all_targets, pcc_weight=model.pcc_weight,
    )
    hvg50_indices = _panel_gene_indices(panels, "train_log1p_variance_top50", gene_names)
    if hvg50_indices:
        index_tensor = torch.as_tensor(hvg50_indices, dtype=torch.long, device=all_predictions.device)
        _hvg50_total, _hvg50_rmse, hvg50_pcc_loss = rmse_pcc_reconstruction_loss(
            all_predictions.index_select(1, index_tensor),
            all_targets.index_select(1, index_tensor),
            pcc_weight=model.pcc_weight,
        )
    else:
        hvg50_pcc_loss = pcc_loss
    return (
        aggregated, float(total), float(rmse), float(pcc_loss), float(hvg50_pcc_loss),
        per_slide_predictions,
    )


def _panel_gene_indices(panels: dict, panel_name: str, gene_names: list[str]) -> list[int]:
    panel = panels.get(panel_name) or []
    positions = {gene: index for index, gene in enumerate(gene_names)}
    return [positions[gene] for gene in panel if gene in positions]


def _tensorboard_gene_indices(config: dict, dataset_manifest: dict,
                              gene_names: list[str], count: int) -> list[int]:
    if count < 1:
        return []
    artifact_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
    if not artifact_path:
        return list(range(min(count, len(gene_names))))
    artifact = load_train_derived_gene_panels(artifact_path, dataset_manifest)
    panel = artifact["panels"].get("train_log1p_variance_top50") or []
    positions = {gene: index for index, gene in enumerate(gene_names)}
    return [positions[gene] for gene in panel[:count] if gene in positions]


def _slide_target_gene_indices(target, gene_names: list[str], count: int) -> list[int]:
    """Genes to PLOT for one slide, ranked by variance in that slide's own truth.

    Purely presentational: this chooses which panels appear in the
    target/prediction figures, never which genes any metric is computed over,
    so using the evaluated slide's ground truth here is a display decision and
    not a leak into a reported number.

    It replaces train-panel-derived selection because that repeatedly plotted
    genes with no signal on the slide being shown. Measured case: LCN2 entered
    the pooled train panel and is non-zero in 0.75% of INT14's 4,552 spots, so
    its "target" map was a flat field and the figure could not show whether the
    model was right or wrong about anything. Ranking on the slide itself
    guarantees the plotted genes actually vary there, which is the only way the
    comparison is readable.
    """
    if count < 1:
        return []
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(gene_names):
        return list(range(min(count, len(gene_names))))
    variance = values.var(axis=0)
    order = sorted(
        range(len(gene_names)),
        key=lambda idx: (-float(variance[idx]), gene_names[idx]),
    )
    return [idx for idx in order[:count] if variance[idx] > 0.0]


def _organ_gene_indices(
    artifact: dict | None, organ: str, gene_names: list[str], count: int,
) -> list[int]:
    """Per-slide gene selection for whole-slide spatial maps: uses the
    smallest organ-specific dispersion-ranked panel for `organ` when the
    loaded panel artifact has one (schema version 2+), falling back to
    the pooled cross-organ panel for older artifacts.

    A gene like ALB (a liver marker) can rank in the top-variance PANEL
    purely because of liver samples elsewhere in a multi-organ training
    cohort, while being near-zero background noise on every other
    organ's slides -- organ-specific ranking avoids plotting genes with
    no real signal on the slide actually being shown."""
    if count < 1:
        return []
    if artifact is None:
        return list(range(min(count, len(gene_names))))
    organ_panels = (artifact.get("panels_by_organ") or {}).get(organ) or {}
    panel = None
    # Use the LARGEST available organ panel so slicing to `count` never
    # truncates to fewer genes than requested just because the smallest
    # named panel (e.g. top1) happened to be picked.
    for _name, genes in sorted(organ_panels.items(), key=lambda item: -len(item[1])):
        panel = genes
        break
    if panel is None:
        panel = artifact["panels"].get("train_log1p_variance_top50") or []
    positions = {gene: index for index, gene in enumerate(gene_names)}
    return [positions[gene] for gene in panel[:count] if gene in positions]


def run_conditional_wae_training(
    config_path: str, *, smoke: bool = False, allow_code_drift: bool = False,
) -> dict:
    config = resolved_config(config_path)
    static_audit_conditional_wae_config(config)
    data_cfg, training_cfg = config["data"], config["training"]
    cpu_threads = _configure_cpu_threads(training_cfg)
    print(f"CPU thread cap: {cpu_threads}", flush=True)
    dataset_manifest = load_dataset_manifest(data_cfg["gen3_manifest_path"])
    train_ids = list(dataset_manifest["train_sample_ids"])
    validation_ids = list(dataset_manifest["validation_sample_ids"])
    if smoke:
        train_ids = train_ids[:1]
        validation_ids = validation_ids[:1]
    cfg_om = OmegaConf.create(config)
    samples, preflight_report = load_and_preflight_samples(
        cfg_om,
        dataset_manifest,
        train_ids + validation_ids,
        expected_tile_encoder_provenance(config),
    )
    strata = config["masking"]["strata"]
    n_smoke_masks = max(1, len(strata))
    validation_masks_per_stratum_per_sample = (
        n_smoke_masks if smoke else int(data_cfg.get("n_validation_masks", 4))
    )
    train_samples = {sample_id: samples[sample_id] for sample_id in train_ids}
    validation_samples = {sample_id: samples[sample_id] for sample_id in validation_ids}
    train_schedule = build_gen3_mask_schedule(
        dataset_manifest,
        train_samples,
        strata,
        "train",
        n_training_masks_per_sample=(
            n_smoke_masks if smoke else int(data_cfg.get("n_training_masks_per_sample", 500))
        ),
    )
    validation_schedule = build_gen3_mask_schedule(
        dataset_manifest,
        validation_samples,
        strata,
        "validation",
        split_counts={
            "validation": validation_masks_per_stratum_per_sample,
        },
        split_seeds={"validation": 700_000},
    ) if validation_samples else None
    base_train = Gen3SpatialFieldDataset(
        dataset_manifest, train_samples, train_schedule, strata, novae_enabled=False,
    )
    include_observed_gex = bool(config["model"]["include_observed_gex"])
    train_dataset = ConditionalWAEMaskedGEXDataset(
        base_train, include_observed_gex=include_observed_gex,
    )
    validation_dataset = None
    if validation_schedule is not None:
        base_validation = Gen3SpatialFieldDataset(
            dataset_manifest, validation_samples, validation_schedule, strata, novae_enabled=False,
        )
        boundary_report = base_validation.validate_boundary_schedule()
        print(
            "training-time validation boundary preflight: PASS "
            f"({boundary_report['n_items_checked']} fixed items = "
            f"{len(validation_ids)} samples x {len(strata)} strata x "
            f"{validation_masks_per_stratum_per_sample} masks per stratum per sample)",
            flush=True,
        )
        validation_dataset = ConditionalWAEMaskedGEXDataset(
            base_validation, include_observed_gex=include_observed_gex,
        )

    seed = int(training_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loop_rng = random.Random(seed)
    device = torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    gene_names = list(dataset_manifest["gene_panel"])
    model = _build_model(config, len(gene_names), gene_names=gene_names).to(device)
    generator_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("discriminator.") and parameter.requires_grad
    ]
    if not generator_parameters:
        raise ValueError("conditional-WAE generator has no trainable parameters")
    parameter_groups = [{"params": generator_parameters, "name": "generator"}]
    if model.discriminator is not None:
        parameter_groups.append({"params": list(model.discriminator.parameters()), "name": "discriminator"})
    optimizer_cfg = training_cfg.get("optimizer") or {}
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(value) for value in optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    checkpoint_dir = Path(training_cfg["checkpoint_dir"])
    run_manifest = _manifest(config, dataset_manifest, preflight_report)
    resume_step = 0
    if not smoke:
        has_checkpoint, old_manifest = _checkpoint_resume_state(checkpoint_dir)
        if old_manifest is not None:
            _verify_resume(old_manifest, run_manifest, allow_code_drift=allow_code_drift)
        if has_checkpoint:
            checkpoint_module.verify_gene_names(checkpoint_dir, gene_names)
            checkpoint_module.load_trainable_state(model, checkpoint_dir)
            if not checkpoint_module.load_optimizer_and_rng_state(
                optimizer, checkpoint_dir, rng=loop_rng,
            ):
                raise ValueError("conditional-WAE checkpoint lacks optimizer/RNG state")
            resume_step = int(checkpoint_module.load_training_state(checkpoint_dir)["step"])
            print(f"resumed conditional WAE from step {resume_step}", flush=True)
        save_gen3_preflight_report(preflight_report, checkpoint_dir / "preflight_report.json")
        _atomic_json(run_manifest, checkpoint_dir / "run_manifest.json")

    total_steps = int(training_cfg.get("total_steps", 1))
    step_target = resume_step + 1 if smoke else total_steps
    max_hours = float(training_cfg.get("max_wall_clock_hours", 8.0))
    log_every = max(1, int(training_cfg.get("log_every_n_steps", 50)))
    eval_every = max(1, int(training_cfg.get("eval_every_n_steps", 2000)))
    checkpoint_every = max(1, int(training_cfg.get("checkpoint_every_n_steps", 2000)))
    keep_last = int(training_cfg.get("checkpoint_keep_last", 2))
    clip_value = float(training_cfg.get("gradient_clip_val", 1.0))
    # Adam's four-arm WAE-GAN ablation: `gradient_accumulation_steps`
    # (default 1, matching every existing config's real behavior exactly)
    # accumulates N VALID masks' gradients -- normalized to a MEAN, not a
    # sum, by dividing each micro-batch loss by N before `.backward()`,
    # so the N accumulated `.backward()` calls sum to the mean gradient
    # -- before one `optimizer.step()` each for the discriminator and the
    # generator. `masks_seen` is the fine-grained, resumable counter of
    # valid masks actually consumed (never counting masks
    # `_select_train_item_with_minimum_queries` itself skipped for being
    # undersized); `step` remains the optimizer-update counter exactly as
    # before, unaffected by accumulation, since every existing piece of
    # resume/wall-clock/checkpoint-cadence logic is already expressed in
    # terms of it. With `gradient_accumulation_steps=1`, `masks_seen`
    # tracks `step` exactly and the per-mask RNG seeds below reduce to
    # the pre-accumulation formula bit-for-bit -- accumulation is
    # strictly additive, not a behavior change for existing configs.
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be a positive integer")
    history_path = checkpoint_dir / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() and not smoke else []
    point_history = [
        entry for entry in history
        if entry.get("primary_prediction_role") == "conditional_mean"
    ]
    best_monitor = str(training_cfg.get("best_monitor", "total"))
    valid_best_monitors = {"total", "generator_total", "wae_prior_total"}
    if best_monitor not in valid_best_monitors:
        raise ValueError(
            f"training.best_monitor must be one of {sorted(valid_best_monitors)}"
        )
    monitored_history = [entry for entry in point_history if best_monitor in entry]
    best_total = min(
        (entry[best_monitor] for entry in monitored_history), default=float("inf")
    )
    best_step = next(
        (entry["step"] for entry in monitored_history
         if entry[best_monitor] == best_total), None,
    ) if monitored_history else None
    if history and not point_history:
        print(
            "legacy validation history uses WAE-prior primary scores; the corrected "
            "deterministic point-prediction best score starts fresh",
            flush=True,
        )
    if not smoke and resume_step > 0:
        prior_training_state = checkpoint_module.load_training_state(checkpoint_dir)
        # Old checkpoints (pre-accumulation) never recorded masks_seen;
        # they are exactly `resume_step` masks in, one mask per step.
        masks_seen = int(prior_training_state.get("masks_seen", resume_step))
    else:
        masks_seen = 0
    tensorboard_cfg = dict((config.get("evaluation") or {}).get("tensorboard") or {})
    tensorboard_logger = None
    if bool(tensorboard_cfg.get("enabled", False)) and not smoke:
        tensorboard_logger = ConditionalWAETensorBoardLogger(
            tensorboard_cfg["log_dir"],
            max_spatial_samples=int(tensorboard_cfg.get("max_spatial_samples", 4)),
            purge_step=(resume_step if resume_step > 0 else None),
        )
        print(f"TensorBoard logging: {tensorboard_cfg['log_dir']}", flush=True)

    # GPT-relayed diagnostics request: whole-slide (every-spot) validation,
    # diagnostic-only. best_masked (the existing best/) stays the ONLY
    # checkpoint the evaluator/cross-arm comparison ever reads; best_whole_slide
    # is a separate, secondary bundle nothing downstream consumes automatically.
    whole_slide_cfg = dict((config.get("evaluation") or {}).get("whole_slide_validation") or {})
    whole_slide_enabled = bool(whole_slide_cfg.get("enabled", False))
    if whole_slide_enabled and config["model"]["task"] != "he_to_st":
        raise ValueError(
            "evaluation.whole_slide_validation is only meaningful for task='he_to_st' "
            "(it hardcodes include_observed_gex=False and predicts every spot as query)"
        )
    whole_slide_every_n_evals = max(1, int(whole_slide_cfg.get("every_n_evals", 5)))
    whole_slide_max_slides = max(1, int(whole_slide_cfg.get("max_slides", 2)))
    whole_slide_chunk_size = max(1, int(whole_slide_cfg.get("chunk_size", 2048)))
    whole_slide_n_samples = whole_slide_cfg.get("n_samples")
    whole_slide_panels = load_configured_gene_panels(config, dataset_manifest) if whole_slide_enabled else {}
    organ_by_validation_sample = {
        sample_id: str(dataset_manifest["samples"][sample_id]["organ"]) for sample_id in validation_ids
    }
    whole_slide_sample_ids = (
        _select_whole_slide_sample_ids(
            validation_ids, whole_slide_max_slides, organ_by_sample=organ_by_validation_sample,
        )
        if whole_slide_enabled else []
    )
    best_whole_slide_total = float("inf")
    best_whole_slide_step = None
    if whole_slide_enabled and not smoke:
        prior_whole_slide_state = checkpoint_module.load_training_state(checkpoint_dir / "best_whole_slide")
        if (
            "whole_slide_total" in prior_whole_slide_state
            and prior_whole_slide_state.get("primary_prediction_role") == "conditional_mean"
        ):
            best_whole_slide_total = float(prior_whole_slide_state["whole_slide_total"])
            best_whole_slide_step = int(prior_whole_slide_state["step"])
        elif "whole_slide_total" in prior_whole_slide_state:
            print(
                "ignoring legacy best_whole_slide score: it was measured on the WAE prior mean, "
                "not the deterministic point prediction",
                flush=True,
            )
    whole_slide_reference_projection = None
    if whole_slide_enabled and tensorboard_logger is not None:
        reference_projection_path = whole_slide_cfg.get("reference_projection_path")
        if not reference_projection_path:
            raise ValueError(
                "evaluation.whole_slide_validation.reference_projection_path is required "
                "when whole-slide validation and TensorBoard are both enabled -- one shared "
                "path lets every arm/dimensionality in a suite reuse the SAME frozen basis"
            )
        whole_slide_reference_projection = ensure_reference_gex_projection(
            reference_projection_path, validation_samples, whole_slide_sample_ids, gene_names,
            n_components=int(whole_slide_cfg.get("reference_pca_components", 3)),
            n_clusters=int(whole_slide_cfg.get("reference_n_clusters", 6)),
            seed=int(whole_slide_cfg.get("reference_seed", seed)),
            max_points=int(whole_slide_cfg.get("reference_max_points", 20_000)),
        )
    whole_slide_gene_panel_artifact = None
    if whole_slide_reference_projection is not None:
        artifact_path = (config.get("evaluation") or {}).get("train_gene_panel_artifact")
        if artifact_path:
            whole_slide_gene_panel_artifact = load_train_derived_gene_panels(artifact_path, dataset_manifest)

    started = time.time()
    step = resume_step
    completion_reason = "total_steps_reached"
    model.train()
    while step < step_target:
        if (time.time() - started) / 3600 >= max_hours:
            completion_reason = "wall_clock_limit_reached"
            break
        # One mask per micro-batch, reused for BOTH its discriminator and
        # generator loss contribution -- exactly mirroring the pre-
        # accumulation contract (one step, one mask, both losses), just
        # repeated `gradient_accumulation_steps` times before either
        # optimizer.step() fires. `masks_seen` is captured PER micro-batch
        # (not reconstructed after the fact) so the RNG seed below is
        # exact and collision-free across every mask this run ever draws.
        skipped_small_masks_total = 0
        micro_batches: list[tuple[int, object, torch.Tensor]] = []
        for _ in range(gradient_accumulation_steps):
            m_inputs, m_target, _identity, skipped = _select_train_item_with_minimum_queries(
                train_dataset, step=masks_seen, seed=seed,
                minimum_query_spots=(2 if getattr(model, "has_latent_model", True) else 1),
            )
            skipped_small_masks_total += skipped
            m_target_tensor = torch.as_tensor(m_target, dtype=torch.float32, device=device)
            micro_batches.append((masks_seen, m_inputs, m_target_tensor))
            masks_seen += 1

        discriminator_loss_value = None
        discriminator_grad_norm = None
        if model.discriminator is not None:
            try:
                discriminator_loss_value, discriminator_grad_norm = accumulate_and_step_discriminator(
                    model, optimizer, micro_batches,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    clip_value=clip_value, seed=seed,
                )
            except RuntimeError as exc:
                raise RuntimeError(f"[step {step}] {exc}") from exc

        try:
            accumulated_losses, generator_grad_norm = accumulate_and_step_generator(
                model, optimizer, generator_parameters, micro_batches,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_value=clip_value, seed=seed,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"[step {step}] {exc}") from exc
        step += 1
        if skipped_small_masks_total and (step - 1 == resume_step or step % log_every == 0):
            print(
                f"[step {step}] skipped {skipped_small_masks_total} undersized mask(s) across "
                f"{gradient_accumulation_steps} accumulated mask(s)",
                flush=True,
            )
        if smoke or step % log_every == 0:
            pieces = [
                f"total={accumulated_losses['total']:.6f}",
                f"reconstruction={accumulated_losses['reconstruction_loss']:.6f}",
                f"conditional_mean={accumulated_losses['conditional_mean_loss']:.6f}",
                f"prior={accumulated_losses['prior_loss']:.6f}",
                f"grad_norm={float(generator_grad_norm):.4f}",
                f"masks_seen={masks_seen}",
            ]
            if discriminator_loss_value is not None:
                pieces.append(f"discriminator={discriminator_loss_value:.6f}")
            for key, label in (
                ("local_gradient_loss", "gradient_local"),
                ("wide_gradient_loss", "gradient_wide"),
            ):
                if key in accumulated_losses:
                    pieces.append(f"{label}={accumulated_losses[key]:.6f}")
            print(f"[step {step}] train: " + ", ".join(pieces), flush=True)
            if tensorboard_logger is not None and bool(tensorboard_cfg.get("log_train_scalars", True)):
                tensorboard_logger.add_train_scalars(
                    step, accumulated_losses, grad_norm=generator_grad_norm,
                    discriminator_loss=discriminator_loss_value,
                    discriminator_grad_norm=discriminator_grad_norm,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    masks_seen=masks_seen,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
        if validation_dataset is not None and (smoke or step % eval_every == 0):
            snapshot = None
            if tensorboard_logger is not None and getattr(model, "has_latent_model", True):
                snapshot_every = max(1, int(tensorboard_cfg.get("snapshot_every_n_evals", 5)))
                evaluation_number = max(1, step // eval_every)
                if (evaluation_number - 1) % snapshot_every == 0:
                    logged_gene_indices = _tensorboard_gene_indices(
                        config, dataset_manifest, gene_names,
                        int(tensorboard_cfg.get("spatial_gene_count", 2)),
                    )
                    snapshot = ConditionalWAESnapshotAccumulator(
                        sample_records={
                            sample_id: dataset_manifest["samples"][sample_id]
                            for sample_id in validation_ids
                        },
                        gene_names=gene_names,
                        logged_gene_indices=logged_gene_indices,
                        max_points=int(tensorboard_cfg.get("embedding_max_points", 5000)),
                        max_points_per_item=int(
                            tensorboard_cfg.get("embedding_max_points_per_item", 64)
                        ),
                        thumbnail_max_points=int(
                            tensorboard_cfg.get("thumbnail_max_points", 512)
                        ),
                        thumbnail_size=int(tensorboard_cfg.get("thumbnail_size", 48)),
                    )
            validation_result = _validate(
                model, validation_dataset, device=device, seed=seed,
                snapshot=snapshot, samples=validation_samples,
                collect_film_diagnostics=(tensorboard_logger is not None),
            )
            # film_diagnostics contains numpy arrays (not JSON-serializable)
            # and is diagnostic-only -- logged to TensorBoard below, never
            # persisted into validation_history.json/checkpoint resume state.
            film_diagnostics = validation_result.pop("film_diagnostics", None)
            entry = {
                "step": step,
                "masks_seen": masks_seen,
                "primary_prediction_role": "conditional_mean",
                **validation_result,
            }
            print(
                f"[step {step}] validation: total={entry['total']:.6f}, "
                f"rmse={entry['rmse']:.6f}, "
                f"wae_prior_rmse={entry['wae_prior_rmse']:.6f}, "
                f"masks_seen={masks_seen}",
                flush=True,
            )
            if tensorboard_logger is not None:
                tensorboard_logger.add_validation_scalars(
                    step, entry, best_total=min(best_total, entry[best_monitor]),
                    best_step=(step if entry[best_monitor] <= best_total else best_step),
                    best_metric_name=best_monitor,
                )
                if snapshot is not None:
                    tensorboard_logger.add_snapshot(step, snapshot)
                if film_diagnostics is not None:
                    tensorboard_logger.add_film_diagnostics(step, film_diagnostics)
            if not smoke:
                history.append(entry)
                _atomic_json(history, history_path)
                monitor_value = float(entry[best_monitor])
                if monitor_value < best_total:
                    best_total = monitor_value
                    best_step = step
                    save_best_checkpoint_bundle(
                        model, config, gene_names, checkpoint_dir / "best",
                        step=step, val_loss=monitor_value, run_manifest=run_manifest,
                        extra_metadata={
                            "masks_seen": masks_seen,
                            "primary_prediction_role": "conditional_mean",
                            "best_monitor": best_monitor,
                        },
                    )
            if whole_slide_enabled:
                evaluation_number = max(1, step // eval_every)
                if smoke or (evaluation_number - 1) % whole_slide_every_n_evals == 0:
                    (
                        _aggregated_whole_slide, whole_slide_total, whole_slide_rmse, whole_slide_pcc_loss,
                        whole_slide_hvg50_pcc_loss, whole_slide_predictions,
                    ) = (
                        _run_whole_slide_validation(
                            model, validation_samples, whole_slide_sample_ids, gene_names,
                            whole_slide_panels, chunk_size=whole_slide_chunk_size,
                            n_samples=whole_slide_n_samples, seed=seed, device=device,
                        )
                    )
                    print(
                        f"[step {step}] whole-slide validation: total={whole_slide_total:.6f} "
                        f"over {len(whole_slide_sample_ids)} slide(s)",
                        flush=True,
                    )
                    if tensorboard_logger is not None:
                        tensorboard_logger.add_whole_slide_scalars(
                            step, whole_slide_total=whole_slide_total, whole_slide_rmse=whole_slide_rmse,
                            whole_slide_pcc_loss=whole_slide_pcc_loss,
                            whole_slide_hvg50_pcc_loss=whole_slide_hvg50_pcc_loss,
                            best_whole_slide_total=min(best_whole_slide_total, whole_slide_total),
                            best_whole_slide_step=(
                                step if whole_slide_total <= best_whole_slide_total else best_whole_slide_step
                            ),
                        )
                        if whole_slide_reference_projection is not None:
                            for prediction in whole_slide_predictions:
                                sample_organ = organ_by_validation_sample[prediction["sample_id"]]
                                tensorboard_logger.add_whole_slide_spatial_maps(
                                    step, prediction["sample_id"], prediction["coords"],
                                    prediction["target"],
                                    prediction["point_prediction"].detach().cpu().numpy(),
                                    gene_names, whole_slide_reference_projection,
                                    gene_indices=_slide_target_gene_indices(
                                        prediction["target"], gene_names,
                                        int(tensorboard_cfg.get("spatial_gene_count", 2)),
                                    ),
                                )
                    if not smoke and whole_slide_total < best_whole_slide_total:
                        best_whole_slide_total = whole_slide_total
                        best_whole_slide_step = step
                        save_best_checkpoint_bundle(
                            model, config, gene_names, checkpoint_dir / "best_whole_slide",
                            step=step, val_loss=whole_slide_total, run_manifest=run_manifest,
                            extra_metadata={
                                "masks_seen": masks_seen, "whole_slide_total": whole_slide_total,
                                "primary_prediction_role": "conditional_mean",
                                "kind": "whole_slide_diagnostic_only",
                            },
                        )
        if not smoke and step % checkpoint_every == 0 and step < step_target:
            checkpoint_module.save_checkpoint(
                model, config, gene_names, checkpoint_dir, step,
                extra_metadata={"completion_reason": "in_progress", "masks_seen": masks_seen},
                keep_last=keep_last, optimizer=optimizer, rng=loop_rng,
                run_manifest=run_manifest,
            )
    if not smoke and step > resume_step:
        checkpoint_module.save_checkpoint(
            model, config, gene_names, checkpoint_dir, step,
            extra_metadata={"completion_reason": completion_reason, "masks_seen": masks_seen},
            keep_last=keep_last, optimizer=optimizer, rng=loop_rng,
            run_manifest=run_manifest,
        )
    summary = {
        "ok": True,
        "smoke": smoke,
        "arm": config["model"]["arm"],
        "task": config["model"]["task"],
        "regularizer": config["model"]["regularizer"],
        "final_step": step,
        "masks_seen": masks_seen,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "best_step": best_step,
        "best_total": (best_total if best_total != float("inf") else None),
        "elapsed_seconds": time.time() - started,
        "completion_reason": completion_reason,
        "checkpoint_dir": str(checkpoint_dir),
    }
    if tensorboard_logger is not None:
        tensorboard_logger.close()
    print(f"conditional WAE training finished: {summary}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-code-drift", action="store_true")
    args = parser.parse_args()
    run_conditional_wae_training(
        args.config, smoke=args.smoke, allow_code_drift=args.allow_code_drift,
    )


if __name__ == "__main__":
    main()
