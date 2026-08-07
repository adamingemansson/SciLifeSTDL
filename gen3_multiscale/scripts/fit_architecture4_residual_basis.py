#!/usr/bin/env python3
"""Adam's Step 6 audit #11 deliverable (of commit 27e1232): the real
Architecture 4 residual-basis fitting pipeline. "There is no real
pipeline for fitting its residual basis" before this script existed --
`required_fingerprints.gene_residual_basis` (train.py::maybe_load_gene_basis)
had to point at SOMETHING, but nothing in this codebase ever produced a
real one.

Adam's own recommended sequence, implemented literally:
    1. Train Architecture 3 (elsewhere -- `train.py --config
       architecture3.yaml`, not this script's job).
    2. Load its selected, already-trained checkpoint.
    3. Generate residuals on TRAINING samples only.
    4. Fit and persist the basis, with dataset/gene/checkpoint/mask
       provenance recorded alongside it.
    5. (Not this script -- see `train.py::
       maybe_load_pretrained_conditioner_for_architecture4`) initialize
       Architecture 4 from that EXACT Architecture 3 conditioner and
       freeze it initially while training the flow apparatus.

"Architecture 4 therefore should not yet run concurrently from random
initialization with Architectures 1-3" -- this script's whole existence
is the fix: Architecture 4 cannot be trained meaningfully until this has
been run once against a real, trained Architecture 3 checkpoint.

    python -m gen3_multiscale.scripts.fit_architecture4_residual_basis \\
        --config gen3_multiscale/configs/architecture3.yaml \\
        --architecture3-checkpoint-dir /path/to/trained/architecture3 \\
        --output-basis-path /path/to/gene_residual_basis.pt \\
        --n-masks-per-sample 20 --rank 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import gene_panel_hash, load_dataset_manifest
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis, save_gene_residual_basis
from gen3_multiscale.training import checkpoint as checkpoint_module
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    build_model_for_inference, config_fingerprint, config_identity_fingerprint, dataset_manifest_fingerprint,
    expected_tile_encoder_provenance, resolved_config, verify_full_checkpoint_identity,
)


def compute_training_residuals(
    architecture3_model: torch.nn.Module, train_dataset, device: torch.device, *, memmap_path: str | Path,
) -> np.memmap:
    """Real residuals -- `target_expression - Architecture3's own
    deterministic conditioner mean` -- over EVERY item currently in
    `train_dataset`. `train_dataset` is a real `Gen3SpatialFieldDataset`
    built with `role="train"`, so this never touches validation/test data
    by construction (Adam's "generate residuals on training samples
    only").

    Codex re-audit of commit 90f853e, launch blocker #11: "Do not
    accumulate all full-gene residuals in RAM for basis fitting; use a
    bounded-memory method or disk-backed matrix." The prior version
    appended every item's residual array to a Python list, then
    `np.concatenate`d the whole thing -- for a real gene panel (tens of
    thousands of genes) and hundreds/thousands of training masks, that
    holds TWO full copies in process RAM simultaneously (the list of
    chunks, plus the freshly concatenated array) at its peak. This
    writes each item's residual directly into a pre-sized
    `numpy.memmap`-backed file at `memmap_path` instead -- a real,
    disk-backed matrix the OS pages in/out as needed, never fully
    resident in process RAM at once. Two passes over `train_dataset` are
    required (the per-item row count is not known ahead of time, since
    hole size varies by stratum/draw): the FIRST is target-shape-only
    (`targets.query_expression.shape`, no model forward pass, cheap);
    the SECOND is the real one, running `architecture3_model` exactly
    once per item exactly as before. The returned `np.memmap` is a real
    `np.ndarray` subclass -- `fit_gene_residual_basis`'s
    `sklearn.utils.extmath.randomized_svd` call accepts it directly, no
    special-casing needed at the call site."""
    if len(train_dataset) == 0:
        raise ValueError("compute_training_residuals: train_dataset produced zero items")

    row_counts = [np.asarray(train_dataset[idx][1].query_expression).shape[0] for idx in range(len(train_dataset))]
    total_rows = int(sum(row_counts))
    if total_rows == 0:
        raise ValueError("compute_training_residuals: train_dataset produced zero residual rows")
    n_genes = np.asarray(train_dataset[0][1].query_expression).shape[1]

    memmap_path = Path(memmap_path)
    memmap_path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=(total_rows, n_genes),
    )
    architecture3_model.eval()
    with torch.no_grad():
        offset = 0
        for idx in range(len(train_dataset)):
            inputs, targets = train_dataset[idx]
            target_expression = torch.as_tensor(targets.query_expression, dtype=torch.float32, device=device)
            out = architecture3_model(inputs)
            residual = target_expression - torch.as_tensor(out["expression"], dtype=torch.float32, device=device)
            n_rows = residual.shape[0]
            residuals[offset:offset + n_rows] = residual.detach().cpu().numpy().astype(np.float32)
            offset += n_rows
    residuals.flush()
    if not np.all(np.isfinite(residuals)):
        raise ValueError("compute_training_residuals: computed residuals contain non-finite values")
    return residuals


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _residual_sidecar_path(residual_path: str | Path) -> Path:
    return Path(f"{residual_path}.provenance.json")


def _json_normalized(value):
    """Return the exact JSON representation persisted in sidecars."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_verified_residual_cache(
    residuals: np.memmap, residual_path: str | Path, expected_identity: dict,
) -> Path:
    """Write the identity sidecar last, after a complete residual matrix.

    A residual file without this sidecar is deliberately not reusable: it
    may be a partial file left by an interrupted inference pass.  Binding
    the file hash and all scientific inputs lets a later SVD retry skip
    Architecture 3 inference without silently accepting stale residuals.
    """
    residuals.flush()
    residual_path = Path(residual_path)
    sidecar_path = _residual_sidecar_path(residual_path)
    payload = {
        "version": 1,
        "kind": "gen3_architecture4_training_residual_cache",
        "identity": expected_identity,
        "shape": [int(value) for value in residuals.shape],
        "dtype": residuals.dtype.str,
        "residual_file_sha256": _file_sha256(residual_path),
    }
    tmp = sidecar_path.with_name(f"{sidecar_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, sidecar_path)
    return sidecar_path


def _load_verified_residual_cache(
    residual_path: str | Path, expected_identity: dict,
) -> np.memmap:
    residual_path = Path(residual_path)
    sidecar_path = _residual_sidecar_path(residual_path)
    if not residual_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            f"reusable residuals require both {residual_path} and {sidecar_path}; "
            "a residual file without its completed provenance sidecar may be partial and is refused"
        )
    payload = json.loads(sidecar_path.read_text())
    if payload.get("version") != 1 or payload.get("kind") != "gen3_architecture4_training_residual_cache":
        raise ValueError(f"{sidecar_path}: unsupported residual-cache schema")
    if payload.get("identity") != expected_identity:
        raise ValueError(
            f"{sidecar_path}: residual-cache scientific identity does not match this basis-fitting run"
        )
    actual_sha256 = _file_sha256(residual_path)
    if actual_sha256 != payload.get("residual_file_sha256"):
        raise ValueError(f"{residual_path}: residual-cache content SHA256 does not match its sidecar")
    residuals = np.load(residual_path, mmap_mode="r")
    if list(residuals.shape) != payload.get("shape") or residuals.dtype.str != payload.get("dtype"):
        raise ValueError(f"{residual_path}: residual-cache shape/dtype does not match its sidecar")
    if residuals.ndim != 2 or residuals.shape[0] == 0 or residuals.shape[1] != expected_identity["n_genes"]:
        raise ValueError(f"{residual_path}: residual-cache matrix has invalid shape {residuals.shape}")
    if not np.all(np.isfinite(residuals)):
        raise ValueError(f"{residual_path}: residual-cache matrix contains non-finite values")
    return residuals


def fit_and_save_architecture4_basis(
    architecture3_config_path: str, architecture3_checkpoint_dir: str, output_basis_path: str,
    *, n_masks_per_sample: int = 20, rank: int = 64, device_str: str = "cpu", allow_code_drift: bool = False,
    reuse_residuals_path: str | None = None, svd_device: str = "cpu", svd_n_iter: int = 4,
    svd_oversamples: int = 16,
) -> dict:
    """The full pipeline. Returns (and persists alongside the basis file,
    as `<output_basis_path>.provenance.json`) a provenance record binding
    the fitted basis to the exact config, dataset manifest, gene panel,
    Architecture 3 checkpoint, and mask schedule it was produced from --
    Adam's "fit and persist the basis with dataset/gene/checkpoint/mask
    provenance".

    Codex re-audit of commit f7bb8a1, launch blocker #5: "Before residual
    fitting, verify Architecture 3 through the same complete best/latest
    identity pipeline used by evaluation." Before this round,
    `architecture3_checkpoint_dir` was only ever checked for EXACT-WEIGHTS
    integrity (`resolve_checkpoint_identity`, via `build_model_for_inference`)
    -- a checkpoint whose weights hash was internally self-consistent but
    was produced under a DIFFERENT config/dataset/gene-panel/synchronized-
    init/gigapath-checkpoint than what THIS basis-fitting run is currently
    using would still pass. `verify_full_checkpoint_identity` (the same
    function `evaluate_gen3_checkpoint` uses for `best`/latest) now runs
    FIRST, before any residual computation, failing closed on exactly
    that mismatch."""
    config = resolved_config(architecture3_config_path)
    architecture_id = str(config["model"]["architecture"])
    if architecture_id != "3":
        raise ValueError(
            f"fit_and_save_architecture4_basis requires an Architecture 3 config (got model.architecture="
            f"{architecture_id!r}) -- the basis must be fit against Architecture 3's own trained "
            "conditioner, per Adam's Step 6 audit #11, never a different architecture's"
        )
    data_cfg = config["data"]
    manifest_path = data_cfg.get("gen3_manifest_path")
    if not manifest_path:
        raise ValueError("data.gen3_manifest_path must be set to a real, already-built dataset manifest")
    dataset_manifest = load_dataset_manifest(manifest_path)
    train_ids = list(dataset_manifest["train_sample_ids"])
    if not train_ids:
        raise ValueError("dataset manifest has zero train_sample_ids -- nothing to fit a residual basis on")
    gene_names = list(dataset_manifest["gene_panel"])

    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, preflight_report = load_and_preflight_samples(cfg_om, dataset_manifest, train_ids, expected_provenance)

    # Codex re-audit of commit 57f0e3c: "fit_and_save_architecture4_basis()
    # still independently uses/resolves the mutable architecture3_
    # checkpoint_dir at least three times: during verify_full_checkpoint_
    # identity, during build_model_for_inference, and again when writing
    # provenance. Architecture 3 could checkpoint between those
    # operations. The basis could be computed using bundle B while its
    # provenance records bundle C." Confirmed real -- fixed by resolving
    # `architecture3_checkpoint_dir` to an immutable bundle identity
    # EXACTLY ONCE, here, before verification even runs. Every subsequent
    # step -- identity verification, model construction/weight loading,
    # and the provenance record itself -- now uses THIS SAME
    # `pinned_arch3_identity`, never a fresh, independently re-resolved
    # one, so residual computation and the provenance describing it are
    # structurally guaranteed to describe one bundle.
    pinned_arch3_identity = checkpoint_module.resolve_checkpoint_identity(architecture3_checkpoint_dir)
    # Launch blocker #5: fail closed BEFORE any residual computation --
    # never spend the (potentially expensive) residual pass against a
    # checkpoint that turns out not to describe the current run.
    verify_full_checkpoint_identity(
        pinned_arch3_identity.resolved_dir, config=config, dataset_manifest=dataset_manifest, gene_names=gene_names,
        cache_content_by_sample=preflight_report.get("cache_content_by_sample"), allow_code_drift=allow_code_drift,
    )

    strata = config["masking"]["strata"]
    train_schedule = build_gen3_mask_schedule(
        dataset_manifest, samples, strata, role="train", n_training_masks_per_sample=n_masks_per_sample,
    )
    train_dataset = Gen3SpatialFieldDataset(dataset_manifest, samples, train_schedule, strata)

    device = torch.device(device_str)

    training_mask_schedule_fingerprint = hashlib.sha256(
        json.dumps(train_schedule.reports, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    residual_identity = _json_normalized({
        "architecture3_config_fingerprint": config_fingerprint(config),
        "architecture3_config_identity_fingerprint": config_identity_fingerprint(config),
        "architecture3_checkpoint_trainable_weights_sha256": pinned_arch3_identity.weights_sha256,
        "architecture3_checkpoint_step": pinned_arch3_identity.step,
        "architecture3_checkpoint_bundle_id": pinned_arch3_identity.bundle_dir,
        "architecture3_checkpoint_manifest_sha256": pinned_arch3_identity.manifest_sha256,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "n_genes": len(gene_names),
        "train_sample_ids": sorted(train_ids),
        "n_masks_per_sample": int(n_masks_per_sample),
        "training_mask_schedule_fingerprint": training_mask_schedule_fingerprint,
        "cache_content_by_sample": preflight_report.get("cache_content_by_sample"),
    })

    residual_path = (
        Path(reuse_residuals_path)
        if reuse_residuals_path is not None
        else Path(f"{output_basis_path}.residuals.tmp.{os.getpid()}.npy")
    )
    residual_sidecar_path = _residual_sidecar_path(residual_path)
    try:
        if reuse_residuals_path is not None:
            residuals = _load_verified_residual_cache(residual_path, residual_identity)
            print(f"reusing verified Architecture 4 residual matrix: {residual_path}")
        else:
            # Audit #2 of commit a32051b: the ONE shared model-
            # reconstruction pipeline, pinned to the exact bundle already
            # verified above.  Model construction is skipped entirely when
            # retrying SVD from a verified residual cache.
            architecture3_model, _info = build_model_for_inference(
                config, gene_names=gene_names, device=device,
                checkpoint_dir=pinned_arch3_identity.resolved_dir, smoke=False,
                dataset_manifest=dataset_manifest,
                cache_content_by_sample=preflight_report.get("cache_content_by_sample"),
            )
            try:
                residuals = compute_training_residuals(
                    architecture3_model, train_dataset, device, memmap_path=residual_path,
                )
            except BaseException:
                # No sidecar was committed, so this can only be incomplete.
                residual_path.unlink(missing_ok=True)
                residual_sidecar_path.unlink(missing_ok=True)
                raise
            _write_verified_residual_cache(residuals, residual_path, residual_identity)

        n_residual_rows = int(residuals.shape[0])
        print(
            f"fitting rank-{rank} residual basis on {svd_device} "
            f"(n_iter={svd_n_iter}, n_oversamples={svd_oversamples}, shape={tuple(residuals.shape)})"
        )
        basis = fit_gene_residual_basis(
            residuals,
            gene_names,
            rank=rank,
            svd_device=svd_device,
            n_iter=svd_n_iter,
            n_oversamples=svd_oversamples,
        )
    except BaseException:
        if residual_path.is_file() and residual_sidecar_path.is_file():
            print(
                f"basis fitting failed; verified residual matrix retained at {residual_path}. "
                f"Retry with --reuse-residuals {residual_path}"
            )
        elif reuse_residuals_path is None:
            # A locally-created file without a committed sidecar is not a
            # valid retry artifact (for example, interruption while hashing).
            residual_path.unlink(missing_ok=True)
            residual_sidecar_path.unlink(missing_ok=True)
        raise
    # Launch blocker #5 / Codex re-audit of commit 2162ff4, finding #5:
    # "write basis + provenance transactionally." `save_gene_residual_
    # basis` is already an atomic tmp-then-`os.replace` write; the
    # provenance sidecar below is written the SAME way, and only AFTER
    # the basis file is fully durable -- a crash can therefore never
    # leave a provenance sidecar referencing a basis file that does not
    # yet exist (or a stale basis with no sidecar at all, since a caller
    # that sees `output_basis_path` exist but `<path>.provenance.json`
    # missing already knows -- and `maybe_load_gene_basis` already
    # enforces -- that the sidecar is mandatory for a non-smoke run).
    saved_basis_path = save_gene_residual_basis(basis, output_basis_path)

    # Codex re-audit of commit 90f853e, launch blocker #2 (and re-audit of
    # commit 57f0e3c): the SAME `pinned_arch3_identity` resolved ONCE at
    # function entry, above -- never a fresh `resolve_checkpoint_identity
    # (architecture3_checkpoint_dir)` call here, which would re-resolve
    # the mutable path a second time and could observe a DIFFERENT bundle
    # than the one verification/residual computation actually used if
    # Architecture 3 checkpointed again during residual computation.
    resolved_identity = pinned_arch3_identity
    # Launch blocker #5: a real, deterministic fingerprint over the
    # complete REALIZED training-mask schedule this basis was fit
    # against -- bound (recorded), though (per `maybe_load_gene_basis`'s
    # own docstring) deliberately not equality-checked at load time,
    # since a legitimately re-diversified training schedule must not
    # invalidate an otherwise-valid basis.
    # Codex re-audit of commit 2162ff4, finding #5: "record the numerical
    # basis SHA256 and verify it when loading" -- the sidecar previously
    # recorded shape (`n_genes`/`rank`) but nothing binding it to the
    # basis's own NUMERICAL content, so a valid sidecar could be copied
    # beside a DIFFERENT, same-shape basis (a hand-edited or re-fit-
    # differently file) without detection. Same hash construction
    # `verify_full_checkpoint_identity` already uses for this exact
    # purpose elsewhere.
    gene_residual_basis_sha256 = hashlib.sha256(
        np.ascontiguousarray(basis.basis.detach().cpu().numpy()).tobytes()
    ).hexdigest()
    provenance = {
        "version": 3,
        "kind": "gen3_architecture4_residual_basis_provenance",
        "architecture3_config_path": str(architecture3_config_path),
        "architecture3_config_fingerprint": config_fingerprint(config),
        "architecture3_config_identity_fingerprint": config_identity_fingerprint(config),
        "architecture3_checkpoint_dir": str(architecture3_checkpoint_dir),
        "architecture3_checkpoint_trainable_weights_sha256": resolved_identity.weights_sha256,
        "architecture3_checkpoint_step": resolved_identity.step,
        # Codex re-audit of commit 2162ff4, finding #5: "record and
        # validate canonical conditioner bundle_id/manifest SHA/step."
        "architecture3_checkpoint_bundle_id": resolved_identity.bundle_dir,
        "architecture3_checkpoint_manifest_sha256": resolved_identity.manifest_sha256,
        "dataset_manifest_fingerprint": dataset_manifest_fingerprint(dataset_manifest),
        "gene_panel_hash": gene_panel_hash(gene_names),
        "n_genes": len(gene_names),
        "train_sample_ids": sorted(train_ids),
        "n_masks_per_sample": int(n_masks_per_sample),
        "n_residual_rows": n_residual_rows,
        "rank": basis.rank,
        "svd_device_type": torch.device(svd_device).type,
        "svd_n_iter": int(svd_n_iter),
        "svd_oversamples": int(svd_oversamples),
        "gene_residual_basis_sha256": gene_residual_basis_sha256,
        "mask_schedule_reports": train_schedule.reports,
        "training_mask_schedule_fingerprint": training_mask_schedule_fingerprint,
        "cache_content_by_sample": preflight_report.get("cache_content_by_sample"),
        "output_basis_path": str(saved_basis_path),
    }
    provenance_path = Path(f"{saved_basis_path}.provenance.json")
    provenance_tmp = provenance_path.with_name(f"{provenance_path.name}.tmp.{os.getpid()}")
    provenance_tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str))
    os.replace(provenance_tmp, provenance_path)

    # Only discard the expensive residual matrix after BOTH basis and
    # provenance have been committed.  Any SVD/save interruption leaves a
    # hash-verified retry artifact rather than forcing inference again.
    del residuals
    residual_path.unlink(missing_ok=True)
    residual_sidecar_path.unlink(missing_ok=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="A real architecture3.yaml (data.gen3_manifest_path must be set)")
    parser.add_argument("--architecture3-checkpoint-dir", required=True, help="A real, already-trained Architecture 3 checkpoint_dir")
    parser.add_argument("--output-basis-path", required=True)
    parser.add_argument("--n-masks-per-sample", type=int, default=20)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--svd-device", default="cpu",
        help="Device for the truncated SVD only. Use 'cuda' on an A100 to avoid the very slow CPU decomposition.",
    )
    parser.add_argument("--svd-n-iter", type=int, default=4)
    parser.add_argument("--svd-oversamples", type=int, default=16)
    parser.add_argument(
        "--reuse-residuals",
        help="Retry basis fitting from a retained, hash-verified residual .npy file produced by a failed prior run",
    )
    parser.add_argument(
        "--allow-code-drift", action="store_true",
        help="Explicit override to fit a basis from an Architecture 3 checkpoint trained under a "
             "different git commit or dirty worktree than the current one. Codex re-audit of commit "
             "2162ff4, finding #4. Never a silent bypass.",
    )
    args = parser.parse_args()
    provenance = fit_and_save_architecture4_basis(
        args.config, args.architecture3_checkpoint_dir, args.output_basis_path,
        n_masks_per_sample=args.n_masks_per_sample, rank=args.rank, device_str=args.device,
        allow_code_drift=args.allow_code_drift, reuse_residuals_path=args.reuse_residuals,
        svd_device=args.svd_device, svd_n_iter=args.svd_n_iter,
        svd_oversamples=args.svd_oversamples,
    )
    print(f"gene residual basis fit and saved: {json.dumps(provenance, indent=2, default=str)}")


if __name__ == "__main__":
    main()
