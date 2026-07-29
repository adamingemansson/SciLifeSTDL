"""The minimal real Step 7 evaluator -- Adam's explicit Step 6 audit #12
deliverable (of commit 27e1232): "Build the minimal Step 7 evaluator
before long training. On fixed patient-held-out masks, report per-
sample/per-mask PCC and RMSE with valid-gene counts, patient-level
aggregation and confidence intervals. Include harmonic, mean and simple
nearest-neighbour baselines. Keep ST-FID/ST-MMD secondary. Never select
using test samples."

Deliberately MINIMAL, per Adam's own earlier explicit instruction ("do
not implement a large new evaluation system yet") -- this is not Step
7's full evaluator, only the smallest real one that can honestly report
whether a checkpoint is learning anything on held-out data, using
already-audited primitives wherever one exists:
  - `evaluation/metrics.py::pearson_per_gene`/`rmse` for the per-item
    numbers, `aggregate_patient_metrics` (already built in Phase 7 for
    exactly this) for patient-level aggregation + 95% CIs.
  - `models/harmonic.py::harmonic_interpolation` for the harmonic
    baseline -- that module's own docstring already states "Harmonic,
    inverse-distance, and nearest-neighbour must also be computed as
    exact-mask external baselines for every arm."
  - `training/gen3_dataset.py`'s FIXED, deterministic held-out mask
    schedule (the exact same one `train.py`'s own validation loop uses)
    -- this evaluator never draws a fresh random mask.

Real, honest limits documented rather than hidden: mean/nearest-neighbour
baselines are new, small, deliberately simple functions (no existing
equivalent found by direct search); ST-FID/ST-MMD are computed only when
explicitly requested (`compute_st_fid_mmd=True`) and are never part of
the reported headline metric, matching "keep ST-FID/ST-MMD secondary."
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gen3_multiscale.data.dataset_manifest import load_dataset_manifest
from gen3_multiscale.evaluation.metrics import (
    aggregate_patient_metrics, embed_pca, gene_panel_metrics, nonzero_auc, pearson_per_gene, resolve_gene_panels,
    rmse, st_fid, st_mmd,
)
from gen3_multiscale.models.harmonic import harmonic_interpolation
from gen3_multiscale.training.gen3_dataset import Gen3SpatialFieldDataset, build_gen3_mask_schedule
from gen3_multiscale.training.gen3_preflight import load_and_preflight_samples
from gen3_multiscale.training.train import (
    build_model_for_inference, expected_tile_encoder_provenance, predict_for_metrics, resolved_config,
    verify_checkpoint_bundle_identity, verify_checkpoint_run_manifest_against_evaluation,
)


def per_item_reconstruction_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """PCC and RMSE for ONE mask/hole, with the "valid-gene count" the
    audit explicitly asked for -- a gene is "valid" for PCC purposes when
    it has non-zero variance in the TRUE values for this item
    (`pearson_per_gene`'s own eligibility rule: a truth-constant gene is
    unidentifiable within this hole and is excluded, not silently scored
    as a perfect or zero match)."""
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must have the same shape")
    per_gene_pcc = pearson_per_gene(pred, true)
    valid = np.isfinite(per_gene_pcc)
    return {
        "pcc": float(np.nanmean(per_gene_pcc)) if valid.any() else float("nan"),
        "rmse": rmse(pred, true),
        "n_valid_genes": int(valid.sum()),
        "n_genes": int(per_gene_pcc.shape[0]),
        # Launch blocker #9: "nonzero AUC" -- can distinguish
        # zero/measured-absent vs. nonzero true expression from predicted
        # magnitude alone, flattened across every gene in this item.
        "nonzero_auc": nonzero_auc(pred, true),
    }


def mean_baseline_prediction(inputs) -> np.ndarray:
    """Predict every query spot as the plain mean of this item's own
    observed expression -- the simplest possible non-trivial baseline."""
    observed = np.asarray(inputs.observed_full_gene_expression, dtype=np.float64)
    mean_expr = observed.mean(axis=0)
    return np.tile(mean_expr, (inputs.query_coords.shape[0], 1)).astype(np.float32)


def nearest_neighbor_baseline_prediction(inputs) -> np.ndarray:
    """Predict every query spot as its single spatially-nearest OBSERVED
    spot's real expression -- the simple nearest-neighbour baseline
    `harmonic.py`'s own docstring names alongside harmonic/inverse-
    distance."""
    observed_coords = np.asarray(inputs.observed_coords, dtype=np.float64)
    query_coords = np.asarray(inputs.query_coords, dtype=np.float64)
    observed_expr = np.asarray(inputs.observed_full_gene_expression, dtype=np.float64)
    dists = np.linalg.norm(query_coords[:, None, :] - observed_coords[None, :, :], axis=-1)
    nearest_idx = np.argmin(dists, axis=1)
    return observed_expr[nearest_idx].astype(np.float32)


def harmonic_baseline_prediction(inputs) -> np.ndarray:
    """The exact-mask harmonic-interpolation baseline
    (`models/harmonic.py`) -- Architecture 2's own anchor input, reused
    here unmodified as an EVALUATION baseline for every architecture,
    never as a model input."""
    return harmonic_interpolation(
        np.asarray(inputs.observed_coords, dtype=np.float64),
        np.asarray(inputs.observed_full_gene_expression, dtype=np.float64),
        np.asarray(inputs.query_coords, dtype=np.float64),
    )


_BASELINE_PREDICTORS = {
    "mean": mean_baseline_prediction,
    "nearest_neighbor": nearest_neighbor_baseline_prediction,
    "harmonic": harmonic_baseline_prediction,
}


def load_configured_gene_panels(config: dict) -> dict[str, list[str]]:
    """Codex re-audit of commit 90f853e, launch blocker #9: "configured
    named panels" -- `evaluation.gene_panels` (every architectureN.yaml
    already declares one: `{panel_name: json_path}`) was never actually
    read by this evaluator; `metrics.py::gene_panel_metrics`/
    `resolve_gene_panels` exist and are tested, but nothing called them
    with real gene NAMES, only the unresolved path strings. Mirrors
    `src/training/train.py::_resolved_evaluation_gene_panels`'s fixed-
    panel loading (a JSON file whose top-level `"genes"` key, or the
    payload itself if it is already a bare list, names the panel) --
    never the training-sample-variance-ranked panel half of that
    function, which has no equivalent config field here."""
    raw_panels = (config.get("evaluation") or {}).get("gene_panels") or {}
    panels: dict[str, list[str]] = {}
    for panel_name, raw_path in raw_panels.items():
        path = Path(str(raw_path))
        if not path.is_file():
            raise FileNotFoundError(f"configured evaluation gene panel {panel_name!r} is missing: {path}")
        payload = json.loads(path.read_text())
        genes = payload.get("genes") if isinstance(payload, dict) else payload
        if not isinstance(genes, list) or not genes:
            raise ValueError(f"evaluation gene panel {panel_name!r} must contain a non-empty gene list: {path}")
        panels[str(panel_name)] = list(dict.fromkeys(str(gene) for gene in genes))
    return panels


def architecture4_calibration_summary(standardized_residuals: list[float]) -> dict:
    """Codex re-audit of commit 90f853e, launch blocker #9: "Architecture
    4 interval coverage/calibration" -- Architecture 4 is the only
    architecture reporting a predictive standard deviation
    (`predict_for_metrics`'s `predictive_std`), and nothing previously
    checked whether it was CALIBRATED (does a nominal X% predictive
    interval actually contain the true value X% of the time?), only that
    it existed (`predictive_std_mean`). Standardized residuals
    `z = (true - pred) / predictive_std`, pooled across every query
    spot/gene/item, should have mean ~0 and std ~1 if well-calibrated;
    empirical coverage of the standard normal's own 68/90/95% intervals
    (|z| <= 1.0 / 1.645 / 1.96) should then match those nominal
    fractions. Systematically low coverage means the model is
    OVERCONFIDENT (intervals too narrow); systematically high coverage
    means it is UNDERCONFIDENT (intervals too wide)."""
    z = np.asarray(standardized_residuals, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"n_values": 0}
    abs_z = np.abs(z)
    return {
        "n_values": int(z.size),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "coverage_68": float(np.mean(abs_z <= 1.0)),
        "coverage_90": float(np.mean(abs_z <= 1.645)),
        "coverage_95": float(np.mean(abs_z <= 1.96)),
    }


def _load_model_for_evaluation(
    config: dict, checkpoint_dir: str | Path, gene_names: list[str], device: torch.device,
    dataset_manifest: dict | None = None,
):
    """Construct the real architecture and load a real, already-trained
    checkpoint's trainable weights onto it -- never a random/untrained
    model. Adam's Step 6 audit #2 of commit a32051b: this now calls the
    ONE shared `train.py::build_model_for_inference` pipeline, which
    additionally loads+freezes Architecture 4's exact Architecture 3
    conditioner -- a real, confirmed gap in the PRIOR version of this
    function, which built the architecture and loaded trainable weights
    but never called `maybe_load_pretrained_conditioner_for_architecture4`
    at all, so Architecture 4 evaluation was silently evaluating a
    conditioner that was never correctly loaded/frozen from the real
    Architecture 3 checkpoint the way training did."""
    model, _info = build_model_for_inference(
        config, gene_names=gene_names, device=device, checkpoint_dir=checkpoint_dir, smoke=False,
        dataset_manifest=dataset_manifest,
    )
    model.eval()
    return model


def evaluate_gen3_checkpoint(
    config_path: str, checkpoint_dir: str | Path, *, split: str = "validation",
    n_masks_per_sample: int = 8, use_best: bool = True, compute_st_fid_mmd: bool = False,
    allow_test: bool = False, device_str: str = "cpu",
) -> dict:
    """The real Step 7 evaluation entrypoint. Runs a REAL, already-trained
    checkpoint over the FIXED, deterministic held-out mask schedule for
    `split`, computing per-item/per-mask PCC+RMSE (with valid-gene
    counts) for the model AND every baseline, then
    `aggregate_patient_metrics` for patient-level means + 95% CIs.

    `split` must be "validation" unless `allow_test=True` is passed
    explicitly -- "Never select using test samples" is enforced here
    structurally, not just documented: this function refuses to touch
    test-split data at all by default.

    Adam's Step 6 audit #7 of commit a32051b extended this report with:
    retained per-item records (`per_item_records`, each carrying
    sample/patient/mask/stratum identity, so a caller can re-slice by any
    of those after the fact -- the prior version discarded every
    per-item value the moment it was folded into the aggregate); paired
    model-vs-baseline deltas with their own patient-level CIs
    (`per_arm_paired_delta_vs_model`, computed item-by-item, not by
    comparing two independently-aggregated means); Architecture 4's
    predictive uncertainty (`predictive_std_mean`, folded into
    `per_arm_patient_aggregated_metrics["model"]` when available); and a
    fail-closed check of the checkpoint's own bundle identity against
    THIS evaluation's dataset/gene-panel before any weights are loaded
    (audit #4's altered-cache/swapped-checkpoint adversarial scenario)."""
    if split == "test" and not allow_test:
        raise ValueError(
            "evaluate_gen3_checkpoint refuses split='test' unless allow_test=True is passed "
            "explicitly -- Adam's Step 6 audit #12: 'Never select using test samples.' Use "
            "split='validation' for any model-selection or development-time evaluation; test-split "
            "evaluation is a final, one-time report, never a decision input."
        )
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")

    config = resolved_config(config_path)
    architecture_id = str(config["model"]["architecture"])
    data_cfg = config["data"]
    dataset_manifest = load_dataset_manifest(data_cfg["gen3_manifest_path"])
    split_ids = list(dataset_manifest[f"{split}_sample_ids"])
    if not split_ids:
        raise ValueError(f"dataset manifest has zero {split}_sample_ids -- nothing to evaluate")

    cfg_om = OmegaConf.create(config)
    expected_provenance = expected_tile_encoder_provenance(config)
    samples, preflight_report = load_and_preflight_samples(cfg_om, dataset_manifest, split_ids, expected_provenance)

    strata = config["masking"]["strata"]
    # A DEDICATED subdirectory, never the raw checkpoint_dir -- train.py's
    # own validation loop persists its own mask banks directly under
    # checkpoint_dir with its own n_masks_per_sample; reusing that same
    # path here with a possibly different n_masks_per_sample would
    # collide with mask_schedule.py's own staleness check (a bank cached
    # for different split_counts at the same path is correctly rejected,
    # not silently regenerated).
    schedule = build_gen3_mask_schedule(
        dataset_manifest, samples, strata, role=split,
        split_counts={split: n_masks_per_sample}, split_seeds={split: 700_000 if split == "validation" else 900_000},
        mask_bank_dir=str(Path(checkpoint_dir) / "evaluation_masks"),
    )
    dataset = Gen3SpatialFieldDataset(dataset_manifest, samples, schedule, strata)

    gene_names = list(dataset_manifest["gene_panel"])
    device = torch.device(device_str)
    checkpoint_dir = Path(checkpoint_dir)
    # Codex re-audit of commit 90f853e, launch blocker #4: "use_best=True
    # silently falls back to latest if best/ is missing... Latest-
    # checkpoint evaluation does not compare the checkpoint's run
    # manifest with the evaluation dataset." Both are now fail-closed:
    # use_best=True with no real best/ bundle RAISES rather than quietly
    # evaluating a different (unselected, unverified-for-this-purpose)
    # checkpoint state; evaluating the latest checkpoint directly now
    # requires and verifies its own run_manifest.json against this
    # evaluation's config/dataset/gene-panel/architecture/synchronized-
    # init/conditioner/basis identity before any weights are loaded.
    if use_best:
        weights_dir = checkpoint_dir / "best"
        if not weights_dir.is_dir():
            raise ValueError(
                f"evaluate_gen3_checkpoint: use_best=True but no best/ bundle exists at {checkpoint_dir} "
                "-- refusing to silently fall back to evaluating the latest checkpoint instead. Pass "
                "use_best=False explicitly if evaluating the latest (not best-selected) checkpoint is "
                "genuinely intended."
            )
        verify_checkpoint_bundle_identity(
            weights_dir, dataset_manifest=dataset_manifest, gene_names=gene_names, architecture_id=architecture_id,
        )
    else:
        weights_dir = checkpoint_dir
        verify_checkpoint_run_manifest_against_evaluation(
            weights_dir, config=config, dataset_manifest=dataset_manifest, gene_names=gene_names,
        )
    model = _load_model_for_evaluation(config, weights_dir, gene_names, device, dataset_manifest=dataset_manifest)

    gene_panels = load_configured_gene_panels(config)
    _panel_indices, panel_metadata = resolve_gene_panels(gene_names, gene_panels) if gene_panels else ({}, {})

    per_item_by_arm: dict[str, list[dict]] = {"model": []}
    for name in _BASELINE_PREDICTORS:
        per_item_by_arm[name] = []
    per_panel_items: dict[str, list[dict]] = {panel: [] for panel in gene_panels}
    per_stratum_model_items: dict[str, list[dict]] = {}
    per_stratum_patient_ids: dict[str, list[str]] = {}
    patient_ids: list[str] = []
    per_item_records: list[dict] = []
    predictive_stds: list[float] = []
    architecture4_standardized_residuals: list[float] = []
    real_expression_for_embedding: list[np.ndarray] = []
    model_expression_for_embedding: list[np.ndarray] = []

    # Audit #1: Architecture 4's reported prediction comes from
    # `sample_predictive_distribution`'s predictive mean, reseeded per
    # item from (a fixed evaluation seed, idx) so repeated evaluation
    # runs against the same checkpoint are exactly reproducible.
    with torch.no_grad():
        for idx in range(len(dataset)):
            inputs, targets = dataset[idx]
            true_expression = np.asarray(targets.query_expression, dtype=np.float32)
            patient_ids.append(str(inputs.patient_id))
            item_identity = dataset.item_identity(idx)

            item_generator = torch.Generator(device=device).manual_seed((idx * 104_729 + 1) % (2**63))
            prediction = predict_for_metrics(architecture_id, model, inputs, generator=item_generator)
            model_pred = np.asarray(prediction["expression"].detach().cpu().numpy(), dtype=np.float32)
            model_item_metrics = per_item_reconstruction_metrics(model_pred, true_expression)
            if "predictive_std" in prediction:
                predictive_std_arr = prediction["predictive_std"].detach().cpu().numpy().astype(np.float64)
                item_predictive_std = float(predictive_std_arr.mean())
                model_item_metrics["predictive_std_mean"] = item_predictive_std
                predictive_stds.append(item_predictive_std)
                # Launch blocker #9: pooled standardized residuals for
                # `architecture4_calibration_summary` below -- every
                # query spot/gene in this item, never just the mean.
                safe_std = np.where(predictive_std_arr > 1e-8, predictive_std_arr, np.nan)
                z = (true_expression.astype(np.float64) - model_pred.astype(np.float64)) / safe_std
                architecture4_standardized_residuals.extend(z.flatten().tolist())
            per_item_by_arm["model"].append(model_item_metrics)

            stratum = item_identity["stratum"]
            record = {
                "idx": idx, "sample_id": item_identity["sample_id"], "patient_id": str(inputs.patient_id),
                "stratum": stratum, "query_fingerprint": item_identity["query_fingerprint"],
                "model": model_item_metrics,
            }
            if stratum is not None:
                per_stratum_model_items.setdefault(str(stratum), []).append(model_item_metrics)
                per_stratum_patient_ids.setdefault(str(stratum), []).append(str(inputs.patient_id))

            for name, predictor in _BASELINE_PREDICTORS.items():
                baseline_pred = predictor(inputs)
                baseline_item_metrics = per_item_reconstruction_metrics(baseline_pred, true_expression)
                per_item_by_arm[name].append(baseline_item_metrics)
                record[name] = baseline_item_metrics

            if gene_panels:
                panel_item_metrics = gene_panel_metrics(model_pred, true_expression, gene_names, gene_panels)
                record["gene_panels"] = panel_item_metrics
                for panel_name, panel_metrics in panel_item_metrics.items():
                    per_panel_items[panel_name].append(panel_metrics)

            per_item_records.append(record)

            if compute_st_fid_mmd:
                real_expression_for_embedding.append(true_expression)
                model_expression_for_embedding.append(model_pred)

    aggregated = {
        arm: aggregate_patient_metrics(items, patient_ids) for arm, items in per_item_by_arm.items()
    }
    if predictive_stds:
        aggregated["model"]["predictive_std_mean"] = float(np.mean(predictive_stds))

    # Launch blocker #9: "per-stratum results" -- the SAME patient-safe
    # aggregation used for the whole split, restricted to items sharing
    # one masking.strata name, so a caller can tell whether performance
    # is uniform across hole sizes/kinds rather than dominated by the
    # easiest stratum.
    per_stratum_aggregated_metrics = {
        stratum: aggregate_patient_metrics(items, per_stratum_patient_ids[stratum])
        for stratum, items in per_stratum_model_items.items()
    }

    # Launch blocker #9: "configured named panels" -- real per-panel
    # patient-safe aggregation, not just per-item numbers buried in
    # per_item_records.
    per_panel_patient_aggregated_metrics = {
        panel: aggregate_patient_metrics(items, patient_ids) for panel, items in per_panel_items.items()
    }

    # Launch blocker #9: "Architecture 4 interval coverage/calibration" --
    # empty ({"n_values": 0}, not omitted) for Architectures 1-3, which
    # report no predictive_std at all.
    architecture4_calibration = architecture4_calibration_summary(architecture4_standardized_residuals)

    # Audit #7: PAIRED model-vs-baseline deltas -- computed ITEM BY ITEM
    # (same mask, same sample) rather than by differencing two
    # independently-aggregated means, then run through the same
    # `aggregate_patient_metrics` patient-level-CI machinery used for the
    # raw metrics. Positive pcc_delta / positive rmse_delta both mean
    # "the model beat this baseline" on that item.
    paired_deltas = {}
    for name in _BASELINE_PREDICTORS:
        deltas = [
            {
                "pcc_delta": model_item["pcc"] - baseline_item["pcc"],
                "rmse_delta": baseline_item["rmse"] - model_item["rmse"],
            }
            for model_item, baseline_item in zip(per_item_by_arm["model"], per_item_by_arm[name])
        ]
        paired_deltas[name] = aggregate_patient_metrics(deltas, patient_ids)

    report = {
        "version": 3,
        "kind": "gen3_step7_evaluation_report",
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "weights_dir": str(weights_dir),
        "split": split,
        "n_samples": len(split_ids),
        "n_items": len(dataset),
        "cache_preflight_report": preflight_report,
        "per_arm_patient_aggregated_metrics": aggregated,
        "per_arm_paired_delta_vs_model": paired_deltas,
        "per_item_records": per_item_records,
        # Launch blocker #9.
        "per_stratum_patient_aggregated_metrics": per_stratum_aggregated_metrics,
        "gene_panel_metadata": panel_metadata,
        "per_panel_patient_aggregated_metrics": per_panel_patient_aggregated_metrics,
        "architecture4_calibration": architecture4_calibration,
    }

    if compute_st_fid_mmd and len(real_expression_for_embedding) >= 2:
        # Secondary, distributional metric only -- never gates selection.
        # embed_pca needs a PCA fit on real data first; a small, local fit
        # here (never persisted/reused) is honest about being a cheap,
        # secondary diagnostic, not a calibrated embedding space.
        from sklearn.decomposition import PCA

        real_matrix = np.concatenate(real_expression_for_embedding, axis=0)
        model_matrix = np.concatenate(model_expression_for_embedding, axis=0)
        n_components = min(10, real_matrix.shape[0] - 1, real_matrix.shape[1])
        if n_components >= 2:
            pca_model = PCA(n_components=n_components).fit(real_matrix)
            real_embeddings = embed_pca(real_matrix, pca_model)
            model_embeddings = embed_pca(model_matrix, pca_model)
            report["secondary_st_fid"] = st_fid(real_embeddings, model_embeddings)
            report["secondary_st_mmd"] = st_mmd(real_embeddings, model_embeddings)

    return report


def save_evaluation_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path
