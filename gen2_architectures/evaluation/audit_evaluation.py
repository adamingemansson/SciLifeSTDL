"""Reproducible mask-bank evaluation for missing-tissue experiments."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gen2_architectures.data.mask_bank import record_masks, split_records
from gen2_architectures.evaluation import metrics as ev
from gen2_architectures.training.validation import move_to_device, predictive_samples


def _fid_n_components(pca_components: int, context_n: int, context_d: int, query_n: int) -> int:
    """Return a covariance-safe PCA width for ST-FID/ST-MMD.

    Inlined from src/evaluation/run_comparison.py (2026-07-25) rather than
    imported, so this evaluation module doesn't drag in run_comparison's own
    dependency on the old src/training/train.py's main()/load_trained_model
    — this folder is meant to be a self-contained unit, not entangled with
    the old repo's much larger training entrypoint. Identical logic, not a
    reimplementation: the PCA basis is fitted on context-derived
    neighborhoods, while FID/MMD covariance is estimated on query-derived
    embeddings, so the component count must respect both sample counts as
    well as the feature width. The one-component floor is retained for
    backward-compatible diagnostics.
    """
    return max(1, min(
        int(pca_components),
        max(1, int(context_n) - 1),
        max(1, int(context_d)),
        max(1, int(query_n) - 1),
    ))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_config(value: Any) -> Any:
    """Convert OmegaConf/plain config values into stable JSON data."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _resume_signature(
    cfg,
    records: list[dict],
    output_path: Path,
    gene_panels: dict[str, list[str]] | None = None,
) -> dict | None:
    """Fingerprint everything required to safely reuse partial audit rows.

    A saved trainable-weights file is mandatory. Without it, two distinct
    in-memory models could share the same config/output path and stale rows
    would be indistinguishable, so resumption fails closed.
    """
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir")
    if not checkpoint_dir:
        return None
    weights_path = Path(str(checkpoint_dir)) / "trainable_weights.pt"
    if not weights_path.is_file():
        return None
    mask_payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    config_payload = json.dumps(
        {
            "data": _canonical_config(cfg.get("data", {})),
            "masking": _canonical_config(cfg.get("masking", {})),
            "model": _canonical_config(cfg.get("model", {})),
            "evaluation": _canonical_config(cfg.get("evaluation", {})),
            "gene_panels": gene_panels or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return {
        # Version 3 also fingerprints the resolved evaluation-only gene
        # panels. Config/weight/mask hashes alone cannot detect train-derived
        # panel membership passed in by the multi-sample training path.
        "version": 3,
        "output_name": output_path.name,
        "weights_sha256": _sha256_file(weights_path),
        "mask_records_sha256": hashlib.sha256(mask_payload.encode()).hexdigest(),
        "config_sha256": hashlib.sha256(config_payload.encode()).hexdigest(),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=True))
    tmp.replace(path)


def _target_for_model(model, target: torch.Tensor) -> torch.Tensor:
    idx = getattr(model, "_decoder_target_col_idx", None)
    return target if idx is None else target[:, idx]


def _mean_and_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return out


def _resolve_gene_panels(
    gene_names: list[str],
    gene_panels: dict[str, list[str]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Map prespecified gene-name panels onto the model output columns.

    Panels are evaluation views only: they never alter the model vocabulary,
    targets, loss, or decoder. Missing names are recorded rather than silently
    changing the requested panel size. Panel names are restricted to stable
    metric-key characters because they become JSON field suffixes.
    """
    import re

    name_to_idx = {str(name): idx for idx, name in enumerate(gene_names)}
    indices: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for raw_name, raw_genes in (gene_panels or {}).items():
        panel_name = str(raw_name)
        if not re.fullmatch(r"[A-Za-z0-9_]+", panel_name):
            raise ValueError(
                f"gene panel name {panel_name!r} must contain only letters, digits and underscores"
            )
        requested = list(dict.fromkeys(str(gene) for gene in raw_genes))
        present = [gene for gene in requested if gene in name_to_idx]
        missing = [gene for gene in requested if gene not in name_to_idx]
        if not present:
            raise ValueError(
                f"gene panel {panel_name!r} has no genes in the model output vocabulary"
            )
        indices[panel_name] = np.asarray([name_to_idx[gene] for gene in present], dtype=np.int64)
        metadata[panel_name] = {
            "requested_count": len(requested),
            "evaluated_count": len(present),
            "genes": present,
            "missing_genes": missing,
        }
    return indices, metadata


def _fixed_pca(records, obs_names, coords3d, expr, requested_components: int, k: int):
    """Fit one PCA basis and dimension for the complete test mask bank.

    2026-07-26: wrapped in threadpoolctl.threadpool_limits -- BLAS
    (OpenBLAS/MKL) defaults to spawning one thread per core with no
    awareness of other processes. With 4 architectures training
    concurrently on the same server, each hitting this PCA fit around the
    same time caused massive oversubscription (observed directly: `top`
    showed a single one of these calls consuming ~9700% CPU, ~97 cores, on
    a box also running 3 other equally CPU-hungry jobs) -- a fit that
    should take seconds stretched to many hours of real wall-clock time
    from contention/cache-thrashing alone, not a hang or deadlock (io
    counters and py-spy-equivalent /proc inspection confirmed the process
    was genuinely computing, not stuck). n_components here is always small
    (evaluation.pca_n_components, typically 50), so capping BLAS threads
    costs essentially nothing in the single-job case."""
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    context_patches = []
    min_query = None
    for record in records:
        context_mask, query_mask = record_masks(record, obs_names)
        context_patches.append(
            ev.pool_knn_neighborhood(coords3d[context_mask], expr[context_mask], k=k)
        )
        n_query = int(query_mask.sum())
        min_query = n_query if min_query is None else min(min_query, n_query)
    reference = np.concatenate(context_patches, axis=0)
    n_components = _fid_n_components(
        requested_components, reference.shape[0], reference.shape[1], min_query or 0
    )
    if n_components < 2:
        return None, 0
    with threadpool_limits(limits=8):
        pca = PCA(n_components=n_components, random_state=0).fit(reference)
    return pca, n_components


def _pseudo_domain_labels(adata, evaluation):
    """Return diagnostic labels and an honest source description."""
    if not bool(evaluation.get("spatial_domain_plausibility", True)):
        return None, "disabled"
    curated_key = evaluation.get("curated_label_key")
    if curated_key is not None and str(curated_key) in adata.obs:
        return adata.obs[str(curated_key)].to_numpy().astype(str), f"curated:{curated_key}"
    try:
        from gen2_architectures.evaluation.cell_type_classifier import cluster_pseudo_labels
        method = str(evaluation.get("spatial_domain_method", "kmeans"))
        labels = cluster_pseudo_labels(
            adata,
            method=method,
            resolution=float(evaluation.get("spatial_domain_resolution", 1.0)),
            n_neighbors=int(evaluation.get("spatial_domain_n_neighbors", 15)),
            n_clusters=evaluation.get("spatial_domain_n_clusters"),
            seed=int(evaluation.get("spatial_domain_seed", 0)),
        )
        return labels, f"unsupervised:{method}"
    except Exception as exc:
        print(f"spatial-domain plausibility skipped: {type(exc).__name__}: {exc}")
        return None, f"failed:{type(exc).__name__}"


def evaluate_model_on_mask_bank(
    model,
    cfg,
    adata,
    coords3d: np.ndarray,
    expr: np.ndarray,
    slice_ids: np.ndarray,
    images: np.ndarray | None,
    bank: dict,
    gene_inputs: dict,
    output_path: str | Path,
    organ: str | None = None,
    tech: str | None = None,
    gene_panels: dict[str, list[str]] | None = None,
    raw_counts: np.ndarray | None = None,
    raw_library_size: np.ndarray | None = None,
    expression_target_sum: float = 1e4,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate predictive means and uncertainty on untouched masks.

    Results are reported separately for each image-availability mode. The
    pseudo-label diagnostic is deliberately named ``spatial_domain`` rather
    than ``cell_type`` because the labels are unsupervised Leiden domains from
    the complete slide, not curated biological annotations.

    split (2026-07-26 bugfix): selects which mask-bank pool to score against
    -- "validation" (evaluation.n_validation_masks, smaller) for periodic
    in-training monitoring, "test" (evaluation.n_test_masks, the full/final
    suite) for the one true held-out report at the end of a run. This used
    to be hardcoded to always read the "test" pool regardless of what the
    caller passed, so periodic validation-time checks during training cost
    exactly as much as the final evaluation -- with n_test_masks=16, that
    meant 16 sequential RandomForestClassifier fits (spatial-domain
    plausibility) plus a full PCA fit, every eval_every_n_steps, for the
    life of a run. Real-server observation: a full 16-mask evaluation
    with 4 concurrent architectures competing for BLAS threads (see
    threadpool_limits below) still took multiple minutes even after fixing
    thread oversubscription -- an unavoidable cost for the final report,
    but wasteful to pay every 5000 steps just to monitor training.

    raw_counts/raw_library_size (2026-07-24, notebook-comparability check):
    optional, row/column-aligned with ``expr`` (see
    src/data/loaders.py::basic_qc_and_normalize's "raw_counts"/
    "_scilifestdl_raw_library_size" stash). When supplied, an ADDITIONAL
    ``pcc_raw_log1p`` metric is computed by inverting the model's prediction
    (which lives in our library-size-normalized log1p space) back to raw-
    count-log1p space using each query spot's TRUE total count, then
    comparing against real log1p(raw_counts) ground truth — the exact space
    STPath's own pretrained weights and the reference notebook evaluate in
    (see stpath/app/pipeline/inference.py's agent.inference internal log1p-
    only convention, verified directly against the cloned STPath source).
    None (default) skips this entirely — purely additive, does not change
    any existing metric for any config that doesn't pass it."""
    from gen2_architectures.data.masked_item import build_masked_item as _build_masked_item

    output_path = Path(output_path)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    evaluation_started = time.monotonic()
    evaluation = cfg.get("evaluation", {})
    records = split_records(bank, split)
    if not records:
        raise ValueError(f"mask bank contains no {split!r} records")
    n_samples = max(1, int(evaluation.get("n_samples", 20)))
    sampling_seed = int(evaluation.get("sampling_seed", 1_200_000))
    image_modes = list(evaluation.get("image_modes", ["full", "target_zero", "all_zero", "shuffled"]))
    primary_image_mode = str(evaluation.get("primary_image_mode", "full"))
    context_gex_mode = str(evaluation.get("context_gex_mode", "full"))
    if primary_image_mode not in {str(mode) for mode in image_modes}:
        raise ValueError(
            f"evaluation.primary_image_mode={primary_image_mode!r} is not present in "
            f"evaluation.image_modes={image_modes!r}"
        )
    k = int(evaluation.get("k_neighborhood", 8))
    requested_pca = int(evaluation.get("pca_n_components", 50))
    total_cells = len(records) * len(image_modes)
    total_draws = total_cells * n_samples
    print(
        f"audit evaluation started: {len(records)} masks x {len(image_modes)} image modes "
        f"x {n_samples} samples = {total_draws} stochastic draws",
        flush=True,
    )
    decoder_idx = getattr(model, "_decoder_target_col_idx", None)
    if decoder_idx is None:
        metric_expr = expr
        metric_gene_names = [str(name) for name in adata.var_names]
        metric_raw_counts = raw_counts
    else:
        idx_np = decoder_idx.detach().cpu().numpy()
        metric_expr = expr[:, idx_np]
        metric_gene_names = [str(adata.var_names[idx]) for idx in idx_np]
        metric_raw_counts = None if raw_counts is None else raw_counts[:, idx_np]
    panel_indices, panel_metadata = _resolve_gene_panels(metric_gene_names, gene_panels)
    # 2026-07-27: _fixed_pca was confirmed (by elimination -- with
    # spatial_domain_plausibility already disabled, and timing prints
    # around both remaining pre-loop steps added, still zero output on a
    # real run) to be the one operation left standing between "audit
    # evaluation started" and the per-mask progress prints ever firing.
    # Root cause not yet identified (a (~1280, ~17000) matrix extracting
    # 50 components via randomized SVD should be a sub-second operation on
    # this hardware) -- made skippable so PCC/RMSE, the metrics that
    # actually matter for the 4-architecture comparison, aren't blocked by
    # it while that's investigated separately. evaluation.compute_st_fid_mmd
    # defaults to False; st_fid/st_mmd simply report nan when skipped
    # (same as when the data is too small to fit a PCA basis at all).
    if bool(evaluation.get("compute_st_fid_mmd", False)):
        _pca_started = time.monotonic()
        pca, effective_pca = _fixed_pca(
            records, adata.obs_names, coords3d, metric_expr, requested_pca, k
        )
        print(f"audit evaluation: PCA basis fit in {time.monotonic() - _pca_started:.1f}s", flush=True)
    else:
        pca, effective_pca = None, 0
    domain_labels, domain_label_source = _pseudo_domain_labels(adata, evaluation)

    model_device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    result: dict[str, Any] = {
        "version": 3,
        "experiment_name": str(cfg.experiment_name),
        "n_test_masks": len(records),
        "n_samples_per_mask": n_samples,
        "requested_pca_components": requested_pca,
        "effective_pca_components": effective_pca,
        "n_evaluated_genes": int(metric_expr.shape[1]),
        "gene_panels": panel_metadata,
        "primary_image_mode": primary_image_mode,
        "context_gex_mode": context_gex_mode,
        "modality_ablation": str(cfg.get("data", {}).get("modality_ablation", "both")),
        "image_modes": {},
        "spatial_domain_label_source": domain_label_source,
        "metric_notes": {
            "spatial_domain_plausibility": (
                "Random-forest agreement with the configured real-expression domain labels. "
                "Unsupervised labels are a diagnostic, not curated cell-type accuracy."
            ),
            "st_fid": "All masks and image modes use the same PCA basis and effective dimensionality.",
            "pcc": (
                "Per-gene PCC eligibility is determined by non-constant ground truth. "
                "A constant prediction for an eligible gene scores 0; n_pcc_genes is reported."
            ),
            "gene_panels": (
                "Evaluation-only slices of the full model output. They do not alter training, "
                "the loss, input genes, or decoder genes."
            ),
        },
    }

    _signature_started = time.monotonic()
    signature = _resume_signature(cfg, records, output_path, gene_panels=gene_panels)
    print(
        f"audit evaluation: resume-signature computed in {time.monotonic() - _signature_started:.1f}s "
        f"(hashes the checkpoint's trainable_weights.pt)",
        flush=True,
    )
    if signature is not None and partial_path.is_file():
        try:
            partial = json.loads(partial_path.read_text())
            if partial.get("signature") == signature and isinstance(partial.get("result"), dict):
                result = partial["result"]
                result.setdefault("primary_image_mode", primary_image_mode)
                result.setdefault("context_gex_mode", context_gex_mode)
                result.setdefault(
                    "modality_ablation",
                    str(cfg.get("data", {}).get("modality_ablation", "both")),
                )
                completed = sum(
                    len(mode.get("per_mask", []))
                    for mode in result.get("image_modes", {}).values()
                )
                print(
                    f"audit evaluation resuming {completed}/{total_cells} completed mask-mode cells "
                    f"from {partial_path}",
                    flush=True,
                )
            else:
                print(
                    f"audit evaluation ignoring stale partial file with a different signature: "
                    f"{partial_path}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"audit evaluation ignoring unreadable partial file {partial_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    domain_classifiers = {}
    if domain_labels is not None:
        from gen2_architectures.evaluation.cell_type_classifier import SpatialDomainPlausibilityClassifier
        n_estimators = int(evaluation.get("spatial_domain_n_estimators", 100))
        # 2026-07-26: this loop used to run fully silent -- a real-server
        # observation showed it can take many minutes for records=16 (100
        # trees x 16 sequential fits, n_jobs=1) with ZERO print in between,
        # which looked identical to a genuine hang from the outside. Added
        # a progress print per fit, and bumped n_jobs from 1 to a small
        # bounded value (not unbounded -- that's exactly the BLAS-
        # oversubscription mistake fixed above for _fixed_pca/kmeans) so
        # each fit is meaningfully faster even with several architectures
        # training concurrently on the same server.
        for fit_index, record in enumerate(records):
            context_mask, _ = record_masks(record, adata.obs_names)
            fit_started = time.monotonic()
            try:
                domain_classifiers[int(record["index"])] = SpatialDomainPlausibilityClassifier(
                    n_estimators=n_estimators, seed=int(record["seed"]), n_jobs=4,
                ).fit(metric_expr[context_mask], domain_labels[context_mask])
                print(
                    f"audit evaluation: spatial-domain classifier {fit_index + 1}/{len(records)} "
                    f"fit in {time.monotonic() - fit_started:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"spatial-domain classifier fit failed for mask {record['index']}: {exc}")

    for mode_index, image_mode in enumerate(image_modes):
        mode_key = str(image_mode)
        existing_mode = result.get("image_modes", {}).get(mode_key, {})
        per_mask = list(existing_mode.get("per_mask", []))
        completed_keys = {
            (int(row["mask_index"]), int(row["seed"]))
            for row in per_mask
            if "mask_index" in row and "seed" in row
        }
        print(
            f"audit evaluation mode {mode_index + 1}/{len(image_modes)}: {mode_key} "
            f"({len(completed_keys)}/{len(records)} masks already complete)",
            flush=True,
        )
        for i, record in enumerate(records):
            record_key = (int(record["index"]), int(record["seed"]))
            if record_key in completed_keys:
                continue
            context_mask, query_mask = record_masks(record, adata.obs_names)
            item = _build_masked_item(
                coords3d, expr, slice_ids, cfg.masking, images, int(record["seed"]),
                context_gene_features=gene_inputs.get("context_gene_features"),
                context_extra_features=gene_inputs.get("context_extra_features"),
                context_gene_feature_provider=gene_inputs.get("context_gene_feature_provider"),
                context_extra_feature_provider=gene_inputs.get("context_extra_feature_provider"),
                organ=organ, tech=tech, augment=False, image_mode=str(image_mode),
                context_gex_mode=context_gex_mode,
                fixed_context_mask=context_mask, fixed_query_mask=query_mask,
                strict_broken_region=bool(cfg.get("data", {}).get("strict_broken_region", False)),
                query_patch_size=float(cfg.get("data", {}).get("query_patch_size_fullres", 224.0)),
            )
            item_device = move_to_device(item, model_device)
            with torch.inference_mode():
                samples = predictive_samples(
                    model, item_device["context"], item_device["query"], n_samples,
                    # Common random numbers across image modes: changing
                    # full/zero/shuffled H&E must not also change the FM
                    # noise draws, otherwise apparent image sensitivity is
                    # confounded by Monte-Carlo variation.
                    seed=sampling_seed + i,
                )
            target_t = _target_for_model(model, item_device["target_expression"])
            pred_t = samples.mean(dim=0)
            if not torch.isfinite(samples).all():
                raise FloatingPointError(
                    f"audit prediction contains non-finite values for image mode "
                    f"{mode_key!r}, mask index {record['index']}; metrics were not written"
                )
            if not torch.isfinite(target_t).all():
                raise FloatingPointError(
                    f"audit target contains non-finite values for mask index {record['index']}"
                )
            lower = torch.quantile(samples, 0.05, dim=0)
            upper = torch.quantile(samples, 0.95, dim=0)
            pred = pred_t.detach().cpu().numpy()
            target = target_t.detach().cpu().numpy()

            pcc_by_gene = ev.pearson_per_gene(pred, target)
            row = {
                "mask_index": int(record["index"]),
                "seed": int(record["seed"]),
                "n_context": int(context_mask.sum()),
                "n_query": int(query_mask.sum()),
                "pcc": float(np.nanmean(pcc_by_gene)),
                "n_pcc_genes": int(np.isfinite(pcc_by_gene).sum()),
                "rmse": float(ev.rmse(pred, target)),
                "nonzero_auc": float(ev.nonzero_auc(pred, target)),
                "predictive_std": float(samples.std(dim=0, unbiased=False).mean().cpu()),
                "interval90_coverage": float(((target_t >= lower) & (target_t <= upper)).float().mean().cpu()),
                "interval90_width": float((upper - lower).mean().cpu()),
            }
            if metric_raw_counts is not None and raw_library_size is not None:
                raw_target = np.asarray(metric_raw_counts[query_mask], dtype=np.float64)
                library_size = np.asarray(raw_library_size, dtype=np.float64)[query_mask][:, None]
                # Invert our library-size-normalized-log1p prediction back to
                # raw-count-log1p space using the query spot's OWN true total
                # count (a per-spot scalar normalizing constant, not per-gene
                # signal) -- the exact space STPath's pretrained weights and
                # the reference notebook evaluate in.
                pred_counts_est = np.clip(
                    np.expm1(pred.astype(np.float64)) * library_size / float(expression_target_sum),
                    a_min=0.0, a_max=None,
                )
                pred_raw_log1p = np.log1p(pred_counts_est)
                target_raw_log1p = np.log1p(raw_target)
                pcc_raw_by_gene = ev.pearson_per_gene(pred_raw_log1p, target_raw_log1p)
                row["pcc_raw_log1p"] = float(np.nanmean(pcc_raw_by_gene))
                row["n_pcc_raw_log1p_genes"] = int(np.isfinite(pcc_raw_by_gene).sum())
                row["rmse_raw_log1p"] = float(ev.rmse(pred_raw_log1p, target_raw_log1p))
            for panel_name, panel_idx in panel_indices.items():
                panel_pred = pred[:, panel_idx]
                panel_target = target[:, panel_idx]
                panel_pcc = ev.pearson_per_gene(panel_pred, panel_target)
                row[f"pcc_{panel_name}"] = float(np.nanmean(panel_pcc))
                row[f"n_pcc_genes_{panel_name}"] = int(np.isfinite(panel_pcc).sum())
                row[f"rmse_{panel_name}"] = float(ev.rmse(panel_pred, panel_target))

            if pca is not None:
                query_coords = coords3d[query_mask]
                query_k = min(k, len(query_coords))
                real_patch = ev.pool_knn_neighborhood(query_coords, target, k=query_k)
                gen_patch = ev.pool_knn_neighborhood(query_coords, pred, k=query_k)
                real_embed, gen_embed = pca.transform(real_patch), pca.transform(gen_patch)
                row["st_fid"] = float(ev.st_fid(real_embed, gen_embed))
                row["st_mmd"] = float(ev.st_mmd(real_embed, gen_embed))
            else:
                row["st_fid"] = float("nan")
                row["st_mmd"] = float("nan")

            classifier = domain_classifiers.get(int(record["index"]))
            if classifier is not None:
                try:
                    row["spatial_domain_plausibility"] = classifier.plausibility_accuracy(
                        pred, domain_labels[query_mask]
                    )
                except Exception as exc:
                    print(f"spatial-domain plausibility failed for mask {i}: {exc}")
                    row["spatial_domain_plausibility"] = float("nan")
            else:
                row["spatial_domain_plausibility"] = float("nan")
            per_mask.append(row)
            completed_keys.add(record_key)

            result.setdefault("image_modes", {})[mode_key] = {
                "summary": _mean_and_std(per_mask),
                "per_mask": per_mask,
            }
            if signature is not None:
                _atomic_json(
                    partial_path,
                    {"signature": signature, "result": result},
                )
            completed_cells = sum(
                len(mode.get("per_mask", []))
                for mode in result.get("image_modes", {}).values()
            )
            elapsed = time.monotonic() - evaluation_started
            print(
                f"audit evaluation progress: {completed_cells}/{total_cells} cells; "
                f"mode={mode_key} mask={i + 1}/{len(records)} "
                f"pcc={row['pcc']:.4f} rmse={row['rmse']:.4f} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

        result["image_modes"][mode_key] = {
            "summary": _mean_and_std(per_mask),
            "per_mask": per_mask,
        }

    _atomic_json(output_path, result)
    try:
        partial_path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"audit evaluation warning: could not remove {partial_path}: {exc}", flush=True)
    elapsed = time.monotonic() - evaluation_started
    print(f"audit evaluation complete in {elapsed / 60:.1f}m: {output_path}", flush=True)
    return result
