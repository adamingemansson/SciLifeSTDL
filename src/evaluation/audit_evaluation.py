"""Reproducible mask-bank evaluation for missing-tissue experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.mask_bank import record_masks, split_records
from src.evaluation import metrics as ev
from src.evaluation.run_comparison import _fid_n_components
from src.training.validation import move_to_device, predictive_samples


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


def _fixed_pca(records, obs_names, coords3d, expr, requested_components: int, k: int):
    """Fit one PCA basis and dimension for the complete test mask bank."""
    from sklearn.decomposition import PCA

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
    return PCA(n_components=n_components, random_state=0).fit(reference), n_components


def _pseudo_domain_labels(adata, evaluation):
    """Return diagnostic labels and an honest source description."""
    if not bool(evaluation.get("spatial_domain_plausibility", True)):
        return None, "disabled"
    curated_key = evaluation.get("curated_label_key")
    if curated_key is not None and str(curated_key) in adata.obs:
        return adata.obs[str(curated_key)].to_numpy().astype(str), f"curated:{curated_key}"
    try:
        from src.evaluation.cell_type_classifier import cluster_pseudo_labels
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
    novae_inputs: dict,
    output_path: str | Path,
    organ: str | None = None,
    tech: str | None = None,
) -> dict[str, Any]:
    """Evaluate predictive means and uncertainty on untouched test masks.

    Results are reported separately for each image-availability mode. The
    pseudo-label diagnostic is deliberately named ``spatial_domain`` rather
    than ``cell_type`` because the labels are unsupervised Leiden domains from
    the complete slide, not curated biological annotations.
    """
    from src.training.train import _build_masked_item

    evaluation = cfg.get("evaluation", {})
    records = split_records(bank, "test")
    if not records:
        raise ValueError("mask bank contains no test records")
    n_samples = max(1, int(evaluation.get("n_samples", 20)))
    sampling_seed = int(evaluation.get("sampling_seed", 1_200_000))
    image_modes = list(evaluation.get("image_modes", ["full", "target_zero", "all_zero", "shuffled"]))
    k = int(evaluation.get("k_neighborhood", 8))
    requested_pca = int(evaluation.get("pca_n_components", 50))
    decoder_idx = getattr(model, "_decoder_target_col_idx", None)
    if decoder_idx is None:
        metric_expr = expr
    else:
        idx_np = decoder_idx.detach().cpu().numpy()
        metric_expr = expr[:, idx_np]
    pca, effective_pca = _fixed_pca(
        records, adata.obs_names, coords3d, metric_expr, requested_pca, k
    )
    domain_labels, domain_label_source = _pseudo_domain_labels(adata, evaluation)

    model_device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    result: dict[str, Any] = {
        "version": 2,
        "experiment_name": str(cfg.experiment_name),
        "n_test_masks": len(records),
        "n_samples_per_mask": n_samples,
        "requested_pca_components": requested_pca,
        "effective_pca_components": effective_pca,
        "image_modes": {},
        "spatial_domain_label_source": domain_label_source,
        "metric_notes": {
            "spatial_domain_plausibility": (
                "Random-forest agreement with the configured real-expression domain labels. "
                "Unsupervised labels are a diagnostic, not curated cell-type accuracy."
            ),
            "st_fid": "All masks and image modes use the same PCA basis and effective dimensionality.",
        },
    }

    domain_classifiers = {}
    if domain_labels is not None:
        from src.evaluation.cell_type_classifier import SpatialDomainPlausibilityClassifier
        n_estimators = int(evaluation.get("spatial_domain_n_estimators", 100))
        for record in records:
            context_mask, _ = record_masks(record, adata.obs_names)
            try:
                domain_classifiers[int(record["index"])] = SpatialDomainPlausibilityClassifier(
                    n_estimators=n_estimators, seed=int(record["seed"])
                ).fit(metric_expr[context_mask], domain_labels[context_mask])
            except Exception as exc:
                print(f"spatial-domain classifier fit failed for mask {record['index']}: {exc}")

    for mode_index, image_mode in enumerate(image_modes):
        per_mask = []
        for i, record in enumerate(records):
            context_mask, query_mask = record_masks(record, adata.obs_names)
            item = _build_masked_item(
                coords3d, expr, slice_ids, cfg.masking, images, int(record["seed"]),
                context_gene_features=novae_inputs.get("context_gene_features"),
                context_novae_features=novae_inputs.get("context_novae_features"),
                context_gene_feature_provider=novae_inputs.get("context_gene_feature_provider"),
                context_novae_feature_provider=novae_inputs.get("context_novae_feature_provider"),
                organ=organ, tech=tech, augment=False, image_mode=str(image_mode),
                fixed_context_mask=context_mask, fixed_query_mask=query_mask,
            )
            item_device = move_to_device(item, model_device)
            with torch.inference_mode():
                samples = predictive_samples(
                    model, item_device["context"], item_device["query"], n_samples,
                    seed=sampling_seed + mode_index * 100_000 + i,
                )
            target_t = _target_for_model(model, item_device["target_expression"])
            pred_t = samples.mean(dim=0)
            lower = torch.quantile(samples, 0.05, dim=0)
            upper = torch.quantile(samples, 0.95, dim=0)
            pred = pred_t.detach().cpu().numpy()
            target = target_t.detach().cpu().numpy()

            row = {
                "mask_index": int(record["index"]),
                "seed": int(record["seed"]),
                "n_context": int(context_mask.sum()),
                "n_query": int(query_mask.sum()),
                "pcc": float(np.nanmean(ev.pearson_per_gene(pred, target))),
                "rmse": float(ev.rmse(pred, target)),
                "nonzero_auc": float(ev.nonzero_auc(pred, target)),
                "predictive_std": float(samples.std(dim=0, unbiased=False).mean().cpu()),
                "interval90_coverage": float(((target_t >= lower) & (target_t <= upper)).float().mean().cpu()),
                "interval90_width": float((upper - lower).mean().cpu()),
            }

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

        result["image_modes"][str(image_mode)] = {
            "summary": _mean_and_std(per_mask),
            "per_mask": per_mask,
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=True))
    tmp.replace(output_path)
    return result
