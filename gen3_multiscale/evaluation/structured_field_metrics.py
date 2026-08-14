"""Whole-slide diagnostics for structured H&E-to-ST predictions.

Pointwise PCC/RMSE answer whether each gene has approximately the right value.
They do not answer whether a prediction preserves (1) the expression profile
within a spot, (2) gene-gene coexpression across a tissue, or (3) the spatial
field and its gradients.  This module measures those claims separately on
held-out whole slides.  It never supplies target expression to a model.

All panels are resolved before calling these functions and must be fixed from
training data.  Gradient differences are divided by training-only per-gene
scales so abundant genes cannot dominate the field diagnostic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.evaluation.metrics import pearson_per_gene
from gen3_multiscale.evaluation.noise_ceiling import normalized_score


def _arrays(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            "predicted and target must be matching [n_spots, n_genes] arrays"
        )
    if predicted.shape[0] < 2 or predicted.shape[1] < 1:
        raise ValueError("structured-field evaluation needs >=2 spots and >=1 gene")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("structured-field inputs must be finite")
    return predicted, target


def undirected_knn_edges(coords: np.ndarray, *, k_neighbors: int) -> np.ndarray:
    """Return sorted, unique undirected edges from the project's k-NN graph."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape [n_spots, 2]")
    if coords.shape[0] < 2:
        raise ValueError("at least two coordinates are required")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    adjacency = build_knn_adjacency(coords, k_neighbors=k_neighbors)
    edges = {
        (min(index, int(neighbor)), max(index, int(neighbor)))
        for index, neighbors in enumerate(adjacency)
        for neighbor in neighbors
        if int(neighbor) != index
    }
    if not edges:
        raise ValueError("k-NN graph produced no edges")
    return np.asarray(sorted(edges), dtype=np.int64)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0 if left.size >= 2 and np.std(right) >= 1e-12 else float("nan")
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def spot_profile_agreement(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Agreement across genes inside each spot, then mean over spots."""
    predicted, target = _arrays(predicted, target)
    # Chunk spots so an all-gene whole slide does not allocate several extra
    # [n_spots, n_genes] float64 arrays at once.  Transposing a spot block maps
    # the shared per-gene PCC implementation onto one PCC per original spot.
    per_spot = np.concatenate([
        pearson_per_gene(predicted[start:end].T, target[start:end].T)
        for start in range(0, predicted.shape[0], 256)
        for end in [min(start + 256, predicted.shape[0])]
    ])
    finite = per_spot[np.isfinite(per_spot)]
    return {
        "mean_spot_profile_pcc": float(finite.mean()) if finite.size else float("nan"),
        "median_spot_profile_pcc": float(np.median(finite)) if finite.size else float("nan"),
        "n_eligible_spots": int(finite.size),
    }


def coexpression_agreement(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Concordance of gene-gene correlation matrices across tissue spots.

    This is intentionally panel-scale: a 17k x 17k correlation matrix is both
    wasteful and dominated by poorly measured genes.  Callers use fixed
    training-derived HVG/marker panels.
    """
    predicted, target = _arrays(predicted, target)
    if predicted.shape[1] < 2:
        return {
            "correlation_matrix_pcc": float("nan"),
            "correlation_matrix_mae": float("nan"),
            "n_gene_pairs": 0,
        }
    target_std = target.std(axis=0)
    eligible = target_std >= 1e-8
    if eligible.sum() < 2:
        return {
            "correlation_matrix_pcc": float("nan"),
            "correlation_matrix_mae": float("nan"),
            "n_gene_pairs": 0,
        }
    target = target[:, eligible]
    predicted = predicted[:, eligible]
    target_z = (target - target.mean(axis=0)) / np.maximum(target.std(axis=0), 1e-12)
    pred_std = predicted.std(axis=0)
    pred_z = np.divide(
        predicted - predicted.mean(axis=0), np.maximum(pred_std, 1e-12),
    )
    # A constant prediction for a truth-variable gene is a failure, not a
    # reason to drop that gene pair. Its correlations are therefore zero.
    pred_z[:, pred_std < 1e-8] = 0.0
    denominator = float(target.shape[0])
    target_corr = target_z.T @ target_z / denominator
    predicted_corr = pred_z.T @ pred_z / denominator
    upper = np.triu_indices(target_corr.shape[0], k=1)
    true_pairs = target_corr[upper]
    pred_pairs = predicted_corr[upper]
    return {
        "correlation_matrix_pcc": _correlation(pred_pairs, true_pairs),
        "correlation_matrix_mae": float(np.mean(np.abs(pred_pairs - true_pairs))),
        "n_gene_pairs": int(true_pairs.size),
    }


def _visium_raster_geometry(
    coords: np.ndarray, *, pixels_per_neighbor: float = 2.0,
    margin_pixels: int = 5, max_side_pixels: int = 768,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    """Map spot centers to one deterministic, bounded regular raster.

    The median nearest-neighbour distance defines the resolution. This keeps
    the raster independent of scanner pixels-per-micron while preserving the
    spot lattice. Extremely large coordinate spans increase the pixel size
    rather than allocating an unbounded image. The same geometry is used for
    target and prediction, and no expression values influence it.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must have shape [n_spots >= 2, 2]")
    if not np.isfinite(coords).all():
        raise ValueError("coords must be finite")
    if pixels_per_neighbor <= 0 or margin_pixels < 0 or max_side_pixels < 16:
        raise ValueError("invalid Visium raster geometry settings")

    distances, _ = cKDTree(coords).query(coords, k=2)
    nearest = np.asarray(distances[:, 1], dtype=np.float64)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if nearest.size == 0:
        raise ValueError("spot coordinates have no positive neighbour spacing")
    pixel_size = float(np.median(nearest) / pixels_per_neighbor)
    spans = np.ptp(coords, axis=0)
    usable_side = max_side_pixels - 2 * margin_pixels - 1
    if usable_side < 1:
        raise ValueError("max_side_pixels is too small for the requested margin")
    pixel_size = max(pixel_size, float(np.max(spans) / usable_side), 1e-12)

    indices = np.rint((coords - coords.min(axis=0)) / pixel_size).astype(np.int64)
    # Coordinates are conventionally x/y; NumPy arrays are row/column.
    cols = indices[:, 0] + margin_pixels
    rows = indices[:, 1] + margin_pixels
    shape = (
        int(rows.max()) + margin_pixels + 1,
        int(cols.max()) + margin_pixels + 1,
    )
    return rows, cols, shape, pixel_size


def spatial_ssim_agreement(
    predicted: np.ndarray, target: np.ndarray, coords: np.ndarray, *,
    pixels_per_neighbor: float = 2.0, interpolation_sigma_pixels: float = 1.0,
    ssim_sigma_pixels: float = 1.5, max_side_pixels: int = 768,
    gene_chunk_size: int = 32,
) -> dict[str, float]:
    """Masked per-gene SSIM after one fixed Visium spot rasterization.

    Spot values are Gaussian-splatted onto a regular grid whose resolution is
    tied to the slide's median nearest-neighbour spacing. Local SSIM moments
    are normalized by the rasterized tissue support and the final score is
    averaged at observed spot centers only. Consequently, empty background
    cannot make two otherwise poor fields appear similar.

    The SSIM constants use each target gene's held-out dynamic range. A
    truth-constant gene is ineligible; a flat prediction for a variable truth
    remains a scored model failure. Predictions are not clipped to the target
    range.
    """
    predicted, target = _arrays(predicted, target)
    if interpolation_sigma_pixels <= 0 or ssim_sigma_pixels <= 0:
        raise ValueError("SSIM raster sigmas must be positive")
    if gene_chunk_size < 1:
        raise ValueError("gene_chunk_size must be positive")
    rows, cols, shape, pixel_size = _visium_raster_geometry(
        coords, pixels_per_neighbor=pixels_per_neighbor,
        max_side_pixels=max_side_pixels,
    )
    # Each chunk keeps several raster tensors alive for local moments. Bound
    # the requested chunk by raster area so pathological coordinate spans do
    # not turn an evaluation metric into a multi-gigabyte allocation.
    gene_chunk_size = min(
        int(gene_chunk_size), max(1, 2_000_000 // int(shape[0] * shape[1])),
    )

    impulses = np.zeros(shape, dtype=np.float64)
    np.add.at(impulses, (rows, cols), 1.0)
    interpolation_weight = gaussian_filter(
        impulses, interpolation_sigma_pixels, mode="constant", truncate=3.0,
    )
    support = interpolation_weight > max(float(interpolation_weight.max()) * 1e-3, 1e-12)
    support_float = support.astype(np.float64)
    local_weight = gaussian_filter(
        support_float, ssim_sigma_pixels, mode="constant", truncate=3.5,
    )
    local_weight = np.maximum(local_weight, 1e-12)

    scores = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    spatial_sigma = (interpolation_sigma_pixels, interpolation_sigma_pixels, 0.0)
    local_sigma = (ssim_sigma_pixels, ssim_sigma_pixels, 0.0)
    denominator = np.maximum(interpolation_weight[:, :, None], 1e-12)
    support_3d = support_float[:, :, None]
    local_denominator = local_weight[:, :, None]
    for start in range(0, predicted.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, predicted.shape[1])
        width = end - start
        pred_impulses = np.zeros((*shape, width), dtype=np.float64)
        true_impulses = np.zeros((*shape, width), dtype=np.float64)
        np.add.at(pred_impulses, (rows, cols), predicted[:, start:end])
        np.add.at(true_impulses, (rows, cols), target[:, start:end])
        pred_field = gaussian_filter(pred_impulses, spatial_sigma, mode="constant", truncate=3.0)
        true_field = gaussian_filter(true_impulses, spatial_sigma, mode="constant", truncate=3.0)
        pred_field = np.where(support_3d > 0, pred_field / denominator, 0.0)
        true_field = np.where(support_3d > 0, true_field / denominator, 0.0)

        pred_mean = gaussian_filter(
            pred_field * support_3d, local_sigma, mode="constant", truncate=3.5,
        ) / local_denominator
        true_mean = gaussian_filter(
            true_field * support_3d, local_sigma, mode="constant", truncate=3.5,
        ) / local_denominator
        pred_second = gaussian_filter(
            np.square(pred_field) * support_3d, local_sigma,
            mode="constant", truncate=3.5,
        ) / local_denominator
        true_second = gaussian_filter(
            np.square(true_field) * support_3d, local_sigma,
            mode="constant", truncate=3.5,
        ) / local_denominator
        cross = gaussian_filter(
            pred_field * true_field * support_3d, local_sigma,
            mode="constant", truncate=3.5,
        ) / local_denominator
        pred_variance = np.maximum(pred_second - np.square(pred_mean), 0.0)
        true_variance = np.maximum(true_second - np.square(true_mean), 0.0)
        covariance = cross - pred_mean * true_mean

        data_range = np.ptp(target[:, start:end], axis=0)
        eligible = data_range >= 1e-8
        c1 = np.square(0.01 * np.maximum(data_range, 1e-8))[None, None, :]
        c2 = np.square(0.03 * np.maximum(data_range, 1e-8))[None, None, :]
        ssim_map = (
            (2.0 * pred_mean * true_mean + c1) * (2.0 * covariance + c2)
            / np.maximum(
                (np.square(pred_mean) + np.square(true_mean) + c1)
                * (pred_variance + true_variance + c2),
                1e-18,
            )
        )
        center_scores = np.mean(ssim_map[rows, cols, :], axis=0)
        center_scores = np.clip(center_scores, -1.0, 1.0)
        block = scores[start:end]
        block[eligible] = center_scores[eligible]
        scores[start:end] = block

    finite = scores[np.isfinite(scores)]
    return {
        "mean_per_gene_ssim": float(np.mean(finite)) if finite.size else float("nan"),
        "median_per_gene_ssim": float(np.median(finite)) if finite.size else float("nan"),
        "gene_ssim_q25": float(np.quantile(finite, 0.25)) if finite.size else float("nan"),
        "gene_ssim_q75": float(np.quantile(finite, 0.75)) if finite.size else float("nan"),
        "n_eligible_genes": int(finite.size),
        "raster_height": int(shape[0]),
        "raster_width": int(shape[1]),
        "raster_pixel_size_coordinate_units": float(pixel_size),
        "pixels_per_median_neighbor": float(pixels_per_neighbor),
        "interpolation_sigma_pixels": float(interpolation_sigma_pixels),
        "ssim_sigma_pixels": float(ssim_sigma_pixels),
        "background_included_in_mean": False,
    }


def _moran_by_gene(expression: np.ndarray, edges: np.ndarray,
                   *, chunk_size: int = 32) -> np.ndarray:
    n_spots, n_genes = expression.shape
    left, right = edges[:, 0], edges[:, 1]
    out = np.full(n_genes, np.nan, dtype=np.float64)
    s0 = float(2 * len(edges))
    for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        centered = expression[:, start:end] - expression[:, start:end].mean(axis=0)
        denominator = np.einsum("ng,ng->g", centered, centered)
        numerator = 2.0 * np.einsum("eg,eg->g", centered[left], centered[right])
        valid = denominator > 1e-12
        block = np.full(end - start, np.nan, dtype=np.float64)
        block[valid] = (n_spots / s0) * numerator[valid] / denominator[valid]
        out[start:end] = block
    return out


def moran_i_agreement(predicted: np.ndarray, target: np.ndarray, coords: np.ndarray,
                      *, k_neighbors: int = 6) -> dict[str, float]:
    """Per-gene Moran's-I preservation on a common held-out spatial graph."""
    predicted, target = _arrays(predicted, target)
    edges = undirected_knn_edges(coords, k_neighbors=k_neighbors)
    target_i = _moran_by_gene(target, edges)
    predicted_i = _moran_by_gene(predicted, edges)
    eligible = np.isfinite(target_i)
    # Keep truth-variable genes whose prediction is constant; zero Moran's I
    # expresses failure while dropping them would reward collapse.
    predicted_i = np.where(eligible & ~np.isfinite(predicted_i), 0.0, predicted_i)
    paired = eligible & np.isfinite(predicted_i)
    if not paired.any():
        return {
            "moran_i_pcc": float("nan"), "moran_i_mae": float("nan"),
            "target_mean_moran_i": float("nan"),
            "predicted_mean_moran_i": float("nan"), "n_eligible_genes": 0,
            "n_edges": int(len(edges)),
        }
    return {
        "moran_i_pcc": _correlation(predicted_i[paired], target_i[paired]),
        "moran_i_mae": float(np.mean(np.abs(predicted_i[paired] - target_i[paired]))),
        "target_mean_moran_i": float(np.mean(target_i[paired])),
        "predicted_mean_moran_i": float(np.mean(predicted_i[paired])),
        "n_eligible_genes": int(paired.sum()),
        "n_edges": int(len(edges)),
    }


def signed_gradient_agreement(
    predicted: np.ndarray, target: np.ndarray, coords: np.ndarray,
    per_gene_scale: np.ndarray, *, k_neighbors: int = 6,
    nontrivial_threshold: float = 0.25, chunk_size: int = 128,
) -> dict[str, float]:
    """Signed edge-gradient agreement in training-standard-deviation units."""
    predicted, target = _arrays(predicted, target)
    scale = np.asarray(per_gene_scale, dtype=np.float64)
    if scale.shape != (predicted.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("per_gene_scale must be finite, positive, and [n_genes]")
    if nontrivial_threshold < 0:
        raise ValueError("nontrivial_threshold must be non-negative")
    edges = undirected_knn_edges(coords, k_neighbors=k_neighbors)
    left, right = edges[:, 0], edges[:, 1]

    total_n = 0
    sum_pred = sum_true = sum_pred2 = sum_true2 = sum_cross = 0.0
    squared_error = absolute_error = predicted_energy = target_energy = 0.0
    sign_matches = sign_total = 0
    per_gene_pcc: list[np.ndarray] = []
    # Edge indexing expands a [n_spots, genes] block into
    # [n_edges, genes]. Keep this deliberately small for large Visium slides.
    chunk_size = min(int(chunk_size), 32)
    for start in range(0, predicted.shape[1], chunk_size):
        end = min(start + chunk_size, predicted.shape[1])
        block_scale = scale[start:end][None, :]
        pred_delta = (predicted[right, start:end] - predicted[left, start:end]) / block_scale
        true_delta = (target[right, start:end] - target[left, start:end]) / block_scale
        total_n += int(pred_delta.size)
        sum_pred += float(pred_delta.sum())
        sum_true += float(true_delta.sum())
        sum_pred2 += float(np.square(pred_delta).sum())
        sum_true2 += float(np.square(true_delta).sum())
        sum_cross += float((pred_delta * true_delta).sum())
        error = pred_delta - true_delta
        squared_error += float(np.square(error).sum())
        absolute_error += float(np.abs(error).sum())
        predicted_energy += float(np.square(pred_delta).sum())
        target_energy += float(np.square(true_delta).sum())
        nontrivial = np.abs(true_delta) >= nontrivial_threshold
        sign_matches += int((np.signbit(pred_delta[nontrivial]) == np.signbit(true_delta[nontrivial])).sum())
        sign_total += int(nontrivial.sum())
        per_gene_pcc.append(pearson_per_gene(pred_delta, true_delta))

    covariance = sum_cross - sum_pred * sum_true / total_n
    pred_ss = sum_pred2 - sum_pred * sum_pred / total_n
    true_ss = sum_true2 - sum_true * sum_true / total_n
    global_pcc = (
        float(np.clip(covariance / np.sqrt(pred_ss * true_ss), -1.0, 1.0))
        if pred_ss > 1e-12 and true_ss > 1e-12 else 0.0
    )
    gene_pcc = np.concatenate(per_gene_pcc)
    gene_pcc = gene_pcc[np.isfinite(gene_pcc)]
    return {
        "signed_gradient_pcc": global_pcc,
        "mean_per_gene_gradient_pcc": float(gene_pcc.mean()) if gene_pcc.size else float("nan"),
        "standardized_gradient_rmse": float(np.sqrt(squared_error / total_n)),
        "standardized_gradient_mae": float(absolute_error / total_n),
        "gradient_energy_ratio": (
            float(predicted_energy / target_energy) if target_energy > 1e-12 else float("nan")
        ),
        "sign_agreement_nontrivial": (
            float(sign_matches / sign_total) if sign_total else float("nan")
        ),
        "nontrivial_threshold_training_sd": float(nontrivial_threshold),
        "n_nontrivial_differences": int(sign_total),
        "n_edges": int(len(edges)),
    }


def structured_field_metrics(
    predicted: np.ndarray, target: np.ndarray, coords: np.ndarray,
    per_gene_scale: np.ndarray, *, panel_indices: dict[str, np.ndarray] | None = None,
    local_k: int = 6, wide_k: int = 18, nontrivial_threshold: float = 0.25,
) -> dict[str, Any]:
    """Complete whole-slide diagnostic suite for one deterministic field."""
    predicted, target = _arrays(predicted, target)
    coords = np.asarray(coords, dtype=np.float64)
    scale = np.asarray(per_gene_scale, dtype=np.float64)
    if coords.shape != (predicted.shape[0], 2):
        raise ValueError("coords must align with expression rows")
    if scale.shape != (predicted.shape[1],):
        raise ValueError("per_gene_scale must align with expression columns")

    selections = {"all_genes": np.arange(predicted.shape[1], dtype=np.int64)}
    selections.update(panel_indices or {})
    panels: dict[str, Any] = {}
    for name, indices in selections.items():
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size < 1:
            raise ValueError(f"structured-field panel {name!r} is empty or invalid")
        if np.any(indices < 0) or np.any(indices >= predicted.shape[1]):
            raise ValueError(f"structured-field panel {name!r} has out-of-range genes")
        panel_pred = predicted[:, indices]
        panel_true = target[:, indices]
        panel_scale = scale[indices]
        entry: dict[str, Any] = {
            "spot_profile": spot_profile_agreement(panel_pred, panel_true),
            "spatial_ssim": spatial_ssim_agreement(
                panel_pred, panel_true, coords,
            ),
            "moran_local": moran_i_agreement(
                panel_pred, panel_true, coords, k_neighbors=local_k,
            ),
            "gradient_local": signed_gradient_agreement(
                panel_pred, panel_true, coords, panel_scale,
                k_neighbors=local_k, nontrivial_threshold=nontrivial_threshold,
            ),
            "gradient_wide": signed_gradient_agreement(
                panel_pred, panel_true, coords, panel_scale,
                k_neighbors=wide_k, nontrivial_threshold=nontrivial_threshold,
            ),
        }
        if indices.size >= 2 and name != "all_genes":
            entry["coexpression"] = coexpression_agreement(panel_pred, panel_true)
        panels[name] = entry
    return {
        "version": 2,
        "scope": "one held_out_whole_slide_every_spot_exactly_once",
        "gradient_units": "training_only_per_gene_standard_deviation",
        "local_k": int(local_k),
        "wide_k": int(wide_k),
        "panels": panels,
    }


def noise_ceiling_adjusted_pcc(
    predicted: np.ndarray, target: np.ndarray, gene_names: list[str],
    ceiling_by_gene: dict[str, float], *,
    panel_indices: dict[str, np.ndarray] | None = None,
    minimum_ceiling: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Report observed PCC as a fraction of count-split measurability.

    The ceiling artifact is external and measured from raw counts; this
    function only aligns it to the immutable output vocabulary.  Ratios are
    not clipped: values above one flag estimation noise or ceiling mismatch
    instead of being silently made to look perfect.
    """
    predicted, target = _arrays(predicted, target)
    if len(gene_names) != predicted.shape[1]:
        raise ValueError("gene_names must align with expression columns")
    ceiling = np.asarray(
        [ceiling_by_gene.get(str(gene), np.nan) for gene in gene_names],
        dtype=np.float64,
    )
    selections = {"all_genes": np.arange(predicted.shape[1], dtype=np.int64)}
    selections.update(panel_indices or {})
    observed = pearson_per_gene(predicted, target)
    result = {}
    for name, indices in selections.items():
        indices = np.asarray(indices, dtype=np.int64)
        fraction = normalized_score(
            observed[indices], ceiling[indices], minimum_ceiling=minimum_ceiling,
        )
        usable = np.isfinite(fraction)
        finite_ceiling = np.isfinite(ceiling[indices])
        result[name] = {
            "mean_observed_pcc": float(np.nanmean(observed[indices])),
            "mean_count_split_ceiling": (
                float(np.mean(ceiling[indices][finite_ceiling]))
                if finite_ceiling.any() else float("nan")
            ),
            "mean_fraction_achievable": (
                float(np.mean(fraction[usable])) if usable.any() else float("nan")
            ),
            "median_fraction_achievable": (
                float(np.median(fraction[usable])) if usable.any() else float("nan")
            ),
            "minimum_usable_ceiling": float(minimum_ceiling),
            "n_usable_genes": int(usable.sum()),
            "n_genes_with_ceiling": int(finite_ceiling.sum()),
        }
    return result
