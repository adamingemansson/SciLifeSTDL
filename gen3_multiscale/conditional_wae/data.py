"""Adapters from verified Gen3 sample data to the supervisor-task schema."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from gen3_multiscale.conditional_wae.inputs import (
    FullImageExpressionInputs,
    validate_full_image_expression_inputs,
)
from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context


def normalize_slide_coordinates(coords: np.ndarray) -> np.ndarray:
    """Match Gen3's slide-centred, median-spot-spacing coordinate frame."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError("coords must contain at least two [x, y] rows")
    if not np.isfinite(coords).all() or np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("coords must be finite and unique")
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    scale = max(float(np.median(distances[:, 1])), 1e-6)
    return ((coords - coords.mean(axis=0)) / scale).astype(np.float32)


def _dense_expression(matrix) -> np.ndarray:
    result = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("sample expression must be a finite rank-2 matrix")
    return result


def padded_spatial_adjacency(adjacency) -> tuple[np.ndarray, np.ndarray]:
    if not adjacency or any(len(neighbors) == 0 for neighbors in adjacency):
        raise ValueError("every slide row must have at least one spatial neighbor")
    width = max(len(neighbors) for neighbors in adjacency)
    indices = np.zeros((len(adjacency), width), dtype=np.int64)
    mask = np.zeros((len(adjacency), width), dtype=bool)
    for row, neighbors in enumerate(adjacency):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        indices[row, :len(neighbors)] = neighbors
        mask[row, :len(neighbors)] = True
    return indices, mask


def build_conditional_wae_example(
    sample,
    *,
    query_indices: np.ndarray | list[int] | None,
    include_observed_gex: bool,
    observed_expression_indices: np.ndarray | list[int] | None = None,
    normalized_coords: np.ndarray | None = None,
    padded_adjacency: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[FullImageExpressionInputs, np.ndarray]:
    """Build Task I or II without masking the query-region image.

    ``sample`` follows ``training.gen3_dataset.Gen3SampleData``. Its verified
    full spot-feature cache is reused directly. For Task II only, selected
    measured GEX rows outside ``query_indices`` are exposed compactly. The
    returned target contains query rows only and never lives in the input.
    """
    features = np.asarray(sample.precomputed_spot_features, dtype=np.float32)
    image_available = np.asarray(sample.image_source_available, dtype=bool)
    coords = np.asarray(sample.full_sample_coords, dtype=np.float64)
    expression_matrix = sample.adata.X
    if len(expression_matrix.shape) != 2:
        raise ValueError("sample expression must be a rank-2 matrix")
    n, n_genes = expression_matrix.shape
    if features.shape[0] != n or coords.shape != (n, 2) or image_available.shape != (n,):
        raise ValueError("sample features, coordinates, availability and expression are not row-aligned")
    if query_indices is None:
        query = np.arange(n, dtype=np.int64)
    else:
        query = np.asarray(query_indices, dtype=np.int64)
    if query.ndim != 1 or query.size == 0 or np.unique(query).size != query.size:
        raise ValueError("query_indices must be a non-empty one-dimensional unique index set")
    if query.min() < 0 or query.max() >= n:
        raise ValueError("query_indices contains an out-of-range row")
    query_mask = np.zeros(n, dtype=bool)
    query_mask[query] = True

    observed_expression = None
    observed_expression_indices = None
    expression_available = None
    if include_observed_gex:
        if query_mask.all():
            raise ValueError("Task II requires at least one surrounding observed-GEX row")
        if observed_expression_indices is None:
            observed_expression_indices = np.flatnonzero(~query_mask).astype(np.int64)
        else:
            observed_expression_indices = np.asarray(observed_expression_indices, dtype=np.int64)
            if (
                observed_expression_indices.ndim != 1
                or observed_expression_indices.size == 0
                or np.unique(observed_expression_indices).size != observed_expression_indices.size
                or observed_expression_indices.min() < 0
                or observed_expression_indices.max() >= n
                or query_mask[observed_expression_indices].any()
            ):
                raise ValueError(
                    "observed_expression_indices must be unique, in-range, non-query rows"
                )
        observed_expression = _dense_expression(
            expression_matrix[observed_expression_indices],
        )
        expression_available = np.zeros(n, dtype=bool)
        expression_available[observed_expression_indices] = True

    target = _dense_expression(expression_matrix[query])
    if target.shape != (query.size, n_genes):
        raise ValueError("query expression rows were not extracted in the requested order")
    if normalized_coords is None:
        normalized_coords = normalize_slide_coordinates(coords)
    else:
        normalized_coords = np.asarray(normalized_coords, dtype=np.float32)
        if normalized_coords.shape != (n, 2) or not np.isfinite(normalized_coords).all():
            raise ValueError("normalized_coords must be finite [N, 2]")

    inputs = FullImageExpressionInputs(
        sample_id=str(sample.sample_id),
        image_features=features,
        coords=normalized_coords,
        image_available=image_available,
        query_mask=query_mask,
        observed_expression=observed_expression,
        observed_expression_indices=observed_expression_indices,
        expression_available=expression_available,
        neighbor_indices=(padded_adjacency[0] if padded_adjacency is not None else None),
        neighbor_mask=(padded_adjacency[1] if padded_adjacency is not None else None),
    )
    validate_full_image_expression_inputs(inputs)
    return inputs, target


class ConditionalWAEMaskedGEXDataset:
    """View an existing verified Gen3 mask schedule under supervisor semantics.

    The same query barcodes/mask strata are reused for comparability, but the
    Gen3 builder is deliberately not called because it removes query-region
    H&E. This view keeps the complete verified spot-image feature matrix and
    masks only GEX.
    """

    def __init__(self, gen3_dataset, *, include_observed_gex: bool):
        self.gen3_dataset = gen3_dataset
        self.include_observed_gex = bool(include_observed_gex)
        self._barcode_positions = {
            sample_id: {
                str(barcode): row
                for row, barcode in enumerate(np.asarray(sample.adata.obs_names, dtype=str))
            }
            for sample_id, sample in gen3_dataset.samples.items()
        }
        self._normalized_coords = {
            sample_id: normalize_slide_coordinates(sample.full_sample_coords)
            for sample_id, sample in gen3_dataset.samples.items()
        }
        self._padded_adjacency = {
            sample_id: padded_spatial_adjacency(sample.spatial_adjacency)
            for sample_id, sample in gen3_dataset.samples.items()
        }

    def __len__(self) -> int:
        return len(self.gen3_dataset)

    def __getitem__(self, index: int):
        base = self.gen3_dataset
        item = base._items[index % len(base._items)]
        sample = base.samples[item.sample_id]
        _context_barcodes, query_barcodes = base._resolve_barcodes(item)
        position = self._barcode_positions[item.sample_id]
        try:
            context_indices = np.asarray([position[str(barcode)] for barcode in _context_barcodes])
            query_indices = np.asarray([position[str(barcode)] for barcode in query_barcodes])
        except KeyError as exc:
            raise ValueError(
                f"{item.sample_id}: mask references unknown query barcode {exc.args[0]!r}"
            ) from exc
        observed_expression_indices = None
        if self.include_observed_gex:
            boundary = extract_boundary_and_local_context(
                sample.full_sample_coords[context_indices],
                sample.full_sample_coords[query_indices],
                k_neighbors=base.k_neighbors,
                local_k=base.local_k,
                max_rings=base.max_rings,
                max_boundary_size=base.max_boundary_size,
                full_adjacency=sample.spatial_adjacency,
                observed_full_idx=context_indices,
                query_full_idx=query_indices,
            )
            selected_context_positions = np.unique(np.concatenate([
                boundary.boundary_idx,
                boundary.query_local_neighbor_idx.reshape(-1),
            ])).astype(np.int64)
            observed_expression_indices = context_indices[selected_context_positions]
        inputs, target = build_conditional_wae_example(
            sample, query_indices=query_indices,
            include_observed_gex=self.include_observed_gex,
            observed_expression_indices=observed_expression_indices,
            normalized_coords=self._normalized_coords[item.sample_id],
            padded_adjacency=self._padded_adjacency[item.sample_id],
        )
        identity = base.item_identity(index)
        return inputs, target, identity


def conditional_wae_identity_collate(batch):
    if len(batch) != 1:
        raise ValueError("conditional WAE uses batch_size=1 for ragged slide/mask items")
    return batch[0]
