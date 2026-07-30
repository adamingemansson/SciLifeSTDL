"""Shared Gen4 test fixtures. NOT a test file itself (no test_/_test
suffix, never collected by pytest). Mirrors _step6_fixtures.py's discipline:
tiny, deterministic, CPU-only, real code paths (no mocking of Gen4's own
logic) with lightweight stand-ins ONLY for the three external, unavailable
pretrained encoders (UNI2, scFoundation, STPath) -- the same role
stub_gigapath already plays for GigaPath in _step6_fixtures.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldTargets, validate_spatial_field_example
from gen3_multiscale.gen4.inputs import Gen4SpatialFieldInputs, validate_gen4_spatial_field_example
from gen3_multiscale.gen4.providers import EncoderIdentity

GEN4_MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4)


def synthetic_gen4_inputs(
    *, n_genes: int = 6, gex_dim: int = 4, image_dim: int = 8, gex_context_dim: int | None = None, seed: int = 0,
) -> tuple[Gen4SpatialFieldInputs, SpatialFieldTargets]:
    """Hand-built square-grid example -- same construction
    test_architectures.py's own `_synthetic_inputs` uses, wrapped into a
    Gen4SpatialFieldInputs (context_gex_embedding populated only when
    gex_context_dim is given)."""
    rng = np.random.default_rng(seed)
    n = 15
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n), np.arange(lo, lo + n))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    dist = np.linalg.norm(grid, axis=1)
    observed_coords, query_coords = grid[dist > 2.5], grid[dist <= 2.5]
    n_observed, n_query = observed_coords.shape[0], query_coords.shape[0]

    result = extract_boundary_and_local_context(observed_coords, query_coords, k_neighbors=6, local_k=6, max_rings=3)

    context_embedding = None
    if gex_context_dim is not None:
        context_embedding = rng.normal(size=(n_observed, gex_context_dim)).astype(np.float32)

    inputs = Gen4SpatialFieldInputs(
        sample_id="s1", patient_id="p1",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed_coords.astype(np.float32),
        query_coords=query_coords.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, image_dim)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=result.query_local_neighbor_idx,
        boundary_idx=result.boundary_idx,
        boundary_ring=result.boundary_ring,
        query_depth_to_boundary=result.query_depth_to_boundary,
        context_gex_embedding=context_embedding,
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    validate_gen4_spatial_field_example(inputs, targets)
    return inputs, targets


def with_synthetic_wsi_context(
    inputs: Gen4SpatialFieldInputs, image_dim: int, n_tiles: int = 12, grid_bound: float = 10.0, seed: int = 1,
    wsi_tile_feature_provenance: str | None = None,
):
    """`wsi_tile_feature_provenance` must be explicitly passed as "uni2" by
    callers exercising `global_context_source="uni2_pool"` -- real
    (non-test) data never has this set, since no real UNI2 dense-WSI cache
    builder exists yet (Codex audit finding #4; see
    Gen4SpatialFieldInputs.wsi_tile_feature_provenance's docstring)."""
    rng = np.random.default_rng(seed)
    longnet_coords = rng.uniform(10_000.0, 20_000.0, size=(n_tiles, 2)).astype(np.float32)
    regional_coords = rng.uniform(-grid_bound + 0.5, grid_bound - 0.5, size=(n_tiles, 2)).astype(np.float32)
    features = rng.normal(size=(n_tiles, image_dim)).astype(np.float32)
    return dataclasses.replace(
        inputs,
        wsi_tile_longnet_coords=longnet_coords, wsi_tile_regional_coords=regional_coords,
        wsi_tile_features=features, full_slide_coord_bounds=(-grid_bound, grid_bound, -grid_bound, grid_bound),
        slide_cache_namespace="gen4-unit-test-slide-abc123", wsi_tile_feature_provenance=wsi_tile_feature_provenance,
    )


def with_synthetic_uni2_features(inputs: Gen4SpatialFieldInputs, image_dim: int, seed: int = 3):
    """Arm 4 (hybrid): a genuinely separate per-spot UNI2 array from
    whatever `observed_gigapath_features` holds for this arm (STPath's own
    GigaPath-shaped tokenizer input) -- see
    `Gen4SpatialFieldInputs.observed_uni2_features`'s docstring."""
    n_observed = inputs.observed_coords.shape[0]
    rng = np.random.default_rng(seed)
    return dataclasses.replace(
        inputs, observed_uni2_features=rng.normal(size=(n_observed, image_dim)).astype(np.float32),
    )


class StubUNI2Encoder:
    """Deterministic ImageContextProvider stand-in -- no real UNI2/timm
    dependency. `encode_available_patches` is a fixed linear function of
    each patch's own mean pixel value, matching stub_gigapath's own
    "deterministic, order-independent per-row function" discipline."""

    def __init__(self, output_dim: int = 8):
        self.output_dim = output_dim
        self.identity = EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256="stub" * 16, pinned_revision="0" * 40,
            package_version="stub-0.0.0", preprocessing_spec="stub_uni2_v1", output_dim=output_dim,
        )

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        means = patches.astype(np.float32).reshape(patches.shape[0], -1).mean(axis=1, keepdims=True)
        return np.tile(means, (1, self.output_dim)).astype(np.float32)


class StubSCFoundationEncoder:
    """Deterministic GexContextProvider stand-in -- row-independent linear
    function of raw expression."""

    def __init__(self, gene_names: list[str], output_dim: int = 12):
        self.gene_names = tuple(gene_names)
        self.output_dim = output_dim
        rng = np.random.default_rng(42)
        self._weight = rng.normal(size=(len(gene_names), output_dim)).astype(np.float32)
        self.identity = EncoderIdentity(
            encoder_name="scfoundation", checkpoint_sha256="stub" * 16, pinned_revision="stubvocab" * 4,
            package_version="stub-0.0.0", preprocessing_spec="stub_scfoundation_v1", output_dim=output_dim,
        )

    def encode_rows(self, expression: np.ndarray) -> np.ndarray:
        return (np.asarray(expression, dtype=np.float32) @ self._weight).astype(np.float32)


class Gen4STPathStub(nn.Module):
    """Deterministic stand-in for Gen4STPathContextEncoder -- same
    `encode_context_only` signature (no query parameter at all), a fixed
    trainable linear layer over concatenated coordinate/expression-summary/
    image-summary features, no real `stpath` package or weights."""

    def __init__(self, n_genes: int, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(2 + n_genes + 1, hidden_dim)

    def encode_context_only(
        self, context_coords: torch.Tensor, context_expression: torch.Tensor,
        context_image_features: torch.Tensor, context_image_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_summary = context_image_features.mean(dim=-1, keepdim=True)
        combined = torch.cat([context_coords, context_expression, image_summary], dim=-1)
        return self.proj(combined)
