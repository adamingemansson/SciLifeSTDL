#!/usr/bin/env python3
"""Gen4 smoke-only launcher -- GEN4_CONTRACT.md section 12.

Constructs each of the four arms' conditioner AND flow model at tiny,
CPU-fast dims, on synthetic (non-HEST) data, using deterministic stub
encoders (no real UNI2/scFoundation/STPath package or weights). Runs
exactly one real optimizer step per model and one real
`sample_predictive_distribution` call. Never touches a GPU, never
downloads or loads a real checkpoint, and NEVER starts a full training run
-- there is no loop here beyond the fixed one-step smoke checks below, by
construction, not by a flag a caller could accidentally leave on.

    python -m gen3_multiscale.scripts.gen4_smoke_launcher
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldTargets
from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.flow import Gen4ResidualFlowModel
from gen3_multiscale.gen4.inputs import Gen4SpatialFieldInputs, validate_gen4_spatial_field_example
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis

N_GENES, GEX_DIM, IMAGE_DIM, CONTEXT_DIM = 6, 4, 8, 5
MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4)


def _synthetic_example(seed: int = 0, gex_context_dim: int | None = None):
    rng = np.random.default_rng(seed)
    n = 15
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n), np.arange(lo, lo + n))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    dist = np.linalg.norm(grid, axis=1)
    observed_coords, query_coords = grid[dist > 2.5], grid[dist <= 2.5]
    n_observed, n_query = observed_coords.shape[0], query_coords.shape[0]
    result = extract_boundary_and_local_context(observed_coords, query_coords, k_neighbors=6, local_k=6, max_rings=3)
    context_embedding = rng.normal(size=(n_observed, gex_context_dim)).astype(np.float32) if gex_context_dim else None
    inputs = Gen4SpatialFieldInputs(
        sample_id="smoke", patient_id="smoke-patient",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed_coords.astype(np.float32), query_coords=query_coords.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, N_GENES)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, IMAGE_DIM)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=result.query_local_neighbor_idx, boundary_idx=result.boundary_idx,
        boundary_ring=result.boundary_ring, query_depth_to_boundary=result.query_depth_to_boundary,
        context_gex_embedding=context_embedding,
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, N_GENES)).astype(np.float32))
    validate_gen4_spatial_field_example(inputs, targets)
    return inputs, targets


class _StubSlideEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.checkpoint_sha256 = "smoke" * 12
        self.proj = nn.Linear(IMAGE_DIM, 16)

    def forward(self, tile_features, tile_coords, cache_namespace):
        return self.proj(tile_features.mean(dim=0, keepdim=True)).squeeze(0)


class _StubSTPathEncoder(nn.Module):
    """Deterministic stand-in for Gen4STPathContextEncoder -- same
    `encode_context_only` signature (no query parameter at all), a real
    trainable linear layer over concatenated coordinate/expression/image
    features, no real `stpath` package or weights. Mirrors
    `tests/_gen4_fixtures.py::Gen4STPathStub`."""

    def __init__(self, n_genes: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(2 + n_genes + 1, hidden_dim)

    def encode_context_only(self, context_coords, context_expression, context_image_features, context_image_available=None):
        image_summary = context_image_features.mean(dim=-1, keepdim=True)
        combined = torch.cat([context_coords, context_expression, image_summary], dim=-1)
        return self.proj(combined)


def _with_wsi_context(inputs, wsi_tile_feature_provenance=None):
    rng = np.random.default_rng(2)
    n_tiles, bound = 12, 10.0
    return dataclasses.replace(
        inputs,
        wsi_tile_longnet_coords=rng.uniform(10_000.0, 20_000.0, size=(n_tiles, 2)).astype(np.float32),
        wsi_tile_regional_coords=rng.uniform(-bound + 0.5, bound - 0.5, size=(n_tiles, 2)).astype(np.float32),
        wsi_tile_features=rng.normal(size=(n_tiles, IMAGE_DIM)).astype(np.float32),
        full_slide_coord_bounds=(-bound, bound, -bound, bound), slide_cache_namespace="smoke-slide",
        wsi_tile_feature_provenance=wsi_tile_feature_provenance,
    )


_ARM_SETUP = {
    "gen4a": dict(gex_feature_source="weighted_linear", image_feature_source="precomputed", global_context_source="uni2_pool", gex_context_dim=None),
    "gen4b": dict(gex_feature_source="frozen_context", image_feature_source="precomputed", global_context_source="gigapath", gex_context_dim=CONTEXT_DIM),
    "gen4c": dict(gex_feature_source="frozen_context", image_feature_source="precomputed", global_context_source="uni2_pool", gex_context_dim=CONTEXT_DIM),
    "gen4d": dict(gex_feature_source="stpath_joint", image_feature_source="stpath_context", global_context_source="none", gex_context_dim=None),
}


def _build_conditioner(arm: str) -> Gen4Conditioner:
    setup = _ARM_SETUP[arm]
    kwargs = dict(
        n_genes=N_GENES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        gex_feature_source=setup["gex_feature_source"], image_feature_source=setup["image_feature_source"],
        global_context_source=setup["global_context_source"],
        use_regional_he=setup["global_context_source"] != "none", **MODEL_KWARGS,
    )
    if setup["gex_context_dim"]:
        kwargs["gex_context_embedding_dim"] = setup["gex_context_dim"]
    if setup["global_context_source"] == "gigapath":
        kwargs["slide_encoder"] = _StubSlideEncoder()
        kwargs["gigapath_checkpoint_sha256"] = "smoke" * 12
        kwargs["global_slide_dim"] = 16
    elif setup["global_context_source"] == "uni2_pool":
        kwargs["uni2_global_pool"] = MaskAwareCoordinateAttentionPool(tile_feature_dim=IMAGE_DIM, output_dim=16, hidden_dim=16, n_heads=2)
        kwargs["global_slide_dim"] = 16
    if setup["image_feature_source"] == "stpath_context":
        kwargs["stpath_encoder"] = _StubSTPathEncoder(n_genes=N_GENES, hidden_dim=IMAGE_DIM)
    return Gen4Conditioner(**kwargs)


def _build_flow(arm: str, gene_basis, gene_names: list[str]) -> Gen4ResidualFlowModel:
    setup = _ARM_SETUP[arm]
    kwargs = dict(
        n_genes=N_GENES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM, gene_basis=gene_basis, gene_names=gene_names,
        gex_feature_source=setup["gex_feature_source"], image_feature_source=setup["image_feature_source"],
        global_context_source=setup["global_context_source"],
        use_regional_he=setup["global_context_source"] != "none", n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **MODEL_KWARGS,
    )
    if setup["gex_context_dim"]:
        kwargs["gex_context_embedding_dim"] = setup["gex_context_dim"]
    if setup["global_context_source"] == "gigapath":
        kwargs["slide_encoder"] = _StubSlideEncoder()
        kwargs["gigapath_checkpoint_sha256"] = "smoke" * 12
        kwargs["global_slide_dim"] = 16
    elif setup["global_context_source"] == "uni2_pool":
        kwargs["uni2_global_pool"] = MaskAwareCoordinateAttentionPool(tile_feature_dim=IMAGE_DIM, output_dim=16, hidden_dim=16, n_heads=2)
        kwargs["global_slide_dim"] = 16
    if setup["image_feature_source"] == "stpath_context":
        kwargs["stpath_encoder"] = _StubSTPathEncoder(n_genes=N_GENES, hidden_dim=IMAGE_DIM)
    return Gen4ResidualFlowModel(**kwargs)


def smoke_one_arm(arm: str) -> dict:
    setup = _ARM_SETUP[arm]
    inputs, targets = _synthetic_example(gex_context_dim=setup["gex_context_dim"])
    if setup["global_context_source"] != "none":
        provenance = "uni2" if setup["global_context_source"] == "uni2_pool" else None
        inputs = _with_wsi_context(inputs, wsi_tile_feature_provenance=provenance)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)

    torch.manual_seed(0)
    conditioner = _build_conditioner(arm)
    optimizer = torch.optim.Adam(conditioner.parameters(), lr=1e-3)
    out = conditioner(inputs)
    loss = torch.nn.functional.mse_loss(out["expression"], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    conditioner_report = {"one_step_conditioner_loss": float(loss.item())}

    gene_names = [f"g{i}" for i in range(N_GENES)]
    residuals = np.random.default_rng(0).normal(size=(20, N_GENES))
    gene_basis = fit_gene_residual_basis(residuals, gene_names, rank=3)
    flow_model = _build_flow(arm, gene_basis, gene_names)
    flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)
    flow_out = flow_model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    flow_optimizer.zero_grad()
    flow_out["flow_loss"].backward()
    flow_optimizer.step()
    eval_out = flow_model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(1))

    return {
        "arm": arm,
        **conditioner_report,
        "one_step_flow_loss": float(flow_out["flow_loss"].item()),
        "eval_predictive_mean_shape": list(eval_out["predictive_mean"].shape),
        "conditioner_n_parameters": sum(p.numel() for p in conditioner.parameters()),
        "flow_n_parameters": sum(p.numel() for p in flow_model.parameters()),
    }


def main() -> None:
    reports = [smoke_one_arm(arm) for arm in ("gen4a", "gen4b", "gen4c", "gen4d")]
    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
