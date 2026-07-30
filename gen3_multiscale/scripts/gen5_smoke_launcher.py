#!/usr/bin/env python3
"""Gen5 smoke-only launcher -- GEN5_CONTRACT.md section 7 (gate 9), 12.

Trains a tiny expression autoencoder on synthetic data, then constructs
each of the four arms' Gen5LatentFlowModel at tiny, CPU-fast dims with
deterministic stub encoders, runs one real optimizer step, and one real
`sample_predictive_distribution` call. Never touches a GPU, never
downloads or loads a real checkpoint, never starts a full training run.

Reports peak CPU RSS via `resource.getrusage` as a stand-in for gate 9's
real CUDA memory smoke test -- this environment has no GPU (see
GEN5_CONTRACT.md section 9's honest gap); a real
`torch.cuda.max_memory_allocated()` profile must be captured separately
on real hardware.

    python -m gen3_multiscale.scripts.gen5_smoke_launcher
"""
from __future__ import annotations

import dataclasses
import resource

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldTargets
from gen3_multiscale.gen4.inputs import Gen4SpatialFieldInputs, validate_gen4_spatial_field_example
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder, evaluate_autoencoder_reconstruction
from gen3_multiscale.gen5.autoencoder_training import train_expression_autoencoder
from gen3_multiscale.gen5.latent_flow import Gen5LatentFlowModel

N_GENES, GEX_DIM, IMAGE_DIM, CONTEXT_DIM, LATENT_DIM = 6, 4, 8, 5, 12
MODEL_KWARGS = dict(hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4)
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]


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
    `encode_context_only` signature. Mirrors
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
    "gen5a": dict(gex_feature_source="weighted_linear", image_feature_source="precomputed", global_context_source="uni2_pool", gex_context_dim=None),
    "gen5b": dict(gex_feature_source="frozen_context", image_feature_source="precomputed", global_context_source="gigapath", gex_context_dim=CONTEXT_DIM),
    "gen5c": dict(gex_feature_source="frozen_context", image_feature_source="precomputed", global_context_source="uni2_pool", gex_context_dim=CONTEXT_DIM),
    "gen5d": dict(gex_feature_source="stpath_joint", image_feature_source="stpath_context", global_context_source="none", gex_context_dim=None),
}


def _build_flow_model(arm: str, autoencoder: ExpressionAutoencoder) -> Gen5LatentFlowModel:
    setup = _ARM_SETUP[arm]
    kwargs = dict(
        n_genes=N_GENES, gene_names=GENE_NAMES, gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        autoencoder=autoencoder, gex_feature_source=setup["gex_feature_source"],
        image_feature_source=setup["image_feature_source"],
        global_context_source=setup["global_context_source"], use_regional_he=setup["global_context_source"] != "none",
        n_flow_blocks=1, n_flow_samples=2, n_ode_steps=2, **MODEL_KWARGS,
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
    return Gen5LatentFlowModel(**kwargs)


def smoke_autoencoder() -> dict:
    rng = np.random.default_rng(0)
    train_expression = rng.normal(size=(64, N_GENES)).astype(np.float32)
    val_expression = rng.normal(size=(16, N_GENES)).astype(np.float32)
    autoencoder, training_report = train_expression_autoencoder(
        train_expression, GENE_NAMES, latent_dim=LATENT_DIM, hidden_dim=32, n_epochs=5, batch_size=16, device="cpu",
    )
    train_report = evaluate_autoencoder_reconstruction(autoencoder, train_expression, GENE_NAMES)
    val_report = evaluate_autoencoder_reconstruction(autoencoder, val_expression, GENE_NAMES)
    return {
        "autoencoder": autoencoder,
        "final_train_loss": training_report.final_train_loss,
        "train_reconstruction_pcc_mean": train_report.pcc_mean, "train_reconstruction_rmse": train_report.rmse,
        "val_reconstruction_pcc_mean": val_report.pcc_mean, "val_reconstruction_rmse": val_report.rmse,
    }


def smoke_one_arm(arm: str, autoencoder: ExpressionAutoencoder) -> dict:
    setup = _ARM_SETUP[arm]
    inputs, targets = _synthetic_example(gex_context_dim=setup["gex_context_dim"])
    if setup["global_context_source"] != "none":
        provenance = "uni2" if setup["global_context_source"] == "uni2_pool" else None
        inputs = _with_wsi_context(inputs, wsi_tile_feature_provenance=provenance)
    target = torch.as_tensor(targets.query_expression, dtype=torch.float32)

    torch.manual_seed(0)
    model = _build_flow_model(arm, autoencoder)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    out = model.compute_losses(inputs, target, generator=torch.Generator().manual_seed(0))
    optimizer.zero_grad()
    out["flow_loss"].backward()
    optimizer.step()

    eval_out = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(1))
    return {
        "arm": arm,
        "one_step_flow_loss": float(out["flow_loss"].item()),
        "predictive_mean_shape": list(eval_out["expression"].shape),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main() -> None:
    autoencoder_report = smoke_autoencoder()
    print({k: v for k, v in autoencoder_report.items() if k != "autoencoder"})
    autoencoder = autoencoder_report["autoencoder"]
    for arm in ("gen5a", "gen5b", "gen5c", "gen5d"):
        print(smoke_one_arm(arm, autoencoder))
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print({"peak_cpu_rss_kb": peak_rss_kb, "note": "CPU-only stand-in for gate 9's real CUDA memory smoke test"})


if __name__ == "__main__":
    main()
