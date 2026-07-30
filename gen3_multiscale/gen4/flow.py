"""Gen4ResidualFlowModel -- GEN4_CONTRACT.md sections 2, 10.

`gen3_multiscale.models.architectures.Architecture4` hardcodes
`self.conditioner = Architecture3(...)` with no injection point for a
different conditioner class, so it cannot be reused directly for Gen4 --
this class is a structurally faithful copy of `Architecture4` with
`self.conditioner` built as a `Gen4Conditioner` instead. Every OTHER piece
is imported and used unmodified: `VelocityNetwork`, `flow_matching_loss`,
`sample_residual_coefficients` (models/flow.py), `GeneResidualBasis`/
`verify_gene_residual_basis` (models/gene_basis.py). The four public
methods below (`forward`, `compute_flow_matching_loss`, `compute_losses`,
`sample_predictive_distribution`) are intentionally byte-for-byte
equivalent in behavior to `Architecture4`'s own -- only the conditioner
type differs -- so anything already true of Gen3 Architecture 4's flow
apparatus (single conditioner pass per step, detached query_hidden/mean,
generator-seeded reproducible sampling) is true here too.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.models.flow import VelocityNetwork, flow_matching_loss, sample_residual_coefficients
from gen3_multiscale.models.gene_basis import GeneResidualBasis, verify_gene_residual_basis


class Gen4ResidualFlowModel(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        gex_feature_dim: int,
        gene_basis: GeneResidualBasis,
        gene_names: list[str],
        image_feature_dim: int = 1536,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        n_flow_blocks: int = 2,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        max_boundary_size: int | None = None,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        gene_gate_mode: str = "per_gene",
        use_query_gate: bool = True,
        use_residual: bool = False,
        residual_rank: int = 32,
        target_gene_scale: torch.Tensor | None = None,
        use_regional_he: bool = False,
        use_global_gex: bool = True,
        regional_grid_size: int = 4,
        n_gex_inducing: int = 16,
        harmonic_k_neighbors: int = 6,
        n_flow_samples: int = 8,
        n_ode_steps: int = 20,
        gex_feature_source: str = "weighted_linear",
        gex_context_embedding_dim: int | None = None,
        image_feature_source: str = "precomputed",
        stpath_encoder: nn.Module | None = None,
        global_context_source: str = "none",
        global_slide_dim: int = 768,
        slide_encoder=None,
        gigapath_checkpoint_sha256: str | None = None,
        uni2_global_pool: nn.Module | None = None,
        model_architecture_version: str = "gen4-flow-v1",
    ):
        super().__init__()
        verify_gene_residual_basis(gene_basis, gene_names)
        self.gene_basis = gene_basis
        self.register_buffer("_gene_basis_matrix", gene_basis.basis.clone())
        self.n_flow_samples = n_flow_samples
        self.n_ode_steps = n_ode_steps

        self.conditioner = Gen4Conditioner(
            n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale,
            use_regional_he=use_regional_he, use_global_gex=use_global_gex, regional_grid_size=regional_grid_size,
            n_gex_inducing=n_gex_inducing, harmonic_k_neighbors=harmonic_k_neighbors,
            gex_feature_source=gex_feature_source, gex_context_embedding_dim=gex_context_embedding_dim,
            image_feature_source=image_feature_source, stpath_encoder=stpath_encoder,
            global_context_source=global_context_source, global_slide_dim=global_slide_dim,
            slide_encoder=slide_encoder, gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
            uni2_global_pool=uni2_global_pool, model_architecture_version=model_architecture_version,
        )
        self.velocity_network = VelocityNetwork(
            residual_rank=gene_basis.rank, hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_flow_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
        )

    def forward(self, inputs) -> dict:
        return self.conditioner(inputs)

    def _prepare_target_expression(self, target_expression, deterministic_mean: torch.Tensor, device: torch.device) -> torch.Tensor:
        target_expression = torch.as_tensor(target_expression, device=device, dtype=deterministic_mean.dtype)
        if target_expression.shape != deterministic_mean.shape:
            raise ValueError(
                f"target_expression {tuple(target_expression.shape)} must match the conditioner's "
                f"deterministic_mean shape {tuple(deterministic_mean.shape)}"
            )
        if not torch.isfinite(target_expression).all():
            raise ValueError("target_expression contains non-finite (NaN/Inf) values")
        return target_expression

    def compute_flow_matching_loss(self, inputs, target_expression, generator: torch.Generator | None = None) -> torch.Tensor:
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        deterministic_mean = conditioner_out["expression"].detach()
        target_expression = self._prepare_target_expression(target_expression, deterministic_mean, device)
        target_residual = target_expression - deterministic_mean
        target_coefficients = target_residual @ self._gene_basis_matrix.T
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        return flow_matching_loss(self.velocity_network, target_coefficients, query_coords, query_hidden, generator=generator)

    def compute_losses(self, inputs, target_expression, generator: torch.Generator | None = None) -> dict:
        """Runs self.conditioner exactly ONCE per step, matching
        Architecture4.compute_losses's own reasoning: calling forward()
        and compute_flow_matching_loss() separately would draw two
        different dropout masks, silently disagreeing on
        deterministic_mean between the two losses."""
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        deterministic_mean = conditioner_out["expression"].detach()
        target_expression = self._prepare_target_expression(target_expression, deterministic_mean, device)
        target_residual = target_expression - deterministic_mean
        target_coefficients = target_residual @ self._gene_basis_matrix.T
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        flow_loss = flow_matching_loss(self.velocity_network, target_coefficients, query_coords, query_hidden, generator=generator)
        return {**conditioner_out, "flow_loss": flow_loss}

    @torch.no_grad()
    def sample_predictive_distribution(
        self, inputs, n_samples: int | None = None, n_steps: int | None = None, generator: torch.Generator | None = None,
    ) -> dict:
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"]
        deterministic_mean = conditioner_out["expression"]
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        n_query = query_coords.shape[0]

        coefficient_samples = sample_residual_coefficients(
            self.velocity_network, n_query, query_coords, query_hidden,
            n_samples=n_samples or self.n_flow_samples, n_steps=n_steps or self.n_ode_steps, generator=generator,
        )
        residual_samples = coefficient_samples @ self._gene_basis_matrix
        predictive_samples = deterministic_mean[None] + residual_samples
        predictive_mean = predictive_samples.mean(dim=0)
        predictive_std = predictive_samples.std(dim=0, unbiased=False)
        return {
            "expression": predictive_mean,
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "predictive_samples": predictive_samples,
            "deterministic_mean": deterministic_mean,
        }
