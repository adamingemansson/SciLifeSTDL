"""Conditional latent flow matched to the MK conditional-WAE architecture.

The data adapter, Architecture-1 conditioner, target-expression encoder,
conditional-mean head, and expression decoder are the same design as MK WAE.
Only prior matching is replaced: a spatial velocity network transports
Gaussian latent noise to the target-expression latent, optionally using a
minibatch Sinkhorn OT pairing that keeps each target attached to its location.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.conditional_wae.model import Architecture1ImageConditioner
from gen3_multiscale.gen5.autoencoder import ExpressionEncoder
from gen3_multiscale.gen6.generative import minibatch_ot_flow_loss
from gen3_multiscale.models.flow import (
    VelocityNetwork,
    flow_matching_loss,
    sample_residual_coefficients,
)
from gen3_multiscale.models.losses import rmse_pcc_reconstruction_loss


class ConditionalLatentFlow(nn.Module):
    """Joint expression autoencoder plus conditional latent rectified flow."""

    def __init__(
        self,
        n_genes: int,
        image_conditioner: Architecture1ImageConditioner,
        *,
        coupling: str,
        latent_dim: int = 256,
        autoencoder_hidden_dim: int = 1024,
        n_flow_blocks: int = 2,
        n_flow_samples: int = 8,
        n_ode_steps: int = 20,
        flow_weight: float = 0.1,
        conditional_mean_weight: float = 1.0,
        pcc_weight: float = 0.1,
        ot_epsilon: float = 0.1,
        ot_sinkhorn_iters: int = 20,
    ):
        super().__init__()
        if coupling not in {"independent", "sinkhorn_ot"}:
            raise ValueError("coupling must be 'independent' or 'sinkhorn_ot'")
        if n_genes < 1 or latent_dim < 1 or n_flow_samples < 1 or n_ode_steps < 1:
            raise ValueError("model dimensions, samples and ODE steps must be positive")
        if flow_weight < 0 or conditional_mean_weight < 0 or pcc_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if ot_epsilon <= 0 or ot_sinkhorn_iters < 1:
            raise ValueError("OT epsilon and Sinkhorn iterations must be positive")
        if image_conditioner.n_genes != n_genes:
            raise ValueError("image_conditioner and flow must use the same n_genes")
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.coupling = coupling
        self.n_flow_samples = int(n_flow_samples)
        self.n_ode_steps = int(n_ode_steps)
        self.flow_weight = float(flow_weight)
        self.conditional_mean_weight = float(conditional_mean_weight)
        self.pcc_weight = float(pcc_weight)
        self.ot_epsilon = float(ot_epsilon)
        self.ot_sinkhorn_iters = int(ot_sinkhorn_iters)
        self.image_conditioner = image_conditioner
        context_dim = image_conditioner.hidden_dim
        self.expression_encoder = ExpressionEncoder(
            n_genes, latent_dim=latent_dim, hidden_dim=autoencoder_hidden_dim,
        )
        self.conditional_mean_head = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, autoencoder_hidden_dim),
            nn.GELU(), nn.Linear(autoencoder_hidden_dim, n_genes),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, autoencoder_hidden_dim),
            nn.GELU(), nn.LayerNorm(autoencoder_hidden_dim),
            nn.Linear(autoencoder_hidden_dim, autoencoder_hidden_dim), nn.GELU(),
            nn.Linear(autoencoder_hidden_dim, n_genes),
        )
        self.velocity_network = VelocityNetwork(
            residual_rank=latent_dim,
            hidden_dim=context_dim,
            n_heads=image_conditioner.blocks[0].cached_attention.n_heads,
            n_blocks=n_flow_blocks,
            dense_threshold=image_conditioner.blocks[0].cached_attention.dense_threshold,
            sparse_k=image_conditioner.blocks[0].sparse_k,
        )

    def _target(self, inputs, target_expression, *, context: torch.Tensor) -> torch.Tensor:
        target = torch.as_tensor(
            target_expression, dtype=context.dtype, device=context.device,
        )
        query_mask = torch.as_tensor(
            inputs.query_mask, dtype=torch.bool, device=context.device,
        )
        if target.shape == (query_mask.shape[0], self.n_genes):
            target = target[query_mask]
        expected = (context.shape[0], self.n_genes)
        if target.shape != expected:
            raise ValueError(f"target_expression must be {expected}, got {tuple(target.shape)}")
        if not torch.isfinite(target).all():
            raise ValueError("target_expression must be finite")
        return target

    def _query_coords(self, inputs, device: torch.device) -> torch.Tensor:
        coords = torch.as_tensor(inputs.coords, dtype=torch.float32, device=device)
        query_mask = torch.as_tensor(inputs.query_mask, dtype=torch.bool, device=device)
        return coords[query_mask]

    def decode(self, z: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.shape != (context.shape[0], self.latent_dim):
            raise ValueError("z must have one configured-width row per context row")
        conditional_mean = self.conditional_mean_head(context)
        residual = self.residual_decoder(torch.cat([context, z], dim=-1))
        return conditional_mean + residual, conditional_mean

    def compute_losses(self, inputs, target_expression, *, generator=None) -> dict:
        context = self.image_conditioner(inputs)
        target = self._target(inputs, target_expression, context=context)
        target_latent = self.expression_encoder(target)
        reconstruction, conditional_mean = self.decode(target_latent, context)
        reconstruction_loss, reconstruction_rmse, reconstruction_pcc = (
            rmse_pcc_reconstruction_loss(
                reconstruction, target, pcc_weight=self.pcc_weight,
            )
        )
        conditional_mean_loss, conditional_mean_rmse, conditional_mean_pcc = (
            rmse_pcc_reconstruction_loss(
                conditional_mean, target, pcc_weight=self.pcc_weight,
            )
        )
        coords = self._query_coords(inputs, context.device)
        # The encoder is learned through reconstruction. Stop-gradient makes
        # the flow chase an informative latent rather than shrinking that
        # latent merely to reduce the transport objective.
        flow_target = target_latent.detach()
        if self.coupling == "sinkhorn_ot":
            flow_loss = minibatch_ot_flow_loss(
                self.velocity_network, flow_target, coords, context,
                epsilon=self.ot_epsilon, n_iters=self.ot_sinkhorn_iters,
                generator=generator,
            )
        else:
            flow_loss = flow_matching_loss(
                self.velocity_network, flow_target, coords, context,
                generator=generator,
            )
        total = (
            reconstruction_loss
            + self.conditional_mean_weight * conditional_mean_loss
            + self.flow_weight * flow_loss
        )
        return {
            "total": total,
            "expression": reconstruction,
            "conditional_mean_expression": conditional_mean,
            "latent": target_latent,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_rmse": reconstruction_rmse,
            "reconstruction_pcc_loss": reconstruction_pcc,
            "conditional_mean_loss": conditional_mean_loss,
            "conditional_mean_rmse": conditional_mean_rmse,
            "conditional_mean_pcc_loss": conditional_mean_pcc,
            "flow_loss": flow_loss,
        }

    def forward(self, inputs) -> dict:
        context = self.image_conditioner(inputs)
        return {
            "expression": self.conditional_mean_head(context),
            "image_context": context,
        }

    @torch.no_grad()
    def sample_predictive_distribution(
        self, inputs, n_samples: int | None = None, n_steps: int | None = None,
        generator=None,
    ) -> dict:
        context = self.image_conditioner(inputs)
        coords = self._query_coords(inputs, context.device)
        latent_samples = sample_residual_coefficients(
            self.velocity_network, context.shape[0], coords, context,
            n_samples=int(n_samples or self.n_flow_samples),
            n_steps=int(n_steps or self.n_ode_steps), generator=generator,
        )
        count, n_query, _ = latent_samples.shape
        expanded_context = context.unsqueeze(0).expand(count, -1, -1)
        decoded, _ = self.decode(
            latent_samples.reshape(count * n_query, self.latent_dim),
            expanded_context.reshape(count * n_query, context.shape[1]),
        )
        samples = decoded.reshape(count, n_query, self.n_genes)
        conditional_mean = self.conditional_mean_head(context)
        return {
            "expression": samples.mean(0),
            "predictive_mean": samples.mean(0),
            "predictive_std": samples.std(0, unbiased=False),
            "predictive_samples": samples,
            "latent_samples": latent_samples,
            "conditional_mean_expression": conditional_mean,
            "image_context": context,
        }
