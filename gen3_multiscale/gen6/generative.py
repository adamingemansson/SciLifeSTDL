"""Staged Gen6 learned-latent OT-flow and WAE-GAN generators."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call

from gen3_multiscale.gen5.autoencoder import (
    ExpressionAutoencoder,
    verify_expression_autoencoder_gene_names,
)
from gen3_multiscale.models.flow import VelocityNetwork, sample_residual_coefficients
from gen3_multiscale.models.losses import rmse_pcc_reconstruction_loss


def sinkhorn_barycentric_ot_pairing(
    noise: torch.Tensor, target: torch.Tensor, *, epsilon: float = 0.1, n_iters: int = 20,
) -> torch.Tensor:
    """Return one OT-barycentric prior-noise row per *fixed* target row.

    Keeping the target row order is essential for conditional flow matching:
    target ``i`` must remain attached to query coordinates/conditioning ``i``.
    The OT plan may choose its prior partner, but must never exchange targets
    between spatial locations.
    """
    if noise.shape != target.shape or noise.ndim != 2:
        raise ValueError("noise and target must be matching [batch, latent_dim] tensors")
    if epsilon <= 0 or n_iters < 1:
        raise ValueError("epsilon and n_iters must be positive")
    with torch.no_grad():
        cost = torch.cdist(noise, target).square()
        log_kernel = -cost / float(epsilon)
        n = noise.shape[0]
        log_mass = -torch.log(torch.tensor(float(n), device=noise.device, dtype=noise.dtype))
        log_u = torch.zeros(n, device=noise.device, dtype=noise.dtype)
        log_v = torch.zeros_like(log_u)
        for _ in range(n_iters):
            log_u = log_mass - torch.logsumexp(log_kernel + log_v[None], dim=1)
            log_v = log_mass - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
        plan = torch.exp(log_kernel + log_u[:, None] + log_v[None])
        paired = (plan.T @ noise) / plan.sum(dim=0, keepdim=False).unsqueeze(1).clamp_min(1e-12)
    return paired


def minibatch_ot_flow_loss(
    network: VelocityNetwork, target: torch.Tensor, coords: torch.Tensor,
    conditioning: torch.Tensor, *, epsilon: float, n_iters: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    paired_noise = sinkhorn_barycentric_ot_pairing(noise, target, epsilon=epsilon, n_iters=n_iters)
    t = torch.rand((), device=target.device, generator=generator)
    x_t = (1 - t) * paired_noise + t * target
    velocity = network(x_t, t, coords, conditioning)
    return nn.functional.mse_loss(velocity, target - paired_noise)


class Gen6LatentOTFlowModel(nn.Module):
    """Frozen Gen6-C conditioner plus OT flow in a learned AE latent space."""

    def __init__(self, conditioner: nn.Module, autoencoder: ExpressionAutoencoder,
                 gene_names: list[str], *, hidden_dim: int = 512, n_heads: int = 8,
                 n_flow_blocks: int = 2, dense_threshold: int = 256,
                 sparse_k: int = 10, chunk_size: int = 1024,
                 n_flow_samples: int = 8, n_ode_steps: int = 20,
                 ot_epsilon: float = 0.1, ot_sinkhorn_iters: int = 20):
        super().__init__()
        verify_expression_autoencoder_gene_names(autoencoder, gene_names)
        self.conditioner = conditioner
        self.autoencoder = autoencoder
        self.velocity_network = VelocityNetwork(
            residual_rank=autoencoder.latent_dim, hidden_dim=hidden_dim, n_heads=n_heads,
            n_blocks=n_flow_blocks, dense_threshold=dense_threshold,
            sparse_k=sparse_k, chunk_size=chunk_size,
        )
        self.n_flow_samples, self.n_ode_steps = int(n_flow_samples), int(n_ode_steps)
        self.ot_epsilon, self.ot_sinkhorn_iters = float(ot_epsilon), int(ot_sinkhorn_iters)
        self._freeze_staged_modules()

    def _freeze_staged_modules(self):
        self.conditioner.eval()
        for parameter in self.conditioner.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.conditioner.eval()
        self.autoencoder.eval()
        return self

    def _condition(self, inputs):
        with torch.no_grad():
            output = self.conditioner(inputs)
        return output["expression"], output["query_hidden"]

    def compute_losses(self, inputs, target_expression, generator=None):
        mean, hidden = self._condition(inputs)
        target = torch.as_tensor(target_expression, dtype=mean.dtype, device=mean.device)
        if target.shape != mean.shape or not torch.isfinite(target).all():
            raise ValueError("target_expression must match the finite conditioner output")
        with torch.no_grad():
            target_latent = self.autoencoder.encode(target)
        coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=mean.device)
        loss = minibatch_ot_flow_loss(
            self.velocity_network, target_latent, coords, hidden,
            epsilon=self.ot_epsilon, n_iters=self.ot_sinkhorn_iters, generator=generator,
        )
        return {
            "query_hidden": hidden,
            "flow_loss": loss,
            "conditioner_expression_diagnostic_only": mean,
        }

    @torch.no_grad()
    def sample_predictive_distribution(self, inputs, n_samples=None, n_steps=None, generator=None):
        mean, hidden = self._condition(inputs)
        coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=mean.device)
        latent_samples = sample_residual_coefficients(
            self.velocity_network, coords.shape[0], coords, hidden,
            n_samples=n_samples or self.n_flow_samples,
            n_steps=n_steps or self.n_ode_steps, generator=generator,
        )
        n_samples_actual, n_query, latent_dim = latent_samples.shape
        samples = self.autoencoder.decode(
            latent_samples.reshape(n_samples_actual * n_query, latent_dim)
        ).reshape(n_samples_actual, n_query, self.autoencoder.n_genes)
        return {
            "expression": samples.mean(0), "predictive_mean": samples.mean(0),
            "predictive_std": samples.std(0, unbiased=False), "predictive_samples": samples,
            "latent_samples": latent_samples,
            "conditioner_expression_diagnostic_only": mean,
        }


class Gen6WAEGANModel(nn.Module):
    """Conditional WAE-GAN trained correctly with the shared one-optimizer loop."""
    def __init__(self, conditioner: nn.Module, n_genes: int, *, latent_dim: int = 256,
                 hidden_dim: int = 1024, conditioner_hidden_dim: int = 512,
                 discriminator_hidden_dim: int = 256, adversarial_weight: float = 0.1,
                 discriminator_weight: float = 1.0, pcc_weight: float = 0.1,
                 n_samples: int = 8):
        super().__init__()
        self.conditioner = conditioner
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + conditioner_hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, discriminator_hidden_dim), nn.GELU(),
            nn.Linear(discriminator_hidden_dim, 1),
        )
        self.latent_dim, self.n_samples = int(latent_dim), int(n_samples)
        self.adversarial_weight = float(adversarial_weight)
        self.discriminator_weight = float(discriminator_weight)
        self.pcc_weight = float(pcc_weight)
        self._freeze_conditioner()

    def _freeze_conditioner(self):
        self.conditioner.eval()
        for parameter in self.conditioner.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.conditioner.eval()
        return self

    def _condition(self, inputs):
        with torch.no_grad():
            output = self.conditioner(inputs)
        return output["expression"], output["query_hidden"]

    def _decode(self, z, hidden):
        return self.decoder(torch.cat([z, hidden], dim=-1))

    def compute_losses(self, inputs, target_expression, generator=None):
        _, hidden = self._condition(inputs)
        target = torch.as_tensor(target_expression, dtype=hidden.dtype, device=hidden.device)
        z_fake = self.encoder(target)
        z_real = torch.randn(z_fake.shape, device=z_fake.device, generator=generator)
        reconstruction = self._decode(z_fake, hidden)
        reconstruction_loss, reconstruction_rmse, reconstruction_pcc_loss = (
            rmse_pcc_reconstruction_loss(
                reconstruction, target, pcc_weight=self.pcc_weight,
            )
        )
        logits_real = self.discriminator(z_real.detach())
        logits_fake_detached = self.discriminator(z_fake.detach())
        discriminator_loss = (
            nn.functional.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
            + nn.functional.binary_cross_entropy_with_logits(
                logits_fake_detached, torch.zeros_like(logits_fake_detached)
            )
        )
        detached_state = {name: value.detach() for name, value in self.discriminator.named_parameters()}
        detached_state.update({name: value for name, value in self.discriminator.named_buffers()})
        fool_logits = functional_call(self.discriminator, detached_state, (z_fake,))
        adversarial_loss = nn.functional.binary_cross_entropy_with_logits(
            fool_logits, torch.ones_like(fool_logits),
        )
        total = (
            reconstruction_loss + self.adversarial_weight * adversarial_loss
            + self.discriminator_weight * discriminator_loss
        )
        return {
            "total": total, "expression": reconstruction,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_rmse": reconstruction_rmse,
            "reconstruction_pcc_loss": reconstruction_pcc_loss,
            "adversarial_loss": adversarial_loss, "discriminator_loss": discriminator_loss,
        }

    def forward(self, inputs):
        _, hidden = self._condition(inputs)
        z = torch.zeros(hidden.shape[0], self.latent_dim, dtype=hidden.dtype, device=hidden.device)
        return {"expression": self._decode(z, hidden), "query_hidden": hidden}

    @torch.no_grad()
    def sample_predictive_distribution(self, inputs, n_samples=None, generator=None, **_):
        conditioner_mean, hidden = self._condition(inputs)
        samples = []
        for _ in range(n_samples or self.n_samples):
            z = torch.randn(hidden.shape[0], self.latent_dim, device=hidden.device, generator=generator)
            samples.append(self._decode(z, hidden))
        samples = torch.stack(samples)
        zero_latent = torch.zeros(
            hidden.shape[0], self.latent_dim, dtype=hidden.dtype, device=hidden.device,
        )
        return {
            "expression": samples.mean(0), "predictive_mean": samples.mean(0),
            "predictive_std": samples.std(0, unbiased=False), "predictive_samples": samples,
            "deterministic_mean": self._decode(zero_latent, hidden),
            "conditioner_expression_diagnostic_only": conditioner_mean,
        }
