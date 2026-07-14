"""
Model registry: model-agnostic generative backbone interface.

Every family (VAE, WAE-GAN, diffusion, ...) implements BaseGenerativeModel,
a thin pytorch_lightning.LightningModule subclass. Lightning owns the
boilerplate (checkpointing, device placement, single- vs. multi-optimizer
training loops via automatic_optimization) so each family only has to
implement its own training_step/configure_optimizers/sample() — necessary
because GAN-style alternating updates and multi-step diffusion sampling
don't fit a single shared loss()/forward() call the way a plain VAE does.
See docs/architecture_plan.md for the full design rationale.

Usage:
    from src.models.registry import build_model

    model = build_model(cfg.model)   # cfg.model.name == "vae_baseline", "wae_gan", ...

To add a new architecture:
    1. Implement a class inheriting from BaseGenerativeModel (below).
    2. Register it with @register_model("your_name").
    3. Reference "your_name" in a config file. Nothing else changes.
"""
from __future__ import annotations
import abc
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.models.conditioning import SpatialContextEncoder

_MODEL_REGISTRY: dict[str, type["BaseGenerativeModel"]] = {}


def register_model(name: str):
    def _wrap(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' already registered.")
        _MODEL_REGISTRY[name] = cls
        return cls
    return _wrap


def build_model(model_cfg: dict) -> "BaseGenerativeModel":
    name = model_cfg["name"]
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](**model_cfg.get("params", {}))


class BaseGenerativeModel(pl.LightningModule, abc.ABC):
    """
    Common interface every generative backbone must implement so the
    training loop, masking simulator, and evaluation code are architecture-
    agnostic.

    Conceptually mirrors the Mimyr-style decomposition (see
    docs/literature_review.md) but keeps it generic:
        context   -> spatial/expression info from observed (unmasked) tissue
            'coords'     [N_obs, D]   spatial coords of observed points (D=2 or 3)
            'expression' [N_obs, G]   gene expression of observed points
            'cell_type'  [N_obs]      optional cell type labels/ids
        query     -> where we want to generate (missing locations / slice)
            'coords'     [N_query, D] target locations
        sample() return -> dict with (a subset of):
            'coords'      [N_gen, D]
            'cell_type'   [N_gen]
            'expression'  [N_gen, G]

    sample() is the ONE entry point evaluation code and every other consumer
    calls, regardless of what happens internally — a single forward pass for
    VAE/WAE-GAN, an iterative denoising loop for diffusion. Never reach into
    a family's internals from outside this class.
    """

    @abc.abstractmethod
    def sample(self, context: dict[str, torch.Tensor], query: dict[str, Any]
               ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch: dict, batch_idx: int):
        """Lightning entry point. Implement the family's own training logic
        here (ELBO for VAE, alternating encoder/decoder vs. discriminator
        updates for WAE-GAN, noise-prediction for diffusion, ...). Use
        self.log(...)/self.log_dict(...) to report training metrics."""
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        """Return one optimizer (VAE, diffusion) or a list of optimizers
        (WAE-GAN: [opt_ae, opt_disc]). Pair a list with
        self.automatic_optimization = False in __init__ — see WAEGAN."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Example minimal baseline: nearest-neighbour / linear interpolation "model"
# with no learned parameters. Useful as a sanity-check floor for the metrics
# pipeline before any real model is trained.
# ---------------------------------------------------------------------------
@register_model("interp_baseline")
class InterpolationBaseline(BaseGenerativeModel):
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def sample(self, context, query):
        coords_obs = context["coords"]          # [N_obs, D]
        expr_obs = context["expression"]         # [N_obs, G]
        coords_q = query["coords"]               # [N_query, D]

        # distance-weighted k-NN interpolation, purely as a floor baseline
        dists = torch.cdist(coords_q, coords_obs)          # [N_query, N_obs]
        knn_d, knn_i = torch.topk(dists, k=min(self.k, dists.shape[1]),
                                   largest=False, dim=1)
        weights = 1.0 / (knn_d + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)
        expr_gen = torch.einsum("nk,nkg->ng", weights, expr_obs[knn_i])
        return {"coords": coords_q, "expression": expr_gen}

    def training_step(self, batch, batch_idx):
        return None  # no learned parameters — nothing to train

    def configure_optimizers(self):
        return None  # no parameters to optimize


# ---------------------------------------------------------------------------
# VAE baseline. Unconditioned placeholder — see docs/architecture_plan.md
# "Known gaps" for the conditioning encoder this still needs.
# ---------------------------------------------------------------------------
@register_model("vae_baseline")
class VAEBaseline(BaseGenerativeModel):
    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dim: int = 256,
                 kl_weight: float = 1e-3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.lr = lr

    def _encode(self, expression):
        h = self.encoder(expression)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def sample(self, context, query):
        # TODO: condition on query['coords']/context once the shared
        # conditioning encoder exists (docs/architecture_plan.md). Left
        # unconditioned here as a structural placeholder — draws z from the
        # prior directly, same as the original stub.
        n = query["coords"].shape[0]
        z = torch.randn(n, self.latent_dim, device=self.device)
        expr_gen = self.decoder(z)
        return {"coords": query["coords"], "expression": expr_gen}

    def training_step(self, batch, batch_idx):
        expression = batch["context"]["expression"]
        mu, logvar = self._encode(expression)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        recon_loss = nn.functional.mse_loss(recon, expression)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_weight * kld
        self.log_dict({"train/recon": recon_loss, "train/kld": kld, "train/loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# WAE-GAN: Wasserstein Auto-Encoder with an adversarial latent regularizer
# (Tolstikhin et al. 2017 — docs/literature_review.md, docs/metrics_notes.md
# SS4). Same reconstruction path as the VAE above, but replaces the KL term
# with a discriminator that pushes the *encoder's* aggregated latent
# distribution toward the prior, instead of judging generated expression
# directly. Chosen over a vanilla conditional GAN because the adversarial
# signal only touches the low-dimensional latent code, not the sparse/
# zero-inflated expression output — a smaller, better-behaved sub-problem
# and the lower-risk way to get a first GAN-family entry working.
#
# Conditioning (docs/model_schematics.md SS1): owns its own
# SpatialContextEncoder instance, trained jointly with this model's own
# gradient signal rather than sharing weights with other registry entries —
# same reasoning as encoder/decoder already being per-model. "One shared
# architecture" means one reusable class, not one shared set of trained
# weights across families.
# ---------------------------------------------------------------------------
@register_model("wae_gan")
class WAEGAN(BaseGenerativeModel):
    def __init__(self, n_genes: int, coord_dim: int = 3, latent_dim: int = 32,
                 hidden_dim: int = 256, cond_hidden_dim: int = 256,
                 disc_hidden_dim: int = 128, adv_weight: float = 1.0,
                 lr: float = 1e-3, lr_disc: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # we alternate encoder/decoder vs. discriminator ourselves

        self.context_encoder = SpatialContextEncoder(
            n_genes=n_genes, coord_dim=coord_dim, hidden_dim=cond_hidden_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # deterministic encoder, no logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, disc_hidden_dim), nn.ReLU(),
            nn.Linear(disc_hidden_dim, 1),   # logit: real-prior-sample vs. encoder output
        )
        self.latent_dim = latent_dim
        self.adv_weight = adv_weight
        self.lr = lr
        self.lr_disc = lr_disc

    def sample(self, context, query):
        # No real target expression at generation time, so z ~ prior (as
        # before) — but the decoder is now also conditioned on c, which
        # carries real local structure from context. z supplies the
        # remaining stochasticity/diversity (docs/architecture_plan.md
        # "mode-averaging risk" — a query location can be genuinely
        # multimodal, e.g. a cell-type boundary; c alone doesn't resolve
        # that, sampling z does).
        n = query["coords"].shape[0]
        c = self.context_encoder(context["coords"], context["expression"], query["coords"])
        z = torch.randn(n, self.latent_dim, device=self.device)
        expr_gen = self.decoder(torch.cat([z, c], dim=-1))
        return {"coords": query["coords"], "expression": expr_gen}

    def training_step(self, batch, batch_idx):
        # Fixes a real gap in the previous placeholder: it trained as a
        # plain autoencoder on context alone and never touched
        # target_expression, i.e. never actually learned to predict the
        # held-out query locations it's meant to reconstruct. Now: encode
        # the REAL target expression (available during training, not at
        # generation time) into z, and train the decoder to reconstruct it
        # from (z, c) — teaches decoder+context_encoder to actually combine
        # local context with a latent code into the right expression.
        context, query = batch["context"], batch["query"]
        target_expression = batch["target_expression"]
        opt_ae, opt_disc = self.optimizers()
        batch_size = target_expression.shape[0]

        c = self.context_encoder(context["coords"], context["expression"], query["coords"])
        z_fake = self.encoder(target_expression)                              # encoder's latent code
        z_real = torch.randn(batch_size, self.latent_dim, device=self.device)  # prior sample

        # --- 1. discriminator step: real prior sample vs. encoder output ---
        logits_real = self.discriminator(z_real.detach())
        logits_fake = self.discriminator(z_fake.detach())
        disc_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_real, torch.ones_like(logits_real)
        ) + nn.functional.binary_cross_entropy_with_logits(
            logits_fake, torch.zeros_like(logits_fake)
        )
        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        # --- 2. encoder/decoder step: reconstruction + fool the discriminator ---
        recon = self.decoder(torch.cat([z_fake, c], dim=-1))
        recon_loss = nn.functional.mse_loss(recon, target_expression)
        logits_fake_for_ae = self.discriminator(z_fake)
        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake_for_ae, torch.ones_like(logits_fake_for_ae)  # fool disc: look like prior
        )
        ae_loss = recon_loss + self.adv_weight * adv_loss
        opt_ae.zero_grad()
        self.manual_backward(ae_loss)
        opt_ae.step()

        self.log_dict({
            "train/recon": recon_loss, "train/adv": adv_loss,
            "train/disc": disc_loss, "train/ae_loss": ae_loss,
        })

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            list(self.context_encoder.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters()),
            lr=self.lr,
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_disc)
        return [opt_ae, opt_disc]


# ---------------------------------------------------------------------------
# FM-OT: Flow Matching with optimal-transport (straight-line) paths
# (Lipman et al. 2022 — docs/literature_review.md). Trains a velocity
# network to regress onto x_1-x_0 along linear interpolation paths between
# noise and the real target expression, conditioned on local spatial
# context via its own SpatialContextEncoder (same reasoning as WAE-GAN:
# per-model instance, not shared trained weights).
#
# diffusers.FlowMatchEulerDiscreteScheduler confirmed to operate on generic
# (non-image) tensors (docs/model_schematics.md), but sampling here uses a
# plain manual Euler integrator instead — our training loop is a direct
# regression, not needing the scheduler's broader image-pipeline feature
# set (s_churn/s_tmin/s_tmax/per_token_timesteps etc.).
#
# The diffusion-path ablation (docs/architecture_plan.md "Prioritization" —
# same network, swap the interpolation formula) is deferred, not
# implemented here — flagged explicitly as a cheap follow-up, not scope
# creep on this entry.
# ---------------------------------------------------------------------------
class _SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for a continuous scalar t in [0,1]
    (Vaswani et al. 2017 positional encoding, adapted from integer
    positions to continuous time) — used across essentially all modern
    diffusion/flow-matching implementations for exactly this purpose."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device))
                           * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


@register_model("fm_ot")
class FlowMatchingOT(BaseGenerativeModel):
    def __init__(self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
                 hidden_dim: int = 512, time_embed_dim: int = 64,
                 n_ode_steps: int = 50, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.context_encoder = SpatialContextEncoder(
            n_genes=n_genes, coord_dim=coord_dim, hidden_dim=cond_hidden_dim
        )
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(n_genes + time_embed_dim + cond_hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.n_genes = n_genes
        self.n_ode_steps = n_ode_steps
        self.lr = lr

    def _velocity(self, x_t, t, c):
        t_embed = self.time_embed(t)
        return self.velocity_net(torch.cat([x_t, t_embed, c], dim=-1))

    def sample(self, context, query):
        n = query["coords"].shape[0]
        c = self.context_encoder(context["coords"], context["expression"], query["coords"])
        x = torch.randn(n, self.n_genes, device=self.device)
        dt = 1.0 / self.n_ode_steps
        for step in range(self.n_ode_steps):
            t = torch.full((n,), step * dt, device=self.device)
            x = x + dt * self._velocity(x, t, c)  # manual Euler ODE integration
        return {"coords": query["coords"], "expression": x}

    def training_step(self, batch, batch_idx):
        context, query = batch["context"], batch["query"]
        x_1 = batch["target_expression"]  # real target expression
        n = x_1.shape[0]
        c = self.context_encoder(context["coords"], context["expression"], query["coords"])

        x_0 = torch.randn_like(x_1)
        t = torch.rand(n, device=self.device)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1   # OT straight-line path
        target_velocity = x_1 - x_0                        # constant along a straight line

        pred_velocity = self._velocity(x_t, t, c)
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        self.log_dict({"train/fm_loss": loss})
        return loss

    def configure_optimizers(self):
        # AdamW (decoupled weight decay) over plain Adam: standard choice in
        # the flow-matching/diffusion literature (Lipman et al. 2022 and
        # essentially all follow-ups use AdamW, not Adam).
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
