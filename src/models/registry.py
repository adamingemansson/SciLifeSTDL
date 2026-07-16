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
from src.models.vqvae import VectorQuantizer, morton_order

_MODEL_REGISTRY: dict[str, type["BaseGenerativeModel"]] = {}


def _build_context_encoder(
    n_genes: int, coord_dim: int, cond_hidden_dim: int,
    context_encoder_type: str = "builtin",
    image_encoder_type: str = "none", image_feat_dim: int = 64, image_patch_size: int = 256,
    gene_encoder_type: str = "raw", gene_feat_dim: int = 256, novae_dim: int | None = None,
    stpath_gene_names: list[str] | None = None, stpath_gene_voc_path: str | None = None,
    stpath_model_weight_path: str | None = None, stpath_organ_type: str = "Kidney",
    stpath_tech_type: str = "Visium",
    stpath_new_gene_encoder_type: str = "none", stpath_novae_dim: int | None = None,
    stpath_pretrained: bool = True,
    storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
    storm_lite_use_relative_bias: bool = True, storm_lite_relative_bias_hidden_dim: int = 32,
    organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
):
    """Shared by WAE-GAN/FM-OT/VQ-VAE+AR so each model's __init__ doesn't
    repeat the context_encoder_type branching. "builtin" (default) is our
    own SpatialContextEncoder (task #17/#20's image_encoder_type switch,
    and the 2026-07-16 gene_encoder_type switch — "mlp"/"novae" — still
    apply here). "stpath" (task #18) replaces it entirely with
    STPathContextEncoder — see src/models/stpath_encoder.py for the full
    setup requirements and grounding; imported lazily since the `stpath`
    package is an opt-in external dependency, not installed by default.
    "storm_lite" (2026-07-16) replaces it with StormLiteContextEncoder
    (src/models/storm_lite_encoder.py) — reuses gene_encoder_type/
    novae_dim (same meaning: which gene encoder, and Novae's real
    dimensionality) since "builtin"/"storm_lite" are mutually exclusive
    per model, no risk of the two conflating.

    stpath_new_gene_encoder_type/stpath_novae_dim (2026-07-16, "Route B"
    GEX-encoder-bottleneck follow-up) are DELIBERATELY separate params
    from gene_encoder_type/novae_dim above, not reused — those swap the
    gene branch of our OWN builtin encoder; these add a residual gene
    signal ON TOP of STPath's real pretrained fusion (see
    STPathContextEncoder's new_gene_encoder_type) — different mechanism,
    different meaning, kept as distinctly-named params so a config can't
    accidentally conflate the two.

    stpath_pretrained=False (2026-07-16, "STPath's own architecture
    trained from scratch on our data" comparison arm — see
    STPathContextEncoder's own pretrained docstring) — stpath_gene_names/
    stpath_gene_voc_path are STILL required (fixed resources, not trained
    parameters); stpath_model_weight_path is not (nothing to load).

    organ_vocab/tech_vocab (2026-07-16, multi-sample training follow-up)
    only apply to "builtin"/"storm_lite" (both route through
    OrganTechEmbedding, see conditioning.py) — NOT "stpath", which already
    has its own fixed-string stpath_organ_type/stpath_tech_type mechanism
    (a single organ/tech per model, matching STPath's own real
    IDTokenizer vocabulary loaded from model_weight_path, not a
    data-driven vocab we build ourselves). Passing both None (default)
    disables organ/tech conditioning entirely, same as before this
    param existed — single-sample/single-organ training is unaffected."""
    if context_encoder_type == "builtin":
        return SpatialContextEncoder(
            n_genes=n_genes, coord_dim=coord_dim, hidden_dim=cond_hidden_dim,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=gene_feat_dim,
            novae_dim=novae_dim,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
    elif context_encoder_type == "stpath":
        from src.models.stpath_encoder import STPathContextEncoder
        assert stpath_gene_names and stpath_gene_voc_path, (
            "context_encoder_type='stpath' requires stpath_gene_names and stpath_gene_voc_path"
        )
        assert not stpath_pretrained or stpath_model_weight_path, (
            "context_encoder_type='stpath' with stpath_pretrained=True (default) "
            "requires stpath_model_weight_path"
        )
        return STPathContextEncoder(
            gene_names=stpath_gene_names, gene_voc_path=stpath_gene_voc_path,
            model_weight_path=stpath_model_weight_path, organ_type=stpath_organ_type,
            tech_type=stpath_tech_type, hidden_dim=cond_hidden_dim,
            new_gene_encoder_type=stpath_new_gene_encoder_type, novae_dim=stpath_novae_dim,
            pretrained=stpath_pretrained,
        )
    elif context_encoder_type == "storm_lite":
        from src.models.storm_lite_encoder import StormLiteContextEncoder
        assert gene_encoder_type in ("mlp", "novae", "both"), (
            f"context_encoder_type='storm_lite' requires gene_encoder_type in "
            f"('mlp', 'novae', 'both'), got {gene_encoder_type!r}"
        )
        return StormLiteContextEncoder(
            n_genes=n_genes, novae_dim=novae_dim, coord_dim=coord_dim,
            hidden_dim=cond_hidden_dim, gene_encoder_type=gene_encoder_type,
            n_transformer_layers=storm_lite_n_layers, n_heads=storm_lite_n_heads,
            use_relative_bias=storm_lite_use_relative_bias,
            relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
    else:
        raise ValueError(f"unknown context_encoder_type {context_encoder_type!r}")


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

    context/query also accept an optional 'images' key (task #17, H&E
    branch) — [N, 3, H, W] float in [0,1], only meaningful if the model's
    context_encoder was built with image_encoder_type != "none"
    (src/models/conditioning.py). Use _encode_context() below rather than
    calling self.context_encoder(...) directly, so this stays a one-line
    addition instead of touching every family's sample()/training_step().
    """

    def _encode_context(self, context: dict, query: dict) -> torch.Tensor:
        return self.context_encoder(
            context["coords"], context["expression"], query["coords"],
            context_images=context.get("images"), query_images=query.get("images"),
            context_novae_features=context.get("novae_features"),
            organ=context.get("organ"), tech=context.get("tech"),
        )

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
                 lr: float = 1e-3, lr_disc: float = 1e-3,
                 image_encoder_type: str = "none", image_feat_dim: int = 64,
                 image_patch_size: int = 256, context_encoder_type: str = "builtin",
                 gene_encoder_type: str = "raw", gene_feat_dim: int = 256,
                 novae_dim: int | None = None,
                 stpath_gene_names: list[str] | None = None, stpath_gene_voc_path: str | None = None,
                 stpath_model_weight_path: str | None = None, stpath_organ_type: str = "Kidney",
                 stpath_tech_type: str = "Visium",
                 stpath_new_gene_encoder_type: str = "none", stpath_novae_dim: int | None = None,
                 stpath_pretrained: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_use_relative_bias: bool = True,
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # we alternate encoder/decoder vs. discriminator ourselves

        self.context_encoder = _build_context_encoder(
            n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
            context_encoder_type=context_encoder_type,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=gene_feat_dim, novae_dim=novae_dim,
            stpath_gene_names=stpath_gene_names, stpath_gene_voc_path=stpath_gene_voc_path,
            stpath_model_weight_path=stpath_model_weight_path, stpath_organ_type=stpath_organ_type,
            stpath_tech_type=stpath_tech_type,
            stpath_new_gene_encoder_type=stpath_new_gene_encoder_type, stpath_novae_dim=stpath_novae_dim,
            stpath_pretrained=stpath_pretrained,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_use_relative_bias=storm_lite_use_relative_bias,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
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
        c = self._encode_context(context, query)
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

        c = self._encode_context(context, query)
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
# Diffusion-path ablation (`path_type="edm"`, added 2026-07-14): same
# network (context_encoder/encoder/decoder/time_embed/velocity_net) —
# swaps the OT straight-line interpolation for EDM's noise/denoising
# formulation (Karras et al. 2022, NeurIPS, "Elucidating the Design Space
# of Diffusion-Based Generative Models") rather than plain DDPM (Ho et al.
# 2020) — EDM is the more carefully-justified, widely-adopted modern
# diffusion formulation, and its "elucidated design" preconditioning
# (c_skip/c_out/c_in/c_noise) is the actual core contribution, not an
# optional add-on, so it's implemented here rather than skipped.
# Simplifications flagged explicitly, not hidden: (1) sampling uses plain
# Euler steps on the probability-flow ODE, not EDM's recommended 2nd-order
# Heun sampler — consistent with this file's existing choice to keep
# sampling loops simple (see the OT path's own manual-Euler-over-
# diffusers-scheduler note above); (2) sigma_data/P_mean/P_std defaults
# are EDM's own published values, tuned for their image-pixel domain, not
# retuned for our latent-code scale — a reasonable starting point, not a
# validated-for-this-domain claim; (3) _SinusoidalTimeEmbedding (built for
# t in [0,1]) is reused for EDM's real-valued c_noise conditioning signal
# rather than building a second embedding module — works numerically, not
# specifically tuned for that input range.
#
# LATENT-SPACE, not raw 16570-gene space: the first real-data runs (flat
# PCC~0, RMSE stuck near noise-scale from 100 to 10000 training steps —
# docs/model_schematics.md) pointed to the velocity network converging to a
# degenerate near-zero solution rather than actually training — consistent
# with flow matching/diffusion directly in a very high-dimensional raw
# space being a much harder regression target than WAE-GAN's direct
# reconstruction. Standard fix, grounded in peer-reviewed and
# domain-specific precedent: run the ODE in a small learned latent space
# instead of raw expression space (Rombach et al. 2022, CVPR — "Latent
# Diffusion Models"; the same recipe applied specifically to single-cell
# gene expression in CFGen, Palma et al. 2025, built on scVI, and scLDM,
# Palla et al. 2025 — both arXiv preprints as of this writing, cited here
# as corroborating domain precedent, not as the sole grounding, which is
# Rombach et al. 2022). FM-OT now owns its own small encoder/decoder (own
# weights, not shared with WAEGAN's — same per-model-weights reasoning as
# elsewhere in this file), trained *jointly* with the flow-matching
# objective in one training_step rather than as a literal separate
# pretraining stage: the encoder/decoder gradient comes only from the
# reconstruction loss (the flow-matching target latent code is detached),
# which approximates a frozen pretrained autoencoder without needing a
# second training script. Documented simplification, not scope creep.
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
                 latent_dim: int = 32, ae_hidden_dim: int = 256,
                 hidden_dim: int = 512, time_embed_dim: int = 64,
                 n_ode_steps: int = 50, recon_weight: float = 1.0,
                 fm_weight: float = 1.0, lr: float = 1e-3,
                 path_type: str = "ot", sigma_min: float = 0.002,
                 sigma_max: float = 80.0, sigma_data: float = 0.5, rho: float = 7.0,
                 edm_p_mean: float = -1.2, edm_p_std: float = 1.2,
                 image_encoder_type: str = "none", image_feat_dim: int = 64,
                 image_patch_size: int = 256, context_encoder_type: str = "builtin",
                 gene_encoder_type: str = "raw", gene_feat_dim: int = 256,
                 novae_dim: int | None = None,
                 stpath_gene_names: list[str] | None = None, stpath_gene_voc_path: str | None = None,
                 stpath_model_weight_path: str | None = None, stpath_organ_type: str = "Kidney",
                 stpath_tech_type: str = "Visium",
                 stpath_new_gene_encoder_type: str = "none", stpath_novae_dim: int | None = None,
                 stpath_pretrained: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_use_relative_bias: bool = True,
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        self.save_hyperparameters()
        assert path_type in ("ot", "edm"), f"unknown path_type {path_type!r}"
        self.context_encoder = _build_context_encoder(
            n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
            context_encoder_type=context_encoder_type,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=gene_feat_dim, novae_dim=novae_dim,
            stpath_gene_names=stpath_gene_names, stpath_gene_voc_path=stpath_gene_voc_path,
            stpath_model_weight_path=stpath_model_weight_path, stpath_organ_type=stpath_organ_type,
            stpath_tech_type=stpath_tech_type,
            stpath_new_gene_encoder_type=stpath_new_gene_encoder_type, stpath_novae_dim=stpath_novae_dim,
            stpath_pretrained=stpath_pretrained,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_use_relative_bias=storm_lite_use_relative_bias,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        # own autoencoder, own weights — compresses expression to a small
        # latent code the velocity net operates on instead of raw n_genes
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_hidden_dim, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, n_genes),
        )
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim + cond_hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.n_genes = n_genes
        self.latent_dim = latent_dim
        self.n_ode_steps = n_ode_steps
        self.recon_weight = recon_weight
        self.fm_weight = fm_weight
        self.lr = lr
        self.path_type = path_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.edm_p_mean = edm_p_mean
        self.edm_p_std = edm_p_std

    def _velocity(self, z_t, t, c):
        t_embed = self.time_embed(t)
        return self.velocity_net(torch.cat([z_t, t_embed, c], dim=-1))

    def _edm_denoise(self, z_sigma, sigma, c):
        """EDM's preconditioned denoiser D_theta (Karras et al. 2022 eq. 7):
        wraps the same velocity_net used by the OT path with c_skip/c_out/
        c_in scaling so the network only has to learn a well-conditioned
        residual at every noise level, not the raw denoising map."""
        sigma = sigma.view(-1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = 0.25 * torch.log(sigma.squeeze(-1))
        t_embed = self.time_embed(c_noise)
        f = self.velocity_net(torch.cat([c_in * z_sigma, t_embed, c], dim=-1))
        return c_skip * z_sigma + c_out * f

    def sample(self, context, query):
        n = query["coords"].shape[0]
        c = self._encode_context(context, query)

        if self.path_type == "ot":
            z = torch.randn(n, self.latent_dim, device=self.device)
            dt = 1.0 / self.n_ode_steps
            for step in range(self.n_ode_steps):
                t = torch.full((n,), step * dt, device=self.device)
                z = z + dt * self._velocity(z, t, c)  # manual Euler ODE integration, in latent space
        else:  # edm
            steps = self.n_ode_steps
            i = torch.arange(steps, device=self.device, dtype=torch.float32)
            sigmas = (self.sigma_max ** (1 / self.rho) + i / (steps - 1) *
                      (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
                      ) ** self.rho
            sigmas = torch.cat([sigmas, torch.zeros(1, device=self.device)])  # sigma_N = 0
            z = torch.randn(n, self.latent_dim, device=self.device) * self.sigma_max
            for step in range(steps):
                sigma_cur = sigmas[step]
                d = self._edm_denoise(z, sigma_cur.expand(n), c)
                d_over_sigma = (z - d) / sigma_cur          # probability-flow ODE: dz/dsigma
                z = z + (sigmas[step + 1] - sigma_cur) * d_over_sigma  # Euler step

        expr_gen = self.decoder(torch.cat([z, c], dim=-1))
        return {"coords": query["coords"], "expression": expr_gen}

    def training_step(self, batch, batch_idx):
        context, query = batch["context"], batch["query"]
        x_1 = batch["target_expression"]  # real target expression
        n = x_1.shape[0]
        c = self._encode_context(context, query)

        z_1 = self.encoder(x_1)
        recon = self.decoder(torch.cat([z_1, c], dim=-1))
        recon_loss = nn.functional.mse_loss(recon, x_1)

        # flow-matching/diffusion target sees a frozen (detached) latent
        # code, so the encoder/decoder are trained only by recon_loss —
        # approximates a pretrained-then-frozen autoencoder without a
        # separate stage
        z_1_target = z_1.detach()

        if self.path_type == "ot":
            z_0 = torch.randn_like(z_1_target)
            t = torch.rand(n, device=self.device)
            z_t = (1 - t[:, None]) * z_0 + t[:, None] * z_1_target   # OT straight-line path
            target_velocity = z_1_target - z_0                        # constant along a straight line
            pred_velocity = self._velocity(z_t, t, c)
            fm_loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        else:  # edm
            log_sigma = self.edm_p_mean + self.edm_p_std * torch.randn(n, device=self.device)
            sigma = torch.exp(log_sigma)
            noise = torch.randn_like(z_1_target)
            z_sigma = z_1_target + sigma[:, None] * noise
            d_pred = self._edm_denoise(z_sigma, sigma, c)
            weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
            fm_loss = (weight[:, None] * (d_pred - z_1_target) ** 2).mean()

        loss = self.recon_weight * recon_loss + self.fm_weight * fm_loss
        self.log_dict({"train/recon": recon_loss, "train/fm_loss": fm_loss, "train/loss": loss})
        return loss

    def configure_optimizers(self):
        # AdamW (decoupled weight decay) over plain Adam: standard choice in
        # the flow-matching/diffusion literature (Lipman et al. 2022 and
        # essentially all follow-ups use AdamW, not Adam).
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ---------------------------------------------------------------------------
# VQ-VAE + autoregressive transformer (docs/architecture_plan.md
# "Prioritization" #3). Own encoder/decoder/VectorQuantizer (own weights,
# same per-model reasoning as WAE-GAN/FM-OT), reusing VectorQuantizer from
# src/models/vqvae.py (the same EMA + dead-code-reset implementation
# validated standalone in task #11, docs/model_schematics.md) rather than
# nesting a full VQVAEStage1 LightningModule inside another one.
#
# Anchor precedent (docs/literature_review.md SS3.2b, re-verified via live
# search 2026-07-14 after finding the citation had gone stale/dangling in
# our own docs): Tudosiu et al., "Realistic morphology-preserving
# generative modelling of the brain" (Nature Machine Intelligence, 2024) —
# VQ-VAE + autoregressive transformer over discrete tokens, fixed raster
# order, evaluated vs. GAN baselines on FID/MMD.
#
# Token order: their raster order (regular voxel grid) doesn't apply to
# our irregular point cloud, so query locations are ordered along a
# Morton/Z-order space-filling curve instead (morton_order(),
# src/models/vqvae.py) — a deterministic, locality-preserving
# generalization of "fixed raster order" to arbitrary point sets.
#
# One token per cell (docs/model_schematics.md "Resolved" token-granularity
# note, task #11) — the transformer predicts one codebook index per query
# location, conditioned on (a) the previously generated tokens via causal
# self-attention (teacher forcing during training) and (b) that location's
# own conditioning vector c, added into each position's input embedding
# (prefix-style conditioning, not cross-attention — simpler, and c is
# already a fixed-size per-location vector, not a variable-length sequence
# that would need cross-attention).
#
# Sampling is a sequential loop with no KV-cache (recomputes the full
# growing sequence's self-attention every step) — fine at the query-set
# sizes this pipeline currently produces (~15-45 points per masking draw),
# flagged explicitly as a follow-up optimization if larger query sets are
# used later, not built here (docs/model_schematics.md "Known cost").
#
# Stochastic (temperature) sampling at generation time, not greedy argmax:
# the first real-data run (2026-07-14) produced the identical token for
# every query point regardless of conditioning — a documented failure mode
# of greedy decoding in autoregressive generation (Holtzman et al. 2019,
# ICLR, "The Curious Case of Neural Text Degeneration"), and inconsistent
# with the rest of this registry, where every other family samples
# stochastically rather than deterministically.
# ---------------------------------------------------------------------------
@register_model("vqvae_ar")
class VQVAEAutoregressive(BaseGenerativeModel):
    def __init__(self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
                 latent_dim: int = 32, ae_hidden_dim: int = 256,
                 codebook_size: int = 64, commitment_weight: float = 0.25,
                 transformer_dim: int = 128, n_transformer_layers: int = 4,
                 n_heads: int = 4, max_seq_len: int = 2048,
                 recon_weight: float = 1.0, ar_weight: float = 1.0,
                 sample_temperature: float = 1.0, lr: float = 1e-3,
                 image_encoder_type: str = "none", image_feat_dim: int = 64,
                 image_patch_size: int = 256, context_encoder_type: str = "builtin",
                 gene_encoder_type: str = "raw", gene_feat_dim: int = 256,
                 novae_dim: int | None = None,
                 stpath_gene_names: list[str] | None = None, stpath_gene_voc_path: str | None = None,
                 stpath_model_weight_path: str | None = None, stpath_organ_type: str = "Kidney",
                 stpath_tech_type: str = "Visium",
                 stpath_new_gene_encoder_type: str = "none", stpath_novae_dim: int | None = None,
                 stpath_pretrained: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_use_relative_bias: bool = True,
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.context_encoder = _build_context_encoder(
            n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
            context_encoder_type=context_encoder_type,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=gene_feat_dim, novae_dim=novae_dim,
            stpath_gene_names=stpath_gene_names, stpath_gene_voc_path=stpath_gene_voc_path,
            stpath_model_weight_path=stpath_model_weight_path, stpath_organ_type=stpath_organ_type,
            stpath_tech_type=stpath_tech_type,
            stpath_new_gene_encoder_type=stpath_new_gene_encoder_type, stpath_novae_dim=stpath_novae_dim,
            stpath_pretrained=stpath_pretrained,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_use_relative_bias=storm_lite_use_relative_bias,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, n_genes),
        )
        self.vq = VectorQuantizer(codebook_size, latent_dim, commitment_weight)

        self.bos_token = codebook_size  # one extra embedding slot for BOS
        self.token_embed = nn.Embedding(codebook_size + 1, transformer_dim)
        self.pos_embed = nn.Embedding(max_seq_len, transformer_dim)
        self.cond_proj = nn.Linear(cond_hidden_dim, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.output_head = nn.Linear(transformer_dim, codebook_size)

        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len
        self.recon_weight = recon_weight
        self.ar_weight = ar_weight
        self.sample_temperature = sample_temperature
        self.lr = lr

    def _transformer_forward(self, input_tokens: torch.Tensor, c_ordered: torch.Tensor):
        """input_tokens, c_ordered: [N]/[N, cond_hidden_dim]. Returns
        per-position hidden states [N, transformer_dim]."""
        n = input_tokens.shape[0]
        assert n <= self.max_seq_len, (
            f"sequence length {n} exceeds max_seq_len={self.max_seq_len}"
        )
        pos = torch.arange(n, device=input_tokens.device)
        h_in = self.token_embed(input_tokens) + self.pos_embed(pos) + self.cond_proj(c_ordered)
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(input_tokens.device)
        h_out = self.transformer(h_in.unsqueeze(0), mask=mask)
        return h_out.squeeze(0)

    def sample(self, context, query):
        n = query["coords"].shape[0]
        c = self._encode_context(context, query)
        order = morton_order(query["coords"]).to(self.device)
        c_ordered = c[order]

        tokens = torch.full((1,), self.bos_token, dtype=torch.long, device=self.device)
        generated = []
        for i in range(n):
            h = self._transformer_forward(tokens, c_ordered[: tokens.shape[0]])
            logits = self.output_head(h[-1])
            # stochastic (temperature) sampling, not greedy argmax: greedy
            # decoding in autoregressive generation is a documented cause of
            # degenerate repetition collapse (Holtzman et al. 2019, ICLR,
            # "The Curious Case of Neural Text Degeneration") - confirmed as
            # the actual failure mode here 2026-07-14 (real-data run
            # produced the exact same token/expression for every query
            # point). Also more consistent with the rest of this registry,
            # where every other family samples stochastically (z ~ prior),
            # not deterministically.
            probs = torch.softmax(logits / self.sample_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            tokens = torch.cat([tokens, next_token])
        idx = torch.cat(generated)  # [n], in Morton order

        z_q = self.vq.embed[idx]
        expr_gen_ordered = self.decoder(z_q)

        expr_gen = torch.empty_like(expr_gen_ordered)
        expr_gen[order] = expr_gen_ordered
        return {"coords": query["coords"], "expression": expr_gen}

    def training_step(self, batch, batch_idx):
        context, query = batch["context"], batch["query"]
        x_1 = batch["target_expression"]  # real target expression
        n = x_1.shape[0]
        c = self._encode_context(context, query)

        order = morton_order(query["coords"]).to(self.device)
        x_1, c = x_1[order], c[order]

        z_e = self.encoder(x_1)
        z_q, idx, vq_loss = self.vq(z_e)
        recon = self.decoder(z_q)
        recon_loss = nn.functional.mse_loss(recon, x_1)

        idx_detached = idx.detach()
        bos = torch.full((1,), self.bos_token, dtype=torch.long, device=self.device)
        input_tokens = torch.cat([bos, idx_detached[:-1]])
        h = self._transformer_forward(input_tokens, c)
        logits = self.output_head(h)  # [N, codebook_size]
        ar_loss = nn.functional.cross_entropy(logits, idx_detached)

        loss = self.recon_weight * recon_loss + vq_loss + self.ar_weight * ar_loss
        self.log_dict({
            "train/recon": recon_loss, "train/vq": vq_loss,
            "train/ar": ar_loss, "train/loss": loss,
        })
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
