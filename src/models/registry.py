"""
Model registry: lets configs pick a generator backbone by name, so the
rest of the pipeline (data, masking, training loop, evaluation) never needs
to know or care which architecture is in use.

Usage:
    from src.models.registry import build_model

    model = build_model(cfg.model)   # cfg.model.name == "vae_baseline", "diffusion_v1", ...

To add a new architecture:
    1. Implement a class inheriting from BaseGenerator (below).
    2. Register it with @register_model("your_name").
    3. Reference "your_name" in a config file. Nothing else changes.
"""
from __future__ import annotations
import abc
from typing import Any

import torch
import torch.nn as nn

_MODEL_REGISTRY: dict[str, type["BaseGenerator"]] = {}


def register_model(name: str):
    def _wrap(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' already registered.")
        _MODEL_REGISTRY[name] = cls
        return cls
    return _wrap


def build_model(model_cfg: dict) -> "BaseGenerator":
    name = model_cfg["name"]
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[name](**model_cfg.get("params", {}))


class BaseGenerator(nn.Module, abc.ABC):
    """
    Common interface every generative backbone must implement so the
    training loop, masking simulator, and evaluation code are architecture-
    agnostic.

    Conceptually mirrors the Mimyr-style decomposition (see
    docs/literature_review.md) but keeps it generic:
        context   -> spatial/expression info from observed (unmasked) tissue
        query     -> where we want to generate (missing locations / slice)
        output    -> reconstructed (location, cell_type, expression) at query

    A simpler baseline (e.g. plain interpolation or a VAE) can ignore parts
    of this interface it doesn't need (e.g. skip explicit cell-location
    generation and just fill in expression on a fixed grid).
    """

    @abc.abstractmethod
    def forward(self, context: dict[str, torch.Tensor], query: dict[str, Any]
                ) -> dict[str, torch.Tensor]:
        """
        context: dict with keys such as
            'coords'     [N_obs, D]   spatial coords of observed points (D=2 or 3)
            'expression' [N_obs, G]   gene expression of observed points
            'cell_type'  [N_obs]      optional cell type labels/ids
        query: dict describing what to generate, e.g.
            'coords'     [N_query, D] target locations (may itself be predicted
                                       by the model for location-generation tasks)
        Returns a dict with (a subset of):
            'coords'      [N_gen, D]
            'cell_type'   [N_gen]
            'expression'  [N_gen, G]
        """
        raise NotImplementedError

    def loss(self, batch: dict, output: dict) -> torch.Tensor:
        """Override per-model; training loop calls this generically."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Example minimal baseline: nearest-neighbour / linear interpolation "model"
# with no learned parameters. Useful as a sanity-check floor for the metrics
# pipeline before any real model is trained.
# ---------------------------------------------------------------------------
@register_model("interp_baseline")
class InterpolationBaseline(BaseGenerator):
    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k

    def forward(self, context, query):
        coords_obs = context["coords"]          # [N_obs, D]
        expr_obs = context["expression"]         # [N_obs, G]
        coords_q = query["coords"]               # [N_query, D]

        # distance-weighted k-NN interpolation, purely as a floor baseline
        dists = torch.cdist(coords_q, coords_obs)          # [N_query, N_obs]
        knn_d, knn_i = torch.topk(dists, k=min(self.k, dists.shape[1]),
                                   largest=False, dim=1)
        weights = 1.0 / (knn_d + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)
        expr_gen = torch.einsum(
            "nk,nkg->ng", weights, expr_obs[knn_i]
        )
        return {"coords": coords_q, "expression": expr_gen}


# ---------------------------------------------------------------------------
# Stub for a real learned backbone. Fill in per architecture decision
# (diffusion / VAE / flow / GNN) once the literature review + Aim 3 baselines
# are chosen. Keeping this as an explicit stub so the registry pattern is
# demonstrated end-to-end.
# ---------------------------------------------------------------------------
@register_model("vae_baseline")
class VAEBaseline(BaseGenerator):
    def __init__(self, n_genes: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )
        self.latent_dim = latent_dim

    def forward(self, context, query):
        # TODO: condition on query['coords'] (e.g. via a small coordinate
        # encoder concatenated to the latent) once the conditioning strategy
        # is decided. Left unconditioned here as a structural placeholder.
        h = self.encoder(context["expression"])
        mu, logvar = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        expr_gen = self.decoder(z)
        return {"coords": query["coords"], "expression": expr_gen,
                "_mu": mu, "_logvar": logvar}

    def loss(self, batch, output):
        recon = nn.functional.mse_loss(output["expression"], batch["target_expression"])
        kld = -0.5 * torch.mean(
            1 + output["_logvar"] - output["_mu"].pow(2) - output["_logvar"].exp()
        )
        return recon + 1e-3 * kld
