"""Shared implementation for Architectures 1 and 2.

Both are "frozen pathology image encoder + local-neighborhood transformer +
dense decoder" — the ONLY real difference between them is which gene
encoder processes context["expression"], and what semantic content
context["expression"] actually holds by the time it reaches the model
(raw normalized-log1p expression for Architecture 1's from-scratch MLP;
precomputed frozen scFoundation cell embeddings for Architecture 2 — see
gen2_architectures/training/data_prep.py's context_gene_feature_provider
wiring, which is what actually populates context["expression"] with one or
the other before this module ever sees it). Mirrors the gene_encoder_type
dispatch pattern already established in the original codebase
(src/models/simple_fusion_encoder.py::_build_gene_encoder) rather than
duplicating this entire module twice for a one-line difference.

See arch1_gpt_baseline.py / arch2_scfoundation.py for the two thin,
named factory functions configs actually point at.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen2_architectures.models.conditioning import (
    GigapathPatchEncoder, MLPGeneEncoder, ScFoundationGeneEncoder, OrganTechEmbedding,
)
from gen2_architectures.models.components import CoordEmbedding, ConfidenceEmbedding, build_local_transformer
from gen2_architectures.models.eval_compat import DeterministicSampleMixin


def nearest_context_neighbors(context_xy: torch.Tensor, query_xy: torch.Tensor, k: int) -> torch.Tensor:
    """For each query spot, the indices of its k nearest context spots.

    context_xy: [N_context, 2]. query_xy: [N_query, 2]. Returns
    [N_query, min(k, N_context)] long indices into context_xy's first
    dimension. This is the actual core of GPT's "small neighborhood
    transformer" design: EACH query spot gets its own tight local window,
    not one shared context capped jointly for a whole (possibly larger,
    irregular) hole the way src/data/mask_bank.py::cap_context_mask's
    "nearest_query" mode does. Pure indexing (torch.cdist + topk) — no
    gradient needed through neighbor SELECTION, only through the features
    gathered afterward, which stays differentiable."""
    k = min(k, context_xy.shape[0])
    with torch.no_grad():
        distances = torch.cdist(query_xy, context_xy)  # [N_query, N_context]
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
    return indices


def _build_gene_encoder(gene_encoder_type: str, n_genes: int, feat_dim: int,
                         scfoundation_dim: int | None) -> nn.Module:
    if gene_encoder_type == "mlp":
        return MLPGeneEncoder(n_genes=n_genes, feat_dim=feat_dim)
    if gene_encoder_type == "scfoundation":
        if scfoundation_dim is None:
            raise ValueError("gene_encoder_type='scfoundation' requires scfoundation_dim")
        return ScFoundationGeneEncoder(scfoundation_dim=scfoundation_dim, feat_dim=feat_dim)
    raise ValueError(f"unknown gene_encoder_type {gene_encoder_type!r}; expected 'mlp' or 'scfoundation'")


class LocalNeighborhoodTransformer(DeterministicSampleMixin, nn.Module):
    """Frozen GigaPath + {MLP | frozen scFoundation} gene encoder + a small
    per-query-spot local-neighborhood transformer + dense decoder.

    forward()'s per-query-spot neighborhood construction, token fusion, and
    query-token readout are IDENTICAL for both gene encoder choices — the
    gene encoder is applied uniformly to context["expression"], whatever
    that array's contents actually are (see module docstring)."""

    def __init__(
        self, n_genes: int, gene_encoder_type: str = "mlp",
        scfoundation_dim: int | None = None,
        feat_dim: int = 256, coord_dim: int = 64, conf_dim: int = 16,
        hidden_dim: int = 512, n_layers: int = 8, n_heads: int = 8, mlp_ratio: float = 4.0,
        dropout: float = 0.1, max_neighbors: int = 80, coord_scale: float = 1000.0,
        decoder_hidden_dim: int = 1024,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.max_neighbors = int(max_neighbors)
        self.hidden_dim = int(hidden_dim)

        self.image_encoder = GigapathPatchEncoder(feat_dim=feat_dim)
        self.gene_encoder = _build_gene_encoder(gene_encoder_type, n_genes, feat_dim, scfoundation_dim)
        self.coord_embed = CoordEmbedding(feat_dim=coord_dim, coord_scale=coord_scale)
        self.conf_embed = ConfidenceEmbedding(embed_dim=conf_dim)

        self.organ_tech = None
        self.organ_tech_proj = None
        if organ_vocab and tech_vocab:
            self.organ_tech = OrganTechEmbedding(organ_vocab, tech_vocab, hidden_dim=hidden_dim)
            self.organ_tech_proj = nn.Linear(hidden_dim, hidden_dim)

        token_in_dim = feat_dim + feat_dim + coord_dim + conf_dim
        self.token_proj = nn.Linear(token_in_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.transformer = build_local_transformer(
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, decoder_hidden_dim), nn.GELU(), nn.LayerNorm(decoder_hidden_dim),
            nn.Linear(decoder_hidden_dim, n_genes),
        )

    def forward(self, context: dict, query: dict) -> torch.Tensor:
        """context/query: see gen2_architectures/data/masked_item.py's
        build_masked_item output. Returns predicted expression
        [N_query, n_genes]."""
        device = context["expression"].device
        context_xy = context["coords"][:, :2]
        query_xy = query["coords"][:, :2]
        n_query = query_xy.shape[0]

        image_feat = self.image_encoder(context["images"])
        gene_feat = self.gene_encoder(context["expression"])
        n_context = context["expression"].shape[0]
        conf_feat = self.conf_embed(
            context.get("image_available", torch.ones(n_context, dtype=torch.bool, device=device))
        )

        neighbor_idx = nearest_context_neighbors(context_xy, query_xy, self.max_neighbors)  # [n_query, k]
        k = neighbor_idx.shape[1]

        gathered_image = image_feat[neighbor_idx]      # [n_query, k, feat_dim]
        gathered_gene = gene_feat[neighbor_idx]         # [n_query, k, feat_dim]
        gathered_conf = conf_feat[neighbor_idx]         # [n_query, k, conf_dim]
        relative_xy = context_xy[neighbor_idx] - query_xy.unsqueeze(1)  # [n_query, k, 2]
        coord_feat = self.coord_embed(relative_xy.reshape(-1, 2)).reshape(n_query, k, -1)

        tokens = torch.cat([gathered_image, gathered_gene, coord_feat, gathered_conf], dim=-1)
        tokens = self.token_proj(tokens)  # [n_query, k, hidden_dim]

        if self.organ_tech is not None and "organ" in context and "tech" in context:
            organ_tech_feat = self.organ_tech(context["organ"], context["tech"], n_query, device)
            tokens = tokens + self.organ_tech_proj(organ_tech_feat).unsqueeze(1)

        query_tok = self.query_token.unsqueeze(0).unsqueeze(0).expand(n_query, 1, -1)  # [n_query, 1, hidden_dim]
        full_tokens = torch.cat([tokens, query_tok], dim=1)  # [n_query, k+1, hidden_dim]

        encoded = self.transformer(full_tokens)  # [n_query, k+1, hidden_dim]
        query_hidden = encoded[:, -1, :]  # last position is always the query token
        return self.decoder(query_hidden)
