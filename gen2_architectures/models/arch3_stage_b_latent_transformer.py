"""Architecture 3, Stage B — spatial transformer predicting a LATENT code.

GPT's own favorite / most novel idea (its PDF's "My preferred solution" and
the review's #2 pick, "Out of the four, I would actually bet on Architecture
3 winning"): decompose the problem into (a) Stage A's self-supervised
denoising transcriptome autoencoder (arch3_stage_a_autoencoder.py, trained
first, separately) and (b) this spatial transformer, which never sees or
predicts raw gene space directly — it predicts the missing spot's LATENT
code, decoded back to genes by Stage A's own decoder.

Flow (see gen2_architectures/README.md for the full spec with dimensions):

    H&E patch -> frozen GigaPath                                -> image_feat [feat_dim]
    context expression -> Stage-A encoder (frozen or fine-tuned) -> gene_latent [latent_dim]
    (query_xy - context_xy) -> CoordEmbedding                    -> coord_feat [coord_dim]
    context image_available -> ConfidenceEmbedding                -> conf_feat [conf_dim]
    [image_feat; gene_latent; coord_feat; conf_feat] -> Linear -> spot_token [hidden_dim]
    per query spot: its own k=max_neighbors nearest context spots + 1
      learnable query token -> shared local transformer
    -> query token's final hidden state -> Linear(hidden_dim, latent_dim) -> predicted_latent
    predicted_latent -> Stage-A decoder (frozen or fine-tuned) -> predicted expression [n_genes]

Loss has two terms (GPT review answer to open question #2: keep the
latent-space loss, the decoder already constrains the latent to stay
biologically meaningful so this doesn't over-constrain the transformer):
  1. latent_loss: MSE(predicted_latent, true_latent) where true_latent =
     Stage-A's own encoder applied to the REAL (uncorrupted) expression of
     the masked/query spot — teacher-forced, available during training
     since this is supervised masking, not genuinely missing data.
  2. gene_loss: StagedGeneLoss(decoded_prediction, true_expression,
     progress) — the same staged MSE->Pearson schedule as Architectures
     1/2, applied AFTER decoding, so the model is never purely optimizing
     an internal latent nobody checks against real genes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen2_architectures.models.arch3_stage_a_autoencoder import DenoisingTranscriptomeAutoencoder
from gen2_architectures.models.conditioning import GigapathPatchEncoder, OrganTechEmbedding
from gen2_architectures.models.components import CoordEmbedding, ConfidenceEmbedding, build_local_transformer
from gen2_architectures.models.local_neighborhood_transformer import nearest_context_neighbors
from gen2_architectures.models.eval_compat import DeterministicLatentSampleMixin


class Architecture3StageB(DeterministicLatentSampleMixin, nn.Module):
    def __init__(
        self, autoencoder: DenoisingTranscriptomeAutoencoder,
        finetune_autoencoder: bool = False, autoencoder_lr_multiplier: float = 0.1,
        feat_dim: int = 256, coord_dim: int = 64, conf_dim: int = 16,
        hidden_dim: int = 512, n_layers: int = 8, n_heads: int = 8, mlp_ratio: float = 4.0,
        dropout: float = 0.1, max_neighbors: int = 80, coord_scale: float = 1000.0,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.finetune_autoencoder = bool(finetune_autoencoder)
        # autoencoder_lr_multiplier is read by the training entrypoint's
        # optimizer param-group construction (GPT's "fine-tune at a much
        # smaller LR than the transformer") -- stored here so the config
        # that builds this module is the single source of truth for it,
        # not duplicated into the training script separately.
        self.autoencoder_lr_multiplier = float(autoencoder_lr_multiplier)
        # Explicit either way (not just "skip freezing" for the finetune
        # case) — real bug caught in testing: a Stage-A checkpoint could
        # arrive already frozen (e.g. reused across multiple StageB
        # instances, or frozen by a previous caller), and requires_grad is
        # NOT part of state_dict, so it never gets reset by loading
        # weights alone. finetune_autoencoder=True must therefore actively
        # re-enable grad, not merely decline to disable it.
        for p in self.autoencoder.parameters():
            p.requires_grad_(bool(finetune_autoencoder))
        self.n_genes = autoencoder.n_genes
        self.latent_dim = autoencoder.latent_dim
        self.max_neighbors = int(max_neighbors)
        self.hidden_dim = int(hidden_dim)

        self.image_encoder = GigapathPatchEncoder(feat_dim=feat_dim)
        self.coord_embed = CoordEmbedding(feat_dim=coord_dim, coord_scale=coord_scale)
        self.conf_embed = ConfidenceEmbedding(embed_dim=conf_dim)

        self.organ_tech = None
        self.organ_tech_proj = None
        if organ_vocab and tech_vocab:
            self.organ_tech = OrganTechEmbedding(organ_vocab, tech_vocab, hidden_dim=hidden_dim)
            self.organ_tech_proj = nn.Linear(hidden_dim, hidden_dim)

        token_in_dim = feat_dim + self.latent_dim + coord_dim + conf_dim
        self.token_proj = nn.Linear(token_in_dim, hidden_dim)
        self.query_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.transformer = build_local_transformer(
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            mlp_ratio=mlp_ratio, dropout=dropout,
        )
        self.latent_head = nn.Linear(hidden_dim, self.latent_dim)

    def _encode_context_genes(self, expression: torch.Tensor) -> torch.Tensor:
        if self.finetune_autoencoder:
            return self.autoencoder.encode(expression)
        with torch.no_grad():
            return self.autoencoder.encode(expression)

    def forward(self, context: dict, query: dict) -> dict:
        """Returns {"predicted_latent": [N_query, latent_dim],
        "predicted_expression": [N_query, n_genes]} — callers needing the
        latent loss use the first key; the gene-space loss uses the
        second (already decoded through Stage A)."""
        device = context["expression"].device
        context_xy = context["coords"][:, :2]
        query_xy = query["coords"][:, :2]
        n_query = query_xy.shape[0]

        image_feat = self.image_encoder(context["images"])
        gene_latent = self._encode_context_genes(context["expression"])
        n_context = context["expression"].shape[0]
        conf_feat = self.conf_embed(
            context.get("image_available", torch.ones(n_context, dtype=torch.bool, device=device))
        )

        neighbor_idx = nearest_context_neighbors(context_xy, query_xy, self.max_neighbors)
        k = neighbor_idx.shape[1]

        gathered_image = image_feat[neighbor_idx]
        gathered_latent = gene_latent[neighbor_idx]
        gathered_conf = conf_feat[neighbor_idx]
        relative_xy = context_xy[neighbor_idx] - query_xy.unsqueeze(1)
        coord_feat = self.coord_embed(relative_xy.reshape(-1, 2)).reshape(n_query, k, -1)

        tokens = torch.cat([gathered_image, gathered_latent, coord_feat, gathered_conf], dim=-1)
        tokens = self.token_proj(tokens)

        if self.organ_tech is not None and "organ" in context and "tech" in context:
            organ_tech_feat = self.organ_tech(context["organ"], context["tech"], n_query, device)
            tokens = tokens + self.organ_tech_proj(organ_tech_feat).unsqueeze(1)

        query_tok = self.query_token.unsqueeze(0).unsqueeze(0).expand(n_query, 1, -1)
        full_tokens = torch.cat([tokens, query_tok], dim=1)

        encoded = self.transformer(full_tokens)
        query_hidden = encoded[:, -1, :]
        predicted_latent = self.latent_head(query_hidden)
        predicted_expression = self.autoencoder.decode(predicted_latent)
        return {"predicted_latent": predicted_latent, "predicted_expression": predicted_expression}

    def true_latent(self, true_expression: torch.Tensor) -> torch.Tensor:
        """Teacher-forced target for the latent-space loss term: Stage-A's
        own encoder applied to the REAL (uncorrupted) expression of the
        query spot. no_grad even when finetune_autoencoder=True — this is
        a TARGET, not something the latent loss should push the
        autoencoder's own weights toward matching its own transformer-
        predicted counterpart (that would be a degenerate shortcut)."""
        with torch.no_grad():
            return self.autoencoder.encode(true_expression)
