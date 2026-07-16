"""
"STORM-lite" fusion transformer context encoder (2026-07-16).

Tests the STORM recipe (Xu et al., arXiv 2604.03630, "A Multimodal
Foundation Model of Spatial Transcriptomics and Histology for Biological
Discovery and Clinical Prediction") at pilot scale: reuse strong,
independently pretrained per-modality encoders (frozen GigaPath for
images, frozen Novae and/or a task-trained MLP for gene expression — see
gene_encoder_type below) and train ONLY a comparatively light fusion
transformer on top — instead of training an entire architecture
(image/gene/organ/tech embeddings AND the fusion transformer) jointly
from scratch the way STPath's own pretraining did.

Direct comparison target: STPathContextEncoder(pretrained=False) — same
"train the fusion on our own pilot data" starting point, but STPath's
version also has to relearn its image/gene embeddings from scratch
alongside its transformer, since none of STPath's own submodules are
separately pretrained (except GigaPath, which it also reuses frozen).
This class tests whether reusing MORE pretrained pieces (both modalities,
not just image) needs LESS data to produce a working fusion — the actual
question behind "is STORM's recipe easier because it took pretrained
models and added a small fusion" (2026-07-16 discussion).

gene_encoder_type reuses the exact same MLPGeneEncoder/NovaeGeneEncoder/
CombinedGeneEncoder classes already built for STPathContextEncoder's
Route-B residual (src/models/conditioning.py) — "mlp" (task-trained,
no pretraining), "novae" (frozen pretrained), "both" (summed) — giving
the same 3-way gene-side ablation for free, on top of this new
fusion-architecture arm.

NOT a verified reproduction of STORM's real fusion transformer — flagged
explicitly, not hidden (this codebase's established practice). Its paper
(arXiv 2604.03630) is on a domain this sandbox's network proxy blocks
(confirmed 2026-07-16 via several direct/mirror attempts, all 403) — the
only confirmed detail (from earlier session research, before that block
was hit) is the high-level shape: per-spot pretrained encoders -> a
spatial encoder fusing neighbor/coordinate context, reportedly over local
5x5 spot windows for Visium-HD-scale density. Exact attention
mechanism/positional encoding/layer count were never independently
verified. Given that, this class is grounded instead in what IS directly
confirmed accessible: MultiST (a real, published cross-attention ST
fusion model) uses a standard Transformer block (LayerNorm + multi-head
self-attention + linear layers) over spot tokens combining morphology and
expression — the same general "attention-based spatial fusion over
per-modality tokens" recipe, implemented here with a plain
nn.TransformerEncoder rather than attempting to match STORM's specific
(unverified) internals.

Simplifications, explicit:
  - Full self-attention over every context+query spot at once (no
    windowing) — our pilot dataset is ~1000 spots per masking draw, well
    within what plain multi-head attention handles; STORM's windowing
    (if the 5x5 detail above is accurate) is a scaling optimization for
    much higher-density data (Visium HD), not fundamental to the
    question this class tests.
  - Coordinate handling: RandomFourierFeatures (already used throughout
    this file), concatenated additively into each token — not STPath's
    own geometry-aware frame-averaging attention bias (verified via
    STPath's real source, see stpath_encoder.py), which STORM may or may
    not also use — unconfirmed either way.
  - STORM's own image encoder is H0-mini, not GigaPath — kept GigaPath
    here for consistency with every other arm in this project's
    comparisons (isolates the fusion-architecture question from a
    separate "which image encoder" question — already a distinct,
    deferred axis per this project's own research notes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.conditioning import (
    RandomFourierFeatures, GigapathPatchEncoder,
    MLPGeneEncoder, NovaeGeneEncoder, CombinedGeneEncoder,
)


class StormLiteContextEncoder(nn.Module):
    def __init__(self, n_genes: int, novae_dim: int | None = None, coord_dim: int = 3,
                 hidden_dim: int = 256, rff_features: int = 64, rff_sigma: float = 1.0,
                 n_transformer_layers: int = 2, n_heads: int = 4,
                 gene_encoder_type: str = "both"):
        super().__init__()
        assert gene_encoder_type in ("mlp", "novae", "both"), (
            f"unknown gene_encoder_type {gene_encoder_type!r}"
        )
        self.gene_encoder_type = gene_encoder_type

        # Per-modality encoders, each projecting to hidden_dim so they can
        # be summed into one token — same additive-fusion pattern STPath's
        # own EncodeInputs uses (img_embed + ge_embed + ...), verified via
        # its real source (see stpath_encoder.py's _ResidualEncodeInputs
        # docstring). GigaPath: frozen+pretrained (RAE pattern, same as
        # every other use of it in this file). Gene side: whichever of
        # MLP (task-trained, no pretraining)/Novae (frozen pretrained)/
        # both this run is testing.
        self.image_encoder = GigapathPatchEncoder(hidden_dim)
        if gene_encoder_type == "mlp":
            self.gene_encoder = MLPGeneEncoder(n_genes, hidden_dim)
        elif gene_encoder_type == "novae":
            assert novae_dim is not None, "gene_encoder_type='novae' requires novae_dim"
            self.gene_encoder = NovaeGeneEncoder(novae_dim, hidden_dim)
        else:  # "both"
            assert novae_dim is not None, "gene_encoder_type='both' requires novae_dim"
            self.gene_encoder = CombinedGeneEncoder(n_genes, novae_dim, hidden_dim)

        self.coord_encoder = RandomFourierFeatures(coord_dim, rff_features, rff_sigma)
        self.coord_proj = nn.Linear(2 * rff_features, hidden_dim)

        # Learned placeholder for query positions' missing gene signal —
        # same role as STPath's own mask_token (its GeneExpTokenizer), but
        # our own trainable parameter, since this class doesn't share
        # STPath's tokenizer/vocabulary at all (no gene_voc_path, no
        # gene_names dependency — a genuine simplification vs STPath,
        # possible specifically because Novae/MLP don't need a fixed
        # cross-dataset gene vocabulary the way STPath's tokenized
        # architecture does).
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.hidden_dim = hidden_dim

    def _encode_gene(self, raw_expr: torch.Tensor,
                      novae_features: torch.Tensor | None) -> torch.Tensor:
        # log1p for numerical stability, same convention used for
        # MLPGeneEncoder/CombinedGeneEncoder everywhere else in this
        # codebase (stpath_encoder.py's residual path).
        if self.gene_encoder_type == "mlp":
            return self.gene_encoder(torch.log1p(raw_expr))
        elif self.gene_encoder_type == "novae":
            assert novae_features is not None, (
                "gene_encoder_type='novae' requires context_novae_features"
            )
            return self.gene_encoder(novae_features)
        else:  # "both"
            assert novae_features is not None, (
                "gene_encoder_type='both' requires context_novae_features"
            )
            return self.gene_encoder(torch.log1p(raw_expr), novae_features)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor,
                query_images: torch.Tensor,
                context_novae_features: torch.Tensor | None = None) -> torch.Tensor:
        """context_images/query_images: raw H&E patches OR precomputed
        GigaPath features (GigapathPatchEncoder dispatches on tensor rank
        — see its own docstring). context_novae_features: [N_context,
        novae_dim] precomputed Novae features, only required when
        gene_encoder_type != "mlp". Returns c [N_query, hidden_dim].

        Query positions get a real image token (H&E at query locations is
        genuinely available — only expression is being predicted, never
        masked in the input) but the LEARNED mask_token for gene signal
        (real expression there would leak the prediction target) — same
        context/query asymmetry STPath's own ge_tokens already encode,
        and the same reasoning STPathContextEncoder's residual_embed
        zero-fills query rows for."""
        n_context = context_coords.shape[0]
        n_query = query_coords.shape[0]
        n_total = n_context + n_query
        device = context_coords.device

        coords = torch.cat([context_coords, query_coords], dim=0)
        coord_embed = self.coord_proj(self.coord_encoder(coords))

        img_embed = torch.cat([
            self.image_encoder(context_images), self.image_encoder(query_images),
        ], dim=0)

        gene_embed = torch.zeros(n_total, self.hidden_dim, device=device)
        gene_embed[:n_context] = self._encode_gene(context_expression, context_novae_features)
        gene_embed[n_context:] = self.mask_token  # broadcasts over n_query rows

        tokens = img_embed + gene_embed + coord_embed  # [N_total, hidden_dim]
        fused = self.transformer(tokens.unsqueeze(0)).squeeze(0)  # self-attention over ALL spots
        return fused[n_context:]  # query positions only
