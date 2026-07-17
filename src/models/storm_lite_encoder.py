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
  - Coordinate handling: RandomFourierFeatures (absolute, concatenated
    into each token) PLUS a relative-position attention bias, selectable
    via bias_type. As of 2026-07-17 the DEFAULT is "frame_averaging" —
    STPath's OWN real mechanism, directly verified by cloning
    github.com/Graph-and-Geometric-Learning/STPath and reading
    stpath/model/nn_utils/fa.py + stpath/model/encoder/spatial_transformer.py
    (see FrameAveragingBias's own docstring in conditioning.py for the
    full mechanism and its provable rotation/reflection invariance).
    "relative_position" (the 2026-07-16 Swin-V2-style CPB MLP, added when
    STORM's own mechanism was still unverifiable behind the arxiv block)
    is kept as a still-tested, still-usable alternative for comparison —
    not deleted, since it's a real, working, if less principled, design.
    STORM's own attention/positional mechanism (arXiv 2604.03630) remains
    genuinely unverified (that domain stays blocked by this sandbox's
    network proxy, confirmed repeatedly) — frame_averaging is STPath's
    real mechanism, used here on the reasoning that a directly-verified
    real architecture is preferable to continuing to guess at STORM's.
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
    RelativePositionBias, FrameAveragingBias,
    OrganTechEmbedding,
)


class _MoMETransformerBlock(nn.Module):
    """One spatial-encoder block matching STORM's real design (verified
    2026-07-17 directly from the uploaded manuscript PDF, arXiv 2604.03630
    Online Methods, "Spatial encoder": "Each block contains a shared
    multi-head self-attention (MSA) module and two modality-specific
    feed-forward networks (modality experts). Tokens are routed to the
    appropriate expert based on modality, while the shared MSA aligns
    features across modalities.").

    Requires tokens to carry their own modality identity (an image token
    and a gene token per spot, not summed into one) — StormLiteContext-
    Encoder's original design (fusion_mode="sum") summed img_embed +
    gene_embed + coord_embed into ONE token per spot specifically because
    a single shared FFN doesn't need per-token modality identity; MoME-FFN
    is the opposite design choice, and needs it. fusion_mode="mome"
    switches StormLiteContextEncoder's token layout accordingly (see its
    own forward() for the doubled-sequence-length token construction)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        # two modality experts (image, gene) — matches STORM's own two
        # modalities (H&E, ST); this project's "gene" expert covers
        # whichever gene_encoder_type this run uses, same as the rest of
        # this class already does for the shared-FFN "sum" path
        self.ffn_experts = nn.ModuleDict({
            "image": nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model),
            ),
            "gene": nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model),
            ),
        })

    def forward(self, x: torch.Tensor, is_image_token: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [1, 2N, d_model]. is_image_token: [2N] bool — True for image
        tokens, False for gene tokens (routes each token to its own FFN
        expert after the SHARED attention pass). attn_mask: same [n_heads,
        2N, 2N] or [2N, 2N] additive bias nn.MultiheadAttention accepts."""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask)
        x = x + attn_out

        normed2 = self.norm2(x)
        ffn_out = torch.zeros_like(normed2)
        if is_image_token.any():
            ffn_out[:, is_image_token] = self.ffn_experts["image"](normed2[:, is_image_token])
        if (~is_image_token).any():
            ffn_out[:, ~is_image_token] = self.ffn_experts["gene"](normed2[:, ~is_image_token])
        return x + ffn_out


class StormLiteContextEncoder(nn.Module):
    def __init__(self, n_genes: int, novae_dim: int | None = None, coord_dim: int = 3,
                 hidden_dim: int = 256, rff_features: int = 64, rff_sigma: float = 1.0,
                 coord_scale: float = 1.0,
                 n_transformer_layers: int = 2, n_heads: int = 4,
                 gene_encoder_type: str = "both",
                 bias_type: str = "frame_averaging", relative_bias_hidden_dim: int = 32,
                 fusion_mode: str = "sum",
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        assert gene_encoder_type in ("mlp", "novae", "both"), (
            f"unknown gene_encoder_type {gene_encoder_type!r}"
        )
        assert bias_type in ("none", "relative_position", "frame_averaging"), (
            f"unknown bias_type {bias_type!r}"
        )
        assert fusion_mode in ("sum", "mome"), f"unknown fusion_mode {fusion_mode!r}"
        self.fusion_mode = fusion_mode
        self.gene_encoder_type = gene_encoder_type
        # relative-position attention bias (2026-07-16/17 — see module
        # docstring's "Coordinate handling" for the full reasoning behind
        # each option). "frame_averaging" (default, 2026-07-17): STPath's
        # own real, verified mechanism (FrameAveragingBias, produces a
        # PER-HEAD [n_heads, N, N] bias). "relative_position" (2026-07-16):
        # the earlier ad hoc Swin-V2-style CPB MLP (RelativePositionBias,
        # a single SHARED [N, N] bias across all heads), kept as a
        # comparison arm. "none": no relative bias at all (only absolute
        # RandomFourierFeatures baked into each token below).
        self.bias_type = bias_type
        if bias_type == "frame_averaging":
            self.pos_bias = FrameAveragingBias(n_heads, coord_scale)
        elif bias_type == "relative_position":
            self.pos_bias = RelativePositionBias(coord_dim, relative_bias_hidden_dim)
        else:
            self.pos_bias = None

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
            # combine_mode="concat", not the default "sum" (2026-07-17,
            # same bug already found+fixed for STPath's Route-B residual,
            # see CombinedGeneEncoder's own docstring and commit 1a26a75:
            # summing before a downstream Linear mathematically prevents
            # it from independently weighting the two gene signals — was
            # never exercised as a DISTINCT issue here since ALL THREE
            # gene_encoder_type variants were broken by the separate
            # RelativePositionBias/coord_scale bugs, but worth fixing on
            # the same principle now that those are fixed). Output is
            # 2*hidden_dim wide (output_dim_multiplier=2 for concat mode)
            # — gene_combine_proj below brings it back to hidden_dim so it
            # can still be summed into `tokens` alongside img_embed/
            # coord_embed (this class's fusion tokens are additive, unlike
            # STPath's Route-B residual which has its own dedicated
            # residual_proj for exactly this same width mismatch).
            self.gene_encoder = CombinedGeneEncoder(n_genes, novae_dim, hidden_dim, combine_mode="concat")
            self.gene_combine_proj = nn.Linear(hidden_dim * self.gene_encoder.output_dim_multiplier, hidden_dim)

        # coord_scale (2026-07-17, see RandomFourierFeatures' own docstring)
        # — safe as a per-call-shared FIXED value here since this class
        # calls coord_encoder ONCE on context+query coords concatenated
        # together (unlike SpatialContextEncoder's separate calls).
        self.coord_encoder = RandomFourierFeatures(coord_dim, rff_features, rff_sigma, coord_scale)
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

        # LayerNorm on each branch before combining (2026-07-17, StormLite
        # underperformance investigation) — a fresh-init numerical check
        # found the three branches (image/gene/coord) similarly scaled at
        # construction time, so this ISN'T confirmed as the root cause of
        # StormLite scoring near/below interp_baseline — but scales can
        # still drift apart from each other during training regardless of
        # a similar start, and normalizing each branch before it's
        # combined is standard practice for exactly this failure mode
        # (one branch's gradient/scale coming to dominate a sum). Applied
        # in BOTH fusion_mode paths below.
        self.img_norm = nn.LayerNorm(hidden_dim)
        self.gene_norm = nn.LayerNorm(hidden_dim)
        self.coord_norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        if fusion_mode == "mome":
            # 2026-07-17: real STORM detail (Online Methods, "Spatial
            # encoder" — see _MoMETransformerBlock's own docstring) —
            # needs separate image/gene tokens (not summed) to route
            # through per-modality FFN experts after shared attention.
            # modality_embed: STORM's own M_i term (formula
            # H_{0,i} = H_i + M_i + P_i) — a learned per-modality offset,
            # one row per modality (0=image, 1=gene).
            self.modality_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
            self.mome_blocks = nn.ModuleList([
                _MoMETransformerBlock(hidden_dim, n_heads) for _ in range(n_transformer_layers)
            ])
            # c must be ONE vector per query spot, not per (spot,
            # modality) pair — combines each query spot's final image-
            # token and gene-token representations after the MoME blocks.
            self.mome_output_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        else:  # "sum" — original design, unchanged
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # 2026-07-16, multi-sample training follow-up — same additive
        # organ/tech offset as SpatialContextEncoder (see that class and
        # OrganTechEmbedding's own docstring for the real caveat: only
        # meaningful once training spans genuinely different
        # organs/platforms). Added post-fusion (to `tokens`, before the
        # transformer sees them) rather than to `fused`, so the fusion
        # transformer's self-attention can actually condition on it —
        # adding it only at the very end would make it a pure output
        # offset every downstream model would trivially fold into its own
        # bias term, same failure mode this project already corrected
        # once for CombinedGeneEncoder's sum-vs-concat issue.
        self.organ_tech_embed = None
        if organ_vocab is not None and tech_vocab is not None:
            self.organ_tech_embed = OrganTechEmbedding(organ_vocab, tech_vocab, hidden_dim)

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
            # CombinedGeneEncoder(combine_mode="concat") output is
            # 2*hidden_dim wide — gene_combine_proj brings it back to
            # hidden_dim (see __init__'s own comment)
            combined = self.gene_encoder(torch.log1p(raw_expr), novae_features)
            return self.gene_combine_proj(combined)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor,
                query_images: torch.Tensor,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
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
        coord_embed = self.coord_norm(self.coord_proj(self.coord_encoder(coords)))

        img_embed = self.img_norm(torch.cat([
            self.image_encoder(context_images), self.image_encoder(query_images),
        ], dim=0))

        gene_embed_raw = torch.zeros(n_total, self.hidden_dim, device=device)
        gene_embed_raw[:n_context] = self._encode_gene(context_expression, context_novae_features)
        gene_embed_raw[n_context:] = self.mask_token  # broadcasts over n_query rows
        gene_embed = self.gene_norm(gene_embed_raw)

        if self.fusion_mode == "sum":
            tokens = img_embed + gene_embed + coord_embed  # [N_total, hidden_dim]
            if self.organ_tech_embed is not None and organ is not None and tech is not None:
                tokens = tokens + self.organ_tech_embed(organ, tech, n_total, device)
            # additive attention bias (see bias_type in __init__) — a
            # FloatTensor `mask` is documented PyTorch behavior for an
            # additive bias added to raw attention logits before softmax,
            # not a boolean keep/drop mask; accepts either a single
            # [N, N] bias shared across heads (RelativePositionBias) or a
            # per-head [n_heads, N, N] bias (FrameAveragingBias) — both
            # verified directly against a real nn.TransformerEncoder call
            # (see tests/test_frame_averaging_bias.py). None
            # (bias_type="none") preserves plain self-attention with no
            # relative-position term.
            bias = self.pos_bias(coords) if self.pos_bias is not None else None
            fused = self.transformer(tokens.unsqueeze(0), mask=bias).squeeze(0)  # self-attention over ALL spots
            return fused[n_context:]  # query positions only
        else:  # "mome" — see _MoMETransformerBlock's own docstring
            img_tokens = img_embed + coord_embed + self.modality_embed[0]
            gene_tokens = gene_embed + coord_embed + self.modality_embed[1]
            if self.organ_tech_embed is not None and organ is not None and tech is not None:
                offset = self.organ_tech_embed(organ, tech, n_total, device)
                img_tokens = img_tokens + offset
                gene_tokens = gene_tokens + offset
            # [1, 2*N_total, hidden_dim] — image tokens first, then gene
            # tokens, same order is_image_token/coords_doubled below use
            tokens = torch.cat([img_tokens, gene_tokens], dim=0).unsqueeze(0)
            is_image_token = torch.cat([
                torch.ones(n_total, dtype=torch.bool, device=device),
                torch.zeros(n_total, dtype=torch.bool, device=device),
            ])
            bias = None
            if self.pos_bias is not None:
                # coords duplicated to match the doubled token sequence —
                # the image token and gene token for the SAME spot share
                # the SAME coordinate, so the bias between them is
                # exactly the "self" (zero relative-offset) case, letting
                # a spot's own image/gene tokens attend to each other
                # most strongly by default, same-spot cross-modality
                # information exchange being the whole point of a shared
                # attention pass over both modalities' tokens.
                coords_doubled = torch.cat([coords, coords], dim=0)
                bias = self.pos_bias(coords_doubled)
            for block in self.mome_blocks:
                tokens = block(tokens, is_image_token, attn_mask=bias)
            tokens = tokens.squeeze(0)  # [2*N_total, hidden_dim]
            img_out, gene_out = tokens[:n_total], tokens[n_total:]
            combined = self.mome_output_proj(torch.cat([img_out, gene_out], dim=-1))  # [N_total, hidden_dim]
            return combined[n_context:]  # query positions only
