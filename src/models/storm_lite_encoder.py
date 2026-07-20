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
    MLPGeneEncoder, NovaeGeneEncoder, CombinedGeneEncoder, TokenizedGeneEncoder,
    RelativePositionBias, FrameAveragingBias,
    OrganTechEmbedding,
)


def _knn_additive_mask(coords: torch.Tensor, k: int) -> torch.Tensor:
    """Additive attention-bias mask (2026-07-20) restricting attention to
    each token's k nearest spatial neighbors (by real coordinate distance,
    always including itself) — ports STFlow's real design choice (Huang
    et al. 2025, "Scalable Generation of Spatial Transcriptomics from
    Histology Images via Whole-Slide Flow Matching", arXiv 2506.05361,
    Section 3.3 "Local Spatial Context" — verified via direct PDF read,
    2026-07-20 literature pass): restricting attention to spatial
    neighbors is a real, published design choice for exactly this kind of
    spot-to-spot spatial transformer, not this project's own guess.

    HONEST SCOPE NOTE: STFlow's own implementation only COMPUTES attention
    over the k neighbors (genuinely O(Nk), their real motivation — see
    their Figure 5 memory comparison). This function instead masks a
    DENSE full N x N attention (0 additive bias on kept edges, -inf on
    dropped edges) computed exactly as before — a real inductive-bias
    change (forces softmax to assign zero weight to spatially distant
    tokens) worth testing on its own merits, but it does NOT reduce
    memory/compute the way STFlow's sparse implementation does. Memory is
    handled separately by masking.max_context_points (src/training/
    train.py, added the same day for a real OOM this was originally
    scoped alongside) — that's still the mechanism to lower if memory is
    the concern; this is purely an architecture/inductive-bias lever.

    k clipped to min(k, N) so a value larger than the actual token count
    never errors — degrades gracefully to full attention (no masking)."""
    N = coords.shape[0]
    k = min(k, N)
    dist = torch.cdist(coords, coords)  # [N, N]
    _, nn_idx = torch.topk(dist, k, dim=-1, largest=False)  # [N, k], nearest incl. self (dist=0)
    mask = torch.full((N, N), float("-inf"), device=coords.device, dtype=coords.dtype)
    mask.scatter_(1, nn_idx, 0.0)
    return mask


class _QKNormAttention(nn.Module):
    """Multi-head self-attention with QK-normalization (Henry et al. 2020,
    EMNLP, "Query-Key Normalization for Transformers"; the LayerNorm-over-
    head-dim variant popularized by ViT-22B, Dehghani et al. 2023, and
    used in Stable Diffusion 3's DiT, Esser et al. 2024) — LayerNorms each
    head's queries and keys before the dot product, so attention LOGIT
    MAGNITUDE is bounded by construction instead of growing with the
    (learnable) Q/K projection norms during training.

    Added 2026-07-19 as the real structural fix for this project's "bigger"
    StormLite mode-collapse: the 4-layer/8-head/512-dim config collapsed to
    a constant output (PCC=nan) in both single- and multi-sample settings,
    and gradient_clip_val=1.0 made it WORSE (ST-FID 9->378), because
    clipping caps gradient norm without addressing the underlying
    attention-entropy collapse (softmax saturating to near-one-hot as
    logits explode) that deeper/wider transformers are documented to hit
    at a flat LR. QK-norm targets that root cause directly — it's the
    standard fix for exactly this failure mode in large-model training.

    Reimplements attention manually (rather than nn.MultiheadAttention)
    ONLY because nn.MultiheadAttention gives no access to Q/K between
    projection and the dot product. Preserves the same call contract
    _MoMETransformerBlock already used: a single [1, L, d_model] input and
    an optional ADDITIVE attn_mask of shape [n_heads, L, L] or [L, L]
    (FrameAveragingBias's per-head bias, or RelativePositionBias's shared
    one) — added to the scaled, QK-normed logits before softmax, exactly
    as nn.MultiheadAttention's float attn_mask does. batch_first, batch
    size 1 (this project's whole design — one masking draw per step)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # QK-norm: LayerNorm over each head's own head_dim channels
        # (elementwise_affine=True — the learnable gain lets the model
        # recover a useful logit temperature, but starting bounded)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [1, L, d_model]
        _, L, _ = x.shape
        # [1, L, d] -> [1, n_heads, L, head_dim]
        def _split(t):
            return t.view(1, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(_split(self.q_proj(x)))   # QK-norm on queries
        k = self.k_norm(_split(self.k_proj(x)))   # QK-norm on keys
        v = _split(self.v_proj(x))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1, n_heads, L, L]
        if attn_mask is not None:
            # additive bias, broadcast to [1, n_heads, L, L] whether it's
            # [n_heads, L, L] (per-head, FrameAveragingBias) or [L, L]
            # (shared, RelativePositionBias) — same semantics as
            # nn.MultiheadAttention's float attn_mask.
            logits = logits + attn_mask.unsqueeze(0)
        attn = self.dropout(torch.softmax(logits, dim=-1))
        out = torch.matmul(attn, v)                  # [1, n_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(1, L, self.n_heads * self.head_dim)
        return self.out_proj(out)


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
    own forward() for the doubled-sequence-length token construction).

    qk_norm (2026-07-19, opt-in): when True, swaps the shared MSA's
    nn.MultiheadAttention for _QKNormAttention (see its own docstring) —
    the real structural fix for the "bigger" StormLite collapse. False
    (default) keeps nn.MultiheadAttention exactly as before, byte-for-byte
    unchanged behavior for every existing config."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 qk_norm: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.qk_norm = qk_norm
        if qk_norm:
            self.attn = _QKNormAttention(d_model, n_heads, dropout=dropout)
        else:
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
        if self.qk_norm:
            attn_out = self.attn(normed, attn_mask=attn_mask)
        else:
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
                 tokenizer_gene_names: list[str] | None = None,
                 tokenizer_full_gene_names: list[str] | None = None,
                 tokenizer_n_pool_layers: int = 1, tokenizer_n_pool_heads: int = 4,
                 bias_type: str = "frame_averaging", relative_bias_hidden_dim: int = 32,
                 fusion_mode: str = "sum", qk_norm: bool = False,
                 knn_k: int | None = None,
                 input_already_log1p: bool = False,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        assert gene_encoder_type in ("mlp", "novae", "both", "tokenizer", "tokenizer_novae"), (
            f"unknown gene_encoder_type {gene_encoder_type!r}"
        )
        assert bias_type in ("none", "relative_position", "frame_averaging"), (
            f"unknown bias_type {bias_type!r}"
        )
        assert fusion_mode in ("sum", "mome"), f"unknown fusion_mode {fusion_mode!r}"
        assert knn_k is None or knn_k >= 1, f"knn_k must be >= 1 or None, got {knn_k!r}"
        self.knn_k = knn_k
        self.fusion_mode = fusion_mode
        self.gene_encoder_type = gene_encoder_type
        # input_already_log1p (2026-07-19, real audit finding — see
        # _encode_gene): basic_qc_and_normalize (src/data/loaders.py)
        # ALREADY applies sc.pp.log1p to adata.X, and _encode_gene then
        # applies torch.log1p AGAIN — a genuine double-log1p that squashes
        # the already-log-normalized expression's dynamic range a second
        # time (log1p([0, ~9.2]) -> [0, ~2.3]). False (default) preserves
        # the exact current (double-log) behavior for every existing config
        # / all historical comparisons; True skips the redundant second
        # log1p, feeding the gene encoder the correctly-single-log-normalized
        # values. An A/B knob, not a blind flip — the double-log is a
        # consistent confound across all models (so comparisons stay fair),
        # and whether removing it actually helps is an empirical question.
        self.input_already_log1p = input_already_log1p
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
        elif gene_encoder_type == "tokenizer":
            # 2026-07-19 research: per-gene identity-aware tokenization,
            # see TokenizedGeneEncoder's own docstring. tokenizer_gene_names
            # (HVG-reduced subset) / tokenizer_full_gene_names (full
            # training panel, for column alignment) are auto-injected by
            # train.py's inject_storm_lite_tokenizer_gene_names, same
            # "fixed vocabulary derived from real data, not hardcoded"
            # pattern as inject_decoder_gene_names.
            assert tokenizer_gene_names is not None and tokenizer_full_gene_names is not None, (
                "gene_encoder_type='tokenizer' requires tokenizer_gene_names "
                "and tokenizer_full_gene_names"
            )
            self.gene_encoder = TokenizedGeneEncoder(
                tokenizer_gene_names, tokenizer_full_gene_names, hidden_dim,
                n_pool_layers=tokenizer_n_pool_layers, n_pool_heads=tokenizer_n_pool_heads,
            )
        elif gene_encoder_type == "tokenizer_novae":
            # combines the gene-tokenizer with Novae's pretrained,
            # spatially-aware whole-profile embedding -- orthogonal axes
            # (see TokenizedGeneEncoder's docstring: Novae's spatial
            # awareness comes from ITS OWN pretrained neighbor graph,
            # independent of per-gene identity), so worth testing together,
            # not just each alone. Same concat+project combine pattern as
            # "both" (mlp+novae), for the same "a Linear on the SUM can't
            # independently reweight what went into it" reasoning (see
            # CombinedGeneEncoder's own docstring).
            assert novae_dim is not None, "gene_encoder_type='tokenizer_novae' requires novae_dim"
            assert tokenizer_gene_names is not None and tokenizer_full_gene_names is not None, (
                "gene_encoder_type='tokenizer_novae' requires tokenizer_gene_names "
                "and tokenizer_full_gene_names"
            )
            self.gene_encoder = TokenizedGeneEncoder(
                tokenizer_gene_names, tokenizer_full_gene_names, hidden_dim,
                n_pool_layers=tokenizer_n_pool_layers, n_pool_heads=tokenizer_n_pool_heads,
            )
            self.novae_encoder = NovaeGeneEncoder(novae_dim, hidden_dim)
            self.gene_combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)
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
            # qk_norm (2026-07-19, opt-in): the structural fix for the
            # "bigger" StormLite collapse — see _QKNormAttention's own
            # docstring. Only the MoME path threads it (that's the flagship
            # and where the collapse happened); False (default) keeps every
            # existing config's attention byte-for-byte unchanged.
            self.mome_blocks = nn.ModuleList([
                _MoMETransformerBlock(hidden_dim, n_heads, qk_norm=qk_norm)
                for _ in range(n_transformer_layers)
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
        # codebase (stpath_encoder.py's residual path). NOTE (2026-07-19
        # audit): raw_expr is adata.X, which basic_qc_and_normalize ALREADY
        # log1p'd — so this is a double-log1p unless input_already_log1p is
        # set (see __init__). _maybe_log1p centralizes that opt-out.
        if self.gene_encoder_type == "mlp":
            return self.gene_encoder(self._maybe_log1p(raw_expr))
        elif self.gene_encoder_type == "novae":
            assert novae_features is not None, (
                "gene_encoder_type='novae' requires context_novae_features"
            )
            return self.gene_encoder(novae_features)
        elif self.gene_encoder_type == "tokenizer":
            return self.gene_encoder(self._maybe_log1p(raw_expr))
        elif self.gene_encoder_type == "tokenizer_novae":
            assert novae_features is not None, (
                "gene_encoder_type='tokenizer_novae' requires context_novae_features"
            )
            tok_out = self.gene_encoder(self._maybe_log1p(raw_expr))
            novae_out = self.novae_encoder(novae_features)
            return self.gene_combine_proj(torch.cat([tok_out, novae_out], dim=-1))
        else:  # "both"
            assert novae_features is not None, (
                "gene_encoder_type='both' requires context_novae_features"
            )
            # CombinedGeneEncoder(combine_mode="concat") output is
            # 2*hidden_dim wide — gene_combine_proj brings it back to
            # hidden_dim (see __init__'s own comment)
            combined = self.gene_encoder(self._maybe_log1p(raw_expr), novae_features)
            return self.gene_combine_proj(combined)

    def _maybe_log1p(self, raw_expr: torch.Tensor) -> torch.Tensor:
        """log1p the raw expression UNLESS input_already_log1p is set (see
        __init__ for the double-log1p audit finding this exists to A/B)."""
        return raw_expr if self.input_already_log1p else torch.log1p(raw_expr)

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
            if self.knn_k is not None:
                knn_mask = _knn_additive_mask(coords, self.knn_k)
                bias = knn_mask if bias is None else bias + knn_mask
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
            if self.knn_k is not None:
                coords_doubled_knn = coords_doubled if self.pos_bias is not None else torch.cat([coords, coords], dim=0)
                knn_mask = _knn_additive_mask(coords_doubled_knn, self.knn_k)
                bias = knn_mask if bias is None else bias + knn_mask
            for block in self.mome_blocks:
                tokens = block(tokens, is_image_token, attn_mask=bias)
            tokens = tokens.squeeze(0)  # [2*N_total, hidden_dim]
            img_out, gene_out = tokens[:n_total], tokens[n_total:]
            combined = self.mome_output_proj(torch.cat([img_out, gene_out], dim=-1))  # [N_total, hidden_dim]
            return combined[n_context:]  # query positions only
