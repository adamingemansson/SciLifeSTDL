"""
STPath as a pretrained H&E+expression conditioning encoder (task #18).

Wraps the real STPath model (Huang et al. 2025, bioRxiv, "STPath: A
Generative Foundation Model for Integrating Spatial Transcriptomics and
Whole Slide Images" — github.com/Graph-and-Geometric-Learning/STPath,
weights at huggingface.co/tlhuang/STPath). Verified 2026-07-15 by cloning
the actual repo and reading its source, not trusting the README alone
(the lesson from task #16/Mimyr) — unlike Mimyr, STPath's real code IS a
genuinely reusable, generic inference API (`STPathInference` in
`stpath/app/pipeline/inference.py`), not hardcoded to the authors' own
cluster paths/datasets. Its documented "in-context learning" mode (feed
real expression for some spots as context, get predictions for the rest)
maps almost exactly onto our own context/query masking setup.

This class replaces our ENTIRE SpatialContextEncoder for this arm, not a
branch fused into it — STPath's own spatial transformer already does
joint image+expression+organ+tech reasoning via k-NN attention over ALL
spots at once, so there's nothing for our own encoder to add on top. This
is the "does STPath's whole architecture help" arm of the task #19
three-way comparison, distinct from task #20 (Gigapath alone, fused into
our own encoder instead).

We call `model.prediction_head(..., return_all=True)` directly rather
than going through `STPathInference.inference()` (which only returns
final gene-expression predictions) — `return_all=True` additionally
returns the pre-head hidden state `x` [N, d_model=512], the actual
embedding we want as conditioning `c`. Token construction (coordinate
rescaling, masked/context gene tokens, organ/tech tokens) mirrors
`STPathInference`'s own internal logic as closely as possible, so this
stays faithful to how the real class calls the model — just exposing the
embedding instead of the final prediction.

SETUP (neither is a default dependency/step of this repo):
  1. Clone github.com/Graph-and-Geometric-Learning/STPath and run
     `pip install -e .` from that directory. Its own setup.py has no
     install_requires — additionally install `einops==0.8.0` (its README
     also lists `torch_geometric==2.6.1`, but inspecting
     stpath/tokenization/ge_tokenizer.py shows the only use of
     torch_geometric — `coalesce`, in `.encode()` — is behind a soft
     try/except ImportError at import time and is NOT on the code path
     this class actually uses (symbol2id / convert_gene_exp_to_one_hot_tensor),
     so it's likely not required for this specific usage — verify this
     yourself if you hit an ImportError).
  2. Download the pretrained weight from huggingface.co/tlhuang/STPath.
     No gated-access requirement was found when checked 2026-07-15
     (unlike Gigapath) — verify this yourself, HF repo settings change.
  3. The bundled gene vocabulary file at
     <cloned STPath repo>/utils_data/symbol2ensembl.json — pass its path
     as gene_voc_path below.
  4. Gigapath itself (task #20's GigapathPatchEncoder setup — STPath's
     image tokens are Gigapath tile-encoder features, feature_dim=1536).

Structural smoke-testing: tests/test_stpath_encoder.py, skips cleanly
without the above. CONFIRMED running end-to-end on real hardware
(2026-07-15, user's Mac, MPS backend, exp_hest1k_wae_gan_stpath.yaml,
50/50 masking draws in ~7s) after fixing two real bugs found along the
way: HEST-1k's real patch .h5 format (src/data/loaders.py) and
PYTORCH_ENABLE_MPS_FALLBACK timing (must be set before ANY MPS op runs
in the process — see src/training/train.py's top-of-file comment, not
the os.environ.setdefault below, which alone was NOT sufficient).
"""
from __future__ import annotations

import os

# STPath's own SpatialTransformer (stpath/model/nn_utils/fa.py create_frame,
# its "frame averaging" geometry step) calls torch.linalg.eigh, which isn't
# implemented on MPS (Apple Silicon) as of this writing - confirmed
# 2026-07-15 via a real crash (NotImplementedError: aten::_linalg_eigh...).
# This is inside the external stpath package's own code, not ours, so we
# can't fix it the way we fixed our own bicubic-interpolate MPS gap
# (moving that one op to CPU manually) without patching code outside this
# repo. PYTORCH_ENABLE_MPS_FALLBACK is PyTorch's own documented workaround
# for exactly this situation - it falls back to CPU only for the specific
# unimplemented op, not the whole model. Set here too (setdefault, so an
# explicit user setting always wins) as a belt-and-suspenders default for
# anyone importing this module directly — but this alone is NOT
# sufficient. Confirmed empirically 2026-07-15: PyTorch checks this env
# var once, early (not lazily per-op as an earlier version of this
# comment assumed) — setting it here, after src/training/train.py's
# _load_images had already dispatched MPS ops via the Gigapath precompute
# step earlier in the same process, produced the identical
# torch.linalg.eigh crash. The real fix is setting it before `import
# torch` at the top of the process's actual entry point
# (src/training/train.py / src/evaluation/run_comparison.py) — see the
# comment there.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

from src.models.conditioning import (
    _load_gigapath_tile_encoder, _gigapath_preprocess_and_encode,
    MLPGeneEncoder, NovaeGeneEncoder, CombinedGeneEncoder,
)


class _ResidualEncodeInputs(nn.Module):
    """Wraps STPath's real `EncodeInputs` (stpath/model/model.py — verified
    2026-07-16 via its real source: `forward(img_tokens, ge_tokens,
    tech_tokens, organ_tokens)` returns `img_embed + ge_embed + tech_embed +
    organ_embed`, all frozen pretrained sub-embeddings) to additionally add
    a NEW, trainable gene-expression signal as a residual, BEFORE the
    spatial Transformer sees it — the actual "Route 1" test from this
    project's GEX-encoder-bottleneck investigation: does a better gene
    encoder help STPath's real (pretrained) fusion, not just our own weak
    one (which task #19-followup's Novae/MLP-in-our-own-encoder test
    already showed near-zero PCC for, regardless of gene encoder quality —
    that result couldn't distinguish "gene encoder doesn't matter" from
    "our fusion is too weak to use ANY gene encoder well").

    Residual, not replacement: STPath's own gene_embed(ge_tokens) still
    contributes as before (its pretrained weights, and the token pipeline
    that already handles context/masked-query positions correctly) — this
    just ADDS residual_proj(extra_embed) on top.

    residual_proj is a Linear(d_model, d_model) with weight AND bias
    zero-initialized — not a single scalar alpha (an earlier version of
    this class used one; a single global scalar can only uniformly scale
    the whole embedding, it can't let training weight different
    dimensions of the new signal differently). Zero-init means
    residual_proj(x) == 0 for ANY x at initialization, so training still
    begins IDENTICAL to the unmodified pretrained model (the same safe-
    initialization property the scalar had — see class docstring
    reasoning in stpath_encoder.py's STPathContextEncoder for why an
    outright replacement, with no safe starting point at all, risks
    unstable training on our small pilot dataset) — just with a strictly
    more expressive combiner once training actually moves it away from
    zero. Standard pattern in adapter/residual-injection literature
    (e.g. ControlNet's "zero convolution", NLP adapter layers).

    extra_embed must be set via set_extra_embed() before each forward()
    call (a side-channel, not a forward() parameter) because STPath's own
    `STFM.inference`/`prediction_head` calls this module internally with a
    fixed signature we don't control — see STPathContextEncoder.forward()
    for where it's actually set."""

    def __init__(self, original: nn.Module, d_model: int):
        super().__init__()
        self.original = original
        self.residual_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        self._extra_embed: torch.Tensor | None = None

    def set_extra_embed(self, extra_embed: torch.Tensor) -> None:
        self._extra_embed = extra_embed

    def forward(self, img_tokens, ge_tokens, tech_tokens, organ_tokens):
        base = self.original(img_tokens, ge_tokens, tech_tokens, organ_tokens)
        assert self._extra_embed is not None, (
            "_ResidualEncodeInputs.forward called without set_extra_embed() first"
        )
        return base + self.residual_proj(self._extra_embed)


class STPathContextEncoder(nn.Module):
    def __init__(self, gene_names: list[str], gene_voc_path: str, model_weight_path: str,
                 organ_type: str = "Kidney", tech_type: str = "Visium",
                 hidden_dim: int = 256, device: str = "cpu",
                 new_gene_encoder_type: str = "none", novae_dim: int | None = None):
        super().__init__()
        assert new_gene_encoder_type in ("none", "mlp", "novae", "both"), (
            f"unknown new_gene_encoder_type {new_gene_encoder_type!r}"
        )
        self.new_gene_encoder_type = new_gene_encoder_type
        from stpath.model.model import STFM
        from stpath.model.nn_utils.config import ModelConfig
        from stpath.tokenization import (
            GeneExpTokenizer, ImageTokenizer, IDTokenizer, TokenizerTools, AnnotationTokenizer,
        )
        from stpath.data.dataset import rescale_coords

        self._rescale_coords = rescale_coords
        self.organ_type = organ_type
        self.tech_type = tech_type
        self.device_str = device

        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        config = ModelConfig.get_default_config()
        config.feature_dim = 1536
        config.activation = "gelu"
        config.n_genes = self.tokenizer.ge_tokenizer.n_tokens
        config.n_tech = self.tokenizer.tech_tokenizer.n_tokens
        config.n_species = self.tokenizer.specie_tokenizer.n_tokens
        config.n_organs = self.tokenizer.organ_tokenizer.n_tokens
        config.backbone = "spatial_transformer"
        self.d_model = config.d_model

        self.model = STFM(config).to(device)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Route B (2026-07-16, GEX-encoder-bottleneck investigation): test
        # whether a better gene encoder helps STPath's REAL pretrained
        # fusion, not just our own weak one (see task #19-followup's
        # SpatialContextEncoder gene_encoder_type — that test couldn't
        # distinguish "gene encoder doesn't matter" from "our fusion is
        # too weak to use any gene encoder well"). Wraps
        # self.model.input_encoder (verified 2026-07-16 via STPath's real
        # source, stpath/model/model.py: EncodeInputs.forward returns
        # img_embed + ge_embed + tech_embed + organ_embed) to additionally
        # add alpha * new_gene_encoder(our_own_raw_expression) as a
        # residual, alpha starting at 0 for safe initialization (see
        # _ResidualEncodeInputs docstring for the full reasoning) — STPath's
        # own gene_embed(ge_tokens) path is untouched, this only adds to it.
        self.new_gene_encoder = None
        if new_gene_encoder_type == "mlp":
            self.new_gene_encoder = MLPGeneEncoder(len(gene_names), self.d_model)
        elif new_gene_encoder_type in ("novae", "both"):
            assert novae_dim is not None, (
                f"new_gene_encoder_type={new_gene_encoder_type!r} requires novae_dim "
                f"(see precompute_novae_features()'s real output shape)"
            )
            if new_gene_encoder_type == "novae":
                self.new_gene_encoder = NovaeGeneEncoder(novae_dim, self.d_model)
            else:  # "both" — 2026-07-16, see CombinedGeneEncoder's own docstring
                self.new_gene_encoder = CombinedGeneEncoder(len(gene_names), novae_dim, self.d_model)
        if self.new_gene_encoder is not None:
            self.model.input_encoder = _ResidualEncodeInputs(self.model.input_encoder, self.d_model)

        # Gigapath tile encoder (frozen) turns our raw H&E patches into the
        # 1536-dim features STPath's image tokenizer expects — shares the
        # same loader as GigapathPatchEncoder (task #20), no separate
        # trainable projection here since STPath wants the raw feature.
        # LAZY (not loaded here): only actually used for raw-patch input
        # (see _ensure_tile_encoder/_gigapath_features below) — the real
        # training pipeline always passes precomputed features
        # (src/models/conditioning.py precompute_gigapath_features) and
        # never touches this. Loading it unconditionally here meant every
        # STPathContextEncoder instance carried an extra ~4.4GB (fp32,
        # ~1.1B params) of dead weight even though it was never called in
        # practice — confirmed 2026-07-15 as a real contributor to a RAM
        # crash when several STPath-based configs were resident in memory
        # at once (src/evaluation/run_comparison.py, see its own fix for
        # the bigger cause: not releasing models between configs).
        self.tile_encoder = None

        # STPath's pre-head hidden state was trained for STPath's own
        # objective, not ours — its activation scale is whatever that
        # training left it at, not necessarily anything close to what an
        # untrained nn.Linear expects. Normalize before projecting rather
        # than relying on `proj` to learn a rescaling from scratch on top
        # of everything else — standard practice for "frozen big model ->
        # small trainable head" (the RAE pattern this class already is).
        # Added 2026-07-15 after a real smoke-test run showed RMSE ~3.7
        # for STPath-conditioned WAE-GAN/FM-OT/FM-EDM vs ~0.55-0.6 for
        # every other encoder variant — plausibly this exact scale
        # mismatch, though that run used a stale epochs=50 config (see
        # exp_hest1k_wae_gan_stpath.yaml's fix) so it wasn't conclusive on
        # its own; adding this regardless since it's safe either way.
        self.embedding_norm = nn.LayerNorm(self.d_model)
        self.proj = nn.Linear(self.d_model, hidden_dim)

        context_gene_ids, valid_gene_pos = self.tokenizer.ge_tokenizer.symbol2id(
            gene_names, return_valid_positions=True
        )
        self._valid_gene_pos = valid_gene_pos
        self.register_buffer("_context_gene_ids", torch.tensor(context_gene_ids, dtype=torch.long))

    def _ensure_tile_encoder(self, device: torch.device) -> nn.Module:
        """Loads the ~4.4GB Gigapath tile encoder on first actual use,
        not at construction time (see the comment on self.tile_encoder in
        __init__). Takes device explicitly rather than relying on
        Lightning's automatic .to(device) — this submodule may not exist
        yet at the point Lightning moved the rest of the model, since it's
        created lazily, possibly mid-training."""
        if self.tile_encoder is None:
            self.tile_encoder = _load_gigapath_tile_encoder().to(device)
            self.tile_encoder.eval()
            for p in self.tile_encoder.parameters():
                p.requires_grad_(False)
        return self.tile_encoder

    def _gigapath_features(self, patches_or_features: torch.Tensor) -> torch.Tensor:
        # accepts EITHER raw patches [B, 3, H, W] float in [0,1] (encodes
        # from scratch — slow, uncached path) OR already-precomputed
        # Gigapath features [B, 1536] (the fast path the real training
        # pipeline uses, see precompute_gigapath_features) - same
        # dispatch-on-rank pattern as GigapathPatchEncoder.forward
        if patches_or_features.dim() == 4:
            tile_encoder = self._ensure_tile_encoder(patches_or_features.device)
            return _gigapath_preprocess_and_encode(tile_encoder, patches_or_features)
        return patches_or_features

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor,
                query_images: torch.Tensor,
                context_novae_features: torch.Tensor | None = None) -> torch.Tensor:
        """context_images/query_images: raw H&E patches [N, 3, H, W] float
        in [0,1] — required (STPath has no meaningful expression-only
        mode). context_novae_features: [N_context, novae_dim] precomputed
        Novae features, only used/required when new_gene_encoder_type=
        "novae" (see __init__) — a SEPARATE channel from context_expression
        (which STPath's own gene_embed pathway always needs raw, whatever
        new_gene_encoder_type is set to). Returns c [N_query, hidden_dim].

        Real bug found 2026-07-15: this method used to be decorated with
        @torch.no_grad(), disabling gradient tracking for the ENTIRE
        forward pass — including self.proj/self.embedding_norm, the only
        trainable parameters in this class. They never received a
        gradient, training or eval, and stayed at random initialization
        regardless of epoch count. Only STPath's own frozen backbone call
        (self.model.prediction_head below) should skip autograd — moved
        to its own `with torch.no_grad():` block, with proj/embedding_norm
        left outside it so they actually train.

        Real gradient-flow subtlety found 2026-07-16 while adding
        new_gene_encoder (Route B residual): that SAME no_grad block would
        silently kill gradient to new_gene_encoder/residual_proj too, even
        though they're set up as trainable — the residual gets injected
        INSIDE self.model.input_encoder, which prediction_head calls INSIDE
        the no_grad block, so any op executed there (including the
        residual addition) produces outputs with requires_grad=False
        regardless of its inputs. Every one of STPath's own parameters
        already has requires_grad_(False) set (see __init__), so removing
        no_grad here doesn't make anything unintentionally trainable — it
        only lets gradient flow THROUGH the frozen layers (using their
        fixed weights) to reach new_gene_encoder/residual_proj on the
        other side, which is exactly what training the residual requires."""
        n_context = context_coords.shape[0]
        n_query = query_coords.shape[0]
        device = context_coords.device

        coords = torch.cat([context_coords[:, :2], query_coords[:, :2]], dim=0)
        coords = coords.clone()
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        coords = self._rescale_coords(coords)

        img_feats = torch.cat([
            self._gigapath_features(context_images), self._gigapath_features(query_images),
        ], dim=0)

        n_total = n_context + n_query
        ge_tokens = self.tokenizer.ge_tokenizer.mask_token.float().to(device).repeat(n_total, 1)
        expr = torch.log1p(context_expression)[:, self._valid_gene_pos]
        context_one_hot = self.tokenizer.ge_tokenizer.convert_gene_exp_to_one_hot_tensor(
            self.tokenizer.ge_tokenizer.n_tokens, expr, self._context_gene_ids.to(device)
        )
        ge_tokens[:n_context] = context_one_hot  # real expression for context, mask token for query

        organ = self.tokenizer.organ_tokenizer.encode(self.organ_type, align_first=True)
        organ_ids = torch.full((n_total,), organ, dtype=torch.long, device=device)
        tech = self.tokenizer.tech_tokenizer.encode(self.tech_type, align_first=True)
        tech_ids = torch.full((n_total,), tech, dtype=torch.long, device=device)

        if self.new_gene_encoder is not None:
            # query positions get zero extra signal, same treatment STPath's
            # own ge_tokens already give query positions (mask token, no
            # real expression) — only context rows carry real information.
            extra_embed = torch.zeros(n_total, self.d_model, device=device)
            # log1p for numerical stability (matches STPath's own convention
            # for its ge_tokens above) — MUST use the FULL, unfiltered
            # context_expression (all our genes), not STPath's `expr` above
            # (that one's already restricted to _valid_gene_pos, a different
            # width than what MLPGeneEncoder/CombinedGeneEncoder were
            # constructed for: len(gene_names)). Branches on the stored type
            # string, not isinstance, so "both" (CombinedGeneEncoder, needing
            # BOTH inputs at once) fits the same dispatch cleanly.
            if self.new_gene_encoder_type == "mlp":
                mlp_input = torch.log1p(context_expression)
                extra_embed[:n_context] = self.new_gene_encoder(mlp_input)
            elif self.new_gene_encoder_type == "novae":
                assert context_novae_features is not None, (
                    "new_gene_encoder_type='novae' requires context_novae_features"
                )
                extra_embed[:n_context] = self.new_gene_encoder(context_novae_features)
            else:  # "both" — CombinedGeneEncoder(raw_expr, novae_features)
                assert context_novae_features is not None, (
                    "new_gene_encoder_type='both' requires context_novae_features"
                )
                mlp_input = torch.log1p(context_expression)
                extra_embed[:n_context] = self.new_gene_encoder(mlp_input, context_novae_features)
            self.model.input_encoder.set_extra_embed(extra_embed)
            # NO torch.no_grad() here — see forward()'s own docstring for
            # why: gradient must flow through the frozen layers to reach
            # new_gene_encoder/residual_proj on the other side of them.
            _, x = self.model.prediction_head(
                img_tokens=img_feats,
                coords=coords,
                ge_tokens=ge_tokens,
                batch_idx=torch.zeros(n_total, dtype=torch.long, device=device),
                tech_tokens=tech_ids,
                organ_tokens=organ_ids,
                return_all=True,
            )
        else:
            with torch.no_grad():  # STPath itself is frozen (see __init__) - skip building its autograd graph
                _, x = self.model.prediction_head(
                    img_tokens=img_feats,
                    coords=coords,
                    ge_tokens=ge_tokens,
                    batch_idx=torch.zeros(n_total, dtype=torch.long, device=device),
                    tech_tokens=tech_ids,
                    organ_tokens=organ_ids,
                    return_all=True,
                )
        x = self.embedding_norm(x[n_context:])  # query positions only; trainable
        return self.proj(x)  # trainable
