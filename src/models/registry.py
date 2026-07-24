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
import math
import abc
from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.models.conditioning import (
    SpatialContextEncoder, PanelInvariantGeneDecoder,
    GeneAttentionDecoder, LLOKIStyleDecoder,
)
from src.models.vqvae import VectorQuantizer, morton_order
from src.models.spatial_baselines import interpolate, harmonic_interpolate

_MODEL_REGISTRY: dict[str, type["BaseGenerativeModel"]] = {}


def _linear_warmup_lr_lambda(warmup_steps: int):
    """Linear LR warmup from 0 to full LR over warmup_steps, then constant
    at full LR afterward — standard fix for deeper/wider transformer
    training instability at a flat LR (2026-07-19, see docs/results_log.md
    "bigger" StormLite collapse investigation: no trainer in this codebase
    scheduled LR at all before this, and gradient_clip_val=1.0 alone made
    that collapse WORSE, not better — clipping treats the symptom, large
    gradient norms, without addressing why the optimization trajectory is
    unstable at a flat LR from step 0, which warmup directly targets).

    warmup_steps=0 (default for every model, unless a config explicitly
    sets warmup_steps > 0) means the caller should NOT attach a scheduler
    at all (see each model's own configure_optimizers) — zero behavior
    change for every existing config, not merely a no-op multiplier
    wrapping the optimizer."""
    return lambda step: min(1.0, (step + 1) / warmup_steps)


def _build_context_encoder(
    n_genes: int, coord_dim: int, cond_hidden_dim: int,
    context_encoder_type: str = "builtin",
    image_encoder_type: str = "none", image_feat_dim: int = 64, image_patch_size: int = 256,
    gene_encoder_type: str = "raw", gene_feat_dim: int = 256, novae_dim: int | None = None,
    coord_scale: float = 1.0,
    stpath_gene_names: list[str] | None = None, stpath_gene_voc_path: str | None = None,
    stpath_model_weight_path: str | None = None, stpath_organ_type: str = "Kidney",
    stpath_tech_type: str = "Visium",
    stpath_new_gene_encoder_type: str = "none", stpath_novae_dim: int | None = None,
    stpath_pretrained: bool = True, stpath_input_already_log1p: bool = True,
    storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
    storm_lite_bias_type: str = "frame_averaging", storm_lite_relative_bias_hidden_dim: int = 32,
    storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
    storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
    storm_lite_local_k: int = 32,
    storm_lite_use_absolute_coords: bool = True,
    storm_lite_input_already_log1p: bool = True,
    storm_lite_tokenizer_gene_names: list[str] | None = None,
    storm_lite_tokenizer_full_gene_names: list[str] | None = None,
    storm_lite_tokenizer_n_pool_layers: int = 1, storm_lite_tokenizer_n_pool_heads: int = 4,
    simple_fusion_knn_k: int = 16, simple_fusion_input_already_log1p: bool = True,
    simple_cross_attn_n_heads: int = 4, simple_cross_attn_mlp_ratio: float = 2.0,
    simple_cross_attn_dropout: float = 0.1, simple_cross_attn_n_layers: int = 2,
    simple_stpath_transformer_n_layers: int = 2, simple_stpath_transformer_n_heads: int = 4,
    simple_stpath_transformer_dropout: float = 0.1,
    simple_stpath_transformer_attn_dropout: float = 0.1,
    simple_stpath_transformer_mlp_ratio: float = 2.0,
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
    param existed — single-sample/single-organ training is unaffected.

    coord_scale (2026-07-17, see RandomFourierFeatures' own docstring in
    conditioning.py for the real bug this fixes: sigma=1.0's default
    absolute-position encoding is essentially random noise on real
    HEST-1k pixel-scale coordinates) only applies to "builtin"/
    "storm_lite" (both construct a RandomFourierFeatures coord_encoder) —
    NOT "stpath", which has its own real, verified geometry-aware
    attention bias from its pretrained checkpoint, independent of this
    mechanism entirely. 1.0 (default) preserves the original behavior for
    any caller that doesn't explicitly opt in; auto-derived from real
    per-sample coordinate spread by inject_coord_scale in
    src/training/train.py."""
    if context_encoder_type == "builtin":
        return SpatialContextEncoder(
            n_genes=n_genes, coord_dim=coord_dim, hidden_dim=cond_hidden_dim,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size,
            gene_encoder_type=gene_encoder_type, gene_feat_dim=gene_feat_dim,
            novae_dim=novae_dim, coord_scale=coord_scale,
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
            pretrained=stpath_pretrained, input_already_log1p=stpath_input_already_log1p,
        )
    elif context_encoder_type == "storm_lite":
        from src.models.storm_lite_encoder import StormLiteContextEncoder
        assert gene_encoder_type in ("mlp", "novae", "both", "tokenizer", "tokenizer_novae"), (
            f"context_encoder_type='storm_lite' requires gene_encoder_type in "
            f"('mlp', 'novae', 'both', 'tokenizer', 'tokenizer_novae'), got {gene_encoder_type!r}"
        )
        return StormLiteContextEncoder(
            n_genes=n_genes, novae_dim=novae_dim, coord_dim=coord_dim,
            hidden_dim=cond_hidden_dim, gene_encoder_type=gene_encoder_type,
            tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            coord_scale=coord_scale,
            n_transformer_layers=storm_lite_n_layers, n_heads=storm_lite_n_heads,
            bias_type=storm_lite_bias_type,
            relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            fusion_mode=storm_lite_fusion_mode, qk_norm=storm_lite_qk_norm,
            knn_k=storm_lite_knn_k, gnn_k=storm_lite_gnn_k,
            local_k=storm_lite_local_k,
            use_absolute_coords=storm_lite_use_absolute_coords,
            input_already_log1p=storm_lite_input_already_log1p,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
    elif context_encoder_type == "simple_fusion":
        from src.models.simple_fusion_encoder import SimpleFusionContextEncoder
        return SimpleFusionContextEncoder(
            n_genes=n_genes, hidden_dim=cond_hidden_dim, knn_k=simple_fusion_knn_k,
            input_already_log1p=simple_fusion_input_already_log1p,
        )
    elif context_encoder_type == "simple_cross_attn":
        from src.models.simple_fusion_encoder import SimpleCrossAttentionContextEncoder
        return SimpleCrossAttentionContextEncoder(
            n_genes=n_genes, hidden_dim=cond_hidden_dim, n_heads=simple_cross_attn_n_heads,
            mlp_ratio=simple_cross_attn_mlp_ratio, dropout=simple_cross_attn_dropout,
            n_layers=simple_cross_attn_n_layers,
            knn_k=simple_fusion_knn_k, input_already_log1p=simple_fusion_input_already_log1p,
        )
    elif context_encoder_type == "simple_stpath_transformer":
        from src.models.simple_fusion_encoder import SimpleFusionSpatialTransformerContextEncoder
        return SimpleFusionSpatialTransformerContextEncoder(
            n_genes=n_genes, hidden_dim=cond_hidden_dim,
            n_layers=simple_stpath_transformer_n_layers,
            n_heads=simple_stpath_transformer_n_heads,
            dropout=simple_stpath_transformer_dropout,
            attn_dropout=simple_stpath_transformer_attn_dropout,
            mlp_ratio=simple_stpath_transformer_mlp_ratio,
            input_already_log1p=simple_fusion_input_already_log1p,
        )
    else:
        raise ValueError(f"unknown context_encoder_type {context_encoder_type!r}")


def _build_decoder(
    in_dim: int, n_genes: int, dense_hidden_dim: int,
    decoder_type: str = "dense", decoder_gene_names: list[str] | None = None,
    decoder_gene_embed_dim: int = 64, tech_vocab: list[str] | None = None,
    decoder_hidden_dim: int | None = None, decoder_mlp_depth: int = 1,
    decoder_combine_mode: str = "concat",
    decoder_attn_n_heads: int = 4, decoder_attn_n_layers: int = 1,
    decoder_lloki_tech_embed_dim: int = 10,
    decoder_lloki_hidden_dims: list[int] | None = None,
) -> nn.Module:
    """Shared by WAE-GAN/FM-OT/VQ-VAE+AR (2026-07-17, diagram-5 gap
    analysis follow-up — see PanelInvariantGeneDecoder's own docstring in
    conditioning.py for the full reasoning). "dense" (default) is the
    original fixed-width nn.Linear(in_dim, n_genes) — unchanged behavior
    for every existing config. "panel_invariant" swaps in
    PanelInvariantGeneDecoder, which looks up genes by name instead of a
    fixed output column, and requires decoder_gene_names (same
    "vocabulary fixed at construction time" requirement as
    stpath_gene_names — see inject_decoder_gene_names in
    src/training/train.py for how it's auto-populated from real data).
    tech_vocab reused as-is from the context-encoder's own
    organ_vocab/tech_vocab params (2026-07-16) — same vocabulary, allowed
    to be queried with a different (target-platform) tech string at decode
    time than the context encoder was conditioned on.

    decoder_hidden_dim/decoder_mlp_depth (2026-07-17, capacity follow-up —
    see docs/results_log.md and PanelInvariantGeneDecoder's own docstring):
    PREVIOUSLY the panel-invariant decoder's internal width silently
    reused dense_hidden_dim (whatever ae_hidden_dim/cond_hidden_dim the
    rest of that model happened to use) — an accidental coupling, not a
    deliberate choice. decoder_hidden_dim now gives it a genuinely
    independent width (defaults to dense_hidden_dim if unset, so every
    existing config's behavior is unchanged); decoder_mlp_depth adds
    capacity along depth instead of width (default 1 == the original
    2-layer structure exactly). decoder_combine_mode ("concat" default,
    "add" — scGPT's real gene-token combination rule, see
    PanelInvariantGeneDecoder's own docstring) roughly halves this
    decoder's training memory footprint (no more [N, n_panel, 2*hidden_dim]
    tensor) and is literature-grounded rather than an ad hoc default.
    These four are only meaningful for decoder_type="panel_invariant" —
    silently unused for other types.

    decoder_attn_n_heads/decoder_attn_n_layers configure "gene_attention"
    (GeneAttentionDecoder — Geneformer-inspired, see its own docstring for
    the real precedent and why it has a hard MAX_SAFE_PANEL_SIZE guard
    instead of silently attempting O(n_panel^2) attention over the full
    training vocabulary).

    decoder_lloki_tech_embed_dim/decoder_lloki_hidden_dims configure
    "lloki" (LLOKIStyleDecoder — faithfully ports LLOKI-CAE's real,
    verified conditional-autoencoder mechanism, see its own docstring for
    the important caveat: fixed n_genes width, NOT panel-invariant, since
    LLOKI's real panel-invariance comes from a separate component
    [LLOKI-FP] not ported here). Requires tech_vocab (reused as-is, same
    vocabulary as the context encoder's own organ_vocab/tech_vocab, or set
    independently if this decoder is used without organ/tech conditioning
    on the context encoder itself)."""
    if decoder_type == "dense":
        return nn.Sequential(
            nn.Linear(in_dim, dense_hidden_dim), nn.ReLU(),
            nn.Linear(dense_hidden_dim, n_genes),
        )
    elif decoder_type in ("panel_invariant", "gene_conditioned_vocabulary"):
        assert decoder_gene_names, (
            "decoder_type='panel_invariant' requires decoder_gene_names"
        )
        return PanelInvariantGeneDecoder(
            gene_names=decoder_gene_names, gene_embed_dim=decoder_gene_embed_dim,
            in_dim=in_dim, tech_vocab=tech_vocab,
            hidden_dim=decoder_hidden_dim or dense_hidden_dim,
            mlp_depth=decoder_mlp_depth, combine_mode=decoder_combine_mode,
        )
    elif decoder_type == "gene_attention":
        assert decoder_gene_names, (
            "decoder_type='gene_attention' requires decoder_gene_names"
        )
        assert len(decoder_gene_names) <= GeneAttentionDecoder.MAX_SAFE_PANEL_SIZE, (
            f"decoder_gene_names has {len(decoder_gene_names)} genes, exceeding "
            f"GeneAttentionDecoder.MAX_SAFE_PANEL_SIZE={GeneAttentionDecoder.MAX_SAFE_PANEL_SIZE} "
            f"— restrict decoder_gene_names to a realistic target panel size "
            f"(see GeneAttentionDecoder's own docstring); this is NOT meant "
            f"for fixed-vocabulary gene-conditioned decoding, use decoder_type='gene_conditioned_vocabulary' "
            f"for that."
        )
        return GeneAttentionDecoder(
            gene_names=decoder_gene_names, gene_embed_dim=decoder_gene_embed_dim,
            in_dim=in_dim, hidden_dim=decoder_hidden_dim or dense_hidden_dim,
            n_heads=decoder_attn_n_heads, n_layers=decoder_attn_n_layers,
            tech_vocab=tech_vocab,
        )
    elif decoder_type == "lloki":
        assert tech_vocab, (
            "decoder_type='lloki' requires tech_vocab (see LLOKIStyleDecoder's "
            "own docstring — its real, verified mechanism is technology-"
            "conditioned, not panel-invariant)"
        )
        return LLOKIStyleDecoder(
            in_dim=in_dim, n_genes=n_genes, tech_vocab=tech_vocab,
            tech_embed_dim=decoder_lloki_tech_embed_dim,
            hidden_dims=tuple(decoder_lloki_hidden_dims or (512, 256, 128)),
        )
    else:
        raise ValueError(f"unknown decoder_type {decoder_type!r}")


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
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            context_novae_features=context.get("novae_features"),
            organ=context.get("organ"), tech=context.get("tech"),
        )

    def _slice_target_for_decoder(self, target_expression: torch.Tensor) -> torch.Tensor:
        """decoder_type="gene_attention" (GeneAttentionDecoder) outputs a
        DELIBERATELY restricted gene panel (its self-attention is
        O(n_panel^2), see its MAX_SAFE_PANEL_SIZE guard) — narrower than
        target_expression's full training-panel width. This slices
        target_expression down to the SAME columns, via an index buffer
        registered at construction time (see each model's __init__,
        "gene_attention" branch), so training_step's loss computation
        compares like-for-like widths. No-op (returns target_expression
        unchanged) for every other decoder_type, including "panel_invariant"
        (whose default gene_names=None already covers the full training
        panel in the same column order, so no slicing is needed there).

        KNOWN GAP (2026-07-17, not yet fixed): this only covers the
        TRAINING loss path. Evaluation (run_comparison.py's shared FID/MMD
        machinery, PCA-fit on the full training-panel width) does NOT yet
        slice consistently — decoder_type="gene_attention" is not yet
        safe to run through the normal evaluation pipeline. Not fixed here
        due to time constraints; flagged clearly rather than silently
        producing wrong FID/MMD numbers. decoder_type="lloki" has no such
        gap (fixed n_genes width, same as "dense")."""
        idx = getattr(self, "_decoder_target_col_idx", None)
        return target_expression if idx is None else target_expression[:, idx]

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



@register_model("spatial_baseline")
class SpatialInterpolationBaseline(BaseGenerativeModel):
    """Unified deterministic baseline family.

    ``mode`` can be ``global_mean``, ``nearest``, ``local_mean``, ``idw`` or
    ``harmonic``. Keeping every floor behind one implementation prevents
    evaluation/config drift between baselines.
    """

    def __init__(self, mode: str = "harmonic", k: int = 8, power: float = 1.0,
                 ridge: float = 1e-4):
        super().__init__()
        self.mode, self.k, self.power, self.ridge = mode, int(k), float(power), float(ridge)

    def sample(self, context, query):
        expression = interpolate(
            self.mode, context["coords"], context["expression"], query["coords"],
            k=self.k, power=self.power, ridge=self.ridge,
        )
        return {"coords": query["coords"], "expression": expression}

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return None


@register_model("stpath_official")
class OfficialSTPathBaseline(BaseGenerativeModel):
    """Released frozen STPath prediction head with no project-specific head.

    The shared STPath wrapper is reused only to construct the authors'
    tokenizer/model inputs.  Its Novae/MLP residual, projection, FM module and
    project decoder are all bypassed.  Missing query H&E is represented by a
    zero 1536-d image feature, making ``target_zero`` an explicitly labelled
    out-of-distribution stress test of released STPath rather than pretending
    that the original model was trained for absent tissue images.
    """

    def __init__(
        self,
        n_genes: int,
        stpath_gene_names: list[str],
        stpath_gene_voc_path: str,
        stpath_model_weight_path: str,
        stpath_organ_type: str = "Kidney",
        stpath_tech_type: str = "Visium",
        stpath_input_already_log1p: bool = True,
        context_encoder_type: str = "stpath",
    ):
        super().__init__()
        if context_encoder_type != "stpath":
            raise ValueError("stpath_official requires context_encoder_type='stpath'")
        from src.models.stpath_encoder import STPathContextEncoder

        self.predictor = STPathContextEncoder(
            gene_names=stpath_gene_names,
            gene_voc_path=stpath_gene_voc_path,
            model_weight_path=stpath_model_weight_path,
            organ_type=stpath_organ_type,
            tech_type=stpath_tech_type,
            hidden_dim=512,
            new_gene_encoder_type="none",
            pretrained=True,
            input_already_log1p=stpath_input_already_log1p,
        )
        with torch.no_grad():
            self.predictor.missing_image_token.zero_()
        for parameter in self.predictor.parameters():
            parameter.requires_grad_(False)
        valid_positions = torch.as_tensor(
            self.predictor._valid_gene_pos, dtype=torch.long
        )
        if valid_positions.numel() == 0:
            raise ValueError("none of the evaluation genes are supported by STPath")
        self.register_buffer("_decoder_target_col_idx", valid_positions)
        self.n_genes = int(n_genes)

    def sample(self, context, query):
        expression = self.predictor(
            context["coords"], context["expression"], query["coords"],
            context_images=context.get("images"),
            query_images=query.get("images"),
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            organ=context.get("organ"), tech=context.get("tech"),
            return_official_predictions=True,
        )
        return {"coords": query["coords"], "expression": expression}

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return None


@register_model("stpath_scratch")
class STPathFromScratch(BaseGenerativeModel):
    """STPath's own real architecture (STFM, same class as stpath_official
    above), randomly initialized and trained from scratch on our data --
    no added components (no gene-table injection, no residual adapter,
    same new_gene_encoder_type='none' as the frozen baseline). The
    comparison arm for "does STPath's architecture help without its
    massive external pretraining, on our data alone" (lung round,
    2026-07-24).

    Deliberately does NOT zero missing_image_token the way
    stpath_official does -- that zero-fill exists there only to
    reproduce, as a labelled stress test, the exact zero-fill the
    RELEASED (frozen) weights were never trained to handle (see that
    class's own docstring, and the real notebook collapse it explains:
    STPath's own fusion sums a biased nn.Linear projection of the image
    feature into every token, so a zeroed feature still injects a real,
    wrong signal). Here the token is trainable from a random init like
    everything else, so training can learn to actually use it correctly.

    Gradient flow through STFM's own layers when pretrained=False is
    STPathContextEncoder's own responsibility (see its forward()
    docstring, needs_grad = ... or not self.pretrained) -- verified
    directly, not assumed."""

    def __init__(
        self,
        n_genes: int,
        stpath_gene_names: list[str],
        stpath_gene_voc_path: str,
        stpath_organ_type: str = "Kidney",
        stpath_tech_type: str = "Visium",
        stpath_input_already_log1p: bool = True,
        stpath_hidden_dim: int = 512,
        context_encoder_type: str = "stpath",
        lr: float = 1e-3,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
    ):
        super().__init__()
        if context_encoder_type != "stpath":
            raise ValueError("stpath_scratch requires context_encoder_type='stpath'")
        from src.models.stpath_encoder import STPathContextEncoder

        self.predictor = STPathContextEncoder(
            gene_names=stpath_gene_names,
            gene_voc_path=stpath_gene_voc_path,
            model_weight_path=None,
            organ_type=stpath_organ_type,
            tech_type=stpath_tech_type,
            hidden_dim=stpath_hidden_dim,
            new_gene_encoder_type="none",
            pretrained=False,
            input_already_log1p=stpath_input_already_log1p,
        )
        self.lr = float(lr)
        valid_positions = torch.as_tensor(
            self.predictor._valid_gene_pos, dtype=torch.long
        )
        if valid_positions.numel() == 0:
            raise ValueError("none of the evaluation genes are supported by STPath")
        self.register_buffer("_decoder_target_col_idx", valid_positions)
        self.n_genes = int(n_genes)

        scale = torch.ones(valid_positions.numel()) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (valid_positions.numel(),):
            raise ValueError(
                f"target_gene_scale must have shape ({valid_positions.numel()},) "
                f"(one entry per STPath-supported gene), got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

    def sample(self, context, query):
        expression = self.predictor(
            context["coords"], context["expression"], query["coords"],
            context_images=context.get("images"),
            query_images=query.get("images"),
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            organ=context.get("organ"), tech=context.get("tech"),
            return_official_predictions=True,
        )
        return {"coords": query["coords"], "expression": expression}

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = self._slice_target_for_decoder(batch["target_expression"])
        standardized_error = (out["expression"] - target) / self.target_gene_scale
        loss = standardized_error.square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        self.log_dict({"train/loss": loss, "train/absolute_mse": absolute_mse})
        return loss

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        return torch.optim.AdamW(parameters, lr=self.lr)


@register_model("stpath_backbone_simple_gene")
class STPathBackboneSimpleGene(BaseGenerativeModel):
    """As close to STPathFromScratch (stpath_scratch, above) as possible,
    with exactly the two changes actually requested (2026-07-24) and
    NOTHING else: a different gene encoder (MLPGeneEncoder, replacing
    STPath's own fixed-vocabulary tokenizer -- not residually added on
    top of it the way new_gene_encoder_type='mlp' on STPathContextEncoder
    does; a genuine replacement) and no organ/tech tokens. Everything
    else stays identical to real STPath: elementwise-sum fusion of
    image+gene tokens (via SimpleFusionSpatialTransformerContextEncoder,
    which already reuses STPath's real SpatialTransformer backbone), and
    critically, the SAME prediction mechanism -- a dense
    LayerNorm+Linear decoder head predicting expression directly,
    not a neighbor-weighted transport average.

    An earlier version of "the STPath-transformer arm without organ/tech"
    (simple_stpath_transformer, in context_transport_regressor) got this
    wrong: it kept the transformer but swapped the whole prediction
    mechanism to transport-over-neighbors, never asked for. That silently
    made it incomparable to stpath_scratch on the single axis that
    actually mattered most. This class fixes that by keeping
    stpath_scratch's own dense-decoder training_step/sample structure
    verbatim and swapping ONLY the encoder that feeds it."""

    def __init__(self, n_genes: int, hidden_dim: int = 512, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, attn_dropout: float = 0.1,
                 mlp_ratio: float = 2.0, input_already_log1p: bool = True,
                 lr: float = 1e-3, target_gene_scale: list[float] | None = None,
                 target_scale_floor: float = 0.05):
        super().__init__()
        from src.models.simple_fusion_encoder import SimpleFusionSpatialTransformerContextEncoder

        self.n_genes = int(n_genes)
        self.lr = float(lr)
        self.encoder = SimpleFusionSpatialTransformerContextEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, attn_dropout=attn_dropout, mlp_ratio=mlp_ratio,
            input_already_log1p=input_already_log1p,
        )
        # Same head shape as STPath's own real prediction_head (LayerNorm +
        # Linear, verified against stpath/model/model.py) -- just to our
        # local n_genes instead of STPath's ~39k-gene vocabulary, since
        # this arm never uses that vocabulary at all (see class docstring).
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_genes))

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}")
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

    def sample(self, context, query):
        condition = self.encoder(
            context["coords"], context["expression"], query["coords"],
            context_images=context.get("images"), query_images=query.get("images"),
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            organ=context.get("organ"), tech=context.get("tech"),
        )
        expression = self.decoder(condition)
        return {"coords": query["coords"], "expression": expression}

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_error = (out["expression"] - target) / self.target_gene_scale
        loss = standardized_error.square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        self.log_dict({"train/loss": loss, "train/absolute_mse": absolute_mse})
        return loss

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        return torch.optim.AdamW(parameters, lr=self.lr)


@register_model("simple_cross_attn_dense_decoder")
class SimpleCrossAttnDenseDecoder(BaseGenerativeModel):
    """Same isolation stpath_backbone_simple_gene applies to the STPath
    transformer, applied to the cross-attention encoder (2026-07-24):
    SimpleCrossAttentionContextEncoder (GigaPath+gene tokens, learned
    cross-attention over k nearest neighbors -- see
    simple_fusion_encoder.py) feeding a dense LayerNorm+Linear decoder
    head that predicts expression directly, instead of
    context_transport_regressor's neighbor-weighted transport average
    (what 303_lung_simple_cross_attn.yaml actually uses).

    Structurally identical to STPathBackboneSimpleGene above except which
    encoder it wraps -- lets "does a dense decoder beat transport" get
    checked on the cross-attention architecture too, not just the STPath
    backbone one."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, knn_k: int = 16,
                 n_layers: int = 2, input_already_log1p: bool = True,
                 lr: float = 1e-3, target_gene_scale: list[float] | None = None,
                 target_scale_floor: float = 0.05):
        super().__init__()
        from src.models.simple_fusion_encoder import SimpleCrossAttentionContextEncoder

        self.n_genes = int(n_genes)
        self.lr = float(lr)
        self.encoder = SimpleCrossAttentionContextEncoder(
            n_genes=n_genes, hidden_dim=hidden_dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
            dropout=dropout, knn_k=knn_k, n_layers=n_layers,
            input_already_log1p=input_already_log1p,
        )
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_genes))

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}")
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

    def sample(self, context, query):
        condition = self.encoder(
            context["coords"], context["expression"], query["coords"],
            context_images=context.get("images"), query_images=query.get("images"),
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            organ=context.get("organ"), tech=context.get("tech"),
        )
        expression = self.decoder(condition)
        return {"coords": query["coords"], "expression": expression}

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_error = (out["expression"] - target) / self.target_gene_scale
        loss = standardized_error.square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        self.log_dict({"train/loss": loss, "train/absolute_mse": absolute_mse})
        return loss

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        return torch.optim.AdamW(parameters, lr=self.lr)


@register_model("set_summary_baseline")
class SetSummaryBaseline(BaseGenerativeModel):
    """Strict learned mean/sum embedding baseline.

    Context rows are encoded independently and aggregated with exactly the
    configured operation. Query coordinates are then combined with that one
    global summary. There is no attention, graph propagation or image branch,
    making this a transparent capacity-matched check of whether sophisticated
    context models beat a learned set statistic.
    """

    def __init__(self, n_genes: int, coord_dim: int = 3, hidden_dim: int = 256,
                 aggregation: str = "mean", lr: float = 1e-3):
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'")
        self.aggregation = aggregation
        self.lr = float(lr)
        self.context_row = nn.Sequential(
            nn.Linear(n_genes + coord_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.query_net = nn.Sequential(
            nn.Linear(hidden_dim + coord_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def sample(self, context, query):
        row = self.context_row(torch.cat([context["expression"], context["coords"]], dim=-1))
        summary = row.mean(dim=0, keepdim=True) if self.aggregation == "mean" else row.sum(dim=0, keepdim=True)
        summary = summary.expand(query["coords"].shape[0], -1)
        pred = self.query_net(torch.cat([summary, query["coords"]], dim=-1))
        return {"coords": query["coords"], "expression": pred}

    def training_step(self, batch, batch_idx):
        pred = self.sample(batch["context"], batch["query"])["expression"]
        loss = nn.functional.mse_loss(pred, batch["target_expression"])
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

@register_model("harmonic_residual")
class HarmonicResidualModel(BaseGenerativeModel):
    """Learn a variance-standardized correction around a harmonic anchor.

    The first implementation zero-initialized the output layer and optimized
    absolute full-panel MSE.  With roughly 16k genes this made the harmonic
    anchor an extremely sticky solution: the zero output layer blocked every
    upstream gradient on the first update, while low-variance genes dominated
    the averaged loss.  Architecturally different encoders consequently
    produced nearly identical anchor-only predictions.

    ``residual_gene_scale`` is derived from training samples only by
    ``src.training.train``.  The network predicts residuals in standardized
    units and converts them back to expression units for sampling.  This gives
    every informative gene a usable optimization signal without changing the
    expression-space prediction or evaluation contract.
    """

    def __init__(
        self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
        hidden_dim: int = 512, harmonic_k: int = 8, harmonic_ridge: float = 1e-4,
        lr: float = 1e-3, context_encoder_type: str = "builtin",
        image_encoder_type: str = "none", image_feat_dim: int = 64,
        image_patch_size: int = 256, gene_encoder_type: str = "mlp",
        gene_feat_dim: int = 256, novae_dim: int | None = None,
        coord_scale: float = 1.0, storm_lite_n_layers: int = 2,
        storm_lite_n_heads: int = 4, storm_lite_bias_type: str = "frame_averaging",
        storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
        storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
        storm_lite_local_k: int = 32,
        storm_lite_use_absolute_coords: bool = True,
        storm_lite_input_already_log1p: bool = True,
        storm_lite_tokenizer_gene_names: list[str] | None = None,
        storm_lite_tokenizer_full_gene_names: list[str] | None = None,
        storm_lite_tokenizer_n_pool_layers: int = 1,
        storm_lite_tokenizer_n_pool_heads: int = 4,
        residual_gene_scale: list[float] | None = None,
        residual_scale_floor: float = 0.05,
        residual_init_std: float = 1e-3,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_encoder = _build_context_encoder(
            n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
            context_encoder_type=context_encoder_type,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size, gene_encoder_type=gene_encoder_type,
            gene_feat_dim=gene_feat_dim, novae_dim=novae_dim, coord_scale=coord_scale,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_bias_type=storm_lite_bias_type,
            storm_lite_fusion_mode=storm_lite_fusion_mode,
            storm_lite_qk_norm=storm_lite_qk_norm,
            storm_lite_knn_k=storm_lite_knn_k, storm_lite_gnn_k=storm_lite_gnn_k,
            storm_lite_local_k=storm_lite_local_k,
            storm_lite_use_absolute_coords=storm_lite_use_absolute_coords,
            storm_lite_input_already_log1p=storm_lite_input_already_log1p,
            storm_lite_tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            storm_lite_tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            storm_lite_tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            storm_lite_tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        self.anchor_proj = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.GELU())
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + cond_hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        if residual_init_std <= 0:
            raise ValueError("residual_init_std must be positive")
        if residual_scale_floor <= 0:
            raise ValueError("residual_scale_floor must be positive")
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=float(residual_init_std))
        nn.init.zeros_(self.residual[-1].bias)
        if residual_gene_scale is None:
            scale = torch.ones(n_genes, dtype=torch.float32)
        else:
            scale = torch.as_tensor(residual_gene_scale, dtype=torch.float32)
            if scale.shape != (n_genes,):
                raise ValueError(
                    f"residual_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}"
                )
            if not torch.isfinite(scale).all():
                raise ValueError("residual_gene_scale contains non-finite values")
        self.register_buffer(
            "residual_gene_scale",
            scale.clamp_min(float(residual_scale_floor)),
        )
        self.harmonic_k, self.harmonic_ridge, self.lr = int(harmonic_k), float(harmonic_ridge), float(lr)

    def _anchor(self, context, query):
        return harmonic_interpolate(
            context["coords"], context["expression"], query["coords"],
            k=self.harmonic_k, ridge=self.harmonic_ridge,
        )

    def sample(self, context, query):
        anchor = self._anchor(context, query).detach()
        c = self._encode_context(context, query)
        standardized_correction = self.residual(
            torch.cat([self.anchor_proj(anchor), c], dim=-1)
        )
        correction = standardized_correction * self.residual_gene_scale
        return {"coords": query["coords"], "expression": anchor + correction,
                "anchor_expression": anchor, "residual_expression": correction,
                "standardized_residual_expression": standardized_correction}

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        residual_target = (target - out["anchor_expression"]) / self.residual_gene_scale
        loss = nn.functional.mse_loss(
            out["standardized_residual_expression"], residual_target
        )
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        anchor_loss = nn.functional.mse_loss(out["anchor_expression"], target)
        correction_rms = out["residual_expression"].square().mean().sqrt()
        target_residual_rms = (target - out["anchor_expression"]).square().mean().sqrt()
        self.log_dict({
            "train/loss": loss,
            "train/absolute_mse": absolute_mse,
            "train/anchor_mse": anchor_loss,
            "train/correction_rms": correction_rms,
            "train/target_residual_rms": target_residual_rms,
        })
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_after_backward(self) -> None:
        """Expose whether the repaired residual signal reaches its encoder."""
        context_sq = torch.zeros((), device=self.device)
        for parameter in self.context_encoder.parameters():
            if parameter.grad is not None:
                context_sq = context_sq + parameter.grad.detach().square().sum()
        output_sq = torch.zeros((), device=self.device)
        for parameter in self.residual[-1].parameters():
            if parameter.grad is not None:
                output_sq = output_sq + parameter.grad.detach().square().sum()
        self.log_dict({
            "train/context_grad_norm": context_sq.sqrt(),
            "train/residual_output_grad_norm": output_sq.sqrt(),
        })


@register_model("hierarchical_missing_tissue_regressor")
class HierarchicalMissingTissueRegressor(BaseGenerativeModel):
    """Deterministic first-stage model for a physically missing tissue hole.

    A frozen, mask-aware GigaPath LongNet supplies global WSI morphology.
    Observed spot H&E, raw GEX and context-only Novae supply local evidence.
    Missing queries cross-attend only to observed tokens; neither query H&E
    nor query GEX is ever consumed.  This is intentionally a regression head
    before reintroducing flow matching: the conditioning architecture must
    first demonstrate genuine held-out predictive signal.
    """

    def __init__(
        self,
        n_genes: int,
        novae_dim: int | None = None,
        hidden_dim: int = 256,
        decoder_hidden_dim: int = 512,
        n_heads: int = 4,
        context_layers: int = 2,
        cross_layers: int = 2,
        query_layers: int = 1,
        local_k: int = 64,
        dropout: float = 0.1,
        use_novae: bool = True,
        use_local_images: bool = True,
        use_slide_context: bool = True,
        gene_encoder_type: str = "weighted_linear",
        slide_checkpoint_path: str | None = None,
        slide_output_dim: int = 768,
        target_gene_mean: list[float] | None = None,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
        correlation_loss_weight: float = 0.25,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        from src.models.hierarchical_slide import HierarchicalMissingTissueEncoder

        if target_scale_floor <= 0:
            raise ValueError("target_scale_floor must be positive")
        if correlation_loss_weight < 0:
            raise ValueError("correlation_loss_weight must be non-negative")
        self.save_hyperparameters()
        self.context_encoder = HierarchicalMissingTissueEncoder(
            n_genes=n_genes,
            novae_dim=novae_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            context_layers=context_layers,
            cross_layers=cross_layers,
            query_layers=query_layers,
            local_k=local_k,
            dropout=dropout,
            use_novae=use_novae,
            use_local_images=use_local_images,
            use_slide_context=use_slide_context,
            gene_encoder_type=gene_encoder_type,
            slide_checkpoint_path=slide_checkpoint_path,
            slide_output_dim=slide_output_dim,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, n_genes),
        )
        mean = torch.zeros(n_genes) if target_gene_mean is None else torch.as_tensor(
            target_gene_mean, dtype=torch.float32
        )
        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if mean.shape != (n_genes,) or scale.shape != (n_genes,):
            raise ValueError(
                f"target gene statistics must both be ({n_genes},), got "
                f"mean={tuple(mean.shape)}, scale={tuple(scale.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("target gene statistics contain non-finite values")
        self.register_buffer("target_gene_mean", mean)
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))
        self.correlation_loss_weight = float(correlation_loss_weight)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

    @staticmethod
    def _mean_per_gene_correlation(prediction: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        target_ss = target_centered.square().sum(dim=0)
        eligible = target_ss > 1e-8
        if not bool(eligible.any()):
            return prediction.new_zeros(())
        numerator = (pred_centered * target_centered).sum(dim=0)
        pred_ss = pred_centered.square().sum(dim=0)
        denominator = torch.sqrt((pred_ss * target_ss).clamp_min(1e-8))
        return (numerator[eligible] / denominator[eligible]).mean()

    def sample(self, context, query):
        hidden = self.context_encoder(context, query)
        standardized = self.decoder(hidden)
        mean = self.target_gene_mean.unsqueeze(0).expand_as(standardized)
        expression = mean + standardized * self.target_gene_scale
        return {
            "coords": query["coords"],
            "expression": expression,
            "anchor_expression": mean,
            "residual_expression": expression - mean,
            "standardized_expression": standardized,
        }

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_target = (
            target - self.target_gene_mean
        ) / self.target_gene_scale
        standardized_mse = nn.functional.mse_loss(
            out["standardized_expression"], standardized_target
        )
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        mean_baseline_mse = nn.functional.mse_loss(out["anchor_expression"], target)
        pcc = self._mean_per_gene_correlation(out["expression"], target)
        loss = standardized_mse + self.correlation_loss_weight * (1.0 - pcc)
        self.log_dict({
            "train/loss": loss,
            "train/standardized_mse": standardized_mse,
            "train/absolute_mse": absolute_mse,
            "train/mean_baseline_mse": mean_baseline_mse,
            "train/per_gene_pcc": pcc,
            "train/prediction_delta_rms": out["residual_expression"].square().mean().sqrt(),
        })
        return loss

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )


@register_model("direct_context_regressor")
class DirectContextRegressor(BaseGenerativeModel):
    """Predict absolute query expression directly from learned context.

    Unlike :class:`HarmonicResidualModel`, this model never computes or feeds
    a spatial interpolation into the prediction path.  Training-only per-gene
    mean/scale statistics condition the regression problem numerically; the
    reported output is converted back to the original normalized-log
    expression units.  The mean-only tensor is exposed as ``anchor_expression``
    solely so fixed-mask validation can report a transparent non-spatial
    baseline gate.
    """

    def __init__(
        self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
        hidden_dim: int = 512, lr: float = 3e-4,
        context_encoder_type: str = "storm_lite",
        image_encoder_type: str = "none", image_feat_dim: int = 64,
        image_patch_size: int = 256, gene_encoder_type: str = "mlp",
        gene_feat_dim: int = 256, novae_dim: int | None = None,
        coord_scale: float = 1.0, storm_lite_n_layers: int = 2,
        storm_lite_n_heads: int = 4, storm_lite_bias_type: str = "frame_averaging",
        storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
        storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
        storm_lite_local_k: int = 32,
        storm_lite_use_absolute_coords: bool = True,
        storm_lite_input_already_log1p: bool = True,
        storm_lite_tokenizer_gene_names: list[str] | None = None,
        storm_lite_tokenizer_full_gene_names: list[str] | None = None,
        storm_lite_tokenizer_n_pool_layers: int = 1,
        storm_lite_tokenizer_n_pool_heads: int = 4,
        target_gene_mean: list[float] | None = None,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.context_encoder = _build_context_encoder(
            n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
            context_encoder_type=context_encoder_type,
            image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
            image_patch_size=image_patch_size, gene_encoder_type=gene_encoder_type,
            gene_feat_dim=gene_feat_dim, novae_dim=novae_dim, coord_scale=coord_scale,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_bias_type=storm_lite_bias_type,
            storm_lite_fusion_mode=storm_lite_fusion_mode,
            storm_lite_qk_norm=storm_lite_qk_norm,
            storm_lite_knn_k=storm_lite_knn_k, storm_lite_gnn_k=storm_lite_gnn_k,
            storm_lite_local_k=storm_lite_local_k,
            storm_lite_use_absolute_coords=storm_lite_use_absolute_coords,
            storm_lite_input_already_log1p=storm_lite_input_already_log1p,
            storm_lite_tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            storm_lite_tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            storm_lite_tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            storm_lite_tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(cond_hidden_dim),
            nn.Linear(cond_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        if target_scale_floor <= 0:
            raise ValueError("target_scale_floor must be positive")
        mean = torch.zeros(n_genes) if target_gene_mean is None else torch.as_tensor(
            target_gene_mean, dtype=torch.float32
        )
        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if mean.shape != (n_genes,) or scale.shape != (n_genes,):
            raise ValueError(
                f"target gene statistics must both have shape ({n_genes},), "
                f"got mean={tuple(mean.shape)}, scale={tuple(scale.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("target gene statistics contain non-finite values")
        self.register_buffer("target_gene_mean", mean)
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))
        self.lr = float(lr)

    def sample(self, context, query):
        condition = self._encode_context(context, query)
        standardized = self.decoder(condition)
        mean = self.target_gene_mean.unsqueeze(0).expand_as(standardized)
        expression = mean + standardized * self.target_gene_scale
        return {
            "coords": query["coords"],
            "expression": expression,
            "anchor_expression": mean,
            "residual_expression": expression - mean,
            "standardized_expression": standardized,
        }

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_target = (
            target - self.target_gene_mean
        ) / self.target_gene_scale
        loss = nn.functional.mse_loss(out["standardized_expression"], standardized_target)
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        mean_baseline_mse = nn.functional.mse_loss(out["anchor_expression"], target)
        prediction_delta_rms = out["residual_expression"].square().mean().sqrt()
        self.log_dict({
            "train/loss": loss,
            "train/absolute_mse": absolute_mse,
            "train/mean_baseline_mse": mean_baseline_mse,
            "train/prediction_delta_rms": prediction_delta_rms,
        })
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_after_backward(self) -> None:
        context_sq = torch.zeros((), device=self.device)
        for parameter in self.context_encoder.parameters():
            if parameter.grad is not None:
                context_sq = context_sq + parameter.grad.detach().square().sum()
        decoder_sq = torch.zeros((), device=self.device)
        for parameter in self.decoder.parameters():
            if parameter.grad is not None:
                decoder_sq = decoder_sq + parameter.grad.detach().square().sum()
        self.log_dict({
            "train/context_grad_norm": context_sq.sqrt(),
            "train/decoder_grad_norm": decoder_sq.sqrt(),
        })


@register_model("context_transport_regressor")
class ContextTransportRegressor(BaseGenerativeModel):
    """Transport observed per-gene values without a gene decoder.

    The direct-regression diagnostic compresses a complete expression profile
    into one latent vector and then asks a dense output head to reconstruct the
    whole gene vocabulary.  That can discard gene identity.  This model keeps
    the raw context expression matrix as the value path: it predicts one set of
    normalized weights over the ``k`` nearest observed spots and applies those
    same weights directly to every gene.

    ``conditioning_mode='uniform'`` is a parameter-free local-mean baseline.
    ``'geometry'`` learns a relative-coordinate kernel.  ``'storm_lite'`` lets
    a Transformer or MoME representation of surrounding H&E/GEX/Novae modulate
    that relative-coordinate kernel.  No mode computes harmonic interpolation,
    sees query expression, or decodes genes from the context embedding.
    """

    def __init__(
        self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
        score_hidden_dim: int = 128, transport_k: int = 32,
        conditioning_mode: str = "geometry", lr: float = 3e-4,
        context_encoder_type: str = "storm_lite",
        image_encoder_type: str = "none", image_feat_dim: int = 64,
        image_patch_size: int = 256, gene_encoder_type: str = "mlp",
        gene_feat_dim: int = 256, novae_dim: int | None = None,
        coord_scale: float = 1.0, storm_lite_n_layers: int = 2,
        storm_lite_n_heads: int = 4, storm_lite_bias_type: str = "frame_averaging",
        storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
        storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
        storm_lite_local_k: int = 32,
        storm_lite_use_absolute_coords: bool = True,
        storm_lite_input_already_log1p: bool = True,
        simple_fusion_knn_k: int = 16,
        simple_fusion_input_already_log1p: bool = True,
        simple_cross_attn_n_heads: int = 4,
        simple_cross_attn_mlp_ratio: float = 2.0,
        simple_cross_attn_dropout: float = 0.1,
        simple_cross_attn_n_layers: int = 2,
        simple_stpath_transformer_n_layers: int = 2,
        simple_stpath_transformer_n_heads: int = 4,
        simple_stpath_transformer_dropout: float = 0.1,
        simple_stpath_transformer_attn_dropout: float = 0.1,
        simple_stpath_transformer_mlp_ratio: float = 2.0,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        _known_conditioning_modes = {
            "uniform", "geometry", "storm_lite", "simple_fusion", "simple_cross_attn",
            "simple_stpath_transformer",
        }
        if conditioning_mode not in _known_conditioning_modes:
            raise ValueError(f"conditioning_mode must be one of {sorted(_known_conditioning_modes)}")
        if transport_k < 1:
            raise ValueError("transport_k must be positive")
        if target_scale_floor <= 0:
            raise ValueError("target_scale_floor must be positive")
        self.save_hyperparameters()
        self.conditioning_mode = conditioning_mode
        self.transport_k = int(transport_k)
        self.lr = float(lr)

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(
                f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

        self.context_encoder = None
        self.geometry_encoder = None
        self.condition_projection = None
        self.weight_scorer = None
        if conditioning_mode in {
            "storm_lite", "simple_fusion", "simple_cross_attn", "simple_stpath_transformer",
        }:
            if context_encoder_type != conditioning_mode:
                raise ValueError(
                    f"conditioning_mode={conditioning_mode!r} requires "
                    f"context_encoder_type={conditioning_mode!r}"
                )
            self.context_encoder = _build_context_encoder(
                n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
                context_encoder_type=context_encoder_type,
                image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
                image_patch_size=image_patch_size, gene_encoder_type=gene_encoder_type,
                gene_feat_dim=gene_feat_dim, novae_dim=novae_dim, coord_scale=coord_scale,
                storm_lite_n_layers=storm_lite_n_layers,
                storm_lite_n_heads=storm_lite_n_heads,
                storm_lite_bias_type=storm_lite_bias_type,
                storm_lite_fusion_mode=storm_lite_fusion_mode,
                storm_lite_qk_norm=storm_lite_qk_norm,
                storm_lite_knn_k=storm_lite_knn_k,
                storm_lite_gnn_k=storm_lite_gnn_k,
                storm_lite_local_k=storm_lite_local_k,
                storm_lite_use_absolute_coords=storm_lite_use_absolute_coords,
                storm_lite_input_already_log1p=storm_lite_input_already_log1p,
                simple_fusion_knn_k=simple_fusion_knn_k,
                simple_fusion_input_already_log1p=simple_fusion_input_already_log1p,
                simple_cross_attn_n_heads=simple_cross_attn_n_heads,
                simple_cross_attn_mlp_ratio=simple_cross_attn_mlp_ratio,
                simple_cross_attn_dropout=simple_cross_attn_dropout,
                simple_cross_attn_n_layers=simple_cross_attn_n_layers,
                simple_stpath_transformer_n_layers=simple_stpath_transformer_n_layers,
                simple_stpath_transformer_n_heads=simple_stpath_transformer_n_heads,
                simple_stpath_transformer_dropout=simple_stpath_transformer_dropout,
                simple_stpath_transformer_attn_dropout=simple_stpath_transformer_attn_dropout,
                simple_stpath_transformer_mlp_ratio=simple_stpath_transformer_mlp_ratio,
                organ_vocab=organ_vocab, tech_vocab=tech_vocab,
            )
            self.condition_projection = nn.Sequential(
                nn.LayerNorm(cond_hidden_dim),
                nn.Linear(cond_hidden_dim, score_hidden_dim),
            )

        if conditioning_mode != "uniform":
            # Keep geometry on its own path. In particular, LayerNorm(1)
            # would map every scalar distance to zero and silently destroy the
            # geometry-only control.
            self.geometry_encoder = nn.Sequential(
                nn.Linear(1, score_hidden_dim),
                nn.GELU(),
            )
            self.weight_scorer = nn.Sequential(
                nn.LayerNorm(score_hidden_dim),
                nn.GELU(),
                nn.Linear(score_hidden_dim, 1),
            )
            # Start close to the transparent local-mean anchor while retaining
            # a nonzero gradient path into the conditioner from update one.
            nn.init.normal_(self.weight_scorer[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.weight_scorer[-1].bias)

    def _neighbors(self, context, query):
        context_coords = context["coords"]
        query_coords = query["coords"]
        if context_coords.shape[0] < 1 or query_coords.shape[0] < 1:
            raise ValueError("context transport requires non-empty context and query sets")
        distances = torch.cdist(query_coords[:, :2], context_coords[:, :2])
        k = min(self.transport_k, context_coords.shape[0])
        nearest_distance, nearest_index = torch.topk(
            distances, k=k, dim=-1, largest=False, sorted=True
        )
        local_scale = nearest_distance[:, -1:].clamp_min(1e-6)
        # Distance/kth-distance is translation, rotation, reflection and
        # global-scale invariant. Do not reintroduce the absolute-coordinate
        # shortcut that the matched StormLite configs explicitly disable.
        geometry = (nearest_distance / local_scale)[..., None]
        neighbour_expression = context["expression"][nearest_index]
        return nearest_index, neighbour_expression, geometry

    def sample(self, context, query):
        nearest_index, neighbour_expression, geometry = self._neighbors(context, query)
        uniform_weights = torch.full(
            geometry.shape[:2], 1.0 / geometry.shape[1],
            device=geometry.device, dtype=geometry.dtype,
        )
        if self.conditioning_mode == "uniform":
            weights = uniform_weights
        else:
            score_hidden = self.geometry_encoder(geometry)
            if self.conditioning_mode == "storm_lite":
                condition = self._encode_context(context, query)
                score_hidden = score_hidden + self.condition_projection(condition)[:, None, :]
            logits = self.weight_scorer(score_hidden).squeeze(-1)
            weights = torch.softmax(logits, dim=-1)

        expression = torch.sum(weights[..., None] * neighbour_expression, dim=1)
        local_mean = torch.sum(uniform_weights[..., None] * neighbour_expression, dim=1)
        return {
            "coords": query["coords"],
            "expression": expression,
            "anchor_expression": local_mean,
            "residual_expression": expression - local_mean,
            "transport_weights": weights,
            "transport_neighbor_indices": nearest_index,
        }

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_error = (out["expression"] - target) / self.target_gene_scale
        loss = standardized_error.square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        anchor_mse = nn.functional.mse_loss(out["anchor_expression"], target)
        correction_rms = out["residual_expression"].square().mean().sqrt()
        entropy = -(
            out["transport_weights"]
            * out["transport_weights"].clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        self.log_dict({
            "train/loss": loss,
            "train/absolute_mse": absolute_mse,
            "train/local_mean_mse": anchor_mse,
            "train/transport_delta_rms": correction_rms,
            "train/transport_entropy": entropy,
        })
        return loss

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        return torch.optim.AdamW(parameters, lr=self.lr)

    def on_after_backward(self) -> None:
        conditioner_sq = torch.zeros((), device=self.device)
        if self.context_encoder is not None:
            for parameter in self.context_encoder.parameters():
                if parameter.grad is not None:
                    conditioner_sq = conditioner_sq + parameter.grad.detach().square().sum()
        scorer_sq = torch.zeros((), device=self.device)
        for module in (self.geometry_encoder, self.condition_projection, self.weight_scorer):
            if module is not None:
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        scorer_sq = scorer_sq + parameter.grad.detach().square().sum()
        self.log_dict({
            "train/conditioner_grad_norm": conditioner_sq.sqrt(),
            "train/transport_scorer_grad_norm": scorer_sq.sqrt(),
        })


@register_model("gene_aware_transport_regressor")
class GeneAwareContextTransportRegressor(BaseGenerativeModel):
    """Low-rank gene-aware transport over observed context expression.

    ``ContextTransportRegressor`` applies one spatial neighbor distribution to
    every gene. The overnight diagnostic showed that this collapses to a local
    mean: genes with genuinely different spatial patterns cannot choose
    different context evidence. This model keeps the same leakage-safe,
    gene-identity-preserving value path but predicts ``transport_heads``
    separate convex neighbor distributions. Every gene learns a simplex gate
    over those heads; StormLite additionally supplies a query-specific gate
    offset. The final prediction is therefore still a convex combination of
    observed values for each gene, not a dense gene decoder, while distinct
    genes can use distinct spatial kernels.

    Geometry is translation/rotation/reflection/global-scale invariant. Query
    expression is never an input, and missing query H&E remains controlled by
    the training/evaluation image mode outside this class.
    """

    def __init__(
        self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
        score_hidden_dim: int = 128, transport_k: int = 64,
        transport_heads: int = 8, conditioning_mode: str = "geometry",
        lr: float = 1e-3, conditioner_lr: float = 3e-4,
        correlation_loss_weight: float = 0.0,
        absolute_loss_weight: float = 0.0, transport_temperature: float = 1.0,
        context_encoder_type: str = "storm_lite",
        image_encoder_type: str = "none", image_feat_dim: int = 64,
        image_patch_size: int = 256, gene_encoder_type: str = "mlp",
        gene_feat_dim: int = 256, novae_dim: int | None = None,
        coord_scale: float = 1.0, storm_lite_n_layers: int = 2,
        storm_lite_n_heads: int = 4,
        storm_lite_bias_type: str = "frame_averaging",
        storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
        storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
        storm_lite_local_k: int = 32,
        storm_lite_use_absolute_coords: bool = True,
        storm_lite_input_already_log1p: bool = True,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
        organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
    ):
        super().__init__()
        if conditioning_mode not in {"geometry", "storm_lite"}:
            raise ValueError("conditioning_mode must be 'geometry' or 'storm_lite'")
        if transport_k < 1 or transport_heads < 2:
            raise ValueError("transport_k must be positive and transport_heads must be at least 2")
        if target_scale_floor <= 0 or transport_temperature <= 0:
            raise ValueError("target_scale_floor and transport_temperature must be positive")
        if correlation_loss_weight < 0 or absolute_loss_weight < 0:
            raise ValueError("loss weights must be non-negative")
        self.save_hyperparameters()
        self.conditioning_mode = str(conditioning_mode)
        self.transport_k = int(transport_k)
        self.transport_heads = int(transport_heads)
        self.transport_temperature = float(transport_temperature)
        self.correlation_loss_weight = float(correlation_loss_weight)
        self.absolute_loss_weight = float(absolute_loss_weight)
        self.lr = float(lr)
        self.conditioner_lr = float(conditioner_lr)

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(
                f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

        self.geometry_encoder = nn.Sequential(
            nn.Linear(1, score_hidden_dim),
            nn.GELU(),
            nn.Linear(score_hidden_dim, score_hidden_dim),
        )
        self.head_embedding = nn.Parameter(
            torch.randn(self.transport_heads, score_hidden_dim) * 0.02
        )
        self.head_score_vector = nn.Parameter(
            torch.randn(self.transport_heads, score_hidden_dim) * 0.02
        )
        self.score_norm = nn.LayerNorm(score_hidden_dim)
        # A full G x H matrix is small (~0.5 MB for 16k genes and 8 heads),
        # directly auditable, and importantly gives every gene its own spatial
        # mixture without introducing a dense GEX decoder.
        self.gene_head_logits = nn.Parameter(torch.zeros(n_genes, self.transport_heads))

        self.context_encoder = None
        self.condition_score_projection = None
        self.condition_gate_projection = None
        if self.conditioning_mode == "storm_lite":
            if context_encoder_type != "storm_lite":
                raise ValueError(
                    "conditioning_mode='storm_lite' requires context_encoder_type='storm_lite'"
                )
            self.context_encoder = _build_context_encoder(
                n_genes=n_genes, coord_dim=coord_dim, cond_hidden_dim=cond_hidden_dim,
                context_encoder_type=context_encoder_type,
                image_encoder_type=image_encoder_type, image_feat_dim=image_feat_dim,
                image_patch_size=image_patch_size, gene_encoder_type=gene_encoder_type,
                gene_feat_dim=gene_feat_dim, novae_dim=novae_dim, coord_scale=coord_scale,
                storm_lite_n_layers=storm_lite_n_layers,
                storm_lite_n_heads=storm_lite_n_heads,
                storm_lite_bias_type=storm_lite_bias_type,
                storm_lite_fusion_mode=storm_lite_fusion_mode,
                storm_lite_qk_norm=storm_lite_qk_norm,
                storm_lite_knn_k=storm_lite_knn_k,
                storm_lite_gnn_k=storm_lite_gnn_k,
                storm_lite_local_k=storm_lite_local_k,
                storm_lite_use_absolute_coords=storm_lite_use_absolute_coords,
                storm_lite_input_already_log1p=storm_lite_input_already_log1p,
                organ_vocab=organ_vocab, tech_vocab=tech_vocab,
            )
            self.condition_score_projection = nn.Sequential(
                nn.LayerNorm(cond_hidden_dim),
                nn.Linear(cond_hidden_dim, self.transport_heads * score_hidden_dim),
            )
            self.condition_gate_projection = nn.Sequential(
                nn.LayerNorm(cond_hidden_dim),
                nn.Linear(cond_hidden_dim, self.transport_heads),
            )

    def _neighbors(self, context: dict, query: dict):
        context_coords = context["coords"]
        query_coords = query["coords"]
        if context_coords.shape[0] < 1 or query_coords.shape[0] < 1:
            raise ValueError("gene-aware transport requires non-empty context and query sets")
        distances = torch.cdist(query_coords[:, :2], context_coords[:, :2])
        k = min(self.transport_k, context_coords.shape[0])
        nearest_distance, nearest_index = torch.topk(
            distances, k=k, dim=-1, largest=False, sorted=True
        )
        local_scale = nearest_distance[:, -1:].clamp_min(1e-6)
        geometry = (nearest_distance / local_scale)[..., None]
        neighbour_expression = context["expression"][nearest_index]
        return nearest_index, neighbour_expression, geometry

    @staticmethod
    def _mean_per_gene_correlation(prediction: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        target_ss = target_centered.square().sum(dim=0)
        eligible = target_ss > 1e-8
        if not bool(eligible.any()):
            return prediction.new_zeros(())
        numerator = (pred_centered * target_centered).sum(dim=0)
        # Clamp the squared product *before* sqrt.  Clamping only after
        # sqrt leaves autograd evaluating sqrt'(0)=inf for a gene whose
        # initial prediction is constant; the following zero numerator can
        # then produce NaN gradients on the very first optimizer step.
        pred_ss = pred_centered.square().sum(dim=0)
        denominator = torch.sqrt((pred_ss * target_ss).clamp_min(1e-8))
        return (numerator[eligible] / denominator[eligible]).mean()

    def sample(self, context, query):
        nearest_index, neighbour_expression, geometry = self._neighbors(context, query)
        n_query, k = geometry.shape[:2]
        geometric = self.geometry_encoder(geometry)[:, :, None, :]
        hidden = geometric + self.head_embedding[None, None, :, :]

        condition = None
        if self.conditioning_mode == "storm_lite":
            condition = self._encode_context(context, query)
            condition_score = self.condition_score_projection(condition).reshape(
                n_query, self.transport_heads, -1
            )
            hidden = hidden + condition_score[:, None, :, :]
        hidden = torch.nn.functional.gelu(self.score_norm(hidden))
        logits = torch.einsum("qkhd,hd->qkh", hidden, self.head_score_vector)
        head_weights = torch.softmax(
            logits.transpose(1, 2) / self.transport_temperature, dim=-1
        )  # [query, head, neighbour]
        head_expression = torch.einsum(
            "qhk,qkg->qhg", head_weights, neighbour_expression
        )

        if condition is None:
            gene_gates = torch.softmax(self.gene_head_logits, dim=-1)
            expression = torch.einsum("qhg,gh->qg", head_expression, gene_gates)
            gate_entropy = -(
                gene_gates * gene_gates.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
        else:
            query_gate = self.condition_gate_projection(condition)
            gene_gates = torch.softmax(
                self.gene_head_logits[None, :, :] + query_gate[:, None, :], dim=-1
            )
            expression = torch.einsum("qhg,qgh->qg", head_expression, gene_gates)
            gate_entropy = -(
                gene_gates * gene_gates.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()

        uniform_weights = torch.full(
            (n_query, k), 1.0 / k, device=geometry.device, dtype=geometry.dtype
        )
        local_mean = torch.sum(
            uniform_weights[..., None] * neighbour_expression, dim=1
        )
        head_entropy = -(
            head_weights * head_weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        return {
            "coords": query["coords"],
            "expression": expression,
            "anchor_expression": local_mean,
            "residual_expression": expression - local_mean,
            "transport_weights": head_weights,
            "transport_neighbor_indices": nearest_index,
            "transport_head_entropy": head_entropy,
            "gene_gate_entropy": gate_entropy,
        }

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_mse = (
            (out["expression"] - target) / self.target_gene_scale
        ).square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        anchor_mse = nn.functional.mse_loss(out["anchor_expression"], target)
        spatial_correlation = self._mean_per_gene_correlation(out["expression"], target)
        correlation_loss = 1.0 - spatial_correlation
        loss = (
            standardized_mse
            + self.absolute_loss_weight * absolute_mse
            + self.correlation_loss_weight * correlation_loss
        )
        correction_rms = out["residual_expression"].square().mean().sqrt()
        self.log_dict({
            "train/loss": loss,
            "train/standardized_mse": standardized_mse,
            "train/absolute_mse": absolute_mse,
            "train/local_mean_mse": anchor_mse,
            "train/per_gene_pcc": spatial_correlation,
            "train/transport_delta_rms": correction_rms,
            "train/transport_head_entropy": out["transport_head_entropy"],
            "train/gene_gate_entropy": out["gene_gate_entropy"],
        })
        return loss

    def configure_optimizers(self):
        if self.context_encoder is None:
            return torch.optim.AdamW(self.parameters(), lr=self.lr)
        conditioner = [
            parameter for parameter in self.context_encoder.parameters()
            if parameter.requires_grad
        ]
        conditioner_ids = {id(parameter) for parameter in conditioner}
        transport = [
            parameter for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in conditioner_ids
        ]
        return torch.optim.AdamW([
            {"params": transport, "lr": self.lr},
            {"params": conditioner, "lr": self.conditioner_lr},
        ])

    def on_after_backward(self) -> None:
        conditioner_sq = torch.zeros((), device=self.device)
        if self.context_encoder is not None:
            for parameter in self.context_encoder.parameters():
                if parameter.grad is not None:
                    conditioner_sq = conditioner_sq + parameter.grad.detach().square().sum()
        scorer_sq = torch.zeros((), device=self.device)
        scorer_modules = (
            self.geometry_encoder, self.condition_score_projection,
            self.condition_gate_projection, self.score_norm,
        )
        for module in scorer_modules:
            if module is not None:
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        scorer_sq = scorer_sq + parameter.grad.detach().square().sum()
        for parameter in (self.head_embedding, self.head_score_vector):
            if parameter.grad is not None:
                scorer_sq = scorer_sq + parameter.grad.detach().square().sum()
        gate_grad = (
            self.gene_head_logits.grad.detach().square().sum().sqrt()
            if self.gene_head_logits.grad is not None
            else torch.zeros((), device=self.device)
        )
        self.log_dict({
            "train/conditioner_grad_norm": conditioner_sq.sqrt(),
            "train/transport_scorer_grad_norm": scorer_sq.sqrt(),
            "train/gene_gate_grad_norm": gate_grad,
        })


@register_model("hierarchical_gene_transport_regressor")
class HierarchicalGeneTransportRegressor(BaseGenerativeModel):
    """Gene-value-preserving transport on top of the hierarchical encoder.

    docs/hierarchical_missing_tissue.md's held-out results showed
    ``HierarchicalMissingTissueRegressor``'s dense ``256 -> 512 -> G`` decoder
    loses to exact IDW/harmonic interpolation (PCC 0.0308 vs harmonic's
    0.0340). The diagnosis: routing ~16k exact observed gene values through
    one 256D bottleneck before decoding back out to ~16k genes discards
    precisely the gene-specific spatial structure a six-slide CCRCC cohort
    cannot re-learn from scratch. This model keeps
    ``HierarchicalMissingTissueEncoder`` (frozen GigaPath local/slide
    context, context-only Novae, STPath-style weighted-linear GEX) as the
    *conditioner* deciding which observed spots matter, but the output stays
    a weighted combination of untouched observed full-gene vectors, never a
    dense gene decoder:

        prediction(q,g) = sum_j alpha(q,j,g) * observed_expression(j,g) + residual(q,g)

    ``alpha`` is a per-gene mixture over ``transport_heads`` separate convex
    neighbor distributions (same factorization as
    ``GeneAwareContextTransportRegressor`` above — an explicit
    ``[n_genes, transport_heads]`` gate table, never a direct learned
    ``[query, gene, neighbor]`` tensor), scored from
    ``HierarchicalMissingTissueEncoder.forward_with_neighbors()``'s query
    state, each candidate neighbor's already-fused multimodal token
    (local/global H&E and Novae already baked in), and the encoder's own
    relative geometry — reusing that forward pass instead of a second,
    redundant k-NN search.

    The candidate transport is blended with an inverse-distance-weighted
    (IDW) anchor rather than trusted outright from step one:

        prediction_transport = anchor + sigmoid(blend_logit) * (candidate - anchor)

    ``blend_logit`` is initialized per-gene so ``sigmoid(blend_logit) ~ 0.05``
    — early training stays close to plain IDW (a reasonable prior, since IDW
    and harmonic both already beat the dense decoder) while keeping a live
    gradient path into the learned transport from the first step. An
    optional rank-``residual_rank`` factorized residual
    (``query_factor(q) . gene_embedding[g]``, scaled by the training-only
    per-gene scale) can add a small per-query-per-gene correction on top; its
    query projection is zero-initialized so it contributes exactly nothing
    until training earns it, and it is explicitly not a second dense
    ``256 -> G`` decoder or a free per-gene bias.

    ``conditioning_mode='geometry'`` disables ``neighbor_hidden``/
    ``query_hidden`` from the transport *scoring* path (pure relative-
    geometry kernel, still spatially adaptive) without touching the encoder
    itself or the gene gate — an isolated ablation of whether multimodal
    information changes *which* neighbor gets weight, independent of
    ``use_query_gate`` (does the per-gene gate depend on the query at all)
    and ``gene_gate_mode='shared'`` (one gate for every gene vs. one per
    gene) so the 20-run suite's C07/C08/C09 controls can be set
    independently of each other and of C05's full-richness baseline.

    ``use_global_candidate`` (2026-07-23 diagnostic follow-up) adds exactly
    one extra candidate to the transport gate's softmax competition, per
    query: a whole-slide fallback built from
    ``HierarchicalMissingTissueEncoder.forward_with_neighbors()``'s
    ``global_hidden`` (mean of every visible context spot's already
    globally-self-attended token) paired with the literal mean of every
    visible context spot's real expression. The k local candidates and the
    IDW anchor are entirely unchanged -- this only widens the *learned*
    candidate's options from "k nearest neighbors only" to "k nearest
    neighbors, or the tissue's general character, whichever this gene's
    gate prefers". Motivation: pure k-nearest-neighbor conditioning has no
    way to recover if a hole's local neighborhood happens to be
    unrepresentative of the tissue it actually contains (e.g. a hole
    straddling a tumor invasive front, where nearby expression can differ
    sharply over a short distance even though the missing tissue is still
    drawn from the same overall section). The global candidate is given a
    sentinel relative-geometry entry (zero direction, distance = 3x the
    query's own local scale, clearly out of the normal k-neighbor range)
    rather than a fabricated real position, so the scorer can learn to
    treat it distinctly by content (``neighbor_hidden``) as well as by that
    sentinel distance.

    ``use_retrieval_candidate`` (2026-07-23 round-4 architecture matrix)
    adds up to ``retrieval_k`` further candidates per query, selected by
    learned content-embedding similarity rather than physical distance --
    directly testing BLEEP's (Xie et al., NeurIPS 2023) published position
    that "the implicit assumption that spatially adjacent regions should
    have similar representations... may not be beneficial... hard coding
    position information could also lead to overfitting in data-scarce
    scenarios" against this project's own opposite finding (geometry-only
    scoring has been the strongest learned configuration in every suite so
    far). Two small linear projections (``retrieval_query_projection``,
    ``retrieval_expression_projection``) map a query's own hidden state and
    any real expression vector into a shared, L2-normalized space; at
    inference the query's projection is compared against every visible
    context spot's projected real expression, and the top ``retrieval_k``
    most similar spots are added as extra transport-gate candidates --
    using their REAL relative geometry and their real
    ``forward_with_neighbors()``-fused token (``context_hidden``), not a
    sentinel, since retrieved candidates are genuine spots with genuine
    positions, unlike the whole-slide global candidate above. Trained via
    an in-batch InfoNCE loss (``retrieval_loss_weight``,
    ``retrieval_temperature``): a query's projection is pulled toward its
    own real target expression's projection and pushed away from every
    other query's target AND every visible context spot's real expression
    in the same draw -- the same real spots the retrieval step ranks
    against at inference, so training and inference share one objective.
    The k local candidates, the IDW anchor, and the global candidate (if
    also enabled) are entirely unaffected; this only adds more options to
    the *learned* candidate's own softmax competition.

    ``use_niche_candidate`` (2026-07-23 round-5 follow-up) adds one further
    candidate: the mean of every OBSERVED context spot that shares the
    query's own spatial-domain ("niche") assignment, rather than the flat
    whole-slide mean ``use_global_candidate`` uses. Motivation: a hole
    straddling a domain boundary (e.g. a tumor invasive front) can have a
    whole-slide mean that is just as unrepresentative as its k nearest
    neighbors, since it averages over every domain in the section, not just
    the one the hole actually sits in. Niche labels are NOT computed inside
    this model -- they must arrive pre-computed on ``context["niche_labels"]``
    (an ``[Nc, 1]`` integer-valued tensor, one label per visible context
    spot), produced upstream by ``src/data/niche_features.py``'s context-only
    BANKSY-style clustering (mirrors ``ContextOnlyNovaeProvider``'s leak-safety
    discipline: the labels must be recomputed fresh on each training draw's
    observed context subgraph only, never on the full slide, since the
    clustering itself is a neighbor-averaging operation that would otherwise
    leak hidden query expression through the graph -- see
    ``src/data/context_features.py``'s module docstring for the general
    argument). The QUERY's own niche is never computed from the query's own
    (hidden) expression -- it is read off the single physically-nearest
    context neighbor's label (``neighbor_indices[:, 0]``, already the
    nearest by construction since ``forward_with_neighbors`` sorts by
    distance), which only uses real spatial position, never hidden content.
    Candidates with no niche-mate other than that nearest neighbor itself
    still get a valid (single-member) mean. Like the global candidate, this
    gets a sentinel relative-geometry entry (distance = 2x local scale,
    distinct from the global candidate's 3x so the two remain distinguishable
    if a config ever stacks both) rather than a fabricated position, since a
    niche mean is not any single real spot.

    ``local_image_encoder_type`` (2026-07-23, "general vs. histology-
    pretrained image encoder" axis) swaps ONLY the local per-spot tile
    encoder between ``"gigapath"`` (default, Prov-GigaPath, pretrained on
    real-world histology) and ``"dinov2"`` (DINOv2, pretrained entirely on
    natural images, never histology -- see
    ``src/models/conditioning.py``'s ``DINOv2PatchEncoder`` docstring for
    the motivating evidence: Wang et al. 2025's *Nat. Commun.* benchmark
    found all 11 SGE-from-H&E methods it tested use general/ImageNet-
    pretrained-or-from-scratch image backbones, none use a histology-
    specific foundation model, and its best overall performer used a
    general ResNet feature extractor). ``use_slide_context``'s whole-WSI
    LongNet aggregator stays Gigapath-only regardless of this setting --
    DINOv2 has no equivalent long-context slide aggregator, so this is
    deliberately a single-axis (local tile encoder only) comparison.

    ``gene_encoder_type='stpath_frozen_table'`` (2026-07-23) isolates
    "does STPath's PRETRAINING help" from "does STPath's whole spatial-
    transformer architecture help" -- a distinction this project's own
    internal ablation (STPath pretrained vs. unfrozen/from-scratch, PCC
    0.503 vs. 0.4706, ``docs/results_log.md``) cannot make, since it always
    uses STPath's whole architecture. Requires ``stpath_frozen_gene_table``
    ([d_model, n_genes], produced by
    ``src/models/stpath_gene_table.py::extract_stpath_gene_embedding_table``,
    grounded directly in STPath's real ``EncodeInputs.gene_embed`` weights,
    gathered per-gene by symbol -> Ensembl ID -> STPath's own vocabulary
    index). Still real observed expression as the only input
    (``real_expression @ frozen_table``, STPath's own frozen gene_embed
    computation, exactly) feeding a small trainable projection -- same RAE
    pattern as every other gene/image encoder in this file; the pretrained
    table itself is a buffer, never a Parameter, so it structurally cannot
    receive gradient.
    """

    def __init__(
        self,
        n_genes: int,
        novae_dim: int | None = None,
        hidden_dim: int = 256,
        n_heads: int = 4,
        context_layers: int = 2,
        cross_layers: int = 2,
        query_layers: int = 1,
        local_k: int = 128,
        dropout: float = 0.1,
        use_novae: bool = True,
        use_local_images: bool = True,
        local_image_encoder_type: str = "gigapath",
        use_slide_context: bool = True,
        gene_encoder_type: str = "weighted_linear",
        tokenized_gene_names: list[str] | None = None,
        tokenized_full_gene_names: list[str] | None = None,
        tokenized_pool_layers: int = 1,
        tokenized_pool_heads: int = 4,
        stpath_frozen_gene_table=None,
        fusion_mode: str = "concat",
        slide_checkpoint_path: str | None = None,
        slide_output_dim: int = 768,
        score_hidden_dim: int = 128,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        idw_power: float = 2.0,
        conditioning_mode: str = "hierarchical",
        gene_gate_mode: str = "per_gene",
        use_global_candidate: bool = False,
        use_niche_candidate: bool = False,
        use_retrieval_candidate: bool = False,
        retrieval_k: int = 8,
        retrieval_dim: int = 64,
        retrieval_temperature: float = 0.1,
        retrieval_loss_weight: float = 0.1,
        use_query_gate: bool = True,
        use_query_gene_gate: bool = False,
        query_gene_gate_rank: int = 16,
        use_residual: bool = False,
        residual_rank: int = 32,
        blend_logit_init: float = -2.9444389791664403,  # logit(0.05)
        correlation_loss_weight: float = 0.25,
        transport_reg_weight: float = 0.0,
        residual_penalty_weight: float = 1e-3,
        target_gene_scale: list[float] | None = None,
        target_scale_floor: float = 0.05,
        lr: float = 1e-3,
        conditioner_lr: float = 3e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        from src.models.hierarchical_slide import HierarchicalMissingTissueEncoder

        if conditioning_mode not in {"hierarchical", "geometry"}:
            raise ValueError("conditioning_mode must be 'hierarchical' or 'geometry'")
        if gene_gate_mode not in {"per_gene", "shared"}:
            raise ValueError("gene_gate_mode must be 'per_gene' or 'shared'")
        if transport_heads < 1:
            raise ValueError("transport_heads must be positive")
        if local_k < 1:
            raise ValueError("local_k must be positive")
        if target_scale_floor <= 0 or transport_temperature <= 0 or idw_power <= 0:
            raise ValueError(
                "target_scale_floor, transport_temperature and idw_power must be positive"
            )
        if correlation_loss_weight < 0 or transport_reg_weight < 0 or residual_penalty_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if use_residual and residual_rank < 1:
            raise ValueError("residual_rank must be positive when use_residual=True")
        if use_retrieval_candidate:
            if retrieval_k < 1:
                raise ValueError("retrieval_k must be positive when use_retrieval_candidate=True")
            if retrieval_dim < 1:
                raise ValueError("retrieval_dim must be positive when use_retrieval_candidate=True")
            if retrieval_temperature <= 0:
                raise ValueError("retrieval_temperature must be positive when use_retrieval_candidate=True")
            if retrieval_loss_weight < 0:
                raise ValueError("retrieval_loss_weight must be non-negative")
        self.save_hyperparameters()

        self.n_genes = int(n_genes)
        self.context_encoder = HierarchicalMissingTissueEncoder(
            n_genes=n_genes, novae_dim=novae_dim, hidden_dim=hidden_dim,
            n_heads=n_heads, context_layers=context_layers, cross_layers=cross_layers,
            query_layers=query_layers, local_k=local_k, dropout=dropout,
            use_novae=use_novae, use_local_images=use_local_images,
            local_image_encoder_type=local_image_encoder_type,
            use_slide_context=use_slide_context, gene_encoder_type=gene_encoder_type,
            tokenized_gene_names=tokenized_gene_names,
            tokenized_full_gene_names=tokenized_full_gene_names,
            tokenized_pool_layers=tokenized_pool_layers,
            tokenized_pool_heads=tokenized_pool_heads,
            stpath_frozen_gene_table=stpath_frozen_gene_table,
            fusion_mode=fusion_mode, slide_checkpoint_path=slide_checkpoint_path,
            slide_output_dim=slide_output_dim,
        )

        self.conditioning_mode = str(conditioning_mode)
        self.gene_gate_mode = str(gene_gate_mode)
        self.use_global_candidate = bool(use_global_candidate)
        self.use_niche_candidate = bool(use_niche_candidate)
        self.use_retrieval_candidate = bool(use_retrieval_candidate)
        self.retrieval_k = int(retrieval_k)
        self.retrieval_temperature = float(retrieval_temperature)
        self.retrieval_loss_weight = float(retrieval_loss_weight)
        self.retrieval_query_projection = None
        self.retrieval_expression_projection = None
        if self.use_retrieval_candidate:
            self.retrieval_query_projection = nn.Linear(hidden_dim, retrieval_dim)
            self.retrieval_expression_projection = nn.Linear(n_genes, retrieval_dim)
        self.use_query_gate = bool(use_query_gate)
        self.use_query_gene_gate = bool(use_query_gene_gate)
        self.use_residual = bool(use_residual)
        self.transport_heads = int(transport_heads)
        self.transport_temperature = float(transport_temperature)
        self.idw_power = float(idw_power)
        self.correlation_loss_weight = float(correlation_loss_weight)
        self.transport_reg_weight = float(transport_reg_weight)
        self.residual_penalty_weight = float(residual_penalty_weight)
        self.lr = float(lr)
        self.conditioner_lr = float(conditioner_lr)
        self.weight_decay = float(weight_decay)

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(
                f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}"
            )
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

        # Transport scorer: relative geometry (3D, straight from
        # forward_with_neighbors) + optionally the neighbor's own fused
        # multimodal token and the query state -> one logit per head.
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, score_hidden_dim), nn.GELU(),
            nn.Linear(score_hidden_dim, score_hidden_dim),
        )
        self.head_embedding = nn.Parameter(
            torch.randn(self.transport_heads, score_hidden_dim) * 0.02
        )
        self.head_score_vector = nn.Parameter(
            torch.randn(self.transport_heads, score_hidden_dim) * 0.02
        )
        self.score_norm = nn.LayerNorm(score_hidden_dim)
        self.neighbor_projection = None
        self.query_score_projection = None
        if self.conditioning_mode == "hierarchical":
            self.neighbor_projection = nn.Linear(hidden_dim, score_hidden_dim)
            self.query_score_projection = nn.Linear(hidden_dim, score_hidden_dim)

        gate_genes = 1 if self.gene_gate_mode == "shared" else n_genes
        # A full [genes, heads] table is small (~0.5 MB for 16k genes and 8
        # heads) and directly auditable — see ContextTransportRegressor's own
        # docstring for why this project avoids a dense gene decoder here too.
        self.gene_head_logits = nn.Parameter(torch.zeros(gate_genes, self.transport_heads))
        self.query_gate_projection = (
            nn.Linear(hidden_dim, self.transport_heads) if self.use_query_gate else None
        )
        self.query_gene_gate_query = None
        self.query_gene_gate_table = None
        if self.use_query_gene_gate:
            self.query_gene_gate_query = nn.Linear(hidden_dim, query_gene_gate_rank)
            self.query_gene_gate_table = nn.Parameter(
                torch.randn(n_genes, query_gene_gate_rank, self.transport_heads) * 0.02
            )

        self.blend_logit = nn.Parameter(torch.full((n_genes,), float(blend_logit_init)))

        self.residual_rank = int(residual_rank) if use_residual else 0
        self.residual_query_projection = None
        self.residual_gene_embedding = None
        if self.use_residual:
            self.residual_query_projection = nn.Linear(hidden_dim, self.residual_rank)
            # "initialize the final query projection to exactly zero" — the
            # residual contributes nothing until training moves this weight.
            nn.init.zeros_(self.residual_query_projection.weight)
            nn.init.zeros_(self.residual_query_projection.bias)
            self.residual_gene_embedding = nn.Parameter(
                torch.randn(n_genes, self.residual_rank) * (1.0 / math.sqrt(self.residual_rank))
            )

    @staticmethod
    def _mean_per_gene_correlation(prediction: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        pred_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        target_ss = target_centered.square().sum(dim=0)
        eligible = target_ss > 1e-8
        if not bool(eligible.any()):
            return prediction.new_zeros(())
        numerator = (pred_centered * target_centered).sum(dim=0)
        pred_ss = pred_centered.square().sum(dim=0)
        denominator = torch.sqrt((pred_ss * target_ss).clamp_min(1e-8))
        return (numerator[eligible] / denominator[eligible]).mean()

    def _idw_anchor(self, neighbor_distances: torch.Tensor,
                     neighbour_expression: torch.Tensor):
        """Non-learned inverse-distance-weighted interpolation — the
        per-step anchor the handoff requires in place of full graph-harmonic
        interpolation (too expensive to run every training step)."""
        weights = 1.0 / neighbor_distances.clamp_min(1e-6).pow(self.idw_power)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        anchor = torch.sum(weights[..., None] * neighbour_expression, dim=1)
        return anchor, weights

    def sample(self, context, query):
        encoded = self.context_encoder.forward_with_neighbors(context, query)
        query_hidden = encoded["query_hidden"]            # [Nq, H]
        neighbor_hidden = encoded["neighbor_hidden"]        # [Nq, k, H]
        neighbor_indices = encoded["neighbor_indices"]      # [Nq, k]
        neighbor_distances = encoded["neighbor_distances"]  # [Nq, k]
        relative_geometry = encoded["relative_geometry"]    # [Nq, k, 3]
        n_query, k = neighbor_indices.shape

        neighbour_expression = context["expression"][neighbor_indices]  # [Nq, k, G]
        anchor_expression, idw_weights = self._idw_anchor(neighbor_distances, neighbour_expression)

        # The IDW anchor above is deliberately computed from ONLY the k real
        # local neighbors -- it must stay a pure, non-learned local
        # interpolation, unaffected by whatever the learned candidate does.
        # The global candidate (if enabled) only ever widens the *learned*
        # transport candidate's own options, one extra slot in its softmax
        # gate alongside the k local ones.
        scoring_relative_geometry = relative_geometry
        scoring_neighbor_hidden = neighbor_hidden
        scoring_neighbour_expression = neighbour_expression
        if self.use_global_candidate:
            global_hidden = encoded["global_hidden"]  # [H]
            global_expression = context["expression"].mean(dim=0)  # [G]
            # relative_geometry's distance channel is already normalized by
            # each query's own local_scale (the k-th/farthest local
            # neighbor's distance, see forward_with_neighbors), so real
            # neighbors fall in roughly (0, 1]. A constant 3.0 here reads as
            # "three times farther than your farthest real local neighbor" --
            # clearly out of that range without needing a fabricated
            # position, so the scorer can learn to treat this slot as
            # categorically different from a real neighbor.
            global_relative = relative_geometry.new_zeros(n_query, 1, 3)
            global_relative[..., 2] = 3.0
            scoring_relative_geometry = torch.cat([relative_geometry, global_relative], dim=1)
            scoring_neighbor_hidden = torch.cat(
                [neighbor_hidden, global_hidden[None, None, :].expand(n_query, 1, -1)], dim=1
            )
            scoring_neighbour_expression = torch.cat(
                [neighbour_expression, global_expression[None, None, :].expand(n_query, 1, -1)], dim=1
            )

        if self.use_niche_candidate:
            if "niche_labels" not in context:
                raise ValueError(
                    "use_niche_candidate=True requires context['niche_labels'] "
                    "([Nc, 1], produced by src/data/niche_features.py's "
                    "context-only clustering) -- see this class's own "
                    "docstring for why niche labels cannot be computed inside "
                    "the model itself."
                )
            niche_labels = context["niche_labels"].reshape(-1)  # [Nc]
            if niche_labels.shape[0] != context["expression"].shape[0]:
                raise ValueError(
                    "context['niche_labels'] must have exactly one row per "
                    f"context spot ({context['expression'].shape[0]}), got "
                    f"{niche_labels.shape[0]}"
                )
            # The query's own niche is read off its single nearest CONTEXT
            # neighbor (real spatial position only) -- never from the
            # query's own hidden expression, which would leak the answer
            # into the lookup used to help predict it.
            query_niche = niche_labels[neighbor_indices[:, 0]]  # [Nq]
            # Dense [Nq, Nc] membership matrix -- Nc is at most a few
            # thousand context spots per slide, so this stays one cheap
            # batched matmul rather than a python loop over niches.
            membership = (niche_labels[None, :] == query_niche[:, None]).to(neighbor_hidden.dtype)
            member_count = membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
            niche_hidden = (membership @ encoded["context_hidden"]) / member_count  # [Nq, H]
            niche_expression = (membership @ context["expression"]) / member_count  # [Nq, G]

            niche_relative = relative_geometry.new_zeros(n_query, 1, 3)
            niche_relative[..., 2] = 2.0  # distinct sentinel from the global candidate's 3.0
            scoring_relative_geometry = torch.cat([scoring_relative_geometry, niche_relative], dim=1)
            scoring_neighbor_hidden = torch.cat(
                [scoring_neighbor_hidden, niche_hidden[:, None, :]], dim=1
            )
            scoring_neighbour_expression = torch.cat(
                [scoring_neighbour_expression, niche_expression[:, None, :]], dim=1
            )

        if self.use_retrieval_candidate:
            # Rank EVERY visible context spot (not just the k physically
            # nearest) by learned content-embedding similarity to this
            # query, take the top retrieval_k. Unlike the global candidate,
            # these are genuine individual spots with real positions, so
            # they get real relative geometry (same normalization/
            # relative_coord embedding as the k local neighbors) rather
            # than a sentinel.
            query_key = torch.nn.functional.normalize(
                self.retrieval_query_projection(query_hidden), dim=-1
            )  # [Nq, D]
            context_key = torch.nn.functional.normalize(
                self.retrieval_expression_projection(context["expression"]), dim=-1
            )  # [Nc, D]
            similarity = query_key @ context_key.T  # [Nq, Nc]
            retrieval_k = min(self.retrieval_k, context["expression"].shape[0])
            _, retrieval_idx = torch.topk(similarity, k=retrieval_k, dim=-1)  # [Nq, rk]

            retrieval_delta = context["coords"][retrieval_idx][..., :2] - query["coords"][:, None, :2]
            retrieval_distance = torch.linalg.norm(retrieval_delta, dim=-1, keepdim=True)
            local_scale = neighbor_distances[:, -1:].clamp_min(1e-6)  # [Nq, 1]
            retrieval_relative = torch.cat(
                [retrieval_delta / local_scale[..., None], retrieval_distance / local_scale[..., None]],
                dim=-1,
            )  # [Nq, rk, 3]
            retrieval_hidden = (
                encoded["context_hidden"][retrieval_idx] + self.context_encoder.relative_coord(retrieval_relative)
            )  # [Nq, rk, H] -- same fused token + positional embedding every k-nearest neighbor gets
            retrieval_expression = context["expression"][retrieval_idx]  # [Nq, rk, G]

            scoring_relative_geometry = torch.cat([scoring_relative_geometry, retrieval_relative], dim=1)
            scoring_neighbor_hidden = torch.cat([scoring_neighbor_hidden, retrieval_hidden], dim=1)
            scoring_neighbour_expression = torch.cat(
                [scoring_neighbour_expression, retrieval_expression], dim=1
            )

        hidden = self.geometry_encoder(scoring_relative_geometry)[:, :, None, :]  # [Nq,k(+1),1,S]
        hidden = hidden + self.head_embedding[None, None, :, :]           # [1,1,H,S]
        if self.conditioning_mode == "hierarchical":
            hidden = hidden + self.neighbor_projection(scoring_neighbor_hidden)[:, :, None, :]
            hidden = hidden + self.query_score_projection(query_hidden)[:, None, None, :]
        hidden = torch.nn.functional.gelu(self.score_norm(hidden))
        logits = torch.einsum("qkhd,hd->qkh", hidden, self.head_score_vector)
        head_weights = torch.softmax(
            logits.transpose(1, 2) / self.transport_temperature, dim=-1
        )  # [Nq, heads, k(+1)]
        head_expression = torch.einsum(
            "qhk,qkg->qhg", head_weights, scoring_neighbour_expression
        )  # [Nq, heads, G]

        base_gate = (
            self.gene_head_logits.expand(self.n_genes, -1)
            if self.gene_gate_mode == "shared" else self.gene_head_logits
        )
        gate_logits = base_gate[None, :, :].expand(n_query, -1, -1)  # [Nq, G, heads]
        if self.query_gate_projection is not None:
            gate_logits = gate_logits + self.query_gate_projection(query_hidden)[:, None, :]
        if self.query_gene_gate_table is not None:
            query_low = self.query_gene_gate_query(query_hidden)  # [Nq, rank]
            gate_logits = gate_logits + torch.einsum(
                "qr,grh->qgh", query_low, self.query_gene_gate_table
            )
        gene_gates = torch.softmax(gate_logits, dim=-1)  # [Nq, G, heads]
        candidate = torch.einsum("qhg,qgh->qg", head_expression, gene_gates)

        blend = torch.sigmoid(self.blend_logit)[None, :]  # [1, G]
        transport = anchor_expression + blend * (candidate - anchor_expression)

        residual = torch.zeros_like(transport)
        if self.use_residual:
            query_factor = self.residual_query_projection(query_hidden)  # [Nq, rank]
            residual = torch.einsum(
                "qr,gr->qg", query_factor, self.residual_gene_embedding
            ) * self.target_gene_scale[None, :]

        expression = transport + residual
        head_entropy = -(
            head_weights * head_weights.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()
        gate_entropy = -(
            gene_gates * gene_gates.clamp_min(1e-12).log()
        ).sum(dim=-1).mean()

        return {
            "coords": query["coords"],
            "expression": expression,
            "anchor_expression": anchor_expression,
            "residual_expression": expression - anchor_expression,
            "transport_expression": transport,
            "factorized_residual": residual,
            "transport_head_entropy": head_entropy,
            "gene_gate_entropy": gate_entropy,
            "idw_weights": idw_weights,
            "query_hidden": query_hidden,
        }

    def training_step(self, batch, batch_idx):
        out = self.sample(batch["context"], batch["query"])
        target = batch["target_expression"]
        standardized_error = (out["expression"] - target) / self.target_gene_scale
        standardized_mse = standardized_error.square().mean()
        absolute_mse = nn.functional.mse_loss(out["expression"], target)
        anchor_mse = nn.functional.mse_loss(out["anchor_expression"], target)
        spatial_correlation = self._mean_per_gene_correlation(out["expression"], target)
        correlation_loss = 1.0 - spatial_correlation

        # "small transport regularization" (handoff, exact form unspecified):
        # originally a negative-entropy bonus rewarding a soft multi-head
        # neighbor distribution. 2026-07-22 20-run suite evidence: with
        # transport_reg_weight=1e-3, transport_head_entropy sat at ~4.81-4.85
        # (ln(128)=4.852, the true maximum for k=128) for the ENTIRE 20k-step
        # run on every config -- the neighbor-weighting mechanism never
        # learned to specialize at all. That plausibly explains why every
        # richness axis (Novae, H&E, extra heads, richer gates) failed to
        # help: a near-uniform selector can't express extra information
        # regardless of how much is available to it. Defaulting the weight
        # to 0.0 (see __init__) and keeping the term itself only so it can
        # be re-tested at a much smaller value if a soft prior is ever
        # actually wanted -- not applied by default.
        transport_regularization = -out["transport_head_entropy"]
        standardized_residual = out["factorized_residual"] / self.target_gene_scale
        residual_penalty = standardized_residual.square().mean()

        retrieval_loss = standardized_mse.new_zeros(())
        if self.use_retrieval_candidate:
            # In-batch InfoNCE: a query's projection should be closest to
            # its OWN real target's projection, among every other query's
            # target AND every visible context spot's real expression in
            # this same draw -- the same real spots the retrieval step in
            # sample() ranks against, so training and inference share one
            # objective (BLEEP's own real contrastive design, adapted:
            # BLEEP's query side is a real image embedding since it always
            # has the query's own real image; ours is query_hidden, since
            # the query's own image/expression are exactly what's hidden).
            query_key = nn.functional.normalize(
                self.retrieval_query_projection(out["query_hidden"]), dim=-1
            )  # [Nq, D]
            key_pool = torch.cat([target, batch["context"]["expression"]], dim=0)  # [Nq+Nc, G]
            pool_key = nn.functional.normalize(
                self.retrieval_expression_projection(key_pool), dim=-1
            )  # [Nq+Nc, D]
            retrieval_logits = query_key @ pool_key.T / self.retrieval_temperature  # [Nq, Nq+Nc]
            retrieval_labels = torch.arange(query_key.shape[0], device=query_key.device)
            retrieval_loss = nn.functional.cross_entropy(retrieval_logits, retrieval_labels)

        loss = (
            standardized_mse
            + self.correlation_loss_weight * correlation_loss
            + self.transport_reg_weight * transport_regularization
            + self.residual_penalty_weight * residual_penalty
            + self.retrieval_loss_weight * retrieval_loss
        )

        transport_delta_rms = (
            out["transport_expression"] - out["anchor_expression"]
        ).square().mean().sqrt()
        factorized_residual_rms = out["factorized_residual"].square().mean().sqrt()
        prediction_delta_rms = out["residual_expression"].square().mean().sqrt()

        self.log_dict({
            "train/loss": loss,
            "train/standardized_mse": standardized_mse,
            "train/absolute_mse": absolute_mse,
            "train/anchor_mse": anchor_mse,
            "train/per_gene_pcc": spatial_correlation,
            "train/transport_delta_rms": transport_delta_rms,
            "train/factorized_residual_rms": factorized_residual_rms,
            "train/prediction_delta_rms": prediction_delta_rms,
            "train/transport_head_entropy": out["transport_head_entropy"],
            "train/gene_gate_entropy": out["gene_gate_entropy"],
            "train/retrieval_loss": retrieval_loss,
        })
        return loss

    def configure_optimizers(self):
        encoder_params = [p for p in self.context_encoder.parameters() if p.requires_grad]
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids
        ]
        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": self.lr})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.conditioner_lr})
        if not groups:
            return None
        return torch.optim.AdamW(groups, weight_decay=self.weight_decay)

    def on_after_backward(self) -> None:
        encoder_params = list(self.context_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        encoder_sq = torch.zeros((), device=self.device)
        for parameter in encoder_params:
            if parameter.grad is not None:
                encoder_sq = encoder_sq + parameter.grad.detach().square().sum()

        residual_params = []
        if self.residual_query_projection is not None:
            residual_params.extend(self.residual_query_projection.parameters())
        if self.residual_gene_embedding is not None:
            residual_params.append(self.residual_gene_embedding)
        residual_ids = {id(p) for p in residual_params}
        residual_sq = torch.zeros((), device=self.device)
        for parameter in residual_params:
            if parameter.grad is not None:
                residual_sq = residual_sq + parameter.grad.detach().square().sum()

        transport_sq = torch.zeros((), device=self.device)
        for parameter in self.parameters():
            if parameter.grad is None:
                continue
            if id(parameter) in encoder_ids or id(parameter) in residual_ids:
                continue
            transport_sq = transport_sq + parameter.grad.detach().square().sum()

        self.log_dict({
            "train/hierarchical_encoder_grad_norm": encoder_sq.sqrt(),
            "train/transport_grad_norm": transport_sq.sqrt(),
            "train/factorized_residual_grad_norm": residual_sq.sqrt(),
        })


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
                 stpath_pretrained: bool = True, stpath_input_already_log1p: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_bias_type: str = "frame_averaging",
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
                 storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
                 storm_lite_input_already_log1p: bool = True,
                 storm_lite_tokenizer_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_full_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_n_pool_layers: int = 1, storm_lite_tokenizer_n_pool_heads: int = 4,
                 coord_scale: float = 1.0,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
                 decoder_type: str = "dense", decoder_gene_names: list[str] | None = None,
                 decoder_gene_embed_dim: int = 64,
                 decoder_hidden_dim: int | None = None, decoder_mlp_depth: int = 1,
                 decoder_combine_mode: str = "concat",
                 decoder_attn_n_heads: int = 4, decoder_attn_n_layers: int = 1,
                 decoder_lloki_tech_embed_dim: int = 10,
                 decoder_lloki_hidden_dims: list[int] | None = None,
                 full_gene_names: list[str] | None = None,
                 warmup_steps: int = 0):
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
            stpath_input_already_log1p=stpath_input_already_log1p,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_bias_type=storm_lite_bias_type,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            storm_lite_fusion_mode=storm_lite_fusion_mode, storm_lite_qk_norm=storm_lite_qk_norm,
            storm_lite_knn_k=storm_lite_knn_k, storm_lite_gnn_k=storm_lite_gnn_k,
            storm_lite_input_already_log1p=storm_lite_input_already_log1p,
            storm_lite_tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            storm_lite_tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            storm_lite_tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            storm_lite_tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            coord_scale=coord_scale, organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # deterministic encoder, no logvar
        )
        self.decoder = _build_decoder(
            in_dim=latent_dim + cond_hidden_dim, n_genes=n_genes, dense_hidden_dim=hidden_dim,
            decoder_type=decoder_type, decoder_gene_names=decoder_gene_names,
            decoder_gene_embed_dim=decoder_gene_embed_dim, tech_vocab=tech_vocab,
            decoder_hidden_dim=decoder_hidden_dim, decoder_mlp_depth=decoder_mlp_depth,
            decoder_combine_mode=decoder_combine_mode,
            decoder_attn_n_heads=decoder_attn_n_heads, decoder_attn_n_layers=decoder_attn_n_layers,
            decoder_lloki_tech_embed_dim=decoder_lloki_tech_embed_dim,
            decoder_lloki_hidden_dims=decoder_lloki_hidden_dims,
        )
        self._decoder_gene_names = decoder_gene_names
        if decoder_type == "gene_attention":
            assert full_gene_names is not None, (
                "decoder_type='gene_attention' requires full_gene_names (auto-injected "
                "by inject_decoder_gene_names) -- needed to align this decoder's "
                "restricted output panel with target_expression's full-width columns, "
                "see BaseGenerativeModel._slice_target_for_decoder"
            )
            name_to_idx = {g: i for i, g in enumerate(full_gene_names)}
            missing = [g for g in decoder_gene_names if g not in name_to_idx]
            assert not missing, (
                f"decoder_gene_names contains {len(missing)} gene(s) not in "
                f"full_gene_names (e.g. {missing[:5]})"
            )
            self.register_buffer(
                "_decoder_target_col_idx",
                torch.tensor([name_to_idx[g] for g in decoder_gene_names], dtype=torch.long),
            )
        self.decoder_type = decoder_type
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, disc_hidden_dim), nn.ReLU(),
            nn.Linear(disc_hidden_dim, 1),   # logit: real-prior-sample vs. encoder output
        )
        self.latent_dim = latent_dim
        self.adv_weight = adv_weight
        self.lr = lr
        self.lr_disc = lr_disc
        self.warmup_steps = warmup_steps

    def _decode(self, h: torch.Tensor, tech: str | None = None) -> torch.Tensor:
        """Dispatches on decoder_type (2026-07-17, see PanelInvariantGeneDecoder
        docstring) so sample()/training_step() don't need their own branching.
        "gene_attention" (GeneAttentionDecoder) needs gene_names passed
        explicitly every call — unlike PanelInvariantGeneDecoder, it does
        NOT default to the full training vocabulary (see its own docstring:
        that default would silently trigger the O(n_panel^2) blowup it
        exists to guard against)."""
        if self.decoder_type == "dense":
            return self.decoder(h)
        elif self.decoder_type == "gene_attention":
            return self.decoder(h, gene_names=self._decoder_gene_names, tech=tech)
        return self.decoder(h, tech=tech)

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
        expr_gen = self._decode(torch.cat([z, c], dim=-1), tech=query.get("tech"))
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
        # only touch lr_schedulers() when warmup is actually enabled — it
        # requires an attached Trainer (raises RuntimeError otherwise),
        # which several existing tests deliberately don't set up (they
        # call training_step() directly against a bare model, monkey-
        # patching just optimizers()/manual_backward()/log_dict() — see
        # tests/test_wae_gan.py's own comment). warmup_steps=0 (default)
        # must stay a genuine no-op, not merely "no scheduler stepped."
        schedulers = self.lr_schedulers() if self.warmup_steps > 0 else None
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
        recon = self._decode(torch.cat([z_fake, c], dim=-1), tech=query.get("tech"))
        recon_loss = nn.functional.mse_loss(recon, self._slice_target_for_decoder(target_expression))
        logits_fake_for_ae = self.discriminator(z_fake)
        adv_loss = nn.functional.binary_cross_entropy_with_logits(
            logits_fake_for_ae, torch.ones_like(logits_fake_for_ae)  # fool disc: look like prior
        )
        ae_loss = recon_loss + self.adv_weight * adv_loss
        opt_ae.zero_grad()
        self.manual_backward(ae_loss)
        opt_ae.step()

        # LR warmup (2026-07-19, opt-in via warmup_steps): manual
        # optimization means Lightning does NOT step schedulers
        # automatically (unlike automatic-optimization models' "interval":
        # "step" config) — must be stepped explicitly here, once per
        # optimizer, every training step. self.lr_schedulers() returns
        # None when warmup_steps=0 (configure_optimizers attaches no
        # scheduler at all in that case — see its own comment).
        if schedulers is not None:
            sched_ae, sched_disc = schedulers
            sched_ae.step()
            sched_disc.step()

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
        if self.warmup_steps > 0:
            # warmup_steps=0 (default) returns [opt_ae, opt_disc] exactly as
            # before this feature existed — zero behavior change unless a
            # config explicitly opts in. See _linear_warmup_lr_lambda's own
            # docstring and training_step's manual .step() calls above
            # (manual optimization means Lightning won't step these itself).
            sched_ae = torch.optim.lr_scheduler.LambdaLR(
                opt_ae, lr_lambda=_linear_warmup_lr_lambda(self.warmup_steps))
            sched_disc = torch.optim.lr_scheduler.LambdaLR(
                opt_disc, lr_lambda=_linear_warmup_lr_lambda(self.warmup_steps))
            return [opt_ae, opt_disc], [sched_ae, sched_disc]
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


class _AdaLNResidualBlock(nn.Module):
    """DiT-style AdaLN-Zero residual block (Peebles & Xie 2023, "Scalable
    Diffusion Models with Transformers") — LayerNorm modulated by a
    (shift, scale) pair derived from the conditioning vector, output
    gated by a third derived value before the residual add. AdaLN-Zero's
    trick (the modulation projection's weight AND bias zero-initialized)
    makes every block the identity function at initialization, so a deep
    stack is trainable from scratch with no separate warmup schedule —
    each block only gradually learns to contribute as training proceeds."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ada = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(cond).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.mlp(h)
        return x + gate * h


class _AdaLNVelocityNet(nn.Module):
    """Residual, per-layer-conditioned replacement for the plain concat-MLP
    velocity_net (2026-07-19 architecture audit, see docs/results_log.md):
    the original velocity_net is a 2-hidden-layer feedforward MLP with NO
    residual connections, injecting the context vector c (derived from a
    context_encoder that can be 60-100M+ params) via one-time
    concatenation at the input layer only — deeper layers see it purely
    secondhand. Every comparable published flow-matching/diffusion
    architecture (DiT, SiT, SD3's MM-DiT) uses residual blocks with
    conditioning re-injected at EVERY layer via AdaLN specifically because
    a plain deep feedforward net without either is hard to optimize and
    dilutes the conditioning signal with depth. Opt-in via
    velocity_net_type="adaln_residual" (default "mlp" = old behavior,
    byte-for-byte unchanged).

    Accepts the SAME single concatenated [z_t, t_embed, c] tensor as the
    plain MLP (splits it back apart internally) so _velocity/_edm_denoise
    call sites need zero changes regardless of which type is active."""

    def __init__(self, latent_dim: int, time_embed_dim: int, cond_hidden_dim: int,
                 hidden_dim: int, n_layers: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        cond_dim = time_embed_dim + cond_hidden_dim
        self.blocks = nn.ModuleList([_AdaLNResidualBlock(hidden_dim, cond_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_t, cond = x[..., :self.latent_dim], x[..., self.latent_dim:]
        h = self.in_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)


@register_model("fm_ot")
class FlowMatchingOT(BaseGenerativeModel):
    def __init__(self, n_genes: int, coord_dim: int = 3, cond_hidden_dim: int = 256,
                 latent_dim: int = 32, ae_hidden_dim: int = 256,
                 hidden_dim: int = 512, time_embed_dim: int = 64,
                 n_ode_steps: int = 50, recon_weight: float = 1.0,
                 fm_weight: float = 1.0, lr: float = 1e-3,
                 fm_time_sampling: str = "uniform",
                 fm_logit_normal_m: float = 0.0, fm_logit_normal_s: float = 1.0,
                 fm_coupling: str = "independent",
                 ode_solver: str = "euler",
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
                 stpath_pretrained: bool = True, stpath_input_already_log1p: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_bias_type: str = "frame_averaging",
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
                 storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
                 storm_lite_input_already_log1p: bool = True,
                 storm_lite_tokenizer_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_full_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_n_pool_layers: int = 1, storm_lite_tokenizer_n_pool_heads: int = 4,
                 coord_scale: float = 1.0,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
                 decoder_type: str = "dense", decoder_gene_names: list[str] | None = None,
                 decoder_gene_embed_dim: int = 64,
                 decoder_hidden_dim: int | None = None, decoder_mlp_depth: int = 1,
                 decoder_combine_mode: str = "concat",
                 decoder_attn_n_heads: int = 4, decoder_attn_n_layers: int = 1,
                 decoder_lloki_tech_embed_dim: int = 10,
                 decoder_lloki_hidden_dims: list[int] | None = None,
                 full_gene_names: list[str] | None = None,
                 warmup_steps: int = 0,
                 velocity_net_type: str = "mlp", velocity_net_n_layers: int = 3,
                 boundary_consistency_weight: float = 0.0,
                 boundary_consistency_bandwidth: float = 100.0):
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
            stpath_input_already_log1p=stpath_input_already_log1p,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_bias_type=storm_lite_bias_type,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            storm_lite_fusion_mode=storm_lite_fusion_mode, storm_lite_qk_norm=storm_lite_qk_norm,
            storm_lite_knn_k=storm_lite_knn_k, storm_lite_gnn_k=storm_lite_gnn_k,
            storm_lite_input_already_log1p=storm_lite_input_already_log1p,
            storm_lite_tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            storm_lite_tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            storm_lite_tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            storm_lite_tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            coord_scale=coord_scale, organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        # own autoencoder, own weights — compresses expression to a small
        # latent code the velocity net operates on instead of raw n_genes
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, latent_dim),
        )
        self.decoder = _build_decoder(
            in_dim=latent_dim + cond_hidden_dim, n_genes=n_genes, dense_hidden_dim=ae_hidden_dim,
            decoder_type=decoder_type, decoder_gene_names=decoder_gene_names,
            decoder_gene_embed_dim=decoder_gene_embed_dim, tech_vocab=tech_vocab,
            decoder_hidden_dim=decoder_hidden_dim, decoder_mlp_depth=decoder_mlp_depth,
            decoder_combine_mode=decoder_combine_mode,
            decoder_attn_n_heads=decoder_attn_n_heads, decoder_attn_n_layers=decoder_attn_n_layers,
            decoder_lloki_tech_embed_dim=decoder_lloki_tech_embed_dim,
            decoder_lloki_hidden_dims=decoder_lloki_hidden_dims,
        )
        self._decoder_gene_names = decoder_gene_names
        if decoder_type == "gene_attention":
            assert full_gene_names is not None, (
                "decoder_type='gene_attention' requires full_gene_names (auto-injected "
                "by inject_decoder_gene_names) -- needed to align this decoder's "
                "restricted output panel with target_expression's full-width columns, "
                "see BaseGenerativeModel._slice_target_for_decoder"
            )
            name_to_idx = {g: i for i, g in enumerate(full_gene_names)}
            missing = [g for g in decoder_gene_names if g not in name_to_idx]
            assert not missing, (
                f"decoder_gene_names contains {len(missing)} gene(s) not in "
                f"full_gene_names (e.g. {missing[:5]})"
            )
            self.register_buffer(
                "_decoder_target_col_idx",
                torch.tensor([name_to_idx[g] for g in decoder_gene_names], dtype=torch.long),
            )
        self.decoder_type = decoder_type
        self.time_embed = _SinusoidalTimeEmbedding(time_embed_dim)
        assert velocity_net_type in ("mlp", "adaln_residual"), (
            f"unknown velocity_net_type {velocity_net_type!r}"
        )
        if velocity_net_type == "mlp":
            self.velocity_net = nn.Sequential(
                nn.Linear(latent_dim + time_embed_dim + cond_hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.velocity_net = _AdaLNVelocityNet(
                latent_dim=latent_dim, time_embed_dim=time_embed_dim,
                cond_hidden_dim=cond_hidden_dim, hidden_dim=hidden_dim,
                n_layers=velocity_net_n_layers,
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
        self.warmup_steps = warmup_steps
        # boundary_consistency_weight (2026-07-20, adapted from DISCO
        # [Duan et al. 2025, "DISCO: A Diffusion Model for Spatial
        # Transcriptomics Data Completion", verified via direct PDF read]
        # -- see _boundary_consistency_loss's own docstring for the full
        # adaptation reasoning and honest scope limits). 0.0 (default) is
        # a true no-op: the term is skipped entirely in training_step, so
        # every existing config is byte-for-byte unaffected.
        self.boundary_consistency_weight = boundary_consistency_weight
        self.boundary_consistency_bandwidth = boundary_consistency_bandwidth
        assert fm_time_sampling in ("uniform", "logit_normal"), (
            f"unknown fm_time_sampling {fm_time_sampling!r}"
        )
        self.fm_time_sampling = fm_time_sampling
        self.fm_logit_normal_m = fm_logit_normal_m
        self.fm_logit_normal_s = fm_logit_normal_s
        assert fm_coupling in ("independent", "minibatch_ot"), (
            f"unknown fm_coupling {fm_coupling!r}"
        )
        self.fm_coupling = fm_coupling
        assert ode_solver in ("euler", "heun"), f"unknown ode_solver {ode_solver!r}"
        self.ode_solver = ode_solver

    def _sample_flow_time(self, n: int) -> torch.Tensor:
        """Timestep sampling for the OT flow-matching loss. "uniform"
        (default) draws t ~ U[0,1] — the original, unchanged behavior for
        every existing config. "logit_normal" (2026-07-19, verified
        improvement from Stable Diffusion 3 / Esser et al. 2024, "Scaling
        Rectified Flow Transformers for High-Resolution Image Synthesis",
        arXiv 2403.03206) instead draws t = sigmoid(m + s*eps), eps ~
        N(0,1), concentrating probability mass around the informative
        MIDDLE of the noise->data trajectory (t~0.5) rather than spending
        equal supervision on the near-noise (t~0, weak structure) and
        near-data (t~1, redundant) ends. SD3 showed this outperforms plain
        uniform rectified-flow sampling and EDM/LDM-linear baselines on
        training efficiency; m=0/s=1 (defaults) is SD3's own default,
        symmetric around t=0.5. Only affects the "ot" path (edm has its
        own separate P_mean/P_std log-normal sigma sampling already)."""
        if self.fm_time_sampling == "logit_normal":
            eps = torch.randn(n, device=self.device)
            return torch.sigmoid(self.fm_logit_normal_m + self.fm_logit_normal_s * eps)
        return torch.rand(n, device=self.device)

    def _couple_noise(self, z_1_target: torch.Tensor) -> torch.Tensor:
        """Sample z_0 (the OT path's noise endpoint) either "independent"
        (default — z_0 = torch.randn_like(z_1_target), i.i.d., the ORIGINAL
        behavior for every existing config) or "minibatch_ot" (2026-07-19,
        Tong et al. 2023, "Improving and Generalizing Flow-Based Generative
        Models with Minibatch Optimal Transport"; Pooladian et al. 2023,
        "Multisample Flow Matching: Straightening Flows with Minibatch
        Couplings", ICML — same technique).

        Real gap this closes: this class is NAMED "FlowMatchingOT" for its
        straight-line OT-style PATH formulation, but its actual z_0<->z_1
        PAIRING was never OT-coupled — z_0 was just an independent i.i.d.
        draw, no different from plain (non-OT) conditional flow matching.
        Real minibatch OT solves the assignment between a POOL of n noise
        samples and the n real targets in THIS training step (this
        project's own natural "minibatch": every query point in one
        masking draw, processed together in one training_step call) via
        the Hungarian algorithm on squared-Euclidean cost — the exact
        discrete optimal-transport plan for that cost, not an
        approximation. Pairing each target with the closest-in-latent-
        space noise sample (rather than a random one) gives straighter,
        less-crossing paths, which both papers show improves sample
        quality and reduces the variance of the training gradient.

        scipy.optimize.linear_sum_assignment (Hungarian/Kuhn-Munkres) is
        exact and already available (scipy is an existing dependency, see
        src/evaluation/metrics.py's own scipy.linalg usage) — no new
        dependency. O(n^3) in the worst case, negligible at this
        project's query-set sizes (~15-45 points per masking draw).
        "independent" (default) never touches scipy at all — zero
        behavior/dependency change unless a config opts in."""
        z_0 = torch.randn_like(z_1_target)
        if self.fm_coupling == "independent":
            return z_0
        from scipy.optimize import linear_sum_assignment
        with torch.no_grad():
            cost = torch.cdist(z_0, z_1_target, p=2).pow(2).cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost)
            # row_idx is already 0..n-1 in order for a square cost matrix;
            # col_idx[i] is the target index z_0[i] gets assigned to —
            # invert so z_0_reordered[j] is the noise paired with z_1[j]
            perm = torch.empty(z_1_target.shape[0], dtype=torch.long, device=z_0.device)
            perm[torch.as_tensor(col_idx, device=z_0.device)] = torch.as_tensor(row_idx, device=z_0.device)
        return z_0[perm]

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

    def _boundary_consistency_loss(self, query_coords: torch.Tensor, context_coords: torch.Tensor,
                                    context_expr: torch.Tensor, pred_expr: torch.Tensor) -> torch.Tensor:
        """2026-07-20, adapted from DISCO (Duan, Li, Zhang, Song, Zhang
        2025, "DISCO: A Diffusion Model for Spatial Transcriptomics Data
        Completion", Proc Int Conf Image Proc — verified via direct PDF
        read, not a guess). DISCO's own ablation (their Section 3.3) found
        that removing "integration with neighboring region" during
        generation was its single biggest lever — MSE 0.89->1.07 (+20%),
        EMD 18.4->21.9 (+19%) — bigger than removing tissue-type
        conditioning entirely. Their mechanism: at EVERY diffusion
        denoising step, real observed neighboring values are re-noised to
        match that step's noise level and spliced back into the SAME
        state tensor being denoised, forcing the generated region to stay
        consistent with real boundary context throughout generation, not
        just via a single conditioning vector computed once up front.

        HONEST ADAPTATION, not a literal port: FM-OT's state space (a
        per-query-point LATENT code integrated via ODE) has no shared
        tensor with context the way DISCO's joint per-cell diffusion state
        does, so DISCO's exact re-noise-and-splice trick isn't
        dimensionally transferable. This ports the SAME underlying idea
        (explicit boundary-consistency supervision, concentrated near real
        observed context) as a training-time auxiliary loss instead:
        pulls each query point's DECODED prediction toward its single
        nearest real context spot's actual measured expression, weighted
        by an exponential decay in coordinate distance (bandwidth
        controls how fast the pull fades — small bandwidth: only points
        immediately at the masked-hole boundary are pulled; large
        bandwidth: pulls extend deep into the hole). This deliberately
        does NOT pull every query point toward a neighbor average
        unconditionally — DISCO's own baselines table shows a plain KNN
        completion method is a WEAK baseline, underperforming every
        learned method; an unweighted version of this loss would risk
        dragging predictions toward that same weak KNN-like behavior.
        Weighting by distance keeps the effect concentrated exactly where
        DISCO's ablation showed it mattered (the boundary), leaving deep-
        hole points free to rely on the model's actual learned generative
        prior instead of a naive local-smoothness assumption.

        0.0-weight callers (default) never call this at all -- see
        boundary_consistency_weight in __init__."""
        dist = torch.cdist(query_coords, context_coords)  # [n_query, n_context]
        min_dist, nn_idx = dist.min(dim=1)
        nearest_context_expr = context_expr[nn_idx]  # [n_query, n_genes] (or decoder-panel width)
        weight = torch.exp(-min_dist / self.boundary_consistency_bandwidth)
        per_point_loss = ((pred_expr - nearest_context_expr) ** 2).mean(dim=-1)
        return (weight * per_point_loss).mean()

    def _decode(self, h: torch.Tensor, tech: str | None = None) -> torch.Tensor:
        """Dispatches on decoder_type (2026-07-17, see PanelInvariantGeneDecoder
        docstring) so sample()/training_step() don't need their own branching.
        "gene_attention" (GeneAttentionDecoder) needs gene_names passed
        explicitly every call — unlike PanelInvariantGeneDecoder, it does
        NOT default to the full training vocabulary (see its own docstring:
        that default would silently trigger the O(n_panel^2) blowup it
        exists to guard against)."""
        if self.decoder_type == "dense":
            return self.decoder(h)
        elif self.decoder_type == "gene_attention":
            return self.decoder(h, gene_names=self._decoder_gene_names, tech=tech)
        return self.decoder(h, tech=tech)

    def prepare_sampling_conditioning(self, context, query) -> dict:
        """Encode deterministic context once for repeated stochastic draws.

        Audit evaluation and fixed-mask validation draw many flow samples for
        the same context/query item.  Re-running a large frozen conditioner
        (especially STPath) for every noise draw is mathematically redundant
        in eval mode and was the reason final evaluation appeared to hang.
        The returned object deliberately retains context/query as well as the
        encoded tensor so subclasses such as ResidualFlowMatchingOT can reuse
        their deterministic anchor without changing their public output.
        """
        return {
            "context": context,
            "query": query,
            "conditioning": self._encode_context(context, query),
        }

    def _sample_from_conditioning(self, c: torch.Tensor, query: dict) -> dict:
        n = query["coords"].shape[0]

        if self.path_type == "ot":
            z = torch.randn(n, self.latent_dim, device=self.device)
            dt = 1.0 / self.n_ode_steps
            if self.ode_solver == "euler":
                for step in range(self.n_ode_steps):
                    t = torch.full((n,), step * dt, device=self.device)
                    z = z + dt * self._velocity(z, t, c)  # manual Euler ODE integration, in latent space
            else:  # heun — 2nd-order predictor-corrector (Karras et al. 2022,
                   # "Elucidating the Design Space of Diffusion-Based
                   # Generative Models", NeurIPS — the EDM paper this
                   # project's own path_type="edm" already cites for its
                   # preconditioning; its OWN recommended sampler is Heun's
                   # method, not plain Euler, which is what "ot" used until
                   # now — a documented simplification, see this class's
                   # module docstring "Simplifications flagged explicitly"
                   # note. Two velocity evaluations per step (predictor +
                   # corrector) instead of one, for O(dt^3) local error
                   # instead of Euler's O(dt^2) — same n_ode_steps, better
                   # accuracy, or equivalent accuracy at fewer steps.
                for step in range(self.n_ode_steps):
                    t_cur = torch.full((n,), step * dt, device=self.device)
                    v_cur = self._velocity(z, t_cur, c)
                    z_pred = z + dt * v_cur                       # Euler predictor
                    t_next = torch.full((n,), (step + 1) * dt, device=self.device)
                    v_next = self._velocity(z_pred, t_next, c)
                    z = z + dt * 0.5 * (v_cur + v_next)           # trapezoidal corrector
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

        expr_gen = self._decode(torch.cat([z, c], dim=-1), tech=query.get("tech"))
        return {"coords": query["coords"], "expression": expr_gen}

    def sample_from_prepared_conditioning(self, prepared: dict) -> dict:
        """Draw once from a value returned by prepare_sampling_conditioning."""
        return self._sample_from_conditioning(
            prepared["conditioning"], prepared["query"]
        )

    def sample(self, context, query):
        # Preserve the public one-shot API. Repeated evaluation calls use the
        # explicit prepare/sample_from_prepared pair through predictive_samples.
        c = self._encode_context(context, query)
        return self._sample_from_conditioning(c, query)

    def training_step(self, batch, batch_idx):
        context, query = batch["context"], batch["query"]
        x_1 = batch["target_expression"]  # real target expression
        n = x_1.shape[0]
        c = self._encode_context(context, query)

        z_1 = self.encoder(x_1)
        recon = self._decode(torch.cat([z_1, c], dim=-1), tech=query.get("tech"))
        recon_loss = nn.functional.mse_loss(recon, self._slice_target_for_decoder(x_1))

        # flow-matching/diffusion target sees a frozen (detached) latent
        # code, so the encoder/decoder are trained only by recon_loss —
        # approximates a pretrained-then-frozen autoencoder without a
        # separate stage
        z_1_target = z_1.detach()

        if self.path_type == "ot":
            z_0 = self._couple_noise(z_1_target)   # independent (default) or minibatch OT, see _couple_noise
            t = self._sample_flow_time(n)   # uniform (default) or logit-normal (SD3, see _sample_flow_time)
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
        log_dict = {"train/recon": recon_loss, "train/fm_loss": fm_loss}
        if self.boundary_consistency_weight > 0:
            # 2026-07-20, adapted from DISCO — see _boundary_consistency_loss's
            # own docstring. Uses `recon` (this step's own decoded query
            # prediction, already computed above for recon_loss) rather than
            # a fresh sample() call — cheap, and the same prediction
            # recon_loss already supervises, just with an added
            # boundary-proximity-weighted pull toward real nearest context.
            boundary_loss = self._boundary_consistency_loss(
                query["coords"], context["coords"],
                self._slice_target_for_decoder(context["expression"]), recon,
            )
            loss = loss + self.boundary_consistency_weight * boundary_loss
            log_dict["train/boundary_consistency"] = boundary_loss
        log_dict["train/loss"] = loss
        self.log_dict(log_dict)
        return loss

    def configure_optimizers(self):
        # AdamW (decoupled weight decay) over plain Adam: standard choice in
        # the flow-matching/diffusion literature (Lipman et al. 2022 and
        # essentially all follow-ups use AdamW, not Adam).
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.warmup_steps > 0:
            # 2026-07-19, opt-in LR warmup (see _linear_warmup_lr_lambda's
            # own docstring) — "interval": "step" means Lightning steps
            # this itself every training batch under AUTOMATIC optimization
            # (unlike WAEGAN's manual .step() calls). warmup_steps=0
            # (default) returns the bare optimizer exactly as before this
            # feature existed.
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=_linear_warmup_lr_lambda(self.warmup_steps))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt



@register_model("residual_fm_ot")
class ResidualFlowMatchingOT(FlowMatchingOT):
    """Flow matching on residuals around a graph-harmonic completion.

    The deterministic spatial interpolation carries low-frequency structure;
    the autoencoder and flow model only learn ``target - harmonic_anchor``.
    This makes the baseline explicit in every prediction and prevents a
    stochastic model from spending capacity relearning simple smoothness.
    """

    def __init__(self, *args, harmonic_k: int = 8, harmonic_ridge: float = 1e-4,
                 pretrained_autoencoder_path: str | None = None,
                 freeze_pretrained_autoencoder: bool = True,
                 pretrained_autoencoder_max_rmse: float | None = None,
                 pretrained_autoencoder_min_pcc: float | None = None,
                 pretrained_autoencoder_gene_names: list[str] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.harmonic_k = int(harmonic_k)
        self.harmonic_ridge = float(harmonic_ridge)
        self.pretrained_ae_decoder = None
        if pretrained_autoencoder_path is not None:
            if self.decoder_type != "dense":
                raise ValueError("pretrained residual autoencoder currently requires decoder_type='dense'")
            checkpoint = torch.load(pretrained_autoencoder_path, map_location="cpu")
            if checkpoint.get("target_type") != "harmonic_residual":
                raise ValueError(
                    "pretrained autoencoder was not validated on harmonic residuals. "
                    "Re-run scripts/pretrain_expression_autoencoder.py with the audit config; "
                    "absolute-expression autoencoders are incompatible with residual_fm_ot."
                )
            checkpoint_k = int(checkpoint.get("harmonic_k", -1))
            checkpoint_ridge = float(checkpoint.get("harmonic_ridge", float("nan")))
            if checkpoint_k != self.harmonic_k or not math.isclose(
                checkpoint_ridge, self.harmonic_ridge, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "pretrained autoencoder harmonic anchor does not match the residual flow "
                    f"configuration: checkpoint k/ridge={checkpoint_k}/{checkpoint_ridge}, "
                    f"model={self.harmonic_k}/{self.harmonic_ridge}"
                )
            checkpoint_genes = [str(g) for g in checkpoint.get("gene_names", [])]
            if pretrained_autoencoder_gene_names is not None:
                expected_gene_names = [str(g) for g in pretrained_autoencoder_gene_names]
                if checkpoint_genes != expected_gene_names:
                    raise ValueError(
                        "pretrained autoencoder gene names/order do not match the fit-derived "
                        "model panel; refusing a width-only match that could permute genes"
                    )
            validation_rmse = float(checkpoint.get("validation_rmse", float("inf")))
            validation_pcc = float(checkpoint.get("validation_pcc", -float("inf")))
            if (pretrained_autoencoder_max_rmse is not None
                    and validation_rmse > float(pretrained_autoencoder_max_rmse)):
                raise ValueError(
                    f"pretrained autoencoder RMSE {validation_rmse:.6f} exceeds required "
                    f"maximum {float(pretrained_autoencoder_max_rmse):.6f}"
                )
            if (pretrained_autoencoder_min_pcc is not None
                    and validation_pcc < float(pretrained_autoencoder_min_pcc)):
                raise ValueError(
                    f"pretrained autoencoder PCC {validation_pcc:.4f} is below required "
                    f"minimum {float(pretrained_autoencoder_min_pcc):.4f}"
                )
            expected_genes = int(self.n_genes)
            if int(checkpoint["n_genes"]) != expected_genes:
                raise ValueError(
                    f"autoencoder gene width {checkpoint['n_genes']} does not match model width {expected_genes}"
                )
            if int(checkpoint["latent_dim"]) != int(self.latent_dim):
                raise ValueError("autoencoder latent_dim does not match residual flow model")
            self.encoder.load_state_dict(checkpoint["encoder_state"])
            ae_hidden_dim = int(checkpoint["hidden_dim"])
            self.pretrained_ae_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, ae_hidden_dim), nn.ReLU(),
                nn.Linear(ae_hidden_dim, expected_genes),
            )
            self.pretrained_ae_decoder.load_state_dict(checkpoint["decoder_state"])
            # The pretrained decoder carries the initial reconstruction. The
            # context-conditioned decoder starts as an exact zero correction.
            last_linear = next(
                (module for module in reversed(list(self.decoder.modules()))
                 if isinstance(module, nn.Linear)), None
            )
            if last_linear is None:
                raise RuntimeError("dense conditional decoder has no Linear output layer")
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
            if freeze_pretrained_autoencoder:
                for parameter in self.encoder.parameters():
                    parameter.requires_grad = False
                for parameter in self.pretrained_ae_decoder.parameters():
                    parameter.requires_grad = False

    def _harmonic_anchor(self, context, query):
        anchor = harmonic_interpolate(
            context["coords"], context["expression"], query["coords"],
            k=self.harmonic_k, ridge=self.harmonic_ridge,
        )
        idx = getattr(self, "_decoder_target_col_idx", None)
        return anchor if idx is None else anchor[:, idx]

    def _decode(self, h: torch.Tensor, tech: str | None = None) -> torch.Tensor:
        conditional = super()._decode(h, tech=tech)
        if self.pretrained_ae_decoder is None:
            return conditional
        base = self.pretrained_ae_decoder(h[:, :self.latent_dim])
        return base + conditional

    def prepare_sampling_conditioning(self, context, query) -> dict:
        prepared = super().prepare_sampling_conditioning(context, query)
        prepared["anchor"] = self._harmonic_anchor(context, query)
        return prepared

    def sample_from_prepared_conditioning(self, prepared: dict) -> dict:
        residual = super().sample_from_prepared_conditioning(prepared)
        anchor = prepared["anchor"]
        residual_expression = residual["expression"]
        expression = anchor + residual_expression
        return {"coords": prepared["query"]["coords"], "expression": expression,
                "anchor_expression": anchor, "residual_expression": residual_expression}

    def sample(self, context, query):
        prepared = self.prepare_sampling_conditioning(context, query)
        return self.sample_from_prepared_conditioning(prepared)

    def training_step(self, batch, batch_idx):
        context, query = batch["context"], batch["query"]
        x_abs = self._slice_target_for_decoder(batch["target_expression"])
        anchor = self._harmonic_anchor(context, query).detach()
        x_1 = x_abs - anchor
        n = x_1.shape[0]
        c = self._encode_context(context, query)

        z_1 = self.encoder(x_1)
        recon_residual = self._decode(torch.cat([z_1, c], dim=-1), tech=query.get("tech"))
        recon_loss = nn.functional.mse_loss(recon_residual, x_1)
        absolute_recon_loss = nn.functional.mse_loss(anchor + recon_residual, x_abs)
        z_1_target = z_1.detach()

        if self.path_type == "ot":
            z_0 = self._couple_noise(z_1_target)
            t = self._sample_flow_time(n)
            z_t = (1 - t[:, None]) * z_0 + t[:, None] * z_1_target
            target_velocity = z_1_target - z_0
            pred_velocity = self._velocity(z_t, t, c)
            fm_loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        else:
            log_sigma = self.edm_p_mean + self.edm_p_std * torch.randn(n, device=self.device)
            sigma = torch.exp(log_sigma)
            noise = torch.randn_like(z_1_target)
            z_sigma = z_1_target + sigma[:, None] * noise
            d_pred = self._edm_denoise(z_sigma, sigma, c)
            weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
            fm_loss = (weight[:, None] * (d_pred - z_1_target) ** 2).mean()

        loss = self.recon_weight * recon_loss + self.fm_weight * fm_loss
        self.log_dict({
            "train/loss": loss,
            "train/residual_recon": recon_loss,
            "train/absolute_recon": absolute_recon_loss,
            "train/fm_loss": fm_loss,
        })
        return loss


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
                 stpath_pretrained: bool = True, stpath_input_already_log1p: bool = True,
                 storm_lite_n_layers: int = 2, storm_lite_n_heads: int = 4,
                 storm_lite_bias_type: str = "frame_averaging",
                 storm_lite_relative_bias_hidden_dim: int = 32,
                 storm_lite_fusion_mode: str = "sum", storm_lite_qk_norm: bool = False,
                 storm_lite_knn_k: int | None = None, storm_lite_gnn_k: int = 8,
                 storm_lite_input_already_log1p: bool = True,
                 storm_lite_tokenizer_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_full_gene_names: list[str] | None = None,
                 storm_lite_tokenizer_n_pool_layers: int = 1, storm_lite_tokenizer_n_pool_heads: int = 4,
                 coord_scale: float = 1.0,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None,
                 decoder_type: str = "dense", decoder_gene_names: list[str] | None = None,
                 decoder_gene_embed_dim: int = 64,
                 decoder_hidden_dim: int | None = None, decoder_mlp_depth: int = 1,
                 decoder_combine_mode: str = "concat",
                 decoder_attn_n_heads: int = 4, decoder_attn_n_layers: int = 1,
                 decoder_lloki_tech_embed_dim: int = 10,
                 decoder_lloki_hidden_dims: list[int] | None = None,
                 full_gene_names: list[str] | None = None,
                 warmup_steps: int = 0):
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
            stpath_input_already_log1p=stpath_input_already_log1p,
            storm_lite_n_layers=storm_lite_n_layers, storm_lite_n_heads=storm_lite_n_heads,
            storm_lite_bias_type=storm_lite_bias_type,
            storm_lite_relative_bias_hidden_dim=storm_lite_relative_bias_hidden_dim,
            storm_lite_fusion_mode=storm_lite_fusion_mode, storm_lite_qk_norm=storm_lite_qk_norm,
            storm_lite_knn_k=storm_lite_knn_k, storm_lite_gnn_k=storm_lite_gnn_k,
            storm_lite_input_already_log1p=storm_lite_input_already_log1p,
            storm_lite_tokenizer_gene_names=storm_lite_tokenizer_gene_names,
            storm_lite_tokenizer_full_gene_names=storm_lite_tokenizer_full_gene_names,
            storm_lite_tokenizer_n_pool_layers=storm_lite_tokenizer_n_pool_layers,
            storm_lite_tokenizer_n_pool_heads=storm_lite_tokenizer_n_pool_heads,
            coord_scale=coord_scale, organ_vocab=organ_vocab, tech_vocab=tech_vocab,
        )
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, ae_hidden_dim), nn.ReLU(),
            nn.Linear(ae_hidden_dim, latent_dim),
        )
        self.decoder = _build_decoder(
            in_dim=latent_dim, n_genes=n_genes, dense_hidden_dim=ae_hidden_dim,
            decoder_type=decoder_type, decoder_gene_names=decoder_gene_names,
            decoder_gene_embed_dim=decoder_gene_embed_dim, tech_vocab=tech_vocab,
            decoder_hidden_dim=decoder_hidden_dim, decoder_mlp_depth=decoder_mlp_depth,
            decoder_combine_mode=decoder_combine_mode,
            decoder_attn_n_heads=decoder_attn_n_heads, decoder_attn_n_layers=decoder_attn_n_layers,
            decoder_lloki_tech_embed_dim=decoder_lloki_tech_embed_dim,
            decoder_lloki_hidden_dims=decoder_lloki_hidden_dims,
        )
        self._decoder_gene_names = decoder_gene_names
        if decoder_type == "gene_attention":
            assert full_gene_names is not None, (
                "decoder_type='gene_attention' requires full_gene_names (auto-injected "
                "by inject_decoder_gene_names) -- needed to align this decoder's "
                "restricted output panel with target_expression's full-width columns, "
                "see BaseGenerativeModel._slice_target_for_decoder"
            )
            name_to_idx = {g: i for i, g in enumerate(full_gene_names)}
            missing = [g for g in decoder_gene_names if g not in name_to_idx]
            assert not missing, (
                f"decoder_gene_names contains {len(missing)} gene(s) not in "
                f"full_gene_names (e.g. {missing[:5]})"
            )
            self.register_buffer(
                "_decoder_target_col_idx",
                torch.tensor([name_to_idx[g] for g in decoder_gene_names], dtype=torch.long),
            )
        self.decoder_type = decoder_type
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
        self.warmup_steps = warmup_steps

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

    def _decode(self, h: torch.Tensor, tech: str | None = None) -> torch.Tensor:
        """Dispatches on decoder_type (2026-07-17, see PanelInvariantGeneDecoder
        docstring) so sample()/training_step() don't need their own branching.
        "gene_attention" (GeneAttentionDecoder) needs gene_names passed
        explicitly every call — unlike PanelInvariantGeneDecoder, it does
        NOT default to the full training vocabulary (see its own docstring:
        that default would silently trigger the O(n_panel^2) blowup it
        exists to guard against)."""
        if self.decoder_type == "dense":
            return self.decoder(h)
        elif self.decoder_type == "gene_attention":
            return self.decoder(h, gene_names=self._decoder_gene_names, tech=tech)
        return self.decoder(h, tech=tech)

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
        expr_gen_ordered = self._decode(z_q, tech=query.get("tech"))

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
        recon = self._decode(z_q, tech=query.get("tech"))
        recon_loss = nn.functional.mse_loss(recon, self._slice_target_for_decoder(x_1))

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
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.warmup_steps > 0:
            # 2026-07-19, opt-in LR warmup — see FlowMatchingOT's own
            # configure_optimizers comment for the full reasoning
            # (identical pattern, automatic optimization).
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=_linear_warmup_lr_lambda(self.warmup_steps))
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt
