"""As-simple-as-possible GigaPath + gene-encoder conditioning (2026-07-24).

Motivated directly by reading STPath's own real architecture (verified
against stpath/model/model.py): its entire multimodal fusion is "project
each modality to d_model, sum the projections, feed a small transformer."
Neither existing context encoder in this project sits at that same point
on the complexity scale -- "builtin"/SpatialContextEncoder and
"storm_lite" (fusion_mode="sum") both still run every context+query spot
through a full nn.TransformerEncoder self-attention pass before returning
the query's row. This module provides the two ends of the minimal
contrast the recovery_suite/lung round wants: SimpleFusionContextEncoder
(sum tokens, uniform mean-pool over the k nearest context spots -- zero
attention parameters) and SimpleCrossAttentionContextEncoder (identical
token construction, but ONE learned cross-attention layer instead of a
fixed uniform average). Comparing the two isolates exactly one variable:
does learned attention over neighbors beat unweighted averaging, holding
the GigaPath encoder, gene encoder, and neighbor set fixed.

Both classes reuse the same tested building blocks as every other context
encoder in this project (GigapathPatchEncoder, MLPGeneEncoder, _knn_indices
-- all from conditioning.py) rather than reimplementing image/gene
encoding from scratch. Query gene expression is never available (that is
the value being predicted) -- both classes use a learned mask_token in its
place, the same real fix StormLiteContextEncoder already uses instead of
STPath's own zero-fill (see stpath's real fusion: a zeroed feature still
passes through a biased nn.Linear and injects a real, wrong signal; a
dedicated learned mask_token has no such failure mode)."""
from __future__ import annotations

import json

import torch
import torch.nn as nn

from src.models.conditioning import GigapathPatchEncoder, MLPGeneEncoder, _knn_indices


def _build_universal_vocab_mapping(gene_names: list[str], gene_voc_path: str, caller: str):
    """Shared by UniversalMLPGeneEncoder and UniversalLinearGeneEncoder.
    Mirrors STPath's real GeneExpTokenizer vocabulary construction exactly
    (stpath/tokenization/ge_tokenizer.py, verified 2026-07-24): symbol ->
    symbol2gene[symbol] (an Ensembl-style ID) -> gene2id[that ID], where
    gene2id enumerates the SORTED SET of unique values in symbol2gene.json,
    offset by 2 (STPath reserves 0/1 for pad/mask -- kept here too so the
    vocabulary SIZE and every gene's ID are identical to STPath's real
    ones, not just the same ordering rule applied fresh). Genes in the
    local panel not in STPath's vocabulary are silently dropped (same as
    STPath's own real out-of-vocabulary handling)."""
    with open(gene_voc_path) as f:
        symbol2gene = json.load(f)
    unique_gene_ids = sorted(set(symbol2gene.values()))
    gene2id = {gene_id: i + 2 for i, gene_id in enumerate(unique_gene_ids)}
    n_vocab_tokens = max(gene2id.values()) + 1

    local_idx, vocab_idx = [], []
    for i, symbol in enumerate(gene_names):
        mapped = symbol2gene.get(symbol)
        if mapped is not None and mapped in gene2id:
            local_idx.append(i)
            vocab_idx.append(gene2id[mapped])
    if not local_idx:
        raise ValueError(f"{caller}: none of the supplied gene_names are in the vocabulary at {gene_voc_path}")
    n_mapped, n_total = len(local_idx), len(gene_names)
    print(f"{caller}: {n_mapped}/{n_total} local genes ({n_mapped / n_total:.0%}) "
          f"mapped into STPath's real {n_vocab_tokens}-token gene-identity vocabulary.")
    return n_vocab_tokens, local_idx, vocab_idx


class UniversalMLPGeneEncoder(nn.Module):
    """MLPGeneEncoder (ours: 2-layer, LayerNorm+GELU, real nonlinear
    depth), but scattered into STPath's own real fixed gene-ID vocabulary
    space first, instead of this dataset's local ad-hoc column order
    (2026-07-24 request: "keep our MLP... but a bit adjusted to better
    look like stpath's").

    Isolates ONE variable against the plain MLPGeneEncoder: does a fixed,
    cross-dataset-stable gene identity space help, holding encoder depth
    and nonlinearity roughly fixed (both are 2-layer MLPs; STPath's own
    real gene_embed is actually a single bias-free Linear with less depth
    than either of these -- see UniversalLinearGeneEncoder below for the
    exact-match version of that)."""

    def __init__(self, gene_names: list[str], gene_voc_path: str, feat_dim: int = 128,
                 hidden_dim: int = 512, bottleneck_dim: int = 256):
        super().__init__()
        n_vocab_tokens, local_idx, vocab_idx = _build_universal_vocab_mapping(
            gene_names, gene_voc_path, "UniversalMLPGeneEncoder"
        )
        self.n_vocab_tokens = n_vocab_tokens
        self.register_buffer("local_idx", torch.as_tensor(local_idx, dtype=torch.long))
        self.register_buffer("vocab_idx", torch.as_tensor(vocab_idx, dtype=torch.long))
        self.mlp_encoder = MLPGeneEncoder(
            n_genes=n_vocab_tokens, feat_dim=feat_dim, hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        n = expression.shape[0]
        scattered = expression.new_zeros((n, self.n_vocab_tokens))
        scattered[:, self.vocab_idx] = expression[:, self.local_idx]
        return self.mlp_encoder(scattered)


class UniversalLinearGeneEncoder(nn.Module):
    """The exact same universal gene-ID vocabulary scatter as
    UniversalMLPGeneEncoder, but encoded with a single bias-free Linear
    layer instead of our 2-layer MLP -- a literal copy of STPath's own
    real gene_embed mechanism (`nn.Linear(n_genes, d_model, bias=False)`,
    verified directly against stpath/model/model.py), reused on top of
    our own architectures (2026-07-24 request: "an exact copy of the gene
    encoder stpath uses"). Isolates the last remaining variable between
    our gene encoding and STPath's real one: with the SAME vocabulary
    (UniversalMLPGeneEncoder already covers that) and now the SAME
    encoder shape too, any remaining difference against 301/305's real
    STPath numbers has to come from elsewhere (the backbone/decoder/
    organ-tech, each already isolated by the other configs this round)."""

    def __init__(self, gene_names: list[str], gene_voc_path: str, feat_dim: int = 128):
        super().__init__()
        n_vocab_tokens, local_idx, vocab_idx = _build_universal_vocab_mapping(
            gene_names, gene_voc_path, "UniversalLinearGeneEncoder"
        )
        self.n_vocab_tokens = n_vocab_tokens
        self.register_buffer("local_idx", torch.as_tensor(local_idx, dtype=torch.long))
        self.register_buffer("vocab_idx", torch.as_tensor(vocab_idx, dtype=torch.long))
        self.gene_embed = nn.Linear(n_vocab_tokens, feat_dim, bias=False)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        n = expression.shape[0]
        scattered = expression.new_zeros((n, self.n_vocab_tokens))
        scattered[:, self.vocab_idx] = expression[:, self.local_idx]
        return self.gene_embed(scattered)


def _build_gene_encoder(gene_encoder_type: str, n_genes: int, feat_dim: int,
                         gene_names: list[str] | None, gene_voc_path: str | None) -> nn.Module:
    """Shared dispatch so SimpleCrossAttentionContextEncoder and
    SimpleFusionSpatialTransformerContextEncoder (2026-07-24) offer the
    same three gene-encoder choices without duplicating the same check
    three times: 'local_mlp' (default, this dataset's own ad-hoc gene
    panel, our 2-layer MLP), 'universal_mlp' (STPath's real fixed
    gene-ID vocabulary, still our MLP), 'universal_linear' (STPath's real
    vocabulary AND STPath's real single-Linear-no-bias encoder shape --
    a literal copy of its gene_embed mechanism)."""
    if gene_encoder_type == "local_mlp":
        return MLPGeneEncoder(n_genes, feat_dim=feat_dim)
    if gene_encoder_type in ("universal_mlp", "universal_linear"):
        if not gene_names or not gene_voc_path:
            raise ValueError(
                f"gene_encoder_type={gene_encoder_type!r} requires both gene_names and gene_voc_path"
            )
        if gene_encoder_type == "universal_mlp":
            return UniversalMLPGeneEncoder(gene_names=gene_names, gene_voc_path=gene_voc_path, feat_dim=feat_dim)
        return UniversalLinearGeneEncoder(gene_names=gene_names, gene_voc_path=gene_voc_path, feat_dim=feat_dim)
    raise ValueError(
        f"unknown gene_encoder_type {gene_encoder_type!r}, must be 'local_mlp', "
        "'universal_mlp', or 'universal_linear'"
    )


class SimpleFusionContextEncoder(nn.Module):
    """GigaPath image embed + gene-MLP embed, summed per spot (STPath's own
    fusion), aggregated over each query's k nearest context spots by a
    plain, unweighted mean -- no attention, no transformer, no extra
    parameters beyond the two encoders themselves."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, knn_k: int = 16,
                 input_already_log1p: bool = True):
        super().__init__()
        if knn_k < 1:
            raise ValueError("knn_k must be positive")
        self.hidden_dim = int(hidden_dim)
        self.knn_k = int(knn_k)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = MLPGeneEncoder(n_genes, feat_dim=hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def _context_tokens(self, context_expression: torch.Tensor,
                         context_images: torch.Tensor | None) -> torch.Tensor:
        n_context = context_expression.shape[0]
        device = context_expression.device
        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            img_embed = self.image_encoder(context_images)
        else:
            img_embed = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        return img_embed + gene_embed

    def _query_own_image(self, n_query: int, query_images: torch.Tensor | None,
                          device: torch.device) -> torch.Tensor:
        if query_images is not None:
            return self.image_encoder(query_images)
        return self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]. context_novae_features/organ/tech
        accepted (matching every other context encoder's call signature in
        _encode_context) and unused -- this encoder is deliberately
        GigaPath+gene-only, no niche/organ conditioning."""
        n_query = query_coords.shape[0]
        device = query_coords.device
        context_tokens = self._context_tokens(context_expression, context_images)
        neighbor_idx = _knn_indices(query_coords[:, :2], context_coords[:, :2], self.knn_k)
        neighbor_tokens = context_tokens[neighbor_idx]  # [n_query, k, hidden_dim]
        pooled = neighbor_tokens.mean(dim=1)
        query_img = self._query_own_image(n_query, query_images, device)
        return pooled + query_img


class _CrossAttnBlock(nn.Module):
    """One pre-norm cross-attention transformer block: query attends over
    context key/value tokens, residual, then a 2-layer FFN, residual. Used
    by SimpleCrossAttentionContextEncoder, stacked n_layers times."""

    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden), nn.GELU(), nn.Linear(ffn_hidden, hidden_dim),
        )

    def forward(self, x: torch.Tensor, kv_source: torch.Tensor) -> torch.Tensor:
        """x: [n_query, 1, hidden_dim] running query representation.
        kv_source: [n_query, k, hidden_dim] the (fixed, same every layer)
        neighbor tokens to attend over. Returns the updated [n_query, 1,
        hidden_dim] representation."""
        q = self.norm_q(x)
        kv = self.norm_kv(kv_source)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class SimpleCrossAttentionContextEncoder(nn.Module):
    """Identical token construction to SimpleFusionContextEncoder, but the
    query attends over its k nearest context tokens via ONE standard
    pre-norm cross-attention block (LayerNorm -> multi-head cross-attention
    -> residual -> LayerNorm -> 2-layer FFN -> residual) instead of a fixed
    uniform average, STACKED n_layers times (default 2 -- one layer alone
    only lets the query look at neighbors once; a second layer lets it
    refine that using what the first layer already gathered, the same
    reason every real transformer stacks more than one block). Real
    learned attention weights, real residual connections around both
    sublayers in every layer (removing those would make it untrainable
    past one layer, see this project's own notes on why residual
    connections are structural, not optional complexity)."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_heads: int = 4,
                 mlp_ratio: float = 2.0, dropout: float = 0.1, knn_k: int = 16,
                 n_layers: int = 2, input_already_log1p: bool = True,
                 gene_encoder_type: str = "local_mlp", gene_names: list[str] | None = None,
                 gene_voc_path: str | None = None):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.hidden_dim = int(hidden_dim)
        self.knn_k = int(knn_k)
        self.input_already_log1p = bool(input_already_log1p)
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = _build_gene_encoder(gene_encoder_type, n_genes, hidden_dim, gene_names, gene_voc_path)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))

        self.layers = nn.ModuleList([
            _CrossAttnBlock(hidden_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)
        ])

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def _context_tokens(self, context_expression: torch.Tensor,
                         context_images: torch.Tensor | None) -> torch.Tensor:
        n_context = context_expression.shape[0]
        device = context_expression.device
        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            img_embed = self.image_encoder(context_images)
        else:
            img_embed = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        return img_embed + gene_embed

    def _query_own_image(self, n_query: int, query_images: torch.Tensor | None,
                          device: torch.device) -> torch.Tensor:
        if query_images is not None:
            return self.image_encoder(query_images)
        return self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]."""
        n_query = query_coords.shape[0]
        device = query_coords.device
        context_tokens = self._context_tokens(context_expression, context_images)
        neighbor_idx = _knn_indices(query_coords[:, :2], context_coords[:, :2], self.knn_k)
        neighbor_tokens = context_tokens[neighbor_idx]  # [n_query, k, hidden_dim]

        query_token = self._query_own_image(n_query, query_images, device)
        x = query_token.unsqueeze(1)  # [n_query, 1, hidden_dim]
        for layer in self.layers:
            x = layer(x, neighbor_tokens)
        return x.squeeze(1)


class SimpleFusionSpatialTransformerContextEncoder(nn.Module):
    """Our own GigaPath + gene-MLP token construction (identical to
    SimpleFusionContextEncoder/SimpleCrossAttentionContextEncoder), fed
    through STPath's REAL SpatialTransformer backbone (the same class the
    released model uses internally) instead of a hand-rolled aggregator --
    "basically STPath, but without STPath's own organ/tech tokens and
    fixed-vocabulary gene tokenizer" (2026-07-24 request). Isolates: does
    STPath's real transformer backbone help when paired with simpler,
    from-scratch encoders that don't need STPath's ~39k-gene vocabulary
    lookup or organ/tech conditioning at all.

    Construction verified directly against STPath's real source
    (stpath/model/model.py's get_backbone() and
    stpath/model/encoder/spatial_transformer.py's SpatialTransformer/
    ModelConfig, fetched 2026-07-24): SpatialTransformer takes ONE
    ModelConfig object, not individual kwargs; ModelConfig accepts
    arbitrary extra fields via **kwargs (mlp_ratio isn't one of its
    explicitly named fields but IS read by SpatialTransformer, exactly as
    STFM's own get_backbone() passes it). forward(features, coords,
    batch_idx) where coords is [N, 2] and batch_idx is one integer per
    token (all zeros here -- single sample, no cross-sample batching,
    same convention STPathContextEncoder already uses).

    First real run (2026-07-24, `stpath` package installed on the actual
    training server) surfaced a real bug the mocked-backbone test could
    never have caught: coordinates were fed to the backbone raw
    (pixel-scale, thousands), never rescaled the way STPathContextEncoder's
    real pipeline always does before calling this exact same backbone
    class. SpatialTransformer's frame-averaging attention bias assumes
    roughly unit-scale coordinates -- unrescaled, it saturates, every
    query loses real positional differentiation, and every query
    collapses to an identical output (PCC exactly 0.0 in both image
    modes -- the actual symptom that caught this). Fixed by mirroring
    STPathContextEncoder.forward's exact real coordinate handling
    (per-axis min-subtract, then STPath's own real rescale_coords)
    instead of skipping it. See forward()'s own inline comment for the
    full mechanism."""

    def __init__(self, n_genes: int, hidden_dim: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, attn_dropout: float = 0.1,
                 mlp_ratio: float = 2.0, input_already_log1p: bool = True,
                 gene_encoder_type: str = "local_mlp", gene_names: list[str] | None = None,
                 gene_voc_path: str | None = None):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        try:
            from stpath.model.encoder.spatial_transformer import SpatialTransformer
            from stpath.model.nn_utils.config import ModelConfig
            from stpath.data.dataset import rescale_coords
        except ImportError as exc:
            raise ImportError(
                "SimpleFusionSpatialTransformerContextEncoder requires the external "
                "`stpath` package (git clone Graph-and-Geometric-Learning/STPath + "
                "pip install -e .) for its SpatialTransformer backbone -- not needed "
                "for any pretrained weights or gene vocabulary here, just the backbone "
                "class itself."
            ) from exc

        self.hidden_dim = int(hidden_dim)
        self.input_already_log1p = bool(input_already_log1p)
        self._rescale_coords = rescale_coords
        self.image_encoder = GigapathPatchEncoder(feat_dim=hidden_dim)
        self.gene_encoder = _build_gene_encoder(gene_encoder_type, n_genes, hidden_dim, gene_names, gene_voc_path)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.backbone = SpatialTransformer(ModelConfig(
            n_genes=n_genes, d_input=hidden_dim, d_model=hidden_dim,
            n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            attn_dropout=attn_dropout, act="gelu", mlp_ratio=mlp_ratio,
        ))

    def _maybe_log1p(self, expr: torch.Tensor) -> torch.Tensor:
        return expr if self.input_already_log1p else torch.log1p(expr)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None,
                query_images: torch.Tensor | None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        """Returns [n_query, hidden_dim]. organ/tech accepted (matching
        every other context encoder's call signature) and unused --
        deliberately no organ/tech conditioning, per the 2026-07-24
        request this class implements."""
        n_context = context_expression.shape[0]
        n_query = query_coords.shape[0]
        n_total = n_context + n_query
        device = query_coords.device

        gene_embed = self.gene_encoder(self._maybe_log1p(context_expression))
        if context_images is not None:
            context_img = self.image_encoder(context_images)
        else:
            context_img = self.mask_token[None, :].expand(n_context, self.hidden_dim).to(device)
        context_tokens = context_img + gene_embed

        if query_images is not None:
            query_img = self.image_encoder(query_images)
        else:
            query_img = self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)
        query_tokens = query_img + self.mask_token[None, :].expand(n_query, self.hidden_dim).to(device)

        tokens = torch.cat([context_tokens, query_tokens], dim=0)  # [n_total, hidden_dim]
        # Real bug found 2026-07-24 (first real run of this class, against
        # the real stpath package -- the mocked-backbone test could never
        # have caught this): raw HEST-1k coordinates are pixel-scale
        # (thousands), and SpatialTransformer's real frame-averaging
        # attention bias assumes roughly unit-scale coordinates -- same
        # class of bug this project already hit with RandomFourierFeatures.
        # Unrescaled, the positional bias saturates/garbages, every query
        # token loses real positional differentiation, and (with no other
        # per-query signal under target_zero, where every query gets the
        # same mask_token for both image and gene) every query collapses
        # to an identical output -- exactly the "PCC exactly 0.0" symptom
        # this fix addresses. Mirrors STPathContextEncoder.forward's own
        # real pipeline exactly (stpath_encoder.py) rather than
        # approximating it: per-axis min-subtract, then STPath's own real
        # rescale_coords (min-max normalize to [0, 100]).
        coords = torch.cat([context_coords[:, :2], query_coords[:, :2]], dim=0)  # [n_total, 2]
        coords = coords.clone()
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        coords = self._rescale_coords(coords)
        batch_idx = torch.zeros(n_total, dtype=torch.long, device=device)

        fused = self.backbone(tokens, coords, batch_idx)  # [n_total, hidden_dim]
        return fused[n_context:]
