"""
Conditioning encoder — docs/architecture_plan.md layer 1, internals in
docs/model_schematics.md. Turns context (observed cells: coords +
expression) into a fixed conditioning representation per query location,
consumed identically by every generator family (WAE-GAN's decoder, FM-OT's
velocity net, VQ-VAE+AR's transformer).

Design grounded in peer-reviewed, independently-published work (not any
single unreviewed preprint):
  - Random Fourier coordinate encoding: Rahimi & Recht, "Random Features
    for Large-Scale Kernel Machines" (NeurIPS 2007); Tancik et al., "Fourier
    Features Let Networks Learn High Frequency Functions in Low Dimensional
    Domains" (NeurIPS 2020).
  - k-NN graph + attention for spatial neighborhoods in ST specifically:
    SpaGCN (Nature Methods 2021), GraphST (Nature Communications 2023),
    GAAEST (Communications Biology 2024) — all use graph-based spatial
    neighborhood encoding for ST; k is dataset-dependent in this
    literature, commonly in the 6-20 range, which is why it's a
    constructor argument here, not a fixed constant.

Deliberately ST-only for now (no H&E) — see docs/architecture_plan.md
"Known gaps". Kept modular on purpose: everything downstream only ever
consumes this module's output `c`, never its internals, so an image-encoder
branch can be fused in later (concatenated into node_repr/query_feat below)
without changing any generator model.

No torch_geometric dependency — k-NN + attention is implemented directly in
plain PyTorch (cdist/topk, same pattern as InterpolationBaseline in
registry.py), consistent with keeping the dependency footprint light.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class RandomFourierFeatures(nn.Module):
    """
    Encodes continuous coordinates into a higher-dimensional feature basis
    (Rahimi & Recht 2007; Tancik et al. 2020 — module docstring above).
    Fixed (non-trainable) random projection, so `sigma` is the one
    hyperparameter that matters — controls the encoding's spatial frequency.
    """

    def __init__(self, in_dim: int, num_features: int = 64, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, num_features) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _knn_indices(queries: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """[N_queries, k] indices into `keys` of each query's k nearest neighbours."""
    dists = torch.cdist(queries, keys)
    k = min(k, keys.shape[0])
    _, idx = torch.topk(dists, k=k, largest=False, dim=-1)
    return idx


class _KNNMessageLayer(nn.Module):
    """One round of message passing: each context node attends to its own
    k nearest neighbours. Stacking a few of these lets information from a
    node's local neighbourhood propagate a couple of hops further."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_repr: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        neighbor_repr = node_repr[knn_idx]           # [N, k, hidden_dim]
        query = node_repr.unsqueeze(1)                 # [N, 1, hidden_dim]
        attn_out, _ = self.attn(query, neighbor_repr, neighbor_repr)
        node_repr = self.norm1(node_repr + attn_out.squeeze(1))
        node_repr = self.norm2(node_repr + self.ff(node_repr))
        return node_repr


class ImagePatchEncoder(nn.Module):
    """
    H&E patch [B, 3, patch_size, patch_size] -> feature vector [B, feat_dim].
    Task #17 — the "own, from-scratch" H&E branch, deliberately a small
    plain CNN (own weights, jointly trained with the rest of the pipeline,
    same per-model-weights reasoning used throughout this codebase), NOT a
    pretrained pathology foundation model — that's task #18 (STPath), kept
    as a separate arm precisely so the two-way ablation (does adding image
    info help at all vs. does a *strong pretrained* image encoder help
    more) stays clean. See src/data/loaders.py load_hest_patches() for
    where the raw 224x224 uint8 patches come from (HEST-1k's own native
    size, not 256 — verified 2026-07-15 against a real downloaded file).

    `patch_size` (constructor param below) is accepted but never actually
    used inside this class — AdaptiveAvgPool2d(1) makes the conv stack
    agnostic to input spatial size, so there's nothing here to resize.
    The REAL resolution fed into this encoder is controlled by
    `_downsample_patches()` in src/training/train.py, applied ONCE at
    data-loading time (not per training step) — a real speed fix
    (2026-07-15, user question: "why are CNN configs so slow"): every
    training step was converting+transferring+convolving the full native
    224x224 patches (up to ~700-900 per masking draw) regardless of any
    config value, since nothing upstream of this class was actually
    resizing anything before that fix.
    """

    def __init__(self, patch_size: int = 256, feat_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),   # ->128
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # ->64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # ->32
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, feat_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, 3, H, W], expected already float in [0, 1]
        h = self.conv(patches).flatten(1)
        return self.proj(h)


def _load_gigapath_tile_encoder():
    """Shared loader for Prov-GigaPath's tile encoder — used by both
    GigapathPatchEncoder below (task #20) and
    src/models/stpath_encoder.py's STPathContextEncoder (task #18, which
    needs the same raw 1536-dim features, without a trainable projection
    on top). Frozen (eval mode, no gradient) — the RAE idea (Zheng et al.
    2025) applied here: reuse a strong pretrained representation as-is.

    Requires, neither a default dependency of this repo:
      1. `pip install timm` (>=1.0.3 per the model's own README) — not in
         environment.yml/requirements.txt, since it's only needed if
         Gigapath is actually used.
      2. A HuggingFace account with access GRANTED to the gated
         prov-gigapath/prov-gigapath repo (research-use-only license —
         request at huggingface.co/prov-gigapath/prov-gigapath) and
         `huggingface-cli login` run once, or `HF_TOKEN` set in the
         environment. Not something this code can obtain on your behalf.
    """
    import timm
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    tile_encoder.eval()
    for p in tile_encoder.parameters():
        p.requires_grad_(False)
    return tile_encoder


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Prov-GigaPath's real tile-encoder output dim, confirmed via a real
# forward pass (GigapathPatchEncoder's original construction-time probe)
# and matching STPath's own hardcoded assumption
# (stpath_encoder.py ImageTokenizer(feature_dim=1536)) — the same real
# model, so the same real number. Kept as a named constant rather than
# re-probed via a real forward pass at every GigapathPatchEncoder
# construction (see that class for why: probing required loading
# Gigapath from HuggingFace even when features are fully precomputed/
# cached and the real model is never actually needed again — a real
# blocking issue, 2026-07-15, HuggingFace HEAD-request timeouts on a
# flaky network, even though the weights were already cached locally
# from an earlier successful download).
_GIGAPATH_FEAT_DIM = 1536


def _gigapath_preprocess_and_encode(tile_encoder, patches: torch.Tensor) -> torch.Tensor:
    """Prov-GigaPath's own documented preprocessing (resize 256 -> center
    crop 224 -> ImageNet normalize, its GitHub README) + a forward pass
    through the frozen tile encoder. patches: [B, 3, H, W] float in
    [0, 1]. Shared by GigapathPatchEncoder, STPathContextEncoder, and
    precompute_gigapath_features below so the preprocessing logic exists
    in exactly one place."""
    device = patches.device
    # bicubic interpolate isn't implemented on MPS (Apple Silicon) as of
    # this writing - do this one op on CPU rather than switch to a mode
    # MPS does support, to stay faithful to the documented preprocessing
    x = nn.functional.interpolate(
        patches.cpu(), size=256, mode="bicubic", align_corners=False
    ).to(device)
    top = (256 - 224) // 2
    x = x[:, :, top:top + 224, top:top + 224]
    x = (x - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
    with torch.no_grad():
        return tile_encoder(x)


def _default_device() -> str:
    """cuda > mps > cpu. Only used by precompute_gigapath_features below —
    every nn.Module in this codebase instead relies on Lightning's Trainer
    moving the whole model to the right device automatically, so encoder
    classes deliberately don't do their own device auto-detection (that
    would risk a mismatch if Lightning later picks a different device)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def precompute_gigapath_features(images, batch_size: int = 16, device: str | None = None):
    """Run Gigapath's frozen tile encoder ONCE over a whole dataset's spot
    patches and cache the result, instead of recomputing it from raw
    pixels on every training step.

    Real bug found 2026-07-15: the original design ran GigapathPatchEncoder
    (and STPathContextEncoder, which uses Gigapath internally) on raw
    patches inside forward(), meaning every single training step re-ran a
    giant ViT over ~1000 context images from scratch — a real training run
    on real INT1 data with STPath conditioning was still stuck after the
    first step. Gigapath's output for a given image never changes (it's
    frozen, eval mode, no gradient), so this is pure waste — precompute
    once here, then MaskedContextQueryDataset just slices/masks the cached
    [N, gigapath_dim] array per draw like any other per-spot feature,
    exactly like it already does for coords/expression.

    This function runs standalone, before any Lightning Trainer exists
    (called from _load_data, not from inside a model), so — unlike every
    nn.Module in this codebase — it has to pick its own device rather than
    rely on Lightning's automatic placement. Defaults to the best
    available (cuda > mps > cpu): confirmed 2026-07-15 that the original
    CPU-only version was the actual bottleneck on Apple Silicon (a
    ~1.1B-parameter ViT running on CPU for ~1000 images).

    images: [N, H, W, 3] uint8. Returns [N, gigapath_dim] float32 numpy
    array (gigapath_dim is read from a real forward pass, not hardcoded —
    same reasoning as GigapathPatchEncoder below)."""
    import numpy as np
    if device is None:
        device = _default_device()
    tile_encoder = _load_gigapath_tile_encoder().to(device)
    n = images.shape[0]
    all_feats = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = images[start:start + batch_size]
            patches_t = torch.tensor(chunk, dtype=torch.float32).permute(0, 3, 1, 2).to(device) / 255.0
            feats = _gigapath_preprocess_and_encode(tile_encoder, patches_t)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


class GigapathPatchEncoder(nn.Module):
    """
    Wraps Prov-GigaPath's tile encoder (Xu et al. 2024, Nature, "A
    whole-slide foundation model for digital pathology from real-world
    data") as an alternative to ImagePatchEncoder above — task #20, "does
    a strong PRETRAINED image encoder help", kept separate from STPath
    (task #18, which uses Gigapath internally but adds its own multi-modal
    fusion): this class is Gigapath ALONE, feeding straight into this same
    module's own k-NN/attention fusion instead of STPath's.

    Only the small trainable projection head is unique to this class — the
    frozen tile encoder loading + preprocessing are shared (see
    _load_gigapath_tile_encoder / _gigapath_preprocess_and_encode above).

    forward() accepts EITHER raw patches [B, 3, H, W] float in [0, 1]
    (encodes them from scratch — the uncached, slow path, useful for
    smoke tests / one-off calls) OR already-precomputed Gigapath features
    [B, gigapath_dim] (the fast, cached path real training should use —
    see precompute_gigapath_features above) — dispatches on tensor rank
    so callers don't need two different method names. Output feature dim
    uses the module-level _GIGAPATH_FEAT_DIM constant rather than probing
    it via a real forward pass at construction time (an earlier version
    did that, unconditionally, on every construction — real problem hit
    2026-07-15: it required loading the actual Gigapath model from
    HuggingFace even when features were fully precomputed/cached and the
    real model was never going to be used again, which hung/timed out on
    a flaky network despite the weights already being cached locally from
    an earlier successful download). The tile encoder is now ONLY ever
    loaded (and _GIGAPATH_FEAT_DIM cross-checked against its real output,
    _ensure_tile_encoder below) if raw patches genuinely show up — never
    in the real training path, which always uses precomputed features.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        # Gigapath's own activation scale was optimized for its own
        # training objective, not for whatever an untrained nn.Linear
        # here expects — normalize before projecting rather than relying
        # on `proj` to learn a rescaling from scratch on top of learning
        # everything else. Same reasoning as
        # stpath_encoder.py's STPathContextEncoder.embedding_norm
        # (2026-07-15), added here too for consistency since both classes
        # are the same "frozen big model -> small trainable head" (RAE)
        # pattern.
        self.embedding_norm = nn.LayerNorm(_GIGAPATH_FEAT_DIM)
        self.proj = nn.Linear(_GIGAPATH_FEAT_DIM, feat_dim)
        # Not loaded here at all (see class docstring) — real training
        # never touches this. Loaded lazily only if raw patches genuinely
        # show up (smoke tests / one-off calls) — same pattern as
        # stpath_encoder.py's STPathContextEncoder._ensure_tile_encoder,
        # both added 2026-07-15 after a real RAM crash running several
        # Gigapath/STPath-backed configs back to back
        # (src/evaluation/run_comparison.py _free()).
        self.tile_encoder = None

    def _ensure_tile_encoder(self, device: torch.device) -> nn.Module:
        if self.tile_encoder is None:
            tile_encoder = _load_gigapath_tile_encoder().to(device)
            with torch.no_grad():
                real_dim = tile_encoder(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
            assert real_dim == _GIGAPATH_FEAT_DIM, (
                f"Gigapath's real tile-encoder output dim ({real_dim}) doesn't match "
                f"the hardcoded _GIGAPATH_FEAT_DIM ({_GIGAPATH_FEAT_DIM}) — update the constant"
            )
            self.tile_encoder = tile_encoder
        return self.tile_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # raw patches - encode from scratch (uncached path)
            tile_encoder = self._ensure_tile_encoder(x.device)
            x = _gigapath_preprocess_and_encode(tile_encoder, x)
        return self.proj(self.embedding_norm(x))


class SpatialContextEncoder(nn.Module):
    """
    context (coords [N_obs, D], expression [N_obs, G]) + query coords
    [N_query, D]  ->  conditioning vector c [N_query, hidden_dim]

    D is 2 for intra-slice (Track A) or 3 for inter-slice (Track B) — same
    module handles both, since it only ever sees generic coordinates.

    image_encoder_type="none" (default) is the ORIGINAL gene-expression-only
    path, completely unchanged — task #17 explicitly keeps this variant
    working as an ablation baseline, not a replacement.
    image_encoder_type="cnn" uses ImagePatchEncoder (task #17, our own
    from-scratch encoder); "gigapath" uses GigapathPatchEncoder (task #20,
    a pretrained encoder) — same fusion code either way, so switching
    between them (or off) for a benchmark is a one-argument change, not a
    rewrite. Whenever images are enabled, context_images/query_images
    become required forward() args and get fused into node_repr/query_feat
    via concatenation, exactly as this module's original docstring
    promised ("designed so an image-encoder branch can be fused in
    later... without changing any generator model").
    """

    def __init__(self, n_genes: int, coord_dim: int = 3, hidden_dim: int = 256,
                 n_message_layers: int = 2, k_neighbors: int = 10,
                 rff_features: int = 64, rff_sigma: float = 1.0,
                 image_encoder_type: str = "none", image_feat_dim: int = 64,
                 image_patch_size: int = 256):
        super().__init__()
        assert image_encoder_type in ("none", "cnn", "gigapath"), (
            f"unknown image_encoder_type {image_encoder_type!r}"
        )
        self.k_neighbors = k_neighbors
        self.use_images = image_encoder_type != "none"
        self.coord_encoder = RandomFourierFeatures(coord_dim, rff_features, rff_sigma)
        coord_feat_dim = 2 * rff_features  # sin + cos

        node_in_dim = n_genes + coord_feat_dim
        query_in_dim = coord_feat_dim
        if image_encoder_type == "cnn":
            self.image_encoder = ImagePatchEncoder(image_patch_size, image_feat_dim)
            node_in_dim += image_feat_dim
            query_in_dim += image_feat_dim
        elif image_encoder_type == "gigapath":
            self.image_encoder = GigapathPatchEncoder(image_feat_dim)
            node_in_dim += image_feat_dim
            query_in_dim += image_feat_dim

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            _KNNMessageLayer(hidden_dim) for _ in range(n_message_layers)
        )
        self.query_proj = nn.Linear(query_in_dim, hidden_dim)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None = None,
                query_images: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_images and (context_images is None or query_images is None):
            raise ValueError("use_images=True requires context_images and query_images")

        # 1. embed each context node from its own expression + coordinate (+ image)
        context_coord_feat = self.coord_encoder(context_coords)
        node_feats = [context_expression, context_coord_feat]
        if self.use_images:
            node_feats.append(self.image_encoder(context_images))
        node_repr = self.node_proj(torch.cat(node_feats, dim=-1))

        # 2. message-pass over the context's own k-NN graph
        context_knn = _knn_indices(context_coords, context_coords, self.k_neighbors)
        for layer in self.message_layers:
            node_repr = layer(node_repr, context_knn)

        # 3. for each query location, attend over its k nearest context nodes
        query_knn = _knn_indices(query_coords, context_coords, self.k_neighbors)
        query_feats = [self.coord_encoder(query_coords)]
        if self.use_images:
            query_feats.append(self.image_encoder(query_images))
        query_feat = self.query_proj(torch.cat(query_feats, dim=-1))
        neighbor_repr = node_repr[query_knn]                     # [N_query, k, hidden_dim]
        c, _ = self.query_attn(query_feat.unsqueeze(1), neighbor_repr, neighbor_repr)
        return c.squeeze(1)                                        # [N_query, hidden_dim]