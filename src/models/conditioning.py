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
    hyperparameter that matters — controls the encoding's spatial frequency,
    IN WHATEVER UNITS THE INPUT COORDINATES ALREADY ARE.

    REAL BUG found 2026-07-17 (same class of bug as RelativePositionBias's
    2026-07-17 fix — see that class's docstring for the first instance):
    sigma=1.0 (the untouched default in every config in this project) only
    makes sense if coordinates are already roughly unit-scale. Real
    HEST-1k coordinates are pixel-scale (thousands — see this project's
    masking configs' own radius_range, e.g. [250, 450]). Checked directly:
    two spots 5 pixel-units apart (essentially the same location) got
    cosine similarity 0.001 between their encodings — indistinguishable
    from spots 2000 units apart (-0.093) — i.e. sin(2*pi*x@B) for
    thousands-scale x wraps around (aliases) so many times that the
    encoding is essentially RANDOM NOISE with respect to real spatial
    locality, not a smooth positional signal at all. Every config using
    "builtin" (SpatialContextEncoder) or "storm_lite"
    (StormLiteContextEncoder) context encoders was silently getting a
    near-useless absolute-position signal on real data (StormLite's
    RelativePositionBias fix, above, still leaves this SEPARATE absolute-
    position pathway broken).

    Fixed via coord_scale: divides x by this BEFORE the sigma-scaled
    projection (mathematically equivalent to sigma/coord_scale, kept as a
    separate constructor arg since sigma still meaningfully controls
    relative frequency in the now-normalized space, while coord_scale is
    purely "what are this dataset's real units"). Deliberately a FIXED
    value set at construction time, not a per-call auto-normalization
    (the fix RelativePositionBias uses) — SpatialContextEncoder calls this
    module SEPARATELY for context_coords and query_coords (two different
    point sets, potentially different extents); per-call normalization
    would give the SAME real position a DIFFERENT encoding depending on
    which call computed it, which is a worse bug than the one being fixed.
    coord_scale=1.0 (default) preserves the exact original (buggy at real
    scale, but correct at small-scale/synthetic-test scale) behavior for
    every existing caller that doesn't explicitly opt in — auto-derived
    from real per-sample coordinate spread by inject_coord_scale in
    src/training/train.py, the same "must be derived from real data, not
    hardcoded" pattern as inject_novae_dim.
    """

    def __init__(self, in_dim: int, num_features: int = 64, sigma: float = 1.0,
                 coord_scale: float = 1.0):
        super().__init__()
        self.coord_scale = coord_scale
        self.register_buffer("B", torch.randn(in_dim, num_features) * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x / self.coord_scale) @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def _knn_indices(queries: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """[N_queries, k] indices into `keys` of each query's k nearest neighbours."""
    dists = torch.cdist(queries, keys)
    k = min(k, keys.shape[0])
    _, idx = torch.topk(dists, k=k, largest=False, dim=-1)
    return idx


class RelativePositionBias(nn.Module):
    """Continuous relative-position attention bias (Liu et al., CVPR 2022,
    "Swin Transformer V2: Scaling Up Capacity and Resolution" — its CPB,
    "continuous position bias", is a small MLP over relative coordinates
    producing an additive attention-score bias) — added 2026-07-16 for
    StormLiteContextEncoder, which previously only had ABSOLUTE position
    information (RandomFourierFeatures concatenated into each token). An
    absolute encoding lets the model infer *something* about geometry
    indirectly through attention's learned Q/K projections, but it has no
    direct signal for "these two spots are close/far" the way STPath's
    own verified geometry-aware frame-averaging attention bias does
    (stpath_encoder.py) — this closes that specific gap for our own
    from-scratch fusion transformer.

    Adapted from Swin V2's 2D windowed-vision setting to arbitrary 3D
    point-cloud coordinates (not a regular grid): for every pair of
    tokens, the MLP takes (dx, dy, dz, euclidean_distance) and outputs a
    single scalar bias, added directly to that pair's raw attention
    logit before softmax (via nn.TransformerEncoder's own `mask` — a
    FloatTensor mask is documented PyTorch behavior for an ADDITIVE bias,
    not a boolean mask — no custom attention implementation needed).

    Simplification flagged explicitly: one shared bias broadcast across
    all attention heads and the batch dimension, not Swin V2's per-head
    bias — this project's models run with batch_size=1 by construction
    (one masking draw at a time, see MaskedContextQueryDataset's own
    docstring), and per-head biases would need one MLP output per head
    rather than a scalar; kept simple for a first version, easy to
    extend later if a per-head bias turns out to matter.

    REAL BUG found 2026-07-17 (this class was directly responsible for
    StormLiteContextEncoder scoring WORSE than a plain k-NN interpolation
    baseline on real HEST-1k data — PCC -0.005 to -0.015, ST-FID 22-36
    vs. interp_baseline's 10.09 — across every gene_encoder_type variant,
    while every STPath arm, unaffected by this class, trained fine): the
    original forward() fed RAW, unnormalized coordinate differences
    directly into self.mlp. Every smoke test in this project used
    synthetic coordinates in [0, 100) (small enough to hide the problem),
    but real HEST-1k spatial coordinates are pixel-scale — see this
    project's masking configs' own radius_range (e.g. [250, 450]),
    implying the full coordinate range is in the thousands. Feeding
    values that large through two nn.Linear layers with standard
    (small-weight) init produces enormous pre-activation magnitudes,
    added directly to raw attention logits BEFORE softmax — completely
    swamping the actual learnable Q/K attention signal with an
    effectively-random, poorly-conditioned bias term the model has no
    way to learn its way out of (the bias itself keeps growing during
    training since its own gradient is dominated by this same scale
    mismatch). Fixed by normalizing diff/dist by this forward call's OWN
    max pairwise distance before the MLP sees them — keeps every feature
    within roughly [-1, 1] regardless of whether the input coordinates
    are in pixels, microns, or the small-scale values every test in this
    file happens to use, so the bias magnitude is governed entirely by
    the MLP's own (trainable, well-scaled) output layer instead of by
    whatever units the caller's coordinates happen to be in."""

    def __init__(self, coord_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [N, coord_dim]. Returns [N, N] additive attention bias
        (bias[i, j] = how much token i's attention to token j should be
        adjusted based on their relative position)."""
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, coord_dim]
        dist = diff.norm(dim=-1, keepdim=True)             # [N, N, 1]
        # per-call scale normalization (2026-07-17 fix, see class
        # docstring) — makes this module invariant to the raw coordinate
        # unit/magnitude instead of assuming small values
        scale = dist.max().clamp(min=1e-6)
        feat = torch.cat([diff / scale, dist / scale], dim=-1)  # [N, N, coord_dim+1], roughly in [-1, 1]
        return self.mlp(feat).squeeze(-1)                    # [N, N]


class FrameAveragingBias(nn.Module):
    """Real, VERIFIED relative-position attention bias, reproducing
    STPath's OWN actual mechanism — added 2026-07-17 after directly
    cloning and reading github.com/Graph-and-Geometric-Learning/STPath
    (stpath/model/nn_utils/fa.py's FrameAveraging class +
    stpath/model/encoder/spatial_transformer.py's Attention class),
    resolving what was previously only described secondhand in this
    project's own comments. Supersedes RelativePositionBias (above,
    2026-07-16) as StormLiteContextEncoder's default bias mechanism —
    that class was an ad hoc Swin-V2-style CPB MLP, invented because
    STORM's own real mechanism was unverifiable (arXiv 2604.03630 blocked
    by this sandbox's network proxy, confirmed repeatedly). STPath's real
    mechanism turned out to be independently accessible and is used here
    instead, on the reasoning that reproducing a VERIFIED real
    architecture beats reproducing an invented one when direct evidence
    is available. RelativePositionBias is kept (not deleted) as a
    still-usable, still-tested alternative.

    Frame averaging (Puny et al. 2022, "Frame Averaging for Invariant and
    Equivariant Network Design") is a technique for exact invariance to a
    symmetry group by averaging a function's output over a small, FINITE
    "frame" of group elements, rather than the whole continuous group
    (which would need e.g. numerical integration over all rotation
    angles). STPath applies it to build a relative-position attention
    bias that is PROVABLY invariant to any rotation/reflection of the
    whole coordinate system — a strictly stronger geometric guarantee
    than RelativePositionBias's plain MLP-over-raw-offset, which has no
    such property (rotating the same tissue 90 degrees would, in
    general, change RelativePositionBias's output, but cannot change
    this class's, by construction — verified numerically in this
    project's own tests via a real random rotation+reflection+translation
    applied to the input coordinates).

    Mechanism, per this forward call's whole point set (STPath's own
    version computes this PER QUERY ROW's own N neighbor offsets — same
    idea, done here in one batched pass over every row at once, which
    this project's small point clouds make cheap): for query row i, its N
    relative offsets to every other point are centered, their covariance
    matrix eigendecomposed for a principal-axis basis (eigenvectors),
    then combined with all 2^dim=4 sign-flip operations
    ((-1,-1),(-1,1),(1,-1),(1,1)) to produce 4 canonical reorientations of
    row i's offsets. edge_bias (a small Linear, one output per attention
    head — this class produces a genuinely PER-HEAD bias, unlike
    RelativePositionBias's single shared scalar, since
    nn.TransformerEncoder's `mask` accepts a [n_heads, N, N] tensor
    exactly as needed for batch_size=1) is applied to each of the 4
    reoriented-offset-plus-norm feature vectors, then averaged over the 4
    frames — averaging over a group's frame is exactly what makes the
    result provably invariant to that group's action on the input.

    dim=2 (not this project's usual coord_dim=3): matches STPath's own
    real Attention class, `super(Attention, self).__init__(dim=2)` — only
    the xy plane, verified directly in its source. STPath itself only
    ever applies this bias to 2D coordinates even though 3D positions
    exist elsewhere in its pipeline; this class does the same (coords[:,
    :2] only), regardless of what coord_dim the caller's coordinates
    actually carry.

    coord_scale (2026-07-17, NOT part of STPath's own real code — see
    RandomFourierFeatures' own docstring for the coordinate-scale bug
    this avoids REINTRODUCING): STPath's real edge_bias weights were
    trained end-to-end on STPath's own real pretraining data's coordinate
    convention. This class is a FRESH, randomly-initialized copy trained
    on OUR data instead, so reusing STPath's implicit raw-coordinate
    convention here would just reintroduce the identical exploding-bias
    failure this project already found and fixed once (see
    RelativePositionBias's own 2026-07-17 docstring) — radial offsets and
    their norm are divided by coord_scale before edge_bias sees them,
    same fixed-at-construction-time approach as RandomFourierFeatures
    (this class is also called ONCE on the full concatenated coord set in
    StormLiteContextEncoder, the same safe usage pattern)."""

    def __init__(self, n_heads: int, coord_scale: float = 1.0):
        super().__init__()
        self.dim = 2
        self.n_frames = 2 ** self.dim  # 4
        self.coord_scale = coord_scale
        # the 4 fixed sign-flip combinations — not learned, matches
        # STPath's own real create_ops (verified via its source)
        self.register_buffer("ops", torch.tensor(
            [[sx, sy] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        ))  # [4, 2]
        self.edge_bias = nn.Linear(self.dim + 1, n_heads, bias=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: [N, coord_dim] (only the first 2 columns are used — see
        class docstring). Returns [n_heads, N, N] additive per-head
        attention bias, directly usable as nn.TransformerEncoder's `mask`
        argument for a batch_size=1 model."""
        coords = coords[:, :2] / self.coord_scale
        n = coords.shape[0]
        radial = coords.unsqueeze(1) - coords.unsqueeze(0)          # [N, N, 2] radial[i,j] = coords[i]-coords[j]
        radial_norm = radial.norm(dim=-1, keepdim=True)              # [N, N, 1]

        # per-query-row (i) frame: center row i's N offsets, eigendecompose
        # their covariance for a principal-axis basis, combine with the 4
        # sign-flip ops (see class docstring / STPath's real create_frame)
        center = radial.mean(dim=1, keepdim=True)                    # [N, 1, 2]
        centered = radial - center                                    # [N, N, 2]
        cov = torch.einsum("nmi,nmj->nij", centered, centered) / n   # [N, 2, 2]
        _, eigvecs = torch.linalg.eigh(cov)                           # eigvecs: [N, 2, 2]

        f_ops = self.ops.view(1, self.n_frames, 1, self.dim) * eigvecs.unsqueeze(1)  # [N, 4, 2, 2]
        # reorient row i's offsets into each of its 4 canonical frames
        frame_feats = torch.einsum("nojk,nmk->nomj", f_ops, radial)  # [N, 4, N, 2]

        radial_norm_exp = radial_norm.unsqueeze(1).expand(n, self.n_frames, n, 1)  # [N, 4, N, 1]
        feat = torch.cat([frame_feats, radial_norm_exp], dim=-1)     # [N, 4, N, 3]
        bias = self.edge_bias(feat).mean(dim=1)                       # [N, N, n_heads] — frame AVERAGING
        return bias.permute(2, 0, 1)                                   # [n_heads, N, N]


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


class MLPGeneEncoder(nn.Module):
    """
    Nonlinear replacement for feeding the raw n_genes expression vector
    straight into node_proj/query_proj. Motivated by the same critique the
    user raised about STPath's gene-expression branch (a single nn.Linear)
    — worth noting our OWN SpatialContextEncoder has the identical
    weakness: today, raw expression is concatenated directly into
    node_proj's input, so the "gene encoder" is effectively also just one
    Linear layer (node_proj's first n_genes input columns), with no
    dedicated nonlinear processing before fusion with coords/image
    features. This class gives gene expression the same kind of dedicated
    encoder image features already get (ImagePatchEncoder/
    GigapathPatchEncoder), instead of being the only modality fused in raw.

    The original implementation hard-coded a 2048-unit first layer. On a
    16,570-gene panel that single matrix contains about 34 million weights,
    making a supposedly lightweight control unnecessarily large and difficult
    to compare fairly with other encoders. The safer default below is a
    configurable 512 -> 256 bottleneck. Callers that intentionally need the
    historical capacity can still pass ``hidden_dim=2048,
    bottleneck_dim=512`` explicitly.

    This remains an encoder-only module trained by the downstream objective.
    The residual-flow audit suite separately pretrains a complete expression
    autoencoder and validates it before freezing it.
    """

    def __init__(self, n_genes: int, feat_dim: int = 256,
                 hidden_dim: int = 512, bottleneck_dim: int = 256):
        super().__init__()
        if min(n_genes, feat_dim, hidden_dim, bottleneck_dim) <= 0:
            raise ValueError("MLPGeneEncoder dimensions must all be positive")
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim), nn.LayerNorm(bottleneck_dim), nn.GELU(),
            nn.Linear(bottleneck_dim, feat_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.net(expression)


def precompute_novae_features(adata, checkpoint: str = "prism-oncology/novae-human-0"):
    """Run pretrained Novae (Novae et al. 2025, Nature Methods — graph-based
    ST foundation model, github.com/MICS-Lab/novae) ONCE over a whole
    sample's spots and cache the result, mirroring
    precompute_gigapath_features above — same reasoning: Novae's own
    published usage pattern is `model.compute_representations(adata,
    zero_shot=True)`, an inference-only call (no documented gradient/
    fine-tuning story — checked 2026-07-16, its docs site returned 403 so
    this is based on the GitHub README's quickstart example, not fully
    verified; re-check yourself if `novae`'s API has since added one).

    Real structural difference from Gigapath/STPath's per-spot encoders:
    Novae's representations depend on the sample's SPATIAL NEIGHBOR GRAPH
    (`novae.spatial_neighbors(adata)`, called here), not just a single
    spot's own data in isolation — so unlike an image patch, a Novae
    embedding cannot even in principle be computed from one row at a time
    inside a model's forward(). Precomputation here is therefore not just
    a speed optimization (as it is for Gigapath) — it's structurally
    required by what the model actually needs as input.

    UNVERIFIED (docs site 403'd when checked): the exact `adata.obsm` key
    `compute_representations` writes to. Defensively diffs obsm keys
    before/after the call rather than hardcoding a guessed name, and
    raises with the real available keys if that doesn't yield exactly one
    new key — verify/hardcode the real key yourself once you can run this
    against the actual package.

    adata: real AnnData with spatial coords in .obsm (novae.spatial_neighbors
    default) and expression in .X. Returns [N, novae_dim] float32 numpy array.
    """
    import novae

    adata = adata.copy()  # spatial_neighbors/compute_representations mutate in place
    keys_before = set(adata.obsm.keys())
    novae.spatial_neighbors(adata)
    model = novae.Novae.from_pretrained(checkpoint)
    model.compute_representations(adata, zero_shot=True)
    new_keys = [k for k in adata.obsm.keys() if k not in keys_before]
    novae_keys = [k for k in new_keys if "novae" in k.lower()] or new_keys
    if len(novae_keys) != 1:
        raise RuntimeError(
            f"Couldn't identify Novae's output obsm key unambiguously — "
            f"new obsm keys after compute_representations: {new_keys!r} "
            f"(all obsm keys: {list(adata.obsm.keys())!r}). Inspect the "
            f"real novae package version installed and hardcode the "
            f"correct key in precompute_novae_features."
        )
    return adata.obsm[novae_keys[0]].astype("float32")


class NovaeGeneEncoder(nn.Module):
    """
    Wraps precomputed Novae embeddings (precompute_novae_features above) as
    an alternative to raw expression / MLPGeneEncoder — frozen pretrained
    "does a strong PRETRAINED, spatially-aware gene-expression encoder
    help" arm, same RAE pattern (Zheng et al. 2025) already used for
    GigapathPatchEncoder/STPathContextEncoder: frozen big representation +
    small trainable LayerNorm+Linear head.

    UNLIKE GigapathPatchEncoder, this does NOT accept raw expression as a
    fallback uncached path — Novae's representations depend on the whole
    sample's spatial neighbor graph (see precompute_novae_features), so
    there is no meaningful "encode this one row from scratch" operation to
    fall back to. forward() therefore always expects already-precomputed
    [B, novae_dim] features; novae_dim is read from the real precomputed
    array's shape at construction (never hardcoded/guessed — same
    "probe, don't assume" reasoning as GigapathPatchEncoder's
    _GIGAPATH_FEAT_DIM, except here there's no live model to probe against
    inside this class, so the caller must supply it directly).
    """

    def __init__(self, novae_dim: int, feat_dim: int = 64):
        super().__init__()
        self.embedding_norm = nn.LayerNorm(novae_dim)
        self.proj = nn.Linear(novae_dim, feat_dim)

    def forward(self, novae_features: torch.Tensor) -> torch.Tensor:
        if novae_features.dim() != 2:
            raise ValueError(
                f"NovaeGeneEncoder expects precomputed [B, novae_dim] features, "
                f"got shape {tuple(novae_features.shape)} — see precompute_novae_features()"
            )
        return self.proj(self.embedding_norm(novae_features))


class CombinedGeneEncoder(nn.Module):
    """Combines MLPGeneEncoder(raw_expr) + NovaeGeneEncoder(novae_features),
    both already projecting to feat_dim — added 2026-07-16 after the real
    STPath-residual comparison (task #19-followup "Route B") showed MLP
    residual beating both the STPath baseline AND the Novae residual on
    5/6 metrics, while Novae residual alone was the only arm to improve
    cell-type plausibility — motivating a combined arm to see whether
    MLP's pointwise-accuracy gain and Novae's plausibility gain are
    complementary or the same underlying signal.

    combine_mode="sum" (default): straight elementwise sum, output stays
    feat_dim — no extra combiner layer, cheapest option. Used by
    StormLiteContextEncoder's additive token fusion (matches STPath's own
    verified img_embed + ge_embed + ... pattern there).

    combine_mode="concat": output is 2*feat_dim (both sub-encoders' raw
    outputs kept separate, not pre-mixed) — REQUIRED when the caller
    itself applies a further learned linear layer on the combined output
    and wants that layer able to weight each source independently.
    Real bug this fixes (2026-07-16): this class's own earlier docstring
    claimed STPathContextEncoder's zero-init residual_proj (a Linear
    applied AFTER this class's output) "can learn to weight the two
    contributions differently if a plain sum isn't optimal" — false. Once
    two vectors are summed into one, a linear layer applied to the SUM
    cannot recover or separately reweight what went into it (a Linear
    layer applied to (a+b) is mathematically NOT equivalent in general to
    independently-weighted a and b — that would require seeing a and b
    as separate inputs). Confirmed by a real run: the combined ("both",
    sum-mode) residual was WORSE than MLP alone on PCC/RMSE/AUC despite
    Novae residual alone being fine on those axes — consistent with the
    optimizer being forced into a single compromise weighting of the
    pre-mixed sum rather than freely calibrating each source. Use
    combine_mode="concat" wherever the caller needs genuine independent
    weighting; sum is still fine (and cheaper) wherever the caller has no
    further learned layer to exploit the distinction, e.g. StormLite's
    additive token fusion above."""

    def __init__(self, n_genes: int, novae_dim: int, feat_dim: int, combine_mode: str = "sum"):
        super().__init__()
        assert combine_mode in ("sum", "concat"), f"unknown combine_mode {combine_mode!r}"
        self.combine_mode = combine_mode
        self.mlp = MLPGeneEncoder(n_genes, feat_dim)
        self.novae = NovaeGeneEncoder(novae_dim, feat_dim)

    @property
    def output_dim_multiplier(self) -> int:
        """1 for "sum" (output is feat_dim), 2 for "concat" (output is
        2*feat_dim) — callers that need to size a downstream layer (e.g.
        STPathContextEncoder's residual_proj) read this rather than
        hardcoding the multiplier themselves."""
        return 2 if self.combine_mode == "concat" else 1

    def forward(self, raw_expr: torch.Tensor, novae_features: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(raw_expr)
        novae_out = self.novae(novae_features)
        if self.combine_mode == "concat":
            return torch.cat([mlp_out, novae_out], dim=-1)
        return mlp_out + novae_out


class TokenizedGeneEncoder(nn.Module):
    """Per-gene identity-aware tokenization + pooling (2026-07-19 research),
    replacing MLPGeneEncoder's single dense-compression bottleneck with
    STPath's real design principle -- its GeneExpTokenizer represents each
    gene as its own identity-aware token (verified via source, see
    stpath_encoder.py's own docstring), rather than collapsing the whole
    expression vector through a shared MLP/pretrained-whole-profile
    embedding the way MLPGeneEncoder/NovaeGeneEncoder both do. Ported as
    our OWN encoder (same "faithful to the design principle, not every
    implementation detail" approach already used for MoME-FFN and
    FrameAveragingBias) rather than porting STPath's tokenizer/binning
    machinery wholesale.

    Mechanism: a learned per-gene IDENTITY embedding (fixed vocabulary,
    same "must be fixed at construction time" reasoning as
    OrganTechEmbedding/inject_stpath_gene_names) ADDED (scGPT's real
    combination rule, Cui et al. 2024 -- already this codebase's own
    established choice for PanelInvariantGeneDecoder's combine_mode="add")
    to a linear projection of that gene's own expression VALUE, forming
    one token per selected gene. A small self-attention layer then lets
    genes interact/contextualize each other (closer to genuine per-gene
    tokenization than plain mean pooling would be) before mean-pooling
    into ONE final vector per spot -- keeping the exact same [N, feat_dim]
    output contract as MLPGeneEncoder/NovaeGeneEncoder, so no downstream
    fusion code needs to change regardless of which gene encoder is active.

    gene_names is the SELECTED (HVG-reduced) subset actually tokenized --
    the full ~16570-gene training panel is not feasible as individual
    self-attended tokens (same O(n_panel^2) cost GeneAttentionDecoder's
    own MAX_SAFE_PANEL_SIZE guard exists for). full_gene_names is the
    full training panel raw_expr's columns are ordered by, needed to
    slice out just the selected genes' values every forward pass."""

    MAX_SAFE_PANEL_SIZE = 4096  # O(n_panel^2) self-attention guard, same reasoning as GeneAttentionDecoder

    def __init__(self, gene_names: list[str], full_gene_names: list[str],
                 feat_dim: int, n_pool_layers: int = 1, n_pool_heads: int = 4):
        super().__init__()
        n_panel = len(gene_names)
        assert n_panel <= self.MAX_SAFE_PANEL_SIZE, (
            f"gene_names has {n_panel} genes, exceeding MAX_SAFE_PANEL_SIZE="
            f"{self.MAX_SAFE_PANEL_SIZE} -- self-attention pooling over gene "
            f"tokens is O(n_panel^2) per spot, same guard as GeneAttentionDecoder"
        )
        name_to_idx = {g: i for i, g in enumerate(full_gene_names)}
        missing = [g for g in gene_names if g not in name_to_idx]
        assert not missing, (
            f"gene_names contains {len(missing)} gene(s) not in full_gene_names "
            f"(e.g. {missing[:5]})"
        )
        self.register_buffer(
            "_gene_col_idx",
            torch.tensor([name_to_idx[g] for g in gene_names], dtype=torch.long),
        )
        self.identity_embed = nn.Embedding(n_panel, feat_dim)
        self.value_proj = nn.Linear(1, feat_dim)
        pool_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=n_pool_heads, dim_feedforward=feat_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.pool_transformer = nn.TransformerEncoder(pool_layer, num_layers=n_pool_layers)
        self.out_norm = nn.LayerNorm(feat_dim)

    def forward(self, raw_expr: torch.Tensor) -> torch.Tensor:
        """raw_expr: [B, n_genes_full] (already log1p'd by the caller, same
        convention as MLPGeneEncoder -- see StormLiteContextEncoder's own
        _maybe_log1p). Returns [B, feat_dim]."""
        selected = raw_expr[:, self._gene_col_idx]           # [B, n_panel]
        values = self.value_proj(selected.unsqueeze(-1))     # [B, n_panel, feat_dim]
        identity = self.identity_embed.weight.unsqueeze(0)   # [1, n_panel, feat_dim], broadcasts over batch
        tokens = values + identity                            # [B, n_panel, feat_dim] -- scGPT's real "add" rule
        pooled = self.pool_transformer(tokens)                # [B, n_panel, feat_dim] -- genes attend to each other
        return self.out_norm(pooled.mean(dim=1))              # [B, feat_dim]


class OrganTechEmbedding(nn.Module):
    """Learned per-organ/per-technology embedding, added identically to
    every spot's representation within one sample (matches STPath's own
    verified pattern — EncodeInputs sums img_embed + ge_embed + tech_embed
    + organ_embed, confirmed via its real source, see stpath_encoder.py —
    built as our own module here since we don't share STPath's
    IDTokenizer/vocabulary).

    Only meaningful for MULTI-sample training spanning genuinely
    different organs/platforms (2026-07-16, multi-sample training
    scaffolding follow-up — see MultiSampleMaskedContextQueryDataset in
    src/training/train.py). On a single-sample or single-organ/single-
    platform training run this contributes a CONSTANT offset every model
    would trivially fold into its own bias terms — zero real signal.
    Built now so the mechanism is ready and tested; not yet validated
    against real organ/platform variation (as of this writing, INT1-
    INT24 — this project's only confirmed-available HEST-1k samples —
    are documented as all-Visium, same ccRCC cohort: see
    load_multi_sample's own docstring in src/data/loaders.py; real
    cross-organ signal requires additional samples not yet downloaded).

    Fixed vocabulary, built ONCE from the training data before model
    construction (same "vocabulary size must be fixed at construction
    time" reasoning as inject_stpath_gene_names) — see
    build_organ_tech_vocab below. Looking up a name outside that
    vocabulary raises loudly rather than silently defaulting to
    something, since defaulting to e.g. index 0 would silently and
    incorrectly claim two different organs are the same."""

    def __init__(self, organ_vocab: list[str], tech_vocab: list[str], hidden_dim: int):
        super().__init__()
        self.organ_to_id = {name: i for i, name in enumerate(organ_vocab)}
        self.tech_to_id = {name: i for i, name in enumerate(tech_vocab)}
        self.organ_embed = nn.Embedding(len(organ_vocab), hidden_dim)
        self.tech_embed = nn.Embedding(len(tech_vocab), hidden_dim)

    def forward(self, organ: str, tech: str, n: int, device) -> torch.Tensor:
        """Returns [n, hidden_dim] — the same (organ, tech) embedding
        broadcast to every one of this sample's n spots (context and
        query combined, or called separately for each — same result
        either way since it doesn't depend on n beyond the broadcast)."""
        if organ not in self.organ_to_id:
            raise KeyError(
                f"organ {organ!r} not in vocab {sorted(self.organ_to_id)} — "
                f"rebuild the vocabulary (build_organ_tech_vocab) to include it"
            )
        if tech not in self.tech_to_id:
            raise KeyError(
                f"tech {tech!r} not in vocab {sorted(self.tech_to_id)} — "
                f"rebuild the vocabulary (build_organ_tech_vocab) to include it"
            )
        organ_t = torch.tensor(self.organ_to_id[organ], device=device)
        tech_t = torch.tensor(self.tech_to_id[tech], device=device)
        embed = self.organ_embed(organ_t) + self.tech_embed(tech_t)  # [hidden_dim]
        return embed.unsqueeze(0).expand(n, -1)  # [n, hidden_dim]


def build_organ_tech_vocab(organs: list[str], techs: list[str]) -> tuple[list[str], list[str]]:
    """Sorted-unique vocabulary lists from real per-sample organ/tech
    values (see load_multi_sample's organs/techs return) — sorted so the
    resulting vocab (and therefore every embedding index) is deterministic
    regardless of input/dict-iteration order, matching this project's
    established practice for other data-driven vocabularies (e.g.
    load_multi_sample's own shared_genes sorting)."""
    return sorted(set(organs)), sorted(set(techs))


class PanelInvariantGeneDecoder(nn.Module):
    """Gene-identity-lookup decoder — the "GEX decoder ... predicted
    expression in target platform space" box from the user's 2026-07-17
    architecture roadmap (diagram 5), and the real gap flagged in
    docs/possible_extensions.md's "Cross-platform decoder" section: every
    generator's decoder up to now (WAE-GAN/FM-OT/VQ-VAE+AR, registry.py)
    is `nn.Linear(hidden_dim, n_genes)`, tied to exactly one fixed-width
    gene panel at construction time — it has no notion of gene IDENTITY,
    only gene POSITION (output column i is "whatever gene was at index i
    in the training panel"). This class predicts expression by looking up
    each target gene's identity in a learned embedding table instead, so
    the same trained decoder can be queried against a DIFFERENT gene
    subset than it was trained on (e.g. context/training from Visium's
    ~16.5k-gene panel, query for Xenium's ~300-gene panel) — provided the
    queried genes are within the known vocabulary.

    Structurally a simplification of STPath's real per-gene-token output
    head (verified via its source, see stpath_encoder.py's GeneExpTokenizer
    usage): STPath predicts a discretized expression BIN per gene token
    via a classification head; this predicts continuous expression
    directly via a small MLP over (decoder_input, gene_embedding) pairs —
    same "gene identity is a lookup, not a fixed output column" structure,
    simpler regression head rather than porting STPath's tokenizer/binning
    machinery wholesale.

    gene_names is the FIXED vocabulary set at construction time (same
    "vocabulary size must be fixed at construction time" reasoning as
    OrganTechEmbedding/inject_stpath_gene_names) — querying a gene outside
    it raises loudly (see gene_indices) rather than silently degrading.
    Calling forward() with gene_names=None (the default) queries the FULL
    training vocabulary in its original order, making this a drop-in
    replacement for a dense decoder: same output shape [N, n_genes], same
    loss code, no changes needed anywhere else in a training loop that
    doesn't explicitly opt into querying a different panel.

    tech_vocab (optional) adds a per-target-technology offset to every
    gene embedding, letting the same decoder shift its predictions for
    "the same gene, but on a different sequencing technology" (diagram 5's
    "GEX decoder ... also fed by tech embedding" arrow) — NOT the same
    tech_vocab conditioning as OrganTechEmbedding/context_encoder (which
    conditions on the SOURCE data's tech), this one is meant to be called
    with the TARGET/query platform's tech, so the two are allowed to
    diverge once genuine cross-platform training pairs exist. On today's
    data context["tech"] == query["tech"] always (single tech per sample —
    see train.py), so this mechanism is exercised but not yet validated
    against a real source != target case.

    NOT validated end-to-end on genuinely cross-platform data (no
    multi-platform training set currently available — see
    possible_extensions.md's own caveat). Built so the architecture is
    ready the moment such data exists; on today's all-Visium data it only
    ever gets queried with gene_names=None (== the training panel), which
    is the correct, expected, non-degenerate use of this class in that
    regime — not a workaround."""

    def __init__(self, gene_names: list[str], gene_embed_dim: int, in_dim: int,
                 tech_vocab: list[str] | None = None, hidden_dim: int = 128,
                 mlp_depth: int = 1, combine_mode: str = "concat"):
        """hidden_dim/mlp_depth (2026-07-17, capacity follow-up — see
        docs/results_log.md): the first real result (gene_embed_dim=64,
        hidden_dim implicitly tied to whatever the caller's dense decoder
        width was) scored PCC 0.1308 vs. the dense decoder's 0.2798 at the
        same epochs; bumping gene_embed_dim alone (64->256) closed about
        half that gap (0.2039) but made ST-FID slightly worse — capacity
        was a real factor but not the whole story. hidden_dim now defaults
        to 128 rather than silently inheriting the caller's dense-decoder
        width, and is a genuinely independent knob (see _build_decoder's
        decoder_hidden_dim in registry.py) rather than reusing
        ae_hidden_dim/cond_hidden_dim. mlp_depth (default 1, matching the
        original 2-layer Linear->GELU->Linear structure exactly) adds
        capacity along a second, different axis — depth, not just width —
        each extra unit inserts one more Linear(hidden_dim,
        hidden_dim)->GELU block before the final Linear(hidden_dim, 1).

        combine_mode (2026-07-17, literature-grounded follow-up): "concat"
        (default, preserves original behavior) concatenates the projected
        query features and gene embedding before the MLP — a [N, n_panel,
        2*hidden_dim] tensor, the real source of this decoder's ~20GB
        training memory footprint. "add" instead combines them via
        element-wise addition, halving that tensor's width — and isn't
        just a cheaper hack: it's scGPT's actual verified mechanism for
        combining gene-identity and expression/context embeddings (Cui
        et al. 2024, Nature Methods — gene identity embedding + expression
        value embedding combined via element-wise addition, not
        concatenation, to form each gene token). Geneformer's real
        architecture (Theodoris et al. 2023, Nature) was also checked as a
        candidate design (genes as literal self-attended sequence
        positions, decoded via a shared per-position head) but full
        self-attention over a ~16570-gene panel is O(n_panel^2) per query
        location — not adopted here, too expensive at this panel size
        without truncation/sparsity machinery this project doesn't have.
        "add" is the practical middle ground: a real, published multi-modal
        gene-token combination rule, not concat's ad hoc default."""
        super().__init__()
        assert combine_mode in ("concat", "add"), f"unknown combine_mode {combine_mode!r}"
        self.gene_names = list(gene_names)
        self._gene_to_idx = {name: i for i, name in enumerate(self.gene_names)}
        assert len(self._gene_to_idx) == len(self.gene_names), (
            "gene_names contains duplicates — decoder vocabulary must be unique"
        )
        assert mlp_depth >= 1, f"mlp_depth must be >= 1, got {mlp_depth}"
        self.combine_mode = combine_mode
        self.gene_embed = nn.Embedding(len(self.gene_names), gene_embed_dim)
        self.tech_vocab = list(tech_vocab) if tech_vocab else None
        if self.tech_vocab:
            self.tech_to_id = {name: i for i, name in enumerate(self.tech_vocab)}
            self.tech_embed = nn.Embedding(len(self.tech_vocab), gene_embed_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gene_proj = nn.Linear(gene_embed_dim, hidden_dim)
        first_in = hidden_dim if combine_mode == "add" else 2 * hidden_dim
        layers = [nn.Linear(first_in, hidden_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.out_mlp = nn.Sequential(*layers)

    def gene_indices(self, gene_names: list[str], device) -> torch.Tensor:
        missing = [g for g in gene_names if g not in self._gene_to_idx]
        assert not missing, (
            f"{len(missing)} gene(s) not in decoder vocabulary "
            f"(e.g. {missing[:5]}) — this decoder can only predict genes "
            f"it was constructed with, see class docstring"
        )
        return torch.tensor(
            [self._gene_to_idx[g] for g in gene_names], device=device, dtype=torch.long
        )

    def forward(self, h: torch.Tensor, gene_names: list[str] | None = None,
                tech: str | None = None) -> torch.Tensor:
        """h: [N, in_dim] per-location decoder input (the same tensor a
        dense decoder would receive, e.g. cat([z, c])). gene_names: target
        panel, defaults to the full training vocabulary in its original
        order (drop-in shape parity with a dense decoder). tech: optional
        target-platform lookup into tech_vocab, ignored if tech_vocab
        wasn't set at construction. Returns [N, len(gene_names)]."""
        device = h.device
        names = gene_names if gene_names is not None else self.gene_names
        idx = self.gene_indices(names, device)
        g = self.gene_embed(idx)                      # [n_panel, gene_embed_dim]
        if tech is not None and self.tech_vocab:
            assert tech in self.tech_to_id, (
                f"tech {tech!r} not in decoder tech_vocab {sorted(self.tech_to_id)}"
            )
            tech_t = torch.tensor(self.tech_to_id[tech], device=device)
            g = g + self.tech_embed(tech_t)            # broadcast over all panel genes
        h_proj = self.in_proj(h)                       # [N, hidden_dim]
        g_proj = self.gene_proj(g)                     # [n_panel, hidden_dim]
        if self.combine_mode == "add":
            # scGPT's real combination rule (see __init__ docstring) — half
            # the memory of "concat" below, [N, n_panel, hidden_dim] not
            # [N, n_panel, 2*hidden_dim].
            combined = h_proj.unsqueeze(1) + g_proj.unsqueeze(0)  # [N, n_panel, hidden_dim]
        else:
            n, n_panel = h_proj.shape[0], g_proj.shape[0]
            h_exp = h_proj.unsqueeze(1).expand(n, n_panel, -1)
            g_exp = g_proj.unsqueeze(0).expand(n, n_panel, -1)
            combined = torch.cat([h_exp, g_exp], dim=-1)          # [N, n_panel, 2*hidden_dim]
        out = self.out_mlp(combined).squeeze(-1)        # [N, n_panel]
        return out


class GeneAttentionDecoder(nn.Module):
    """Geneformer-inspired gene decoder (Theodoris et al. 2023, Nature —
    "Transfer learning enables predictions in network biology"): genes are
    literal self-attended sequence positions, decoded via a SHARED
    per-position output head, rather than PanelInvariantGeneDecoder's
    independent-per-gene scoring (every gene predicted in isolation, no
    way for gene co-expression structure to influence the prediction).
    Real, considered-and-initially-rejected design (see
    docs/results_log.md's 2026-07-17 decoder research entry) — rejected
    only because full self-attention over our ~16570-gene FULL training
    vocabulary is O(n_panel^2) per query location, genuinely too
    expensive. Implemented here WITH a hard safety guard instead
    (MAX_SAFE_PANEL_SIZE) rather than dropped entirely: for the actual
    target use case this decoder exists for — a genuinely smaller
    cross-platform panel (Xenium's is typically ~300-500 genes, not the
    full transcriptome) — full attention is completely tractable
    (300^2 = 90k, trivial), and lets gene predictions influence each
    other, which the independent-scoring design fundamentally cannot do.
    NOT a literal port of Geneformer's own architecture (rank-based input
    encoding, its own masked-pretraining objective) — this borrows only
    the structural idea (genes as attended tokens, shared output head),
    grounded in real published precedent rather than invented from
    scratch, same "structural echo, not a full port" relationship
    PanelInvariantGeneDecoder itself has to STPath's real per-gene-token
    classification head.

    Token construction: additive (see PanelInvariantGeneDecoder's own
    combine_mode="add" docstring for why — scGPT's real, verified
    mechanism, and cheaper than concatenation), not a second design
    decision made independently here.

    NOT validated on genuine cross-platform data (none available — see
    docs/possible_extensions.md). On today's all-Visium data this can
    only be exercised with gene_names restricted to a SUBSET smaller than
    MAX_SAFE_PANEL_SIZE (see tests/test_panel_invariant_decoder.py) —
    querying the full training vocabulary raises loudly rather than
    silently attempting an O(n_panel^2) computation that would OOM."""

    MAX_SAFE_PANEL_SIZE = 4096  # O(n_panel^2) attention guard, see class docstring

    def __init__(self, gene_names: list[str], gene_embed_dim: int, in_dim: int,
                 hidden_dim: int = 128, n_heads: int = 4, n_layers: int = 1,
                 tech_vocab: list[str] | None = None):
        super().__init__()
        self.gene_names = list(gene_names)
        self._gene_to_idx = {name: i for i, name in enumerate(self.gene_names)}
        assert len(self._gene_to_idx) == len(self.gene_names), (
            "gene_names contains duplicates — decoder vocabulary must be unique"
        )
        self.gene_embed = nn.Embedding(len(self.gene_names), gene_embed_dim)
        self.tech_vocab = list(tech_vocab) if tech_vocab else None
        if self.tech_vocab:
            self.tech_to_id = {name: i for i, name in enumerate(self.tech_vocab)}
            self.tech_embed = nn.Embedding(len(self.tech_vocab), gene_embed_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.gene_proj = nn.Linear(gene_embed_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(hidden_dim, 1)  # SHARED across every gene position

    def gene_indices(self, gene_names: list[str], device) -> torch.Tensor:
        missing = [g for g in gene_names if g not in self._gene_to_idx]
        assert not missing, (
            f"{len(missing)} gene(s) not in decoder vocabulary "
            f"(e.g. {missing[:5]}) — this decoder can only predict genes "
            f"it was constructed with, see class docstring"
        )
        return torch.tensor(
            [self._gene_to_idx[g] for g in gene_names], device=device, dtype=torch.long
        )

    def forward(self, h: torch.Tensor, gene_names: list[str] | None = None,
                tech: str | None = None) -> torch.Tensor:
        """h: [N, in_dim]. gene_names: target panel — unlike
        PanelInvariantGeneDecoder, does NOT default to the full training
        vocabulary, since that would silently trigger the O(n_panel^2)
        blowup this class exists to guard against; always pass an
        explicit, reasonably-sized panel. Returns [N, len(gene_names)]."""
        assert gene_names is not None, (
            "GeneAttentionDecoder requires an explicit gene_names panel "
            "(no full-vocabulary default — see class docstring, "
            "MAX_SAFE_PANEL_SIZE guard)"
        )
        n_panel = len(gene_names)
        assert n_panel <= self.MAX_SAFE_PANEL_SIZE, (
            f"gene_names has {n_panel} genes, exceeding MAX_SAFE_PANEL_SIZE="
            f"{self.MAX_SAFE_PANEL_SIZE} — full self-attention over a panel "
            f"this large is O(n_panel^2) and will exhaust GPU memory. This "
            f"decoder is designed for realistic target-platform panel sizes "
            f"(e.g. Xenium's ~300-500 genes), not a full transcriptome — use "
            f"PanelInvariantGeneDecoder (decoder_type='panel_invariant') "
            f"instead for full-vocabulary decoding."
        )
        device = h.device
        idx = self.gene_indices(gene_names, device)
        g = self.gene_embed(idx)                       # [n_panel, gene_embed_dim]
        if tech is not None and self.tech_vocab:
            assert tech in self.tech_to_id, (
                f"tech {tech!r} not in decoder tech_vocab {sorted(self.tech_to_id)}"
            )
            tech_t = torch.tensor(self.tech_to_id[tech], device=device)
            g = g + self.tech_embed(tech_t)
        g_proj = self.gene_proj(g)                      # [n_panel, hidden_dim]
        h_proj = self.in_proj(h)                        # [N, hidden_dim]
        n = h_proj.shape[0]
        tokens = g_proj.unsqueeze(0).expand(n, n_panel, -1) + h_proj.unsqueeze(1)  # [N, n_panel, hidden_dim]
        contextualized = self.transformer(tokens)       # self-attention among gene tokens, per query location
        return self.output_head(contextualized).squeeze(-1)  # [N, n_panel]


class LLOKIStyleDecoder(nn.Module):
    """Faithfully ports LLOKI-CAE's real, verified conditional-autoencoder
    mechanism (Levy et al. 2025, Genome Research — source verified
    directly: github.com/ma-compbio/LLOKI, lloki/cae/conditional_autoencoder.py,
    2026-07-17). Real architecture, confirmed from source: encoder input =
    concat(features, technology_embedding), decoder input =
    concat(latent, technology_embedding), multi-layer stack of Linear+ReLU
    (final layer has no activation), technology embedding is a learned
    nn.Embedding looked up by a discrete batch/technology index and
    concatenated ONCE at the input (not injected at every layer).

    IMPORTANT — this is deliberately NOT a claim of full LLOKI parity, and
    NOT panel-invariant the way PanelInvariantGeneDecoder is (fixed
    n_genes width, tied at construction, same limitation as the original
    "dense" decoder): LLOKI's real panel-invariance comes from a SEPARATE,
    heavier component (LLOKI-FP), which uses an external pretrained
    single-cell foundation model (scGPT) to impute/embed arbitrary gene
    panels into a shared space BEFORE LLOKI-CAE ever sees them — LLOKI-CAE
    itself only does cross-technology batch integration within that
    already-shared space. Standing up an FP-equivalent here would mean
    integrating a full external single-cell foundation model, out of
    scope for a decoder swap. What's ported here is the one directly
    transferable, verified piece: technology-conditioned decoding via
    input concatenation, useful for a DIFFERENT real problem than panel
    mismatch — adapting the output distribution to the source
    technology's detection biases/dropout patterns within the SAME gene
    panel. `tech` is therefore REQUIRED here (not optional, unlike
    PanelInvariantGeneDecoder/GeneAttentionDecoder), since the whole
    mechanism depends on it.

    hidden_dims defaults to (512, 256, 128), the real shape LLOKI-CAE's
    paper/repo describes for its own encoder/decoder stacks — kept as the
    default for fidelity to the source, not re-tuned for this project's
    much smaller pilot scale."""

    def __init__(self, in_dim: int, n_genes: int, tech_vocab: list[str],
                 tech_embed_dim: int = 10, hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        assert tech_vocab, "LLOKIStyleDecoder requires a non-empty tech_vocab"
        self.tech_to_id = {name: i for i, name in enumerate(tech_vocab)}
        self.tech_embed = nn.Embedding(len(tech_vocab), tech_embed_dim)
        dims = [in_dim + tech_embed_dim] + list(hidden_dims) + [n_genes]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # ReLU on every layer except the final output layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, tech: str) -> torch.Tensor:
        """h: [N, in_dim]. tech: REQUIRED (see class docstring). Returns
        [N, n_genes] — fixed width, not panel-invariant."""
        assert tech in self.tech_to_id, (
            f"tech {tech!r} not in decoder tech_vocab {sorted(self.tech_to_id)}"
        )
        device = h.device
        tech_t = torch.tensor(self.tech_to_id[tech], device=device)
        tech_vec = self.tech_embed(tech_t).unsqueeze(0).expand(h.shape[0], -1)
        return self.net(torch.cat([h, tech_vec], dim=-1))


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

    gene_encoder_type mirrors the same pattern for the expression side
    (2026-07-16, "is our own gene-expression branch also just a thin
    linear projection, same critique as STPath's" investigation):
    "raw" (default, unchanged) concatenates the raw n_genes expression
    vector directly, same as always. "mlp" swaps in MLPGeneEncoder (own,
    trained-jointly nonlinear encoder). "novae" swaps in NovaeGeneEncoder
    (frozen pretrained ST foundation model, precompute_novae_features
    required upstream — see that function and _load_novae_features in
    src/training/train.py). context_expression passed to forward() becomes
    [N_obs, novae_dim] precomputed features instead of [N_obs, n_genes]
    raw expression when gene_encoder_type="novae" — same "which tensor
    this positional arg actually holds depends on config" pattern
    context_images already has for image_encoder_type. Query locations
    never carry a gene-encoder input at all — predicting expression AT
    the query is the task, so this was already true for "raw"/"mlp" and
    stays true for "novae" too.
    """

    def __init__(self, n_genes: int, coord_dim: int = 3, hidden_dim: int = 256,
                 n_message_layers: int = 2, k_neighbors: int = 10,
                 rff_features: int = 64, rff_sigma: float = 1.0, coord_scale: float = 1.0,
                 image_encoder_type: str = "none", image_feat_dim: int = 64,
                 image_patch_size: int = 256,
                 gene_encoder_type: str = "raw", gene_feat_dim: int = 256,
                 novae_dim: int | None = None,
                 organ_vocab: list[str] | None = None, tech_vocab: list[str] | None = None):
        super().__init__()
        assert image_encoder_type in ("none", "cnn", "gigapath"), (
            f"unknown image_encoder_type {image_encoder_type!r}"
        )
        assert gene_encoder_type in ("raw", "mlp", "novae"), (
            f"unknown gene_encoder_type {gene_encoder_type!r}"
        )
        self.k_neighbors = k_neighbors
        self.use_images = image_encoder_type != "none"
        self.gene_encoder_type = gene_encoder_type
        # coord_scale (2026-07-17, see RandomFourierFeatures' own docstring
        # for the real bug this fixes): a FIXED value set once here, not
        # per-call — this class calls coord_encoder SEPARATELY for
        # context_coords/query_coords below, so a per-call auto-scale
        # would give the same real position different encodings depending
        # on which call computed it.
        self.coord_encoder = RandomFourierFeatures(coord_dim, rff_features, rff_sigma, coord_scale)
        coord_feat_dim = 2 * rff_features  # sin + cos

        if gene_encoder_type == "raw":
            self.gene_encoder = None
            gene_out_dim = n_genes
        elif gene_encoder_type == "mlp":
            self.gene_encoder = MLPGeneEncoder(n_genes, gene_feat_dim)
            gene_out_dim = gene_feat_dim
        else:  # novae
            assert novae_dim is not None, (
                "gene_encoder_type='novae' requires novae_dim (read it from "
                "precompute_novae_features()'s real output shape — see "
                "src/training/train.py's Novae-loading path)"
            )
            self.gene_encoder = NovaeGeneEncoder(novae_dim, gene_feat_dim)
            gene_out_dim = gene_feat_dim

        node_in_dim = gene_out_dim + coord_feat_dim
        query_in_dim = coord_feat_dim
        if image_encoder_type == "cnn":
            self.image_encoder = ImagePatchEncoder(image_patch_size, image_feat_dim)
            node_in_dim += image_feat_dim
            query_in_dim += image_feat_dim
        elif image_encoder_type == "gigapath":
            self.image_encoder = GigapathPatchEncoder(image_feat_dim)
            node_in_dim += image_feat_dim
            query_in_dim += image_feat_dim

        if self.use_images:
            self.missing_image_token = nn.Parameter(torch.randn(image_feat_dim) * 0.02)

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            _KNNMessageLayer(hidden_dim) for _ in range(n_message_layers)
        )
        self.query_proj = nn.Linear(query_in_dim, hidden_dim)
        self.query_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # 2026-07-16, multi-sample training follow-up (see OrganTechEmbedding's
        # own docstring for the real caveat: only meaningful once training
        # spans genuinely different organs/platforms). Added AFTER
        # node_proj/query_proj (not concatenated into their input) so
        # enabling/disabling it never changes node_in_dim/query_in_dim —
        # a pure additive offset at hidden_dim, same fusion style as
        # STPath's own verified organ_embed/tech_embed pattern.
        self.organ_tech_embed = None
        if organ_vocab is not None and tech_vocab is not None:
            self.organ_tech_embed = OrganTechEmbedding(organ_vocab, tech_vocab, hidden_dim)

    def _encode_gene(self, expression: torch.Tensor) -> torch.Tensor:
        return expression if self.gene_encoder is None else self.gene_encoder(expression)

    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor | None = None,
                query_images: torch.Tensor | None = None,
                context_image_available: torch.Tensor | None = None,
                query_image_available: torch.Tensor | None = None,
                context_novae_features: torch.Tensor | None = None,
                organ: str | None = None, tech: str | None = None) -> torch.Tensor:
        # NOTE: gene_encoder_type applies to context_expression only, same
        # as "raw"/"mlp" always did — query locations never carry an
        # expression feature at all (predicting it is the task), so there
        # is no query-side counterpart to add here for "novae" either.
        # context_novae_features is accepted-and-ignored here — it exists
        # so BaseGenerativeModel._encode_context (registry.py) can call
        # every context encoder type with the same call signature; this
        # class's own gene_encoder_type="novae" path (added 2026-07-16)
        # already gets Novae features THROUGH context_expression itself
        # (context_gene_features replaces it upstream in train.py), unlike
        # STPathContextEncoder's Route-B residual, which needs both raw
        # expression AND Novae features simultaneously and so needs this
        # as a genuinely separate channel.
        #
        # organ/tech (2026-07-16, multi-sample follow-up): unlike gene
        # expression, organ/platform metadata is fully known for BOTH
        # context and query (it describes the whole sample, not a
        # per-point measurement that could leak the prediction target),
        # so the same embedding gets added to every node AND every query
        # position — no context/query asymmetry needed here, unlike
        # gene_encoder_type.
        if self.use_images and (context_images is None or query_images is None):
            raise ValueError("use_images=True requires context_images and query_images")

        # 1. embed each context node from its own expression + coordinate (+ image)
        context_coord_feat = self.coord_encoder(context_coords)
        node_feats = [self._encode_gene(context_expression), context_coord_feat]
        if self.use_images:
            context_image_feat = self.image_encoder(context_images)
            if context_image_available is not None:
                available = context_image_available.to(context_image_feat.device).bool()
                context_image_feat = torch.where(
                    available[:, None], context_image_feat, self.missing_image_token[None, :]
                )
            node_feats.append(context_image_feat)
        node_repr = self.node_proj(torch.cat(node_feats, dim=-1))
        if self.organ_tech_embed is not None and organ is not None and tech is not None:
            node_repr = node_repr + self.organ_tech_embed(
                organ, tech, node_repr.shape[0], node_repr.device
            )

        # 2. message-pass over the context's own k-NN graph
        context_knn = _knn_indices(context_coords, context_coords, self.k_neighbors)
        for layer in self.message_layers:
            node_repr = layer(node_repr, context_knn)

        # 3. for each query location, attend over its k nearest context nodes
        query_knn = _knn_indices(query_coords, context_coords, self.k_neighbors)
        query_feats = [self.coord_encoder(query_coords)]
        if self.use_images:
            query_image_feat = self.image_encoder(query_images)
            if query_image_available is not None:
                available = query_image_available.to(query_image_feat.device).bool()
                query_image_feat = torch.where(
                    available[:, None], query_image_feat, self.missing_image_token[None, :]
                )
            query_feats.append(query_image_feat)
        query_feat = self.query_proj(torch.cat(query_feats, dim=-1))
        if self.organ_tech_embed is not None and organ is not None and tech is not None:
            query_feat = query_feat + self.organ_tech_embed(
                organ, tech, query_feat.shape[0], query_feat.device
            )
        neighbor_repr = node_repr[query_knn]                     # [N_query, k, hidden_dim]
        c, _ = self.query_attn(query_feat.unsqueeze(1), neighbor_repr, neighbor_repr)
        return c.squeeze(1)                                        # [N_query, hidden_dim]