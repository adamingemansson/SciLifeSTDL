"""Mask-aware GigaPath slide context: one frozen LongNet global token plus
regional H&E tokens computed from spatial pooling -- Phase 3 of the
multiscale spatial-field handoff ("Adapt the mask-aware WSI path").

FrozenGigaPathSlideEncoder below is copied VERBATIM (narrow extraction,
not the whole file) from src/models/hierarchical_slide.py at commit
c4cba30 -- see gen3_multiscale/data/hest1k_catalog.py's identical copy-
provenance note for why. This is the audited, fail-closed-on-missing-
checkpoint official Prov-GigaPath LongNet wrapper; the rest of that
file's HierarchicalMissingTissueEncoder/QueryCrossAttentionBlock (the
OLDER fusion design) is deliberately NOT copied here -- the handoff is
explicit: "Do not copy the older fusion design blindly, and do not
compare old and new result numbers as if they came from the same
protocol." Do not let this class drift from its src/ origin without a
deliberate reason.

pool_regional_tokens is NEW: GigaPath's stable public slide-encoder
interface returns only one global CLS vector per call -- it does not
expose per-tile contextualized states -- so regional tokens come from
spatially pooling the VISIBLE pre-LongNet tile embeddings into a fixed
grid instead, exactly as the handoff specifies: "If the official, stable
GigaPath interface exposes contextualized tile states, pool them
spatially into regional tokens. Otherwise, spatially pool the visible
pre-LongNet GigaPath tile embeddings into a fixed grid... Do not depend
on private unstable internals merely to obtain regional tokens."
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class FrozenGigaPathSlideEncoder(nn.Module):
    """Frozen official Prov-GigaPath slide encoder with a small output cache.

    Prov-GigaPath's public slide model returns one CLS representation per
    requested layer.  It does not return spatially aligned query features, so
    its scientifically honest role here is a global WSI-context vector.  The
    local missing-region prediction remains the responsibility of explicit
    query-to-observed attention in the spatial-field backbone.

    The checkpoint is mandatory.  The upstream ``create_model`` function
    otherwise silently constructs a randomly initialized LongNet when a path
    is wrong, which would invalidate an experiment while still allowing it to
    run.  We reject that state before importing or constructing the model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_arch: str = "gigapath_slide_enc12l768d",
        tile_feature_dim: int = 1536,
        output_dim: int = 768,
        cache_entries: int = 512,
    ):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"GigaPath slide checkpoint not found: {path}. Download the official "
                "prov-gigapath/prov-gigapath slide_encoder.pth and set "
                "GIGAPATH_SLIDE_CHECKPOINT to that exact file. Random LongNet "
                "weights are not permitted."
            )
        try:
            import gigapath.slide_encoder as slide_encoder
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise ImportError(
                "The official Prov-GigaPath package is required for slide context. "
                "Install https://github.com/prov-gigapath/prov-gigapath in the "
                "training environment."
            ) from exc
        # GigaPath's vendored DilatedAttention asserts that FlashAttention is
        # active, and its A100 path sets the callable to None when the optional
        # compiled package is absent.  Detect that state while constructing the
        # model rather than failing deep inside the first validation forward.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7:
            from gigapath.torchscale.component.flash_attention import flash_attn_func

            if flash_attn_func is None:
                raise ImportError(
                    "Prov-GigaPath LongNet requires FlashAttention on A100-class GPUs, "
                    "but its CUDA kernel is unavailable. Install the upstream-pinned "
                    "dependency with `MAX_JOBS=4 python3 -m pip install "
                    "flash-attn==2.5.8 --no-build-isolation`, then start a new Python process."
                )

        self.model = slide_encoder.create_model(
            str(path), model_arch, int(tile_feature_dim)
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tile_feature_dim = int(tile_feature_dim)
        self.output_dim = int(output_dim)
        self.cache_entries = int(cache_entries)
        self._cache: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        # Lightning recursively calls train() on every child module.  Keep the
        # frozen LongNet deterministic (its upstream default has dropout).
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _tensor_digest(coords: torch.Tensor, namespace: str) -> str:
        payload = coords.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes()
        return hashlib.sha256(namespace.encode() + b"\0" + payload).hexdigest()

    @staticmethod
    def _extract_last_embedding(output) -> torch.Tensor:
        # Official repository: list[Tensor[B, D]].  STPath's vendored copy
        # historically returned (outcomes, hidden_states); accepting that
        # shape makes the error message robust without depending on STPath.
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if not torch.is_tensor(output) or output.ndim != 2:
            raise RuntimeError(
                "Unexpected GigaPath slide-encoder output; expected [B,D] or a list of [B,D] tensors"
            )
        return output

    def forward(
        self,
        tile_features: torch.Tensor,
        tile_coords: torch.Tensor,
        cache_namespace: str,
    ) -> torch.Tensor:
        if tile_features.ndim != 2 or tile_features.shape[1] != self.tile_feature_dim:
            raise ValueError(
                f"slide tile features must be [N,{self.tile_feature_dim}], got "
                f"{tuple(tile_features.shape)}"
            )
        if tile_coords.ndim != 2 or tile_coords.shape != (tile_features.shape[0], 2):
            raise ValueError(
                f"slide tile coordinates must be [N,2] aligned with features, got "
                f"{tuple(tile_coords.shape)}"
            )
        if tile_features.shape[0] < 1:
            raise ValueError("GigaPath slide context contains no visible tiles")
        if not torch.isfinite(tile_features).all() or not torch.isfinite(tile_coords).all():
            raise ValueError("GigaPath slide context contains non-finite values")

        key = self._tensor_digest(tile_coords, cache_namespace)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.to(device=tile_features.device, dtype=tile_features.dtype)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Prov-GigaPath LongNet slide inference requires CUDA FlashAttention"
            )
        caller_device = tile_features.device
        caller_dtype = tile_features.dtype
        slide_device = (
            caller_device
            if caller_device.type == "cuda"
            else torch.device("cuda", torch.cuda.current_device())
        )
        # LongNet's positional buffer is FP32 in the released checkpoint.
        # Autocasting only the linear operations is insufficient because
        # adding that buffer can promote the token stream back to FP32, which
        # FlashAttention rejects. Keep the complete frozen inference module
        # (parameters and buffers) and its image-token input explicitly in
        # FP16 on CUDA.  Audit evaluation may place the surrounding learned
        # model back on CPU after Lightning teardown, so the slide encoder has
        # its own device boundary and returns only its compact vector.
        parameter = next(self.model.parameters())
        if parameter.device != slide_device or parameter.dtype != torch.float16:
            self.model.to(device=slide_device, dtype=torch.float16)
        model_features = tile_features.to(device=slide_device, dtype=torch.float16)
        model_coords = tile_coords.to(device=slide_device, dtype=torch.float32)

        # This mirrors the official Prov-GigaPath inference pipeline, which
        # runs LongNet under CUDA autocast.  It materially reduces WSI-token
        # memory while the returned vector is converted back to the local
        # trainable path's dtype before projection.
        with torch.no_grad(), torch.autocast(
            device_type=slide_device.type,
            dtype=torch.float16,
            enabled=True,
        ):
            output = self.model(
                model_features.unsqueeze(0), model_coords.unsqueeze(0)
            )
            embedding = self._extract_last_embedding(output).squeeze(0)
        embedding = embedding.to(device=caller_device, dtype=caller_dtype)
        if embedding.shape != (self.output_dim,):
            raise RuntimeError(
                f"GigaPath slide embedding must be [{self.output_dim}], got {tuple(embedding.shape)}"
            )
        if self.cache_entries > 0:
            if len(self._cache) >= self.cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = embedding.detach().to(device="cpu", dtype=torch.float32)
        return embedding


def pool_regional_tokens(
    visible_tile_features: np.ndarray,
    visible_tile_coords: np.ndarray,
    full_slide_coord_bounds: tuple[float, float, float, float],
    grid_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool visible pre-LongNet tile embeddings into a
    grid_size x grid_size grid of FIXED, slide-stable spatial regions.

    full_slide_coord_bounds = (xmin, xmax, ymin, ymax) must be computed
    from the slide's COMPLETE tile set (before any hole removes tiles),
    not from visible_tile_coords alone -- otherwise grid cell (i, j)
    would refer to a different physical region depending on which hole
    happened to be cut for this particular training item, defeating the
    point of a "regional" token a model could learn to interpret
    consistently across examples on the same slide.

    Returns (tokens, available) where tokens is
    [grid_size*grid_size, tile_feature_dim] (a cell with zero visible
    tiles -- e.g. one fully covered by the hole -- gets an explicit zero
    vector) and available is a [grid_size*grid_size] bool tensor marking
    which cells actually had at least one visible tile, so a caller never
    silently treats a hole-covered region's zero vector as real content.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    xmin, xmax, ymin, ymax = full_slide_coord_bounds
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"invalid full_slide_coord_bounds {full_slide_coord_bounds}")

    n_tiles, feature_dim = visible_tile_features.shape
    if visible_tile_coords.shape != (n_tiles, 2):
        raise ValueError(
            f"visible_tile_coords must be [{n_tiles}, 2] aligned with visible_tile_features, "
            f"got {visible_tile_coords.shape}"
        )

    tokens = np.zeros((grid_size * grid_size, feature_dim), dtype=np.float32)
    counts = np.zeros(grid_size * grid_size, dtype=np.int64)

    if n_tiles > 0:
        x = np.clip(visible_tile_coords[:, 0], xmin, xmax)
        y = np.clip(visible_tile_coords[:, 1], ymin, ymax)
        col = np.clip(((x - xmin) / (xmax - xmin) * grid_size).astype(int), 0, grid_size - 1)
        row = np.clip(((y - ymin) / (ymax - ymin) * grid_size).astype(int), 0, grid_size - 1)
        cell = row * grid_size + col
        for c in range(grid_size * grid_size):
            in_cell = cell == c
            n_in_cell = int(in_cell.sum())
            if n_in_cell > 0:
                tokens[c] = visible_tile_features[in_cell].mean(axis=0)
                counts[c] = n_in_cell

    available = counts > 0
    return torch.from_numpy(tokens), torch.from_numpy(available)


def regional_grid_cell_centers(
    full_slide_coord_bounds: tuple[float, float, float, float], grid_size: int = 4,
) -> torch.Tensor:
    """Real (x, y) center coordinates for each of pool_regional_tokens's
    grid_size x grid_size cells, in the SAME row-major `cell = row *
    grid_size + col` ordering that function uses internally -- so a
    caller can zip cell index i from pool_regional_tokens's output with
    row i here and get the correct physical center for that exact cell,
    needed to compute a real relative geometry from each query to each
    regional token (models.geometry_utils.compute_relative_geometry)."""
    if grid_size < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    xmin, xmax, ymin, ymax = full_slide_coord_bounds
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"invalid full_slide_coord_bounds {full_slide_coord_bounds}")
    cell_w = (xmax - xmin) / grid_size
    cell_h = (ymax - ymin) / grid_size
    centers = np.zeros((grid_size * grid_size, 2), dtype=np.float32)
    for row in range(grid_size):
        for col in range(grid_size):
            cell = row * grid_size + col
            centers[cell, 0] = xmin + (col + 0.5) * cell_w
            centers[cell, 1] = ymin + (row + 0.5) * cell_h
    return torch.from_numpy(centers)
