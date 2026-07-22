"""Hierarchical, mask-aware H&E/GEX context for missing-tissue prediction.

This module deliberately separates three spatial scales:

1. a frozen Prov-GigaPath LongNet slide encoder over *visible* WSI tiles;
2. observed, spot-aligned H&E/GEX/Novae tokens; and
3. query-to-observed cross-attention for the physically missing region.

The slide encoder never receives a tile intersecting the query hole.  That
constraint is enforced by the data builder (``src.training.train``), while
this module fails closed if slide inputs are missing or malformed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from src.models.conditioning import GigapathPatchEncoder, MLPGeneEncoder, NovaeGeneEncoder


class WeightedGeneExpressionEncoder(nn.Module):
    """STPath-style expression-weighted gene-vocabulary embedding.

    With log-normalized expression ``x`` and one learned vector per gene,
    ``x @ W`` is exactly a weighted bag of gene embeddings.  Cross-spot and
    nonlinear processing belongs to the spatial context Transformer, rather
    than an oversized pointwise MLP before spatial information is introduced.
    """

    def __init__(self, n_genes: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(n_genes, output_dim, bias=False)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(f"expression must be [N,G], got {tuple(expression.shape)}")
        return self.projection(expression)


class FrozenGigaPathSlideEncoder(nn.Module):
    """Frozen official Prov-GigaPath slide encoder with a small output cache.

    Prov-GigaPath's public slide model returns one CLS representation per
    requested layer.  It does not return spatially aligned query features, so
    its scientifically honest role here is a global WSI-context vector.  The
    local missing-region prediction remains the responsibility of explicit
    query-to-observed attention below.

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

        model_features = tile_features
        if tile_features.device.type == "cuda":
            # LongNet's positional buffer is FP32 in the released checkpoint.
            # Autocasting only the linear operations is insufficient because
            # adding that buffer can promote the token stream back to FP32,
            # which FlashAttention rejects.  The slide model is frozen, so
            # keep the complete inference module (parameters and buffers) and
            # its image-token input explicitly in FP16 on CUDA. Coordinates
            # remain FP32 because they are used only for discrete position
            # indexing before attention.
            parameter = next(self.model.parameters())
            if parameter.device != tile_features.device or parameter.dtype != torch.float16:
                self.model.to(device=tile_features.device, dtype=torch.float16)
            model_features = tile_features.to(dtype=torch.float16)

        # This mirrors the official Prov-GigaPath inference pipeline, which
        # runs LongNet under CUDA autocast.  It materially reduces WSI-token
        # memory while the returned vector is converted back to the local
        # trainable path's dtype before projection.
        with torch.no_grad(), torch.autocast(
            device_type=tile_features.device.type,
            dtype=torch.float16,
            enabled=tile_features.device.type == "cuda",
        ):
            output = self.model(
                model_features.unsqueeze(0), tile_coords.unsqueeze(0)
            )
            embedding = self._extract_last_embedding(output).squeeze(0)
        embedding = embedding.to(dtype=tile_features.dtype)
        if embedding.shape != (self.output_dim,):
            raise RuntimeError(
                f"GigaPath slide embedding must be [{self.output_dim}], got {tuple(embedding.shape)}"
            )
        if self.cache_entries > 0:
            if len(self._cache) >= self.cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = embedding.detach().to(device="cpu", dtype=torch.float32)
        return embedding


class QueryCrossAttentionBlock(nn.Module):
    """One query-to-observed attention block; queries never become values."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, query: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        # query [Nq,H], observed [Nq,K,H]
        q = self.query_norm(query).unsqueeze(1)
        kv = self.context_norm(observed)
        update, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + update.squeeze(1)
        return query + self.ff(self.ff_norm(query))


class HierarchicalMissingTissueEncoder(nn.Module):
    """Observed multimodal tissue -> one representation per missing spot."""

    def __init__(
        self,
        n_genes: int,
        novae_dim: int | None,
        hidden_dim: int = 256,
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
    ):
        super().__init__()
        if local_k < 1:
            raise ValueError("local_k must be positive")
        if use_novae and novae_dim is None:
            raise ValueError("use_novae=True requires the real context-only Novae dimension")
        if use_slide_context and not slide_checkpoint_path:
            raise ValueError("use_slide_context=True requires slide_checkpoint_path")

        self.hidden_dim = int(hidden_dim)
        self.local_k = int(local_k)
        self.use_novae = bool(use_novae)
        self.use_local_images = bool(use_local_images)
        self.use_slide_context = bool(use_slide_context)

        self.image_encoder = GigapathPatchEncoder(hidden_dim) if use_local_images else None
        if gene_encoder_type == "weighted_linear":
            self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, hidden_dim)
        elif gene_encoder_type == "mlp":
            self.gene_encoder = MLPGeneEncoder(n_genes, hidden_dim)
        else:
            raise ValueError("gene_encoder_type must be 'weighted_linear' or 'mlp'")
        self.novae_encoder = NovaeGeneEncoder(int(novae_dim), hidden_dim) if use_novae else None
        modality_count = 1 + int(use_local_images) + int(use_novae)
        self.context_fusion = nn.Sequential(
            nn.Linear(modality_count * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.relative_coord = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.normalized_coord = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

        if use_slide_context:
            self.slide_encoder = FrozenGigaPathSlideEncoder(
                checkpoint_path=str(slide_checkpoint_path), output_dim=slide_output_dim
            )
            self.slide_projection = nn.Sequential(
                nn.LayerNorm(slide_output_dim), nn.Linear(slide_output_dim, hidden_dim)
            )
            self.missing_slide_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        else:
            self.slide_encoder = None
            self.slide_projection = None
            self.register_parameter("missing_slide_token", None)

        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(
            context_layer, num_layers=context_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.query_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.cross_blocks = nn.ModuleList([
            QueryCrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(cross_layers)
        ])
        if query_layers > 0:
            query_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.query_transformer = nn.TransformerEncoder(
                query_layer, num_layers=query_layers, norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.query_transformer = None

    @staticmethod
    def _normalized_coords(context_coords: torch.Tensor, query_coords: torch.Tensor):
        all_coords = torch.cat([context_coords, query_coords], dim=0)
        center = context_coords.mean(dim=0, keepdim=True)
        scale = context_coords.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0)
        return (all_coords - center) / scale

    def _slide_token(self, context: dict) -> torch.Tensor:
        if not self.use_slide_context:
            return torch.zeros(
                self.hidden_dim, device=context["coords"].device,
                dtype=context["expression"].dtype,
            )
        available = bool(context.get("slide_available", False))
        if not available:
            return self.missing_slide_token
        required = ("slide_images", "slide_coords", "slide_context_id")
        missing = [key for key in required if key not in context]
        if missing:
            raise ValueError(f"slide context is marked available but missing keys: {missing}")
        raw = self.slide_encoder(
            context["slide_images"], context["slide_coords"],
            str(context["slide_context_id"]),
        )
        return self.slide_projection(raw)

    def forward(self, context: dict, query: dict) -> torch.Tensor:
        context_coords = context["coords"]
        query_coords = query["coords"]
        if context_coords.shape[0] < 1 or query_coords.shape[0] < 1:
            raise ValueError("hierarchical missing-tissue encoder requires non-empty context and query")

        modalities = [self.gene_encoder(context["expression"])]
        if self.use_local_images:
            if "images" not in context:
                raise ValueError("use_local_images=True requires observed GigaPath tile features")
            image = self.image_encoder(context["images"])
            if "image_available" in context:
                available = context["image_available"].bool()
                image = image * available[:, None].to(image.dtype)
            modalities.append(image)
        if self.use_novae:
            if "novae_features" not in context:
                raise ValueError("use_novae=True requires context-only Novae features")
            modalities.append(self.novae_encoder(context["novae_features"]))

        normalized = self._normalized_coords(context_coords, query_coords)
        context_coord = self.normalized_coord(normalized[: context_coords.shape[0]])
        query_coord = self.normalized_coord(normalized[context_coords.shape[0] :])
        slide = self._slide_token(context)
        observed = self.context_fusion(torch.cat(modalities, dim=-1)) + context_coord + slide
        observed = self.context_transformer(observed.unsqueeze(0)).squeeze(0)

        distance = torch.cdist(query_coords[:, :2], context_coords[:, :2])
        k = min(self.local_k, context_coords.shape[0])
        nearest_distance, nearest_index = torch.topk(
            distance, k=k, dim=-1, largest=False, sorted=True
        )
        neighbor = observed[nearest_index]
        delta = context_coords[nearest_index, :2] - query_coords[:, None, :2]
        local_scale = nearest_distance[:, -1:].clamp_min(1e-6)
        relative = torch.cat(
            [delta / local_scale[..., None], nearest_distance[..., None] / local_scale[..., None]],
            dim=-1,
        )
        neighbor = neighbor + self.relative_coord(relative)

        query_hidden = self.query_token[None, :] + query_coord + slide
        for block in self.cross_blocks:
            query_hidden = block(query_hidden, neighbor)
        if self.query_transformer is not None:
            query_hidden = self.query_transformer(query_hidden.unsqueeze(0)).squeeze(0)
        return query_hidden
