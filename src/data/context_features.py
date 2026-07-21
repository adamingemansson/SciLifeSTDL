"""Leakage-safe context-only feature computation.

Graph encoders such as Novae propagate information between neighboring spots.
Their features therefore must be computed on the observed context subgraph,
not on the complete slide before query expression is hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import json

import numpy as np


FeatureFn = Callable[[object], np.ndarray]


def _mask_digest(obs_names: np.ndarray, context_mask: np.ndarray) -> str:
    selected = np.asarray(obs_names, dtype=str)[context_mask]
    payload = "\n".join(selected.tolist()).encode("utf-8")
    return sha256(payload).hexdigest()[:20]


def _adata_feature_signature(adata, feature_fn: FeatureFn) -> str:
    """Fingerprint data/model inputs that can change graph-derived features.

    The historical cache key used only context barcodes. That could silently
    reuse embeddings after changing the gene panel, preprocessing contract or
    expression matrix or coordinates. Hash the actual sparse/dense matrix,
    observation order and spatial coordinates. This is intentionally stronger
    than the historical per-gene-sum fingerprint: two matrices can have
    identical column sums while assigning expression to different spots.
    """
    x = adata.X
    uns = getattr(adata, "uns", {})
    preprocessing = {
        "expression_state": uns.get("_scilifestdl_expression_state", {}),
        "legacy_expression_preprocessing": uns.get("expression_preprocessing", {}),
    }
    fn_name = f"{getattr(feature_fn, '__module__', '')}.{getattr(feature_fn, '__qualname__', repr(feature_fn))}"
    n_obs = int(getattr(adata, "n_obs", x.shape[0]))
    n_vars = int(getattr(adata, "n_vars", x.shape[1]))
    var_names = getattr(adata, "var_names", [f"column-{i}" for i in range(n_vars)])
    digest = sha256()
    digest.update(b"context-novae-signature-v2\0")
    digest.update(str((n_obs, n_vars)).encode())
    digest.update("\n".join(map(str, getattr(adata, "obs_names", []))).encode())
    digest.update("\n".join(map(str, var_names)).encode())
    if hasattr(x, "tocsr"):
        csr = x.tocsr(copy=False)
        for array in (csr.data, csr.indices, csr.indptr):
            contiguous = np.ascontiguousarray(array)
            digest.update(str((contiguous.shape, contiguous.dtype)).encode())
            digest.update(memoryview(contiguous).cast("B"))
    else:
        contiguous = np.ascontiguousarray(np.asarray(x))
        digest.update(str((contiguous.shape, contiguous.dtype)).encode())
        digest.update(memoryview(contiguous).cast("B"))
    spatial = getattr(adata, "obsm", {}).get("spatial")
    if spatial is not None:
        spatial = np.ascontiguousarray(np.asarray(spatial))
        digest.update(str((spatial.shape, spatial.dtype)).encode())
        digest.update(memoryview(spatial).cast("B"))
    digest.update(json.dumps(preprocessing, sort_keys=True, default=str).encode())
    digest.update(fn_name.encode())
    return digest.hexdigest()[:24]


@dataclass
class ContextOnlyNovaeProvider:
    """Compute Novae features on the observed context subgraph only.

    The provider receives the complete, QC'd AnnData only so it can subset it.
    The feature function is invoked on ``adata[context_mask].copy()``; query
    rows and their expression are physically absent from that object. Optional
    disk caching is keyed by the exact observed barcode set.

    Context-only Novae for a fresh random mask every training step is costly.
    It is intentionally explicit rather than silently falling back to the
    invalid full-slide shortcut. Clean benchmark configs should normally use
    MLP/tokenizer gene encoders; this provider is for fixed-mask studies where
    the extra compute is acceptable.
    """

    adata: object
    cache_dir: str | Path | None = None
    sample_id: str = "sample"
    feature_fn: FeatureFn | None = None

    def __post_init__(self) -> None:
        if self.feature_fn is None:
            from src.models.conditioning import precompute_novae_features
            self.feature_fn = precompute_novae_features
        self.cache_dir = Path(self.cache_dir) if self.cache_dir is not None else None
        self.obs_names = np.asarray(self.adata.obs_names, dtype=str)
        self.feature_signature = _adata_feature_signature(self.adata, self.feature_fn)
        self._output_dim: int | None = None

    @property
    def output_dim(self) -> int | None:
        return self._output_dim

    def __call__(self, context_mask: np.ndarray) -> np.ndarray:
        context_mask = np.asarray(context_mask, dtype=bool)
        if context_mask.shape != (len(self.obs_names),):
            raise ValueError(
                f"context mask shape {context_mask.shape} does not match {len(self.obs_names)} observations"
            )
        if not context_mask.any():
            raise ValueError("cannot compute context features for an empty context")

        cache_path = None
        digest = _mask_digest(self.obs_names, context_mask)
        selected_names = self.obs_names[context_mask]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / (
                f"{self.sample_id}.data-{self.feature_signature}.context-{digest}.npz"
            )
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                cached_signature = str(cached["feature_signature"].item()) if "feature_signature" in cached else ""
                if (cached_signature == self.feature_signature
                        and np.array_equal(cached["obs_names"].astype(str), selected_names)):
                    features = cached["features"].astype(np.float32, copy=False)
                    self._output_dim = int(features.shape[1])
                    return features

        context_adata = self.adata[context_mask].copy()
        features = np.asarray(self.feature_fn(context_adata), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != int(context_mask.sum()):
            raise ValueError(
                "context feature function must return [n_context, feature_dim], "
                f"got {features.shape} for n_context={int(context_mask.sum())}"
            )
        self._output_dim = int(features.shape[1])

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(f"{cache_path.stem}.tmp{__import__('os').getpid()}.npz")
            np.savez(
                tmp, features=features, obs_names=selected_names,
                feature_signature=np.asarray(self.feature_signature),
            )
            __import__("os").replace(tmp, cache_path)
        return features


def model_uses_novae(model_params: dict) -> bool:
    """Return whether a model configuration requests any Novae-derived input."""
    encoder = model_params.get("context_encoder_type", "builtin")
    if encoder == "builtin":
        return model_params.get("gene_encoder_type") == "novae"
    if encoder == "stpath":
        return model_params.get("stpath_new_gene_encoder_type") in {"novae", "both"}
    if encoder == "storm_lite":
        return model_params.get("gene_encoder_type") in {"novae", "both", "tokenizer_novae"}
    return False


def novae_input_mode(cfg, model_params: dict) -> str:
    """Validate and return the requested Novae safety mode.

    Modes:
      - ``disabled`` (default): clean benchmark default; requesting Novae raises.
      - ``context_only``: compute on each observed context subgraph.
      - ``unsafe_full_graph``: historical reproduction only and requires the
        explicit ``data.allow_unsafe_novae=true`` acknowledgement.
    """
    if not model_uses_novae(model_params):
        return "disabled"
    mode = str(cfg.data.get("novae_mode", "disabled"))
    if mode == "disabled":
        raise ValueError(
            "This model requests Novae features, but data.novae_mode is 'disabled'. "
            "Full-slide Novae features leak hidden query expression through graph message passing. "
            "Use a clean MLP/tokenizer config, or explicitly set novae_mode=context_only."
        )
    if mode == "unsafe_full_graph":
        if not bool(cfg.data.get("allow_unsafe_novae", False)):
            raise ValueError(
                "novae_mode=unsafe_full_graph is scientifically contaminated and requires "
                "data.allow_unsafe_novae=true for historical reproduction."
            )
        return mode
    if mode != "context_only":
        raise ValueError(f"unknown data.novae_mode {mode!r}")
    return mode
