"""Narrow encoder-provider interfaces (GEN4_CONTRACT.md section 5).

These are documentation-as-code Protocols, not base classes anything must
inherit from -- Gen4's real cache builders/encoders (uni2_encoder.py,
scfoundation_encoder.py, stpath_context.py) satisfy them structurally.
Kept separate from any one encoder module so `Gen4Conditioner` can type its
dependencies without importing UNI2/scFoundation/STPath-specific code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EncoderIdentity:
    """Common provenance shape every frozen Gen4 encoder wrapper records at
    construction time -- mirrors the fields `data/spot_feature_cache.py`
    already requires for GigaPath (checkpoint hash, pinned revision,
    package version, preprocessing spec), generalized across encoder
    families so cache files can be validated the same way regardless of
    which encoder produced them."""
    encoder_name: str  # "uni2" | "scfoundation" | "stpath"
    checkpoint_sha256: str
    pinned_revision: str
    package_version: str
    preprocessing_spec: str
    output_dim: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in ("encoder_name", "checkpoint_sha256", "pinned_revision", "package_version", "preprocessing_spec"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"EncoderIdentity.{field_name} must be a non-empty string, got {value!r}")
        if self.output_dim <= 0:
            raise ValueError(f"EncoderIdentity.output_dim must be positive, got {self.output_dim}")

    def as_dict(self) -> dict:
        return {
            "encoder_name": self.encoder_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "pinned_revision": self.pinned_revision,
            "package_version": self.package_version,
            "preprocessing_spec": self.preprocessing_spec,
            "output_dim": int(self.output_dim),
            "schema_version": int(self.schema_version),
        }


class ImageContextProvider(Protocol):
    """Offline, row-independent (per-spot) image encoder. Never called at
    forward time by Gen4Conditioner -- only by a cache builder. Must never
    silently fall back to a different checkpoint/model on failure."""

    identity: EncoderIdentity

    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        """patches: [N, H, W, 3] uint8/float in [0,255] for AVAILABLE spots
        only (a caller must never pass an unavailable/placeholder patch --
        see GEN4_CONTRACT.md section 9). Returns [N, output_dim] float32,
        finite."""
        ...


class GexContextProvider(Protocol):
    """Offline, row-independent frozen GEX-context encoder. Must never
    compute a statistic across rows (e.g. batch normalization using the
    batch's own mean/std) -- each row's output must depend only on that
    row's own raw expression, so it is safe to call on training rows,
    validation rows, or test rows independently without any cross-split
    leakage risk from encoding order or batch composition."""

    identity: EncoderIdentity
    gene_names: tuple[str, ...]

    def encode_rows(self, expression: np.ndarray, raw_library_size: np.ndarray | None = None) -> np.ndarray:
        """`expression`: [N, n_genes], row order arbitrary (row-
        independent) -- this codebase's own normalize_log1p transform
        (data/loaders.py::basic_qc_and_normalize, the pipeline default),
        NOT raw counts; see `gen4.scfoundation_encoder`'s own docstring
        for why a provider that needs the real raw per-row total count
        (a read-depth token, e.g. scFoundation) cannot derive it from
        this matrix by summing it. `raw_library_size`: optional [N] real,
        PRE-normalization total count per row (data/loaders.py's own
        `adata.obs['_scilifestdl_raw_library_size']`), row-aligned with
        `expression` -- `None` for providers that do not need it; a
        provider that does must raise if it is not given. Returns
        [N, output_dim] float32, finite."""
        ...
