"""scFoundation per-spot GEX-context cache -- GEN4_CONTRACT.md sections 7/9.
Mirrors uni2_spot_cache.py's discipline exactly, applied to expression rows
instead of image patches: row-independent encoding, mandatory provenance
(checkpoint hash + gene-vocabulary hash), atomic writes, strict validation
on load, its own on-disk directory (`scfoundation_gen3_spot_cache/`).

"Row-independent" is enforced by construction here, not merely documented:
`build_scfoundation_spot_feature_cache` calls the encoder once per manifest
sample on that sample's own `n_spots x n_genes` matrix -- never assembling
a cross-sample batch, and `FrozenSCFoundationEncoder.encode_rows`'s own
contract (providers.py) is that per-row output depends only on that row.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

_REQUIRED_FIELDS = {
    "features", "barcodes", "feature_available", "gene_panel_hash",
    "scfoundation_checkpoint_sha256", "scfoundation_vocab_sha256",
    "scfoundation_package_version", "scfoundation_output_dim", "scfoundation_schema_version",
    "expression_content_hash",
}


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    return Path(cache_root) / "scfoundation_gen3_spot_cache" / f"{sample_id}.npz"


def _expression_content_hash(expression: np.ndarray) -> str:
    """Codex audit finding: cache identity previously hashed neither the
    expression VALUES nor the preprocessing that produced them, so a
    stale/changed input could silently reuse an old cached embedding. This
    hashes the exact float32 bytes of the row-aligned expression matrix
    fed into the encoder -- any change to a single value (a re-run QC
    fix, a different normalization, a corrected count) changes the hash
    and forces a rebuild rather than silently reusing a stale cache."""
    expression = np.ascontiguousarray(expression, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(expression.tobytes())
    return digest.hexdigest()


def build_scfoundation_spot_feature_cache(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    expression: np.ndarray,
    gene_panel_hash: str,
    encoder,
    batch_size: int = 256,
) -> Path:
    """`encoder` must satisfy `gen4.providers.GexContextProvider`. Every
    row of `expression` is treated as available (scFoundation's own
    row-independent encoding never fails per-spot the way an H&E patch
    can be physically missing) -- `feature_available` is still recorded,
    all-True, so the on-disk schema stays uniform with uni2_spot_cache.py's
    availability convention and any future encoder that DOES have partial
    per-row failures can reuse the identical loader contract."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    barcodes = np.asarray([str(b) for b in barcodes])
    expression = np.asarray(expression, dtype=np.float32)
    n = barcodes.shape[0]
    if expression.shape[0] != n:
        raise ValueError(f"{sample_id}: barcodes and expression must be row-aligned")
    unique_barcodes, counts = np.unique(barcodes, return_counts=True)
    duplicated = unique_barcodes[counts > 1]
    if duplicated.size:
        raise ValueError(f"{sample_id}: barcodes contain {duplicated.size} duplicate value(s)")

    output_dim = encoder.identity.output_dim
    features = np.zeros((n, output_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = encoder.encode_rows(expression[start:end])
        if out.shape != (end - start, output_dim):
            raise ValueError(f"{sample_id}: encoder returned shape {out.shape}, expected ({end - start}, {output_dim})")
        features[start:end] = out.astype(np.float32)
    feature_available = np.ones(n, dtype=bool)

    path = _cache_path(cache_root, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            features=features, barcodes=barcodes, feature_available=feature_available,
            gene_panel_hash=np.asarray(gene_panel_hash),
            scfoundation_checkpoint_sha256=np.asarray(encoder.identity.checkpoint_sha256),
            scfoundation_vocab_sha256=np.asarray(encoder.identity.pinned_revision),
            scfoundation_package_version=np.asarray(encoder.identity.package_version),
            scfoundation_output_dim=np.asarray(output_dim),
            scfoundation_schema_version=np.asarray(1),
            expression_content_hash=np.asarray(_expression_content_hash(expression)),
        )
    os.replace(tmp, path)
    return path


def load_scfoundation_spot_features(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    gene_panel_hash: str,
    expression: np.ndarray,
) -> dict:
    """`expression` must be the SAME live `n_spots x n_genes` matrix (row-
    aligned with `barcodes`) that would be fed into the encoder right now.
    Codex audit finding: cache identity previously keyed only on
    (sample_id, barcodes, gene_panel_hash) -- none of which change when
    the underlying expression VALUES change (a re-run QC fix, a corrected
    count, a different normalization), so a stale cache could be reused
    silently. Now fails closed on any content mismatch instead."""
    path = _cache_path(cache_root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"scFoundation spot-feature cache missing for {sample_id}: {path}. Build it with "
            "gen4.scfoundation_cache.build_scfoundation_spot_feature_cache before training."
        )
    cached = np.load(path, allow_pickle=False)
    missing = sorted(_REQUIRED_FIELDS.difference(cached.files))
    if missing:
        raise ValueError(f"scFoundation spot-feature cache {path} is missing fields {missing} -- rebuild it")

    if str(cached["gene_panel_hash"]) != str(gene_panel_hash):
        raise ValueError(
            f"scFoundation spot-feature cache {path} was built against a different gene panel "
            "than the live manifest -- rebuild it"
        )
    real_barcodes = np.asarray([str(b) for b in barcodes])
    cached_barcodes = np.asarray([str(b) for b in cached["barcodes"]])
    if not np.array_equal(cached_barcodes, real_barcodes):
        raise ValueError(f"scFoundation spot-feature cache {path} barcode identity/order mismatch -- rebuild it")
    if expression.shape[0] != real_barcodes.shape[0]:
        raise ValueError(
            f"scFoundation spot-feature cache {path}: live expression has {expression.shape[0]} rows, "
            f"but {real_barcodes.shape[0]} barcodes were supplied -- must be row-aligned"
        )
    live_hash = _expression_content_hash(expression)
    if str(cached["expression_content_hash"]) != live_hash:
        raise ValueError(
            f"scFoundation spot-feature cache {path} was built from different expression values than "
            "the live input (stale cache) -- rebuild it with gen4.scfoundation_cache."
            "build_scfoundation_spot_feature_cache"
        )

    output_dim = int(np.asarray(cached["scfoundation_output_dim"]).item())
    features = np.asarray(cached["features"], dtype=np.float32)
    if features.shape != (real_barcodes.shape[0], output_dim):
        raise ValueError(f"scFoundation spot-feature cache {path} features has wrong shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"scFoundation spot-feature cache {path} contains non-finite values")

    return {
        "features": features,
        "barcodes": real_barcodes,
        "feature_available": np.asarray(cached["feature_available"], dtype=bool),
        "provenance": {
            "checkpoint_sha256": str(cached["scfoundation_checkpoint_sha256"]),
            "vocab_sha256": str(cached["scfoundation_vocab_sha256"]),
            "package_version": str(cached["scfoundation_package_version"]),
            "output_dim": output_dim,
            "schema_version": int(np.asarray(cached["scfoundation_schema_version"]).item()),
        },
    }


def barcode_embedding_lookup(cached: dict) -> dict[str, np.ndarray]:
    """Convert a loaded cache dict into the `{barcode: row}` mapping
    `gen4.inputs.build_gen4_spatial_field_example`'s `gex_context_embedding`
    argument expects."""
    return {str(b): cached["features"][i] for i, b in enumerate(cached["barcodes"])}
