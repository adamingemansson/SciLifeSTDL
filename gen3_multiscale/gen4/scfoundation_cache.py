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
    "scfoundation_package_version", "scfoundation_preprocessing_spec",
    "scfoundation_output_dim", "scfoundation_schema_version",
    "expression_content_hash",
}


def _cache_path(cache_root: str | Path, sample_id: str) -> Path:
    return Path(cache_root) / "scfoundation_gen3_spot_cache" / f"{sample_id}.npz"


def _dense_expression_rows(expression, start: int, end: int) -> np.ndarray:
    """Materialize only ``expression[start:end]`` as contiguous float32.

    Real HEST expression is commonly scipy sparse.  Converting the complete
    sample before batching defeats the cache builder's bounded-memory
    contract, so every caller goes through this row-sliced helper instead.
    """
    rows = expression[start:end]
    if hasattr(rows, "toarray"):
        rows = rows.toarray()
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"expression rows must be two-dimensional, got {rows.shape}")
    return np.ascontiguousarray(rows)


def _expression_content_hash(
    expression,
    raw_library_size: np.ndarray | None = None,
    *,
    row_batch_size: int = 256,
) -> str:
    """Codex audit finding: cache identity previously hashed neither the
    expression VALUES nor the preprocessing that produced them, so a
    stale/changed input could silently reuse an old cached embedding. This
    hashes the exact float32 bytes of the row-aligned expression matrix
    fed into the encoder -- any change to a single value (a re-run QC
    fix, a different normalization, a corrected count) changes the hash
    and forces a rebuild rather than silently reusing a stale cache.

    Item 6 (six-launch-blocker audit): `raw_library_size`, when given, is
    folded in too -- it is now a REAL second input the encoder's output
    depends on (the read-depth token), so a cache built from one
    raw_library_size must not be silently reused after that changes
    (e.g. a re-run QC fix that changes per-spot raw counts) even if
    `expression` itself happens to stay byte-identical."""
    if row_batch_size <= 0:
        raise ValueError(f"row_batch_size must be positive, got {row_batch_size}")
    if not hasattr(expression, "shape") or len(expression.shape) != 2:
        raise ValueError(
            f"expression must be a two-dimensional array or sparse matrix, got "
            f"{getattr(expression, 'shape', None)}"
        )
    digest = hashlib.sha256()
    for start in range(0, int(expression.shape[0]), row_batch_size):
        end = min(start + row_batch_size, int(expression.shape[0]))
        digest.update(_dense_expression_rows(expression, start, end).tobytes())
    if raw_library_size is not None:
        digest.update(np.ascontiguousarray(raw_library_size, dtype=np.float32).tobytes())
    return digest.hexdigest()


def build_scfoundation_spot_feature_cache(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    expression,
    gene_panel_hash: str,
    encoder,
    batch_size: int = 256,
    raw_library_size: np.ndarray | None = None,
) -> Path:
    """`encoder` must satisfy `gen4.providers.GexContextProvider`. Every
    row of `expression` is treated as available (scFoundation's own
    row-independent encoding never fails per-spot the way an H&E patch
    can be physically missing) -- `feature_available` is still recorded,
    all-True, so the on-disk schema stays uniform with uni2_spot_cache.py's
    availability convention and any future encoder that DOES have partial
    per-row failures can reuse the identical loader contract.

    `raw_library_size` (Item 6, six-launch-blocker audit): [N] real,
    pre-normalization total count per row (e.g. `adata.obs[
    '_scilifestdl_raw_library_size']`, data/loaders.py's own stash),
    row-aligned with `barcodes`/`expression` -- forwarded to
    `encoder.encode_rows` unchanged. Required for `FrozenSCFoundationEncoder`
    (it cannot derive a real read-depth token from the already-
    normalized `expression` matrix alone); `None` only for encoders
    (e.g. `StubSCFoundationEncoder`) that don't need it."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    barcodes = np.asarray([str(b) for b in barcodes])
    n = barcodes.shape[0]
    if not hasattr(expression, "shape") or len(expression.shape) != 2:
        raise ValueError(
            f"{sample_id}: expression must be a two-dimensional array or sparse matrix"
        )
    if expression.shape[0] != n:
        raise ValueError(f"{sample_id}: barcodes and expression must be row-aligned")
    if raw_library_size is not None:
        raw_library_size = np.asarray(raw_library_size, dtype=np.float32).reshape(-1)
        if raw_library_size.shape[0] != n:
            raise ValueError(f"{sample_id}: raw_library_size has {raw_library_size.shape[0]} rows, expected {n}")
    unique_barcodes, counts = np.unique(barcodes, return_counts=True)
    duplicated = unique_barcodes[counts > 1]
    if duplicated.size:
        raise ValueError(f"{sample_id}: barcodes contain {duplicated.size} duplicate value(s)")

    output_dim = encoder.identity.output_dim
    features = np.zeros((n, output_dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_library_size = raw_library_size[start:end] if raw_library_size is not None else None
        expression_batch = _dense_expression_rows(expression, start, end)
        out = encoder.encode_rows(expression_batch, batch_library_size)
        if out.shape != (end - start, output_dim):
            raise ValueError(f"{sample_id}: encoder returned shape {out.shape}, expected ({end - start}, {output_dim})")
        features[start:end] = out.astype(np.float32)
        print(
            f"scFoundation {sample_id}: encoded {end}/{n} spots",
            flush=True,
        )
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
            scfoundation_preprocessing_spec=np.asarray(encoder.identity.preprocessing_spec),
            scfoundation_output_dim=np.asarray(output_dim),
            # Codex audit finding (Item 2): preprocessing_spec was
            # previously never persisted to the cache at all, so a
            # preprocessing change (e.g. a fixed read-depth-token
            # derivation bug) could silently keep reusing a stale cache
            # built under the old preprocessing. schema_version bumped so
            # any pre-existing cache missing this field fails closed via
            # _REQUIRED_FIELDS below, rather than silently loading.
            scfoundation_schema_version=np.asarray(2),
            expression_content_hash=np.asarray(
                _expression_content_hash(
                    expression, raw_library_size, row_batch_size=batch_size,
                )
            ),
        )
    os.replace(tmp, path)
    return path


def load_scfoundation_spot_features(
    cache_root: str | Path,
    sample_id: str,
    barcodes: np.ndarray,
    gene_panel_hash: str,
    expression,
    raw_library_size: np.ndarray | None = None,
) -> dict:
    """`expression` must be the SAME live `n_spots x n_genes` matrix (row-
    aligned with `barcodes`) that would be fed into the encoder right now.
    Codex audit finding: cache identity previously keyed only on
    (sample_id, barcodes, gene_panel_hash) -- none of which change when
    the underlying expression VALUES change (a re-run QC fix, a corrected
    count, a different normalization), so a stale cache could be reused
    silently. Now fails closed on any content mismatch instead.
    `raw_library_size`, when the cache was built with one, must be passed
    here too (same row-aligned array) -- see `_expression_content_hash`'s
    own docstring (Item 6)."""
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
    live_hash = _expression_content_hash(expression, raw_library_size)
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
            "preprocessing_spec": str(cached["scfoundation_preprocessing_spec"]),
            "output_dim": output_dim,
            "schema_version": int(np.asarray(cached["scfoundation_schema_version"]).item()),
        },
    }


def barcode_embedding_lookup(cached: dict) -> dict[str, np.ndarray]:
    """Convert a loaded cache dict into the `{barcode: row}` mapping
    `gen4.inputs.build_gen4_spatial_field_example`'s `gex_context_embedding`
    argument expects."""
    return {str(b): cached["features"][i] for i, b in enumerate(cached["barcodes"])}
