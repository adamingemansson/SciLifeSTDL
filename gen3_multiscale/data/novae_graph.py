"""Context-only Novae input construction -- Step 4 of the real Gen3 data
builder/trainer (Adam's explicit 9-step implementation order,
CONTRACT.md section 30, item 4: "Build context-only Novae graphs";
sharpened by the 11th Codex re-audit's decision on Step 4, then
corrected by the 12th Codex re-audit of commit 1bb66d6 finding #1).

**The 12th re-audit's finding #1 (CONFIRMED against Novae's real
published quickstart, github.com/MICS-Lab/novae) was a real design bug
in the first version of this module, not a style nit.** Real Novae's
interface is AnnData-in, AnnData-mutated-in-place-out:
```
novae.spatial_neighbors(adata)
model = novae.Novae.from_pretrained("prism-oncology/novae-human-0")
model.compute_representations(adata, zero_shot=True)
```
`novae.spatial_neighbors` builds Novae's OWN spatial graph internally
from `adata.obsm["spatial"]` -- it does not accept a caller-supplied
graph at all. A prior version of this module built its own separate
k-NN adjacency (`boundary_graph.build_knn_adjacency`) and fed
`(node_expression, edge_index)` to the injected feature function --
that graph was never proven to be the one real Novae would actually
build or use, and the function signature could not even accept real
Novae's real call pattern. `gen2_architectures.models.conditioning.precompute_novae_features`
already implements the REAL, correct adapter (confirmed by reading it
in full): it takes a whole `adata`, calls `novae.spatial_neighbors(adata)`
itself, loads a cached `Novae.from_pretrained(checkpoint)`, calls
`model.compute_representations(adata, zero_shot=True)`, and extracts
the new `obsm` key defensively. This module's job is therefore NOT to
build a competing graph -- it is to build the PHYSICALLY context-only
AnnData real Novae needs (ordered `var_names`, context-only `obs_names`,
context-only `obsm["spatial"]`, context-only `X`), prove it excludes
every query composite identity, and pass THAT to an injected adapter
function with the same `Callable[[adata], np.ndarray]` contract
`gen2_architectures.data.context_features.ContextOnlyNovaeProvider`
already uses and this project has already audited -- not a bespoke
`(expression, edge_index)` signature nobody could actually plug a real
Novae adapter into.

Since the context-only AnnData structurally excludes every query row,
and real Novae's OWN internal `spatial_neighbors` graph is built FROM
that exact AnnData (inside the injected function, not here), the graph
Novae's `spatial_neighbors` produces cannot contain a query node or
edge EITHER -- proving the input AnnData is query-free is what actually
proves the downstream graph is too. This module does not fabricate a
second, unverified graph structure to police instead.

**Cache fingerprinting (12th re-audit finding #2, CONFIRMED: the first
version's cache key hashed only context-node identities, ignoring
expression content, preprocessing, gene panel, coordinates, and the
feature function's own identity/checkpoint).** Fixed by reusing
`gen2_architectures.data.context_features._adata_feature_signature` --
the already-audited fingerprint (real matrix bytes, obs_names,
var_names, spatial coordinates, preprocessing state, and the feature
function's module-qualified name) this project has already relied on
for the exact same problem, rather than reinventing a weaker one.
`checkpoint_signature`/`extra_signature` lets a caller fold in a real
checkpoint identity (e.g. `f"{repo}:{revision}"` or a file hash) so a
changed checkpoint invalidates the cache even when the wrapping
Python function's name doesn't change -- exactly `_adata_feature_signature`'s
own documented `extra_signature` use case.

**Cached embeddings are now fully validated on load (12th re-audit
finding #3, CONFIRMED: the first version only checked node-identity
overlap with the query set, never shape/dtype/finiteness/width)**: see
`ensure_cached_novae_embeddings`.

**The preflight report is now honest about what it can and cannot prove
(12th re-audit finding #4, CONFIRMED: a prior "checks" dict asserted
`True` for claims like "an arbitrary injected function didn't leak
external data," which this module has no way to actually verify)**: see
`build_novae_preflight_report`'s `verified` vs `not_provable_from_this_module_alone`
split.

GEX availability, not H&E availability, is what this module reads:
`example_builder.build_spatial_field_example` (Step 2) EXCLUDES context
spots whose H&E patch physically overlaps the query hole from
`observed_*` entirely (both modalities) -- a real, currently-accepted
scope gap this project's own audit trail flags as needing a
per-modality `image_available` fix in Step 5 (CONTRACT.md section 33,
finding #4). Novae operates on GEX alone and has no H&E-overlap
constraint, so this module reads directly from the REALIZED mask's
`context_barcodes` (physically disjoint from `query_barcodes` -- the
ONLY real constraint) rather than from a SpatialFieldInputs' already
H&E-filtered `observed_barcodes` -- using every GEX-available context
spot, exactly as instructed.

The actual Novae model/checkpoint is never called here (no such
dependency exists anywhere in this repo -- confirmed: no `novae` or
`torch_geometric` package is installed). `novae_feature_fn` is
INJECTED and receives the context-only AnnData ITSELF, never the full
adata.

**Sanitized construction (13th Codex re-audit of commit 65611c7,
finding #1, CONFIRMED).** That AnnData is now built EXPLICITLY,
field-by-field -- `X` (row-subset), `var` (gene identities Novae needs
to match its own vocabulary by name), `obsm['spatial']` (row-subset),
and only an explicit whitelist of `.uns` keys (`_ALLOWED_CONTEXT_UNS_KEYS`
below, exactly the keys `_adata_feature_signature` itself reads) --
rather than via `adata[node_pos].copy()`'s blanket inheritance of
every `.obsm`/`.obsp`/`.layers`/`.uns`/`.raw` entry on the original
object. Array-shaped obs-indexed fields ARE correctly row-subset by
that blanket copy, but arbitrary non-obs-indexed `.uns` entries (e.g.
a full-slide summary statistic computed once and stashed in `.uns`)
are not meaningfully "subset" at all and would have carried forward
into the context-only object unchanged -- a real, if narrow, leakage
surface the prior version left open. This module cannot prevent an
injected `novae_feature_fn` from independently capturing the original
`adata` via its own Python closure -- no callee can ever police what a
caller's closure captures -- which is why that specific claim is
listed under `not_provable_from_this_module_alone` in
`build_novae_preflight_report`, not asserted as verified."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import anndata as ad
import numpy as np
import pandas as pd

from gen2_architectures.data.context_features import _adata_feature_signature
from gen3_multiscale.data.dataset_manifest import composite_spot_id

_REPORT_VERSION = 2

# Bumped whenever THIS module's cache file schema (required keys,
# meaning of stored fields) changes -- folded into the cache fingerprint
# via _adata_feature_signature's extra_signature so an old-schema cache
# file is never misread as the new schema, even if every other input is
# unchanged.
_CACHE_SCHEMA_VERSION = "1"

# The ONLY .uns keys carried forward into a context-only AnnData (13th
# Codex re-audit finding #1) -- exactly the keys
# gen2_architectures.data.context_features._adata_feature_signature
# itself reads (`expression_state` / `legacy_expression_preprocessing`
# there). Anything else in the original adata's .uns is real-slide
# metadata with no obs-indexed subsetting semantics and must never be
# assumed safe to carry forward into a context-only object.
_ALLOWED_CONTEXT_UNS_KEYS = ("_scilifestdl_expression_state", "expression_preprocessing")

# Required keys of a structured Novae checkpoint provenance dict (13th
# Codex re-audit finding #7, CONFIRMED: an optional, default-empty
# `checkpoint_signature: str = ""` meant a caller could silently omit
# real checkpoint identity and still get a cache hit/preflight pass).
# `checkpoint_revision` may be a real repo revision (for a
# HuggingFace-style `checkpoint_repo`) OR a local checkpoint file's own
# SHA256 -- either way it must be a real, non-empty identity string, not
# a guess. `spatial_neighbors_settings` records whatever real settings
# (e.g. technology, n_neighs) the caller's adapter passes to Novae's own
# `spatial_neighbors(adata)` call, since that call's OWN graph
# construction is part of what determines the embeddings, not just the
# model checkpoint.
_REQUIRED_CHECKPOINT_PROVENANCE_KEYS = (
    "checkpoint_repo", "checkpoint_revision", "novae_package_version",
    "adapter_version", "spatial_neighbors_settings",
)

NovaeFeatureFn = Callable[[object], np.ndarray]  # (context_only_adata) -> [n_context, dim]


def _validate_checkpoint_provenance(checkpoint_provenance: dict) -> str:
    """Validate a REQUIRED structured Novae checkpoint provenance dict
    and return its canonical (sorted-key JSON) signature string for
    folding into the cache fingerprint. Fails closed on a missing dict,
    a missing required key, or an empty/whitespace-only value for a
    required key -- this module has no way to verify the CONTENT of a
    caller-supplied identity string is actually correct (that's why it
    stays under `not_provable_from_this_module_alone`), but it can and
    does refuse to accept the absence of one entirely."""
    if not isinstance(checkpoint_provenance, dict):
        raise TypeError(
            f"checkpoint_provenance must be a dict with keys {_REQUIRED_CHECKPOINT_PROVENANCE_KEYS}, "
            f"got {type(checkpoint_provenance).__name__}"
        )
    missing = [k for k in _REQUIRED_CHECKPOINT_PROVENANCE_KEYS if k not in checkpoint_provenance]
    if missing:
        raise ValueError(f"checkpoint_provenance is missing required key(s): {missing}")
    empty = [k for k in _REQUIRED_CHECKPOINT_PROVENANCE_KEYS if not str(checkpoint_provenance[k]).strip()]
    if empty:
        raise ValueError(f"checkpoint_provenance has empty/whitespace value(s) for required key(s): {empty}")
    canonical = {k: str(checkpoint_provenance[k]) for k in _REQUIRED_CHECKPOINT_PROVENANCE_KEYS}
    return json.dumps(canonical, sort_keys=True)


@dataclass(frozen=True)
class NovaeContextInputs:
    """The physically context-only AnnData real Novae needs, plus
    composite-identity bookkeeping for leakage verification.
    `node_barcodes` is sorted (deterministic, independent of the
    caller's context_barcodes order); `context_adata.obs_names` is
    identical to it by construction."""
    sample_id: str
    node_barcodes: np.ndarray          # [n_context] str, sorted
    node_composite_ids: np.ndarray     # [n_context] str, composite_spot_id(sample_id, barcode)
    context_adata: object              # "ad.AnnData" -- physically excludes every query row


def _build_sanitized_context_adata(
    adata: "ad.AnnData", node_barcodes: np.ndarray, node_pos: list[int],
) -> "ad.AnnData":
    """Build a genuinely NEW, EXPLICITLY-constructed context-only AnnData
    (13th Codex re-audit finding #1) -- never `adata[node_pos].copy()`,
    which blanket-inherits every `.obsm`/`.obsp`/`.layers`/`.uns`/`.raw`
    entry from the original object. Array-shaped obs-indexed fields ARE
    correctly row-subset by that blanket copy, but arbitrary
    non-obs-indexed `.uns` entries have no subsetting semantics at all
    and would carry forward unchanged -- a real leakage surface for a
    full-slide summary statistic stashed in `.uns`.

    Only three things are carried forward, all explicitly row/identity
    subset: `X` (expression), `var` (gene IDENTITY ONLY -- see below),
    and `obsm['spatial']` (coordinates, the only spatial input real
    `novae.spatial_neighbors` reads). `.uns` is populated ONLY from
    `_ALLOWED_CONTEXT_UNS_KEYS` -- exactly the keys
    `_adata_feature_signature` itself reads, so the cache fingerprint
    still reflects real preprocessing state. `.obsp`, `.layers`, `.raw`,
    and every other `.uns` key are never copied -- if a real
    `novae_feature_fn` adapter needs something beyond this, that is a
    disclosed, real integration gap (`build_novae_preflight_report`'s
    `not_provable_from_this_module_alone`), not something silently
    assumed safe to smuggle through.

    14th Codex re-audit of commit 8d4e276, finding #1 (CONFIRMED): `var`
    used to be `adata.var.copy()` -- copying every column, not just gene
    identity. Scanpy/AnnData conventionally stores full-slide-derived
    per-gene statistics in `.var` (detection counts, means, dispersion/
    variability), computed over EVERY spot including query spots -- a
    real leakage surface identical in kind to the `.uns` one above, just
    on the gene axis instead of the observation axis. Fixed: `var` is
    now an INDEX-ONLY DataFrame built straight from `adata.var_names`
    (gene identity strings, the only thing Novae needs to match its own
    vocabulary by name), never `adata.var`'s other columns."""
    X = adata.X[node_pos]
    X = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)
    obs_index_name = adata.obs.index.name if hasattr(adata.obs, "index") else None
    var_index_name = adata.var.index.name if hasattr(adata.var, "index") else None
    context_adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(node_barcodes, name=obs_index_name)),
        var=pd.DataFrame(index=pd.Index(np.asarray(adata.var_names, dtype=str), name=var_index_name)),
    )
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required to build a context-only Novae input")
    context_adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"])[node_pos].copy()
    source_uns = getattr(adata, "uns", {})
    for key in _ALLOWED_CONTEXT_UNS_KEYS:
        if key in source_uns:
            context_adata.uns[key] = source_uns[key]
    return context_adata


def build_context_only_novae_input(
    adata: "ad.AnnData",  # noqa: F821
    context_barcodes: list[str], query_barcodes: list[str], *, sample_id: str,
) -> NovaeContextInputs:
    """Build the PHYSICALLY context-only AnnData real Novae needs (real
    `var_names`/gene identities, context-only `obs_names`, context-only
    `obsm['spatial']`, context-only `X`) -- not a fabricated graph. Query
    spots are structurally absent: this function never reads
    `query_barcodes`' expression or coordinates for anything beyond the
    disjointness/leakage checks. Raises (fail-closed) if the result is
    found to reference any query composite identity -- verified
    immediately via `verify_novae_context_excludes_query_identities`
    before returning."""
    obs_names = np.asarray(adata.obs_names, dtype=str)
    barcode_to_pos = {b: i for i, b in enumerate(obs_names)}

    missing_context = [b for b in context_barcodes if b not in barcode_to_pos]
    missing_query = [b for b in query_barcodes if b not in barcode_to_pos]
    if missing_context or missing_query:
        raise ValueError(
            f"{sample_id}: mask references barcodes absent from the aligned sample data -- "
            f"context missing {missing_context[:5]}, query missing {missing_query[:5]}"
        )
    if set(context_barcodes) & set(query_barcodes):
        raise ValueError(f"{sample_id}: context and query barcodes overlap")
    if not context_barcodes:
        raise ValueError(f"{sample_id}: context_barcodes is empty -- no context to build Novae input from")

    node_barcodes = np.asarray(sorted(str(b) for b in context_barcodes), dtype=str)
    node_pos = [barcode_to_pos[b] for b in node_barcodes]
    context_adata = _build_sanitized_context_adata(adata, node_barcodes, node_pos)
    node_composite_ids = np.asarray(
        [composite_spot_id(sample_id, b) for b in node_barcodes], dtype=str,
    )

    inputs = NovaeContextInputs(
        sample_id=sample_id, node_barcodes=node_barcodes, node_composite_ids=node_composite_ids,
        context_adata=context_adata,
    )
    verify_novae_context_excludes_query_identities(inputs, query_barcodes)
    return inputs


def verify_novae_context_excludes_query_identities(
    inputs: NovaeContextInputs, query_barcodes: Iterable[str],
) -> dict:
    """Fail-closed: prove no query composite identity appears among the
    context-only AnnData's own `obs_names`, and that `node_barcodes`/
    `node_composite_ids`/`context_adata.obs_names` are all exactly
    aligned (so a bug that desynchronized them, rather than the
    subsetting itself, would also be caught here, not silently
    trusted)."""
    query_composite_ids = {composite_spot_id(inputs.sample_id, b) for b in query_barcodes}
    node_composite_ids = set(inputs.node_composite_ids.tolist())
    leaked = query_composite_ids & node_composite_ids
    if leaked:
        raise ValueError(
            f"{inputs.sample_id}: {len(leaked)} query composite identit(y/ies) leaked into the "
            f"context-only Novae input: {sorted(leaked)[:5]}"
        )
    context_obs_names = np.asarray(inputs.context_adata.obs_names, dtype=str)
    if not np.array_equal(context_obs_names, inputs.node_barcodes):
        raise ValueError(
            f"{inputs.sample_id}: context_adata.obs_names does not match node_barcodes -- the "
            "context-only AnnData is desynchronized from its own identity bookkeeping"
        )
    query_barcode_set = {str(b) for b in query_barcodes}
    leaked_barcodes = set(context_obs_names.tolist()) & query_barcode_set
    if leaked_barcodes:
        raise ValueError(
            f"{inputs.sample_id}: context_adata itself contains {len(leaked_barcodes)} query "
            f"barcode(s): {sorted(leaked_barcodes)[:5]} -- this must structurally never happen"
        )
    return {
        "n_context": int(context_obs_names.shape[0]),
        "n_query_checked": len(query_composite_ids),
        "passed": True,
    }


def pool_novae_embeddings(node_embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool node-level Novae embeddings into one 'global/induced'
    per-sample summary vector. Leakage-safe BY CONSTRUCTION: it operates
    only on `node_embeddings`, an array whose rows already correspond
    1:1 with the context-only AnnData's `obs_names` -- there is no code
    path here that could introduce a query row that wasn't already
    excluded upstream."""
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    if node_embeddings.ndim != 2 or node_embeddings.shape[0] == 0:
        raise ValueError(
            f"node_embeddings must be a non-empty [n_context, dim] array, got shape {node_embeddings.shape}"
        )
    return node_embeddings.mean(axis=0)


def compute_novae_embeddings(inputs: NovaeContextInputs, novae_feature_fn: NovaeFeatureFn) -> np.ndarray:
    """Call the injected Novae adapter on the context-only AnnData --
    the SAME object built by `build_context_only_novae_input`, never a
    closure capturing the original full `adata`. Injected rather than
    called directly: this module has no hard dependency on a real
    Novae checkpoint, matching `ContextOnlyNovaeProvider`'s own
    `feature_fn` contract exactly (`Callable[[adata], np.ndarray]`), so
    a real adapter (`gen2_architectures.models.conditioning.precompute_novae_features`,
    or an equivalent wrapping the real
    `novae.spatial_neighbors`/`Novae.from_pretrained`/`compute_representations`
    call sequence) can be plugged in directly without adapting its
    signature."""
    embeddings = np.asarray(novae_feature_fn(inputs.context_adata), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != inputs.node_barcodes.shape[0]:
        raise ValueError(
            f"{inputs.sample_id}: novae_feature_fn must return [n_context, dim] with n_context="
            f"{inputs.node_barcodes.shape[0]}, got shape {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{inputs.sample_id}: novae_feature_fn returned non-finite values")
    return embeddings


def ensure_cached_novae_embeddings(
    cache_dir: str | Path, inputs: NovaeContextInputs, novae_feature_fn: NovaeFeatureFn,
    query_barcodes: Iterable[str], *, expected_dim: int | None = None, checkpoint_provenance: dict,
) -> tuple[np.ndarray, Path]:
    """Load-or-compute-and-cache the Novae embeddings for this
    context-only input, atomically. The cache key is
    `_adata_feature_signature(context_adata, novae_feature_fn,
    extra_signature=...)` -- a COMPREHENSIVE fingerprint over real
    expression bytes, obs/var names, spatial coordinates, preprocessing
    state, the feature function's identity, `checkpoint_provenance` (a
    REQUIRED structured dict -- `checkpoint_repo`, `checkpoint_revision`,
    `novae_package_version`, `adapter_version`, `spatial_neighbors_settings`,
    validated by `_validate_checkpoint_provenance`; 13th Codex re-audit
    finding #7, CONFIRMED: an optional default-empty-string signature let
    a caller silently omit real checkpoint identity and still get a
    cache hit), and this module's own cache-schema version -- never just
    context-node identities alone (12th Codex re-audit finding #2).

    Before ever returning a CACHED result, this function validates: the
    cache file has every required key; its recorded `cache_key` matches
    exactly; its recorded node identities/order match the CURRENT
    context exactly (never just "no query leak", which alone doesn't
    prove it's even the RIGHT context); its embeddings array has the
    correct row count, is 2D, is `float32`, is all-finite, and (if
    `expected_dim` is supplied) has the expected width (12th Codex
    re-audit finding #3) -- any failure is a hard refusal to reuse the
    cache, not a silent fallback to treating it as a miss."""
    verify_novae_context_excludes_query_identities(inputs, query_barcodes)  # fail closed before touching the cache
    query_composite_ids = {composite_spot_id(inputs.sample_id, b) for b in query_barcodes}
    checkpoint_signature = _validate_checkpoint_provenance(checkpoint_provenance)

    cache_key = _adata_feature_signature(
        inputs.context_adata, novae_feature_fn,
        extra_signature=f"gen3-novae-cache-schema-v{_CACHE_SCHEMA_VERSION}:{checkpoint_signature}",
    )
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{inputs.sample_id}.novae.{cache_key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        required_keys = {"embeddings", "node_composite_ids", "cache_key"}
        missing_keys = required_keys - set(cached.files)
        if missing_keys:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} are missing required "
                f"key(s) {sorted(missing_keys)} -- refusing to trust a malformed cache file"
            )
        cached_cache_key = str(cached["cache_key"].item())
        if cached_cache_key != cache_key:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} record cache_key "
                f"{cached_cache_key!r}, expected {cache_key!r} -- refusing a mismatched cache"
            )
        cached_node_ids = cached["node_composite_ids"].astype(str)
        leaked = set(cached_node_ids.tolist()) & query_composite_ids
        if leaked:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} record {len(leaked)} "
                f"query composite identit(y/ies) -- refusing to reuse a leaked cache: {sorted(leaked)[:5]}"
            )
        if not np.array_equal(cached_node_ids, inputs.node_composite_ids):
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} record a different "
                "node identity/order than the current context -- refusing a mismatched cache"
            )
        embeddings = cached["embeddings"]
        n_context = inputs.node_barcodes.shape[0]
        if embeddings.ndim != 2 or embeddings.shape[0] != n_context:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} have shape "
                f"{embeddings.shape}, expected ({n_context}, dim) -- refusing a malformed cache"
            )
        if expected_dim is not None and embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} have width "
                f"{embeddings.shape[1]}, expected {expected_dim} -- refusing a stale/mismatched cache"
            )
        if embeddings.dtype != np.float32:
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} have dtype "
                f"{embeddings.dtype}, expected float32 -- refusing a malformed cache"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(
                f"{inputs.sample_id}: cached Novae embeddings at {cache_path} contain non-finite "
                "values -- refusing a corrupted cache"
            )
        return embeddings.astype(np.float32, copy=False), cache_path

    embeddings = compute_novae_embeddings(inputs, novae_feature_fn)
    if expected_dim is not None and embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"{inputs.sample_id}: novae_feature_fn returned width {embeddings.shape[1]}, expected "
            f"{expected_dim}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp, embeddings=embeddings, node_composite_ids=inputs.node_composite_ids,
        cache_key=np.asarray(cache_key),
    )
    os.replace(tmp, cache_path)
    return embeddings, cache_path


def build_novae_preflight_report(
    adata: "ad.AnnData",  # noqa: F821
    context_barcodes: list[str], query_barcodes: list[str], novae_feature_fn: NovaeFeatureFn, *,
    sample_id: str, cache_dir: str | Path, checkpoint_provenance: dict, expected_dim: int | None = None,
) -> dict:
    """Build the context-only Novae input, compute (or reuse cached)
    embeddings, pool them, and record what was ACTUALLY verified versus
    what this module cannot prove on its own (12th Codex re-audit
    finding #4: a prior version's "checks" dict asserted `True` for
    claims -- like "an arbitrary injected function didn't leak external
    data" -- that no amount of checking the INPUT/OUTPUT shape here can
    actually establish). `checkpoint_provenance` is REQUIRED (13th Codex
    re-audit finding #7) -- see `_validate_checkpoint_provenance` for its
    schema. This is the artifact Step 8's preflight gate is meant to
    require before a real training run touches Novae features."""
    inputs = build_context_only_novae_input(adata, context_barcodes, query_barcodes, sample_id=sample_id)
    context_check = verify_novae_context_excludes_query_identities(inputs, query_barcodes)
    embeddings, cache_path = ensure_cached_novae_embeddings(
        cache_dir, inputs, novae_feature_fn, query_barcodes,
        expected_dim=expected_dim, checkpoint_provenance=checkpoint_provenance,
    )
    pooled = pool_novae_embeddings(embeddings)

    return {
        "version": _REPORT_VERSION,
        "sample_id": sample_id,
        "cache_path": str(cache_path),
        "checkpoint_provenance": {k: str(checkpoint_provenance[k]) for k in _REQUIRED_CHECKPOINT_PROVENANCE_KEYS},
        "n_context": context_check["n_context"],
        "n_query_checked": context_check["n_query_checked"],
        "pooled_embedding_dim": int(pooled.shape[0]),
        "verified": [
            "context_adata.obs_names contains no query composite identity or query barcode "
            "(checked directly against the AnnData object actually passed to novae_feature_fn)",
            "context_adata.obs_names, node_barcodes, and node_composite_ids are mutually consistent",
            "the returned embeddings are row-aligned 1:1 with context_adata (shape checked)",
            "context_adata was built EXPLICITLY field-by-field (X/var/obsm['spatial']/whitelisted "
            "uns keys only), not by blanket-copying the full adata's .obsm/.obsp/.layers/.uns/.raw",
            "checkpoint_provenance has every required key (checkpoint_repo, checkpoint_revision, "
            "novae_package_version, adapter_version, spatial_neighbors_settings) non-empty",
            "if reused from cache: the cache's recorded cache_key, node identity/order, required "
            "keys, shape, dtype, and finiteness were all validated before trusting it",
        ],
        "not_provable_from_this_module_alone": [
            "that an arbitrary injected novae_feature_fn did not internally read data beyond the "
            "context_adata object it was given (this module cannot inspect the function's body)",
            "that cached or freshly-computed embeddings came from the exact real Novae checkpoint "
            "claimed, beyond whatever identity strings the caller supplied via checkpoint_provenance",
        ],
        "passed": True,
    }


def save_novae_preflight_report(report: dict, path: str | Path) -> Path:
    """Atomic write, mirroring save_dataset_manifest/save_mask_fingerprint_report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def load_novae_preflight_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
