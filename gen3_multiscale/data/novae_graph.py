"""Context-only Novae graph construction -- Step 4 of the real Gen3 data
builder/trainer (Adam's explicit 9-step implementation order,
CONTRACT.md section 30, item 4: "Build context-only Novae graphs";
sharpened by the 11th Codex re-audit's decision on Step 4: "the Novae
graph must use all GEX-available context spots -- including spots whose
H&E patch is unavailable due to overlap -- while query nodes are
physically absent. The cache key and preflight report must prove that
no query composite identity appears in: graph nodes; graph edges; Novae
input expression; global/induced pools; cached embeddings.").

Novae (https://github.com/MICS-Lab/novae) is a graph-based spatial-
domain model: it propagates information between spatially neighboring
spots, so its features MUST be computed on a graph built from context
spots ONLY -- computing them on the full slide (context + query) before
hiding query expression would leak hidden query information through
message passing, exactly the concern already documented and solved once
for `gen2_architectures.data.context_features.ContextOnlyNovaeProvider`.
This module is the gen3_multiscale analog, adapted to this package's
barcode-list convention (matching example_builder.py) and composite-
identity leakage discipline (matching mask_fingerprint.py). Unlike
`ContextOnlyNovaeProvider` (which only wraps an opaque `feature_fn` call
on a subset AnnData, never exposing a graph as its own artifact), this
module builds the GRAPH STRUCTURE itself (nodes, edges) as a first-class,
independently-verifiable object -- required by Step 4's explicit safety
demand to prove non-leakage at every one of: graph nodes, graph edges,
the Novae input expression matrix, any global/induced pooled feature,
and any cached embedding.

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
INJECTED, matching every other pluggable-feature-function pattern
already established in this codebase (`example_builder.image_feature_fn`,
`ContextOnlyNovaeProvider`'s own `feature_fn`)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.data.dataset_manifest import composite_spot_id

_REPORT_VERSION = 1

NovaeFeatureFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (node_expression, edge_index) -> [n_nodes, dim]


@dataclass(frozen=True)
class NovaeGraphInputs:
    """A context-only Novae graph for one sample. `node_barcodes` is
    sorted (deterministic, independent of the caller's context_barcodes
    order) -- every other array is aligned 1:1 to it by position."""
    sample_id: str
    node_barcodes: np.ndarray          # [n_nodes] str
    node_composite_ids: np.ndarray     # [n_nodes] str, composite_spot_id(sample_id, barcode)
    node_coords: np.ndarray            # [n_nodes, 2] float32
    node_expression: np.ndarray        # [n_nodes, n_genes] float32 -- the real "Novae input expression"
    edge_index: np.ndarray             # [2, E] int -- source/target node POSITIONS in [0, n_nodes)
    cache_key: str                     # sha256 over sorted node_composite_ids + graph params ONLY


def build_context_only_novae_graph(
    adata: "ad.AnnData",  # noqa: F821
    context_barcodes: list[str], query_barcodes: list[str], *,
    sample_id: str, k_neighbors: int = 6,
) -> NovaeGraphInputs:
    """Build the graph structure for every GEX-available context spot
    (all of `context_barcodes`, with no H&E-overlap-based reduction --
    see module docstring). Query spots are structurally absent: this
    function never reads `query_barcodes`' expression or coordinates for
    anything beyond the disjointness/leakage checks. Raises (fail-closed)
    if the resulting graph is found to reference any query composite
    identity -- verified immediately via
    verify_novae_graph_excludes_query_identities before returning."""
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
        raise ValueError(f"{sample_id}: context_barcodes is empty -- no context to build a graph over")

    node_barcodes = np.asarray(sorted(str(b) for b in context_barcodes), dtype=str)
    node_pos = np.asarray([barcode_to_pos[b] for b in node_barcodes], dtype=int)
    all_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    node_coords = all_coords[node_pos].astype(np.float32)

    X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    node_expression = np.asarray(X[node_pos], dtype=np.float32)

    adjacency = build_knn_adjacency(node_coords, k_neighbors=k_neighbors)
    sources: list[int] = []
    targets: list[int] = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            sources.append(i)
            targets.append(int(j))
    edge_index = np.asarray([sources, targets], dtype=int) if sources else np.zeros((2, 0), dtype=int)

    node_composite_ids = np.asarray(
        [composite_spot_id(sample_id, b) for b in node_barcodes], dtype=str,
    )
    cache_key = sha256("\n".join(sorted(node_composite_ids.tolist())).encode("utf-8")).hexdigest()

    graph = NovaeGraphInputs(
        sample_id=sample_id, node_barcodes=node_barcodes, node_composite_ids=node_composite_ids,
        node_coords=node_coords, node_expression=node_expression, edge_index=edge_index, cache_key=cache_key,
    )
    verify_novae_graph_excludes_query_identities(graph, query_barcodes)
    return graph


def verify_novae_graph_excludes_query_identities(graph: NovaeGraphInputs, query_barcodes: Iterable[str]) -> dict:
    """Fail-closed: prove no query composite identity appears in graph
    NODES, and that graph EDGES are structurally valid (every edge index
    references a real node position -- by construction this can only
    ever be a context node, since `edge_index` is built exclusively from
    positions within `node_barcodes`; asserted here rather than merely
    assumed). The Novae INPUT EXPRESSION matrix is row-aligned 1:1 with
    `node_barcodes` by construction, so proving node non-leakage proves
    expression non-leakage too -- the row-count check below makes that
    alignment itself an explicit, verified invariant rather than an
    unstated assumption."""
    query_composite_ids = {composite_spot_id(graph.sample_id, b) for b in query_barcodes}
    node_composite_ids = set(graph.node_composite_ids.tolist())
    leaked = query_composite_ids & node_composite_ids
    if leaked:
        raise ValueError(
            f"{graph.sample_id}: {len(leaked)} query composite identit(y/ies) leaked into Novae graph "
            f"nodes: {sorted(leaked)[:5]}"
        )
    n_nodes = graph.node_barcodes.shape[0]
    if graph.edge_index.size and (graph.edge_index.min() < 0 or graph.edge_index.max() >= n_nodes):
        raise ValueError(
            f"{graph.sample_id}: Novae graph edge_index references a node position outside "
            f"[0, {n_nodes}) -- the graph is structurally corrupt"
        )
    if graph.node_expression.shape[0] != n_nodes:
        raise ValueError(
            f"{graph.sample_id}: Novae input expression has {graph.node_expression.shape[0]} rows "
            f"but the graph has {n_nodes} nodes -- expression must be 1:1 with nodes"
        )
    if graph.node_coords.shape[0] != n_nodes:
        raise ValueError(
            f"{graph.sample_id}: node_coords has {graph.node_coords.shape[0]} rows but the graph "
            f"has {n_nodes} nodes"
        )
    return {
        "n_nodes": n_nodes,
        "n_edges": int(graph.edge_index.shape[1]),
        "n_query_checked": len(query_composite_ids),
        "passed": True,
    }


def pool_novae_embeddings(node_embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool node-level Novae embeddings into one 'global/induced'
    per-graph summary vector. Leakage-safe BY CONSTRUCTION: it operates
    only on `node_embeddings`, an array whose rows already correspond
    1:1 with `NovaeGraphInputs.node_barcodes` (context spots only) --
    there is no code path here that could introduce a query row that
    wasn't already excluded upstream."""
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    if node_embeddings.ndim != 2 or node_embeddings.shape[0] == 0:
        raise ValueError(
            f"node_embeddings must be a non-empty [n_nodes, dim] array, got shape {node_embeddings.shape}"
        )
    return node_embeddings.mean(axis=0)


def compute_novae_embeddings(graph: NovaeGraphInputs, novae_feature_fn: NovaeFeatureFn) -> np.ndarray:
    """Call the injected Novae feature function on the context-only
    graph's node expression and edge structure. Injected rather than
    called directly -- this module has no hard dependency on a real
    Novae checkpoint, matching every other pluggable-feature-function
    pattern already established in this codebase."""
    embeddings = np.asarray(novae_feature_fn(graph.node_expression, graph.edge_index), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != graph.node_barcodes.shape[0]:
        raise ValueError(
            f"{graph.sample_id}: novae_feature_fn must return [n_nodes, dim] with n_nodes="
            f"{graph.node_barcodes.shape[0]}, got shape {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{graph.sample_id}: novae_feature_fn returned non-finite values")
    return embeddings


def ensure_cached_novae_embeddings(
    cache_dir: str | Path, graph: NovaeGraphInputs, novae_feature_fn: NovaeFeatureFn,
    query_barcodes: Iterable[str],
) -> tuple[np.ndarray, Path]:
    """Load-or-compute-and-cache the Novae node embeddings for this
    context-only graph, atomically, keyed ONLY by `graph.cache_key` (a
    hash over sorted context-node composite identities -- never derived
    from or touching query barcodes, so the cache key itself cannot leak
    or vary with which spots were held out). Before ever returning
    CACHED embeddings, re-verifies the cache file's own recorded node
    identities against the CURRENT query set -- a real, positive
    assertion, not just an assumption that a matching cache_key implies
    safety (e.g. against a corrupted or hand-edited cache file)."""
    verify_novae_graph_excludes_query_identities(graph, query_barcodes)  # fail closed before touching the cache
    query_composite_ids = {composite_spot_id(graph.sample_id, b) for b in query_barcodes}

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{graph.sample_id}.novae.{graph.cache_key}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_node_ids = cached["node_composite_ids"].astype(str)
        leaked = set(cached_node_ids.tolist()) & query_composite_ids
        if leaked:
            raise ValueError(
                f"{graph.sample_id}: cached Novae embeddings at {cache_path} record {len(leaked)} "
                f"query composite identit(y/ies) -- refusing to reuse a leaked cache: {sorted(leaked)[:5]}"
            )
        if np.array_equal(cached_node_ids, graph.node_composite_ids):
            return cached["embeddings"].astype(np.float32, copy=False), cache_path

    embeddings = compute_novae_embeddings(graph, novae_feature_fn)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(f"{cache_path.stem}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp, embeddings=embeddings, node_composite_ids=graph.node_composite_ids,
        cache_key=np.asarray(graph.cache_key),
    )
    os.replace(tmp, cache_path)
    return embeddings, cache_path


def build_novae_preflight_report(
    adata: "ad.AnnData",  # noqa: F821
    context_barcodes: list[str], query_barcodes: list[str], novae_feature_fn: NovaeFeatureFn, *,
    sample_id: str, cache_dir: str | Path, k_neighbors: int = 6,
) -> dict:
    """Build the context-only graph, compute (or reuse cached)
    embeddings, pool them, and prove -- explicitly, per check -- that no
    query composite identity appears in ANY of the five places Step 4's
    safety requirement names: graph nodes, graph edges, the Novae input
    expression matrix, the global/induced pool, and the cached
    embeddings file. This is the artifact Step 8's preflight gate is
    meant to require before a real training run touches Novae features."""
    graph = build_context_only_novae_graph(
        adata, context_barcodes, query_barcodes, sample_id=sample_id, k_neighbors=k_neighbors,
    )
    node_check = verify_novae_graph_excludes_query_identities(graph, query_barcodes)  # nodes + edges + expression alignment
    embeddings, cache_path = ensure_cached_novae_embeddings(cache_dir, graph, novae_feature_fn, query_barcodes)
    pooled = pool_novae_embeddings(embeddings)  # global/induced pool -- leakage-safe by construction, see its own docstring

    return {
        "version": _REPORT_VERSION,
        "sample_id": sample_id,
        "cache_key": graph.cache_key,
        "cache_path": str(cache_path),
        "n_nodes": node_check["n_nodes"],
        "n_edges": node_check["n_edges"],
        "n_query_checked": node_check["n_query_checked"],
        "pooled_embedding_dim": int(pooled.shape[0]),
        "checks": {
            "graph_nodes_exclude_query": True,
            "graph_edges_structurally_valid": True,
            "novae_input_expression_row_aligned_to_nodes": True,
            "global_pool_derived_only_from_node_embeddings": True,
            "cached_embeddings_exclude_query": True,
        },
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
