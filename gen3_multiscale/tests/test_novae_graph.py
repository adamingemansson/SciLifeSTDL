"""Tests for context-only Novae input construction (Step 4 of the real
Gen3 data builder/trainer, CONTRACT.md section 37)."""
import numpy as np
import pandas as pd
import pytest
import anndata as ad

from gen3_multiscale.data.dataset_manifest import composite_spot_id
from gen3_multiscale.data.example_builder import build_spatial_field_example
from gen3_multiscale.data.novae_graph import (
    NovaeContextInputs, _validate_checkpoint_provenance, build_context_only_novae_input,
    build_novae_preflight_report, compute_novae_embeddings, ensure_cached_novae_embeddings,
    load_novae_preflight_report, pool_novae_embeddings, save_novae_preflight_report,
    verify_novae_context_excludes_query_identities,
)

_N_GENES = 5

_CHECKPOINT_PROVENANCE = {
    "checkpoint_repo": "prism-oncology/novae-human-0",
    "checkpoint_revision": "checkpoint-v1",
    "novae_package_version": "0.1.0",
    "adapter_version": "gen2_architectures.models.conditioning.precompute_novae_features-v1",
    "spatial_neighbors_settings": "default",
}


def _square_grid_adata(n_side: int = 6, spacing: float = 10.0, n_genes: int = _N_GENES, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    coords = np.array([[x * spacing, y * spacing] for x in range(n_side) for y in range(n_side)], dtype=np.float64)
    n = coords.shape[0]
    barcodes = [f"SPOT{i}-1" for i in range(n)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    counts = rng.poisson(5, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obsm["spatial"] = coords
    return adata


def _stub_novae_feature_fn(context_adata) -> np.ndarray:
    """Deterministic per-spot feature: the spot's own expression mean,
    tiled to a small feature width -- no real Novae checkpoint needed,
    matching every other pluggable-feature-function pattern already
    established in this codebase. Receives the REAL context-only
    AnnData, exactly like ContextOnlyNovaeProvider's feature_fn."""
    X = context_adata.X if isinstance(context_adata.X, np.ndarray) else context_adata.X.toarray()
    means = X.mean(axis=1)
    return np.tile(means[:, None], (1, 4)).astype(np.float32)


def test_build_context_only_novae_input_basic_shapes_and_disjointness():
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[14:16]
    context = [b for b in barcodes if b not in query]

    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")
    assert inputs.node_barcodes.tolist() == sorted(context)
    assert inputs.context_adata.n_obs == len(context)
    assert list(inputs.context_adata.var_names) == list(adata.var_names)
    assert inputs.context_adata.obsm["spatial"].shape == (len(context), 2)
    assert set(inputs.node_barcodes.tolist()).isdisjoint(set(query))


def test_build_context_only_novae_input_is_a_real_copy_not_a_view_of_the_full_adata():
    """context_adata is a genuinely independent object -- mutating it
    must never mutate the original adata. This does NOT (and cannot)
    prove an injected novae_feature_fn's own Python closure couldn't
    independently capture the original adata too -- no callee can
    police what a caller's closure captures, which is exactly why that
    specific claim lives under build_novae_preflight_report's
    `not_provable_from_this_module_alone`, not asserted here."""
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    inputs.context_adata.X[0, 0] = -12345.0
    assert adata.X[adata.obs_names.get_loc(inputs.node_barcodes[0]), 0] != -12345.0


def test_build_context_only_novae_input_excludes_unwhitelisted_uns_and_obsp_layers_raw():
    """13th Codex re-audit finding #1 regression test: a full-slide
    summary statistic stashed in adata.uns under a key OUTSIDE the
    whitelist, plus arbitrary .obsp/.layers entries, must never appear
    on the context-only AnnData -- only the whitelisted preprocessing
    keys (which _adata_feature_signature itself reads) and the
    explicitly-built X/var/obsm['spatial'] survive."""
    adata = _square_grid_adata()
    adata.uns["full_slide_summary_statistic"] = {"mean_total_counts": 12345.0}
    adata.uns["_scilifestdl_expression_state"] = {"kind": "raw"}
    adata.layers["some_extra_layer"] = adata.X.copy()
    adata.obsp["some_extra_graph"] = np.eye(adata.n_obs)

    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    assert "full_slide_summary_statistic" not in inputs.context_adata.uns
    assert inputs.context_adata.uns.get("_scilifestdl_expression_state") == {"kind": "raw"}
    assert dict(inputs.context_adata.layers) == {}
    assert dict(inputs.context_adata.obsp) == {}
    assert not hasattr(inputs.context_adata, "raw") or inputs.context_adata.raw is None


def test_build_context_only_novae_input_var_is_index_only_not_full_slide_gene_stats():
    """14th Codex re-audit of commit 8d4e276, finding #1 (CONFIRMED): a
    prior version copied `adata.var` wholesale (`adata.var.copy()`) --
    Scanpy/AnnData conventionally stores full-slide-derived per-gene
    statistics there (detection counts, means, dispersion/variability),
    computed over EVERY spot including query spots. Only gene IDENTITY
    (var_names) may survive into the context-only object."""
    adata = _square_grid_adata()
    adata.var["mean_counts_full_slide"] = np.arange(adata.n_vars, dtype=np.float64) * 1000.0
    adata.var["n_cells_by_counts"] = np.arange(adata.n_vars) + 1

    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    assert list(inputs.context_adata.var_names) == list(adata.var_names)
    assert list(inputs.context_adata.var.columns) == []
    assert "mean_counts_full_slide" not in inputs.context_adata.var.columns
    assert "n_cells_by_counts" not in inputs.context_adata.var.columns


def test_build_context_only_novae_input_includes_all_gex_available_context_spots_even_with_he_overlap():
    """The core requirement Step 4 exists to satisfy: the Novae input
    must include EVERY GEX-available context spot regardless of H&E
    availability -- Novae operates on GEX alone and has no H&E-overlap
    concept at all. (15th Codex re-audit, Step 5: build_spatial_field_example
    ALSO now retains every GEX-available spot -- see
    test_example_builder.py's *_flags_but_retains_* test -- but it
    additionally tracks per-spot H&E availability as an explicit flag,
    a concept Novae's context-only AnnData has no field for and does not
    need, since it never reads image features at all.)"""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = np.zeros((adata.n_obs, 4, 4, 3), dtype=np.uint8)
    barcodes = list(adata.obs_names)
    coords = adata.obsm["spatial"]

    query_idx = 14
    query_barcode = barcodes[query_idx]
    query_xy = coords[query_idx]
    distances = np.linalg.norm(coords - query_xy, axis=1)
    distances[query_idx] = np.inf
    neighbor_idx = int(np.argmin(distances))
    neighbor_barcode = barcodes[neighbor_idx]

    context = [b for b in barcodes if b != query_barcode]

    def _stub_image_feature_fn(p):
        return np.zeros((p.shape[0], 4), dtype=np.float32)

    example_inputs, _ = build_spatial_field_example(
        adata, patches, context, [query_barcode], _stub_image_feature_fn,
        sample_id="S0", patient_id="P0", patch_size_fullres=30.0, require_full_sample_coords=False,
    )
    assert neighbor_barcode in example_inputs.observed_barcodes.tolist()  # GEX retained
    neighbor_pos = example_inputs.observed_barcodes.tolist().index(neighbor_barcode)
    assert example_inputs.observed_image_available[neighbor_pos] == False  # noqa: E712 -- but H&E flagged unavailable

    novae_inputs = build_context_only_novae_input(adata, context, [query_barcode], sample_id="S0")
    assert neighbor_barcode in novae_inputs.node_barcodes.tolist()  # Novae has no H&E-availability concept at all


def test_build_context_only_novae_input_rejects_missing_barcodes():
    adata = _square_grid_adata()
    with pytest.raises(ValueError, match="absent from the aligned sample data"):
        build_context_only_novae_input(
            adata, ["NOT-A-REAL-BARCODE"], [list(adata.obs_names)[0]], sample_id="S0",
        )


def test_build_context_only_novae_input_rejects_context_query_overlap():
    adata = _square_grid_adata()
    shared = list(adata.obs_names)[:5]
    with pytest.raises(ValueError, match="overlap"):
        build_context_only_novae_input(adata, shared, shared[:1], sample_id="S0")


def test_build_context_only_novae_input_rejects_empty_context():
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    with pytest.raises(ValueError, match="no context to build Novae input from"):
        build_context_only_novae_input(adata, [], barcodes[:1], sample_id="S0")


def test_verify_novae_context_excludes_query_identities_detects_a_leaked_node():
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    # Tamper after construction (frozen dataclass, mutable numpy array
    # contents) to simulate a corrupted object and confirm the
    # verification function actually catches it.
    inputs.node_composite_ids[0] = composite_spot_id("S0", query[0])
    with pytest.raises(ValueError, match="leaked into the context-only Novae input"):
        verify_novae_context_excludes_query_identities(inputs, query)


def test_pool_novae_embeddings_mean_pools_and_rejects_empty():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pooled = pool_novae_embeddings(embeddings)
    assert np.allclose(pooled, [2.0, 3.0])
    with pytest.raises(ValueError, match="non-empty"):
        pool_novae_embeddings(np.zeros((0, 2), dtype=np.float32))


def test_compute_novae_embeddings_validates_shape_and_finiteness():
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    embeddings = compute_novae_embeddings(inputs, _stub_novae_feature_fn)
    assert embeddings.shape == (len(context), 4)

    with pytest.raises(ValueError, match="must return"):
        compute_novae_embeddings(inputs, lambda a: np.zeros((1, 4)))  # wrong row count

    def _nan_fn(a):
        out = _stub_novae_feature_fn(a)
        out[0, 0] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        compute_novae_embeddings(inputs, _nan_fn)


def test_ensure_cached_novae_embeddings_computes_then_reuses_from_cache(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    calls = []

    def _counting_fn(a):
        calls.append(1)
        return _stub_novae_feature_fn(a)

    embeddings_1, path = ensure_cached_novae_embeddings(
        tmp_path, inputs, _counting_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    assert path.is_file()
    assert len(calls) == 1

    embeddings_2, path_2 = ensure_cached_novae_embeddings(
        tmp_path, inputs, _counting_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    assert path_2 == path
    assert len(calls) == 1  # not recomputed
    assert np.array_equal(embeddings_1, embeddings_2)


def test_ensure_cached_novae_embeddings_rejects_a_cache_file_recording_a_query_identity(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    embeddings, path = ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )

    cached = dict(np.load(path, allow_pickle=False))
    tampered_ids = cached["node_composite_ids"].copy()
    tampered_ids[0] = composite_spot_id("S0", query[0])
    np.savez(
        path, embeddings=cached["embeddings"], node_composite_ids=tampered_ids, cache_key=cached["cache_key"],
    )

    with pytest.raises(ValueError, match="refusing to reuse a leaked cache"):
        ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )


def test_ensure_cached_novae_embeddings_rejects_a_cache_missing_required_keys(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    embeddings, path = ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    cached = dict(np.load(path, allow_pickle=False))
    np.savez(path, embeddings=cached["embeddings"])  # drop node_composite_ids/cache_key

    with pytest.raises(ValueError, match="missing required key"):
        ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )


def test_ensure_cached_novae_embeddings_rejects_a_wrong_dtype_cache(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    embeddings, path = ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    cached = dict(np.load(path, allow_pickle=False))
    np.savez(
        path, embeddings=cached["embeddings"].astype(np.float64), node_composite_ids=cached["node_composite_ids"],
        cache_key=cached["cache_key"],
    )

    with pytest.raises(ValueError, match="expected float32"):
        ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )


def test_ensure_cached_novae_embeddings_rejects_an_unexpected_width(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    with pytest.raises(ValueError, match="expected"):
        ensure_cached_novae_embeddings(
            tmp_path, inputs, _stub_novae_feature_fn, query, expected_dim=99,
            checkpoint_provenance=_CHECKPOINT_PROVENANCE,
        )


def test_cache_key_changes_when_checkpoint_provenance_changes(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    provenance_v2 = dict(_CHECKPOINT_PROVENANCE, checkpoint_revision="checkpoint-v2")
    _, path_a = ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    _, path_b = ensure_cached_novae_embeddings(
        tmp_path, inputs, _stub_novae_feature_fn, query, checkpoint_provenance=provenance_v2,
    )
    assert path_a != path_b  # a changed checkpoint identity must invalidate the cache


def test_ensure_cached_novae_embeddings_requires_checkpoint_provenance(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    inputs = build_context_only_novae_input(adata, context, query, sample_id="S0")

    with pytest.raises(TypeError):
        ensure_cached_novae_embeddings(tmp_path, inputs, _stub_novae_feature_fn, query)  # missing required kwarg


def test_validate_checkpoint_provenance_rejects_missing_and_empty_fields():
    with pytest.raises(TypeError):
        _validate_checkpoint_provenance("not-a-dict")

    incomplete = dict(_CHECKPOINT_PROVENANCE)
    del incomplete["novae_package_version"]
    with pytest.raises(ValueError, match="missing required key"):
        _validate_checkpoint_provenance(incomplete)

    blank = dict(_CHECKPOINT_PROVENANCE, adapter_version="   ")
    with pytest.raises(ValueError, match="empty/whitespace"):
        _validate_checkpoint_provenance(blank)

    signature = _validate_checkpoint_provenance(_CHECKPOINT_PROVENANCE)
    assert isinstance(signature, str) and signature


def test_build_novae_preflight_report_end_to_end_and_persists(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:2]
    context = barcodes[2:]

    report = build_novae_preflight_report(
        adata, context, query, _stub_novae_feature_fn, sample_id="S0", cache_dir=tmp_path / "cache",
        checkpoint_provenance=_CHECKPOINT_PROVENANCE,
    )
    assert report["passed"] is True
    assert report["n_context"] == len(context)
    assert report["n_query_checked"] == len(query)
    assert report["checkpoint_provenance"]["checkpoint_repo"] == _CHECKPOINT_PROVENANCE["checkpoint_repo"]
    assert len(report["verified"]) > 0
    assert len(report["not_provable_from_this_module_alone"]) > 0

    report_path = tmp_path / "report.json"
    save_novae_preflight_report(report, report_path)
    assert load_novae_preflight_report(report_path) == report


def test_build_novae_preflight_report_raises_on_context_query_overlap(tmp_path):
    adata = _square_grid_adata()
    shared = list(adata.obs_names)[:5]
    with pytest.raises(ValueError, match="overlap"):
        build_novae_preflight_report(
            adata, shared, shared[:1], _stub_novae_feature_fn, sample_id="S0", cache_dir=tmp_path / "cache",
            checkpoint_provenance=_CHECKPOINT_PROVENANCE,
        )


def test_build_novae_preflight_report_requires_checkpoint_provenance(tmp_path):
    adata = _square_grid_adata()
    barcodes = list(adata.obs_names)
    query = barcodes[:1]
    context = barcodes[1:]
    with pytest.raises(TypeError):
        build_novae_preflight_report(
            adata, context, query, _stub_novae_feature_fn, sample_id="S0", cache_dir=tmp_path / "cache",
        )
