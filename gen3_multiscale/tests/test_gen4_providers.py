"""EncoderIdentity + cache round-trip tests, plus a real end-to-end
build_gen4_spatial_field_example integration test on synthetic HEST-shaped
data (reusing _step6_fixtures.py's real dataset-manifest/patch machinery)."""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gen3_multiscale.data import loaders
from gen3_multiscale.data.example_builder import load_sample_for_examples
from gen3_multiscale.gen4.inputs import build_gen4_spatial_field_example
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.scfoundation_cache import (
    _expression_content_hash,
    barcode_embedding_lookup,
    build_scfoundation_spot_feature_cache,
    load_scfoundation_spot_features,
)
from gen3_multiscale.scripts import precompute_gen45_features
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache, load_uni2_spot_features
from gen3_multiscale.tests._gen4_fixtures import StubSCFoundationEncoder, StubUNI2Encoder
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment


def _square_grid_adata(n_side: int = 6, spacing: float = 10.0, n_genes: int = 5, seed: int = 0) -> "ad.AnnData":
    """Mirrors tests/test_example_builder.py's own helper -- kept local
    rather than cross-imported so this file's geometry tests don't
    depend on that module's internals."""
    rng = np.random.default_rng(seed)
    coords = np.array([[x * spacing, y * spacing] for x in range(n_side) for y in range(n_side)], dtype=np.float64)
    n = coords.shape[0]
    barcodes = [f"SPOT{i}-1" for i in range(n)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    counts = rng.poisson(5, size=(n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)))
    adata.obsm["spatial"] = coords
    return adata


def test_encoder_identity_rejects_blank_fields():
    with pytest.raises(ValueError):
        EncoderIdentity(
            encoder_name="", checkpoint_sha256="a" * 8, pinned_revision="0" * 40,
            package_version="1.0", preprocessing_spec="spec", output_dim=8,
        )


def test_encoder_identity_rejects_nonpositive_output_dim():
    with pytest.raises(ValueError):
        EncoderIdentity(
            encoder_name="uni2", checkpoint_sha256="a" * 8, pinned_revision="0" * 40,
            package_version="1.0", preprocessing_spec="spec", output_dim=0,
        )


def test_uni2_cache_round_trip(tmp_path):
    encoder = StubUNI2Encoder(output_dim=6)
    barcodes = np.array(["a", "b", "c"])
    patches = np.random.default_rng(0).integers(0, 255, size=(3, 4, 4, 3)).astype(np.uint8)
    availability = np.array([True, False, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    loaded = load_uni2_spot_features(tmp_path, "s1", barcodes, patches, availability)
    assert loaded["features"].shape == (3, 6)
    assert np.array_equal(loaded["features"][1], np.zeros(6))  # unavailable spot -> explicit zero
    expected_available = encoder.encode_available_patches(patches[[0, 2]])
    assert np.allclose(loaded["features"][[0, 2]], expected_available)
    assert loaded["provenance"]["output_dim"] == 6


def test_scfoundation_cache_round_trip(tmp_path):
    gene_names = [f"g{i}" for i in range(4)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=7)
    barcodes = np.array(["a", "b"])
    expression = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "hash123", encoder)
    loaded = load_scfoundation_spot_features(tmp_path, "s1", barcodes, "hash123", expression)
    assert loaded["features"].shape == (2, 7)
    assert np.allclose(loaded["features"], encoder.encode_rows(expression))
    lookup = barcode_embedding_lookup(loaded)
    assert set(lookup) == {"a", "b"}


def test_scfoundation_sparse_cache_materializes_only_bounded_row_batches(tmp_path):
    class TrackingSparseExpression:
        def __init__(self, array):
            self.matrix = sp.csr_matrix(array)
            self.shape = self.matrix.shape
            self.row_batch_sizes = []

        def __getitem__(self, key):
            row_key = key[0] if isinstance(key, tuple) else key
            start, stop, step = row_key.indices(self.shape[0])
            assert step == 1
            self.row_batch_sizes.append(stop - start)
            return self.matrix[key]

        def __array__(self, *args, **kwargs):
            raise AssertionError("the complete sparse expression matrix must never be densified")

    gene_names = [f"g{i}" for i in range(4)]
    dense = np.arange(28, dtype=np.float32).reshape(7, 4)
    expression = TrackingSparseExpression(dense)
    encoder = StubSCFoundationEncoder(gene_names, output_dim=7)
    barcodes = np.asarray([f"b{i}" for i in range(7)])

    build_scfoundation_spot_feature_cache(
        tmp_path, "s1", barcodes, expression, "hash123", encoder, batch_size=2,
    )

    assert expression.row_batch_sizes
    assert max(expression.row_batch_sizes) <= 2
    assert _expression_content_hash(sp.csr_matrix(dense)) == _expression_content_hash(dense)


def test_scfoundation_only_precompute_never_loads_hest_patches(tmp_path, monkeypatch):
    gene_names = ["g0", "g1"]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=3)
    adata = ad.AnnData(
        X=sp.csr_matrix(np.ones((2, 2), dtype=np.float32)),
        obs=pd.DataFrame(
            {"_scilifestdl_raw_library_size": [10.0, 20.0]},
            index=pd.Index(["a", "b"]),
        ),
        var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    manifest = {
        "samples": {"s1": {}},
        "gene_panel": gene_names,
        "hest_data_dir": str(tmp_path / "hest"),
        "hest_cache_dir": str(tmp_path / "cache"),
    }
    captured = {}

    monkeypatch.setattr(precompute_gen45_features, "load_dataset_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        precompute_gen45_features,
        "load_expression_for_model_target_space",
        lambda _manifest, _sample_id: adata,
    )
    monkeypatch.setattr(
        precompute_gen45_features,
        "load_sample_for_examples",
        lambda *_args, **_kwargs: pytest.fail(
            "scFoundation-only precompute must not load H&E patches"
        ),
    )
    monkeypatch.setattr(
        precompute_gen45_features,
        "FrozenSCFoundationEncoder",
        lambda *_args, **_kwargs: encoder,
    )

    def capture_cache(_root, sample_id, _barcodes, expression, *_args, **_kwargs):
        captured[sample_id] = expression

    monkeypatch.setattr(
        precompute_gen45_features,
        "build_scfoundation_spot_feature_cache",
        capture_cache,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "precompute_gen45_features",
            "--manifest", "manifest.json",
            "--cache-root", str(tmp_path / "cache"),
            "--modalities", "scfoundation",
            "--scfoundation-checkpoint", "models.ckpt",
            "--scfoundation-vocab", "vocab.tsv",
            "--scfoundation-repo", "repo",
            "--scfoundation-revision", "a" * 40,
            "--device", "cpu",
        ],
    )

    precompute_gen45_features.main()

    assert captured == {"s1": adata.X}


def test_build_gen4_spatial_field_example_real_pipeline(tmp_path, monkeypatch):
    """End-to-end: real synthetic HEST-shaped sample -> real
    build_gen4_spatial_field_example, with a real (stubbed-encoder) GEX
    context embedding lookup attached."""
    cfg, manifest, _manifest_path = prepare_step6_experiment(tmp_path, monkeypatch, n_side=6, n_genes=5)
    train_sample_id = manifest["train_sample_ids"][0]
    adata, patches, image_source_available = load_sample_for_examples(manifest, train_sample_id, cfg.data.hest_data_dir)
    barcodes = list(adata.obs_names)
    context_barcodes, query_barcodes = barcodes[: len(barcodes) // 2], barcodes[len(barcodes) // 2:]

    image_features = np.random.default_rng(0).normal(size=(adata.n_obs, 6)).astype(np.float32)

    gene_names = [f"g{i}" for i in range(5)]
    scfoundation = StubSCFoundationEncoder(gene_names, output_dim=9)
    embedding_lookup = {
        str(barcode): row for barcode, row in zip(adata.obs_names, scfoundation.encode_rows(np.asarray(adata.X)))
    }

    inputs, targets = build_gen4_spatial_field_example(
        adata, patches, context_barcodes, query_barcodes,
        sample_id=train_sample_id, patient_id=manifest["samples"][train_sample_id]["patient_id"],
        precomputed_spot_features=image_features,
        gex_context_embedding=embedding_lookup, gex_context_provenance=scfoundation.identity.as_dict(),
        full_sample_coords=np.asarray(adata.obsm["spatial"]),
    )
    assert inputs.context_gex_embedding is not None
    assert inputs.context_gex_embedding.shape == (inputs.observed_coords.shape[0], 9)
    assert inputs.context_gex_embedding_provenance["encoder_name"] == "scfoundation"


def test_build_gen4_spatial_field_example_populates_uni2_features_provenance_and_organ(tmp_path, monkeypatch):
    """Item 1 (audit finding: "the hybrid arm receives
    observed_uni2_features=None") + Item 3 (real per-sample organ):
    uni2_spot_embedding/wsi_tile_feature_provenance/sample_organ are real,
    barcode-aligned, query/hole-excluded-by-construction inputs, not
    synthetic-fixture-only fields."""
    cfg, manifest, _manifest_path = prepare_step6_experiment(tmp_path, monkeypatch, n_side=6, n_genes=5)
    train_sample_id = manifest["train_sample_ids"][0]
    adata, patches, _availability = load_sample_for_examples(manifest, train_sample_id, cfg.data.hest_data_dir)
    barcodes = list(adata.obs_names)
    context_barcodes, query_barcodes = barcodes[: len(barcodes) // 2], barcodes[len(barcodes) // 2:]
    image_features = np.random.default_rng(0).normal(size=(adata.n_obs, 6)).astype(np.float32)

    uni2_lookup = {
        str(barcode): row
        for barcode, row in zip(adata.obs_names, np.random.default_rng(1).normal(size=(adata.n_obs, 6)).astype(np.float32))
    }

    inputs, _targets = build_gen4_spatial_field_example(
        adata, patches, context_barcodes, query_barcodes,
        sample_id=train_sample_id, patient_id=manifest["samples"][train_sample_id]["patient_id"],
        precomputed_spot_features=image_features,
        uni2_spot_embedding=uni2_lookup, wsi_tile_feature_provenance="uni2", sample_organ="Lung",
        full_sample_coords=np.asarray(adata.obsm["spatial"]),
    )
    assert inputs.observed_uni2_features is not None
    assert inputs.observed_uni2_features.shape == (inputs.observed_coords.shape[0], 6)
    # Query/hole exclusion is inherited by construction -- every row
    # corresponds exactly to a REALIZED observed_barcodes entry, and no
    # query barcode ever appears there (validate_spatial_field_example's
    # own disjointness check, delegated to unconditionally above).
    expected = np.stack([uni2_lookup[str(b)] for b in inputs.observed_barcodes])
    assert np.array_equal(inputs.observed_uni2_features, expected)
    assert inputs.wsi_tile_feature_provenance == "uni2"
    assert inputs.sample_organ == "Lung"


def test_build_gen4_spatial_field_example_zeroes_uni2_features_for_physically_overlapping_context_rows():
    """Integration-audit finding #4 (CONFIRMED real bug, fixed): a
    context spot's H&E can physically overlap the query hole (patch
    footprint, not barcode identity) even though its own barcode is
    disjoint from every query barcode -- example_builder.py already
    computes this as observed_image_available=False and zeroes
    precomputed_spot_features there. The first version of
    observed_uni2_features selected directly from uni2_spot_embedding
    with no reference to that flag at all, leaking the real cached UNI2
    feature for a physically-hidden row. Proves the fix: mutating that
    exact row's cached UNI2 feature to a distinctive value must NOT
    change the constructed model input."""
    adata = _square_grid_adata(n_side=6, spacing=10.0)
    patches = np.zeros((adata.n_obs, 4, 4, 3), dtype=np.uint8)
    barcodes = list(adata.obs_names)
    coords = adata.obsm["spatial"]

    query_idx = 14
    query_barcode = barcodes[query_idx]
    query_xy = coords[query_idx]
    distances = np.linalg.norm(coords - query_xy, axis=1)
    distances[query_idx] = np.inf
    neighbor_idx = int(np.argmin(distances))  # close enough to physically overlap -> unavailable
    neighbor_barcode = barcodes[neighbor_idx]

    context_barcodes = [b for b in barcodes if b != query_barcode]
    image_features = np.zeros((adata.n_obs, 6), dtype=np.float32)
    uni2_lookup_a = {b: np.zeros(6, dtype=np.float32) for b in barcodes}
    uni2_lookup_b = dict(uni2_lookup_a)
    uni2_lookup_b[neighbor_barcode] = np.full(6, 12345.0, dtype=np.float32)  # distinctive, real-looking value

    kwargs = dict(
        sample_id="S0", patient_id="P0", patch_size_fullres=30.0, require_full_sample_coords=False,
        precomputed_spot_features=image_features, full_sample_coords=coords,
    )
    inputs_a, _ = build_gen4_spatial_field_example(
        adata, patches, context_barcodes, [query_barcode], uni2_spot_embedding=uni2_lookup_a, **kwargs,
    )
    inputs_b, _ = build_gen4_spatial_field_example(
        adata, patches, context_barcodes, [query_barcode], uni2_spot_embedding=uni2_lookup_b, **kwargs,
    )
    neighbor_pos = inputs_b.observed_barcodes.tolist().index(neighbor_barcode)
    assert inputs_b.observed_image_available[neighbor_pos] == False  # noqa: E712 -- real numpy bool, physical overlap confirmed
    assert np.all(inputs_b.observed_uni2_features[neighbor_pos] == 0.0)  # zeroed, not leaked
    np.testing.assert_array_equal(inputs_a.observed_uni2_features, inputs_b.observed_uni2_features)


def test_build_gen4_spatial_field_example_rejects_bad_wsi_tile_feature_provenance(tmp_path, monkeypatch):
    cfg, manifest, _manifest_path = prepare_step6_experiment(tmp_path, monkeypatch, n_side=6, n_genes=5)
    train_sample_id = manifest["train_sample_ids"][0]
    adata, patches, _availability = load_sample_for_examples(manifest, train_sample_id, cfg.data.hest_data_dir)
    barcodes = list(adata.obs_names)
    context_barcodes, query_barcodes = barcodes[: len(barcodes) // 2], barcodes[len(barcodes) // 2:]
    image_features = np.random.default_rng(0).normal(size=(adata.n_obs, 6)).astype(np.float32)
    with pytest.raises(ValueError, match="wsi_tile_feature_provenance"):
        build_gen4_spatial_field_example(
            adata, patches, context_barcodes, query_barcodes,
            sample_id=train_sample_id, patient_id=manifest["samples"][train_sample_id]["patient_id"],
            precomputed_spot_features=image_features, wsi_tile_feature_provenance="gigapath",
            full_sample_coords=np.asarray(adata.obsm["spatial"]),
        )


def test_build_gen4_spatial_field_example_missing_barcode_fails_closed(tmp_path, monkeypatch):
    cfg, manifest, _manifest_path = prepare_step6_experiment(tmp_path, monkeypatch, n_side=6, n_genes=5)
    train_sample_id = manifest["train_sample_ids"][0]
    adata, patches, _availability = load_sample_for_examples(manifest, train_sample_id, cfg.data.hest_data_dir)
    barcodes = list(adata.obs_names)
    context_barcodes, query_barcodes = barcodes[: len(barcodes) // 2], barcodes[len(barcodes) // 2:]
    image_features = np.random.default_rng(0).normal(size=(adata.n_obs, 6)).astype(np.float32)

    incomplete_lookup = {str(barcodes[0]): np.zeros(9, dtype=np.float32)}  # missing every other context barcode
    with pytest.raises(KeyError):
        build_gen4_spatial_field_example(
            adata, patches, context_barcodes, query_barcodes,
            sample_id=train_sample_id, patient_id=manifest["samples"][train_sample_id]["patient_id"],
            precomputed_spot_features=image_features, gex_context_embedding=incomplete_lookup,
            full_sample_coords=np.asarray(adata.obsm["spatial"]),
        )
