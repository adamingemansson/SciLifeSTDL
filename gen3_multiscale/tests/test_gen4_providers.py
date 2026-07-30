"""EncoderIdentity + cache round-trip tests, plus a real end-to-end
build_gen4_spatial_field_example integration test on synthetic HEST-shaped
data (reusing _step6_fixtures.py's real dataset-manifest/patch machinery)."""
from __future__ import annotations

import numpy as np
import pytest

from gen3_multiscale.data import loaders
from gen3_multiscale.data.example_builder import load_sample_for_examples
from gen3_multiscale.gen4.inputs import build_gen4_spatial_field_example
from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.gen4.scfoundation_cache import (
    barcode_embedding_lookup, build_scfoundation_spot_feature_cache, load_scfoundation_spot_features,
)
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache, load_uni2_spot_features
from gen3_multiscale.tests._gen4_fixtures import StubSCFoundationEncoder, StubUNI2Encoder
from gen3_multiscale.tests._step6_fixtures import prepare_step6_experiment


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
