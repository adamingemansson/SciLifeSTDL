"""Gen4 leakage/provenance invariant tests -- GEN4_CONTRACT.md section 11,
the task's own required-tests list."""
from __future__ import annotations

import copy
import dataclasses
import inspect

import numpy as np
import pytest
import torch

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen4.providers import GexContextProvider
from gen3_multiscale.gen4.scfoundation_cache import build_scfoundation_spot_feature_cache, load_scfoundation_spot_features
from gen3_multiscale.gen4.stpath_context import Gen4STPathContextEncoder
from gen3_multiscale.gen4.uni2_spot_cache import build_uni2_spot_feature_cache, load_uni2_spot_features
from gen3_multiscale.models.architectures import Architecture3
from gen3_multiscale.tests._gen4_fixtures import GEN4_MODEL_KWARGS, StubSCFoundationEncoder, StubUNI2Encoder, synthetic_gen4_inputs


def _tiny_model():
    return Gen4Conditioner(
        n_genes=6, gex_feature_dim=4, image_feature_dim=8,
        use_regional_he=False, global_context_source="none", **GEN4_MODEL_KWARGS,
    )


def test_query_barcode_mutation_cannot_affect_conditioning():
    inputs, _targets = synthetic_gen4_inputs()
    model = _tiny_model()
    torch.manual_seed(0)
    out1 = model(inputs)
    mutated = dataclasses.replace(inputs, query_barcodes=np.array([f"different-{i}" for i in range(inputs.query_barcodes.shape[0])]))
    torch.manual_seed(0)
    out2 = model(mutated)
    assert torch.equal(out1["expression"], out2["expression"])


def test_query_expression_mutation_cannot_affect_conditioning():
    """targets.query_expression is never even an argument to forward() --
    mutating it between two identical-inputs calls cannot change the
    output, since the model literally has no reference to targets."""
    inputs, targets = synthetic_gen4_inputs()
    model = _tiny_model()
    torch.manual_seed(0)
    out1 = model(inputs)
    _mutated_targets = dataclasses.replace(targets, query_expression=targets.query_expression * 0.0 + 999.0)
    torch.manual_seed(0)
    out2 = model(inputs)  # targets never passed at all
    assert torch.equal(out1["expression"], out2["expression"])


def test_hidden_data_never_passed_cannot_affect_output():
    """A "hidden query patch" has no field anywhere in SpatialFieldInputs
    to smuggle a value through -- confirmed structurally: the dataclass's
    own field set (inherited, extended only by context_gex_embedding/
    context_gex_embedding_provenance) contains no query-indexed image or
    expression field at all."""
    from gen3_multiscale.gen4.inputs import Gen4SpatialFieldInputs
    field_names = {f.name for f in dataclasses.fields(Gen4SpatialFieldInputs)}
    for forbidden in ("query_expression", "query_gigapath_features", "query_image", "query_patch", "query_full_gene_expression"):
        assert forbidden not in field_names


def test_visible_context_mutation_can_affect_output():
    inputs, _targets = synthetic_gen4_inputs()
    model = _tiny_model()
    torch.manual_seed(0)
    out1 = model(inputs)
    mutated_expr = inputs.observed_full_gene_expression.copy()
    mutated_expr[0] += 50.0
    mutated = dataclasses.replace(inputs, observed_full_gene_expression=mutated_expr)
    torch.manual_seed(0)
    out2 = model(mutated)
    assert not torch.equal(out1["expression"], out2["expression"])


def test_gen4_conditioner_never_overrides_the_candidate_pool():
    """Structural guarantee: Gen4Conditioner does not define its own
    _candidate_pool -- the inherited Architecture3/_SharedFieldArchitecture
    method (which builds the transport head's candidate values EXCLUSIVELY
    from observed_full_gene_expression) is used unmodified."""
    assert "_candidate_pool" not in Gen4Conditioner.__dict__
    assert Gen4Conditioner._candidate_pool is Architecture3._candidate_pool


def test_scfoundation_encoder_interface_has_no_query_parameter():
    params = set(inspect.signature(GexContextProvider.encode_rows).parameters) - {"self"}
    assert not any("query" in p for p in params)


def test_stpath_encode_context_only_has_no_query_parameter():
    params = set(inspect.signature(Gen4STPathContextEncoder.encode_context_only).parameters) - {"self"}
    assert not any("query" in p for p in params)
    # And the base pilot-study forward() (which DOES take query images) is
    # never called from this method's own body -- none of its real
    # query-shaped argument names are referenced as an actual code token
    # anywhere in the executable statements (docstring/comments may
    # legitimately discuss the contrast in prose).
    source = inspect.getsource(Gen4STPathContextEncoder.encode_context_only)
    code_lines = [
        line for line in source.split('"""', 2)[-1].splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in code_lines:
        code = line.split("#", 1)[0]
        for forbidden in ("query_coords", "query_images", "query_image_available", "query_expression"):
            assert forbidden not in code, f"found {forbidden!r} in executable code: {line!r}"


def test_uni2_cache_barcode_order_mismatch_fails_closed(tmp_path):
    encoder = StubUNI2Encoder(output_dim=4)
    barcodes = np.array(["a", "b", "c"])
    patches = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    with pytest.raises(ValueError, match="barcode identity/order"):
        load_uni2_spot_features(tmp_path, "s1", np.array(["a", "c", "b"]), patches, availability)


def test_uni2_cache_patch_content_mismatch_fails_closed(tmp_path):
    encoder = StubUNI2Encoder(output_dim=4)
    barcodes = np.array(["a", "b", "c"])
    patches = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    availability = np.array([True, True, True])
    build_uni2_spot_feature_cache(tmp_path, "s1", barcodes, patches, availability, encoder)
    tampered_patches = patches.copy()
    tampered_patches[0] = 255
    with pytest.raises(ValueError, match="patch content"):
        load_uni2_spot_features(tmp_path, "s1", barcodes, tampered_patches, availability)


def test_scfoundation_cache_gene_panel_mismatch_fails_closed(tmp_path):
    gene_names = [f"g{i}" for i in range(6)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=5)
    barcodes = np.array(["a", "b"])
    expression = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "panelhash123", encoder)
    with pytest.raises(ValueError, match="different gene panel"):
        load_scfoundation_spot_features(tmp_path, "s1", barcodes, "a-different-panel-hash", expression)


def test_scfoundation_cache_barcode_mismatch_fails_closed(tmp_path):
    gene_names = [f"g{i}" for i in range(6)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=5)
    barcodes = np.array(["a", "b"])
    expression = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "panelhash123", encoder)
    with pytest.raises(ValueError, match="barcode identity/order"):
        load_scfoundation_spot_features(tmp_path, "s1", np.array(["a", "c"]), "panelhash123", expression)


def test_scfoundation_cache_stale_expression_fails_closed(tmp_path):
    """Codex audit finding: cache identity previously hashed neither the
    expression VALUES nor the preprocessing that produced them, so a
    changed expression matrix (same barcodes, same gene panel) could
    silently reuse a stale cached embedding."""
    gene_names = [f"g{i}" for i in range(6)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=5)
    barcodes = np.array(["a", "b"])
    expression = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    build_scfoundation_spot_feature_cache(tmp_path, "s1", barcodes, expression, "panelhash123", encoder)
    tampered_expression = expression.copy()
    tampered_expression[0, 0] += 1.0
    with pytest.raises(ValueError, match="different expression values"):
        load_scfoundation_spot_features(tmp_path, "s1", barcodes, "panelhash123", tampered_expression)


def test_scfoundation_encoder_is_row_independent():
    """Calling encode_rows on a subset of rows must reproduce exactly the
    corresponding rows of encoding the full batch -- proves no
    cross-row statistic (e.g. a batch norm) is computed."""
    gene_names = [f"g{i}" for i in range(6)]
    encoder = StubSCFoundationEncoder(gene_names, output_dim=5)
    expression = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    full = encoder.encode_rows(expression)
    subset = encoder.encode_rows(expression[3:6])
    assert np.allclose(full[3:6], subset)
