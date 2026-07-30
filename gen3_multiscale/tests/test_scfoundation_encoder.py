"""Item 6 (six-launch-blocker audit): scFoundation's read-depth-token
derivation must use the real, pre-normalization raw total count per row,
never re-derived by summing the already-normalized/log1p'd expression
matrix. `FrozenSCFoundationEncoder.__init__` needs a real `scfoundation`
package + checkpoint (not installed in this sandbox -- GEN4_CONTRACT.md
section 13's documented gap), so `_to_scfoundation_input`'s pure-numpy
logic (the actual bug) is exercised directly against a bypassed-__init__
instance carrying only the attributes that method reads."""
from __future__ import annotations

import numpy as np
import pytest

from gen3_multiscale.gen4.scfoundation_encoder import FrozenSCFoundationEncoder


def _bare_encoder(gene_names, vocab, target_total_count=1e4):
    encoder = FrozenSCFoundationEncoder.__new__(FrozenSCFoundationEncoder)
    encoder.scfoundation_vocab = list(vocab)
    encoder.gene_names = tuple(gene_names)
    vocab_position = {gene: idx for idx, gene in enumerate(vocab)}
    encoder._manifest_to_vocab_pos = [
        (row, vocab_position[gene]) for row, gene in enumerate(gene_names) if gene in vocab_position
    ]
    encoder.target_total_count = float(target_total_count)
    return encoder


def test_to_scfoundation_input_uses_raw_library_size_not_expression_sum():
    """CONFIRMED real bug: the prior version computed the read-depth
    token as log1p(expression.sum(axis=1)) -- but expression here is
    ALREADY normalize_log1p-transformed (never raw counts), so its
    row-sum is not a real read depth at all. The fixed version must
    ignore expression's own magnitude entirely for this token and use
    only the real raw_library_size argument."""
    encoder = _bare_encoder(["g0", "g1"], ["g0", "g1", "g2"])
    # Two rows with IDENTICAL already-normalized expression (so the old,
    # buggy sum-of-expression approach would produce the SAME token for
    # both) but genuinely DIFFERENT real raw library sizes.
    expression = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    raw_library_size = np.array([500.0, 50_000.0], dtype=np.float32)

    model_input = encoder._to_scfoundation_input(expression, raw_library_size)

    total_count_col = model_input[:, len(encoder.scfoundation_vocab)]
    assert not np.allclose(total_count_col[0], total_count_col[1])
    assert np.allclose(total_count_col, np.log1p(raw_library_size))
    # Never derived from expression's own row sum (the old, buggy path).
    assert not np.allclose(total_count_col, np.log1p(expression.sum(axis=1)))


def test_to_scfoundation_input_reindexes_into_fixed_vocabulary_order():
    encoder = _bare_encoder(["g2", "g0"], ["g0", "g1", "g2"])
    expression = np.array([[10.0, 20.0]], dtype=np.float32)  # g2=10, g0=20
    raw_library_size = np.array([100.0], dtype=np.float32)

    model_input = encoder._to_scfoundation_input(expression, raw_library_size)

    assert model_input[0, 0] == 20.0  # g0
    assert model_input[0, 1] == 0.0   # g1 -- absent from manifest, stays zero
    assert model_input[0, 2] == 10.0  # g2
    assert np.isclose(model_input[0, 3], np.log1p(100.0))  # real read-depth token
    assert np.isclose(model_input[0, 4], np.log1p(encoder.target_total_count))  # target token


def test_encode_rows_requires_raw_library_size():
    encoder = _bare_encoder(["g0"], ["g0"])
    encoder.device = "cpu"
    with pytest.raises(ValueError, match="raw_library_size"):
        encoder.encode_rows(np.zeros((2, 1), dtype=np.float32))


def test_encode_rows_rejects_misaligned_raw_library_size():
    encoder = _bare_encoder(["g0"], ["g0"])
    encoder.device = "cpu"
    with pytest.raises(ValueError, match="raw_library_size has"):
        encoder.encode_rows(np.zeros((2, 1), dtype=np.float32), raw_library_size=np.zeros(3, dtype=np.float32))
