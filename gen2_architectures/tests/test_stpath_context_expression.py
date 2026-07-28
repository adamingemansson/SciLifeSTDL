"""GPT-audit-flagged bug (2026-07-27, confirmed and fixed): Architecture 4's
context["expression"] must be log1p(RAW counts), not the project's usual
library-size-normalized log1p -- STPath's frozen pretrained weights expect
the former, every gen2 config configures the latter for everything else."""
import numpy as np
import pytest

from gen2_architectures.training.train_local_neighborhood import _stpath_context_expression


class _FakeAdata:
    def __init__(self, layers):
        self.layers = layers


def test_computes_log1p_of_raw_counts_not_normalized_expression():
    raw = np.array([[0.0, 3.0, 7.0], [1.0, 0.0, 2.0]], dtype=np.float32)
    adata = _FakeAdata(layers={"raw_counts": raw})

    out = _stpath_context_expression(adata)

    assert np.allclose(out, np.log1p(raw))


def test_handles_a_sparse_raw_counts_layer():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    raw = np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float32)
    adata = _FakeAdata(layers={"raw_counts": scipy_sparse.csr_matrix(raw)})

    out = _stpath_context_expression(adata)

    assert np.allclose(out, np.log1p(raw))


def test_raises_a_clear_error_when_raw_counts_is_missing():
    adata = _FakeAdata(layers={})
    with pytest.raises(ValueError, match="raw_counts"):
        _stpath_context_expression(adata)
