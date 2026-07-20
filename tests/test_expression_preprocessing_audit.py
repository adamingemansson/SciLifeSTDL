import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from src.data.loaders import EXPRESSION_STATE_KEY, basic_qc_and_normalize


def test_expression_transform_is_idempotent_and_conflicts_raise():
    x = np.array([[1, 0, 2], [0, 3, 1], [2, 1, 0]], dtype=np.float32)
    adata = ad.AnnData(x)
    first = basic_qc_and_normalize(
        adata, min_genes=0, min_cells=0, transform="normalize_log1p"
    )
    values = first.X.copy()
    second = basic_qc_and_normalize(
        first, min_genes=0, min_cells=0, transform="normalize_log1p"
    )
    assert np.array_equal(second.X, values)
    assert second.uns[EXPRESSION_STATE_KEY]["transform"] == "normalize_log1p"
    with pytest.raises(ValueError, match="already has expression preprocessing"):
        basic_qc_and_normalize(second, min_genes=0, min_cells=0, transform="normalize")
