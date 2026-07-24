"""Architecture 4 depends on the real `stpath` package plus real
STPATH_GENE_VOC_PATH/STPATH_MODEL_WEIGHT_PATH resources. Mirrors this
project's existing convention (tests/test_stpath_encoder.py in the main
repo) of skipping cleanly rather than failing when those aren't available
in the current environment -- not something this test suite can fake, and
consistent with how every other STPath-dependent test in this codebase
already handles the same real constraint.
"""
import os

import pytest


def _stpath_available() -> bool:
    try:
        import stpath  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("STPATH_GENE_VOC_PATH")) and bool(os.environ.get("STPATH_MODEL_WEIGHT_PATH"))


@pytest.mark.skipif(not _stpath_available(), reason="stpath package or STPATH_*_PATH env vars not available")
def test_architecture4_scfoundation_residual_wiring():
    import torch

    from gen2_architectures.models.arch4_stpath_hybrid import Architecture4

    gene_names = [f"g{i}" for i in range(200)]
    model = Architecture4(
        gene_names=gene_names, gene_voc_path=os.environ["STPATH_GENE_VOC_PATH"],
        model_weight_path=os.environ["STPATH_MODEL_WEIGHT_PATH"],
        organ_type="Lung", tech_type="Visium", scfoundation_dim=128, hidden_dim=64,
    )
    n_context, n_query = 20, 3
    context = {
        "coords": torch.randn(n_context, 2) * 500,
        "expression": torch.randn(n_context, len(gene_names)),  # raw-log1p, in a real run
        "images": torch.randn(n_context, 1536),
        "extra_features": torch.randn(n_context, 128),
    }
    query = {"coords": torch.randn(n_query, 2) * 500, "images": torch.randn(n_query, 1536)}
    pred = model(context, query)
    assert pred.shape[0] == n_query
    assert hasattr(model, "_decoder_target_col_idx")


def test_architecture4_module_imports_without_the_stpath_package():
    """The stpath package is only needed at STPathContextEncoder
    CONSTRUCTION time (a lazy import inside __init__), not at module
    import time -- this must always pass, even with stpath unavailable."""
    from gen2_architectures.models.arch4_stpath_hybrid import Architecture4  # noqa: F401
