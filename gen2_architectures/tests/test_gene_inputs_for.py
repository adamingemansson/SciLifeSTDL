"""GPT-audit-flagged bug (2026-07-27, second-pass re-audit, confirmed and
fixed): the final held-out test-eval block in train_local_neighborhood.py
was a hand-duplicated copy of the periodic validation block. When
context_gene_features was added for Architecture 4, a replace_all string
edit only matched the copies whose surrounding text happened to be
identical -- the final-eval block's differed just enough not to match, so
Architecture 4's automatic end-of-training test metrics were silently
computed with the wrong (library-size-normalized, not raw-count) log1p
context. Fixed by extracting one shared _gene_inputs_for() used by every
call site (validation loop, final test loop, and the standalone
run_held_out_evaluation.py) so this class of bug is structurally
impossible to reintroduce by duplication."""
import numpy as np
import pytest

from gen2_architectures.training.train_local_neighborhood import _gene_inputs_for


class _FakeAdata:
    def __init__(self, layers):
        self.layers = layers


def _adata_with_raw_counts():
    raw = np.array([[0.0, 3.0], [1.0, 0.0]], dtype=np.float32)
    return _FakeAdata(layers={"raw_counts": raw}), raw


def test_architecture_4_gets_stpath_raw_log1p_context_and_scfoundation_extra_features():
    adata, raw = _adata_with_raw_counts()
    providers = {"s1": "fake_provider"}

    gene_inputs = _gene_inputs_for("4", "s1", adata, providers)

    assert gene_inputs["context_gene_feature_provider"] is None
    assert np.allclose(gene_inputs["context_gene_features"], np.log1p(raw))
    assert gene_inputs["context_extra_feature_provider"] == "fake_provider"


def test_architecture_2_gets_a_scfoundation_context_gene_feature_provider_only():
    adata, _ = _adata_with_raw_counts()
    providers = {"s1": "fake_provider"}

    gene_inputs = _gene_inputs_for("2", "s1", adata, providers)

    assert gene_inputs["context_gene_feature_provider"] == "fake_provider"
    assert gene_inputs["context_gene_features"] is None
    assert gene_inputs["context_extra_feature_provider"] is None


def test_architecture_1_gets_no_special_gene_inputs():
    adata, _ = _adata_with_raw_counts()

    gene_inputs = _gene_inputs_for("1", "s1", adata, {})

    assert gene_inputs["context_gene_feature_provider"] is None
    assert gene_inputs["context_gene_features"] is None
    assert gene_inputs["context_extra_feature_provider"] is None


def test_architecture_4_raises_a_clear_error_when_raw_counts_is_missing():
    adata = _FakeAdata(layers={})
    with pytest.raises(ValueError, match="raw_counts"):
        _gene_inputs_for("4", "s1", adata, {})
