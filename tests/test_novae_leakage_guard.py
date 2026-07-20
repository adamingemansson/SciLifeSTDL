from types import SimpleNamespace

import numpy as np
import pytest

from src.data.context_features import ContextOnlyNovaeProvider, novae_input_mode


class FakeAdata:
    def __init__(self, names, x):
        self.obs_names = np.asarray(names)
        self.X = np.asarray(x, dtype=np.float32)

    def __getitem__(self, mask):
        return FakeAdata(self.obs_names[np.asarray(mask)], self.X[np.asarray(mask)])

    def copy(self):
        return FakeAdata(self.obs_names.copy(), self.X.copy())


def test_context_provider_physically_excludes_query_rows(tmp_path):
    adata = FakeAdata(["a", "b", "c", "d"], [[1], [2], [1000], [2000]])
    seen = []

    def feature_fn(context):
        seen.append((context.obs_names.tolist(), context.X.copy()))
        return np.concatenate([context.X, context.X + 1], axis=1)

    provider = ContextOnlyNovaeProvider(
        adata, cache_dir=tmp_path, sample_id="S", feature_fn=feature_fn
    )
    mask = np.array([True, True, False, False])
    features = provider(mask)
    assert features.shape == (2, 2)
    assert seen[0][0] == ["a", "b"]
    assert seen[0][1].max() == 2

    # Cached lookup remains tied to the exact context barcodes.
    again = provider(mask)
    assert np.array_equal(again, features)
    assert len(seen) == 1


def test_unsafe_full_graph_requires_double_opt_in():
    params = {"context_encoder_type": "storm_lite", "gene_encoder_type": "novae"}
    cfg = SimpleNamespace(data={"novae_mode": "disabled"})
    with pytest.raises(ValueError, match="leak"):
        novae_input_mode(cfg, params)

    cfg = SimpleNamespace(data={"novae_mode": "unsafe_full_graph"})
    with pytest.raises(ValueError, match="allow_unsafe_novae"):
        novae_input_mode(cfg, params)

    cfg = SimpleNamespace(data={
        "novae_mode": "unsafe_full_graph", "allow_unsafe_novae": True
    })
    assert novae_input_mode(cfg, params) == "unsafe_full_graph"
