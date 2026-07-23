"""Leak-safety tests for the niche candidate's data plumbing
(src/data/niche_features.py). Mirrors tests/test_novae_leakage_guard.py --
ContextOnlyFeatureProvider is the exact same generic engine
ContextOnlyNovaeProvider already uses (see context_features.py), so these
tests exercise it with a fake feature_fn rather than the real
sklearn/scanpy-dependent compute_banksy_augmented_niche_labels (not
importable in every environment this test suite runs in; its own
correctness is a separate, environment-dependent concern -- this file only
covers the leak-safety CONTRACT: context-only recomputation, and the
fail-closed data.niche_mode gate).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.context_features import ContextOnlyFeatureProvider, ContextOnlyNovaeProvider
from src.data.niche_features import model_uses_niche_candidate, niche_input_mode


class FakeAdata:
    def __init__(self, names, x):
        self.obs_names = np.asarray(names)
        self.X = np.asarray(x, dtype=np.float32)

    def __getitem__(self, mask):
        return FakeAdata(self.obs_names[np.asarray(mask)], self.X[np.asarray(mask)])

    def copy(self):
        return FakeAdata(self.obs_names.copy(), self.X.copy())


def test_context_only_feature_provider_is_the_same_engine_novae_uses():
    assert ContextOnlyFeatureProvider is ContextOnlyNovaeProvider


def test_niche_provider_physically_excludes_query_rows(tmp_path):
    adata = FakeAdata(["a", "b", "c", "d"], [[1], [2], [1000], [2000]])
    seen = []

    def fake_niche_fn(context):
        seen.append(context.obs_names.tolist())
        return np.zeros((context.X.shape[0], 1), dtype=np.float32)

    provider = ContextOnlyFeatureProvider(
        adata, cache_dir=tmp_path, sample_id="S", feature_fn=fake_niche_fn
    )
    mask = np.array([True, True, False, False])
    labels = provider(mask)
    assert labels.shape == (2, 1)
    assert seen[0] == ["a", "b"]

    again = provider(mask)
    assert np.array_equal(again, labels)
    assert len(seen) == 1  # second call served from cache, not recomputed
    print("[niche_candidate] OK — niche labels are computed on the context-only "
          "subgraph and physically exclude query rows, exactly like Novae features")


def test_model_uses_niche_candidate():
    assert model_uses_niche_candidate({}) is False
    assert model_uses_niche_candidate({"use_niche_candidate": False}) is False
    assert model_uses_niche_candidate({"use_niche_candidate": True}) is True
    print("[niche_candidate] OK — model_uses_niche_candidate reads the model params correctly")


def test_niche_mode_requires_double_opt_in():
    params = {"use_niche_candidate": True}
    cfg = SimpleNamespace(data={"niche_mode": "disabled"})
    with pytest.raises(ValueError, match="leak"):
        niche_input_mode(cfg, params)

    cfg = SimpleNamespace(data={"niche_mode": "nonsense"})
    with pytest.raises(ValueError, match="unknown"):
        niche_input_mode(cfg, params)

    cfg = SimpleNamespace(data={"niche_mode": "context_only"})
    assert niche_input_mode(cfg, params) == "context_only"

    # A model that doesn't request the niche candidate at all is never
    # gated, regardless of data.niche_mode.
    cfg = SimpleNamespace(data={"niche_mode": "disabled"})
    assert niche_input_mode(cfg, {"use_niche_candidate": False}) == "disabled"
    print("[niche_candidate] OK — use_niche_candidate requires an explicit "
          "data.niche_mode=context_only opt-in, exactly like Novae's own gate")


if __name__ == "__main__":
    test_context_only_feature_provider_is_the_same_engine_novae_uses()
    test_niche_provider_physically_excludes_query_rows(
        __import__("pathlib").Path(__import__("tempfile").mkdtemp())
    )
    test_model_uses_niche_candidate()
    test_niche_mode_requires_double_opt_in()
    print("\nAll niche-candidate leakage-guard tests passed.")
