import inspect

import numpy as np

from gen3_multiscale.scripts.plot_mk_gene_spatial_maps import (
    _global_scales,
    _load_cache,
    _load_model_and_samples,
    _metrics,
    _save_cache,
    _select_split_sample_ids,
)


def _records():
    return [
        {
            "sample_id": "slide_a",
            "coords": np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            "target": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            "prediction": np.asarray([0.0, 1.5, 1.5], dtype=np.float32),
        },
        {
            "sample_id": "slide_b",
            "coords": np.asarray([[2, 2], [3, 2]], dtype=np.float32),
            "target": np.asarray([3.0, 4.0], dtype=np.float32),
            "prediction": np.asarray([30.0, 40.0], dtype=np.float32),
        },
    ]


def test_single_gene_cache_round_trip_is_pickle_free_and_identity_bound(tmp_path):
    metadata = {"kind": "test", "weights_sha256": "abc", "gene": "GENE"}
    path = tmp_path / "values.npz"
    _save_cache(path, metadata, _records())

    loaded = _load_cache(path, metadata)
    assert [row["sample_id"] for row in loaded] == ["slide_a", "slide_b"]
    np.testing.assert_array_equal(loaded[0]["target"], _records()[0]["target"])
    assert _load_cache(path, {**metadata, "weights_sha256": "different"}) is None


def test_standardized_scales_depend_on_targets_not_model_predictions():
    records = _records()
    first = _global_scales(records)
    for row in records:
        row["prediction"] *= 1000
    assert _global_scales(records) == first
    assert first[1] > first[0]
    assert first[2] == first[1] - first[0]


def test_slide_metrics_use_unclipped_values():
    result = _metrics(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 1.0, 4.0]),
    )
    assert result["rmse"] == np.sqrt(4.0 / 3.0)
    assert result["mae"] == 2.0 / 3.0
    assert np.isfinite(result["pcc"])


def test_model_loader_accepts_an_explicit_sample_subset():
    signature = inspect.signature(_load_model_and_samples)
    assert signature.parameters["sample_ids"].default is None


def test_sample_selection_keeps_an_explicit_subset_in_requested_order():
    manifest = {"validation_sample_ids": ["slide_a", "slide_b", "slide_c"]}
    available, selected = _select_split_sample_ids(
        manifest, split="validation", requested=["slide_c", "slide_a"],
    )
    assert available == ["slide_a", "slide_b", "slide_c"]
    assert selected == ["slide_c", "slide_a"]


def test_renderer_keeps_legacy_marker_scale_as_default():
    signature = inspect.signature(__import__(
        "gen3_multiscale.scripts.plot_mk_gene_spatial_maps",
        fromlist=["_render"],
    )._render)
    assert signature.parameters["marker_size_scale"].default == 1.0
