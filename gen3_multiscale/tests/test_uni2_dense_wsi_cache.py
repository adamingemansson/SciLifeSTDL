"""Item 1 (six-launch-blocker audit): a real UNI2 dense-WSI cache loader
that produces a `data.slide_context.load_slide_context`-shaped dict, so
it plugs into `example_builder.build_spatial_field_example`'s existing
`slide_context=` parameter and reuses the same audited hole-overlap
masking (`visible_slide_context`) unmodified. `build_uni2_dense_wsi_cache`
itself needs a real WSI reader (openslide/tiffslide, neither installed in
this sandbox) and is therefore only exercised structurally elsewhere;
`load_uni2_dense_wsi_context` is fully real-code-path tested here against
hand-written cache fixtures, mirroring tests/test_slide_context.py's own
`_write_dense_wsi_cache` convention.
"""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.data.slide_context import tile_centers, visible_slide_context
from gen3_multiscale.gen4.uni2_dense_wsi_cache import load_uni2_dense_wsi_context


def _write_uni2_dense_cache(path, features, coords, level0_coords, wsi_dimensions, tile_size=256.0, level0_tile_size=256.0):
    np.savez(
        path, features=features, coords=coords, level0_coords=level0_coords,
        tile_size=np.asarray(tile_size, dtype=np.float32), level0_tile_size=np.asarray(level0_tile_size, dtype=np.float32),
        coords_are_centers=np.asarray(False), wsi_dimensions=np.asarray(wsi_dimensions, dtype=np.int64),
        uni2_checkpoint_sha256=np.asarray("uni2" * 16), uni2_pinned_revision=np.asarray("0" * 40),
        uni2_package_version=np.asarray("1.0.3"), uni2_preprocessing_spec=np.asarray("uni2_tile_v1:vit_giant_patch14_224:resize224:imagenet_norm"),
        uni2_output_dim=np.asarray(1536), uni2_schema_version=np.asarray(1),
    )


def _cfg(tmp_path):
    return OmegaConf.create({"data": {"hest_data_dir": str(tmp_path / "hest1k")}})


def test_load_uni2_dense_wsi_context_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="UNI2 dense-WSI cache missing"):
        load_uni2_dense_wsi_context(_cfg(tmp_path), "S0", np.zeros((1, 2), dtype=np.float32))


def test_load_uni2_dense_wsi_context_missing_fields_raises(tmp_path):
    cache_dir = tmp_path / "hest1k" / "uni2_dense_wsi_cache"
    cache_dir.mkdir(parents=True)
    np.savez(cache_dir / "S0.npz", features=np.ones((1, 1536), dtype=np.float32))
    with pytest.raises(ValueError, match="missing fields"):
        load_uni2_dense_wsi_context(_cfg(tmp_path), "S0", np.zeros((1, 2), dtype=np.float32))


def test_load_uni2_dense_wsi_context_round_trips_into_slide_context_shape(tmp_path):
    cache_dir = tmp_path / "hest1k" / "uni2_dense_wsi_cache"
    cache_dir.mkdir(parents=True)
    features = np.ones((3, 1536), dtype=np.float32)
    coords = np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32)
    spots = np.repeat(np.asarray([[10.0, 10.0]], dtype=np.float32), 10, axis=0)
    _write_uni2_dense_cache(cache_dir / "S0.npz", features, coords, coords, wsi_dimensions=[1024, 1024])
    context = load_uni2_dense_wsi_context(_cfg(tmp_path), "S0", spots)
    assert context["source"] == "uni2_dense_wsi_cache"
    assert context["tile_encoder_provenance"]["uni2_output_dim"] == 1536
    # Exact same shape/keys load_slide_context returns -- reuse of the
    # unmodified, already-audited masking logic downstream depends on this.
    for key in ("features", "coords", "mask_coords", "tile_size", "mask_tile_size", "coords_are_centers", "context_id", "source"):
        assert key in context
    # visible_slide_context/tile_centers (the real hole-overlap masking
    # code) run completely unmodified against this dict.
    centers = tile_centers(context)
    assert centers.shape == (3, 2)
    visible = visible_slide_context(
        context, np.asarray([[384.0, 128.0, 0.0]], dtype=np.float32), image_mode="target_zero", query_patch_size=224.0,
    )
    assert visible["available"]
    assert visible["n_visible"] == 2


def test_load_uni2_dense_wsi_context_rejects_duplicate_tile_coordinates(tmp_path):
    cache_dir = tmp_path / "hest1k" / "uni2_dense_wsi_cache"
    cache_dir.mkdir(parents=True)
    features = np.ones((2, 1536), dtype=np.float32)
    coords = np.asarray([[0, 0], [0, 0]], dtype=np.float32)  # duplicate
    spots = np.repeat(np.asarray([[10.0, 10.0]], dtype=np.float32), 10, axis=0)
    _write_uni2_dense_cache(cache_dir / "S0.npz", features, coords, coords, wsi_dimensions=[1024, 1024])
    with pytest.raises(ValueError, match="duplicate tile coordinates"):
        load_uni2_dense_wsi_context(_cfg(tmp_path), "S0", spots)


def test_load_uni2_dense_wsi_context_rejects_mismatched_coordinate_frame(tmp_path):
    cache_dir = tmp_path / "hest1k" / "uni2_dense_wsi_cache"
    cache_dir.mkdir(parents=True)
    features = np.ones((3, 1536), dtype=np.float32)
    coords = np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32)
    # Spots nowhere near any cached tile -- a real coordinate-frame mismatch.
    spots = np.repeat(np.asarray([[100_000.0, 100_000.0]], dtype=np.float32), 10, axis=0)
    _write_uni2_dense_cache(cache_dir / "S0.npz", features, coords, coords, wsi_dimensions=[200_000, 200_000])
    with pytest.raises(ValueError, match="align near retained WSI tissue"):
        load_uni2_dense_wsi_context(_cfg(tmp_path), "S0", spots)
