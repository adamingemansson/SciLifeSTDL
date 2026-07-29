"""Phase 3 (multiscale spatial-field handoff): mask-aware WSI tile
overlap removal. Adapted from tests/test_hierarchical_missing_tissue.py's
existing coverage of the same (copied, §"Reused audited infrastructure")
slide_context.py logic -- kept self-contained here rather than imported,
since gen3_multiscale's copy must be independently verified, not just
assumed identical."""
import numpy as np
import pytest
from omegaconf import OmegaConf

from gen3_multiscale.data.slide_context import (
    load_slide_context, nonoverlapping_context_patch_mask, tile_centers, visible_slide_context,
)


def _slide_context():
    # Three 256px tiles at x=0,256,512. Query at x=384 intersects only the
    # middle tile when represented by a 224px missing patch.
    return {
        "features": np.ones((3, 1536), dtype=np.float32),
        "coords": np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32),
        "mask_coords": np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32),
        "tile_size": 256.0,
        "mask_tile_size": 256.0,
        "coords_are_centers": False,
        "context_id": "unit-test",
        "source": "dense_wsi_cache",
    }


def test_target_zero_removes_every_wsi_tile_intersecting_the_hole():
    visible = visible_slide_context(
        _slide_context(), np.asarray([[384, 128, 0]], dtype=np.float32),
        image_mode="target_zero", query_patch_size=224.0,
    )
    assert visible["available"]
    assert visible["n_total"] == 3
    assert visible["n_visible"] == 2
    assert np.array_equal(visible["coords"][:, 0], np.asarray([0, 512]))


def _slide_context_with_differing_mpp():
    """17th Codex re-audit (Step 5 Part 2 launch blocker #1): a real
    dense_wsi_cache can have `coords` (GigaPath's target-MPP frame) on a
    genuinely different scale/origin than `mask_coords` (level-0/HEST-
    aligned) whenever the source slide's MPP differs from 0.5 um/px.
    `coords == mask_coords` (as `_slide_context()` above uses) would
    never catch a caller that accidentally mixed the two frames up."""
    level0 = np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32)
    longnet = level0 * 1.7 + np.asarray([9000.0, 9000.0], dtype=np.float32)  # different MPP scale + origin
    return {
        "features": np.ones((3, 1536), dtype=np.float32),
        "coords": longnet,
        "mask_coords": level0,
        "tile_size": 256.0,
        "mask_tile_size": 256.0,
        "coords_are_centers": False,
        "context_id": "unit-test-differing-mpp",
        "source": "dense_wsi_cache",
    }


def test_visible_slide_context_returns_level0_coords_independent_of_the_longnet_frame():
    """17th Codex re-audit (Step 5 Part 2 launch blocker #1), CONFIRMED:
    a prior version of visible_slide_context returned ONLY `coords`
    (the LongNet frame) -- callers had no way to derive regional
    coordinates in the level-0/HEST-aligned frame without incorrectly
    reusing the LongNet one. `level0_coords` must be the real tile
    CENTERS in the mask_coords frame (here, non-centers with tile_size
    256 -- center = corner + 128), independent of `coords`'s own scale."""
    ctx = _slide_context_with_differing_mpp()
    visible = visible_slide_context(
        ctx, np.asarray([[384 * 1.0, 128, 0]], dtype=np.float32),  # level-0-frame hole coordinates
        image_mode="target_zero", query_patch_size=224.0,
    )
    # The hole overlap test itself runs in the level-0/mask frame -- same
    # result as test_target_zero_removes_every_wsi_tile_intersecting_the_hole.
    assert visible["n_visible"] == 2
    expected_level0_centers = np.asarray([[128.0, 128.0], [640.0, 128.0]], dtype=np.float32)  # corner + tile_size/2
    assert np.allclose(visible["level0_coords"], expected_level0_centers)
    # coords stays the untouched LongNet frame -- large-magnitude, on a
    # completely different scale from level0_coords.
    expected_longnet = np.asarray([[0.0, 0.0], [512.0, 0.0]], dtype=np.float32) * 1.7 + np.asarray([9000.0, 9000.0])
    assert np.allclose(visible["coords"], expected_longnet)
    assert not np.allclose(visible["coords"], visible["level0_coords"])


def test_tile_centers_computes_the_complete_unmasked_level0_center_set():
    ctx = _slide_context_with_differing_mpp()
    centers = tile_centers(ctx)
    assert np.allclose(centers, np.asarray([[128.0, 128.0], [384.0, 128.0], [640.0, 128.0]], dtype=np.float32))


def test_all_zero_removes_the_complete_slide_context():
    visible = visible_slide_context(
        _slide_context(), np.asarray([[384, 128, 0]], dtype=np.float32),
        image_mode="all_zero", query_patch_size=224.0,
    )
    assert visible == {"available": False}


def test_full_image_mode_keeps_every_tile():
    visible = visible_slide_context(
        _slide_context(), np.asarray([[384, 128, 0]], dtype=np.float32),
        image_mode="full", query_patch_size=224.0,
    )
    assert visible["n_visible"] == 3


def test_hole_covering_every_tile_raises_rather_than_returning_empty_content():
    huge_hole = np.asarray([[0, 0, 0], [256, 0, 0], [512, 0, 0]], dtype=np.float32)
    try:
        visible_slide_context(
            _slide_context(), huge_hole, image_mode="target_zero", query_patch_size=1000.0,
        )
        assert False, "expected a ValueError when every WSI tile is removed"
    except ValueError as exc:
        assert "removed every WSI context tile" in str(exc)


def test_local_context_patch_overlap_is_removed():
    context_coords = np.asarray([[0, 0, 0], [700, 0, 0]], dtype=np.float32)
    query_coords = np.asarray([[100, 0, 0]], dtype=np.float32)
    keep = nonoverlapping_context_patch_mask(context_coords, query_coords, patch_size=224.0)
    # 224px patches: half-size 112 each, overlap limit = 224. The first
    # context spot (x=0, distance 100 from the query) overlaps -- its own
    # footprint intersects the hole and must be removed. The second
    # (x=700, distance 600) is well clear.
    assert list(keep) == [False, True]


def test_visible_slide_context_id_changes_when_the_visible_tile_set_changes():
    """15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED: a
    prior version's context_id only appended the literal image_mode
    string -- two DIFFERENT query holes on the SAME cached slide produce
    two different VISIBLE tile sets but would collide on an identical
    context_id, risking a cached LongNet global vector computed for the
    wrong visible-tile set being silently reused."""
    ctx = _slide_context()
    hole_a = visible_slide_context(
        ctx, np.asarray([[384, 128, 0]], dtype=np.float32), image_mode="target_zero", query_patch_size=224.0,
    )
    hole_b = visible_slide_context(
        ctx, np.asarray([[0, 0, 0]], dtype=np.float32), image_mode="target_zero", query_patch_size=224.0,
    )
    assert hole_a["n_visible"] != hole_b["n_visible"] or not np.array_equal(hole_a["coords"], hole_b["coords"])
    assert hole_a["context_id"] != hole_b["context_id"]


def test_visible_slide_context_id_is_stable_for_the_same_hole():
    ctx = _slide_context()
    query = np.asarray([[384, 128, 0]], dtype=np.float32)
    a = visible_slide_context(ctx, query, image_mode="target_zero", query_patch_size=224.0)
    b = visible_slide_context(ctx, query, image_mode="target_zero", query_patch_size=224.0)
    assert a["context_id"] == b["context_id"]


def _write_dense_wsi_cache(path, features, coords, tile_size=256.0):
    np.savez(
        path, features=features, coords=coords, tile_size=np.asarray(tile_size),
        coords_are_centers=np.asarray(False),
    )


def test_load_slide_context_id_is_bound_to_real_content_not_just_file_stat(tmp_path):
    """15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED: a
    prior version identified the cache by file path+size+mtime -- a
    proxy for content, not content itself. Two files with genuinely
    different tile feature content (same shape) must produce different
    context_ids."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)

    features_a = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(cache_dir / "S0.npz", features_a, coords)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)
    ctx_a = load_slide_context(cfg, "S0", None, spot_coords)

    features_b = features_a * 2.0  # genuinely different content, same shape/size
    _write_dense_wsi_cache(cache_dir / "S0.npz", features_b, coords)
    ctx_b = load_slide_context(cfg, "S0", None, spot_coords)

    assert ctx_a["context_id"] != ctx_b["context_id"]


def test_load_slide_context_id_changes_when_only_the_level0_mask_fields_change(tmp_path):
    """16th Codex re-audit (Step 5 Part 2 acceptance criteria), CONFIRMED:
    a prior content-hash version only covered features/coords/tile_size
    -- NOT mask_coords/mask_tile_size/coords_are_centers, which
    independently drive WHICH tiles visible_slide_context keeps for a
    given hole (a dense_wsi_cache's level0_coords/level0_tile_size can
    differ entirely from its coords/tile_size). A cache changed ONLY in
    those masking-relevant level0 fields must still get a new context_id."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    np.savez(
        cache_dir / "S0.npz", features=features, coords=coords, tile_size=np.asarray(256.0),
        coords_are_centers=np.asarray(False),
        level0_coords=np.asarray([[0, 0], [512, 0]], dtype=np.float32),  # different level0 mask coords
        level0_tile_size=np.asarray(256.0),
    )
    ctx_a = load_slide_context(cfg, "S0", None, spot_coords)

    np.savez(
        cache_dir / "S0.npz", features=features, coords=coords, tile_size=np.asarray(256.0),
        coords_are_centers=np.asarray(False),
        level0_coords=np.asarray([[0, 0], [1024, 0]], dtype=np.float32),  # changed ONLY this
        level0_tile_size=np.asarray(256.0),
    )
    ctx_b = load_slide_context(cfg, "S0", None, spot_coords)

    assert ctx_a["context_id"] != ctx_b["context_id"]


def test_load_slide_context_rejects_duplicate_tile_coordinates(tmp_path):
    """16th Codex re-audit (Step 5 Part 2 acceptance criteria), CONFIRMED:
    no check existed for duplicate tile coordinates -- a corrupted or
    badly-generated cache with two tiles at the identical position would
    silently double-count that region's real contribution."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    duplicate_coords = np.asarray([[0, 0], [0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((3, 1536), dtype=np.float32)
    _write_dense_wsi_cache(cache_dir / "S0.npz", features, duplicate_coords)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate tile coordinates"):
        load_slide_context(cfg, "S0", None, spot_coords)
