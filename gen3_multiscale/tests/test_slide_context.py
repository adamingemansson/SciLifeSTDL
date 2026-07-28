"""Phase 3 (multiscale spatial-field handoff): mask-aware WSI tile
overlap removal. Adapted from tests/test_hierarchical_missing_tissue.py's
existing coverage of the same (copied, §"Reused audited infrastructure")
slide_context.py logic -- kept self-contained here rather than imported,
since gen3_multiscale's copy must be independently verified, not just
assumed identical."""
import numpy as np

from gen3_multiscale.data.slide_context import nonoverlapping_context_patch_mask, visible_slide_context


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
