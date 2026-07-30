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


# 19th Codex re-audit (Step 5 Part 2, remaining launch blocker #3):
# load_slide_context now validates every provenance field, not just a
# nonblank state_dict_sha256 -- these defaults must therefore be
# well-formed (a real 40-hex-char commit SHA, the exact expected repo id
# and preprocessing spec, a supported schema version) for every existing
# test that doesn't deliberately corrupt one field.
_VALID_HF_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars

_TILE_ENCODER_PROVENANCE_KWARGS = dict(
    tile_encoder_hf_repo_id=np.asarray("prov-gigapath/prov-gigapath"),
    tile_encoder_hf_revision=np.asarray(_VALID_HF_REVISION),
    tile_encoder_timm_version=np.asarray("1.0.3"),
    tile_encoder_preprocessing_spec=np.asarray("centercrop224_no_resize_v2_2026-07-24"),
    tile_encoder_state_dict_sha256=np.asarray("a" * 64),
    tile_encoder_schema_version=np.asarray(1),
)


def _write_dense_wsi_cache(path, features, coords, tile_size=256.0, **extra_provenance):
    kwargs = dict(_TILE_ENCODER_PROVENANCE_KWARGS)
    kwargs.update(extra_provenance)
    np.savez(
        path, features=features, coords=coords, tile_size=np.asarray(tile_size),
        coords_are_centers=np.asarray(False), **kwargs,
    )


def _load_edge_alignment_fixture(tmp_path, spot_coords):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    features = np.ones((1, 1536), dtype=np.float32)
    tile_coords = np.asarray([[0.0, 0.0]], dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, tile_coords,
        level0_coords=tile_coords,
        level0_tile_size=np.asarray(256.0),
        wsi_dimensions=np.asarray([1024.0, 1024.0]),
    )
    cfg = OmegaConf.create({
        "data": {
            "slide_context_source": "dense_wsi_cache",
            "hest_data_dir": str(tmp_path / "hest1k"),
        }
    })
    return load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_allows_a_rare_subtile_crop_edge_offset(tmp_path):
    spots = np.repeat(np.asarray([[10.0, 10.0]], dtype=np.float32), 1000, axis=0)
    spots[-1] = [-139.0, 10.0]  # 0.1% outside, less than one 256px tile
    context = _load_edge_alignment_fixture(tmp_path, spots)
    assert context["source"] == "dense_wsi_cache"


def test_load_slide_context_rejects_a_far_crop_edge_offset(tmp_path):
    spots = np.repeat(np.asarray([[10.0, 10.0]], dtype=np.float32), 1000, axis=0)
    spots[-1] = [-300.0, 10.0]  # beyond one 256px tile
    with pytest.raises(ValueError, match="maximum edge offset"):
        _load_edge_alignment_fixture(tmp_path, spots)


def test_load_slide_context_rejects_systematic_crop_edge_offsets(tmp_path):
    spots = np.repeat(np.asarray([[10.0, 10.0]], dtype=np.float32), 1000, axis=0)
    spots[-10:, 0] = -20.0  # 1% outside, despite each offset being small
    with pytest.raises(ValueError, match="allowed: <=0.5%"):
        _load_edge_alignment_fixture(tmp_path, spots)


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
        **_TILE_ENCODER_PROVENANCE_KWARGS,
    )
    ctx_a = load_slide_context(cfg, "S0", None, spot_coords)

    np.savez(
        cache_dir / "S0.npz", features=features, coords=coords, tile_size=np.asarray(256.0),
        coords_are_centers=np.asarray(False),
        level0_coords=np.asarray([[0, 0], [1024, 0]], dtype=np.float32),  # changed ONLY this
        level0_tile_size=np.asarray(256.0),
        **_TILE_ENCODER_PROVENANCE_KWARGS,
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


def test_load_slide_context_rejects_duplicate_level0_coords_even_with_unique_coords(tmp_path):
    """18th Codex re-audit (Step 5 Part 2, "Other real gaps"), CONFIRMED
    real: only `coords` (the LongNet frame) was checked for duplicates
    -- `mask_coords` (level0_coords) is an independently-sourced field
    for a dense_wsi_cache and could contain duplicates of its own even
    when `coords` has none."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    unique_coords = np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32)  # no duplicates here
    duplicate_level0 = np.asarray([[0, 0], [0, 0], [512, 0]], dtype=np.float32)  # but here
    features = np.ones((3, 1536), dtype=np.float32)
    kwargs = dict(_TILE_ENCODER_PROVENANCE_KWARGS)
    np.savez(
        cache_dir / "S0.npz", features=features, coords=unique_coords, tile_size=np.asarray(256.0),
        coords_are_centers=np.asarray(False), level0_coords=duplicate_level0,
        level0_tile_size=np.asarray(256.0), **kwargs,
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate level0_coords"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_requires_tile_encoder_provenance(tmp_path):
    """18th Codex re-audit (Step 5 Part 2 launch blocker #2), CONFIRMED
    real: a dense_wsi_cache written before this fix (no tile-encoder
    provenance fields at all) must be rejected explicitly -- rebuilding
    is required, not a silent "assume it's fine"."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    np.savez(  # deliberately WITHOUT any tile_encoder_* field -- an old-format cache
        cache_dir / "S0.npz", features=features, coords=coords, tile_size=np.asarray(256.0),
        coords_are_centers=np.asarray(False),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_hf_repo_id"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_id_changes_when_only_the_tile_encoder_provenance_changes(tmp_path):
    """18th Codex re-audit (Step 5 Part 2 launch blocker #2), CONFIRMED
    real: a cache regenerated with a DIFFERENT tile encoder (or a fixed
    preprocessing bug), with IDENTICAL features/coords/tile_size, must
    still get a new context_id -- otherwise two caches produced with
    different tile-encoder weights/preprocessing could both appear
    equally valid under the same identity."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    _write_dense_wsi_cache(cache_dir / "S0.npz", features, coords)
    ctx_a = load_slide_context(cfg, "S0", None, spot_coords)

    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_state_dict_sha256=np.asarray("b" * 64),  # changed ONLY this
    )
    ctx_b = load_slide_context(cfg, "S0", None, spot_coords)

    assert ctx_a["context_id"] != ctx_b["context_id"]


def test_load_slide_context_exposes_real_tile_encoder_provenance_for_dense_cache(tmp_path):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(cache_dir / "S0.npz", features, coords)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    ctx = load_slide_context(cfg, "S0", None, spot_coords)
    provenance = ctx["tile_encoder_provenance"]
    assert provenance["hf_repo_id"] == "prov-gigapath/prov-gigapath"
    assert provenance["hf_revision"] == _VALID_HF_REVISION
    assert provenance["state_dict_sha256"] == "a" * 64
    assert provenance["schema_version"] == 1


def test_load_slide_context_rejects_a_blank_tile_encoder_state_dict_sha256(tmp_path):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(cache_dir / "S0.npz", features, coords, tile_encoder_state_dict_sha256=np.asarray("   "))
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_state_dict_sha256"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_spot_aligned_has_no_tile_encoder_provenance():
    """spot_aligned is an explicit diagnostic fallback with no
    accompanying tile-encoder metadata to validate -- explicitly None,
    never fabricated."""
    cfg = OmegaConf.create({"data": {"slide_context_source": "spot_aligned"}})
    spot_features = np.ones((2, 1536), dtype=np.float32)
    spot_coords = np.asarray([[10, 10], [20, 20]], dtype=np.float32)
    ctx = load_slide_context(cfg, "S0", spot_features, spot_coords)
    assert ctx["tile_encoder_provenance"] is None


def test_load_slide_context_rejects_an_unexpected_tile_encoder_hf_repo_id(tmp_path):
    """19th Codex re-audit (Step 5 Part 2, remaining launch blocker #3):
    a cache claiming a DIFFERENT source repository must be refused, not
    silently trusted -- there is nothing else tying the cached features
    to prov-gigapath/prov-gigapath specifically."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_hf_repo_id=np.asarray("some-other-org/some-other-model"),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_hf_repo_id"):
        load_slide_context(cfg, "S0", None, spot_coords)


@pytest.mark.parametrize("bad_revision", [
    "main", "latest", "unpinned", "deadbeef", "a" * 39, "A" * 40, "g" * 40,
])
def test_load_slide_context_rejects_a_non_immutable_tile_encoder_hf_revision(tmp_path, bad_revision):
    """19th Codex re-audit (Step 5 Part 2, remaining launch blockers
    #1-3): a cache recording a moving ref (or malformed string) instead
    of a real, resolved, immutable 40-character lowercase hex commit SHA
    must be rejected -- this is the same discipline
    scripts/precompute_gigapath_wsi_tiles.py now enforces at build time,
    re-checked here so an old/hand-edited cache can't bypass it."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_hf_revision=np.asarray(bad_revision),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_hf_revision"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_rejects_a_blank_tile_encoder_timm_version(tmp_path):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_timm_version=np.asarray("None"),  # str(None) -- best-effort load-time fallback
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_timm_version"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_rejects_an_unexpected_tile_encoder_preprocessing_spec(tmp_path):
    """A cache whose recorded preprocessing string doesn't match the
    current real pipeline (src.models.conditioning._GIGAPATH_PREPROCESS_
    VERSION) must be refused -- it was built with different (possibly
    stale) pixel-processing logic and would silently produce features
    that mean something different from a freshly-built cache."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_preprocessing_spec=np.asarray("some_stale_preprocessing_v0"),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_preprocessing_spec"):
        load_slide_context(cfg, "S0", None, spot_coords)


@pytest.mark.parametrize("bad_sha", ["", "   ", "a" * 63, "a" * 65, "g" * 64, "A" * 64])
def test_load_slide_context_rejects_a_malformed_tile_encoder_state_dict_sha256(tmp_path, bad_sha):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_state_dict_sha256=np.asarray(bad_sha),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_state_dict_sha256"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_rejects_an_unsupported_tile_encoder_schema_version(tmp_path):
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_schema_version=np.asarray(999),
    )
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="tile_encoder_schema_version"):
        load_slide_context(cfg, "S0", None, spot_coords)


def test_load_slide_context_id_changes_when_only_a_non_state_dict_provenance_field_changes(tmp_path):
    """19th Codex re-audit (Step 5 Part 2, remaining launch blocker #4),
    CONFIRMED real: content_digest previously only hashed
    state_dict_sha256 -- hf_revision (and every other provenance field)
    was validated but NOT bound into context_id. Two caches with
    IDENTICAL weights (same state_dict_sha256) but a DIFFERENT recorded
    revision -- e.g. one built at commit A, one at commit B, that happen
    to produce byte-identical weights -- must still get different
    context_ids, since the full provenance object is now what's hashed,
    not just one field of it."""
    cache_dir = tmp_path / "hest1k" / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True)
    coords = np.asarray([[0, 0], [256, 0]], dtype=np.float32)
    features = np.ones((2, 1536), dtype=np.float32)
    cfg = OmegaConf.create({"data": {"slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_path / "hest1k")}})
    spot_coords = np.asarray([[10, 10]], dtype=np.float32)

    _write_dense_wsi_cache(cache_dir / "S0.npz", features, coords)  # hf_revision = _VALID_HF_REVISION
    ctx_a = load_slide_context(cfg, "S0", None, spot_coords)

    other_revision = "1234567890abcdef1234567890abcdef12345678"
    _write_dense_wsi_cache(
        cache_dir / "S0.npz", features, coords,
        tile_encoder_hf_revision=np.asarray(other_revision),  # changed ONLY this; state_dict_sha256 unchanged
    )
    ctx_b = load_slide_context(cfg, "S0", None, spot_coords)

    assert ctx_a["context_id"] != ctx_b["context_id"]
    assert ctx_a["tile_encoder_provenance"]["state_dict_sha256"] == ctx_b["tile_encoder_provenance"]["state_dict_sha256"]
