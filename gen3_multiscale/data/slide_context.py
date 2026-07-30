"""Mask-aware WSI tile-cache loading for hierarchical missing tissue.

2026-07-28: copied VERBATIM from src/data/slide_context.py into
gen3_multiscale/ -- see data/hest1k_catalog.py's identical copy-
provenance note in this package for why (the multiscale spatial-field
handoff's "reuse audited data hygiene and cache logic" instruction, and
its Phase 3 instruction to "reuse the audited dense-tile footprint
calculation" rather than reimplement mask-aware WSI tile filtering).
Includes the conservative axis-aligned tile/hole overlap test
(_overlaps_query_hole) already used to guarantee no WSI tile whose
footprint intersects the missing region ever reaches GigaPath LongNet.
Do not let this drift from src/data/slide_context.py without a
deliberate reason.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

# Must match src.models.conditioning's real GigaPath tile-encoder identity
# constants exactly. gen3_multiscale deliberately never imports from src/
# (see loaders.py's own copy-provenance note in this package) -- kept as a
# synchronized copy instead of a cross-package import. If either drifts,
# every real dense_wsi_cache built with the current tile encoder starts
# failing this validation, which is the intended fail-closed behavior, not
# a bug to silently work around.
_EXPECTED_GIGAPATH_HF_REPO_ID = "prov-gigapath/prov-gigapath"
_EXPECTED_GIGAPATH_PREPROCESSING_SPEC = "centercrop224_no_resize_v2_2026-07-24"
_SUPPORTED_TILE_ENCODER_SCHEMA_VERSIONS = {1}
_HF_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_tile_encoder_provenance(source: str, tile_encoder_provenance: dict) -> None:
    """Fail-closed validation of a real GigaPath tile-encoder provenance
    dict (the same 6-field shape `gigapath_tile_encoder_provenance` --
    src.models.conditioning -- produces): expected repository, an
    immutable 40-hex-char commit SHA, a nonblank library version, the
    exact current preprocessing spec, a well-formed 64-hex-char weights
    digest, and a supported schema version.

    Factored out of load_slide_context's dense_wsi_cache branch (19th
    Codex re-audit, remaining launch blocker #3) so
    gen3_multiscale/data/spot_feature_cache.py's Gen3 spot-feature cache
    -- a genuinely different cache format -- can apply the IDENTICAL
    validation to its own tile-encoder provenance rather than
    reimplementing (and risking silently drifting from) these same six
    checks. `source` is a human-readable identifier (e.g. a file path)
    used only in error messages."""
    if tile_encoder_provenance["hf_repo_id"] != _EXPECTED_GIGAPATH_HF_REPO_ID:
        raise ValueError(
            f"{source} tile_encoder_hf_repo_id="
            f"{tile_encoder_provenance['hf_repo_id']!r}, expected "
            f"{_EXPECTED_GIGAPATH_HF_REPO_ID!r}"
        )
    if not _HF_COMMIT_SHA_RE.match(tile_encoder_provenance["hf_revision"]):
        raise ValueError(
            f"{source} tile_encoder_hf_revision="
            f"{tile_encoder_provenance['hf_revision']!r} is not a full 40-character "
            "lowercase hex Hugging Face commit SHA -- rebuild with a pinned, "
            "immutable --tile-encoder-revision"
        )
    if not tile_encoder_provenance["timm_version"].strip() or tile_encoder_provenance["timm_version"] == "None":
        raise ValueError(f"{source} has a blank/missing tile_encoder_timm_version")
    if tile_encoder_provenance["preprocessing_spec"] != _EXPECTED_GIGAPATH_PREPROCESSING_SPEC:
        raise ValueError(
            f"{source} tile_encoder_preprocessing_spec="
            f"{tile_encoder_provenance['preprocessing_spec']!r}, expected "
            f"{_EXPECTED_GIGAPATH_PREPROCESSING_SPEC!r} -- rebuild it with the current "
            "scripts/precompute_gigapath_wsi_tiles.py"
        )
    if not _SHA256_HEX_RE.match(tile_encoder_provenance["state_dict_sha256"]):
        raise ValueError(
            f"{source} tile_encoder_state_dict_sha256 is not a well-formed "
            "64-character lowercase hex SHA256 digest"
        )
    if tile_encoder_provenance["schema_version"] not in _SUPPORTED_TILE_ENCODER_SCHEMA_VERSIONS:
        raise ValueError(
            f"{source} tile_encoder_schema_version="
            f"{tile_encoder_provenance['schema_version']} is not supported "
            f"(supported: {sorted(_SUPPORTED_TILE_ENCODER_SCHEMA_VERSIONS)})"
        )


def _cache_path(cfg, sample_id: str) -> Path:
    configured = cfg.data.get("slide_context_cache_dir")
    if configured:
        root = Path(str(configured))
    else:
        cache_root = cfg.data.get("hest_cache_dir", cfg.data.hest_data_dir)
        root = Path(str(cache_root)) / "gigapath_slide_cache"
    return root / f"{sample_id}.npz"


def load_slide_context(
    cfg,
    sample_id: str,
    spot_features: np.ndarray | None,
    spot_coords: np.ndarray,
) -> dict | None:
    """Load the configured slide tiles without silently changing semantics.

    ``dense_wsi_cache`` is the intended biological path: a tissue-wide,
    non-overlapping WSI tile grid produced by
    ``scripts/precompute_gigapath_wsi_tiles.py``.  ``spot_aligned`` is an
    explicit diagnostic fallback over the ST-covered patch lattice; it is
    never presented as complete WSI coverage.
    """
    source = str(cfg.data.get("slide_context_source", "disabled"))
    if source == "disabled":
        return None
    if source == "spot_aligned":
        if spot_features is None or spot_features.ndim != 2:
            raise ValueError(
                "slide_context_source=spot_aligned requires precomputed GigaPath spot features"
            )
        tile_size = float(cfg.data.get("spot_patch_size_fullres", 224.0))
        features = np.asarray(spot_features, dtype=np.float32)
        # HEST spot coordinates are patch centers; Prov-GigaPath expects
        # level-0 tile coordinates. Convert explicitly instead of silently
        # shifting its 2-D positional bins by half a patch.
        coords = np.asarray(spot_coords[:, :2], dtype=np.float32) - tile_size / 2.0
        mask_coords = coords
        mask_tile_size = tile_size
        coords_are_centers = False
        identity = f"{sample_id}:spot_aligned:{features.shape[0]}"
        # 18th Codex re-audit (Step 5 Part 2, "Require spot and dense-WSI
        # caches to use the same tile-encoder provenance"): spot_features
        # arrives here as a bare precomputed array with no accompanying
        # metadata -- there is nothing to validate or bind, so
        # tile_encoder_provenance is explicitly None for this path, never
        # fabricated. spot_aligned is already documented (this function's
        # own docstring) as an explicit diagnostic fallback, never
        # production WSI coverage; cross-validating it against a real
        # dense-cache's provenance is deferred, not silently skipped.
        tile_encoder_provenance = None
    elif source == "dense_wsi_cache":
        path = _cache_path(cfg, sample_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"Dense WSI GigaPath cache missing for {sample_id}: {path}. Run "
                "scripts/precompute_gigapath_wsi_tiles.py before training."
            )
        cached = np.load(path, allow_pickle=False)
        required = {"features", "coords", "tile_size", "coords_are_centers"}
        # 18th Codex re-audit (Step 5 Part 2 launch blocker #2), CONFIRMED
        # real: the dense WSI cache stored no resolved tile-encoder
        # identity at all -- two caches built from different tile-encoder
        # weights/preprocessing (or a stale cache built before a
        # preprocessing fix) could both appear equally valid. Now
        # mandatory for every dense_wsi_cache -- fails closed on an old
        # cache built before this fix, rather than silently trusting it.
        required_tile_encoder_provenance = {
            "tile_encoder_hf_repo_id", "tile_encoder_hf_revision", "tile_encoder_timm_version",
            "tile_encoder_preprocessing_spec", "tile_encoder_state_dict_sha256", "tile_encoder_schema_version",
        }
        required = required | required_tile_encoder_provenance
        missing = sorted(required.difference(cached.files))
        if missing:
            raise ValueError(
                f"slide cache {path} is missing fields {missing} -- rebuild it with the current "
                "scripts/precompute_gigapath_wsi_tiles.py (real tile-encoder provenance is now "
                "mandatory)"
            )
        tile_encoder_provenance = {
            "hf_repo_id": str(cached["tile_encoder_hf_repo_id"]),
            "hf_revision": str(cached["tile_encoder_hf_revision"]),
            "timm_version": str(cached["tile_encoder_timm_version"]),
            "preprocessing_spec": str(cached["tile_encoder_preprocessing_spec"]),
            "state_dict_sha256": str(cached["tile_encoder_state_dict_sha256"]),
            "schema_version": int(np.asarray(cached["tile_encoder_schema_version"]).item()),
        }
        # 19th Codex re-audit (Step 5 Part 2, remaining launch blocker
        # #3), CONFIRMED real: only a nonblank state_dict_sha256 was
        # meaningfully checked -- hf_repo_id/hf_revision/timm_version/
        # preprocessing_spec/schema_version were read but never validated,
        # so a cache built against the wrong repo, an unpinned/malformed
        # revision, a blank library version, stale preprocessing, or an
        # unrecognized schema could all silently pass. Every field is now
        # validated explicitly, fail-closed, via the shared validator
        # (factored out in the 20th re-audit so
        # spot_feature_cache.py's Gen3 spot-feature cache applies the
        # identical checks rather than risking a second, drifting copy).
        validate_tile_encoder_provenance(f"slide cache {path}", tile_encoder_provenance)
        features = np.asarray(cached["features"], dtype=np.float32)
        coords = np.asarray(cached["coords"], dtype=np.float32)
        tile_size = float(np.asarray(cached["tile_size"]).item())
        coords_are_centers = bool(np.asarray(cached["coords_are_centers"]).item())
        # ``coords`` may be expressed in the 0.5-um/px virtual coordinate
        # system expected by GigaPath.  Hole filtering must instead happen
        # in the HEST level-0 coordinate frame used by spot coordinates.
        mask_coords = np.asarray(
            cached["level0_coords"] if "level0_coords" in cached.files else coords,
            dtype=np.float32,
        )
        mask_tile_size = float(
            np.asarray(
                cached["level0_tile_size"]
                if "level0_tile_size" in cached.files else cached["tile_size"]
            ).item()
        )
        wsi_dimensions = (
            np.asarray(cached["wsi_dimensions"], dtype=np.float64)
            if "wsi_dimensions" in cached.files else None
        )
        # 15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED: a
        # prior version identified the cache by file path+size+mtime --
        # a proxy for content, not content itself (a file replaced
        # in-place with different bytes but the same size, at a moment
        # that rounds to the same mtime granularity, would silently
        # collide). Hash the REAL tile-cache content.
        #
        # 16th Codex re-audit (Step 5 Part 2), CONFIRMED: the first
        # content-hash version only covered features/coords/tile_size --
        # NOT mask_coords/mask_tile_size/coords_are_centers, every one of
        # which independently affects WHICH tiles visible_slide_context
        # actually keeps (its hole-overlap test runs entirely in the
        # mask_coords/mask_tile_size/coords_are_centers frame, a
        # DIFFERENT coordinate system than coords/tile_size for a
        # dense_wsi_cache with a separate level0_coords/level0_tile_size).
        # A cache file changed ONLY in those masking-relevant fields
        # (same features/coords/tile_size) would have silently kept the
        # old context_id despite producing different visible tiles for
        # every hole. All six real fields are now hashed together.
        content_digest = hashlib.sha256()
        content_digest.update(np.ascontiguousarray(features).tobytes())
        content_digest.update(np.ascontiguousarray(coords).tobytes())
        content_digest.update(np.ascontiguousarray(mask_coords).tobytes())
        content_digest.update(str(tile_size).encode())
        content_digest.update(str(mask_tile_size).encode())
        content_digest.update(str(coords_are_centers).encode())
        # 18th Codex re-audit (Step 5 Part 2 launch blocker #2), EXTENDED
        # by the 19th re-audit (remaining launch blocker #4), CONFIRMED
        # real: the 18th-round fix only hashed state_dict_sha256, despite
        # its own comment claiming "the real tile-encoder identity" was
        # bound -- hf_repo_id/hf_revision/timm_version/preprocessing_spec/
        # schema_version were validated (above) but NOT folded into
        # content_digest, so a cache with identical weights but a
        # different (still-valid-looking) recorded revision/repo/
        # preprocessing string would silently collide on context_id. Hash
        # the COMPLETE canonical provenance object (json.dumps with
        # sort_keys=True is deterministic regardless of dict insertion
        # order) so ANY provenance field changing changes context_id.
        content_digest.update(json.dumps(tile_encoder_provenance, sort_keys=True).encode())
        identity = f"{sample_id}:dense:{content_digest.hexdigest()}"
        # 16th Codex re-audit (Step 5 Part 2), CONFIRMED: no check existed
        # for duplicate tile coordinates -- a corrupted or badly-generated
        # cache with two tiles at the identical position would silently
        # double-count that region's contribution to both regional
        # pooling and the LongNet global vector, and pool_regional_tokens
        # would attribute it to one grid cell twice.
        if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
            raise ValueError(
                f"dense WSI cache {path} contains duplicate tile coordinates -- refusing a "
                "corrupted/malformed tile cache"
            )
        # 18th Codex re-audit (Step 5 Part 2, "Other real gaps"),
        # CONFIRMED real: only `coords` (the LongNet frame) was checked
        # for duplicates -- `mask_coords` (the level-0/HEST-aligned
        # frame the hole-overlap test actually runs in) is an
        # independently-sourced field for a dense_wsi_cache with a
        # separate level0_coords, and could contain duplicates of its
        # own even when `coords` has none, silently double-counting a
        # region's contribution to hole-overlap filtering.
        if np.unique(mask_coords, axis=0).shape[0] != mask_coords.shape[0]:
            raise ValueError(
                f"dense WSI cache {path} contains duplicate level0_coords -- refusing a "
                "corrupted/malformed tile cache"
            )
    else:
        raise ValueError(
            "data.slide_context_source must be disabled, spot_aligned, or dense_wsi_cache"
        )

    if features.ndim != 2 or features.shape[1] != 1536:
        raise ValueError(
            f"slide context features must be [N,1536], got {features.shape} for {sample_id}"
        )
    if coords.shape != (features.shape[0], 2):
        raise ValueError(
            f"slide context coords must be [N,2] aligned with features, got {coords.shape}"
        )
    if mask_coords.shape != (features.shape[0], 2):
        raise ValueError(
            f"slide context level0_coords must be [N,2] aligned with features, got "
            f"{mask_coords.shape}"
        )
    if features.shape[0] < 1 or tile_size <= 0:
        raise ValueError(f"slide context for {sample_id} is empty or has invalid tile_size")
    if mask_tile_size <= 0:
        raise ValueError(f"slide context for {sample_id} has invalid level0_tile_size")
    if not np.isfinite(features).all() or not np.isfinite(coords).all() or not np.isfinite(mask_coords).all():
        raise ValueError(f"slide context for {sample_id} contains non-finite values")
    if source == "dense_wsi_cache":
        spot_xy = np.asarray(spot_coords[:, :2], dtype=np.float64)
        if wsi_dimensions is not None:
            if wsi_dimensions.shape != (2,) or np.any(wsi_dimensions <= 0):
                raise ValueError(f"slide cache for {sample_id} has invalid wsi_dimensions")
            inside = (
                (spot_xy[:, 0] >= 0) & (spot_xy[:, 0] < wsi_dimensions[0])
                & (spot_xy[:, 1] >= 0) & (spot_xy[:, 1] < wsi_dimensions[1])
            )
            # A few HEST spot centres can sit just outside a cropped WSI by
            # tens of level-0 pixels (for example, a Visium spot centred on
            # the crop boundary).  That is not a coordinate-frame mismatch.
            # Permit only a very small fraction of such points, and only when
            # every one is within one cached level-0 tile of the slide.  The
            # independent >=90% retained-tissue-tile check below remains
            # unchanged and catches systematic crop/origin mismatches.
            outside_fraction = float(np.mean(~inside))
            x_edge_distance = np.maximum(
                np.maximum(-spot_xy[:, 0], spot_xy[:, 0] - wsi_dimensions[0]),
                0.0,
            )
            y_edge_distance = np.maximum(
                np.maximum(-spot_xy[:, 1], spot_xy[:, 1] - wsi_dimensions[1]),
                0.0,
            )
            max_outside_distance = float(np.max(np.maximum(x_edge_distance, y_edge_distance)))
            if outside_fraction > 0.005 or max_outside_distance > mask_tile_size:
                raise ValueError(
                    f"{(~inside).sum()}/{len(inside)} ST spots fall outside the cached WSI; "
                    f"maximum edge offset is {max_outside_distance:.1f} level-0 pixels "
                    f"(allowed: <=0.5% of spots and <=one {mask_tile_size:.1f}px tile); "
                    "the H5AD and WSI coordinate frames likely do not match"
                )
        # Most measured spots must fall in, or immediately beside, a retained
        # tissue tile.  This catches the far more dangerous case where both
        # arrays have plausible positive coordinates but refer to different
        # crops/origins of the slide.
        tile_bins = {
            (int(np.floor(x / mask_tile_size)), int(np.floor(y / mask_tile_size)))
            for x, y in mask_coords
        }
        covered = []
        for x, y in spot_xy:
            bx, by = int(np.floor(x / mask_tile_size)), int(np.floor(y / mask_tile_size))
            covered.append(any(
                (bx + dx, by + dy) in tile_bins
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            ))
        coverage = float(np.mean(covered))
        if coverage < 0.90:
            raise ValueError(
                f"only {coverage:.1%} of {sample_id} ST spots align near retained WSI tissue "
                "tiles; refusing a likely mismatched WSI/H5AD coordinate frame"
            )
    return {
        "features": features,
        "coords": coords,
        "mask_coords": mask_coords,
        "tile_size": tile_size,
        "mask_tile_size": mask_tile_size,
        "coords_are_centers": coords_are_centers,
        "context_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "source": source,
        # 18th Codex re-audit (Step 5 Part 2 launch blocker #2): real for
        # dense_wsi_cache (validated above), explicitly None for
        # spot_aligned (nothing to validate) -- never fabricated. Exposed
        # so a caller/future manifest-or-preflight report (Step 8) can
        # bind it, and so real and diagnostic caches are distinguishable
        # by more than just `source`.
        "tile_encoder_provenance": tile_encoder_provenance,
    }


def _overlaps_query_hole(
    tile_centers: np.ndarray,
    tile_half_size: float,
    query_centers: np.ndarray,
    query_half_size: float,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Conservative axis-aligned tile/target-patch intersection test."""
    overlap = np.zeros(tile_centers.shape[0], dtype=bool)
    limit = float(tile_half_size + query_half_size)
    for start in range(0, tile_centers.shape[0], chunk_size):
        block = tile_centers[start : start + chunk_size]
        delta = np.abs(block[:, None, :] - query_centers[None, :, :])
        overlap[start : start + len(block)] = np.any(
            (delta[..., 0] < limit) & (delta[..., 1] < limit), axis=1
        )
    return overlap


def tile_centers(slide_context: dict) -> np.ndarray:
    """Real level-0/HEST-aligned tile CENTER coordinates for the
    COMPLETE (unmasked) tile set -- the same physical frame `spot_coords`
    (ST spot coordinates) are already expressed in, and the ONLY frame
    `example_builder.py` may validly derive regional-attention geometry
    from (17th Codex re-audit, Step 5 Part 2 launch blocker #1).

    `slide_context["coords"]` is a DIFFERENT frame: GigaPath LongNet's
    own target-MPP tile coordinates, used for real LongNet inference
    only. For a dense_wsi_cache with a genuinely different source MPP
    (i.e. `coords != mask_coords`), the two frames are not
    interchangeable -- computing regional coordinates from `coords`
    against a level-0 reference/scale would be physically invalid.
    `mask_coords`/`mask_tile_size` are always the level-0/HEST-aligned
    fields (see load_slide_context's own field documentation)."""
    mask_coords = slide_context["mask_coords"]
    mask_tile_size = float(slide_context["mask_tile_size"])
    if bool(slide_context["coords_are_centers"]):
        return mask_coords
    return mask_coords + mask_tile_size / 2.0


def visible_slide_context(
    slide_context: dict | None,
    query_coords: np.ndarray,
    image_mode: str,
    query_patch_size: float,
) -> dict:
    """Return only WSI tiles that could still exist after physical damage."""
    if slide_context is None or image_mode == "all_zero":
        return {"available": False}
    features = slide_context["features"]
    coords = slide_context["coords"]
    mask_tile_size = float(slide_context["mask_tile_size"])
    mask_centers = tile_centers(slide_context)

    visible = np.ones(features.shape[0], dtype=bool)
    if image_mode == "target_zero":
        query_xy = np.asarray(query_coords[:, :2], dtype=np.float32)
        visible &= ~_overlaps_query_hole(
            mask_centers, mask_tile_size / 2.0, query_xy, float(query_patch_size) / 2.0
        )
    elif image_mode not in {"full", "shuffled"}:
        raise ValueError(f"unsupported slide image mode {image_mode!r}")
    if not visible.any():
        raise ValueError("physical missing-tissue mask removed every WSI context tile")
    # 15th Codex re-audit (Step 5 acceptance criteria), CONFIRMED: a prior
    # version's context_id appended only the LITERAL mode string
    # (e.g. "target_zero"), never anything about WHICH tiles the hole
    # actually removed -- two different query holes on the SAME cached
    # slide produce two different VISIBLE tile sets but would collide on
    # an identical context_id, silently reusing a cached LongNet global
    # vector computed for the wrong visible-tile set. Bind the real,
    # ordered visible tile coordinates (post-filtering) into the id.
    visible_coords = coords[visible]
    visible_digest = hashlib.sha256(np.ascontiguousarray(visible_coords).tobytes()).hexdigest()[:24]
    return {
        "available": True,
        "features": features[visible],
        # GigaPath LongNet's own target-MPP tile coordinates -- fed to
        # FrozenGigaPathSlideEncoder unchanged, never mixed with the
        # level-0/HEST-aligned frame below.
        "coords": visible_coords,
        # Real level-0/HEST-aligned tile CENTER coordinates for the same
        # visible tiles, in the SAME physical frame as spot_coords -- the
        # only frame example_builder.py may validly derive
        # wsi_tile_regional_coords from (17th Codex re-audit, Step 5
        # Part 2 launch blocker #1).
        "level0_coords": mask_centers[visible],
        "context_id": f"{slide_context['context_id']}:{image_mode}:{visible_digest}",
        "n_total": int(features.shape[0]),
        "n_visible": int(visible.sum()),
    }


def nonoverlapping_context_patch_mask(
    context_coords: np.ndarray,
    query_coords: np.ndarray,
    patch_size: float,
) -> np.ndarray:
    """Context spot patches whose raw pixels do not intersect the hole."""
    return ~_overlaps_query_hole(
        np.asarray(context_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
        np.asarray(query_coords[:, :2], dtype=np.float32),
        float(patch_size) / 2.0,
    )
