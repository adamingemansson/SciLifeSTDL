"""Tests for gen3_multiscale/data/tile_encoder_preflight.py -- the
cross-cache, cross-sample tile-encoder provenance gate (20th Codex
re-audit, Step 5/6 boundary #2: "Step 6 preflight must require identical
provenance across every selected sample and exact agreement with the
experiment manifest/config"), hardened by the 21st Codex re-audit
(CONFIRMED real: the first version accepted `{"a": {}, "b": {}}`
silently, and let `expected_provenance` be omitted entirely)."""
import pytest

from gen3_multiscale.data.tile_encoder_preflight import (
    require_consistent_tile_encoder_provenance,
)


def _provenance(**overrides) -> dict:
    base = dict(
        hf_repo_id="prov-gigapath/prov-gigapath",
        hf_revision="a" * 40,
        timm_version="1.0.3",
        preprocessing_spec="centercrop224_no_resize_v2_2026-07-24",
        state_dict_sha256="b" * 64,
        schema_version=1,
    )
    base.update(overrides)
    return base


def test_require_consistent_tile_encoder_provenance_accepts_identical_entries():
    require_consistent_tile_encoder_provenance(
        {
            "INT1:dense_wsi": _provenance(),
            "INT1:spot_features": _provenance(),
            "INT2:dense_wsi": _provenance(),
            "INT2:spot_features": _provenance(),
        },
        expected_provenance=_provenance(),
    )


def test_require_consistent_tile_encoder_provenance_rejects_a_mismatched_revision():
    with pytest.raises(ValueError, match="hf_revision"):
        require_consistent_tile_encoder_provenance(
            {
                "INT1:dense_wsi": _provenance(),
                "INT2:dense_wsi": _provenance(hf_revision="c" * 40),  # different revision
            },
            expected_provenance=_provenance(),
        )


def test_require_consistent_tile_encoder_provenance_rejects_a_mismatched_state_dict_sha256():
    """The same revision string could, in principle, be reused after the
    upstream repository's weights changed under it (a broken but
    real-world-possible scenario) -- the weights hash must also agree,
    not only the revision string."""
    with pytest.raises(ValueError, match="state_dict_sha256"):
        require_consistent_tile_encoder_provenance(
            {
                "INT1:dense_wsi": _provenance(),
                "INT1:spot_features": _provenance(state_dict_sha256="d" * 64),
            },
            expected_provenance=_provenance(),
        )


def test_require_consistent_tile_encoder_provenance_rejects_disagreement_with_expected():
    """A cache built from a DIFFERENT, still validly-pinned revision would
    pass internal per-cache validation and pass the pairwise-consistency
    check above if EVERY cache in the experiment used that same wrong
    revision -- exact agreement with an experiment-declared expected
    provenance is a separate, necessary check."""
    entries = {
        "INT1:dense_wsi": _provenance(),
        "INT1:spot_features": _provenance(),
    }
    with pytest.raises(ValueError, match="hf_revision"):
        require_consistent_tile_encoder_provenance(
            entries, expected_provenance=_provenance(hf_revision="e" * 40),
        )


def test_require_consistent_tile_encoder_provenance_accepts_agreement_with_expected():
    entries = {
        "INT1:dense_wsi": _provenance(),
        "INT1:spot_features": _provenance(),
    }
    require_consistent_tile_encoder_provenance(entries, expected_provenance=_provenance())


def test_require_consistent_tile_encoder_provenance_allows_a_partial_expected_provenance():
    """A caller pinning only hf_revision (say) should not be forced to
    also restate every other field."""
    entries = {
        "INT1:dense_wsi": _provenance(),
        "INT1:spot_features": _provenance(),
    }
    require_consistent_tile_encoder_provenance(
        entries, expected_provenance={"hf_revision": "a" * 40},
    )


def test_require_consistent_tile_encoder_provenance_rejects_empty_input():
    with pytest.raises(ValueError, match="no provenance entries"):
        require_consistent_tile_encoder_provenance({}, expected_provenance={"hf_revision": "a" * 40})


def test_require_consistent_tile_encoder_provenance_rejects_malformed_entries():
    """21st Codex re-audit, CONFIRMED real: two EMPTY provenance dicts
    previously "agreed" with each other on every field being None and
    passed silently. Every entry must now fail its own syntactic
    validation first."""
    with pytest.raises(ValueError, match="missing field"):
        require_consistent_tile_encoder_provenance(
            {"INT1:dense_wsi": {}, "INT1:spot_features": {}},
            expected_provenance=_provenance(),
        )


def test_require_consistent_tile_encoder_provenance_rejects_an_entry_with_an_invalid_field():
    """A dict with every field present but one syntactically invalid
    (e.g. an unpinned-looking revision) must also be rejected by the
    per-entry validation pass, not just a missing-key check."""
    with pytest.raises(ValueError, match="hf_revision"):
        require_consistent_tile_encoder_provenance(
            {
                "INT1:dense_wsi": _provenance(hf_revision="main"),
                "INT1:spot_features": _provenance(hf_revision="main"),
            },
            expected_provenance=_provenance(hf_revision="main"),
        )


def test_require_consistent_tile_encoder_provenance_requires_expected_provenance():
    """expected_provenance is now a MANDATORY positional/keyword argument
    -- omitting it entirely is a TypeError (missing argument), which is
    the correct, loudest possible failure mode for a caller that forgot
    to declare what revision this experiment actually expects."""
    with pytest.raises(TypeError):
        require_consistent_tile_encoder_provenance({"INT1:dense_wsi": _provenance()})


def test_require_consistent_tile_encoder_provenance_rejects_an_empty_expected_provenance():
    with pytest.raises(ValueError, match="expected_provenance"):
        require_consistent_tile_encoder_provenance(
            {"INT1:dense_wsi": _provenance()}, expected_provenance={},
        )


def test_require_consistent_tile_encoder_provenance_rejects_an_expected_provenance_missing_hf_revision():
    """21st Codex re-audit: 'production use should require an experiment-
    declared expected revision -- optional/empty expected provenance is
    insufficient because all caches could consistently use the wrong
    revision.' A nonempty expected_provenance that doesn't even mention
    hf_revision is exactly that gap."""
    with pytest.raises(ValueError, match="hf_revision"):
        require_consistent_tile_encoder_provenance(
            {"INT1:dense_wsi": _provenance()},
            expected_provenance={"schema_version": 1},
        )
