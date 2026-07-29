"""Tile-encoder provenance preflight gate.

20th Codex re-audit (Step 5/6 boundary #2, response to the confirmed-
closed dense-WSI launch blocker in the 19th round): both
``slide_context.load_slide_context`` and
``spot_feature_cache.load_gen3_spot_features`` validate one cache's
provenance SYNTACTICALLY (real repo, immutable revision, nonblank
library version, exact preprocessing, well-formed digest, supported
schema) -- but neither checks that EVERY sample's dense-WSI and spot-
feature cache in one experiment used the SAME tile-encoder revision, nor
that it matches what the experiment's own config/manifest declares. A
cache built from a different, still validly-pinned revision would
silently pass each cache's own internal validation while mixing tile-
encoder identities within a single experiment -- exactly the ambiguity
pinning was meant to remove in the first place.

This is a pure, standalone function (no I/O, no torch/timm import) so it
is trivially unit-testable and ready to be wired into Step 8's mandatory
preflight gates (not yet built) once the real trainer exists. Until then
it has no caller of its own -- it is deliberately callable independently
of any specific pipeline stage.
"""
from __future__ import annotations

_PROVENANCE_FIELDS = (
    "hf_repo_id", "hf_revision", "timm_version", "preprocessing_spec",
    "state_dict_sha256", "schema_version",
)


def require_consistent_tile_encoder_provenance(
    provenance_by_source: dict[str, dict],
    expected_provenance: dict | None = None,
) -> None:
    """Every provenance dict in ``provenance_by_source`` (keyed by a
    human-readable source identifier, e.g. ``"INT1:dense_wsi"`` or
    ``"INT1:spot_features"`` -- one entry per dense-WSI AND per spot-
    feature cache across every sample selected for an experiment) must
    be IDENTICAL across every entry. When ``expected_provenance`` is
    given (the experiment's own declared config/manifest value), every
    entry must ALSO exactly match it field-by-field for every field
    ``expected_provenance`` specifies -- a caller that only wants to pin
    ``hf_revision`` (say) may omit the other fields from
    ``expected_provenance`` rather than being forced to restate every
    one of them.

    Raises ``ValueError`` with the offending source/field on the first
    mismatch found. Deliberately does not attempt to "fix" or ignore a
    mismatch -- this is a fail-closed gate, not a best-effort merge."""
    if not provenance_by_source:
        raise ValueError("require_consistent_tile_encoder_provenance: no provenance entries given")
    reference_source, reference = next(iter(provenance_by_source.items()))
    for source, provenance in provenance_by_source.items():
        for field in _PROVENANCE_FIELDS:
            if provenance.get(field) != reference.get(field):
                raise ValueError(
                    f"tile-encoder provenance mismatch: {source}.{field}={provenance.get(field)!r} "
                    f"!= {reference_source}.{field}={reference.get(field)!r} -- every dense-WSI and "
                    "spot-feature cache used in one experiment must share the exact same "
                    "tile-encoder identity"
                )
    if expected_provenance is not None:
        for field in _PROVENANCE_FIELDS:
            if field not in expected_provenance:
                continue  # expected_provenance may deliberately pin only a subset of fields
            if reference.get(field) != expected_provenance[field]:
                raise ValueError(
                    f"tile-encoder provenance mismatch: every cache has {field}="
                    f"{reference.get(field)!r}, but the experiment config/manifest declares an "
                    f"expected {field}={expected_provenance[field]!r}"
                )
