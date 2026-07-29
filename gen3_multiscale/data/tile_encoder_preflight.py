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
21st Codex re-audit, CONFIRMED real gap fixed: the first version of this
function compared ``.get(field)`` across entries WITHOUT first requiring
each entry to actually be a real, well-formed provenance dict --
``require_consistent_tile_encoder_provenance({"a": {}, "b": {}})``
passed silently, since two empty dicts agree with each other on every
field being ``None``. Every entry is now run through
``slide_context.validate_tile_encoder_provenance`` first. Separately,
``expected_provenance`` was optional (default ``None``), which let every
cache in an experiment consistently agree with each other while ALL
being built from the wrong tile-encoder revision -- this gate would
never have noticed. ``expected_provenance`` is now MANDATORY and must
declare at least ``hf_revision``.
"""
from __future__ import annotations

from gen3_multiscale.data.slide_context import validate_tile_encoder_provenance

_PROVENANCE_FIELDS = (
    "hf_repo_id", "hf_revision", "timm_version", "preprocessing_spec",
    "state_dict_sha256", "schema_version",
)


def require_consistent_tile_encoder_provenance(
    provenance_by_source: dict[str, dict],
    expected_provenance: dict,
) -> None:
    """Every provenance dict in ``provenance_by_source`` (keyed by a
    human-readable source identifier, e.g. ``"INT1:dense_wsi"`` or
    ``"INT1:spot_features"`` -- one entry per dense-WSI AND per spot-
    feature cache across every sample selected for an experiment) is
    first validated individually (same checks
    ``load_slide_context``/``load_gen3_spot_features`` already apply on
    load -- real repo, immutable revision, nonblank library version,
    exact preprocessing, well-formed digest, supported schema), then
    required to be IDENTICAL across every entry, then required to
    exactly match ``expected_provenance`` field-by-field for every field
    it specifies.

    ``expected_provenance`` is MANDATORY and must declare at least
    ``hf_revision`` -- an omitted or empty ``expected_provenance`` would
    let every cache in an experiment consistently agree with each other
    while ALL being built from the WRONG tile-encoder revision, which
    this gate would then never catch (21st Codex re-audit, CONFIRMED
    real). A caller that only wants to pin ``hf_revision`` may still
    omit the other fields from ``expected_provenance`` rather than being
    forced to restate every one of them.

    Raises ``ValueError`` with the offending source/field on the first
    mismatch found. Deliberately does not attempt to "fix" or ignore a
    mismatch -- this is a fail-closed gate, not a best-effort merge."""
    if not provenance_by_source:
        raise ValueError("require_consistent_tile_encoder_provenance: no provenance entries given")
    if not expected_provenance or "hf_revision" not in expected_provenance:
        raise ValueError(
            "require_consistent_tile_encoder_provenance: expected_provenance must declare at "
            "least hf_revision -- an omitted or empty expected_provenance would let every cache "
            "in this experiment consistently agree on the WRONG tile-encoder revision without "
            "this gate ever noticing"
        )
    for source, provenance in provenance_by_source.items():
        missing_fields = [field for field in _PROVENANCE_FIELDS if field not in provenance]
        if missing_fields:
            raise ValueError(
                f"tile-encoder provenance for {source!r} is missing field(s) {missing_fields} -- "
                "refusing to preflight-check an incomplete provenance record"
            )
        validate_tile_encoder_provenance(f"preflight entry {source!r}", provenance)

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
    for field in _PROVENANCE_FIELDS:
        if field not in expected_provenance:
            continue  # expected_provenance may deliberately pin only a subset of fields
        if reference.get(field) != expected_provenance[field]:
            raise ValueError(
                f"tile-encoder provenance mismatch: every cache has {field}="
                f"{reference.get(field)!r}, but the experiment config/manifest declares an "
                f"expected {field}={expected_provenance[field]!r}"
            )
