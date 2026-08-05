"""Step 6's mandatory preflight gate.

Adam's explicit Step 6 requirements #3/#4: "Build expected cache coverage
from every manifest-selected sample. Require exactly `<sample>:dense_wsi`
and `<sample>:spot_features`. Reject missing, duplicate or additional
entries. Run cache-coverage verification and
`require_consistent_tile_encoder_provenance(...)` before constructing the
model, optimizer or DataLoader."

Reuses `load_gen3_sample_data` (gen3_dataset.py) to do the ONE real,
verified load per sample this gate needs -- the real trainer then reuses
these SAME already-loaded `Gen3SampleData` objects to build its dataset,
so a sample is never loaded from disk twice just because preflight and
dataset construction both need it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from gen3_multiscale.data.dataset_manifest import verify_metadata_csv_provenance
from gen3_multiscale.data.tile_encoder_preflight import require_consistent_tile_encoder_provenance
from gen3_multiscale.gen4.uni2_spot_cache import require_consistent_uni2_tile_encoder_provenance
from gen3_multiscale.training.gen3_dataset import Gen3SampleData, load_gen3_sample_data


def expected_cache_source_labels(sample_ids, *, require_dense_wsi: bool = True) -> set[str]:
    """Exactly `<sample>:spot_features` per sample, plus `<sample>:dense_wsi`
    too when `require_dense_wsi` -- never more, never fewer.

    `require_dense_wsi` (Adam's Step 6 audit #6 of commit a32051b: "do
    not load dense WSI caches for architectures that do not consume
    them"): `gen3_dataset.py::load_gen3_sample_data` now skips loading
    the dense-WSI cache entirely for an architecture with
    `use_regional_he=false` and `use_global_slide=false` -- this gate
    must not then demand dense-WSI provenance that was never (and
    correctly never) produced."""
    labels: set[str] = set()
    for sample_id in sample_ids:
        if require_dense_wsi:
            labels.add(f"{sample_id}:dense_wsi")
        labels.add(f"{sample_id}:spot_features")
    return labels


def verify_cache_coverage(expected_labels: set[str], available_labels) -> dict:
    """Fail closed on missing, duplicate, OR extra cache-coverage
    entries -- a consistent SUBSET (or an unexpected superset, e.g. a
    stale entry for a sample no longer selected by this experiment) must
    never silently pass."""
    available_list = list(available_labels)
    available_set = set(available_list)
    if len(available_list) != len(available_set):
        duplicates = sorted({label for label in available_list if available_list.count(label) > 1})
        raise ValueError(f"gen3 preflight: duplicate cache-coverage entries: {duplicates}")
    missing = sorted(expected_labels - available_set)
    if missing:
        raise ValueError(f"gen3 preflight: missing cache-coverage entries: {missing}")
    extra = sorted(available_set - expected_labels)
    if extra:
        raise ValueError(
            f"gen3 preflight: unexpected extra cache-coverage entries not part of the "
            f"manifest-selected sample set: {extra}"
        )
    return {"n_expected": len(expected_labels), "n_available": len(available_set), "passed": True}


def collect_sample_cache_provenance(sample: Gen3SampleData, *, require_dense_wsi: bool = True) -> dict:
    """The provenance entries one already-loaded `Gen3SampleData`
    contributes to a preflight's `provenance_by_source` map -- both
    `dense_wsi` and `spot_features` when `require_dense_wsi`, only
    `spot_features` otherwise (see `expected_cache_source_labels`)."""
    entries = {f"{sample.sample_id}:spot_features": sample.tile_encoder_provenance["spot_features"]}
    if require_dense_wsi:
        if sample.tile_encoder_provenance.get("dense_wsi") is None:
            raise ValueError(
                f"{sample.sample_id}: no dense-WSI tile-encoder provenance available -- is "
                "data.slide_context_source configured to dense_wsi_cache for this experiment?"
            )
        entries[f"{sample.sample_id}:dense_wsi"] = sample.tile_encoder_provenance["dense_wsi"]
    return entries


def collect_sample_cache_content_identity(sample: Gen3SampleData, *, require_dense_wsi: bool = True) -> dict:
    """Codex re-audit of commit 90f853e, launch blocker #5: "Bind every
    per-sample spot-feature and dense-WSI cache CONTENT digest, barcode
    identity and availability identity into the run manifest, and
    compare them on resume. Encoder provenance alone is insufficient" --
    `require_consistent_tile_encoder_provenance` (used by
    `load_and_preflight_samples` below) only checks the tile ENCODER's
    own identity (repo/revision/weights sha256), which stays identical
    if a cache is validly rebuilt (same encoder, same real H&E patches)
    but produces DIFFERENT numeric feature content -- a non-deterministic
    encoding bug, a corrupted rebuild, or an accidental wrong-sample
    write would all pass provenance checks while silently changing what
    the model actually trains/evaluates on. This binds the REAL,
    already-verified content identity `Gen3SampleData` computed at load
    time (`precomputed_spot_features_digest` -- sha256 of the verified
    features array's own bytes -- plus the exact barcode order and
    availability mask) so a resume can detect the cache changing under
    it, not merely the encoder identity staying superficially the same.
    """
    barcodes = np.asarray(sample.precomputed_spot_features_barcodes, dtype=str)
    availability = np.asarray(sample.image_source_available, dtype=bool)
    identity = {
        "spot_features_content_sha256": sample.precomputed_spot_features_digest,
        "spot_features_barcodes_sha256": hashlib.sha256(
            b"\x1f".join(b.encode("utf-8") for b in barcodes)
        ).hexdigest(),
        "spot_features_availability_sha256": hashlib.sha256(availability.tobytes()).hexdigest(),
    }
    if require_dense_wsi:
        context_id = (sample.slide_context_record or {}).get("context_id")
        if context_id is None:
            raise ValueError(
                f"{sample.sample_id}: no dense-WSI content identity (context_id) available -- is "
                "data.slide_context_source configured to dense_wsi_cache for this experiment?"
            )
        identity["dense_wsi_context_id"] = str(context_id)
    return identity


def cache_content_fingerprint(content_identity_by_sample: dict[str, dict]) -> str:
    """A single fingerprint over EVERY sample's real cache-content
    identity (see `collect_sample_cache_content_identity`), deterministic
    regardless of dict/sample iteration order -- this is what
    `train.py::verify_resume_consistency` actually compares, so a
    validly-regenerated-but-different cache for even one sample changes
    this fingerprint and refuses the resume."""
    return hashlib.sha256(
        json.dumps(content_identity_by_sample, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_and_preflight_samples(
    cfg, manifest: dict, sample_ids: list[str], expected_tile_encoder_provenance: dict,
) -> tuple[dict[str, Gen3SampleData], dict]:
    """Load every sample in `sample_ids` EXACTLY once (real I/O -- the
    real, one-time preflight cost, not a per-epoch one), verify cache
    coverage and tile-encoder provenance BEFORE returning, and hand back
    the already-loaded samples for the caller's dataset/DataLoader to
    reuse without a second load.

    Raises (fail-closed) on any preflight failure -- the caller must not
    proceed to construct a model, optimizer, or DataLoader if this
    raises."""
    # Real, confirmed gap (Codex audit of commit 27e1232): the shared
    # metadata CSV's own content provenance was recorded by
    # dataset_manifest.py but never re-verified before training -- check
    # it once here (cheap, one shared file) before the per-sample h5ad/
    # patch checks (load_gen3_sample_data -> verify_content_provenance)
    # do the same for each sample individually.
    verify_metadata_csv_provenance(manifest)

    # Adam's Step 6 audit #6 of commit a32051b: mirrors gen3_dataset.py::
    # load_gen3_sample_data's own use_regional_he/use_global_slide check
    # -- an architecture using NEITHER never loads (and must not be
    # required to produce) dense-WSI cache provenance.
    model_params = cfg.get("model", {}).get("params", {}) or {}
    require_dense_wsi = bool(model_params.get("use_regional_he", False)) or bool(
        model_params.get("use_global_slide", False)
    )
    image_encoder = str(cfg.data.get("image_encoder", "gigapath"))
    if image_encoder not in {"gigapath", "uni2"}:
        raise ValueError(f"data.image_encoder must be 'gigapath' or 'uni2', got {image_encoder!r}")
    if image_encoder == "uni2" and require_dense_wsi:
        raise ValueError(
            "data.image_encoder='uni2' combined with use_regional_he/use_global_slide is not "
            "supported -- dense-WSI regional/global context only has a GigaPath tile-encoder "
            "path in this codebase"
        )

    samples: dict[str, Gen3SampleData] = {}
    provenance_by_source: dict[str, dict] = {}
    content_identity_by_sample: dict[str, dict] = {}
    for sample_id in sample_ids:
        sample = load_gen3_sample_data(cfg, manifest, sample_id)
        samples[sample_id] = sample
        provenance_by_source.update(collect_sample_cache_provenance(sample, require_dense_wsi=require_dense_wsi))
        content_identity_by_sample[sample_id] = collect_sample_cache_content_identity(
            sample, require_dense_wsi=require_dense_wsi,
        )

    coverage = verify_cache_coverage(
        expected_cache_source_labels(sample_ids, require_dense_wsi=require_dense_wsi), provenance_by_source.keys(),
    )
    if image_encoder == "gigapath":
        require_consistent_tile_encoder_provenance(provenance_by_source, expected_tile_encoder_provenance)
    else:
        require_consistent_uni2_tile_encoder_provenance(provenance_by_source, expected_tile_encoder_provenance)

    report = {
        "version": 2,
        "kind": "gen3_step6_cache_preflight",
        "n_samples": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "cache_coverage": coverage,
        "tile_encoder_provenance_expected": expected_tile_encoder_provenance,
        "tile_encoder_provenance_reference": next(iter(provenance_by_source.values())),
        # Launch blocker #5: real per-sample cache CONTENT identity
        # (never merely encoder provenance), bound into a single
        # deterministic fingerprint `train.py::build_run_manifest` lifts
        # to the top level for `verify_resume_consistency` to compare.
        "cache_content_by_sample": content_identity_by_sample,
        "cache_content_fingerprint": cache_content_fingerprint(content_identity_by_sample),
        "passed": True,
    }
    return samples, report


def save_gen3_preflight_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path


def load_gen3_preflight_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
