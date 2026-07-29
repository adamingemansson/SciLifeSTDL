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

import json
import os
from pathlib import Path

from gen3_multiscale.data.tile_encoder_preflight import require_consistent_tile_encoder_provenance
from gen3_multiscale.training.gen3_dataset import Gen3SampleData, load_gen3_sample_data


def expected_cache_source_labels(sample_ids) -> set[str]:
    """Exactly two labels per sample -- `<sample>:dense_wsi` and
    `<sample>:spot_features` -- never more, never fewer."""
    labels: set[str] = set()
    for sample_id in sample_ids:
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


def collect_sample_cache_provenance(sample: Gen3SampleData) -> dict:
    """The two provenance entries one already-loaded `Gen3SampleData`
    contributes to a preflight's `provenance_by_source` map."""
    if sample.tile_encoder_provenance.get("dense_wsi") is None:
        raise ValueError(
            f"{sample.sample_id}: no dense-WSI tile-encoder provenance available -- is "
            "data.slide_context_source configured to dense_wsi_cache for this experiment?"
        )
    return {
        f"{sample.sample_id}:dense_wsi": sample.tile_encoder_provenance["dense_wsi"],
        f"{sample.sample_id}:spot_features": sample.tile_encoder_provenance["spot_features"],
    }


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
    samples: dict[str, Gen3SampleData] = {}
    provenance_by_source: dict[str, dict] = {}
    for sample_id in sample_ids:
        sample = load_gen3_sample_data(cfg, manifest, sample_id)
        samples[sample_id] = sample
        provenance_by_source.update(collect_sample_cache_provenance(sample))

    coverage = verify_cache_coverage(expected_cache_source_labels(sample_ids), provenance_by_source.keys())
    require_consistent_tile_encoder_provenance(provenance_by_source, expected_tile_encoder_provenance)

    report = {
        "version": 1,
        "kind": "gen3_step6_cache_preflight",
        "n_samples": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "cache_coverage": coverage,
        "tile_encoder_provenance_expected": expected_tile_encoder_provenance,
        "tile_encoder_provenance_reference": next(iter(provenance_by_source.values())),
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
