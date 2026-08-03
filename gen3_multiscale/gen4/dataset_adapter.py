"""Integration audit item 2: a Gen4-aware dataset adapter around the
EXISTING Gen3 dataset/mask machinery -- no new mask scheduler, no new
trainer, no re-derived sample loading. `Gen4SpatialFieldDataset` is a
`training.gen3_dataset.Gen3SpatialFieldDataset` SUBCLASS: it inherits
`__init__` (sample/schedule validation, per-sample masking-config
preparation), `__len__`, `item_identity`, `validate_boundary_schedule`,
and `_resolve_barcodes` completely unmodified -- real, manifest-driven
sample loading (`Gen3SampleData`/`load_gen3_sample_data`) and the real
stratified mask schedule (`build_gen3_mask_schedule`) stay exactly as
Gen3 built them. The only thing this subclass changes is which per-spot
image/GEX-context cache records get attached before calling
`gen4.inputs.build_gen4_spatial_field_example` instead of Gen3's own
`example_builder.build_spatial_field_example`.

Per-arm wiring (confirmed against `gen4.model_factory.ARM_TABLE` and
matching the user's own explicit per-arm spec):

  gen4a (baseline_uni2_weighted_linear): UNI2 spot features (SWAPS OUT
    Gen3SampleData's own GigaPath-cached `precomputed_spot_features` --
    this arm's real image path IS the UNI2 cache, never GigaPath's) +
    UNI2 dense-WSI context (SWAPS OUT `sample.slide_context_record`,
    tagged `wsi_tile_feature_provenance="uni2"` so
    `global_context_source="uni2_pool"` never silently consumes
    GigaPath-encoded tiles -- see gen4/inputs.py's own docstring).
  gen4c (arm1_uni2_scfoundation): the same UNI2 swap as gen4a, PLUS
    scFoundation `gex_context_embedding` (`gex_feature_source=
    "frozen_context"` per ARM_TABLE).
  gen4b (arm2_gigapath_scfoundation): Gen3's own GigaPath-cached
    `precomputed_spot_features`/`slide_context_record` are REUSED
    UNCHANGED (`image_feature_source="precomputed"`,
    `global_context_source="gigapath"` -- exactly what Gen3 already
    caches; the live GigaPath slide encoder/LongNet global vector is
    wired at model-construction time by `gen4.trainer_adapter`, not
    here), PLUS scFoundation `gex_context_embedding`.
  gen4d (arm3_stpath): Gen3's own GigaPath-cached
    `precomputed_spot_features` is reused UNCHANGED as a structurally
    required (but, for this arm, conditioner-ignored --
    `image_feature_source="stpath_context"` reads live STPath output
    instead) placeholder, PLUS this sample's real manifest `organ`
    (`gen4.conditioner.Gen4Conditioner` fails closed on a missing organ
    for any STPath-consuming arm).
  gen4e (arm4_hybrid_stpath_uni2_scfoundation): Gen3's own GigaPath
    cache reused as the STPath-arm placeholder (same as gen4d) PLUS a
    SEPARATE UNI2 spot-feature lookup (`uni2_spot_embedding`, arm 4's
    own `observed_uni2_features` hybrid field) PLUS scFoundation
    `gex_context_embedding` PLUS `sample_organ`.

Gen5's five arms consume the IDENTICAL per-arm conditioning input as
their mapped Gen4 arm (`gen5.model_factory.GEN5_TO_GEN4_ARM`) -- this
class is used unchanged for Gen5 configs too; only the arm-to-Gen4-arm
resolution differs (`gen4.trainer_adapter._resolve_gen4_arm`, reused
here rather than re-implemented)."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gen3_multiscale.data import example_builder
from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.data.dataset_manifest import gene_panel_hash
from gen3_multiscale.data.dataset_manifest import verify_content_provenance, verify_metadata_csv_provenance
from gen3_multiscale.gen4 import scfoundation_cache, uni2_dense_wsi_cache, uni2_spot_cache
from gen3_multiscale.gen4.inputs import build_gen4_spatial_field_example
from gen3_multiscale.gen4.model_factory import ARM_TABLE
from gen3_multiscale.gen4.trainer_adapter import _resolve_gen4_arm
from gen3_multiscale.training.gen3_dataset import (
    EmptyBoundaryError,
    Gen3SampleData,
    Gen3SpatialFieldDataset,
    load_gen3_sample_data,
)
from gen3_multiscale.training.gen3_preflight import (
    cache_content_fingerprint,
    collect_sample_cache_content_identity,
    collect_sample_cache_provenance,
    expected_cache_source_labels,
    verify_cache_coverage,
)
from gen3_multiscale.data.tile_encoder_preflight import require_consistent_tile_encoder_provenance

_RAW_LIBRARY_SIZE_OBS_KEY = "_scilifestdl_raw_library_size"


def _resolve_cache_root(config: dict) -> str:
    data_cfg = config.get("data") or {}
    cache_root = data_cfg.get("hest_cache_dir") or data_cfg.get("hest_data_dir")
    if not cache_root:
        raise ValueError("data.hest_cache_dir (or data.hest_data_dir) must be set to resolve Gen4 cache paths")
    return str(cache_root)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _barcode_sha256(barcodes: np.ndarray) -> str:
    return hashlib.sha256(
        b"\x1f".join(str(barcode).encode("utf-8") for barcode in np.asarray(barcodes, dtype=str))
    ).hexdigest()


def _availability_sha256(availability: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(availability, dtype=bool).tobytes()).hexdigest()


def _arm_requirements(config: dict) -> dict[str, bool]:
    from gen3_multiscale.gen6.contract import is_gen6_config, gen6_cache_requirements

    if is_gen6_config(config):
        return gen6_cache_requirements(config)
    arm = _resolve_gen4_arm(config)
    if arm not in ARM_TABLE:
        raise ValueError(f"unknown model.arm {arm!r} -- expected one of {sorted(ARM_TABLE)}")
    spec = ARM_TABLE[arm]
    return {
        "uses_uni2_primary": (
            spec["image_feature_source"] == "precomputed"
            and spec["global_context_source"] == "uni2_pool"
        ),
        "uses_gigapath_spot": spec["image_feature_source"] in {
            "precomputed", "stpath_context", "hybrid_context",
        } and spec["global_context_source"] != "uni2_pool",
        "uses_gigapath_dense": spec["global_context_source"] == "gigapath",
        "uses_scfoundation": spec["gex_feature_source"] in {"frozen_context", "hybrid_context"},
        "uses_uni2_hybrid": spec["image_feature_source"] == "hybrid_context",
        "uses_sample_organ": (
            spec["gex_feature_source"] == "stpath_joint"
            or spec["image_feature_source"] in {"stpath_context", "hybrid_context"}
        ),
    }


def _load_uni2_primary_sample(
    cfg, manifest: dict, sample_id: str, *, require_dense: bool = True,
) -> Gen3SampleData:
    """Load a Gen3-compatible sample record whose primary image caches are
    UNI2, not GigaPath.

    This is the important boundary the first adapter missed: using
    ``load_gen3_sample_data`` for a UNI2-only arm still required and bound
    an unrelated GigaPath spot cache before later swapping it out.  The
    returned object keeps the already-audited Gen3 sample/mask machinery,
    but every image feature and dense context it carries is the modality
    the arm really consumes.
    """
    record = manifest["samples"].get(sample_id)
    if record is None:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    verify_content_provenance(cfg.data.hest_data_dir, manifest, sample_id)
    adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
    obs_names = np.asarray(adata.obs_names, dtype=str)
    full_sample_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    cache_root = _resolve_cache_root(cfg)
    spot_record = uni2_spot_cache.load_uni2_spot_features(
        cache_root, sample_id, obs_names, patches, image_source_available,
    )
    dense_record = None
    if require_dense:
        dense_record = uni2_dense_wsi_cache.load_uni2_dense_wsi_context(
            cfg, sample_id, full_sample_coords.astype(np.float32),
        )
    coords3d = np.concatenate(
        [full_sample_coords, np.zeros((full_sample_coords.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return Gen3SampleData(
        sample_id=sample_id,
        patient_id=str(record["patient_id"]),
        split=str(record["split"]),
        adata=adata,
        patches=patches,
        image_source_available=image_source_available,
        precomputed_spot_features=spot_record["features"],
        precomputed_spot_features_barcodes=spot_record["barcodes"],
        precomputed_spot_features_digest=hashlib.sha256(
            np.ascontiguousarray(spot_record["features"]).tobytes()
        ).hexdigest(),
        full_sample_coords=full_sample_coords,
        coords3d=coords3d,
        slice_ids=np.full(obs_names.shape[0], sample_id, dtype=object),
        spatial_adjacency=tuple(build_knn_adjacency(full_sample_coords, k_neighbors=6)),
        slide_context_record=dense_record,
        tile_encoder_provenance={
            **({"dense_wsi": dense_record["tile_encoder_provenance"]} if dense_record else {}),
            "spot_features": spot_record["provenance"],
        },
    )


def load_and_preflight_gen4_samples(
    cfg,
    manifest: dict,
    sample_ids: list[str],
    config: dict,
    *,
    require_resolved_artifacts: bool = True,
) -> tuple[dict[str, Gen3SampleData], dict]:
    """Load and bind exactly the cache modalities a Gen4/Gen5 arm uses.

    The report deliberately has the same top-level cache-content fields
    as Gen3's preflight, so the existing run-manifest, resume, and
    evaluator identity checks continue to work unchanged.
    """
    if not sample_ids:
        raise ValueError("Gen4/Gen5 preflight requires at least one manifest-selected sample")
    verify_metadata_csv_provenance(manifest)
    requirements = _arm_requirements(config)
    cache_root = _resolve_cache_root(config)

    # Structural config validation and exact manifest-derived auxiliary
    # cache coverage happen before loading a model or constructing a
    # dataset.  Construction-only smoke may use stub identities; a real
    # or staged run must have every declared artifact pinned.
    model_kind = str((config.get("model") or {}).get("kind", ""))
    from gen3_multiscale.gen6.contract import is_gen6_config

    if is_gen6_config(config):
        from gen3_multiscale.gen6.preflight import static_audit_gen6_config

        static_report = static_audit_gen6_config(config)
    elif model_kind == "latent_flow":
        from gen3_multiscale.gen5.preflight import static_audit_gen5_config

        static_report = static_audit_gen5_config(config)
    else:
        from gen3_multiscale.gen4.preflight import static_audit_gen4_config

        static_report = static_audit_gen4_config(config)
    if require_resolved_artifacts and not static_report["ready_for_real_training"]:
        raise ValueError(
            "Gen4/Gen5 config is not ready for real training; unresolved required fingerprints: "
            f"{static_report['unset_required_fingerprints']}"
        )
    if is_gen6_config(config):
        from gen3_multiscale.gen6.preflight import audit_gen6_manifest_cache_coverage

        modality_report = audit_gen6_manifest_cache_coverage(
            cache_root, sample_ids, config,
            require_staged_artifacts=require_resolved_artifacts,
        )
    else:
        from gen3_multiscale.gen4.preflight import audit_gen4_manifest_cache_coverage

        modality_report = audit_gen4_manifest_cache_coverage(cache_root, sample_ids, config)

    samples: dict[str, Gen3SampleData] = {}
    gigapath_provenance: dict[str, dict] = {}
    content_by_sample: dict[str, dict] = {}
    gene_names = list(manifest["gene_panel"])
    expected_gigapath_provenance = None
    if requirements["uses_gigapath_spot"]:
        from gen3_multiscale.training.train import expected_tile_encoder_provenance

        expected_gigapath_provenance = expected_tile_encoder_provenance(config)

    for sample_id in sample_ids:
        if requirements["uses_uni2_primary"]:
            sample = _load_uni2_primary_sample(
                cfg, manifest, sample_id,
                require_dense=requirements.get("uses_uni2_dense", True),
            )
            identity = {
                "primary_modality": "uni2",
                "uni2_spot_features_content_sha256": _array_sha256(sample.precomputed_spot_features),
                "uni2_spot_features_barcodes_sha256": _barcode_sha256(
                    sample.precomputed_spot_features_barcodes
                ),
                "uni2_spot_features_availability_sha256": _availability_sha256(
                    sample.image_source_available
                ),
            }
            if sample.slide_context_record is not None:
                identity["uni2_dense_wsi_context_id"] = str(
                    sample.slide_context_record["context_id"]
                )
        else:
            # Gen6 derives this requirement from its canonical arm table.
            # Do not infer it from inherited model.params: a prepared
            # component-screen config may intentionally replace the base
            # model while retaining otherwise matched data settings.
            sample = load_gen3_sample_data(
                cfg, manifest, sample_id,
                require_dense_wsi=requirements["uses_gigapath_dense"],
            )
            # All non-UNI2-primary arms genuinely consume the GigaPath
            # spot features (directly or as STPath's image tokens).
            gigapath_provenance.update(
                collect_sample_cache_provenance(
                    sample, require_dense_wsi=requirements["uses_gigapath_dense"],
                )
            )
            identity = {
                "primary_modality": "gigapath",
                **collect_sample_cache_content_identity(
                    sample, require_dense_wsi=requirements["uses_gigapath_dense"],
                ),
            }
        samples[sample_id] = sample

        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        expression = np.asarray(
            sample.adata.X.toarray() if hasattr(sample.adata.X, "toarray") else sample.adata.X,
            dtype=np.float32,
        )
        if requirements["uses_scfoundation"]:
            if _RAW_LIBRARY_SIZE_OBS_KEY not in sample.adata.obs:
                raise ValueError(
                    f"{sample_id}: scFoundation requires adata.obs[{_RAW_LIBRARY_SIZE_OBS_KEY!r}] "
                    "with pre-normalization library sizes; refusing to infer it from log1p expression"
                )
            raw_library_size = np.asarray(
                sample.adata.obs[_RAW_LIBRARY_SIZE_OBS_KEY], dtype=np.float32,
            )
            cached_sc = scfoundation_cache.load_scfoundation_spot_features(
                cache_root,
                sample_id,
                obs_names,
                gene_panel_hash(gene_names),
                expression,
                raw_library_size,
            )
            identity.update({
                "scfoundation_features_content_sha256": _array_sha256(cached_sc["features"]),
                "scfoundation_barcodes_sha256": _barcode_sha256(cached_sc["barcodes"]),
                "scfoundation_availability_sha256": _availability_sha256(
                    cached_sc["feature_available"]
                ),
                "scfoundation_provenance_sha256": hashlib.sha256(
                    json.dumps(cached_sc["provenance"], sort_keys=True).encode("utf-8")
                ).hexdigest(),
            })

        if requirements["uses_uni2_hybrid"]:
            cached_uni2 = uni2_spot_cache.load_uni2_spot_features(
                cache_root, sample_id, obs_names, sample.patches, sample.image_source_available,
            )
            identity.update({
                "hybrid_uni2_features_content_sha256": _array_sha256(cached_uni2["features"]),
                "hybrid_uni2_barcodes_sha256": _barcode_sha256(cached_uni2["barcodes"]),
                "hybrid_uni2_availability_sha256": _availability_sha256(
                    cached_uni2["image_source_available"]
                ),
            })
        content_by_sample[sample_id] = identity

    if gigapath_provenance:
        expected_labels = expected_cache_source_labels(
            sample_ids, require_dense_wsi=requirements["uses_gigapath_dense"],
        )
        verify_cache_coverage(expected_labels, gigapath_provenance.keys())
        require_consistent_tile_encoder_provenance(
            gigapath_provenance, expected_gigapath_provenance,
        )

    report = {
        "version": 1,
        "kind": "gen4_gen5_consumed_cache_preflight",
        "arm": str((config.get("model") or {}).get("arm", "")),
        "model_kind": model_kind,
        "n_samples": len(sample_ids),
        "sample_ids": sorted(sample_ids),
        "static_config_audit": static_report,
        "modality_coverage": modality_report,
        "cache_content_by_sample": content_by_sample,
        "cache_content_fingerprint": cache_content_fingerprint(content_by_sample),
        "passed": True,
    }
    return samples, report


class Gen4SpatialFieldDataset(Gen3SpatialFieldDataset):
    """See module docstring. `config` is the resolved Gen4 or Gen5 config
    (used only to resolve the arm and the on-disk cache root -- never to
    re-derive sample selection or masking, which stay the base class's
    own job). `gene_names` is the manifest's own gene panel, in manifest
    order (required for scFoundation's `gene_panel_hash` cache-identity
    check, exactly like `gen4.trainer_adapter`'s own use of it)."""

    def __init__(self, manifest, samples, schedule, strata, config: dict, gene_names: list[str], **kwargs):
        super().__init__(manifest, samples, schedule, strata, **kwargs)
        self.config = config
        self.gene_names = list(gene_names)
        self.gen4_arm = str((config.get("model") or {}).get("arm", ""))
        requirements = _arm_requirements(config)
        cache_root = _resolve_cache_root(config)

        uses_uni2_primary = requirements["uses_uni2_primary"]
        needs_scfoundation = requirements["uses_scfoundation"]
        needs_uni2_hybrid = requirements["uses_uni2_hybrid"]
        needs_sample_organ = requirements["uses_sample_organ"]
        self._precomputed_spot_features: dict[str, np.ndarray] = {}
        self._slide_context: dict[str, dict] = {}
        self._wsi_tile_feature_provenance: dict[str, str | None] = {}
        self._gex_context_embedding: dict[str, dict[str, np.ndarray]] = {}
        self._uni2_spot_embedding: dict[str, dict[str, np.ndarray]] = {}
        self._sample_organ: dict[str, str] = {}

        for sample_id, sample in samples.items():
            obs_names = np.asarray(sample.adata.obs_names, dtype=str)

            if uses_uni2_primary:
                # Load the explicitly UNI2-bound records. The real
                # Gen4-aware preflight has already validated these exact
                # files; loading again here also preserves this dataset
                # class's useful standalone contract for tests/tools
                # supplied a generic Gen3SampleData object.
                cached = uni2_spot_cache.load_uni2_spot_features(
                    cache_root, sample_id, obs_names, sample.patches, sample.image_source_available,
                )
                self._precomputed_spot_features[sample_id] = cached["features"]
                if requirements.get("uses_uni2_dense", True):
                    self._slide_context[sample_id] = uni2_dense_wsi_cache.load_uni2_dense_wsi_context(
                        config, sample_id, sample.full_sample_coords.astype(np.float32),
                    )
                    self._wsi_tile_feature_provenance[sample_id] = "uni2"
            elif needs_uni2_hybrid:
                # Arm 4 (hybrid): a SEPARATE UNI2 spot-feature lookup on
                # top of the reused GigaPath precomputed_spot_features --
                # see gen4/inputs.py's own `uni2_spot_embedding` docstring
                # (Integration finding #4's H&E-leakage fix applies to
                # this exact field).
                cached_uni2 = uni2_spot_cache.load_uni2_spot_features(
                    cache_root, sample_id, obs_names, sample.patches, sample.image_source_available,
                )
                self._uni2_spot_embedding[sample_id] = {
                    str(b): cached_uni2["features"][i] for i, b in enumerate(cached_uni2["barcodes"])
                }

            if needs_scfoundation:
                expression = np.asarray(sample.adata.X.toarray() if hasattr(sample.adata.X, "toarray") else sample.adata.X, dtype=np.float32)
                if _RAW_LIBRARY_SIZE_OBS_KEY not in sample.adata.obs:
                    raise ValueError(
                        f"{sample_id}: adata.obs is missing {_RAW_LIBRARY_SIZE_OBS_KEY!r} -- scFoundation's "
                        "gex_context requires the real pre-normalization total count per spot "
                        "(data.loaders.basic_qc_and_normalize stashes this during QC)"
                    )
                raw_library_size = np.asarray(sample.adata.obs[_RAW_LIBRARY_SIZE_OBS_KEY], dtype=np.float32)
                cached_sc = scfoundation_cache.load_scfoundation_spot_features(
                    cache_root, sample_id, obs_names, gene_panel_hash(self.gene_names), expression, raw_library_size,
                )
                self._gex_context_embedding[sample_id] = scfoundation_cache.barcode_embedding_lookup(cached_sc)

            if needs_sample_organ:
                organ = manifest["samples"][sample_id].get("organ")
                if not organ:
                    raise ValueError(f"{sample_id}: manifest has no organ recorded, required for arm {self.gen4_arm!r}")
                self._sample_organ[sample_id] = str(organ)

    def __getitem__(self, idx: int):
        item = self._items[idx % len(self._items)]
        sample_id = item.sample_id
        sample = self.samples[sample_id]
        context_barcodes, query_barcodes = self._resolve_barcodes(item)

        precomputed_spot_features = self._precomputed_spot_features.get(sample_id, sample.precomputed_spot_features)
        slide_context = self._slide_context.get(sample_id, sample.slide_context_record)
        wsi_tile_feature_provenance = self._wsi_tile_feature_provenance.get(sample_id)

        try:
            inputs, targets = build_gen4_spatial_field_example(
                sample.adata, sample.patches, context_barcodes, query_barcodes,
                sample_id=sample_id, patient_id=sample.patient_id,
                precomputed_spot_features=precomputed_spot_features,
                gex_context_embedding=self._gex_context_embedding.get(sample_id),
                full_sample_coords=sample.full_sample_coords, require_full_sample_coords=True,
                patch_size_fullres=self.patch_size_fullres, k_neighbors=self.k_neighbors,
                local_k=self.local_k, max_rings=self.max_rings, max_boundary_size=self.max_boundary_size,
                expected_feature_width=int(precomputed_spot_features.shape[1]),
                slide_context=slide_context,
                image_source_available=sample.image_source_available,
                uni2_spot_embedding=self._uni2_spot_embedding.get(sample_id),
                wsi_tile_feature_provenance=wsi_tile_feature_provenance,
                sample_organ=self._sample_organ.get(sample_id),
            )
        except EmptyBoundaryError as exc:
            identity = self.item_identity(idx)
            raise EmptyBoundaryError(
                f"{sample_id}: {self.schedule.role} mask has no usable observed boundary "
                f"(stratum={identity['stratum']!r}, "
                f"query_fingerprint={identity['query_fingerprint']}); {exc}"
            ) from exc

        novae_diagnostic = None
        if self.novae_enabled:
            from gen3_multiscale.data import novae_graph

            novae_inputs = novae_graph.build_context_only_novae_input(
                sample.adata, context_barcodes, query_barcodes, sample_id=sample_id,
            )
            novae_diagnostic = novae_graph.verify_novae_context_excludes_query_identities(
                novae_inputs, query_barcodes,
            )
        if novae_diagnostic is not None:
            inputs.provenance["novae_context_only_check"] = novae_diagnostic
        return inputs, targets
