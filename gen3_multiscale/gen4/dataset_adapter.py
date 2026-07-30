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

import numpy as np

from gen3_multiscale.data.dataset_manifest import gene_panel_hash
from gen3_multiscale.gen4 import scfoundation_cache, uni2_dense_wsi_cache, uni2_spot_cache
from gen3_multiscale.gen4.inputs import build_gen4_spatial_field_example
from gen3_multiscale.gen4.model_factory import ARM_TABLE
from gen3_multiscale.gen4.trainer_adapter import _resolve_gen4_arm
from gen3_multiscale.training.gen3_dataset import EmptyBoundaryError, Gen3SpatialFieldDataset

_RAW_LIBRARY_SIZE_OBS_KEY = "_scilifestdl_raw_library_size"


def _resolve_cache_root(config: dict) -> str:
    data_cfg = config.get("data") or {}
    cache_root = data_cfg.get("hest_cache_dir") or data_cfg.get("hest_data_dir")
    if not cache_root:
        raise ValueError("data.hest_cache_dir (or data.hest_data_dir) must be set to resolve Gen4 cache paths")
    return str(cache_root)


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
        self.gen4_arm = _resolve_gen4_arm(config)
        if self.gen4_arm not in ARM_TABLE:
            raise ValueError(f"unknown model.arm {self.gen4_arm!r} -- expected one of {sorted(ARM_TABLE)}")
        arm_spec = ARM_TABLE[self.gen4_arm]
        cache_root = _resolve_cache_root(config)

        uses_uni2_primary = (
            arm_spec["image_feature_source"] == "precomputed" and arm_spec["global_context_source"] == "uni2_pool"
        )
        needs_scfoundation = arm_spec["gex_feature_source"] in ("frozen_context", "hybrid_context")
        needs_uni2_hybrid = arm_spec["image_feature_source"] == "hybrid_context"
        needs_sample_organ = (
            arm_spec["gex_feature_source"] == "stpath_joint"
            or arm_spec["image_feature_source"] in ("stpath_context", "hybrid_context")
        )
        self._precomputed_spot_features: dict[str, np.ndarray] = {}
        self._slide_context: dict[str, dict] = {}
        self._wsi_tile_feature_provenance: dict[str, str | None] = {}
        self._gex_context_embedding: dict[str, dict[str, np.ndarray]] = {}
        self._uni2_spot_embedding: dict[str, dict[str, np.ndarray]] = {}
        self._sample_organ: dict[str, str] = {}

        for sample_id, sample in samples.items():
            obs_names = np.asarray(sample.adata.obs_names, dtype=str)

            if uses_uni2_primary:
                cached = uni2_spot_cache.load_uni2_spot_features(
                    cache_root, sample_id, obs_names, sample.patches, sample.image_source_available,
                )
                self._precomputed_spot_features[sample_id] = cached["features"]
                dense = uni2_dense_wsi_cache.load_uni2_dense_wsi_context(
                    config, sample_id, sample.full_sample_coords.astype(np.float32),
                )
                self._slide_context[sample_id] = dense
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
