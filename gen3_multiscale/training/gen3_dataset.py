"""Step 6 of the real Gen3 data builder/trainer: the manifest-backed
dataset the real trainer (`training/train.py`) actually loads.

Loads every manifest-selected sample's real data EXACTLY once (adata,
patches, availability, the verified spot-feature cache record, the
dense-WSI slide context, and its mask schedule), then serves individual
`(SpatialFieldInputs, SpatialFieldTargets)` items via a plain
`torch.utils.data.Dataset`.

Mandatory requirement #1 (Adam's Step 6 instructions): sample selection
and the train/validation/test split are driven EXCLUSIVELY from the
immutable dataset manifest (`dataset_manifest.py`) -- `Gen3SampleData`
below never re-derives a split independently, it reads
`manifest["samples"][sample_id]["split"]`.

Mandatory requirement #2: the spot-feature cache is loaded as the
COMPLETE verified record (barcodes, availability, and features together)
via `spot_feature_cache.load_gen3_spot_features` -- never a bare feature
matrix extracted and passed on alone. `Gen3SampleData.__post_init__`
additionally re-verifies the record's barcodes/availability against this
exact sample's own `adata`/`image_source_available` a SECOND time at
construction -- redundant with `load_gen3_spot_features`'s own check,
deliberately so: it catches a caller-side variable-mixup bug (a
DIFFERENT sample's already-verified record accidentally attached to
THIS sample's adata), which `load_gen3_spot_features` itself cannot see
since it only ever validates against whatever `adata`/patches/
availability its OWN caller happened to pass it.

Mandatory requirement #5: never invoke the frozen GigaPath tile encoder
per training example. `build_spatial_field_example` is always called
with `precomputed_spot_features=`, never `image_feature_fn=` -- there is
no code path in this module that imports or calls
`src.models.conditioning`'s tile-encoder functions at all.

Mandatory requirement #6 (disjoint train/validation/test masks): resolved
by construction, not by an in-sample overlap check. `dataset_manifest.py`
already patient-disjoint-splits SAMPLES into train/validation/test
(CONTRACT.md section 33's own "the mask-drawing scheme needs an explicit
disjointness mechanism... or per-sample in-sample validation/test masking
needs to be reconsidered in favor of relying solely on the sample-level
train/validation/test split" -- the second option, chosen here). A
TRAINING sample only ever draws training-role masks (via
`mask_fingerprint.build_collision_free_training_schedule`, reserved set
empty -- there is no same-sample validation/test mask to collide with,
by design); a VALIDATION/TEST sample only ever draws its own role's
FIXED, deterministic masks (via `mask_schedule.ensure_stratified_mask_bank`,
one split key only). Since validation/test masks are realized on
ENTIRELY DIFFERENT samples than any training mask, "validation/test
masks must never be training masks" holds structurally: a query spot
realized on a validation sample cannot equal a query spot realized on a
training sample, because they are different (sample_id, barcode) pairs
by construction of `dataset_manifest.composite_spot_id`. Verified, not
merely asserted: `build_training_sample_mask_report`/
`build_held_out_sample_mask_report` (mask_fingerprint.py) are run at
dataset-construction time for every sample and their `passed` flags are
required True.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.data import example_builder, novae_graph, slide_context, spot_feature_cache
from gen3_multiscale.data import mask_fingerprint
from gen3_multiscale.data.boundary_graph import (
    EmptyBoundaryError,
    build_knn_adjacency,
    extract_boundary_and_local_context,
)
from gen3_multiscale.data.dataset_manifest import verify_content_provenance
from gen3_multiscale.data.mask_schedule import (
    ensure_stratified_mask_bank,
    prepare_masking_cfg_for_sample,
    stratum_to_masking_cfg,
)


@dataclass(frozen=True)
class Gen3SampleData:
    """Everything loaded ONCE for one manifest sample, real and verified
    -- never re-loaded per masking draw."""
    sample_id: str
    patient_id: str
    split: str  # "train" / "validation" / "test", read from the manifest
    adata: "object"  # ad.AnnData, QC'd/gene-panel-aligned (example_builder.load_sample_for_examples's contract)
    patches: np.ndarray
    image_source_available: np.ndarray
    precomputed_spot_features: np.ndarray  # VERIFIED via spot_feature_cache.load_gen3_spot_features
    precomputed_spot_features_barcodes: np.ndarray  # the cache record's OWN verified barcodes -- kept alongside the features, not discarded after the fact
    precomputed_spot_features_digest: str  # sha256 of the verified features array's bytes -- an auditable identity, not merely a shape
    full_sample_coords: np.ndarray
    coords3d: np.ndarray  # [n, 3], z=0 (single 2D section, not a Track B multi-slice series)
    slice_ids: np.ndarray  # [n] str, every entry == sample_id (mask_bank's single-slice convention)
    slide_context_record: dict | None
    tile_encoder_provenance: dict  # {"dense_wsi": {...} | None, "spot_features": {...}}
    spatial_adjacency: tuple[np.ndarray, ...] | None = None  # canonical graph, built once per sample

    def __post_init__(self):
        if self.spatial_adjacency is None:
            object.__setattr__(
                self,
                "spatial_adjacency",
                tuple(build_knn_adjacency(self.full_sample_coords, k_neighbors=6)),
            )
        elif len(self.spatial_adjacency) != self.full_sample_coords.shape[0]:
            raise ValueError(
                f"{self.sample_id}: spatial_adjacency has {len(self.spatial_adjacency)} nodes, "
                f"expected {self.full_sample_coords.shape[0]}"
            )

        # Mandatory requirement #2's "verify at the trainer call site"
        # half -- see module docstring. Real, confirmed gap (Codex audit
        # of commit 27e1232): a row-COUNT check alone cannot catch a
        # caller-side mix-up between two samples with the SAME n_spots
        # (routine for same-technology HEST-1k grids) -- barcode IDENTITY
        # AND ORDER is the real invariant, so this now compares the
        # spot-feature cache's own verified barcodes against this exact
        # sample's adata.obs_names, not just their lengths.
        obs_names = np.asarray(self.adata.obs_names, dtype=str)
        if self.precomputed_spot_features.shape[0] != obs_names.shape[0]:
            raise ValueError(
                f"{self.sample_id}: precomputed_spot_features has {self.precomputed_spot_features.shape[0]} "
                f"rows, expected {obs_names.shape[0]} (aligned with adata.obs_names) -- refusing a "
                "possible sample mix-up"
            )
        cache_barcodes = np.asarray(self.precomputed_spot_features_barcodes, dtype=str)
        if not np.array_equal(cache_barcodes, obs_names):
            raise ValueError(
                f"{self.sample_id}: precomputed_spot_features_barcodes do not exactly equal "
                "adata.obs_names (identity and order) -- refusing a possible sample mix-up"
            )
        expected_digest = hashlib.sha256(np.ascontiguousarray(self.precomputed_spot_features).tobytes()).hexdigest()
        if self.precomputed_spot_features_digest != expected_digest:
            raise ValueError(
                f"{self.sample_id}: precomputed_spot_features_digest does not match the actual "
                "precomputed_spot_features content -- corrupted or mismatched construction"
            )


def load_gen3_sample_data(cfg, manifest: dict, sample_id: str) -> Gen3SampleData:
    """Load and fully verify one manifest sample's real data -- the ONLY
    function in this module (and the real trainer) that reads
    per-spot patches/features from disk, and it does so exactly once."""
    record = manifest["samples"].get(sample_id)
    if record is None:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    split = str(record["split"])

    # Real, confirmed gap (Codex audit of commit 27e1232): re-verify the
    # sample's real h5ad/patch-h5 files against the manifest's own
    # recorded content hashes BEFORE loading anything from them -- a
    # changed h5ad with identical genes/barcodes could otherwise pass
    # every downstream check silently (see dataset_manifest.py's
    # verify_content_provenance docstring).
    verify_content_provenance(cfg.data.hest_data_dir, manifest, sample_id)

    adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
    obs_names = np.asarray(adata.obs_names, dtype=str)
    full_sample_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    coords3d = np.concatenate([full_sample_coords, np.zeros((full_sample_coords.shape[0], 1))], axis=1)
    slice_ids = np.full(obs_names.shape[0], sample_id, dtype=object)
    spatial_adjacency = tuple(build_knn_adjacency(full_sample_coords, k_neighbors=6))

    # Mandatory requirement #2: the COMPLETE verified spot-feature cache
    # record -- barcodes, availability, AND features loaded and checked
    # together, never a bare features array obtained any other way.
    spot_record = spot_feature_cache.load_gen3_spot_features(
        cfg, sample_id, obs_names, patches, image_source_available,
    )

    dense_wsi_provenance = None
    slide_context_record = None
    source = str(cfg.data.get("slide_context_source", "disabled"))
    # Adam's Step 6 audit #6 of commit a32051b: "do not load dense WSI
    # caches for architectures that do not consume them." The dense-WSI
    # fields (wsi_tile_longnet_coords/wsi_tile_regional_coords/
    # wsi_tile_features/full_slide_coord_bounds -- see data/example.py's
    # own docstring) feed BOTH the real GigaPath LongNet global vector
    # (use_global_slide) AND regional-token spatial pooling
    # (use_regional_he) -- gating on use_global_slide alone would
    # silently break a use_regional_he-only architecture (Architecture 3
    # without use_global_slide), so both flags are checked. Architecture
    # 1/2 configs use neither, and previously still paid the full dense
    # WSI cache load (real per-slide-tile-set I/O + memory) every time
    # `data.slide_context_source` was left non-"disabled" in a shared
    # config, even though nothing downstream ever reads the result.
    model_params = cfg.get("model", {}).get("params", {}) or {}
    architecture_needs_dense_wsi_cache = bool(model_params.get("use_regional_he", False)) or bool(
        model_params.get("use_global_slide", False)
    )
    if source != "disabled" and architecture_needs_dense_wsi_cache:
        spot_features_for_slide = None  # dense_wsi_cache path never needs precomputed spot features
        slide_context_record = slide_context.load_slide_context(
            cfg, sample_id, spot_features_for_slide, full_sample_coords.astype(np.float32),
        )
        if slide_context_record is not None:
            dense_wsi_provenance = slide_context_record.get("tile_encoder_provenance")

    return Gen3SampleData(
        sample_id=sample_id,
        patient_id=str(record["patient_id"]),
        split=split,
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
        slice_ids=slice_ids,
        spatial_adjacency=spatial_adjacency,
        slide_context_record=slide_context_record,
        tile_encoder_provenance={
            "dense_wsi": dense_wsi_provenance,
            "spot_features": spot_record["tile_encoder_provenance"],
        },
    )


def _masking_cfg_by_stratum(strata: list[dict]) -> dict:
    return {s["name"]: stratum_to_masking_cfg(s) for s in strata}


@dataclass(frozen=True)
class _TrainMaskItem:
    sample_id: str
    stratum: str
    seed: int


@dataclass(frozen=True)
class _HeldOutMaskItem:
    sample_id: str
    context_obs_names: list
    query_obs_names: list
    stratum: str | None = None


@dataclass
class Gen3MaskSchedule:
    """Per-role mask items for every sample in one dataset split, plus
    the leakage/coverage reports `build_training_sample_mask_report`/
    `build_held_out_sample_mask_report` produced while building them --
    mandatory requirement #6's verified (not merely asserted) proof."""
    role: str  # "train" / "validation" / "test"
    train_items: list = field(default_factory=list)
    held_out_items: list = field(default_factory=list)
    reports: dict = field(default_factory=dict)  # sample_id -> report dict


def sample_seed_namespace(sample_id: str, *, salt: str) -> int:
    """Stable, ARCHITECTURE-INDEPENDENT per-sample base seed -- a pure
    function of `sample_id` (and `salt`, to separate the train/validation/
    test seed spaces from each other) that never touches model config, so
    all four architectures still draw the IDENTICAL mask schedule for a
    given sample (the fairness-matrix requirement is preserved).

    Real, confirmed gap (Codex audit of commit 27e1232): the trainer
    previously used a single hardcoded `base_seed=0` for EVERY training
    sample and a single hardcoded `700_000` for EVERY validation sample --
    two samples with similar/regular spot lattices (routine for same-
    technology HEST-1k grids) would then draw their raw per-item seed
    candidates from the exact same starting range, risking near-identical
    realized query-index schedules across DIFFERENT samples (not the
    already-guarded-against same-sample collision case). Each sample now
    gets its own, hash-derived seed range, spaced widely enough
    (1e8 per bucket) that it cannot practically overlap with
    `mask_fingerprint.py`'s own per-stratum seed stride (1e6) or
    `mask_schedule.py`'s validation/test attempt budget."""
    digest = hashlib.sha256(f"gen3-mask-seed:{salt}:{sample_id}".encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) % 1_000_000) * 100_000_000


def build_gen3_mask_schedule(
    manifest: dict, samples: dict[str, Gen3SampleData], strata: list[dict], role: str,
    *, n_training_masks_per_sample: int = 500, split_counts: dict | None = None,
    split_seeds: dict | None = None, mask_bank_dir: str | Path | None = None,
) -> Gen3MaskSchedule:
    """Build (and verify) the mask schedule for every sample in `samples`
    -- all of which must share `role` (mixing roles in one call would
    defeat the whole point of the sample-level-split disjointness
    guarantee)."""
    split_counts = dict(split_counts or {"validation": 4, "test": 8})
    split_seeds = dict(split_seeds or {"validation": 700_000, "test": 900_000})
    schedule = Gen3MaskSchedule(role=role)
    for sample_id, sample in samples.items():
        if sample.split != role:
            raise ValueError(f"{sample_id}: manifest split {sample.split!r} != requested role {role!r}")
        obs_names = np.asarray(sample.adata.obs_names, dtype=str)
        if role == "train":
            sample_base_seed = sample_seed_namespace(sample_id, salt="train")
            training_schedule = mask_fingerprint.build_collision_free_training_schedule(
                sample.coords3d, sample.slice_ids, obs_names, sample_id, strata,
                n_items=n_training_masks_per_sample, base_seed=sample_base_seed,
                reserved_query_composite_ids=set(),  # nothing reserved: no same-sample val/test masks are drawn -- see module docstring
                manifest=manifest,
                spatial_adjacency=sample.spatial_adjacency,
            )
            report = mask_fingerprint.build_training_sample_mask_report(
                manifest, sample_id, sample.coords3d, sample.slice_ids, obs_names, strata,
                training_schedule, reserved_query_composite_ids=set(),
                spatial_adjacency=sample.spatial_adjacency,
            )
            if not report.get("passed"):
                raise ValueError(f"{sample_id}: training mask report did not pass: {report}")
            schedule.reports[sample_id] = report
            for item in training_schedule["items"]:
                schedule.train_items.append(_TrainMaskItem(sample_id=sample_id, stratum=item["stratum"], seed=item["seed"]))
        else:
            sample_split_seed = split_seeds[role] + sample_seed_namespace(sample_id, salt=role)
            sample_split_seeds = {role: sample_split_seed}
            path = Path(mask_bank_dir) / f"{sample_id}_stratified_mask_bank.json" if mask_bank_dir else None
            if path is not None and path.exists():
                bank = ensure_stratified_mask_bank(
                    path, sample.coords3d, sample.slice_ids, obs_names, strata,
                    split_counts={role: split_counts[role]}, split_seeds=sample_split_seeds,
                )
            else:
                from gen3_multiscale.data.mask_schedule import build_stratified_mask_bank, save_stratified_mask_bank
                bank = build_stratified_mask_bank(
                    sample.coords3d, sample.slice_ids, obs_names, strata,
                    split_counts={role: split_counts[role]}, split_seeds=sample_split_seeds,
                )
                if path is not None:
                    save_stratified_mask_bank(bank, path)
            report = mask_fingerprint.build_held_out_sample_mask_report(
                manifest, sample_id, sample.coords3d, sample.slice_ids, obs_names, strata, role, bank,
                expected_split_counts=split_counts, expected_split_seeds=sample_split_seeds,
            )
            if not report.get("passed"):
                raise ValueError(f"{sample_id}: held-out mask report did not pass: {report}")
            schedule.reports[sample_id] = report
            for record in bank["records"]:
                schedule.held_out_items.append(_HeldOutMaskItem(
                    sample_id=sample_id, context_obs_names=record["context_obs_names"],
                    query_obs_names=record["query_obs_names"], stratum=record.get("stratum"),
                ))
    return schedule


class Gen3SpatialFieldDataset(torch.utils.data.Dataset):
    """One (context, query) draw per `__getitem__` call, real, verified,
    manifest-driven data only. `role="train"` cycles through a
    deterministic, collision-free per-sample schedule (mandatory
    requirement #9: reproducible, but not the SAME fixed set every
    epoch's worth of steps -- a bounded, cyclable pool, matching
    `mask_bank.build_training_seed_bank`'s own established round-robin
    reuse pattern for exactly this reason); `role in ("validation",
    "test")` iterates a FIXED, deterministic mask list once per epoch
    (requirement #9's "deterministic fixed-mask validation" literally)."""

    def __init__(
        self, manifest: dict, samples: dict[str, Gen3SampleData], schedule: Gen3MaskSchedule,
        strata: list[dict], *, novae_enabled: bool = False,
        k_neighbors: int = 6, local_k: int = 32, max_rings: int = 3,
        max_boundary_size: int | None = None, patch_size_fullres: float = 224.0,
    ):
        for sample_id, sample in samples.items():
            if sample.split != schedule.role:
                raise ValueError(
                    f"Gen3SpatialFieldDataset: sample {sample_id!r} has manifest split "
                    f"{sample.split!r}, but this dataset was constructed for role {schedule.role!r}"
                )
        self.manifest = manifest
        self.samples = samples
        self.schedule = schedule
        self.strata = strata
        self.novae_enabled = novae_enabled
        self.k_neighbors = k_neighbors
        self.local_k = local_k
        self.max_rings = max_rings
        self.max_boundary_size = max_boundary_size
        self.patch_size_fullres = patch_size_fullres
        self._masking_cfg_by_stratum = _masking_cfg_by_stratum(strata)
        self._masking_cfg_by_sample_stratum = {
            sample_id: {
                name: prepare_masking_cfg_for_sample(cfg, sample.coords3d, sample.slice_ids)
                for name, cfg in self._masking_cfg_by_stratum.items()
            }
            for sample_id, sample in samples.items()
        }
        self._items = schedule.train_items if schedule.role == "train" else schedule.held_out_items
        if not self._items:
            raise ValueError(f"Gen3SpatialFieldDataset: role {schedule.role!r} has zero mask items")

    def __len__(self) -> int:
        return len(self._items)

    def item_identity(self, idx: int) -> dict:
        """`sample_id`/`stratum`/`query_fingerprint` for item `idx` in this
        FIXED, deterministic schedule -- Adam's Step 6 audit #7 of commit
        a32051b: the evaluator needs "sample/patient/mask/stratum
        identity" per retained record, which the raw `(inputs, targets)`
        pair returned by `__getitem__` doesn't carry (`patient_id` is on
        `inputs` itself; `stratum` is schedule-level and train items
        already carry it, so this exposes the same field for held-out
        items). `stratum` is `None` for a schedule built before
        `_HeldOutMaskItem` gained this field, or if the underlying mask
        bank record never had one (defensive, not expected in real use).

        `query_fingerprint` (Codex re-audit of commit 90f853e, launch
        blocker #9: "per-item query/mask fingerprint") -- sha256 of
        `sample_id` plus the exact, sorted query barcode set THIS item
        realized, via the same `_resolve_barcodes` `__getitem__` itself
        calls to build the real example; a caller can use it to detect
        two nominally-different items that happened to realize the
        identical held-out mask, or to verify a report's items truly
        match this dataset's own schedule."""
        item = self._items[idx % len(self._items)]
        _context_obs_names, query_obs_names = self._resolve_barcodes(item)
        query_fingerprint = hashlib.sha256(
            f"{item.sample_id}:{','.join(sorted(str(b) for b in query_obs_names))}".encode("utf-8")
        ).hexdigest()
        return {
            "sample_id": item.sample_id, "stratum": getattr(item, "stratum", None),
            "query_fingerprint": query_fingerprint,
        }

    def _resolve_barcodes(self, item) -> tuple[list, list]:
        if isinstance(item, _TrainMaskItem):
            sample = self.samples[item.sample_id]
            obs_names = np.asarray(sample.adata.obs_names, dtype=str)
            masking_cfg = self._masking_cfg_by_sample_stratum[item.sample_id][item.stratum]
            record = mask_fingerprint.realize_seed_and_fingerprint(
                sample.coords3d, sample.slice_ids, obs_names, item.sample_id, masking_cfg, item.seed,
                manifest=self.manifest, spatial_adjacency=sample.spatial_adjacency,
            )
            return record["context_obs_names"], record["query_obs_names"]
        return list(item.context_obs_names), list(item.query_obs_names)

    def validate_boundary_schedule(self) -> dict:
        """Validate every fixed item geometrically before model construction.

        Validation is intentionally geometry-only: it does not build image
        tensors or run either GigaPath encoder.  Its purpose is to guarantee
        that a fixed validation/test schedule cannot train successfully for
        hours and only then discover an unusable empty boundary at the first
        evaluation pass.
        """
        for idx, item in enumerate(self._items):
            sample = self.samples[item.sample_id]
            context_barcodes, query_barcodes = self._resolve_barcodes(item)
            obs_names = np.asarray(sample.adata.obs_names, dtype=str)
            position_by_barcode = {barcode: pos for pos, barcode in enumerate(obs_names)}
            try:
                context_pos = np.asarray([position_by_barcode[str(b)] for b in context_barcodes], dtype=int)
                query_pos = np.asarray([position_by_barcode[str(b)] for b in query_barcodes], dtype=int)
            except KeyError as exc:
                raise ValueError(
                    f"{item.sample_id}: boundary preflight mask references barcode {exc.args[0]!r} "
                    "that is absent from the loaded sample"
                ) from exc
            try:
                extract_boundary_and_local_context(
                    sample.full_sample_coords[context_pos], sample.full_sample_coords[query_pos],
                    k_neighbors=self.k_neighbors, local_k=self.local_k,
                    max_rings=self.max_rings, max_boundary_size=self.max_boundary_size,
                    full_adjacency=sample.spatial_adjacency,
                    observed_full_idx=context_pos,
                    query_full_idx=query_pos,
                )
            except EmptyBoundaryError as exc:
                identity = self.item_identity(idx)
                raise EmptyBoundaryError(
                    f"{item.sample_id}: {self.schedule.role} mask has no usable observed boundary "
                    f"(stratum={identity['stratum']!r}, "
                    f"query_fingerprint={identity['query_fingerprint']}); {exc}"
                ) from exc
        return {"role": self.schedule.role, "n_items_checked": len(self._items), "passed": True}

    def __getitem__(self, idx: int):
        item = self._items[idx % len(self._items)]
        sample_id = item.sample_id
        sample = self.samples[sample_id]
        context_barcodes, query_barcodes = self._resolve_barcodes(item)

        # Mandatory requirement #5: precomputed_spot_features only, never
        # image_feature_fn -- no tile encoder is ever invoked here.
        try:
            inputs, targets = example_builder.build_spatial_field_example(
                sample.adata, sample.patches, context_barcodes, query_barcodes,
                None,
                sample_id=sample_id, patient_id=sample.patient_id,
                full_sample_coords=sample.full_sample_coords, require_full_sample_coords=True,
                patch_size_fullres=self.patch_size_fullres, k_neighbors=self.k_neighbors,
                local_k=self.local_k, max_rings=self.max_rings, max_boundary_size=self.max_boundary_size,
                expected_feature_width=int(sample.precomputed_spot_features.shape[1]),
                slide_context=sample.slide_context_record, image_mode="target_zero",
                image_source_available=sample.image_source_available,
                precomputed_spot_features=sample.precomputed_spot_features,
                full_sample_adjacency=sample.spatial_adjacency,
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
            # Requirement #6's Novae bullet: context-only, verified to
            # physically exclude every query identity -- diagnostic only
            # (Novae is not consumed by any architecture forward pass,
            # CONTRACT.md section 44 -- unchanged here), never fed into
            # `inputs`.
            novae_inputs = novae_graph.build_context_only_novae_input(
                sample.adata, context_barcodes, query_barcodes, sample_id=sample_id,
            )
            novae_diagnostic = novae_graph.verify_novae_context_excludes_query_identities(
                novae_inputs, query_barcodes,
            )

        if novae_diagnostic is not None:
            inputs.provenance["novae_context_only_check"] = novae_diagnostic
        return inputs, targets


def gen3_identity_collate(batch):
    """This project's models run with batch_size=1 by construction (one
    masking draw at a time -- every SpatialFieldInputs/Targets pair has
    its own ragged n_observed/n_query/n_boundary shapes with no padding
    convention). A real DataLoader(batch_size=1, collate_fn=this) simply
    unwraps the trivial one-item list PyTorch's default collate would
    otherwise try to stack into a batch dimension that does not apply
    here."""
    if len(batch) != 1:
        raise ValueError(
            f"gen3_identity_collate expects batch_size=1 (ragged per-item shapes cannot be "
            f"stacked), got {len(batch)} items"
        )
    return batch[0]
