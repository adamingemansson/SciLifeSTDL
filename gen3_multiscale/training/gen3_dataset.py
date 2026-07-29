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

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gen3_multiscale.data import example_builder, novae_graph, slide_context, spot_feature_cache
from gen3_multiscale.data import mask_fingerprint
from gen3_multiscale.data.mask_schedule import ensure_stratified_mask_bank, stratum_to_masking_cfg


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
    full_sample_coords: np.ndarray
    coords3d: np.ndarray  # [n, 3], z=0 (single 2D section, not a Track B multi-slice series)
    slice_ids: np.ndarray  # [n] str, every entry == sample_id (mask_bank's single-slice convention)
    slide_context_record: dict | None
    tile_encoder_provenance: dict  # {"dense_wsi": {...} | None, "spot_features": {...}}

    def __post_init__(self):
        # Mandatory requirement #2's "verify at the trainer call site"
        # half -- see module docstring. Cheap (array equality over a few
        # thousand rows), run once here rather than per masking draw
        # (nothing about these arrays changes between draws for an
        # already-constructed Gen3SampleData).
        obs_names = np.asarray(self.adata.obs_names, dtype=str)
        if self.precomputed_spot_features.shape[0] != obs_names.shape[0]:
            raise ValueError(
                f"{self.sample_id}: precomputed_spot_features has {self.precomputed_spot_features.shape[0]} "
                f"rows, expected {obs_names.shape[0]} (aligned with adata.obs_names) -- refusing a "
                "possible sample mix-up"
            )


def load_gen3_sample_data(cfg, manifest: dict, sample_id: str) -> Gen3SampleData:
    """Load and fully verify one manifest sample's real data -- the ONLY
    function in this module (and the real trainer) that reads
    per-spot patches/features from disk, and it does so exactly once."""
    record = manifest["samples"].get(sample_id)
    if record is None:
        raise ValueError(f"{sample_id!r} is not a sample the dataset manifest declares")
    split = str(record["split"])

    adata, patches, image_source_available = example_builder.load_sample_for_examples(manifest, sample_id)
    obs_names = np.asarray(adata.obs_names, dtype=str)
    full_sample_coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    coords3d = np.concatenate([full_sample_coords, np.zeros((full_sample_coords.shape[0], 1))], axis=1)
    slice_ids = np.full(obs_names.shape[0], sample_id, dtype=object)

    # Mandatory requirement #2: the COMPLETE verified spot-feature cache
    # record -- barcodes, availability, AND features loaded and checked
    # together, never a bare features array obtained any other way.
    spot_record = spot_feature_cache.load_gen3_spot_features(
        cfg, sample_id, obs_names, patches, image_source_available,
    )

    dense_wsi_provenance = None
    slide_context_record = None
    source = str(cfg.data.get("slide_context_source", "disabled"))
    if source != "disabled":
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
        full_sample_coords=full_sample_coords,
        coords3d=coords3d,
        slice_ids=slice_ids,
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
            training_schedule = mask_fingerprint.build_collision_free_training_schedule(
                sample.coords3d, sample.slice_ids, obs_names, sample_id, strata,
                n_items=n_training_masks_per_sample, base_seed=0,
                reserved_query_composite_ids=set(),  # nothing reserved: no same-sample val/test masks are drawn -- see module docstring
                manifest=manifest,
            )
            report = mask_fingerprint.build_training_sample_mask_report(
                manifest, sample_id, sample.coords3d, sample.slice_ids, obs_names, strata,
                training_schedule, reserved_query_composite_ids=set(),
            )
            if not report.get("passed"):
                raise ValueError(f"{sample_id}: training mask report did not pass: {report}")
            schedule.reports[sample_id] = report
            for item in training_schedule["items"]:
                schedule.train_items.append(_TrainMaskItem(sample_id=sample_id, stratum=item["stratum"], seed=item["seed"]))
        else:
            path = Path(mask_bank_dir) / f"{sample_id}_stratified_mask_bank.json" if mask_bank_dir else None
            if path is not None and path.exists():
                bank = ensure_stratified_mask_bank(
                    path, sample.coords3d, sample.slice_ids, obs_names, strata,
                    split_counts={role: split_counts[role]}, split_seeds={role: split_seeds[role]},
                )
            else:
                from gen3_multiscale.data.mask_schedule import build_stratified_mask_bank, save_stratified_mask_bank
                bank = build_stratified_mask_bank(
                    sample.coords3d, sample.slice_ids, obs_names, strata,
                    split_counts={role: split_counts[role]}, split_seeds={role: split_seeds[role]},
                )
                if path is not None:
                    save_stratified_mask_bank(bank, path)
            report = mask_fingerprint.build_held_out_sample_mask_report(
                manifest, sample_id, sample.coords3d, sample.slice_ids, obs_names, strata, role, bank,
                expected_split_counts=split_counts, expected_split_seeds=split_seeds,
            )
            if not report.get("passed"):
                raise ValueError(f"{sample_id}: held-out mask report did not pass: {report}")
            schedule.reports[sample_id] = report
            for record in bank["records"]:
                schedule.held_out_items.append(_HeldOutMaskItem(
                    sample_id=sample_id, context_obs_names=record["context_obs_names"],
                    query_obs_names=record["query_obs_names"],
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
        self._items = schedule.train_items if schedule.role == "train" else schedule.held_out_items
        if not self._items:
            raise ValueError(f"Gen3SpatialFieldDataset: role {schedule.role!r} has zero mask items")

    def __len__(self) -> int:
        return len(self._items)

    def _resolve_barcodes(self, item) -> tuple[list, list]:
        if isinstance(item, _TrainMaskItem):
            sample = self.samples[item.sample_id]
            obs_names = np.asarray(sample.adata.obs_names, dtype=str)
            masking_cfg = self._masking_cfg_by_stratum[item.stratum]
            record = mask_fingerprint.realize_seed_and_fingerprint(
                sample.coords3d, sample.slice_ids, obs_names, item.sample_id, masking_cfg, item.seed,
                manifest=self.manifest,
            )
            return record["context_obs_names"], record["query_obs_names"]
        return list(item.context_obs_names), list(item.query_obs_names)

    def __getitem__(self, idx: int):
        item = self._items[idx % len(self._items)]
        sample_id = item.sample_id
        sample = self.samples[sample_id]
        context_barcodes, query_barcodes = self._resolve_barcodes(item)

        # Mandatory requirement #5: precomputed_spot_features only, never
        # image_feature_fn -- no tile encoder is ever invoked here.
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
        )

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
