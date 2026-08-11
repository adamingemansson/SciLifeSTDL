"""Canonical Gen6 arm contract.

The table below is the single source of truth for construction, preflight,
data/cache loading and reports.  In particular, cache requirements are
derived here rather than inferred from a vaguely similar Gen4 arm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Gen6ArmSpec:
    arm: str
    description: str
    gene_encoder: str
    image_encoder: str
    fusion: str
    geometry: str
    spatial_model: str
    generator: str
    uses_uni2_spot: bool = False
    uses_uni2_dense: bool = False
    uses_gigapath_spot: bool = False
    uses_gigapath_dense: bool = False
    uses_scfoundation: bool = False
    uses_stpath: bool = False
    uses_sample_organ: bool = False
    staged_conditioner: bool = False
    staged_autoencoder: bool = False

    def cache_requirements(self) -> dict[str, bool]:
        return {
            "uses_uni2_primary": self.uses_uni2_spot,
            "uses_uni2_dense": self.uses_uni2_dense,
            "uses_gigapath_spot": self.uses_gigapath_spot,
            "uses_gigapath_dense": self.uses_gigapath_dense,
            "uses_scfoundation": self.uses_scfoundation,
            "uses_uni2_hybrid": False,
            "uses_sample_organ": self.uses_sample_organ,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _spec(arm: str, description: str, gene: str, image: str, fusion: str,
          geometry: str, spatial: str, generator: str, **requirements) -> Gen6ArmSpec:
    return Gen6ArmSpec(
        arm=arm, description=description, gene_encoder=gene,
        image_encoder=image, fusion=fusion, geometry=geometry,
        spatial_model=spatial, generator=generator, **requirements,
    )


# G6-6 through G6-12 deliberately use scFoundation + UNI2 as the fixed
# encoder pair.  The purpose of those arms is to isolate fusion/geometry/
# generator components; changing encoders at the same time would make the
# comparison uninterpretable.  The first four factorial arms establish the
# MLP/scFoundation x UNI2/GigaPath encoder comparison separately.
GEN6_ARM_SPECS: dict[str, Gen6ArmSpec] = {
    "gen6a": _spec(
        "gen6a", "pure pretrained STPath, unfrozen and fine-tuned end-to-end",
        "stpath_joint", "stpath_gigapath", "stpath_native", "frame_averaging",
        "stpath_native", "stpath_native_full_panel",
        uses_gigapath_spot=True, uses_stpath=True, uses_sample_organ=True,
    ),
    "gen6b": _spec(
        "gen6b", "MLP GEX + UNI2 with simple fusion",
        "weighted_linear", "uni2", "simple", "fourier_absolute",
        "spatial_field", "gene_transport",
        uses_uni2_spot=True,
    ),
    "gen6c": _spec(
        "gen6c", "scFoundation + UNI2 with simple fusion",
        "scfoundation", "uni2", "simple", "fourier_absolute",
        "spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6d": _spec(
        "gen6d", "MLP GEX + GigaPath/LongNet with simple fusion",
        "weighted_linear", "gigapath_longnet", "simple", "fourier_absolute",
        "spatial_field", "gene_transport",
        uses_gigapath_spot=True, uses_gigapath_dense=True,
    ),
    "gen6e": _spec(
        "gen6e", "scFoundation + GigaPath/LongNet with simple fusion",
        "scfoundation", "gigapath_longnet", "simple", "fourier_absolute",
        "spatial_field", "gene_transport",
        uses_gigapath_spot=True, uses_gigapath_dense=True, uses_scfoundation=True,
    ),
    "gen6f": _spec(
        "gen6f", "MoME fusion with frame-averaged geometry",
        "scfoundation", "uni2", "mome", "frame_averaging",
        "spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6g": _spec(
        "gen6g", "MoME fusion with learned relative-geometry bias",
        "scfoundation", "uni2", "mome", "relative_bias",
        "spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6h": _spec(
        "gen6h", "MoME fusion with Fourier spatial attention",
        "scfoundation", "uni2", "mome", "fourier_attention",
        "spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6i": _spec(
        "gen6i", "bidirectional image/GEX cross-attention",
        "scfoundation", "uni2", "bidirectional_cross_attention",
        "relative_bias", "spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6j": _spec(
        "gen6j", "full boundary/local/regional/global spatial-field transformer",
        "scfoundation", "uni2", "mome", "frame_averaging",
        "full_spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_uni2_dense=True, uses_scfoundation=True,
    ),
    "gen6k": _spec(
        "gen6k", "frozen Gen6-C conditioner plus learned-latent minibatch OT flow",
        "selected_conditioner", "selected_conditioner", "staged", "staged",
        "staged_conditioner", "latent_ot_flow",
        staged_conditioner=True, staged_autoencoder=True,
    ),
    "gen6l": _spec(
        "gen6l", "frozen Gen6-C conditioner plus WAE-GAN expression generator",
        "selected_conditioner", "selected_conditioner", "staged", "staged",
        "staged_conditioner", "wae_gan",
        staged_conditioner=True,
    ),
    # --- second component screen -------------------------------------------
    # Each arm below changes exactly ONE thing relative to an arm that has
    # already been trained, so the comparison stays readable:
    #
    #   gen6m, gen6n  vs gen6c  -- refine the prediction over the spot graph
    #   gen6o         vs gen6l  -- add the supervised conditional-mean head
    #   gen6p         vs gen6k  -- hard OT assignment instead of barycentric
    #
    # gen6m/gen6n follow from a measurement rather than a hunch.  On this
    # manifest's validation slides, simply averaging a spot's OBSERVED
    # neighbours scores 0.392 on the top-200 panel against a measured
    # count-split noise ceiling of 0.767.  Neighbour expression is the
    # strongest cheap signal in the data, and nothing in gen6b-gen6j
    # propagates PREDICTED expression between query spots: every query spot is
    # decoded independently from its own conditioning.  Refinement is the one
    # component that closes that gap.
    "gen6m": _spec(
        "gen6m", "Gen6-C plus two rounds of spatial refinement over predicted expression",
        "scfoundation", "uni2", "simple", "fourier_absolute",
        "refined_spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6n": _spec(
        "gen6n", "Gen6-C plus four rounds of spatial refinement over a wider neighbourhood",
        "scfoundation", "uni2", "simple", "fourier_absolute",
        "refined_spatial_field", "gene_transport",
        uses_uni2_spot=True, uses_scfoundation=True,
    ),
    "gen6o": _spec(
        "gen6o", "Gen6-L plus a supervised conditional-mean head and correlatable latent",
        "selected_conditioner", "selected_conditioner", "staged", "staged",
        "staged_conditioner", "wae_gan",
        staged_conditioner=True,
    ),
    "gen6p": _spec(
        "gen6p", "Gen6-K with hard OT assignment instead of barycentric averaging",
        "selected_conditioner", "selected_conditioner", "staged", "staged",
        "staged_conditioner", "latent_ot_flow",
        staged_conditioner=True, staged_autoencoder=True,
    ),
}

# Arms whose conditioner must refine its own predicted expression over the
# spot graph. Derived from the table rather than hard-coded in the factory, so
# the table stays the single source of truth for what an arm IS.
REFINED_SPATIAL_FIELD_ARMS = frozenset(
    arm for arm, spec in GEN6_ARM_SPECS.items()
    if spec.spatial_model == "refined_spatial_field"
)


def get_gen6_arm_spec(arm: str) -> Gen6ArmSpec:
    try:
        return GEN6_ARM_SPECS[str(arm)]
    except KeyError as exc:
        raise ValueError(
            f"unknown Gen6 arm {arm!r}; expected one of {sorted(GEN6_ARM_SPECS)}"
        ) from exc


def is_gen6_config(config: dict) -> bool:
    return str((config.get("model") or {}).get("arm", "")) in GEN6_ARM_SPECS


def gen6_cache_requirements(config: dict) -> dict[str, bool]:
    arm = str((config.get("model") or {}).get("arm", ""))
    spec = get_gen6_arm_spec(arm)
    if spec.staged_conditioner:
        selected = str(((config.get("model") or {}).get("params") or {}).get("conditioner_arm", ""))
        if selected != "gen6c":
            raise ValueError(
                f"{arm} must use the shared deterministic control conditioner gen6c, "
                f"got {selected!r}"
            )
        selected_spec = get_gen6_arm_spec(selected)
        if selected_spec.staged_conditioner:
            raise ValueError("a staged Gen6 generator must select a deterministic Gen6 conditioner")
        return selected_spec.cache_requirements()
    return spec.cache_requirements()
