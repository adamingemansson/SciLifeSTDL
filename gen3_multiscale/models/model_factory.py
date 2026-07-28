"""Model factory -- bridges a resolved config dict (configs/architectureN.yaml)
to a real Architecture1-4 instance.

Confirmed real gap (Codex audit finding #4 against commit 386bcf4,
verified against the actual code before fixing): `Architecture1(**config["model"]["params"])`
would raise a TypeError, because the configs' `model.params` blocks
include fields no architecture constructor accepts at all --
`gene_encoder_type` (a Phase-4 documentation-only field recording which
conditioning encoder produced `observed_gex_conditioning` upstream of
this package -- CONTRACT.md section 10) and `init_seed` (consumed by a
CALLER via `torch.manual_seed()` before construction, never passed into
any constructor). Architecture 4's config additionally has
`gene_basis_rank`/`n_flow_samples`/`n_ode_steps`, only some of which are
real Architecture4 constructor kwargs. This module is the missing glue,
built so `static_config_audit` (training/launch_four_gpu_suite.py) can be
strengthened to compare RESOLVED constructor kwargs, not just raw YAML
text -- "normalize effective configs before comparing them" (the same
audit's finding #9).
"""
from __future__ import annotations

import inspect

import torch

from gen3_multiscale.models.architectures import (
    Architecture1, Architecture2, Architecture3, Architecture4, _SharedFieldArchitecture,
)
from gen3_multiscale.models.gene_basis import GeneResidualBasis

_ARCHITECTURE_CLASSES = {"1": Architecture1, "2": Architecture2, "3": Architecture3, "4": Architecture4}

# Architecture1/2/3's own __init__ is a thin `**kwargs`-forwarding
# wrapper around _SharedFieldArchitecture.__init__ (setting a few
# defaults, e.g. use_anchor_blend, before calling super().__init__) --
# inspecting Architecture1.__init__ directly would only ever see
# `(self, **kwargs)` and accept nothing, so the REAL accepted parameter
# set for those three lives on the shared base class. Architecture4's
# own __init__ is not a pass-through wrapper (its parameter list is
# genuinely different from the base class), so it is inspected directly.
_CONSTRUCTOR_SIGNATURE_SOURCE = {
    Architecture1: _SharedFieldArchitecture, Architecture2: _SharedFieldArchitecture,
    Architecture3: _SharedFieldArchitecture, Architecture4: Architecture4,
}

# Config fields under model.params that are metadata for something OTHER
# than an architecture constructor -- known and explicitly accounted for,
# never silently dropped. Anything present in a config that is neither a
# real constructor parameter nor listed here raises (fail closed on an
# unrecognized field, e.g. a typo, rather than silently ignoring it).
_KNOWN_NON_CONSTRUCTOR_FIELDS = frozenset({"gene_encoder_type", "init_seed"})

# Fields Architecture4's config declares under a DIFFERENT name/shape
# than its constructor kwarg: gene_basis_rank configures the rank
# fit_gene_residual_basis was/will be called with OFFLINE (this factory
# never fits a basis itself -- a caller must supply an already-fit
# GeneResidualBasis, mirroring Architecture4's own "never fits one
# itself" discipline), so it is metadata here, not passed through.
_ARCHITECTURE4_EXTRA_NON_CONSTRUCTOR_FIELDS = frozenset({"gene_basis_rank"})


def _constructor_signature(architecture_cls: type) -> inspect.Signature:
    return inspect.signature(_CONSTRUCTOR_SIGNATURE_SOURCE[architecture_cls].__init__)


def _constructor_param_names(architecture_cls: type) -> frozenset[str]:
    params = _constructor_signature(architecture_cls).parameters
    return frozenset(params) - {"self"}


def resolve_model_kwargs(
    config: dict,
    *,
    n_genes: int,
    gex_feature_dim: int,
    gene_basis: GeneResidualBasis | None = None,
    gene_names: list[str] | None = None,
) -> dict:
    """Turn a resolved config dict into the exact kwargs its
    architecture's constructor accepts. Fails closed (raises ValueError)
    on any `model.params` field that is neither a real constructor
    parameter nor a known non-constructor field -- silently dropping an
    unrecognized field would hide a typo or a genuinely missing wiring
    decision, not just cosmetic config noise.
    """
    model_cfg = config.get("model") or {}
    architecture_id = str(model_cfg.get("architecture", ""))
    if architecture_id not in _ARCHITECTURE_CLASSES:
        raise ValueError(f"unknown model.architecture {architecture_id!r} -- expected one of {sorted(_ARCHITECTURE_CLASSES)}")
    architecture_cls = _ARCHITECTURE_CLASSES[architecture_id]
    accepted = _constructor_param_names(architecture_cls)

    known_metadata = _KNOWN_NON_CONSTRUCTOR_FIELDS
    if architecture_id == "4":
        known_metadata = known_metadata | _ARCHITECTURE4_EXTRA_NON_CONSTRUCTOR_FIELDS

    raw_params = dict(model_cfg.get("params") or {})
    kwargs: dict = {}
    for key, value in raw_params.items():
        if key in {"n_genes", "gex_feature_dim"}:
            continue  # always supplied explicitly below, from real data, never from the static config
        if key in accepted:
            kwargs[key] = value
        elif key in known_metadata:
            continue
        else:
            raise ValueError(
                f"config field model.params.{key!r} is neither a constructor parameter of "
                f"{architecture_cls.__name__} nor a known non-constructor field "
                f"({sorted(known_metadata)}) -- refusing to silently drop it"
            )

    kwargs["n_genes"] = n_genes
    kwargs["gex_feature_dim"] = gex_feature_dim
    if architecture_id == "4":
        if gene_basis is None or gene_names is None:
            raise ValueError("Architecture 4 requires gene_basis and gene_names (fit offline, outside this factory)")
        kwargs["gene_basis"] = gene_basis
        kwargs["gene_names"] = gene_names

    required = {
        name for name, param in _constructor_signature(architecture_cls).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
    }
    missing = required - set(kwargs)
    if missing:
        raise ValueError(
            f"config is missing required constructor arguments for {architecture_cls.__name__}: "
            f"{sorted(missing)}"
        )
    return kwargs


def build_architecture(
    config: dict,
    *,
    n_genes: int,
    gex_feature_dim: int,
    gene_basis: GeneResidualBasis | None = None,
    gene_names: list[str] | None = None,
    seed: int | None = None,
) -> torch.nn.Module:
    """Construct a real Architecture1-4 instance from a resolved config.
    `seed` overrides `model.params.init_seed` when given; otherwise that
    config field (if present) is used -- either way, `torch.manual_seed`
    is called immediately before construction so "identical shared
    initialization... not merely identical seeds" (Codex audit finding
    #9) is actually exercised here, not left to a caller to remember."""
    model_cfg = config.get("model") or {}
    architecture_id = str(model_cfg.get("architecture", ""))
    architecture_cls = _ARCHITECTURE_CLASSES[architecture_id]
    kwargs = resolve_model_kwargs(
        config, n_genes=n_genes, gex_feature_dim=gex_feature_dim, gene_basis=gene_basis, gene_names=gene_names,
    )
    effective_seed = seed if seed is not None else (model_cfg.get("params") or {}).get("init_seed")
    if effective_seed is not None:
        torch.manual_seed(int(effective_seed))
    return architecture_cls(**kwargs)
