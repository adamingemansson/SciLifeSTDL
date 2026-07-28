"""Model factory -- fixes a real, confirmed gap (Codex audit finding #4
against commit 386bcf4): the real configs/architectureN.yaml files
include fields (gene_encoder_type, init_seed, ...) no architecture
constructor accepts, so `Architecture1(**config["model"]["params"])`
would raise a TypeError. Tests exercise the factory against the REAL
four config files, not just synthetic fixtures, and directly verify
finding #9's "guarantee identical shared initialization -- not merely
identical seeds" at the resolved-config level (constructing real models
from the real configs and diffing their actual parameters, not just
diffing kwargs dictionaries)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.models.model_factory import build_architecture, resolve_model_kwargs, synchronize_shared_initialization

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(_CONFIG_DIR / f"{name}.yaml"), resolve=True)


def _gene_basis(n_genes=6, rank=4, seed=0):
    rng = np.random.default_rng(seed)
    gene_names = [f"g{i}" for i in range(n_genes)]
    return fit_gene_residual_basis(rng.normal(size=(20, n_genes)), gene_names, rank=rank), gene_names


def test_resolve_model_kwargs_strips_known_metadata_fields():
    config = _load("architecture1")
    kwargs = resolve_model_kwargs(config, n_genes=6, gex_feature_dim=4)
    assert "gene_encoder_type" not in kwargs
    assert "init_seed" not in kwargs
    assert kwargs["hidden_dim"] == 512
    assert kwargs["use_anchor_blend"] is False


def test_resolve_model_kwargs_rejects_a_gene_encoder_type_other_than_weighted_linear():
    """gene_encoder_type is no longer pure metadata: _SharedFieldArchitecture
    unconditionally builds a WeightedGeneExpressionEncoder now, so a config
    claiming a different encoder would silently get a mismatched model --
    must fail closed instead."""
    config = _load("architecture1")
    config["model"]["params"]["gene_encoder_type"] = "mlp"
    with pytest.raises(ValueError, match="gene_encoder_type"):
        resolve_model_kwargs(config, n_genes=6, gex_feature_dim=4)


def test_resolve_model_kwargs_raises_on_an_unrecognized_field():
    config = _load("architecture1")
    config["model"]["params"]["totally_unknown_field"] = 1
    with pytest.raises(ValueError, match="totally_unknown_field"):
        resolve_model_kwargs(config, n_genes=6, gex_feature_dim=4)


def test_resolve_model_kwargs_rejects_an_unknown_architecture_id():
    config = _load("architecture1")
    config["model"]["architecture"] = "99"
    with pytest.raises(ValueError, match="unknown model.architecture"):
        resolve_model_kwargs(config, n_genes=6, gex_feature_dim=4)


def test_build_architecture_raises_value_error_not_key_error_for_an_unknown_id():
    """Regression test for a real, confirmed bug (2nd Codex re-audit of
    commit 547f51e): build_architecture indexed _ARCHITECTURE_CLASSES
    BEFORE calling resolve_model_kwargs's validation, so an unknown
    architecture id raised a raw KeyError instead of the intended,
    actionable ValueError."""
    config = _load("architecture1")
    config["model"]["architecture"] = "99"
    with pytest.raises(ValueError, match="unknown model.architecture"):
        build_architecture(config, n_genes=6, gex_feature_dim=4)


def test_build_architecture4_requires_gene_basis_and_gene_names():
    config = _load("architecture4")
    with pytest.raises(ValueError, match="gene_basis and gene_names"):
        build_architecture(config, n_genes=6, gex_feature_dim=4)


@pytest.mark.parametrize("name", ["architecture1", "architecture2", "architecture3", "architecture4"])
def test_build_architecture_constructs_a_real_model_from_each_real_config(name):
    """The core regression test for finding #4: every real config file
    must actually construct its architecture without a TypeError."""
    config = _load(name)
    kwargs = {}
    if name == "architecture4":
        basis, gene_names = _gene_basis()
        kwargs = {"gene_basis": basis, "gene_names": gene_names}
    model = build_architecture(config, n_genes=6, gex_feature_dim=4, **kwargs)
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_build_architecture_from_real_configs_gives_architecture_1_and_2_identical_shared_initialization():
    """Finding #9: "guarantee identical shared initialization -- not
    merely identical seeds." Both architecture1.yaml and architecture2.yaml
    declare init_seed: 0 -- verified here at the level that actually
    matters: constructing REAL models from the REAL config files (via the
    factory, exercising the exact path a real training entrypoint would
    use) and diffing their actual parameters, not hand-crafted test
    kwargs."""
    model1 = build_architecture(_load("architecture1"), n_genes=6, gex_feature_dim=4)
    model2 = build_architecture(_load("architecture2"), n_genes=6, gex_feature_dim=4)
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    shared_names = set(params1) & set(params2)
    assert len(shared_names) > 10  # sanity: they really do share most modules
    for name in shared_names:
        if name == "transport_head.blend_logit":
            continue  # Architecture 2's only real divergence (deterministically filled, not RNG-consuming)
        assert torch.equal(params1[name], params2[name]), f"{name} diverged despite identical init_seed"


# ---------------------------------------------------------------------------
# synchronize_shared_initialization -- fixes a real, confirmed gap (2nd
# Codex re-audit of commit 547f51e): "Architecture 3 constructs additional
# randomly initialized global-GEX modules before some shared modules...
# causing later shared parameters to differ despite using the same seed."
# ---------------------------------------------------------------------------
def test_synchronize_shared_initialization_copies_matching_named_parameters():
    reference = torch.nn.Linear(4, 4)
    other = torch.nn.Linear(4, 4)
    assert not torch.equal(reference.weight, other.weight)  # different RNG draws
    models = {"reference": reference, "other": other}
    synchronized = synchronize_shared_initialization(models, reference="reference")
    assert torch.equal(reference.weight, other.weight)
    assert torch.equal(reference.bias, other.bias)
    assert set(synchronized["other"]) == {"weight", "bias"}


def test_synchronize_shared_initialization_supports_a_name_prefix():
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.conditioner = inner

    reference = torch.nn.Linear(3, 3)
    wrapped = Wrapper(torch.nn.Linear(3, 3))
    models = {"reference": reference, "wrapped": wrapped}
    synchronized = synchronize_shared_initialization(
        models, reference="reference", name_prefixes={"wrapped": "conditioner."},
    )
    assert torch.equal(reference.weight, wrapped.conditioner.weight)
    assert synchronized["wrapped"] == ["conditioner.weight", "conditioner.bias"]


def test_synchronize_shared_initialization_never_touches_parameters_with_a_shape_mismatch():
    reference = torch.nn.Linear(4, 4)
    other = torch.nn.Linear(4, 4)
    other.extra = torch.nn.Linear(9, 9)  # no counterpart in reference at all -- must be left alone
    original_extra_weight = other.extra.weight.clone()
    synchronize_shared_initialization({"reference": reference, "other": other}, reference="reference")
    assert torch.equal(other.extra.weight, original_extra_weight)


def test_synchronize_shared_initialization_gives_all_four_real_architectures_identical_shared_parameters():
    """The full fix, not just the 1-vs-2 case Phase 6's original tests
    honestly scoped down to: build all FOUR real architectures from the
    real config files, synchronize, and verify every genuinely shared
    parameter is byte-identical across every pair -- including
    Architecture 3 and Architecture 4 (via its conditioner), which
    legitimately diverge in RNG-stream order during construction and were
    never proven identical before this fix."""
    basis, gene_names = _gene_basis(n_genes=6)
    model1 = build_architecture(_load("architecture1"), n_genes=6, gex_feature_dim=4)
    model2 = build_architecture(_load("architecture2"), n_genes=6, gex_feature_dim=4)
    model3 = build_architecture(_load("architecture3"), n_genes=6, gex_feature_dim=4)
    model4 = build_architecture(_load("architecture4"), n_genes=6, gex_feature_dim=4, gene_basis=basis, gene_names=gene_names)
    models = {"architecture1": model1, "architecture2": model2, "architecture3": model3, "architecture4": model4}

    synchronized = synchronize_shared_initialization(
        models, reference="architecture1", name_prefixes={"architecture4": "conditioner."},
    )
    assert len(synchronized["architecture2"]) > 10
    assert len(synchronized["architecture3"]) > 10
    assert len(synchronized["architecture4"]) > 10  # the previously-unaddressed case

    flat_params = {
        name: {
            (p_name[len("conditioner."):] if name == "architecture4" and p_name.startswith("conditioner.") else p_name): p
            for p_name, p in model.named_parameters()
        }
        for name, model in models.items()
    }
    reference_params = flat_params["architecture1"]
    for name, params in flat_params.items():
        if name == "architecture1":
            continue
        shared = set(reference_params) & set(params)
        for param_name in shared:
            if param_name == "transport_head.blend_logit":
                continue  # Architecture 2's only real, deterministic (non-RNG) divergence
            if reference_params[param_name].shape != params[param_name].shape:
                # A genuine STRUCTURAL divergence, not something
                # synchronize_shared_initialization could or should copy
                # -- e.g. backbone branch_gate has 2 output branches for
                # Architecture 1 (local+boundary) but 3 for Architecture
                # 3/4 (local+boundary+global-GEX), so the two tensors can
                # never be shape-compatible regardless of seed.
                continue
            assert torch.equal(reference_params[param_name], params[param_name]), (
                f"{name}.{param_name} diverged after synchronize_shared_initialization"
            )
