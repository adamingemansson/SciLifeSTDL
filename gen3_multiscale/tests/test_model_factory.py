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
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.models.model_factory import (
    build_architecture, load_synchronized_initialization, persist_synchronized_initializations,
    resolve_model_kwargs, synchronize_four_architecture_initialization, synchronize_shared_initialization,
)

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


def _build_all_four(n_genes=6, gex_feature_dim=4):
    basis, gene_names = _gene_basis(n_genes=n_genes)
    return {
        "architecture1": build_architecture(_load("architecture1"), n_genes=n_genes, gex_feature_dim=gex_feature_dim),
        "architecture2": build_architecture(_load("architecture2"), n_genes=n_genes, gex_feature_dim=gex_feature_dim),
        "architecture3": build_architecture(_load("architecture3"), n_genes=n_genes, gex_feature_dim=gex_feature_dim),
        "architecture4": build_architecture(
            _load("architecture4"), n_genes=n_genes, gex_feature_dim=gex_feature_dim,
            gene_basis=basis, gene_names=gene_names,
        ),
    }


def _flat_params(models: dict) -> dict:
    return {
        name: {
            (p_name[len("conditioner."):] if name == "architecture4" and p_name.startswith("conditioner.") else p_name): p
            for p_name, p in model.named_parameters()
        }
        for name, model in models.items()
    }


def test_synchronize_shared_initialization_alone_cannot_reach_architecture_3_only_modules():
    """Documents the real gap a 3rd-round Codex re-audit found: a naive
    single-reference call (reference="architecture1") can never even
    ATTEMPT to synchronize gex_pool.* -- Architecture 1 has no gex_pool
    at all, so no name in the reference dict can ever match it. This is
    a structural fact about which NAMES get copied, independent of
    whether the values happen to already coincide for some other reason
    in a given run -- and it's what makes the second hop in
    synchronize_four_architecture_initialization below structurally
    necessary, not merely convenient. transport_head.* is checked as a
    positive control: it DOES exist on Architecture 1 too, so the
    single-hop call must still reach it."""
    models = _build_all_four()
    synchronized = synchronize_shared_initialization(
        {name: models[name] for name in ("architecture1", "architecture2", "architecture3")}, reference="architecture1",
    )
    assert not any(name.startswith("gex_pool.") for name in synchronized["architecture3"])
    assert any(name.startswith("transport_head.") for name in synchronized["architecture3"])


def test_synchronize_four_architecture_initialization_gives_all_four_real_architectures_identical_shared_parameters():
    """The full, corrected fix: build all FOUR real architectures from
    the real config files, synchronize with the two-hop helper, and
    verify every genuinely shared parameter is byte-identical across
    every pair -- including gex_pool.* between Architecture 3 and
    Architecture 4's conditioner, the exact case the single-hop version
    could never reach."""
    models = _build_all_four()
    synchronized = synchronize_four_architecture_initialization(models)
    assert len(synchronized["architecture2"]) > 10
    assert len(synchronized["architecture3"]) > 10
    assert len(synchronized["architecture4"]) > 10

    # The specific case the previous, single-hop test could never check:
    # gex_pool has no counterpart on Architecture 1 at all.
    assert "conditioner.gex_pool.inducing_queries" in synchronized["architecture4"]
    assert torch.equal(
        models["architecture3"].gex_pool.inducing_queries, models["architecture4"].conditioner.gex_pool.inducing_queries,
    )

    flat_params = _flat_params(models)
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
                f"{name}.{param_name} diverged after synchronize_four_architecture_initialization"
            )
    # Every parameter architecture3/architecture4's conditioner both
    # have (a strict superset of what architecture1 has, e.g. gex_pool.*)
    # must ALSO match -- not just the architecture1-shared subset checked above.
    arch3_only = set(flat_params["architecture3"]) & set(flat_params["architecture4"]) - set(reference_params)
    assert len(arch3_only) > 0  # sanity: gex_pool.* really is in this set
    for param_name in arch3_only:
        assert torch.equal(flat_params["architecture3"][param_name], flat_params["architecture4"][param_name]), (
            f"architecture4.conditioner.{param_name} diverged from architecture3.{param_name}"
        )


def test_synchronize_four_architecture_initialization_requires_exactly_the_four_expected_keys():
    models = _build_all_four()
    del models["architecture4"]
    with pytest.raises(ValueError, match="requires exactly"):
        synchronize_four_architecture_initialization(models)


# ---------------------------------------------------------------------------
# persist_synchronized_initializations / load_synchronized_initialization --
# fixes a real, confirmed gap (3rd Codex re-audit of commit ca7cf53): the
# real launcher spawns separate subprocesses per GPU and never calls the
# in-process synchronizer, so in-process synchronization alone can't reach
# the actual training jobs. These give the synchronized weights a durable,
# cross-process form on disk.
# ---------------------------------------------------------------------------
def test_persist_synchronized_initializations_writes_weights_and_a_manifest(tmp_path):
    reference = torch.nn.Linear(4, 4)
    other = torch.nn.Linear(4, 4)
    synchronize_shared_initialization({"reference": reference, "other": other}, reference="reference")

    manifest = persist_synchronized_initializations({"reference": reference, "other": other}, tmp_path)

    assert (tmp_path / "reference" / "trainable_weights.pt").is_file()
    assert (tmp_path / "other" / "trainable_weights.pt").is_file()
    assert json.loads((tmp_path / "initialization_manifest.json").read_text()) == manifest
    assert set(manifest["architectures"]) == {"reference", "other"}
    assert manifest["architectures"]["reference"]["n_parameters"] > 0


def test_persist_synchronized_initializations_shared_hash_groups_reflect_real_synchronization(tmp_path):
    reference = torch.nn.Linear(4, 4)
    unsynchronized = torch.nn.Linear(4, 4)
    synchronized = torch.nn.Linear(4, 4)
    synchronize_shared_initialization({"reference": reference, "synchronized": synchronized}, reference="reference")

    manifest = persist_synchronized_initializations(
        {"reference": reference, "unsynchronized": unsynchronized, "synchronized": synchronized}, tmp_path,
    )
    groups = manifest["shared_hash_groups"]
    weight_group = next(g for g in groups.values() if any(loc.endswith(".weight") for loc in g))
    assert "reference.weight" in weight_group
    assert "synchronized.weight" in weight_group
    assert "unsynchronized.weight" not in weight_group


def test_persist_and_load_synchronized_initialization_round_trips(tmp_path):
    trained_looking = torch.nn.Linear(5, 5)
    with torch.no_grad():
        trained_looking.weight.fill_(3.5)
    persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)

    fresh = torch.nn.Linear(5, 5)  # different random init
    assert not torch.equal(fresh.weight, trained_looking.weight)
    load_synchronized_initialization(fresh, tmp_path / "architecture1")
    assert torch.equal(fresh.weight, trained_looking.weight)


def test_persist_synchronized_initializations_on_all_four_real_architectures(tmp_path):
    """End-to-end: synchronize, persist, and verify the manifest's
    shared_hash_groups actually contain cross-architecture entries for a
    genuinely shared module (spot_token), including the previously-
    unreachable Architecture 3/4 conditioner case."""
    models = _build_all_four()
    synchronize_four_architecture_initialization(models)
    manifest = persist_synchronized_initializations(models, tmp_path)

    groups = manifest["shared_hash_groups"]
    spot_token_group = next(
        g for g in groups.values() if any("architecture1.spot_token.image_proj.weight" == loc for loc in g)
    )
    assert "architecture3.spot_token.image_proj.weight" in spot_token_group
    assert "architecture4.conditioner.spot_token.image_proj.weight" in spot_token_group

    gex_pool_group = next(
        g for g in groups.values() if any("architecture3.gex_pool.inducing_queries" == loc for loc in g)
    )
    assert "architecture4.conditioner.gex_pool.inducing_queries" in gex_pool_group
