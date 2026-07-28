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
    build_architecture, load_ad_hoc_checkpoint_unverified, load_synchronized_initialization,
    persist_four_architecture_initializations, persist_synchronized_initializations, resolve_model_kwargs,
    synchronize_four_architecture_initialization, synchronize_shared_initialization,
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
    assert synchronized["other"] == {"weight": "reference.weight", "bias": "reference.bias"}


def test_synchronize_shared_initialization_also_synchronizes_buffers():
    """Regression test for a real, confirmed gap (4th Codex re-audit of
    commit 0fd46e5): an earlier version only iterated named_parameters(),
    silently leaving buffers (e.g. GeneValueTransportHead's
    target_gene_scale, which affects predictions) unsynchronized."""
    class WithBuffer(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("scale", torch.full((4,), value))

    reference = WithBuffer(1.0)
    other = WithBuffer(2.0)
    assert not torch.equal(reference.scale, other.scale)
    synchronized = synchronize_shared_initialization({"reference": reference, "other": other}, reference="reference")
    assert torch.equal(reference.scale, other.scale)
    assert synchronized["other"]["scale"] == "reference.scale"


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
    assert synchronized["wrapped"] == {"conditioner.weight": "reference.weight", "conditioner.bias": "reference.bias"}


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


def test_synchronize_four_architecture_initialization_also_synchronizes_target_gene_scale_buffer():
    """The real-architecture-level version of the buffer-sync regression
    test above: GeneValueTransportHead.target_gene_scale is a buffer, not
    a parameter, and affects predictions -- deliberately perturbed here
    (defaults to all-ones for every architecture otherwise, which would
    make a before/after check meaningless) to prove the two-hop
    synchronizer reaches it too."""
    models = _build_all_four()
    with torch.no_grad():
        models["architecture3"].transport_head.target_gene_scale.fill_(7.0)
    assert not torch.equal(
        models["architecture1"].transport_head.target_gene_scale, models["architecture3"].transport_head.target_gene_scale,
    )
    synchronize_four_architecture_initialization(models)
    assert torch.equal(
        models["architecture1"].transport_head.target_gene_scale, models["architecture3"].transport_head.target_gene_scale,
    )
    assert torch.equal(
        models["architecture1"].transport_head.target_gene_scale,
        models["architecture4"].conditioner.transport_head.target_gene_scale,
    )


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


def test_persist_synchronized_initializations_records_the_synchronizers_own_provenance_not_inferred_hashes(tmp_path):
    """Regression test for a real, confirmed gap (4th Codex re-audit of
    commit 0fd46e5): an earlier version inferred "sharing" from raw hash
    COLLISIONS across all parameters, which could mistake two unrelated,
    coincidentally-identical tensors (e.g. two independently zero-
    initialized biases) for a real shared-parameter relationship. Two
    UNSYNCHRONIZED zero-initialized biases are used deliberately here --
    they WOULD collide under the old hash-grouping approach but must NOT
    appear in shared_parameter_mapping, since no synchronization actually
    related them."""
    reference = torch.nn.Linear(4, 4, bias=True)
    unsynchronized = torch.nn.Linear(4, 4, bias=True)
    synchronized = torch.nn.Linear(4, 4, bias=True)
    with torch.no_grad():
        reference.bias.zero_()
        unsynchronized.bias.zero_()  # coincidentally identical to reference.bias -- but never synchronized
    sync_result = synchronize_shared_initialization(
        {"reference": reference, "synchronized": synchronized}, reference="reference",
    )

    manifest = persist_synchronized_initializations(
        {"reference": reference, "unsynchronized": unsynchronized, "synchronized": synchronized}, tmp_path,
        synchronized=sync_result,
    )
    mapping = manifest["shared_parameter_mapping"]
    assert mapping["synchronized"]["weight"] == "reference.weight"
    assert mapping["synchronized"]["bias"] == "reference.bias"
    assert "unsynchronized" not in mapping  # never passed to the synchronizer -- no ground-truth relationship exists


def test_persist_synchronized_initializations_records_gene_basis_gene_names_hash_for_architecture_4(tmp_path):
    """Field renamed from gene_basis_hash (5th Codex re-audit of commit
    c02a5d1: "is actually only gene_names_hash... does not describe how
    the basis was fitted") -- the name now says precisely what it is."""
    basis, gene_names = _gene_basis(n_genes=6)
    model4 = build_architecture(_load("architecture4"), n_genes=6, gex_feature_dim=4, gene_basis=basis, gene_names=gene_names)
    manifest = persist_synchronized_initializations({"architecture4": model4}, tmp_path)
    assert manifest["architectures"]["architecture4"]["gene_basis_gene_names_hash"] == basis.gene_names_hash


def test_persist_and_load_synchronized_initialization_round_trips(tmp_path):
    trained_looking = torch.nn.Linear(5, 5)
    with torch.no_grad():
        trained_looking.weight.fill_(3.5)
    manifest = persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)

    fresh = torch.nn.Linear(5, 5)  # different random init
    assert not torch.equal(fresh.weight, trained_looking.weight)
    load_synchronized_initialization(fresh, tmp_path / "architecture1", manifest=manifest, architecture_name="architecture1")
    assert torch.equal(fresh.weight, trained_looking.weight)


def test_load_synchronized_initialization_requires_a_manifest_and_architecture_name():
    """Regression test for a real, confirmed gap (5th Codex re-audit of
    commit c02a5d1): manifest/architecture_name used to be optional,
    which is not what "fail-closed" means -- a caller could accidentally
    skip verification just by forgetting to pass them. Now a TypeError
    (missing required positional arguments), not a silent unverified
    load."""
    import inspect
    params = inspect.signature(load_synchronized_initialization).parameters
    assert params["manifest"].default is inspect.Parameter.empty
    assert params["architecture_name"].default is inspect.Parameter.empty


def test_load_ad_hoc_checkpoint_unverified_is_the_only_way_to_skip_verification(tmp_path):
    """The old unverified behavior still exists, but only under a
    separately named function that cannot be reached by accident."""
    trained_looking = torch.nn.Linear(5, 5)
    persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)
    fresh = torch.nn.Linear(5, 5)
    load_ad_hoc_checkpoint_unverified(fresh, tmp_path / "architecture1")
    assert torch.equal(fresh.weight, trained_looking.weight)


def test_load_synchronized_initialization_rejects_a_corrupted_checkpoint_file(tmp_path):
    """Regression test for a real, confirmed gap (4th Codex re-audit):
    load_synchronized_initialization used to ignore the manifest
    entirely, so a modified/corrupted checkpoint file would be loaded
    silently."""
    trained_looking = torch.nn.Linear(5, 5)
    manifest = persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)

    weights_path = tmp_path / "architecture1" / "trainable_weights.pt"
    weights_path.write_bytes(weights_path.read_bytes() + b"\x00")  # corrupt it

    fresh = torch.nn.Linear(5, 5)
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        load_synchronized_initialization(fresh, tmp_path / "architecture1", manifest=manifest, architecture_name="architecture1")


def test_load_synchronized_initialization_rejects_a_checkpoint_copied_to_the_wrong_architecture(tmp_path):
    """The other half of the "corrupted or copied from a different
    architecture's directory" case -- a real, plausible operational
    mistake (copying architecture1's checkpoint into architecture2's
    directory) must be caught, not silently loaded as if it were correct."""
    model1 = torch.nn.Linear(5, 5)
    model2 = torch.nn.Linear(5, 5)
    with torch.no_grad():
        model2.weight.fill_(9.0)  # deliberately different from model1
    manifest = persist_synchronized_initializations({"architecture1": model1, "architecture2": model2}, tmp_path)

    # Simulate the mistake: architecture2's directory actually holds architecture1's file.
    import shutil
    shutil.copy(tmp_path / "architecture1" / "trainable_weights.pt", tmp_path / "architecture2" / "trainable_weights.pt")

    fresh = torch.nn.Linear(5, 5)
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        load_synchronized_initialization(fresh, tmp_path / "architecture2", manifest=manifest, architecture_name="architecture2")


def test_load_synchronized_initialization_rejects_a_manifest_claiming_the_wrong_tensor_shape(tmp_path):
    """Regression test for a real, confirmed gap (6th Codex re-audit of
    commit 06f5cce): `_tensor_hash` hashes raw tensor bytes only
    (`tensor.numpy().tobytes()`), which does not encode shape -- e.g. a
    [2,3] and a [3,2] all-zeros tensor hash identically. shape/dtype were
    already recorded in the manifest by persist_synchronized_initializations
    but never checked on load. A manifest entry claiming the wrong shape
    for a tensor (however it arose) must now be caught explicitly, not
    silently accepted just because the byte hash happens to match."""
    trained_looking = torch.nn.Linear(5, 5)
    manifest = persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)
    manifest["architectures"]["architecture1"]["tensor_hashes"]["weight"]["shape"] = [25, 1]

    fresh = torch.nn.Linear(5, 5)
    with pytest.raises(ValueError, match="has shape"):
        load_synchronized_initialization(fresh, tmp_path / "architecture1", manifest=manifest, architecture_name="architecture1")


def test_load_synchronized_initialization_rejects_a_manifest_claiming_the_wrong_tensor_dtype(tmp_path):
    trained_looking = torch.nn.Linear(5, 5)
    manifest = persist_synchronized_initializations({"architecture1": trained_looking}, tmp_path)
    manifest["architectures"]["architecture1"]["tensor_hashes"]["weight"]["dtype"] = "torch.float64"

    fresh = torch.nn.Linear(5, 5)
    with pytest.raises(ValueError, match="has dtype"):
        load_synchronized_initialization(fresh, tmp_path / "architecture1", manifest=manifest, architecture_name="architecture1")


def test_persist_synchronized_initializations_on_all_four_real_architectures(tmp_path):
    """End-to-end: synchronize, persist (passing the synchronizer's own
    provenance), and verify the manifest's shared_parameter_mapping
    records the previously-unreachable Architecture 3/4 conditioner case
    (gex_pool) alongside an Architecture-1-shared module (spot_token).
    Also verifies every persisted architecture round-trips through
    load_synchronized_initialization's full fail-closed verification."""
    models = _build_all_four()
    sync_result = synchronize_four_architecture_initialization(models)
    manifest = persist_synchronized_initializations(models, tmp_path, synchronized=sync_result)

    mapping = manifest["shared_parameter_mapping"]
    assert mapping["architecture3"]["spot_token.image_proj.weight"] == "architecture1.spot_token.image_proj.weight"
    assert (
        mapping["architecture4"]["conditioner.spot_token.image_proj.weight"]
        == "architecture3.spot_token.image_proj.weight"
    )
    assert (
        mapping["architecture4"]["conditioner.gex_pool.inducing_queries"]
        == "architecture3.gex_pool.inducing_queries"
    )

    for name in models:
        fresh_kwargs = dict(n_genes=6, gex_feature_dim=4)
        if name == "architecture4":
            basis, gene_names = _gene_basis(n_genes=6)
            fresh_kwargs.update(gene_basis=basis, gene_names=gene_names)
        fresh = build_architecture(_load(name), **fresh_kwargs)
        load_synchronized_initialization(fresh, tmp_path / name, manifest=manifest, architecture_name=name)  # must not raise


# ---------------------------------------------------------------------------
# persist_four_architecture_initializations -- the strict, purpose-built
# entry point (5th Codex re-audit of commit c02a5d1): "For a function with
# that name, it should require the exact four architectures, require the
# two-hop mapping, verify every expected shared tensor is equal, and then
# persist."
# ---------------------------------------------------------------------------
def test_persist_four_architecture_initializations_requires_exactly_the_four_expected_keys(tmp_path):
    models = _build_all_four()
    del models["architecture4"]
    with pytest.raises(ValueError, match="requires exactly"):
        persist_four_architecture_initializations(models, tmp_path)


def test_persist_four_architecture_initializations_synchronizes_persists_and_round_trips(tmp_path):
    models = _build_all_four()
    manifest = persist_four_architecture_initializations(models, tmp_path)

    mapping = manifest["shared_parameter_mapping"]
    assert mapping["architecture3"]["spot_token.image_proj.weight"] == "architecture1.spot_token.image_proj.weight"
    assert (
        mapping["architecture4"]["conditioner.gex_pool.inducing_queries"]
        == "architecture3.gex_pool.inducing_queries"
    )

    for name in models:
        fresh_kwargs = dict(n_genes=6, gex_feature_dim=4)
        if name == "architecture4":
            basis, gene_names = _gene_basis(n_genes=6)
            fresh_kwargs.update(gene_basis=basis, gene_names=gene_names)
        fresh = build_architecture(_load(name), **fresh_kwargs)
        load_synchronized_initialization(fresh, tmp_path / name, manifest=manifest, architecture_name=name)


def test_persist_four_architecture_initializations_refuses_to_persist_an_inconsistent_state(tmp_path, monkeypatch):
    """Verifies the extra paranoia check actually does something: if
    synchronization somehow left a "shared" tensor genuinely unequal
    (simulated here via a monkeypatched synchronizer that lies about what
    it did), persistence must refuse rather than write a manifest
    claiming a sharing relationship that isn't true."""
    import gen3_multiscale.models.model_factory as model_factory_module

    models = _build_all_four()
    with torch.no_grad():
        # Force a genuine, deliberate mismatch -- otherwise architecture1
        # and architecture2's gene_head_logits already coincide by
        # construction (both built with the same init_seed), which would
        # make this test pass for the wrong reason.
        models["architecture2"].transport_head.gene_head_logits.fill_(123.0)

    def _lying_synchronizer(_models):
        return {"architecture2": {"transport_head.gene_head_logits": "architecture1.transport_head.gene_head_logits"}}

    monkeypatch.setattr(model_factory_module, "synchronize_four_architecture_initialization", _lying_synchronizer)
    with pytest.raises(ValueError, match="does not match"):
        model_factory_module.persist_four_architecture_initializations(models, tmp_path)
