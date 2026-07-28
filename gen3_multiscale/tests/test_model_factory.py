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
from gen3_multiscale.models.model_factory import build_architecture, resolve_model_kwargs

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
