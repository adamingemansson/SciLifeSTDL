"""All four arms' real committed configs (configs/gen4/*.yaml) must pass
the static preflight audit and construct a real model at their ACTUAL
(non-test-shrunk) dims -- the "all four real configs construct
successfully" requirement."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from gen3_multiscale.gen4.model_factory import build_gen4_conditioner, build_gen4_flow
from gen3_multiscale.gen4.preflight import static_audit_gen4_config
from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.tests._gen4_fixtures import Gen4STPathStub, StubUNI2Encoder

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "gen4"
_ARMS = ["gen4a", "gen4b", "gen4c", "gen4d"]


class _StubSlideEncoder(torch.nn.Module):
    def __init__(self, image_dim, out_dim):
        super().__init__()
        self.checkpoint_sha256 = "deadbeef" * 8
        self.proj = torch.nn.Linear(image_dim, out_dim)

    def forward(self, tile_features, tile_coords, cache_namespace):
        return self.proj(tile_features.mean(dim=0, keepdim=True)).squeeze(0)


def _load(name: str) -> dict:
    return yaml.safe_load((_CONFIG_DIR / name).read_text())


@pytest.mark.parametrize("arm", _ARMS)
def test_conditioner_config_passes_static_audit(arm):
    config = _load(f"{arm}_conditioner.yaml")
    report = static_audit_gen4_config(config)
    assert report["arm"] == arm
    assert report["kind"] == "conditioner"
    assert not report["ready_for_real_training"]  # every fingerprint is still null in the committed config


@pytest.mark.parametrize("arm", _ARMS)
def test_flow_config_passes_static_audit(arm):
    config = _load(f"{arm}_flow.yaml")
    report = static_audit_gen4_config(config)
    assert report["arm"] == arm
    assert report["kind"] == "flow"


@pytest.mark.parametrize("arm", _ARMS)
def test_conditioner_config_constructs_a_real_model_at_its_actual_dims(arm):
    config = _load(f"{arm}_conditioner.yaml")
    params = config["model"]["params"]
    n_genes, gex_dim, image_dim = 6, params["gex_feature_dim"], params["image_feature_dim"]
    context_dim = params.get("gex_context_embedding_dim")

    kwargs = dict(config=config, n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim)
    if context_dim:
        kwargs["gex_context_embedding_dim"] = context_dim
    if arm == "gen4b":
        kwargs["slide_encoder"] = _StubSlideEncoder(image_dim, params["global_slide_dim"])
        kwargs["gigapath_checkpoint_sha256"] = "deadbeef" * 8
    if arm in {"gen4a", "gen4c"}:
        kwargs["uni2_global_pool"] = MaskAwareCoordinateAttentionPool(
            tile_feature_dim=image_dim, output_dim=params["global_slide_dim"], hidden_dim=32, n_heads=2,
        )
    if arm == "gen4d":
        kwargs["stpath_encoder"] = Gen4STPathStub(n_genes=n_genes, hidden_dim=image_dim)
    model = build_gen4_conditioner(**kwargs)
    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("arm", _ARMS)
def test_flow_config_constructs_a_real_model_at_its_actual_dims(arm):
    config = _load(f"{arm}_flow.yaml")
    params = config["model"]["params"]
    n_genes, gex_dim, image_dim = 6, params["gex_feature_dim"], params["image_feature_dim"]
    context_dim = params.get("gex_context_embedding_dim")

    gene_names = [f"g{i}" for i in range(n_genes)]
    import numpy as np
    residuals = np.random.default_rng(0).normal(size=(20, n_genes))
    gene_basis = fit_gene_residual_basis(residuals, gene_names, rank=min(params["gene_basis_rank"], n_genes))

    kwargs = dict(
        config=config, n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim,
        gene_basis=gene_basis, gene_names=gene_names,
    )
    if context_dim:
        kwargs["gex_context_embedding_dim"] = context_dim
    if arm == "gen4b":
        kwargs["slide_encoder"] = _StubSlideEncoder(image_dim, params["global_slide_dim"])
        kwargs["gigapath_checkpoint_sha256"] = "deadbeef" * 8
    if arm in {"gen4a", "gen4c"}:
        kwargs["uni2_global_pool"] = MaskAwareCoordinateAttentionPool(
            tile_feature_dim=image_dim, output_dim=params["global_slide_dim"], hidden_dim=32, n_heads=2,
        )
    if arm == "gen4d":
        kwargs["stpath_encoder"] = Gen4STPathStub(n_genes=n_genes, hidden_dim=image_dim)
    model = build_gen4_flow(**kwargs)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_unknown_field_in_params_fails_closed():
    config = _load("gen4a_conditioner.yaml")
    config["model"]["params"]["totally_bogus_field"] = 1
    with pytest.raises(ValueError, match="totally_bogus_field"):
        build_gen4_conditioner(config=config, n_genes=6, gex_feature_dim=4, image_feature_dim=8)
