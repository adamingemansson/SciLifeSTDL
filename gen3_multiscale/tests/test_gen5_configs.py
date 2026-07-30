"""All four arms' real committed Gen5 configs must pass the static
preflight audit, share one common velocity-network size (fairness,
GEN5_CONTRACT.md section 6), and construct a real model at their ACTUAL
dims."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from gen3_multiscale.gen4.uni2_global_pool import MaskAwareCoordinateAttentionPool
from gen3_multiscale.gen5.model_factory import build_gen5_model
from gen3_multiscale.gen5.preflight import static_audit_gen5_config
from gen3_multiscale.tests._gen5_fixtures import tiny_autoencoder

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "gen5"
_ARMS = ["gen5a", "gen5b", "gen5c", "gen5d"]
_SHARED_VELOCITY_KEYS = ("hidden_dim", "n_heads", "n_flow_blocks", "dense_threshold", "sparse_k", "chunk_size", "n_flow_samples", "n_ode_steps", "latent_dim")


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
def test_config_passes_static_audit(arm):
    config = _load(f"{arm}.yaml")
    report = static_audit_gen5_config(config)
    assert report["arm"] == arm and report["kind"] == "latent_flow"
    assert not report["ready_for_real_training"]


def test_all_four_arms_share_one_common_velocity_network_size():
    configs = {arm: _load(f"{arm}.yaml") for arm in _ARMS}
    reference = configs["gen5a"]["model"]["params"]
    for arm, config in configs.items():
        params = config["model"]["params"]
        for key in _SHARED_VELOCITY_KEYS:
            assert params[key] == reference[key], f"{arm}.{key} diverges from gen5a: {params[key]!r} != {reference[key]!r}"


@pytest.mark.parametrize("arm", _ARMS)
def test_config_constructs_a_real_model_at_its_actual_dims(arm):
    config = _load(f"{arm}.yaml")
    params = config["model"]["params"]
    n_genes = 6
    gene_names = [f"g{i}" for i in range(n_genes)]
    autoencoder = tiny_autoencoder(n_genes=n_genes, latent_dim=params["latent_dim"])
    gex_dim, image_dim = params["gex_feature_dim"], params["image_feature_dim"]
    context_dim = params.get("gex_context_embedding_dim")

    kwargs = dict(
        config=config, n_genes=n_genes, gene_names=gene_names, gex_feature_dim=gex_dim,
        image_feature_dim=image_dim, autoencoder=autoencoder,
    )
    if context_dim:
        kwargs["gex_context_embedding_dim"] = context_dim
    if arm == "gen5b":
        kwargs["slide_encoder"] = _StubSlideEncoder(image_dim, params["global_slide_dim"])
        kwargs["gigapath_checkpoint_sha256"] = "deadbeef" * 8
    if arm in {"gen5a", "gen5c"}:
        kwargs["uni2_global_pool"] = MaskAwareCoordinateAttentionPool(
            tile_feature_dim=image_dim, output_dim=params["global_slide_dim"], hidden_dim=32, n_heads=2,
        )
    model = build_gen5_model(**kwargs)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_unknown_field_in_params_fails_closed():
    config = _load("gen5a.yaml")
    config["model"]["params"]["totally_bogus_field"] = 1
    autoencoder = tiny_autoencoder(n_genes=6, latent_dim=256)
    with pytest.raises(ValueError, match="totally_bogus_field"):
        build_gen5_model(
            config=config, n_genes=6, gene_names=[f"g{i}" for i in range(6)], gex_feature_dim=4,
            image_feature_dim=8, autoencoder=autoencoder,
        )
