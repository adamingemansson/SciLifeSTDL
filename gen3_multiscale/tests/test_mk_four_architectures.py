import copy

import numpy as np
import pytest
import torch
import yaml

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAE,
    DeterministicSpatialPredictor,
    FullImageExpressionInputs,
    LocalImageConditioner,
)
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.training.train_conditional_wae import _build_model
from gen3_multiscale.scripts import prepare_mk_architecture_suite as suite_module


def _inputs(n=8, image_dim=12):
    rng = np.random.default_rng(4)
    return FullImageExpressionInputs(
        sample_id="slide",
        image_features=rng.normal(size=(n, image_dim)).astype(np.float32),
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        image_available=np.ones(n, dtype=bool),
        query_mask=np.array([True, False, True, False, True, False, True, False]),
    )


def test_local_conditioner_is_structurally_coordinate_and_neighbour_invariant():
    inputs = _inputs()
    changed = FullImageExpressionInputs(
        **{
            **inputs.__dict__,
            "coords": np.full_like(inputs.coords, 999.0),
            "neighbor_indices": np.tile(np.arange(3), (8, 1)).astype(np.int64),
            "neighbor_mask": np.ones((8, 3), dtype=bool),
        }
    )
    model = LocalImageConditioner(6, image_feature_dim=12, hidden_dim=16, dropout=0.0).eval()
    torch.testing.assert_close(model(inputs), model(changed))
    assert not any("coord" in name or "attention" in name for name, _ in model.named_modules())


def test_deterministic_spatial_predictor_has_no_wae_components_and_trains_supervised():
    inputs = _inputs()
    conditioner = Architecture1ImageConditioner(
        6, image_feature_dim=12, gex_feature_dim=5, hidden_dim=16,
        n_heads=4, n_blocks=1, dense_threshold=20, sparse_k=3, dropout=0.0,
    )
    model = DeterministicSpatialPredictor(6, conditioner, hidden_dim=20)
    assert not hasattr(model, "expression_encoder")
    assert not hasattr(model, "residual_decoder")
    assert not hasattr(model, "conditional_prior")
    target = torch.randn(4, 6)
    losses = model.compute_generator_losses(inputs, target)
    assert losses["prior_loss"].item() == 0.0
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    prediction = model.sample_predictive_distribution(inputs, n_samples=3)
    torch.testing.assert_close(prediction["predictive_mean"], prediction["point_prediction"])
    assert torch.count_nonzero(prediction["predictive_std"]) == 0


@pytest.mark.parametrize("conditioner_cls", [LocalImageConditioner, Architecture1ImageConditioner])
def test_conditional_wae_prior_is_image_dependent_and_receives_gradients(conditioner_cls):
    if conditioner_cls is LocalImageConditioner:
        conditioner = conditioner_cls(6, image_feature_dim=12, hidden_dim=16, dropout=0.0)
    else:
        conditioner = conditioner_cls(
            6, image_feature_dim=12, gex_feature_dim=5, hidden_dim=16,
            n_heads=4, n_blocks=1, dense_threshold=20, sparse_k=3, dropout=0.0,
        )
    model = ConditionalWAE(
        6, conditioner, regularizer="mmd", latent_dim=4,
        autoencoder_hidden_dim=20, discriminator_hidden_dim=8,
        encoder_conditioning="film", prior_mode="conditional",
        conditional_prior_hidden_dim=10,
    )
    inputs = _inputs()
    losses = model.compute_generator_losses(
        inputs, torch.randn(4, 6), generator=torch.Generator().manual_seed(3),
    )
    assert "conditional_alignment_loss" in losses
    assert "prior_anchor_loss" in losses
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.conditional_prior.parameters())
    context = model.image_conditioner(inputs)
    z = model.sample_inference_latent(context, generator=torch.Generator().manual_seed(5))
    assert z.shape == (4, 4)
    with pytest.raises(ValueError, match="incompatible"):
        model.sample_inference_latent(context, z_mean=torch.zeros(4))


def _contract_config(arm, *, conditioner_mode, prior_mode, deterministic_only):
    regularizer = "none" if deterministic_only else "mmd"
    params = {
        "image_feature_dim": 1536, "hidden_dim": 32, "gex_feature_dim": 16,
        "autoencoder_hidden_dim": 24, "n_heads": 4, "n_blocks": 1,
        "dense_threshold": 20, "sparse_k": 3, "latent_dim": 4,
        "conditioner_mode": conditioner_mode, "prior_mode": prior_mode,
        "deterministic_only": deterministic_only, "encoder_conditioning": "film",
        "film_layers": ["first", "second"], "n_refinement_steps": 0,
    }
    if deterministic_only:
        params["encoder_conditioning"] = "none"
        params.pop("film_layers")
    return {
        "model": {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st",
            "regularizer": regularizer, "include_observed_gex": False,
            "image_mode": "full_visible", "params": params,
        },
        "data": {
            "gen3_manifest_path": "/manifest.json", "image_encoder": "uni2",
            "uni2_pinned_revision": "deadbeef",
        },
        "training": {"checkpoint_dir": "/checkpoint"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }


def test_four_arm_contracts_fail_closed_on_claimed_architecture():
    designs = {
        "mk_local_wae_mmd": ("local", "standard", False),
        "mk_spatial_deterministic": ("spatial", "none", True),
        "mk_local_conditional_wae_mmd": ("local", "conditional", False),
        "mk_spatial_conditional_wae_mmd": ("spatial", "conditional", False),
    }
    for arm, design in designs.items():
        config = _contract_config(
            arm, conditioner_mode=design[0], prior_mode=design[1],
            deterministic_only=design[2],
        )
        assert static_audit_conditional_wae_config(config)["passed"]
        if design[0] == "local":
            broken = copy.deepcopy(config)
            broken["model"]["params"]["n_refinement_steps"] = 1
            with pytest.raises(ValueError, match="local conditioner"):
                static_audit_conditional_wae_config(broken)


def test_training_factory_constructs_the_exact_four_architectures():
    designs = {
        "mk_local_wae_mmd": (LocalImageConditioner, "standard", True),
        "mk_spatial_deterministic": (Architecture1ImageConditioner, "none", False),
        "mk_local_conditional_wae_mmd": (LocalImageConditioner, "conditional", True),
        "mk_spatial_conditional_wae_mmd": (
            Architecture1ImageConditioner, "conditional", True,
        ),
    }
    for arm, (conditioner_type, prior_mode, has_latent) in designs.items():
        mode = "local" if conditioner_type is LocalImageConditioner else "spatial"
        config = _contract_config(
            arm, conditioner_mode=mode, prior_mode=prior_mode,
            deterministic_only=not has_latent,
        )
        params = config["model"]["params"]
        params.update({
            "image_feature_dim": 12, "gex_feature_dim": 5,
            "hidden_dim": 16, "autoencoder_hidden_dim": 20,
            "discriminator_hidden_dim": 8, "n_inference_samples": 3,
            "conditional_prior_hidden_dim": 10,
        })
        model = _build_model(config, n_genes=6, gene_names=[f"g{i}" for i in range(6)])
        assert isinstance(model.image_conditioner, conditioner_type)
        assert model.has_latent_model is has_latent
        assert model.prior_mode == prior_mode
        if prior_mode == "conditional":
            assert model.conditional_prior is not None
        elif has_latent:
            assert model.conditional_prior is None
        else:
            assert isinstance(model, DeterministicSpatialPredictor)


def test_suite_preparer_writes_four_gpu_mask_safe_configs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "gen3_multiscale" / "results").mkdir(parents=True)
    raw = source / "data" / "raw" / "hest1k"
    cache = source / "data" / "cache" / "hest1k"
    uni2_cache = cache / "uni2_gen3_spot_cache"
    raw.mkdir(parents=True)
    uni2_cache.mkdir(parents=True)
    comparison = source / "gen3_multiscale" / "results" / "arch1.yaml"
    comparison.write_text(yaml.safe_dump({
        "model": {"architecture": "1", "params": {
            "image_feature_dim": 1536, "hidden_dim": 32, "n_heads": 4,
            "n_blocks": 1, "dense_threshold": 20, "sparse_k": 3,
        }},
        "data": {
            "hest_data_dir": str(raw), "hest_cache_dir": str(cache),
            "gex_feature_dim": 16,
        },
        "masking": {"strata": {"small": {"kind": "round"}}},
        "training": {"seed": 7},
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    panels_path = tmp_path / "panels.json"
    panels_path.write_text("{}")
    manifest = {
        "gene_panel": ["g0", "g1"],
        "validation_sample_ids": ["v0"],
        "samples": {"v0": {"organ": "kidney"}},
    }
    monkeypatch.setattr(suite_module, "load_dataset_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        suite_module, "load_train_derived_gene_panels",
        lambda _path, _manifest: {"panels": {
            "train_log1p_variance_top50": ["g0"],
            "train_log1p_variance_top200": ["g0", "g1"],
        }},
    )
    root = tmp_path / "suite"
    plan = suite_module.prepare_mk_architecture_suite(
        comparison_config=str(comparison), manifest=str(manifest_path),
        train_gene_panels=str(panels_path), output_root=str(root),
        uni2_pinned_revision="pinned-revision",
        uni2_spot_feature_cache_dir=str(uni2_cache),
        gpus=(0, 2, 3, 5),
    )
    assert tuple(plan["arm_order"]) == suite_module.ARM_ORDER
    assert [plan["arms"][arm]["gpu"] for arm in suite_module.ARM_ORDER] == [0, 2, 3, 5]
    for arm in suite_module.ARM_ORDER:
        config = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
        # The launcher masks the physical GPU, so the process must use its
        # only visible logical device rather than cuda:<physical id>.
        assert config["training"]["device"] == "cuda"
        assert config["data"]["image_encoder"] == "uni2"
        assert config["data"]["retain_patches_in_memory"] is False
        assert config["model"]["params"]["n_refinement_steps"] == 0
    assert (root.parent / "LATEST_MK_FOUR_ARCHITECTURE_SUITE_ROOT.txt").read_text().strip() == str(root)
