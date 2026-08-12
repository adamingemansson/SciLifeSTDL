import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.data import build_conditional_wae_example
from gen3_multiscale.conditional_wae.inputs import FullImageExpressionInputs
from gen3_multiscale.conditional_wae.model import (
    Architecture1ImageConditioner,
    DeterministicSpatialPredictor,
    LocalImageConditioner,
)
from gen3_multiscale.conditional_wae.structured_field import (
    fit_centered_organ_balanced_gene_structure,
    load_centered_gene_structure_artifact,
    save_centered_gene_structure_artifact,
)
from gen3_multiscale.training.train_conditional_wae import _build_model
from gen3_multiscale.scripts import prepare_mk_structured_field_suite as suite_module
from gen3_multiscale.scripts.fit_mk_centered_gene_structure import _select_spot_rows
from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide


def _fit_artifact():
    rng = np.random.default_rng(9)
    expression = {
        "a": rng.normal(size=(12, 6)).astype(np.float32) + 20,
        "b": rng.normal(size=(8, 6)).astype(np.float32) - 30,
        "c": rng.normal(size=(10, 6)).astype(np.float32) + 100,
    }
    return fit_centered_organ_balanced_gene_structure(
        expression, ["a", "b", "c"],
        {"a": "kidney", "b": "kidney", "c": "lung"},
        [f"g{i}" for i in range(6)], rank=3, seed=2,
    )


def test_centered_structure_spot_selection_is_bounded_and_deterministic():
    expression = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    first, first_indices = _select_spot_rows(
        expression, "slide-a", max_spots_per_slide=12, seed=7,
    )
    second, second_indices = _select_spot_rows(
        expression, "slide-a", max_spots_per_slide=12, seed=7,
    )
    assert first.shape == (12, 4)
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first, second)
    assert np.array_equal(first, expression[first_indices])
    assert np.all(np.diff(first_indices) > 0)


def test_centered_structure_spot_selection_copies_small_slides():
    expression = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    selected, indices = _select_spot_rows(
        expression, "slide-a", max_spots_per_slide=12, seed=7,
    )
    assert np.array_equal(indices, np.arange(5))
    assert np.array_equal(selected, expression)
    selected[0, 0] = -1
    assert expression[0, 0] == 0


def _inputs(n=8):
    rng = np.random.default_rng(4)
    return FullImageExpressionInputs(
        sample_id="slide",
        image_features=rng.normal(size=(n, 12)).astype(np.float32),
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        image_available=np.ones(n, dtype=bool),
        query_mask=np.array([True, False, True, False, True, False, True, False]),
    )


def _local():
    return LocalImageConditioner(6, image_feature_dim=12, hidden_dim=16, dropout=0.0)


def _spatial():
    return Architecture1ImageConditioner(
        6, image_feature_dim=12, gex_feature_dim=5, hidden_dim=16,
        n_heads=4, n_blocks=1, dense_threshold=20, sparse_k=3, dropout=0.0,
    )


def test_centering_makes_program_subspace_invariant_to_slide_offsets(tmp_path):
    rng = np.random.default_rng(1)
    base = {
        "a": rng.normal(size=(12, 6)).astype(np.float32),
        "b": rng.normal(size=(8, 6)).astype(np.float32),
        "c": rng.normal(size=(10, 6)).astype(np.float32),
    }
    shifted = {
        key: matrix + np.float32(offset)
        for (key, matrix), offset in zip(base.items(), (100, -50, 999))
    }
    organs = {"a": "kidney", "b": "kidney", "c": "lung"}
    names = [f"g{i}" for i in range(6)]
    first = fit_centered_organ_balanced_gene_structure(
        base, list(base), organs, names, rank=3, seed=7,
    )
    second = fit_centered_organ_balanced_gene_structure(
        shifted, list(base), organs, names, rank=3, seed=7,
    )
    # Compare projection matrices, which are invariant to arbitrary SVD signs.
    torch.testing.assert_close(
        first.basis.basis.T @ first.basis.basis,
        second.basis.basis.T @ second.basis.basis,
        atol=2e-5, rtol=2e-5,
    )
    torch.testing.assert_close(first.per_gene_scale, second.per_gene_scale, atol=2e-5, rtol=2e-5)
    assert first.metadata["centering"] == "per_slide_gene_mean"
    assert first.metadata["weighting"] == "equal_organ_equal_slide"

    path = save_centered_gene_structure_artifact(first, tmp_path / "structure.pt")
    loaded = load_centered_gene_structure_artifact(path, names)
    torch.testing.assert_close(loaded.basis.basis, first.basis.basis)
    with pytest.raises(ValueError, match="gene"):
        load_centered_gene_structure_artifact(path, names[::-1])


def test_within_between_gradient_and_combined_have_exact_components():
    artifact = _fit_artifact()
    inputs = _inputs()
    target = torch.randn(4, 6)
    arms = {
        "within": DeterministicSpatialPredictor(
            6, _local(), hidden_dim=20, gene_structure_artifact=artifact,
            per_gene_scale=artifact.per_gene_scale,
        ),
        "between": DeterministicSpatialPredictor(
            6, _spatial(), hidden_dim=20, n_refinement_steps=1,
            per_gene_scale=artifact.per_gene_scale,
        ),
        "gradient": DeterministicSpatialPredictor(
            6, _local(), hidden_dim=20, per_gene_scale=artifact.per_gene_scale,
            local_gradient_weight=0.025, wide_gradient_weight=0.025,
            local_gradient_k=2, wide_gradient_k=3,
        ),
        "combined": DeterministicSpatialPredictor(
            6, _spatial(), hidden_dim=20, gene_structure_artifact=artifact,
            n_refinement_steps=1, per_gene_scale=artifact.per_gene_scale,
            local_gradient_weight=0.025, wide_gradient_weight=0.025,
            local_gradient_k=2, wide_gradient_k=3,
        ),
    }
    assert arms["within"].coexpression_refinement is not None
    assert arms["within"].spatial_refiner is None
    assert arms["between"].coexpression_refinement is None
    assert arms["between"].spatial_refiner is not None
    assert arms["gradient"].coexpression_refinement is None
    assert arms["gradient"].spatial_refiner is None
    assert arms["combined"].coexpression_refinement is not None
    assert arms["combined"].spatial_refiner is not None
    for name, model in arms.items():
        losses = model.compute_generator_losses(inputs, target)
        assert torch.isfinite(losses["total"])
        if name in {"gradient", "combined"}:
            assert losses["local_gradient_loss"] > 0
            assert losses["wide_gradient_loss"] > 0
            assert losses["total"] > losses["reconstruction_loss"]
        else:
            assert losses["local_gradient_loss"] == 0
            assert losses["wide_gradient_loss"] == 0


def _contract(arm, design, family="deterministic"):
    conditioner, structure, refinement, local_gradient, wide_gradient = design
    if family == "deterministic":
        regularizer, prior_mode, deterministic, encoder_conditioning = "none", "none", True, "none"
    elif family == "standard_wae":
        regularizer, prior_mode, deterministic, encoder_conditioning = "mmd", "standard", False, "film"
    elif family == "conditional_wae":
        regularizer, prior_mode, deterministic, encoder_conditioning = "mmd", "conditional", False, "film"
    else:
        raise ValueError(family)
    return {
        "model": {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st",
            "regularizer": regularizer, "include_observed_gex": False,
            "image_mode": "full_visible", "params": {
                "image_feature_dim": 1536, "hidden_dim": 32,
                "gex_feature_dim": 16, "autoencoder_hidden_dim": 24,
                "n_heads": 4, "n_blocks": 1, "dense_threshold": 20,
                "sparse_k": 3, "conditioner_mode": conditioner,
                "prior_mode": prior_mode, "deterministic_only": deterministic,
                "encoder_conditioning": encoder_conditioning,
                "film_layers": ["first", "second"],
                "film_shared_generator": False,
                "latent_dim": 4, "discriminator_hidden_dim": 8,
                "n_inference_samples": 3,
                "conditional_prior_hidden_dim": 8,
                "conditional_prior_context_weight": 1.0,
                "conditional_prior_anchor_weight": 0.1,
                "n_refinement_steps": refinement,
                "refinement_k_neighbors": 6, "refinement_hidden_dim": 16,
                "refinement_gex_feature_dim": 8,
                "use_centered_gene_structure": structure,
                "local_gradient_k": 6, "wide_gradient_k": 18,
            },
        },
        "data": {
            "gen3_manifest_path": "/manifest.json", "image_encoder": "uni2",
            "uni2_pinned_revision": "pinned",
            "centered_gene_structure_path": "/structure.pt",
            "centered_gene_structure_basis_sha256": "basis-hash",
        },
        "training": {"checkpoint_dir": "/checkpoint"},
        "loss": {
            "pcc_weight": 0.1,
            "regularizer_weight": 0.0 if deterministic else 0.1,
            "conditional_mean_weight": 1.0,
            "local_gradient_weight": 0.025 if local_gradient else 0.0,
            "wide_gradient_weight": 0.025 if wide_gradient else 0.0,
        },
    }


def test_structured_field_contracts_fail_closed():
    designs = {
        "mk_field_within": ("local", True, 0, False, False),
        "mk_field_between": ("spatial", False, 1, False, False),
        "mk_field_gradient": ("local", False, 0, True, True),
        "mk_field_combined": ("spatial", True, 1, True, True),
    }
    for arm, design in designs.items():
        config = _contract(arm, design)
        assert static_audit_conditional_wae_config(config)["passed"]
        broken = copy.deepcopy(config)
        broken["data"].pop("centered_gene_structure_path")
        with pytest.raises(ValueError, match="centered_gene_structure_path"):
            static_audit_conditional_wae_config(broken)
        broken = copy.deepcopy(config)
        broken["model"]["params"]["use_centered_gene_structure"] = not design[1]
        with pytest.raises(ValueError, match="immutable contract"):
            static_audit_conditional_wae_config(broken)


def test_training_factory_builds_each_structured_field_arm(tmp_path):
    artifact = _fit_artifact()
    path = save_centered_gene_structure_artifact(artifact, tmp_path / "structure.pt")
    designs = {
        "mk_field_within": ("local", True, 0, False, False),
        "mk_field_between": ("spatial", False, 1, False, False),
        "mk_field_gradient": ("local", False, 0, True, True),
        "mk_field_combined": ("spatial", True, 1, True, True),
    }
    for arm, design in designs.items():
        config = _contract(arm, design)
        config["data"]["centered_gene_structure_path"] = str(path)
        config["data"]["centered_gene_structure_basis_sha256"] = artifact.metadata["basis_sha256"]
        params = config["model"]["params"]
        params.update({
            "image_feature_dim": 12, "hidden_dim": 16,
            "gex_feature_dim": 5, "autoencoder_hidden_dim": 20,
        })
        model = _build_model(config, 6, gene_names=[f"g{i}" for i in range(6)])
        assert model.has_latent_model is False
        assert (model.coexpression_refinement is not None) is design[1]
        assert model.n_refinement_steps == design[2]
        assert (model.local_gradient_weight > 0) is design[3]
        assert (model.wide_gradient_weight > 0) is design[4]


@pytest.mark.parametrize(
    "family,prefix,expected_prior",
    [("standard_wae", "wae", "standard"),
     ("conditional_wae", "cwae", "conditional")],
)
def test_structured_wae_families_use_identical_modules_in_all_paths(
    tmp_path, family, prefix, expected_prior,
):
    artifact = _fit_artifact()
    path = save_centered_gene_structure_artifact(artifact, tmp_path / f"{family}.pt")
    designs = {
        "within": ("local", True, 0, False, False),
        "between": ("spatial", False, 1, False, False),
        "gradient": ("local", False, 0, True, True),
        "combined": ("spatial", True, 1, True, True),
    }
    inputs = _inputs()
    target = torch.randn(4, 6)
    for suffix, design in designs.items():
        config = _contract(f"{prefix}_{suffix}", design, family=family)
        config["data"]["centered_gene_structure_path"] = str(path)
        config["data"]["centered_gene_structure_basis_sha256"] = artifact.metadata["basis_sha256"]
        params = config["model"]["params"]
        params.update({"image_feature_dim": 12, "hidden_dim": 16, "gex_feature_dim": 5})
        assert static_audit_conditional_wae_config(config)["passed"]
        model = _build_model(config, 6, gene_names=[f"g{i}" for i in range(6)])
        assert model.has_latent_model is True
        assert model.prior_mode == expected_prior
        assert (model.centered_gene_structure_refinement is not None) is design[1]
        assert model.n_refinement_steps == design[2]
        losses = model.compute_generator_losses(inputs, target)
        assert torch.isfinite(losses["total"])
        assert bool((losses["local_gradient_loss"] > 0).item()) is design[3]
        assert bool((losses["wide_gradient_loss"] > 0).item()) is design[4]
        context = model.image_conditioner(inputs)
        point = model.predict_point_from_context(context, inputs)
        sampled = model.decode_latent_from_context(torch.zeros(4, 4), context, inputs)
        assert point.shape == sampled.shape == (4, 6)


def test_suite_preparer_writes_four_audited_configs(tmp_path, monkeypatch):
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
        "gene_panel": [f"g{i}" for i in range(6)],
        "train_sample_ids": ["a", "b", "c"],
        "validation_sample_ids": ["v0"],
        "test_sample_ids": [],
        "samples": {
            "a": {"organ": "kidney"}, "b": {"organ": "kidney"},
            "c": {"organ": "lung"}, "v0": {"organ": "kidney"},
        },
    }
    artifact = _fit_artifact()
    structure_path = save_centered_gene_structure_artifact(
        artifact, tmp_path / "structure.pt",
    )
    monkeypatch.setattr(suite_module, "load_dataset_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        suite_module, "load_train_derived_gene_panels",
        lambda _path, _manifest: {"panels": {
            "train_log1p_variance_top50": ["g0"],
            "train_log1p_variance_top200": ["g0", "g1"],
        }},
    )
    root = tmp_path / "suite"
    plan = suite_module.prepare_mk_structured_field_suite(
        comparison_config=str(comparison), manifest=str(manifest_path),
        train_gene_panels=str(panels_path),
        centered_gene_structure=str(structure_path), output_root=str(root),
        uni2_pinned_revision="pinned", uni2_spot_feature_cache_dir=str(uni2_cache),
    )
    assert tuple(plan["arm_order"]) == suite_module.ARM_ORDER
    for arm in suite_module.ARM_ORDER:
        config = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
        assert static_audit_conditional_wae_config(config)["passed"]
        assert config["model"]["regularizer"] == "none"
        assert config["model"]["include_observed_gex"] is False
        assert config["data"]["image_encoder"] == "uni2"
        structured = config["evaluation"]["structured_field_metrics"]
        assert structured["enabled"] is True
        assert structured["all_split_slides"] is True
        assert (structured["local_k"], structured["wide_k"]) == (6, 18)


@pytest.mark.parametrize(
    "family,expected_arms,expected_prior,deterministic",
    [
        ("standard_wae", ("wae_within", "wae_between", "wae_gradient", "wae_combined"),
         "standard", False),
        ("conditional_wae", ("cwae_within", "cwae_between", "cwae_gradient", "cwae_combined"),
         "conditional", False),
    ],
)
def test_suite_preparer_writes_structured_latent_families(
    tmp_path, monkeypatch, family, expected_arms, expected_prior, deterministic,
):
    source = tmp_path / family / "source"
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
        "data": {"hest_data_dir": str(raw), "hest_cache_dir": str(cache),
                 "gex_feature_dim": 16},
        "masking": {"strata": {"small": {"kind": "round"}}},
        "training": {"seed": 7},
    }))
    manifest_path = tmp_path / family / "manifest.json"
    manifest_path.write_text("{}")
    panels_path = tmp_path / family / "panels.json"
    panels_path.write_text("{}")
    manifest = {
        "gene_panel": [f"g{i}" for i in range(6)],
        "train_sample_ids": ["a", "b", "c"], "validation_sample_ids": ["v0"],
        "test_sample_ids": [],
        "samples": {"a": {"organ": "kidney"}, "b": {"organ": "kidney"},
                    "c": {"organ": "lung"}, "v0": {"organ": "kidney"}},
    }
    artifact = _fit_artifact()
    structure_path = save_centered_gene_structure_artifact(
        artifact, tmp_path / family / "structure.pt",
    )
    monkeypatch.setattr(suite_module, "load_dataset_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        suite_module, "load_train_derived_gene_panels",
        lambda _path, _manifest: {"panels": {
            "train_log1p_variance_top50": ["g0"],
            "train_log1p_variance_top200": ["g0", "g1"],
        }},
    )
    root = tmp_path / family / "suite"
    plan = suite_module.prepare_mk_structured_field_suite(
        comparison_config=str(comparison), manifest=str(manifest_path),
        train_gene_panels=str(panels_path), centered_gene_structure=str(structure_path),
        output_root=str(root), uni2_pinned_revision="pinned",
        uni2_spot_feature_cache_dir=str(uni2_cache), family=family,
    )
    assert tuple(plan["arm_order"]) == expected_arms
    assert plan["family"] == family
    for arm in expected_arms:
        config = yaml.safe_load((root / "configs" / f"{arm}.yaml").read_text())
        audit = static_audit_conditional_wae_config(config)
        assert audit["passed"]
        assert audit["prior_mode"] == expected_prior
        assert audit["deterministic_only"] is deterministic
        assert config["model"]["regularizer"] == "mmd"
        assert config["model"]["params"]["encoder_conditioning"] == "film"


def test_whole_slide_prediction_applies_deterministic_field_refinement():
    rng = np.random.default_rng(12)
    sample = SimpleNamespace(
        sample_id="slide",
        precomputed_spot_features=rng.normal(size=(9, 12)).astype(np.float32),
        image_source_available=np.ones(9, dtype=bool),
        full_sample_coords=np.stack([np.arange(9), np.arange(9) % 3], axis=1).astype(np.float64),
        adata=SimpleNamespace(
            X=rng.normal(size=(9, 6)).astype(np.float32),
            var_names=[f"g{i}" for i in range(6)],
        ),
    )
    model = DeterministicSpatialPredictor(
        6, _spatial(), hidden_dim=20, n_refinement_steps=1,
    ).eval()
    inputs, _target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    direct = model(inputs)["point_prediction"]
    whole = predict_whole_slide(model, sample, chunk_size=3)["point_prediction"]
    torch.testing.assert_close(whole, direct)
