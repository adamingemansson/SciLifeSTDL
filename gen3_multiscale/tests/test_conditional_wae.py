import json
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAEMaskedGEXDataset,
    ConditionalWAE,
    FullImageExpressionInputs,
    build_conditional_wae_example,
    imq_mmd,
)
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.tensorboard import (
    ConditionalWAESnapshotAccumulator,
    ConditionalWAETensorBoardLogger,
    METADATA_HEADER,
)
from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.training import train_conditional_wae
from gen3_multiscale.scripts.prepare_conditional_wae_suite import (
    _absolutize_existing_source_paths,
    _source_repository_root,
)


def _inputs(n=12, image_dim=16):
    rng = np.random.default_rng(0)
    return FullImageExpressionInputs(
        sample_id="slide",
        image_features=rng.normal(size=(n, image_dim)).astype(np.float32),
        coords=rng.normal(size=(n, 2)).astype(np.float32),
        image_available=np.ones(n, dtype=bool),
        query_mask=np.ones(n, dtype=bool),
    )


def _model(regularizer):
    return ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer=regularizer, latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
    )


def test_inputs_forbid_hidden_image_content_in_unavailable_rows():
    inputs = _inputs()
    inputs = FullImageExpressionInputs(
        inputs.sample_id, inputs.image_features,
        inputs.coords, np.zeros(12, dtype=bool), inputs.query_mask,
    )
    with pytest.raises(ValueError, match="exactly zero"):
        _model("mmd")(inputs)


def test_task_ii_forbids_query_gex_but_accepts_surrounding_gex():
    base = _inputs()
    query = np.zeros(12, dtype=bool)
    query[:4] = True
    expression = np.random.default_rng(1).normal(size=(12, 7)).astype(np.float32)
    expression[:4] = 0.0
    available = ~query
    inputs = FullImageExpressionInputs(
        base.sample_id, base.image_features, base.coords, base.image_available,
        query, expression[available], np.flatnonzero(available), available,
    )
    assert _model("mmd")(inputs)["expression"].shape == (4, 7)

    leaking = FullImageExpressionInputs(
        base.sample_id, base.image_features, base.coords, base.image_available,
        query, np.ones((12, 7), dtype=np.float32), np.arange(12), np.ones(12, dtype=bool),
    )
    with pytest.raises(ValueError, match="target-expression leakage"):
        _model("mmd")(leaking)


def test_real_sample_adapter_keeps_query_he_visible_but_masks_only_query_gex():
    n, genes, image_dim = 9, 7, 16
    features = np.arange(n * image_dim, dtype=np.float32).reshape(n, image_dim)
    sample = SimpleNamespace(
        sample_id="slide",
        precomputed_spot_features=features,
        image_source_available=np.ones(n, dtype=bool),
        full_sample_coords=np.stack([np.arange(n), np.arange(n) % 3], axis=1),
        adata=SimpleNamespace(X=np.arange(n * genes, dtype=np.float32).reshape(n, genes)),
    )
    query = np.array([2, 4, 7])
    inputs, target = build_conditional_wae_example(
        sample, query_indices=query, include_observed_gex=True,
    )
    np.testing.assert_array_equal(inputs.image_features[query], features[query])
    assert inputs.image_available[query].all()
    assert not inputs.expression_available[query].any()
    np.testing.assert_array_equal(inputs.observed_expression_indices, np.flatnonzero(~inputs.query_mask))
    assert inputs.observed_expression.shape == (n - len(query), genes)
    np.testing.assert_array_equal(target, sample.adata.X[query])


def test_dataset_view_reuses_mask_identity_but_not_gen3_target_zero_builder():
    n, genes, image_dim = 8, 7, 16
    obs_names = np.asarray([f"s{i}" for i in range(n)])
    coords = np.stack([np.arange(n), np.arange(n) % 2], axis=1)
    sample = SimpleNamespace(
        sample_id="slide", obs_names=obs_names,
        precomputed_spot_features=np.ones((n, image_dim), dtype=np.float32),
        image_source_available=np.ones(n, dtype=bool),
        full_sample_coords=coords,
        spatial_adjacency=tuple(build_knn_adjacency(coords, k_neighbors=2)),
        adata=SimpleNamespace(
            X=np.ones((n, genes), dtype=np.float32), obs_names=obs_names,
        ),
    )
    item = SimpleNamespace(sample_id="slide")

    class Base:
        _items = [item]
        samples = {"slide": sample}
        k_neighbors = 2
        local_k = 3
        max_rings = 2
        max_boundary_size = None

        @staticmethod
        def _resolve_barcodes(_item):
            return list(obs_names[2:]), list(obs_names[:2])

        @staticmethod
        def item_identity(_index):
            return {"sample_id": "slide", "query_fingerprint": "fixed"}

        def __len__(self):
            return 1

    inputs, target, identity = ConditionalWAEMaskedGEXDataset(
        Base(), include_observed_gex=True,
    )[0]
    assert inputs.image_available[:2].all()
    assert np.all(inputs.image_features[:2] == 1)
    assert not inputs.expression_available[:2].any()
    assert 0 not in inputs.observed_expression_indices and 1 not in inputs.observed_expression_indices
    assert target.shape == (2, genes)
    assert identity["query_fingerprint"] == "fixed"
    assert _model("mmd")(inputs)["expression"].shape == target.shape


def test_imq_mmd_is_zero_for_identical_samples_and_finite_otherwise():
    sample = torch.randn(10, 4)
    torch.testing.assert_close(imq_mmd(sample, sample), torch.zeros(()))
    assert torch.isfinite(imq_mmd(sample, torch.randn_like(sample)))


def test_mmd_variant_routes_gradients_and_inference_never_needs_gex():
    model = _model("mmd")
    inputs, target = _inputs(), torch.randn(12, 7)
    losses = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(1),
    )
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.image_conditioner.parameters())
    assert any(parameter.grad is not None for parameter in model.expression_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.residual_decoder.parameters())
    prediction = model.sample_predictive_distribution(
        inputs, generator=torch.Generator().manual_seed(2),
    )
    assert prediction["predictive_mean"].shape == target.shape
    assert prediction["predictive_samples"].shape == (3, 12, 7)
    assert model(inputs)["expression"].shape == target.shape


def test_configured_dense_sparse_attention_switch_is_real():
    model = _model("mmd")  # dense_threshold=20 in the shared test model
    model(_inputs(n=12))
    assert {
        block.cached_attention.last_attention_mode
        for block in model.image_conditioner.blocks
    } == {"dense"}
    model(_inputs(n=21))
    assert {
        block.cached_attention.last_attention_mode
        for block in model.image_conditioner.blocks
    } == {"sparse"}


def test_gan_variant_separates_discriminator_and_generator_gradients():
    model = _model("gan")
    inputs, target = _inputs(), torch.randn(12, 7)
    discriminator_loss = model.compute_discriminator_loss(
        target, generator=torch.Generator().manual_seed(3),
    )
    discriminator_loss.backward()
    assert any(parameter.grad is not None for parameter in model.discriminator.parameters())
    assert all(parameter.grad is None for parameter in model.expression_encoder.parameters())

    model.zero_grad(set_to_none=True)
    losses = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(4),
    )
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in model.discriminator.parameters())
    assert any(parameter.grad is not None for parameter in model.expression_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.image_conditioner.parameters())


def test_static_contract_distinguishes_full_he_tasks():
    base = {
        "model": {
            "arm": "wae_he_st_mmd", "kind": "conditional_wae",
            "task": "he_plus_st_to_st", "regularizer": "mmd",
            "include_observed_gex": True, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "latent_dim": 5, "hidden_dim": 24,
                "gex_feature_dim": 6, "autoencoder_hidden_dim": 20,
            },
        },
        "data": {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"},
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    report = static_audit_conditional_wae_config(base)
    assert report["query_he_visible"] is True
    assert report["query_gex_visible"] is False
    assert report["surrounding_gex_visible"] is True
    base["model"]["image_mode"] = "target_zero"
    with pytest.raises(ValueError, match="full_visible"):
        static_audit_conditional_wae_config(base)


def test_precheckpoint_root_manifest_is_not_mistaken_for_resumable_weights(tmp_path):
    payload = {"kind": "conditional_wae_supervisor_run"}
    (tmp_path / "run_manifest.json").write_text(json.dumps(payload))
    has_checkpoint, old_manifest = train_conditional_wae._checkpoint_resume_state(tmp_path)
    assert has_checkpoint is False
    assert old_manifest == payload


def test_checkpoint_pointer_without_bound_manifest_fails_closed(tmp_path, monkeypatch):
    (tmp_path / "latest_bundle.json").write_text("{}")
    monkeypatch.setattr(
        train_conditional_wae.checkpoint_module,
        "load_checkpoint_run_manifest",
        lambda _path: None,
    )
    with pytest.raises(ValueError, match="bundle-bound run manifest"):
        train_conditional_wae._checkpoint_resume_state(tmp_path)


def test_training_item_selection_deterministically_skips_single_spot_masks(monkeypatch):
    items = [
        ("one", np.zeros((1, 7), dtype=np.float32), {"mask": "one"}),
        ("valid", np.zeros((3, 7), dtype=np.float32), {"mask": "valid"}),
    ]
    monkeypatch.setattr(
        train_conditional_wae,
        "deterministic_train_index_for_step",
        lambda *_args, **_kwargs: 0,
    )
    inputs, target, identity, skipped = (
        train_conditional_wae._select_train_item_with_minimum_queries(
            items, step=3200, seed=0,
        )
    )
    assert inputs == "valid"
    assert target.shape == (3, 7)
    assert identity == {"mask": "valid"}
    assert skipped == 1


def test_training_item_selection_fails_if_every_mask_is_undersized(monkeypatch):
    items = [("one", np.zeros((1, 7), dtype=np.float32), {})]
    monkeypatch.setattr(
        train_conditional_wae,
        "deterministic_train_index_for_step",
        lambda *_args, **_kwargs: 0,
    )
    with pytest.raises(ValueError, match="no eligible mask"):
        train_conditional_wae._select_train_item_with_minimum_queries(
            items, step=0, seed=0,
        )


def test_tensorboard_snapshot_is_bounded_aligned_and_has_he_thumbnails(monkeypatch, tmp_path):
    inputs = _inputs(n=7)
    target = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    sample = SimpleNamespace(
        sample_id="slide", patient_id="patient-1",
        full_sample_coords=np.stack([np.arange(7), np.arange(7) + 10], axis=1),
        image_source_available=np.ones(7, dtype=bool),
        patches=np.full((7, 8, 8, 3), 128, dtype=np.uint8),
        adata=SimpleNamespace(obs_names=np.asarray([f"spot-{i}" for i in range(7)])),
    )
    accumulator = ConditionalWAESnapshotAccumulator(
        sample_records={"slide": {"organ": "kidney"}},
        gene_names=[f"g{i}" for i in range(7)], logged_gene_indices=[1, 3],
        max_points=5, max_points_per_item=3, thumbnail_max_points=2,
        thumbnail_size=4,
    )
    prediction = {
        "predictive_mean": target + 1,
        "image_context": torch.arange(42, dtype=torch.float32).reshape(7, 6),
    }
    identity = {"stratum": "medium", "query_fingerprint": "fixed-mask"}
    accumulator.add(
        inputs=inputs, target=target, identity=identity, prediction=prediction,
        posterior_z=torch.arange(35, dtype=torch.float32).reshape(7, 5), sample=sample,
    )
    accumulator.add(
        inputs=inputs, target=target, identity=identity, prediction=prediction,
        posterior_z=torch.arange(35, dtype=torch.float32).reshape(7, 5), sample=sample,
    )
    arrays = accumulator.arrays()
    assert accumulator.n_points == 5
    assert arrays["posterior_z"].shape == (5, 5)
    assert arrays["context"].shape == (5, 6)
    assert arrays["prediction_genes"].shape == (5, 2)
    assert arrays["thumbnails"].shape == (2, 3, 4, 4)
    assert len(accumulator.metadata) == 5
    assert accumulator.metadata[0][METADATA_HEADER.index("organ")] == "kidney"

    class FakeWriter:
        def __init__(self):
            self.scalars = []
            self.embeddings = []

        def add_scalar(self, *args):
            self.scalars.append(args)

        def add_embedding(self, *args, **kwargs):
            self.embeddings.append((args, kwargs))

        def flush(self):
            pass

        def close(self):
            pass

    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    monkeypatch.setattr(logger, "_add_spatial_figures", lambda *_args: None)
    losses = _model("mmd").compute_generator_losses(
        _inputs(), torch.randn(12, 7), generator=torch.Generator().manual_seed(7),
    )
    logger.add_train_scalars(10, losses, grad_norm=1.0, learning_rate=1e-4)
    logger.add_validation_scalars(10, {
        "total": 1.0, "rmse": 0.5, "pcc_loss": 0.8, "conditional_mean_rmse": 0.6,
    })
    logger.add_snapshot(10, accumulator)
    assert {entry[0] for entry in writer.scalars} >= {
        "train/total", "train/prior", "validation/total", "validation/rmse",
    }
    assert len(writer.embeddings) == 4
    assert sum("label_img" in kwargs for _args, kwargs in writer.embeddings) == 2


def test_suite_absolutizes_only_existing_paths_from_comparison_repository(tmp_path):
    source = tmp_path / "source"
    comparison = source / "gen3_multiscale" / "results" / "run" / "config.yaml"
    comparison.parent.mkdir(parents=True)
    comparison.write_text("model: {}\n")
    asset = source / "data" / "raw" / "hest1k"
    asset.mkdir(parents=True)

    assert _source_repository_root(comparison) == source.resolve()
    payload = {
        "data": {"hest_data_dir": "data/raw/hest1k", "missing": "data/not-created"},
        "device": "cuda",
        "revision": "d517a8dd",
    }
    resolved = _absolutize_existing_source_paths(payload, source)
    assert resolved["data"]["hest_data_dir"] == str(asset.resolve())
    assert resolved["data"]["missing"] == "data/not-created"
    assert resolved["device"] == "cuda"
    assert resolved["revision"] == "d517a8dd"
