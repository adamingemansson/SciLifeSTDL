import json
import math
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAEMaskedGEXDataset,
    ConditionalWAE,
    FiLMConditionedExpressionEncoder,
    FrozenGeneEmbeddingExpressionEncoder,
    FullImageExpressionInputs,
    build_conditional_wae_example,
    imq_mmd,
)
from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.spatial_refinement import (
    SpatialExpressionRefiner,
    padded_neighbor_graph,
)
from gen3_multiscale.conditional_wae.film_diagnostics import compute_film_diagnostics
from gen3_multiscale.conditional_wae.whole_slide import predict_whole_slide
from gen3_multiscale.conditional_wae.tensorboard import (
    ConditionalWAESnapshotAccumulator,
    ConditionalWAETensorBoardLogger,
    METADATA_HEADER,
)
from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.evaluation.conditional_wae_evaluator import _latent_path_predictions
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


def test_latent_path_diagnostic_preserves_the_standard_prior_mean_and_supports_film():
    inputs = _inputs()
    target = torch.randn(12, 7)
    for model in (_model("mmd"), _film_model(regularizer="mmd")):
        model.eval()
        diagnostic = _latent_path_predictions(
            model,
            inputs,
            target,
            n_prior_samples=4,
            generator=torch.Generator().manual_seed(17),
        )
        ordinary = model.sample_predictive_distribution(
            inputs,
            n_samples=4,
            generator=torch.Generator().manual_seed(17),
        )
        torch.testing.assert_close(diagnostic["model"], ordinary["predictive_mean"])
        torch.testing.assert_close(diagnostic["predictive_std"], ordinary["predictive_std"])
        torch.testing.assert_close(
            diagnostic["conditional_mean"], ordinary["conditional_mean_expression"],
        )
        assert diagnostic["posterior_reconstruction"].shape == target.shape
        assert diagnostic["zero_latent"].shape == target.shape
        assert diagnostic["shuffled_posterior"].shape == target.shape
        assert diagnostic["posterior_z"].shape == (12, model.latent_dim)


def test_latent_path_diagnostic_refuses_distributional_head_that_bypasses_z():
    model = ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
        likelihood="zero_inflated_gaussian", distributional_hidden_dim=16,
    )
    with pytest.raises(ValueError, match="bypasses the WAE latent decoder"):
        _latent_path_predictions(
            model,
            _inputs(),
            torch.randn(12, 7),
            n_prior_samples=4,
            generator=torch.Generator().manual_seed(0),
        )


def _film_model(*, film_layers=("first", "second"), shared=False, regularizer="gan"):
    return ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer=regularizer, latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
        encoder_conditioning="film", film_layers=film_layers, film_shared_generator=shared,
    )


def _frozen_table_model(*, use_film, embedding_dim=9, regularizer="mmd"):
    torch.manual_seed(0)
    table = torch.randn(embedding_dim, 7)
    return ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer=regularizer, latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
        encoder_conditioning=("film" if use_film else "none"),
        gene_encoder_table=table,
    )


def test_frozen_gene_embedding_encoder_forward_shape_without_film():
    table = torch.randn(9, 7)
    encoder = FrozenGeneEmbeddingExpressionEncoder(table, context_dim=24, latent_dim=5, hidden_dim=20)
    output = encoder(torch.randn(6, 7))
    assert output.shape == (6, 5)


def test_frozen_gene_embedding_encoder_table_is_a_buffer_not_a_trainable_parameter():
    table = torch.randn(9, 7)
    encoder = FrozenGeneEmbeddingExpressionEncoder(table, context_dim=24, latent_dim=5, hidden_dim=20)
    parameter_names = {name for name, _ in encoder.named_parameters()}
    buffer_names = {name for name, _ in encoder.named_buffers()}
    assert "frozen_table" not in parameter_names
    assert "frozen_table" in buffer_names
    torch.testing.assert_close(encoder.frozen_table, table)


def test_frozen_gene_embedding_encoder_empty_film_layers_is_valid_and_ignores_context():
    table = torch.randn(9, 7)
    encoder = FrozenGeneEmbeddingExpressionEncoder(table, context_dim=24, latent_dim=5, hidden_dim=20, film_layers=())
    assert encoder.film_first is None and encoder.film_second is None
    output_no_context = encoder(torch.randn(6, 7))
    assert output_no_context.shape == (6, 5)


def test_frozen_gene_embedding_encoder_requires_context_when_film_layers_set():
    table = torch.randn(9, 7)
    encoder = FrozenGeneEmbeddingExpressionEncoder(
        table, context_dim=24, latent_dim=5, hidden_dim=20, film_layers=("first",),
    )
    with pytest.raises(ValueError, match="requires 'context'"):
        encoder(torch.randn(6, 7))


def test_frozen_gene_embedding_encoder_rejects_gene_dim_mismatch():
    table = torch.randn(9, 7)
    encoder = FrozenGeneEmbeddingExpressionEncoder(table, context_dim=24, latent_dim=5, hidden_dim=20)
    with pytest.raises(ValueError, match="genes"):
        encoder(torch.randn(6, 5))


def test_conditional_wae_rejects_a_negative_z_noise_std():
    with pytest.raises(ValueError, match="z_noise_std"):
        ConditionalWAE(
            7,
            Architecture1ImageConditioner(
                7, image_feature_dim=16, gex_feature_dim=6,
                hidden_dim=24, n_heads=4, n_blocks=1,
                dense_threshold=20, sparse_k=3, dropout=0.0,
            ),
            regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
            discriminator_hidden_dim=12, n_inference_samples=3, z_noise_std=-0.1,
        )


def test_z_noise_std_zero_reproduces_the_pre_fix_deterministic_reconstruction():
    """Default z_noise_std=0.0 must be a strict no-op: the decoder always
    sees the clean encoded z regardless of which generator is passed, so
    reconstruction is identical across generator seeds (only prior_loss,
    which samples an independent prior draw, would differ)."""
    model = _model("mmd")
    inputs, target = _inputs(), torch.rand(12, 7)
    first = model.compute_generator_losses(inputs, target, generator=torch.Generator().manual_seed(1))
    second = model.compute_generator_losses(inputs, target, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(first["expression"], second["expression"])
    torch.testing.assert_close(first["latent"], second["latent"])


def test_z_noise_std_positive_perturbs_the_decoder_input_but_not_the_regularized_latent():
    """The whole point of this fix: decode from a noised z (so the decoder
    is forced to be robust to prior-scale z variation at inference), while
    the MMD/GAN regularizer still sees the CLEAN encoded z (so the
    aggregate-posterior-matching objective itself is unaffected)."""
    model = ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3, z_noise_std=0.5,
    )
    inputs, target = _inputs(), torch.rand(12, 7)
    first = model.compute_generator_losses(inputs, target, generator=torch.Generator().manual_seed(1))
    second = model.compute_generator_losses(inputs, target, generator=torch.Generator().manual_seed(2))
    # Different generator seeds -> different noise draws -> different
    # reconstruction, even though the target/context are identical.
    assert not torch.allclose(first["expression"], second["expression"])
    # The clean encoded latent (what the regularizer sees) never depends
    # on z_noise_std or the generator's noise draw.
    torch.testing.assert_close(first["latent"], second["latent"])
    # conditional_mean_head only ever takes image context, never z -- must
    # stay completely unaffected by z_noise_std.
    torch.testing.assert_close(
        first["conditional_mean_expression"], second["conditional_mean_expression"],
    )
    for losses in (first, second):
        assert torch.isfinite(losses["total"])


def test_conditional_wae_uses_frozen_gene_embedding_encoder_when_table_provided():
    model = _frozen_table_model(use_film=False)
    assert isinstance(model.expression_encoder, FrozenGeneEmbeddingExpressionEncoder)
    inputs = _inputs()
    target = torch.rand(12, 7)
    losses = model.compute_generator_losses(inputs, target)
    assert torch.isfinite(losses["total"])


def test_conditional_wae_frozen_gene_embedding_encoder_composes_with_film():
    model = _frozen_table_model(use_film=True)
    assert isinstance(model.expression_encoder, FrozenGeneEmbeddingExpressionEncoder)
    assert model.expression_encoder.film_first is not None
    assert model.expression_encoder.film_second is not None
    inputs = _inputs()
    target = torch.rand(12, 7)
    losses = model.compute_generator_losses(inputs, target)
    assert torch.isfinite(losses["total"])


def test_conditional_wae_gene_encoder_table_rejects_wrong_gene_count():
    with pytest.raises(ValueError, match="n_genes"):
        ConditionalWAE(
            7,
            Architecture1ImageConditioner(
                7, image_feature_dim=16, gex_feature_dim=6,
                hidden_dim=24, n_heads=4, n_blocks=1,
                dense_threshold=20, sparse_k=3, dropout=0.0,
            ),
            regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
            discriminator_hidden_dim=12, n_inference_samples=3,
            gene_encoder_table=torch.randn(9, 6),  # 6 != n_genes=7
        )


def test_film_encoder_output_is_identical_across_contexts_at_init():
    encoder = FiLMConditionedExpressionEncoder(7, context_dim=24, latent_dim=5, hidden_dim=20)
    expression = torch.randn(6, 7)
    context_a = torch.randn(6, 24)
    context_b = torch.randn(6, 24) * 10.0
    torch.testing.assert_close(encoder(expression, context_a), encoder(expression, context_b))


def test_film_layer_selection_only_builds_the_requested_generators():
    both = FiLMConditionedExpressionEncoder(7, 24, film_layers=("first", "second"))
    assert both.film_first is not None and both.film_second is not None
    first_only = FiLMConditionedExpressionEncoder(7, 24, film_layers=("first",))
    assert first_only.film_first is not None and first_only.film_second is None
    last_only = FiLMConditionedExpressionEncoder(7, 24, film_layers=("second",))
    assert last_only.film_first is None and last_only.film_second is not None


def test_film_shared_generator_ties_both_layers_to_one_module():
    shared = FiLMConditionedExpressionEncoder(7, 24, film_layers=("first", "second"), shared_film_generator=True)
    assert shared.film_first is shared.film_second
    independent = FiLMConditionedExpressionEncoder(7, 24, film_layers=("first", "second"), shared_film_generator=False)
    assert independent.film_first is not independent.film_second


def test_film_shared_generator_requires_both_layers():
    with pytest.raises(ValueError, match="both layers"):
        FiLMConditionedExpressionEncoder(7, 24, film_layers=("first",), shared_film_generator=True)


def test_film_layers_rejects_invalid_or_empty_values():
    with pytest.raises(ValueError, match="film_layers"):
        FiLMConditionedExpressionEncoder(7, 24, film_layers=())
    with pytest.raises(ValueError, match="film_layers"):
        FiLMConditionedExpressionEncoder(7, 24, film_layers=("bogus",))


def test_film_variant_routes_gradients_into_the_film_generators():
    model = _film_model(regularizer="mmd")
    inputs, target = _inputs(), torch.randn(12, 7)
    losses = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(1),
    )
    losses["total"].backward()
    encoder = model.expression_encoder
    assert encoder.film_first.to_gamma_beta.weight.grad is not None
    assert encoder.film_second.to_gamma_beta.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.image_conditioner.parameters())
    assert any(parameter.grad is not None for parameter in model.residual_decoder.parameters())


def test_film_variant_discriminator_loss_requires_inputs():
    model = _film_model(regularizer="gan")
    target = torch.randn(12, 7)
    with pytest.raises(ValueError, match="requires 'inputs'"):
        model.compute_discriminator_loss(target, generator=torch.Generator().manual_seed(2))
    loss = model.compute_discriminator_loss(
        target, inputs=_inputs(), generator=torch.Generator().manual_seed(2),
    )
    assert torch.isfinite(loss)


def test_film_variant_still_enforces_query_gex_leakage_contract():
    model = _film_model(regularizer="gan")
    base = _inputs()
    query = np.ones(12, dtype=bool)
    leaking = FullImageExpressionInputs(
        base.sample_id, base.image_features, base.coords, base.image_available,
        query, np.ones((12, 7), dtype=np.float32), np.arange(12), np.ones(12, dtype=bool),
    )
    with pytest.raises(ValueError, match="target-expression leakage"):
        model.compute_generator_losses(leaking, torch.randn(12, 7), generator=torch.Generator().manual_seed(3))


def test_encode_posterior_requires_inputs_when_film_active_but_not_otherwise():
    film_model = _film_model(regularizer="mmd")
    target = torch.randn(12, 7)
    with pytest.raises(ValueError, match="requires 'inputs'"):
        film_model.encode_posterior(target)
    posterior = film_model.encode_posterior(target, inputs=_inputs())
    assert posterior.shape == (12, 5)

    plain_model = _model("mmd")
    assert plain_model.encode_posterior(target).shape == (12, 5)


def _film_batch(model, n=8):
    inputs, target = _inputs(n=n), torch.randn(n, 7)
    context = model.image_conditioner(inputs)
    posterior_z = model.encode_posterior(target, inputs=inputs)
    prediction = model.sample_predictive_distribution(inputs, generator=torch.Generator().manual_seed(0))
    return posterior_z, context, prediction


def test_compute_film_diagnostics_requires_a_film_encoder():
    model = _model("mmd")
    posterior_z, context, prediction = _film_batch(_film_model(regularizer="mmd"))
    with pytest.raises(ValueError, match="encoder_conditioning='film'"):
        compute_film_diagnostics(
            model, posterior_z, context, prediction["predictive_std"],
            prediction["predictive_mean"], prediction["conditional_mean_expression"],
        )


def test_compute_film_diagnostics_requires_at_least_two_rows():
    model = _film_model(regularizer="mmd")
    posterior_z, context, prediction = _film_batch(model, n=1)
    with pytest.raises(ValueError, match="at least two"):
        compute_film_diagnostics(
            model, posterior_z, context, prediction["predictive_std"],
            prediction["predictive_mean"], prediction["conditional_mean_expression"],
        )


def test_compute_film_diagnostics_reports_effective_rank_and_active_dims_within_latent_dim():
    model = _film_model(regularizer="mmd")
    posterior_z, context, prediction = _film_batch(model, n=10)
    diagnostics = compute_film_diagnostics(
        model, posterior_z, context, prediction["predictive_std"],
        prediction["predictive_mean"], prediction["conditional_mean_expression"],
    )
    assert 0.0 <= diagnostics["effective_rank"] <= model.latent_dim
    assert 0 <= diagnostics["active_dimensions"] <= model.latent_dim
    assert diagnostics["latent_dim"] == model.latent_dim
    assert diagnostics["latent_dim_mean"].shape == (model.latent_dim,)
    assert diagnostics["latent_dim_std"].shape == (model.latent_dim,)
    assert diagnostics["predictive_std_mean"] >= 0.0
    assert diagnostics["stochastic_vs_conditional_mean_diff"] >= 0.0


def test_compute_film_diagnostics_reports_gamma_beta_only_for_active_layers():
    model = _film_model(film_layers=("first",), regularizer="mmd")
    posterior_z, context, prediction = _film_batch(model, n=6)
    diagnostics = compute_film_diagnostics(
        model, posterior_z, context, prediction["predictive_std"],
        prediction["predictive_mean"], prediction["conditional_mean_expression"],
    )
    assert set(diagnostics["gamma_beta"]) == {"first"}
    gamma, beta = diagnostics["gamma_beta"]["first"]
    assert gamma.shape == beta.shape == (6, model.expression_encoder.linear1.out_features)


def test_build_model_wires_film_config_into_the_constructed_encoder():
    config = {
        "model": {
            "arm": "wae_he_gan_film_first_only", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "gex_feature_dim": 6,
                "hidden_dim": 24, "n_heads": 4, "n_blocks": 1,
                "dense_threshold": 20, "sparse_k": 3, "dropout": 0.1,
                "latent_dim": 5, "autoencoder_hidden_dim": 20,
                "discriminator_hidden_dim": 12, "n_inference_samples": 3,
                "encoder_conditioning": "film", "film_layers": ["first"],
                "film_shared_generator": False,
            },
        },
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    model = train_conditional_wae._build_model(config, n_genes=7)
    assert isinstance(model.expression_encoder, FiLMConditionedExpressionEncoder)
    assert model.expression_encoder.film_first is not None
    assert model.expression_encoder.film_second is None


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


def test_predictive_output_names_separate_point_prediction_from_wae_prior_mean():
    model = _model("mmd")
    inputs = _inputs()
    prediction = model.sample_predictive_distribution(
        inputs, n_samples=3, generator=torch.Generator().manual_seed(2),
    )
    torch.testing.assert_close(prediction["expression"], model(inputs)["expression"])
    torch.testing.assert_close(prediction["point_prediction"], prediction["expression"])
    torch.testing.assert_close(
        prediction["conditional_mean_expression"], prediction["point_prediction"],
    )
    torch.testing.assert_close(
        prediction["wae_predictive_mean"], prediction["predictive_mean"],
    )


def test_masked_validation_uses_the_deterministic_point_prediction_as_primary():
    model = _model("mmd").eval()
    inputs = _inputs()
    target = model(inputs)["expression"].detach().cpu().numpy()

    class OneItem:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return inputs, target, {
                "sample_id": "slide", "stratum": "test", "query_fingerprint": "fixed",
            }

    result = train_conditional_wae._validate(
        model, OneItem(), device=torch.device("cpu"), seed=0,
    )
    # rmse_pcc_reconstruction_loss deliberately evaluates sqrt(MSE + 1e-8)
    # for finite gradients at an exact match, so its perfect-prediction floor
    # is 1e-4 rather than zero.
    assert result["rmse"] == pytest.approx(1.0e-4, abs=1e-8)
    assert result["conditional_mean_rmse"] == result["rmse"]
    assert result["wae_prior_rmse"] > result["rmse"]


def test_latent_spatial_correlation_zero_reproduces_the_historical_sampler():
    model = _model("mmd")
    inputs = _inputs()
    default = model.sample_predictive_distribution(
        inputs, n_samples=4, generator=torch.Generator().manual_seed(5),
    )
    explicit = model.sample_predictive_distribution(
        inputs, n_samples=4, generator=torch.Generator().manual_seed(5),
        latent_spatial_correlation=0.0,
    )
    assert torch.equal(default["predictive_samples"], explicit["predictive_samples"])


def test_latent_spatial_correlation_keeps_the_marginal_prior_but_couples_spots():
    """The mixing must leave each spot's z marginally N(0,I) -- the
    distribution MMD actually trained the encoder to match -- while making
    a drawn field spatially coherent rather than salt-and-pepper."""
    rho, latent_dim, n_spots, draws = 0.75, 32, 16, 3000
    shared_weight, local_weight = math.sqrt(rho), math.sqrt(1.0 - rho)
    generator = torch.Generator().manual_seed(0)
    stacked = []
    for _ in range(draws):
        noise = torch.randn(n_spots, latent_dim, generator=generator)
        shared = torch.randn(1, latent_dim, generator=generator)
        stacked.append(shared_weight * shared + local_weight * noise)
    z = torch.stack(stacked)
    assert abs(float(z.std(0).mean()) - 1.0) < 0.05
    correlation = torch.corrcoef(
        torch.stack([z[:, 0, :].flatten(), z[:, 1, :].flatten()])
    )[0, 1]
    assert abs(float(correlation) - rho) < 0.05


def test_latent_spatial_correlation_rejects_values_outside_the_unit_interval():
    model = _model("mmd")
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="latent_spatial_correlation"):
            model.sample_predictive_distribution(
                _inputs(), latent_spatial_correlation=bad,
            )


def _distributional_model(**kwargs):
    return ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
        likelihood="zero_inflated_gaussian", distributional_hidden_dim=16, **kwargs,
    )


def _sparse_target(n=12, genes=7, seed=0):
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand(n, genes, generator=generator)
    target[target < 0.6] = 0.0  # a real log1p target has a large point mass at 0
    return target


def test_default_likelihood_builds_no_distributional_head_and_adds_no_loss_key():
    """The trainer's accumulator calls .detach() on every scalar loss entry,
    so a None placeholder would break every existing gaussian_mse run."""
    model = _model("mmd")
    assert model.distributional_head is None
    losses = model.compute_generator_losses(
        _inputs(), _sparse_target(), generator=torch.Generator().manual_seed(0),
    )
    assert "distributional_loss" not in losses
    for key, value in losses.items():
        if key not in ("expression", "conditional_mean_expression", "latent"):
            float(value.detach())  # must not raise


def test_zero_inflated_head_contributes_a_real_loss_and_receives_gradient():
    model = _distributional_model()
    losses = model.compute_generator_losses(
        _inputs(), _sparse_target(), generator=torch.Generator().manual_seed(0),
    )
    assert "distributional_loss" in losses
    assert torch.isfinite(losses["distributional_loss"])
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.distributional_head.parameters())


def test_distributional_predictive_std_is_analytic_not_monte_carlo():
    """The measured failure was Monte-Carlo scatter over a latent producing
    z_std of 9-19. With a likelihood head the spread is a trained output, so
    it must not change with the number of drawn samples."""
    model = _distributional_model()
    inputs = _inputs()
    few = model.sample_predictive_distribution(
        inputs, n_samples=2, generator=torch.Generator().manual_seed(0),
    )
    many = model.sample_predictive_distribution(
        inputs, n_samples=64, generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(few["predictive_std"], many["predictive_std"])
    torch.testing.assert_close(few["predictive_mean"], many["predictive_mean"])
    assert few["predictive_samples"].shape[0] == 2
    assert many["predictive_samples"].shape[0] == 64
    assert bool((few["predictive_std"] >= 0).all())


def test_zero_inflated_head_can_commit_to_an_exact_zero():
    """The flat-UMOD failure was an inability to say 'this spot is zero'.
    A confident zero prediction must beat a confident nonzero one on a
    genuinely all-zero target."""
    from gen3_multiscale.conditional_wae.distributional import zero_inflated_gaussian_nll
    target = torch.zeros(4, 3)
    shape = target.shape
    confident_zero = zero_inflated_gaussian_nll(
        torch.full(shape, 8.0), torch.zeros(shape), torch.full(shape, -2.0), target,
    )
    confident_nonzero = zero_inflated_gaussian_nll(
        torch.full(shape, -8.0), torch.full(shape, 2.0), torch.full(shape, -2.0), target,
    )
    assert float(confident_zero) < float(confident_nonzero)


def test_static_contract_validates_and_reports_the_likelihood():
    config = _geneencoder_config(
        "wae_he_mmd_geneencoder_mlp_nofilm",
        encoder_conditioning="none", gene_encoder_source="linear",
    )
    assert static_audit_conditional_wae_config(config)["likelihood"] == "gaussian_mse"
    config["model"]["params"]["likelihood"] = "zero_inflated_gaussian"
    assert static_audit_conditional_wae_config(config)["likelihood"] == "zero_inflated_gaussian"
    config["model"]["params"]["likelihood"] = "poisson"
    with pytest.raises(ValueError, match="likelihood"):
        static_audit_conditional_wae_config(config)


def test_model_rejects_an_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        _distributional_model.__wrapped__ if False else ConditionalWAE(
            7,
            Architecture1ImageConditioner(
                7, image_feature_dim=16, gex_feature_dim=6, hidden_dim=24,
                n_heads=4, n_blocks=1, dense_threshold=20, sparse_k=3, dropout=0.0,
            ),
            regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
            discriminator_hidden_dim=12, n_inference_samples=3, likelihood="poisson",
        )


def _refining_model(n_steps, regularizer="mmd"):
    return ConditionalWAE(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        regularizer=regularizer, latent_dim=5, autoencoder_hidden_dim=20,
        discriminator_hidden_dim=12, n_inference_samples=3,
        n_refinement_steps=n_steps, refinement_k_neighbors=3,
        refinement_hidden_dim=16, refinement_gex_feature_dim=8,
    )


def test_zero_refinement_steps_builds_no_refiner_and_changes_nothing():
    model = _refining_model(0)
    assert model.spatial_refiner is None
    baseline = _model("mmd")
    assert model(_inputs())["expression"].shape == baseline(_inputs())["expression"].shape


def test_refiner_starts_at_the_identity_so_enabling_it_cannot_break_a_checkpoint():
    model = _refining_model(3)
    inputs = _inputs()
    context = model.image_conditioner(inputs)
    base = model.conditional_mean_head(context)
    # The update head is zero-initialised, so before any training the
    # refinement is exactly a no-op regardless of step count.
    torch.testing.assert_close(model._refine(base, context, inputs), base)


def test_refinement_changes_the_prediction_once_the_update_head_is_nonzero():
    model = _refining_model(2)
    inputs = _inputs()
    with torch.no_grad():
        for parameter in model.spatial_refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    context = model.image_conditioner(inputs)
    base = model.conditional_mean_head(context)
    refined = model._refine(base, context, inputs)
    assert not torch.allclose(refined, base)
    assert refined.shape == base.shape


def test_refinement_never_reads_target_expression():
    """The refiner must depend only on the model's OWN prediction, image
    context and coordinates -- never on the held-out target."""
    model = _refining_model(2)
    inputs = _inputs()
    with torch.no_grad():
        for parameter in model.spatial_refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    generator_kwargs = {"generator": torch.Generator().manual_seed(0)}
    first = model.compute_generator_losses(inputs, torch.zeros(12, 7), **generator_kwargs)
    second = model.compute_generator_losses(
        inputs, torch.zeros(12, 7), generator=torch.Generator().manual_seed(0),
    )
    torch.testing.assert_close(first["expression"], second["expression"])
    # A completely different target must not change what the refiner does to
    # the image-only conditional mean.
    third = model.compute_generator_losses(
        inputs, torch.randn(12, 7) * 5.0, generator=torch.Generator().manual_seed(0),
    )
    torch.testing.assert_close(
        first["conditional_mean_expression"], third["conditional_mean_expression"],
    )


def test_refinement_gradients_reach_the_refiner_and_the_conditioner():
    model = _refining_model(2)
    inputs, target = _inputs(), torch.randn(12, 7)
    losses = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(1),
    )
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.spatial_refiner.parameters())
    assert any(p.grad is not None for p in model.image_conditioner.parameters())


def test_refiner_rejects_malformed_construction():
    with pytest.raises(ValueError, match="n_refinement_steps"):
        _refining_model(-1)
    with pytest.raises(ValueError, match="positive"):
        SpatialExpressionRefiner(7, 24, hidden_dim=0)
    with pytest.raises(ValueError, match="positive"):
        SpatialExpressionRefiner(0, 24)


def test_padded_neighbor_graph_masks_padding_and_excludes_self_loops():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(9, 2)).astype(np.float32)
    indices, mask = padded_neighbor_graph(coords, k_neighbors=3)
    assert indices.shape == mask.shape
    assert indices.shape[0] == 9
    assert bool(mask.any(dim=1).all())  # no isolated spot
    for row in range(9):
        neighbors = indices[row][mask[row]].tolist()
        assert row not in neighbors, "self-loop leaked into the refinement graph"


def test_static_contract_reports_and_validates_refinement_steps():
    config = _geneencoder_config(
        "wae_he_mmd_geneencoder_mlp_nofilm",
        encoder_conditioning="none", gene_encoder_source="linear",
    )
    config["model"]["params"].update({"n_refinement_steps": 4})
    assert static_audit_conditional_wae_config(config)["n_refinement_steps"] == 4
    config["model"]["params"]["n_refinement_steps"] = -1
    with pytest.raises(ValueError, match="n_refinement_steps"):
        static_audit_conditional_wae_config(config)


def test_build_model_wires_refinement_from_config():
    config = _geneencoder_config(
        "wae_he_gan_control", encoder_conditioning="none", gene_encoder_source="linear",
    )
    config["model"]["params"].update({
        "n_heads": 4, "n_blocks": 1, "dense_threshold": 20, "sparse_k": 3,
        "discriminator_hidden_dim": 12, "n_inference_samples": 3,
        "n_refinement_steps": 3, "refinement_k_neighbors": 5,
        "refinement_hidden_dim": 16, "refinement_gex_feature_dim": 8,
    })
    model = train_conditional_wae._build_model(config, n_genes=7)
    assert model.n_refinement_steps == 3
    assert model.spatial_refiner is not None
    assert model.spatial_refiner.k_neighbors == 5


def test_sample_predictive_distribution_defaults_reproduce_the_pre_ex_post_prior_behavior():
    model = _model("mmd")
    inputs = _inputs()
    baseline = model.sample_predictive_distribution(
        inputs, generator=torch.Generator().manual_seed(7),
    )
    explicit_standard_normal = model.sample_predictive_distribution(
        inputs, generator=torch.Generator().manual_seed(7),
        z_mean=torch.zeros(model.latent_dim), z_std=torch.ones(model.latent_dim),
    )
    torch.testing.assert_close(
        baseline["predictive_samples"], explicit_standard_normal["predictive_samples"],
    )


def test_sample_predictive_distribution_ex_post_prior_with_zero_std_is_deterministic_at_the_mean():
    model = _model("mmd")
    inputs = _inputs()
    z_mean = torch.randn(model.latent_dim)
    prediction = model.sample_predictive_distribution(
        inputs, n_samples=5, generator=torch.Generator().manual_seed(3),
        z_mean=z_mean, z_std=torch.zeros(model.latent_dim),
    )
    samples = prediction["predictive_samples"]
    for index in range(1, samples.shape[0]):
        torch.testing.assert_close(samples[0], samples[index])
    context = model.image_conditioner(inputs)
    expected, _ = model.decode(z_mean.unsqueeze(0).expand(context.shape[0], -1), context)
    torch.testing.assert_close(samples[0], expected)


def test_sample_predictive_distribution_ex_post_prior_shifts_and_scales_z_relative_to_default():
    model = _model("mmd")
    inputs = _inputs()
    narrow = model.sample_predictive_distribution(
        inputs, n_samples=32, generator=torch.Generator().manual_seed(11),
        z_mean=torch.zeros(model.latent_dim), z_std=torch.full((model.latent_dim,), 0.01),
    )
    wide = model.sample_predictive_distribution(
        inputs, n_samples=32, generator=torch.Generator().manual_seed(11),
        z_mean=torch.zeros(model.latent_dim), z_std=torch.ones(model.latent_dim),
    )
    assert float(narrow["predictive_std"].mean()) < float(wide["predictive_std"].mean())


def test_sample_predictive_distribution_rejects_malformed_ex_post_prior():
    model = _model("mmd")
    inputs = _inputs()
    with pytest.raises(ValueError, match="z_mean"):
        model.sample_predictive_distribution(inputs, z_mean=torch.zeros(model.latent_dim + 1))
    with pytest.raises(ValueError, match="z_std"):
        model.sample_predictive_distribution(inputs, z_std=torch.zeros(model.latent_dim + 1))
    with pytest.raises(ValueError, match="non-negative"):
        model.sample_predictive_distribution(inputs, z_std=-torch.ones(model.latent_dim))


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


def _geneencoder_config(arm, *, encoder_conditioning, gene_encoder_source="frozen_table",
                       gene_encoder_table_path="basis.pt"):
    params = {
        "image_feature_dim": 16, "latent_dim": 64, "hidden_dim": 24,
        "gex_feature_dim": 6, "autoencoder_hidden_dim": 20, "discriminator_hidden_dim": 12,
        "gene_encoder_source": gene_encoder_source, "encoder_conditioning": encoder_conditioning,
    }
    if encoder_conditioning == "film":
        params["film_layers"] = ["first", "second"]
    return {
        "model": {
            "arm": arm, "kind": "conditional_wae", "task": "he_to_st", "regularizer": "mmd",
            "include_observed_gex": False, "image_mode": "full_visible", "params": params,
        },
        "data": {
            "gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc",
            "gene_encoder_table_path": gene_encoder_table_path,
        },
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }


@pytest.mark.parametrize("arm,use_film,gene_encoder_source", [
    ("wae_he_mmd_geneencoder_scfoundation_film", True, "frozen_table"),
    ("wae_he_mmd_geneencoder_scfoundation_nofilm", False, "frozen_table"),
    ("wae_he_mmd_geneencoder_mlp_film", True, "linear"),
    ("wae_he_mmd_geneencoder_mlp_nofilm", False, "linear"),
])
def test_static_contract_passes_for_every_geneencoder_arm(arm, use_film, gene_encoder_source):
    config = _geneencoder_config(
        arm, encoder_conditioning=("film" if use_film else "none"),
        gene_encoder_source=gene_encoder_source,
    )
    if gene_encoder_source == "linear":
        del config["data"]["gene_encoder_table_path"]
    report = static_audit_conditional_wae_config(config)
    assert report["passed"] is True
    assert report["gene_encoder_source"] == gene_encoder_source


def test_static_contract_rejects_unknown_gene_encoder_source():
    config = _geneencoder_config(
        "wae_he_mmd_geneencoder_scfoundation_film", encoder_conditioning="film",
        gene_encoder_source="bogus",
    )
    with pytest.raises(ValueError, match="gene_encoder_source"):
        static_audit_conditional_wae_config(config)


def test_static_contract_requires_table_path_for_frozen_table_source():
    config = _geneencoder_config(
        "wae_he_mmd_geneencoder_scfoundation_film", encoder_conditioning="film",
        gene_encoder_table_path="",
    )
    with pytest.raises(ValueError, match="gene_encoder_table_path"):
        static_audit_conditional_wae_config(config)


def test_static_contract_gene_encoder_source_defaults_to_linear():
    config = _geneencoder_config("wae_he_gan_control", encoder_conditioning="none")
    del config["model"]["params"]["gene_encoder_source"]
    config["model"]["regularizer"] = "gan"
    report = static_audit_conditional_wae_config(config)
    assert report["gene_encoder_source"] == "linear"


def test_static_contract_z_noise_std_defaults_to_zero_and_is_reported():
    config = _geneencoder_config("wae_he_gan_control", encoder_conditioning="none")
    config["model"]["regularizer"] = "gan"
    report = static_audit_conditional_wae_config(config)
    assert report["z_noise_std"] == 0.0


def test_static_contract_rejects_a_negative_z_noise_std():
    config = _geneencoder_config("wae_he_gan_control", encoder_conditioning="none")
    config["model"]["regularizer"] = "gan"
    config["model"]["params"]["z_noise_std"] = -0.1
    with pytest.raises(ValueError, match="z_noise_std"):
        static_audit_conditional_wae_config(config)


def test_static_contract_accepts_a_positive_z_noise_std():
    config = _geneencoder_config("wae_he_gan_control", encoder_conditioning="none")
    config["model"]["regularizer"] = "gan"
    config["model"]["params"]["z_noise_std"] = 0.5
    report = static_audit_conditional_wae_config(config)
    assert report["z_noise_std"] == 0.5


def test_build_model_wires_z_noise_std_from_config():
    config = _geneencoder_config(
        "wae_he_gan_control", encoder_conditioning="none", gene_encoder_source="linear",
    )
    config["model"]["regularizer"] = "gan"
    config["model"]["params"]["n_heads"] = 4
    config["model"]["params"]["n_blocks"] = 1
    config["model"]["params"]["dense_threshold"] = 20
    config["model"]["params"]["sparse_k"] = 3
    config["model"]["params"]["discriminator_hidden_dim"] = 12
    config["model"]["params"]["n_inference_samples"] = 3
    config["model"]["params"]["z_noise_std"] = 0.5
    model = train_conditional_wae._build_model(config, n_genes=7)
    assert model.z_noise_std == 0.5


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
        "point_prediction": target + 1,
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
    monkeypatch.setattr(logger, "_add_latent_scatter", lambda *_args: None)
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
        "validation/point_total", "validation/point_rmse",
    }
    assert len(writer.embeddings) == 4
    assert sum("label_img" in kwargs for _args, kwargs in writer.embeddings) == 2


def test_pca_2d_produces_two_columns_and_is_deterministic():
    from gen3_multiscale.conditional_wae.tensorboard import _pca_2d

    rng = np.random.default_rng(0)
    values = rng.normal(size=(20, 64)).astype(np.float32)
    coords_a = _pca_2d(values)
    coords_b = _pca_2d(values)
    assert coords_a.shape == (20, 2)
    np.testing.assert_array_equal(coords_a, coords_b)


def test_pca_2d_rejects_non_finite_input():
    from gen3_multiscale.conditional_wae.tensorboard import _pca_2d

    values = np.zeros((3, 4), dtype=np.float32)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _pca_2d(values)


def test_add_latent_scatter_writes_one_figure_tagged_by_organ_pca(tmp_path):
    class FakeFigureWriter:
        def __init__(self):
            self.figures = []

        def add_figure(self, tag, fig, step, close=True):
            self.figures.append((tag, step))

    writer = FakeFigureWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    metadata = [
        ["s0", "p0", "kidney", "medium", "fp", "b0", "0", "0", "True", "query"],
        ["s0", "p0", "kidney", "medium", "fp", "b1", "1", "0", "True", "query"],
        ["s1", "p1", "liver", "medium", "fp", "b2", "0", "1", "True", "query"],
    ]
    arrays = {"posterior_z": np.random.default_rng(1).normal(size=(3, 5)).astype(np.float32)}
    logger._add_latent_scatter(10, metadata, arrays)
    assert writer.figures == [("embeddings/posterior_z_pca_2d", 10)]


def _accumulation_optimizer(model):
    generator_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("discriminator.")
    ]
    parameter_groups = [{"params": generator_parameters, "name": "generator"}]
    if model.discriminator is not None:
        parameter_groups.append(
            {"params": list(model.discriminator.parameters()), "name": "discriminator"}
        )
    optimizer = torch.optim.AdamW(parameter_groups, lr=1e-3, weight_decay=0.0)
    return optimizer, generator_parameters


def test_generator_accumulation_takes_one_step_and_averages_not_sums_gradients():
    model = _model("mmd")
    optimizer, generator_parameters = _accumulation_optimizer(model)
    inputs, target = _inputs(), torch.randn(12, 7)

    reference = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(5 + 2 * 3 + 1),
    )
    reference["total"].backward()
    reference_grad = next(model.image_conditioner.blocks[0].parameters()).grad.clone()
    model.zero_grad(set_to_none=True)

    micro_batches = [(3, inputs, target)] * 4
    _accumulated_losses, grad_norm = train_conditional_wae.accumulate_and_step_generator(
        model, optimizer, generator_parameters, micro_batches,
        gradient_accumulation_steps=4, clip_value=1e6, seed=5,
    )
    accumulated_grad = next(model.image_conditioner.blocks[0].parameters()).grad
    torch.testing.assert_close(accumulated_grad, reference_grad)
    assert grad_norm > 0
    parameter_steps = {
        int(optimizer.state[p]["step"]) for p in generator_parameters if p in optimizer.state
    }
    assert parameter_steps == {1}


def test_discriminator_accumulation_takes_one_step_independent_of_generator():
    model = _model("gan")
    optimizer, generator_parameters = _accumulation_optimizer(model)
    target = torch.randn(12, 7)

    reference_loss = model.compute_discriminator_loss(
        target, generator=torch.Generator().manual_seed(9 + 2 * 2),
    )
    reference_loss.backward()
    reference_grad = next(model.discriminator.parameters()).grad.clone()
    model.zero_grad(set_to_none=True)

    micro_batches = [(2, None, target)] * 5
    loss_value, grad_norm = train_conditional_wae.accumulate_and_step_discriminator(
        model, optimizer, micro_batches,
        gradient_accumulation_steps=5, clip_value=1e6, seed=9,
    )
    accumulated_grad = next(model.discriminator.parameters()).grad
    torch.testing.assert_close(accumulated_grad, reference_grad)
    assert grad_norm > 0
    assert loss_value == pytest.approx(float(reference_loss.detach()))
    discriminator_params = list(model.discriminator.parameters())
    assert {
        int(optimizer.state[p]["step"]) for p in discriminator_params if p in optimizer.state
    } == {1}
    assert all(p not in optimizer.state for p in generator_parameters)


def test_generator_accumulation_still_enforces_query_gex_leakage_contract():
    model = _model("gan")
    optimizer, generator_parameters = _accumulation_optimizer(model)
    base = _inputs()
    query = np.ones(12, dtype=bool)
    leaking = FullImageExpressionInputs(
        base.sample_id, base.image_features, base.coords, base.image_available,
        query, np.ones((12, 7), dtype=np.float32), np.arange(12), np.ones(12, dtype=bool),
    )
    target = torch.randn(12, 7)
    with pytest.raises(ValueError, match="target-expression leakage"):
        train_conditional_wae.accumulate_and_step_generator(
            model, optimizer, generator_parameters, [(0, leaking, target)],
            gradient_accumulation_steps=1, clip_value=1.0, seed=0,
        )


def test_accumulation_functions_reject_non_positive_steps_and_empty_batches():
    model = _model("gan")
    optimizer, generator_parameters = _accumulation_optimizer(model)
    inputs, target = _inputs(), torch.randn(12, 7)
    micro_batches = [(0, inputs, target)]
    for kwargs in (
        {"gradient_accumulation_steps": 0, "clip_value": 1.0, "seed": 0},
    ):
        with pytest.raises(ValueError, match="positive integer"):
            train_conditional_wae.accumulate_and_step_generator(
                model, optimizer, generator_parameters, micro_batches, **kwargs,
            )
        with pytest.raises(ValueError, match="positive integer"):
            train_conditional_wae.accumulate_and_step_discriminator(
                model, optimizer, micro_batches, **kwargs,
            )
    with pytest.raises(ValueError, match="non-empty"):
        train_conditional_wae.accumulate_and_step_generator(
            model, optimizer, generator_parameters, [],
            gradient_accumulation_steps=1, clip_value=1.0, seed=0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        train_conditional_wae.accumulate_and_step_discriminator(
            model, optimizer, [], gradient_accumulation_steps=1, clip_value=1.0, seed=0,
        )


def test_compact_dimensions_reach_the_constructed_layers():
    config = {
        "model": {
            "arm": "wae_he_gan_small", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "gan",
            "include_observed_gex": False, "image_mode": "full_visible",
            "params": {
                "image_feature_dim": 16, "gex_feature_dim": 6,
                "hidden_dim": 256, "n_heads": 4, "n_blocks": 2,
                "dense_threshold": 400, "sparse_k": 32, "dropout": 0.1,
                "latent_dim": 128, "autoencoder_hidden_dim": 512,
                "discriminator_hidden_dim": 256, "n_inference_samples": 8,
            },
        },
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }
    model = train_conditional_wae._build_model(config, n_genes=7)
    assert model.latent_dim == 128
    assert model.image_conditioner.hidden_dim == 256
    assert len(model.image_conditioner.blocks) == 2
    assert model.expression_encoder.net[0].out_features == 512
    assert model.expression_encoder.net[-1].out_features == 128
    assert model.residual_decoder[0].in_features == 256 + 128
    assert model.residual_decoder[0].out_features == 512
    assert model.discriminator[0].in_features == 128
    assert model.discriminator[0].out_features == 256
    inputs, target = _inputs(n=12, image_dim=16), torch.randn(12, 7)
    losses = model.compute_generator_losses(
        inputs, target, generator=torch.Generator().manual_seed(0),
    )
    assert losses["latent"].shape == (12, 128)


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


def _whole_slide_sample(n=15, genes=7, image_dim=16, seed=0, sample_id="slide"):
    rng = np.random.default_rng(seed)
    coords = np.stack([np.arange(n), (np.arange(n) * 5) % 4], axis=1).astype(np.float64)
    return SimpleNamespace(
        sample_id=sample_id,
        precomputed_spot_features=rng.normal(size=(n, image_dim)).astype(np.float32),
        image_source_available=np.ones(n, dtype=bool),
        full_sample_coords=coords,
        adata=SimpleNamespace(
            X=rng.normal(size=(n, genes)).astype(np.float32),
            var_names=[f"g{i}" for i in range(genes)],
        ),
    )


def test_whole_slide_matches_masked_inference_with_active_spatial_refinement():
    sample = _whole_slide_sample(n=15, seed=11)
    inputs, _target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    model = _refining_model(2)
    # Make refinement genuinely non-identity so this catches the historical
    # whole-slide path that silently skipped it.
    with torch.no_grad():
        for parameter in model.spatial_refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    model.eval()
    expected = model.sample_predictive_distribution(
        inputs, n_samples=4, generator=torch.Generator().manual_seed(19),
    )
    actual = predict_whole_slide(
        model, sample, chunk_size=4, n_samples=4, seed=19,
    )
    torch.testing.assert_close(actual["point_prediction"], expected["point_prediction"])
    torch.testing.assert_close(actual["predictive_mean"], expected["predictive_mean"])
    torch.testing.assert_close(actual["predictive_std"], expected["predictive_std"])


def test_whole_slide_output_is_independent_of_decoder_chunk_size():
    sample = _whole_slide_sample(n=15, seed=12)
    model = _refining_model(2).eval()
    small = predict_whole_slide(model, sample, chunk_size=3, n_samples=3, seed=7)
    large = predict_whole_slide(model, sample, chunk_size=100, n_samples=3, seed=7)
    for key in ("point_prediction", "predictive_mean", "predictive_std"):
        torch.testing.assert_close(small[key], large[key])


def test_aggregate_whole_slide_metrics_averages_across_slides_ignoring_nan_auc():
    per_slide_metrics = [
        {"per_arm": {"model": {"all_genes": {"pcc": 0.2, "rmse": 1.0, "auc": 0.6}}}},
        {"per_arm": {"model": {"all_genes": {"pcc": 0.4, "rmse": 2.0, "auc": float("nan")}}}},
    ]
    aggregated = train_conditional_wae._aggregate_whole_slide_metrics(per_slide_metrics)
    assert aggregated["model"]["all_genes"]["pcc"] == pytest.approx(0.3)
    assert aggregated["model"]["all_genes"]["rmse"] == pytest.approx(1.5)
    assert aggregated["model"]["all_genes"]["auc"] == pytest.approx(0.6)


def test_aggregate_whole_slide_metrics_rejects_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        train_conditional_wae._aggregate_whole_slide_metrics([])


def test_select_whole_slide_sample_ids_is_bounded_and_deterministic():
    validation_ids = ["S9", "S1", "S5", "S2", "S8", "S3"]
    selected = train_conditional_wae._select_whole_slide_sample_ids(validation_ids, 2)
    assert selected == ["S1", "S2"]  # sorted, then capped -- never grows with more slides
    assert train_conditional_wae._select_whole_slide_sample_ids(validation_ids, 100) == sorted(validation_ids)


def test_select_whole_slide_sample_ids_rejects_non_positive_max_slides():
    with pytest.raises(ValueError, match="max_slides"):
        train_conditional_wae._select_whole_slide_sample_ids(["S0"], 0)


def test_select_whole_slide_sample_ids_round_robins_across_organs():
    validation_ids = ["K0", "K1", "L0", "B0", "B1", "B2"]
    organ_by_sample = {
        "K0": "Kidney", "K1": "Kidney", "L0": "Liver",
        "B0": "Bowel", "B1": "Bowel", "B2": "Bowel",
    }
    # A small max_slides still covers every organ present, not whichever
    # organ happens to sort first alphabetically overall.
    selected = train_conditional_wae._select_whole_slide_sample_ids(
        validation_ids, 3, organ_by_sample=organ_by_sample,
    )
    assert {organ_by_sample[sid] for sid in selected} == {"Kidney", "Liver", "Bowel"}
    assert selected == ["B0", "K0", "L0"]  # one per organ, alphabetical within/across


def test_select_whole_slide_sample_ids_fills_a_second_round_after_every_organ_has_one():
    validation_ids = ["K0", "K1", "L0"]
    organ_by_sample = {"K0": "Kidney", "K1": "Kidney", "L0": "Liver"}
    selected = train_conditional_wae._select_whole_slide_sample_ids(
        validation_ids, 3, organ_by_sample=organ_by_sample,
    )
    assert selected == ["K0", "L0", "K1"]  # round 1: one per organ; round 2: Kidney's second


def test_select_whole_slide_sample_ids_without_organ_info_keeps_old_alphabetical_behavior():
    validation_ids = ["S9", "S1", "S5"]
    assert train_conditional_wae._select_whole_slide_sample_ids(validation_ids, 2) == ["S1", "S5"]


def test_organ_gene_indices_uses_the_organs_own_dispersion_panel():
    gene_names = ["ALB", "G1", "G2"]
    artifact = {
        "panels": {"train_log1p_variance_top50": ["ALB", "G1", "G2"]},
        "panels_by_organ": {
            "Kidney": {"train_dispersion_top1": ["G1"], "train_dispersion_top5": ["G1", "G2", "ALB"]},
            "Liver": {"train_dispersion_top1": ["ALB"], "train_dispersion_top5": ["ALB", "G1", "G2"]},
        },
    }
    kidney_indices = train_conditional_wae._organ_gene_indices(artifact, "Kidney", gene_names, 2)
    assert kidney_indices == [1, 2]  # G1, G2 -- not ALB
    liver_indices = train_conditional_wae._organ_gene_indices(artifact, "Liver", gene_names, 1)
    assert liver_indices == [0]  # ALB


def test_organ_gene_indices_falls_back_to_pooled_panel_for_an_unlisted_organ():
    gene_names = ["ALB", "G1"]
    artifact = {
        "panels": {"train_log1p_variance_top50": ["ALB", "G1"]},
        "panels_by_organ": {"Kidney": {"train_dispersion_top1": ["G1"]}},
    }
    assert train_conditional_wae._organ_gene_indices(artifact, "Lung", gene_names, 2) == [0, 1]


def test_organ_gene_indices_falls_back_to_identity_when_artifact_is_none():
    assert train_conditional_wae._organ_gene_indices(None, "Kidney", ["G0", "G1", "G2"], 2) == [0, 1]


def test_film_gamma_beta_genuinely_affect_the_encoder_after_training_moves_the_weights():
    """Distinguishes real FiLM conditioning from a no-op: at init (gamma=1,
    beta=0) two different contexts must give IDENTICAL output (see
    test_film_encoder_output_is_identical_across_contexts_at_init); once the
    generators' weights are non-zero, two different contexts must give
    DIFFERENT output for the SAME expression."""
    encoder = FiLMConditionedExpressionEncoder(7, context_dim=24, latent_dim=5, hidden_dim=20)
    with torch.no_grad():
        for film in (encoder.film_first, encoder.film_second):
            film.to_gamma_beta.weight.normal_(mean=0.0, std=1.0)
            film.to_gamma_beta.bias.normal_(mean=0.0, std=1.0)
    expression = torch.randn(6, 7)
    context_a = torch.randn(6, 24)
    context_b = torch.randn(6, 24) * 5.0
    output_a = encoder(expression, context_a)
    output_b = encoder(expression, context_b)
    assert not torch.allclose(output_a, output_b)


def test_run_whole_slide_validation_computes_metrics_and_a_comparable_total():
    genes = [f"g{i}" for i in range(7)]
    model = _model("mmd")
    samples = {"S0": _whole_slide_sample(n=13, seed=1, sample_id="S0"),
               "S1": _whole_slide_sample(n=9, seed=2, sample_id="S1")}
    (
        aggregated, total, rmse, pcc_loss, hvg50_pcc_loss, per_slide_predictions,
    ) = train_conditional_wae._run_whole_slide_validation(
        model, samples, ["S0", "S1"], genes, {"panelA": ["g0", "g1"]},
        chunk_size=4, n_samples=2, seed=0, device=torch.device("cpu"),
    )
    for arm in ("model", "conditional_mean"):
        assert set(aggregated[arm]) == {"all_genes", "panelA"}
        for panel in aggregated[arm].values():
            assert set(panel) == {"pcc", "rmse", "auc"}
    assert np.isfinite(total)
    assert np.isfinite(rmse)
    assert np.isfinite(pcc_loss)
    # No "train_log1p_variance_top50" key in this test's panels dict ->
    # _panel_gene_indices finds nothing -> falls back to the full-panel
    # pcc_loss verbatim (never silently 0/NaN).
    assert hvg50_pcc_loss == pcc_loss
    assert len(per_slide_predictions) == 2
    assert per_slide_predictions[0]["sample_id"] == "S0"


def test_run_whole_slide_validation_hvg50_pcc_loss_uses_only_the_top50_panel_columns():
    genes = [f"g{i}" for i in range(7)]
    model = _model("mmd")
    samples = {"S0": _whole_slide_sample(n=13, seed=1, sample_id="S0")}
    top50_genes = ["g1", "g3"]
    (
        _aggregated, _total, _rmse, pcc_loss, hvg50_pcc_loss, _per_slide_predictions,
    ) = train_conditional_wae._run_whole_slide_validation(
        model, samples, ["S0"], genes,
        {"panelA": ["g0", "g1"], "train_log1p_variance_top50": top50_genes},
        chunk_size=4, n_samples=2, seed=0, device=torch.device("cpu"),
    )
    assert np.isfinite(hvg50_pcc_loss)
    # A real, distinct computation over a 2-gene subset -- not silently
    # aliased to the full 7-gene pcc_loss (would be a near-impossible
    # coincidence with random model weights/data).
    assert hvg50_pcc_loss != pcc_loss


def test_panel_gene_indices_maps_panel_gene_names_to_positions():
    gene_names = ["g0", "g1", "g2", "g3"]
    assert train_conditional_wae._panel_gene_indices(
        {"train_log1p_variance_top50": ["g2", "g0", "gMISSING"]}, "train_log1p_variance_top50", gene_names,
    ) == [2, 0]
    assert train_conditional_wae._panel_gene_indices({}, "train_log1p_variance_top50", gene_names) == []
