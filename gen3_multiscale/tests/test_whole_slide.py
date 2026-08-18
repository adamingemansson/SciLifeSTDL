from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_wae import Architecture1ImageConditioner, ConditionalWAE
from gen3_multiscale.conditional_wae.model import DeterministicSpatialPredictor
from gen3_multiscale.conditional_wae.structured_field import (
    fit_centered_organ_balanced_gene_structure,
)
from gen3_multiscale.conditional_wae.data import build_conditional_wae_example
from gen3_multiscale.conditional_wae.whole_slide import (
    predict_deterministic_refinement_stages,
    predict_whole_slide,
    whole_slide_metrics,
)


def _sample(n=17, genes=7, image_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.stack([np.arange(n), (np.arange(n) * 7) % 5], axis=1).astype(np.float64)
    return SimpleNamespace(
        sample_id="slide",
        precomputed_spot_features=rng.normal(size=(n, image_dim)).astype(np.float32),
        image_source_available=np.ones(n, dtype=bool),
        full_sample_coords=coords,
        adata=SimpleNamespace(
            X=rng.normal(size=(n, genes)).astype(np.float32),
            var_names=[f"g{i}" for i in range(genes)],
        ),
    )


def _model(genes=7, image_dim=16):
    return ConditionalWAE(
        genes,
        Architecture1ImageConditioner(
            genes, image_feature_dim=image_dim, gex_feature_dim=6,
            hidden_dim=24, n_heads=4, n_blocks=1,
            dense_threshold=5, sparse_k=3, dropout=0.0,
        ),
        regularizer="mmd", latent_dim=5, autoencoder_hidden_dim=20,
        n_inference_samples=4,
    )


def test_chunked_whole_slide_prediction_equals_unchunked():
    torch.manual_seed(0)
    model = _model()
    sample = _sample(n=17)
    unchunked = predict_whole_slide(model, sample, chunk_size=1000, n_samples=3, seed=7)
    chunked = predict_whole_slide(model, sample, chunk_size=4, n_samples=3, seed=7)
    torch.testing.assert_close(unchunked["predictive_mean"], chunked["predictive_mean"])
    torch.testing.assert_close(unchunked["predictive_std"], chunked["predictive_std"])
    torch.testing.assert_close(
        unchunked["conditional_mean_expression"], chunked["conditional_mean_expression"],
    )


def test_every_tissue_spot_is_predicted_exactly_once():
    model = _model()
    sample = _sample(n=17)
    result = predict_whole_slide(model, sample, chunk_size=5, n_samples=2, seed=0)
    assert result["predictive_mean"].shape[0] == 17
    assert result["n_spots"] == 17
    assert result["target"].shape == (17, 7)
    assert result["coords"].shape == (17, 2)


def test_whole_slide_construction_never_admits_gex_into_the_predictor():
    sample = _sample(n=9)
    inputs, target = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    assert inputs.query_mask.all()
    assert inputs.observed_expression is None
    assert inputs.expression_available is None
    np.testing.assert_array_equal(target, sample.adata.X)


def test_predict_whole_slide_rejects_non_positive_chunk_size():
    model = _model()
    sample = _sample(n=9)
    with pytest.raises(ValueError, match="chunk_size"):
        predict_whole_slide(model, sample, chunk_size=0)


def test_whole_slide_prediction_uses_the_configured_default_sample_count():
    model = _model()
    sample = _sample(n=6)
    result = predict_whole_slide(model, sample, chunk_size=3, seed=0)
    assert result["predictive_mean"].shape == (6, 7)


def test_whole_slide_metrics_reports_all_genes_and_configured_panels():
    model = _model()
    sample = _sample(n=17)
    gene_names = [f"g{i}" for i in range(7)]
    prediction = predict_whole_slide(model, sample, chunk_size=6, n_samples=3, seed=0)
    metrics = whole_slide_metrics(
        prediction, gene_names, gene_panels={"small_panel": ["g0", "g1", "g2"]},
    )
    assert metrics["n_spots"] == 17
    assert metrics["sample_id"] == "slide"
    for arm in ("model", "conditional_mean"):
        assert {"pcc", "rmse", "auc"}.issubset(
            metrics["per_arm"][arm]["all_genes"]
        )
        assert {"pcc", "rmse", "auc"}.issubset(
            metrics["per_arm"][arm]["small_panel"]
        )
        assert np.isfinite(metrics["per_arm"][arm]["all_genes"]["rmse"])


def test_whole_slide_metrics_without_panels_only_reports_all_genes():
    model = _model()
    sample = _sample(n=9)
    gene_names = [f"g{i}" for i in range(7)]
    prediction = predict_whole_slide(model, sample, chunk_size=9, n_samples=2, seed=0)
    metrics = whole_slide_metrics(prediction, gene_names)
    assert list(metrics["per_arm"]["model"]) == ["all_genes"]
    assert metrics["gene_panel_metadata"] == {}


def test_whole_slide_metrics_can_add_structured_field_diagnostics():
    model = _model()
    sample = _sample(n=17)
    gene_names = [f"g{i}" for i in range(7)]
    prediction = predict_whole_slide(model, sample, chunk_size=6, n_samples=2, seed=0)
    metrics = whole_slide_metrics(
        prediction, gene_names,
        gene_panels={"small_panel": ["g0", "g1", "g2"]},
        per_gene_scale=np.ones(7),
        structured_field_config={
            "enabled": True, "local_k": 3, "wide_k": 5,
            "nontrivial_gradient_threshold_training_sd": 0.25,
        },
    )
    structured = metrics["structured_field"]
    assert structured["scope"] == "one held_out_whole_slide_every_spot_exactly_once"
    assert set(structured["panels"]) == {"all_genes", "small_panel"}
    assert "coexpression" not in structured["panels"]["all_genes"]
    assert "coexpression" in structured["panels"]["small_panel"]


def test_structured_whole_slide_metrics_require_training_scale():
    model = _model()
    sample = _sample(n=9)
    prediction = predict_whole_slide(model, sample, chunk_size=9, n_samples=2, seed=0)
    with pytest.raises(ValueError, match="training-only per_gene_scale"):
        whole_slide_metrics(
            prediction, [f"g{i}" for i in range(7)],
            structured_field_config={"enabled": True},
        )


def test_deterministic_stage_audit_matches_each_refinement_path():
    torch.manual_seed(5)
    sample = _sample(n=9, genes=7)
    artifact = fit_centered_organ_balanced_gene_structure(
        {"slide": np.asarray(sample.adata.X, dtype=np.float32)}, ["slide"],
        {"slide": "organ"}, list(sample.adata.var_names), rank=3, seed=0,
    )
    model = DeterministicSpatialPredictor(
        7,
        Architecture1ImageConditioner(
            7, image_feature_dim=16, gex_feature_dim=6, hidden_dim=24,
            n_heads=4, n_blocks=1, dense_threshold=20, sparse_k=3, dropout=0.0,
        ),
        hidden_dim=20, gene_structure_artifact=artifact,
        n_refinement_steps=1, structured_composition="parallel_gated",
        refinement_k_neighbors=3, refinement_hidden_dim=12,
        refinement_gex_feature_dim=8, per_gene_scale=artifact.per_gene_scale,
    ).eval()
    result = predict_deterministic_refinement_stages(model, sample, chunk_size=4)
    inputs, _ = build_conditional_wae_example(
        sample, query_indices=None, include_observed_gex=False,
    )
    with torch.no_grad():
        context = model.image_conditioner(inputs)
        base = model.decode_base_from_context(context)
        expected = {
            "base": base,
            "within_only": model._apply_within(base),
            "between_only": model._apply_between(base, context, inputs),
            "full": model.refine_prediction(base, context, inputs),
        }
    assert result["n_spots"] == 9
    assert result["composition_gates"].shape == (2,)
    for stage, value in expected.items():
        torch.testing.assert_close(result["stages"][stage], value)


def test_deterministic_stage_audit_rejects_latent_models():
    with pytest.raises(ValueError, match="deterministic"):
        predict_deterministic_refinement_stages(_model(), _sample(n=9))
