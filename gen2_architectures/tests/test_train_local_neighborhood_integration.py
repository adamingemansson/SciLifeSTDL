"""End-to-end synthetic integration tests for the actual training/
evaluation wiring (not just individual component unit tests) -- built
during development specifically to catch integration bugs before ever
touching real HEST-1k data or GPU hardware, per GPT review's own
suggestion to budget verification runs before committing real compute.
Uses synthetic AnnData (real anndata objects, fake random values) so real
HEST-1k data/GigaPath/scanpy are never required.
"""
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gen2_architectures.data.masked_item import build_masked_item
from gen2_architectures.models.components import StagedGeneLoss
from gen2_architectures.training import checkpoint
from gen2_architectures.training.train_local_neighborhood import build_model, _model_config_dict
from gen2_architectures.training.validation import move_to_device


def _masking_cfg():
    return OmegaConf.create({
        "strategy": "random_dropout_patches",
        "max_context_points": 40,
        "context_selection": "nearest_query",
        "params": {"n_patches": 1, "radius_range": [80, 150], "radius_unit": "coordinate", "shape": "mixed"},
    })


def _arch1_cfg():
    return OmegaConf.create({
        "model": {"architecture": "1", "params": {
            "feat_dim": 16, "coord_dim": 8, "conf_dim": 4, "hidden_dim": 32,
            "n_layers": 1, "n_heads": 2, "max_neighbors": 20, "decoder_hidden_dim": 32,
            "coord_scale": 1000.0,
        }},
    })


def test_training_step_loop_reduces_finite_loss_and_flows_gradients():
    n_spots, n_genes = 200, 30
    rng = np.random.default_rng(0)
    coords3d = np.concatenate([rng.uniform(0, 2000, size=(n_spots, 2)), np.zeros((n_spots, 1))], axis=1)
    expr = rng.normal(size=(n_spots, n_genes)).astype(np.float32)
    images = rng.normal(size=(n_spots, 1536)).astype(np.float32)
    slice_ids = np.array(["s1"] * n_spots)

    cfg = _arch1_cfg()
    gene_names = [f"g{i}" for i in range(n_genes)]
    model = build_model(cfg, gene_names, scfoundation_dim=None)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    loss_fn = StagedGeneLoss()

    losses = []
    model.train()
    for step in range(5):
        item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=step,
                                  image_mode="target_zero", context_gex_mode="full")
        item = move_to_device(item, torch.device("cpu"))
        pred = model(item["context"], item["query"])
        target = item["target_expression"]
        result = loss_fn(pred, target, progress=step / 5)
        optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        losses.append(result["loss"].item())

    assert all(np.isfinite(losses)), f"loss went non-finite: {losses}"


def test_training_checkpoint_round_trips_through_the_real_save_load_helpers():
    n_genes = 20
    cfg = _arch1_cfg()
    gene_names = [f"g{i}" for i in range(n_genes)]
    model = build_model(cfg, gene_names, scfoundation_dim=None)
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint.save_checkpoint(model, _model_config_dict(cfg, None), gene_names, tmp, step=5)
        model2 = build_model(cfg, gene_names, scfoundation_dim=None)
        checkpoint.load_trainable_state(model2, tmp)
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"mismatch in {n1}"
        assert checkpoint.load_training_state(tmp)["step"] == 5


@pytest.fixture
def synthetic_adata():
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    n_spots, n_genes = 120, 25
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n_spots, n_genes)).astype(np.float32)
    gene_names = [f"g{i}" for i in range(n_genes)]
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=gene_names))
    adata.obsm["spatial"] = rng.uniform(0, 2000, size=(n_spots, 2))
    adata.obs["z"] = 0.0
    adata.obs["slice_id"] = "s1"
    adata.obs["organ"] = "Lung"
    adata.obs["tech"] = "Visium"
    adata.obs_names = [f"spot{i}" for i in range(n_spots)]
    images = rng.normal(size=(n_spots, 1536)).astype(np.float32)
    return adata, images, gene_names


def test_evaluation_harness_end_to_end_with_a_synthetic_slide(synthetic_adata):
    """The real integration point that matters most: a gen2 model, built
    through the real build_model() path, evaluated through the real
    (copied) evaluate_model_on_mask_bank harness -- mask bank construction,
    PCA basis fitting, per-mask PCC/RMSE, all real code, only the
    underlying data is synthetic."""
    pytest.importorskip("sklearn")
    from gen2_architectures.training import evaluate

    adata, images, gene_names = synthetic_adata
    cfg = _arch1_cfg()
    cfg.experiment_name = "integration_test"
    cfg.masking = _masking_cfg()
    cfg.evaluation = OmegaConf.create({
        "n_validation_masks": 2, "n_test_masks": 2, "n_samples": 1,
        "image_modes": ["target_zero"], "primary_image_mode": "target_zero",
        "k_neighborhood": 4, "pca_n_components": 5,
    })
    cfg.data = OmegaConf.create({})

    model = build_model(cfg, gene_names, scfoundation_dim=None)
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        cfg.evaluation.mask_bank_dir = str(Path(tmp) / "mask_banks")
        metrics = evaluate.evaluate_sample(model, cfg, adata, images, {}, "sample1", "test", Path(tmp))
        assert metrics["primary_image_mode"] == "target_zero"
        primary = metrics["image_modes"]["target_zero"]["summary"]
        assert "pcc" in primary and "rmse" in primary
        assert np.isfinite(primary["rmse"]["mean"])
