"""Integration test for the standalone offline evaluation entrypoint --
build a real checkpoint, then run run_held_out_evaluation.main() against
it exactly as it would be invoked on the real server, with only the
(expensive, real-data-only) sample loading swapped for synthetic data."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from gen2_architectures.training import checkpoint, run_held_out_evaluation
from gen2_architectures.training.train_local_neighborhood import build_model, _model_config_dict


@pytest.fixture
def synthetic_adata():
    ad = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    n_spots, n_genes = 120, 20
    rng = np.random.default_rng(2)
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


def _arch1_cfg():
    return OmegaConf.create({
        "model": {"architecture": "1", "params": {
            "feat_dim": 16, "coord_dim": 8, "conf_dim": 4, "hidden_dim": 32,
            "n_layers": 1, "n_heads": 2, "max_neighbors": 20, "decoder_hidden_dim": 32,
            "coord_scale": 1000.0,
        }},
    })


def test_run_held_out_evaluation_rebuilds_model_from_checkpoint_and_scores_test_ids(
    monkeypatch, synthetic_adata,
):
    pytest.importorskip("sklearn")
    adata, images, gene_names = synthetic_adata
    cfg = _arch1_cfg()
    model = build_model(cfg, gene_names, scfoundation_dim=None)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "ckpt"
        checkpoint.save_checkpoint(model, _model_config_dict(cfg, None), gene_names, checkpoint_dir, step=42)

        config_path = Path(tmp) / "config.yaml"
        full_cfg = OmegaConf.create({
            "experiment_name": "standalone_eval_test",
            "data": {"test_sample_ids": ["sample1"]},
            "masking": {
                "strategy": "random_dropout_patches", "max_context_points": 40,
                "context_selection": "nearest_query",
                "params": {"n_patches": 1, "radius_range": [80, 150], "radius_unit": "coordinate", "shape": "mixed"},
            },
            "training": {"checkpoint_dir": str(checkpoint_dir), "device": "cpu"},
            "evaluation": {
                "mask_bank_dir": str(Path(tmp) / "mask_banks"),
                "n_validation_masks": 2, "n_test_masks": 2, "n_samples": 1,
                "image_modes": ["target_zero"], "primary_image_mode": "target_zero",
                "k_neighborhood": 4, "pca_n_components": 5,
            },
        })
        OmegaConf.save(full_cfg, config_path)

        monkeypatch.setattr(run_held_out_evaluation.data_prep, "apply_sample_selection", lambda cfg: None)
        monkeypatch.setattr(
            run_held_out_evaluation.data_prep, "load_held_out_samples_with_images",
            lambda cfg, ids, gene_names: (list(ids), [adata] * len(ids), [images] * len(ids)),
        )

        run_held_out_evaluation.main(str(config_path))


def test_missing_checkpoint_raises_a_clear_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    full_cfg = OmegaConf.create({
        "data": {"test_sample_ids": ["sample1"]},
        "training": {"checkpoint_dir": str(tmp_path / "no_such_checkpoint")},
    })
    OmegaConf.save(full_cfg, config_path)
    with pytest.raises(FileNotFoundError, match="no saved checkpoint"):
        run_held_out_evaluation.main(str(config_path))
