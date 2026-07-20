"""
Smoke test for the full model comparison script
(src/evaluation/run_comparison.py, task #15). Builds a small synthetic
dataset on disk (matching the naming pattern load_hest_sample expects) and
tiny per-model configs, then runs main() for real end-to-end — this
orchestrates several moving parts (masking, Leiden clustering, patch-pooled
PCA, all four registry model families) worth catching bugs in cheaply
before an expensive real multi-thousand-epoch comparison run.

Run with: python -m tests.test_run_comparison
"""
import tempfile
from pathlib import Path

import numpy as np
import anndata as ad
import yaml

from src.evaluation.run_comparison import main


def _make_synthetic_hest_dir(tmp_dir: Path, n_points=200, n_genes=60, n_clusters=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1000, size=(n_clusters, 2))
    cluster_id = rng.integers(0, n_clusters, size=n_points)
    coords = centers[cluster_id] + rng.normal(scale=50, size=(n_points, 2))
    gene_means = rng.uniform(0, 3, size=(n_clusters, n_genes))
    expr = np.clip(gene_means[cluster_id] + rng.normal(scale=0.3, size=(n_points, n_genes)), 0, None)

    adata = ad.AnnData(X=expr.astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["z"] = 0.0

    hest_dir = tmp_dir / "hest1k"
    hest_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(hest_dir / "TEST_INT1.h5ad")
    return hest_dir


def _write_config(path: Path, model_name: str, model_params: dict, hest_dir: Path, epochs: int):
    cfg = {
        "experiment_name": f"smoke_{model_name}",
        "data": {
            "source": "hest1k", "hest_data_dir": str(hest_dir), "sample_id": "INT1",
            "task": "intra_slice", "min_genes": 1, "min_cells": 1,
        },
        "masking": {
            "strategy": "random_dropout_patches",
            "params": {"n_patches": 2, "radius_range": [30, 60]},
        },
        "model": {"name": model_name, "params": model_params},
        "training": {
            "epochs": epochs, "seed": 0, "log_every_n_steps": 10,
            "checkpoint_dir": str(path.parent / f"checkpoints_{model_name}"),
        },
        "validation": {"enabled": False},
        "evaluation": {
            "mask_bank_path": str(path.parent / f"mask_bank_{model_name}.json"),
            "n_validation_masks": 1, "n_test_masks": 2,
            "n_samples": 2, "image_modes": ["full"],
            "pca_n_components": 5,
        },
        "logging": {"backend": "none"},
    }
    path.write_text(yaml.safe_dump(cfg))


def test_run_comparison_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        hest_dir = _make_synthetic_hest_dir(tmp_dir)
        n_genes = 60

        configs = []
        specs = [
            ("vae_baseline", {"n_genes": n_genes, "latent_dim": 4, "hidden_dim": 16}),
            ("wae_gan", {"n_genes": n_genes, "coord_dim": 3, "latent_dim": 4,
                         "hidden_dim": 16, "cond_hidden_dim": 16, "disc_hidden_dim": 8}),
            ("fm_ot", {"n_genes": n_genes, "coord_dim": 3, "cond_hidden_dim": 16,
                       "latent_dim": 4, "ae_hidden_dim": 16, "hidden_dim": 16,
                       "time_embed_dim": 8, "n_ode_steps": 3}),
            ("vqvae_ar", {"n_genes": n_genes, "coord_dim": 3, "cond_hidden_dim": 16,
                          "latent_dim": 4, "ae_hidden_dim": 16, "codebook_size": 8,
                          "transformer_dim": 16, "n_transformer_layers": 1, "n_heads": 2,
                          "max_seq_len": 64}),
        ]
        for name, params in specs:
            path = tmp_dir / f"{name}.yaml"
            _write_config(path, name, params, hest_dir, epochs=5)
            configs.append(str(path))

        main(configs, pca_components=5)  # just needs to run without raising
        print("[run_comparison] OK — ran all 5 models (4 trained + interp_baseline) end-to-end")


if __name__ == "__main__":
    test_run_comparison_end_to_end()
    print("\nrun_comparison smoke test passed.")
