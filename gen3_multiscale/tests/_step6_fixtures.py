"""Shared Step 6 test fixtures. NOT a test file itself (no test_/_test
suffix, never collected by pytest) -- imported by test_gen3_preflight.py,
test_gen3_dataset.py, and test_train.py so all three exercise the exact
same real, small, end-to-end synthetic Gen3 experiment rather than three
independently-drifting fixture builders."""
from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

from gen3_multiscale.data import loaders
from gen3_multiscale.data.dataset_manifest import build_dataset_manifest, save_dataset_manifest
from gen3_multiscale.data.spot_feature_cache import build_gen3_spot_feature_cache

VALID_HF_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars
GIGAPATH_FEAT_DIM = 1536

STEP6_TRAIN_STRATA = [
    {"name": "small", "radius_range": [1.5, 2.5], "radius_unit": "spot_spacing", "shape": "circle"},
]


def stub_gigapath(monkeypatch, feat_dim: int = GIGAPATH_FEAT_DIM) -> None:
    """Cheap, deterministic GigaPath stand-in -- same pattern as
    test_spot_feature_cache.py's _stub_gigapath. Real
    gigapath_tile_encoder_provenance is left unstubbed (only needs a
    real state_dict(), which nn.Linear provides); `timm` itself is
    injected into sys.modules since it is not installed in this sandbox."""
    def _fake_load(revision=None):
        assert revision is not None
        # Deterministic weights (never randomly initialized) -- the real
        # system's whole point is that loading the SAME pinned revision
        # twice (once for a dense-WSI cache, once for a spot-feature
        # cache) produces byte-identical weights; nn.Linear's default
        # random init would falsely make two stubbed loads disagree.
        encoder = nn.Linear(4, 4)
        with torch.no_grad():
            encoder.weight.fill_(0.5)
            encoder.bias.fill_(0.0)
        return encoder

    def _fake_encode(tile_encoder, tensor):
        means = tensor.mean(dim=(1, 2, 3)).view(tensor.shape[0], 1)
        return means.expand(tensor.shape[0], feat_dim).clone()

    fake_timm = types.ModuleType("timm")
    # Some Torch versions ask importlib for timm's spec while AdamW
    # initializes torch._dynamo. A module stub without a spec makes that
    # standard package-presence check raise instead of returning True.
    fake_timm.__spec__ = ModuleSpec("timm", loader=None)
    fake_timm.__version__ = "0.0.0-test-stub"
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    monkeypatch.setattr("src.models.conditioning._load_gigapath_tile_encoder", _fake_load)
    monkeypatch.setattr("src.models.conditioning._gigapath_preprocess_and_encode", _fake_encode)


def build_synthetic_gen3_experiment(
    tmp_path: Path,
    monkeypatch,
    *,
    n_side: int = 6,
    spacing: float = 300.0,
    n_genes: int = 6,
    samples_per_split: dict[str, int] | None = None,
    organ: str = "Lung",
    tile_encoder_revision: str = VALID_HF_REVISION,
) -> tuple["OmegaConf", dict]:
    """Build a real, small, end-to-end Gen3 experiment on disk: real
    HEST-1k-shaped h5ad/patch files on a genuine 2D grid (so
    radius-based masking strata produce real, non-degenerate holes), a
    real dataset manifest (dataset_manifest.build_dataset_manifest), and
    real dense-WSI + spot-feature caches (GigaPath itself monkeypatched)
    sharing ONE pinned tile-encoder revision. Returns (cfg, manifest)."""
    samples_per_split = samples_per_split or {"train": 3, "validation": 1, "test": 1}
    n_total = sum(samples_per_split.values())
    sample_ids = [f"S{i}" for i in range(n_total)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    hest_dir = tmp_path / "hest1k"
    (hest_dir / "st").mkdir(parents=True, exist_ok=True)
    (hest_dir / "patches").mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    (cache_dir / "gigapath_slide_cache").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    coords_by_sample: dict[str, np.ndarray] = {}
    for sid in sample_ids:
        barcodes = [f"{sid}-SPOT{i}-1" for i in range(n_side * n_side)]
        coords = np.array(
            [[x * spacing, y * spacing] for x in range(n_side) for y in range(n_side)], dtype=np.float64,
        )
        coords_by_sample[sid] = coords
        counts = rng.poisson(5, size=(len(barcodes), n_genes)).astype(np.float32)
        adata = ad.AnnData(
            X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)),
        )
        adata.obsm["spatial"] = coords
        adata.write_h5ad(hest_dir / "st" / f"{sid}.h5ad")
        with h5py.File(hest_dir / "patches" / f"{sid}.h5", "w") as f:
            f.create_dataset("img", data=np.full((len(barcodes), 4, 4, 3), 100, dtype=np.uint8))
            f.create_dataset("barcode", data=np.array([[b.encode()] for b in barcodes]))
        rows.append({
            "id": sid, "organ": organ, "st_technology": "Visium", "species": "Homo sapiens",
            "nb_genes": n_genes, "patient": sid,
        })
    meta_path = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    manifest = build_dataset_manifest(
        hest_dir, str(meta_path), organs="all",
        min_samples_per_organ=n_total, max_samples_per_organ=n_total,
        n_validation_per_organ=samples_per_split["validation"], n_test_per_organ=samples_per_split["test"],
        split_seed=0, check_gene_panel_compatibility=False, min_nb_genes=None,
        gene_min_genes_per_spot=1, gene_min_cells=0, hest_cache_dir=str(cache_dir),
    )

    stub_gigapath(monkeypatch)
    cfg = OmegaConf.create({
        "data": {
            "hest_data_dir": str(hest_dir), "hest_cache_dir": str(cache_dir),
            "slide_context_source": "dense_wsi_cache",
        },
    })

    from src.models.conditioning import _load_gigapath_tile_encoder, gigapath_tile_encoder_provenance

    encoder = _load_gigapath_tile_encoder(revision=tile_encoder_revision)
    provenance = gigapath_tile_encoder_provenance(encoder, revision=tile_encoder_revision)

    # A real dense-WSI tile grid (256px, HEST-1k's own convention) is
    # independent of spot spacing -- one tile per spot position here is
    # a synthetic simplification (matches
    # scripts/smoke_test_gigapath_slide_encoder.py's own regularly-
    # gridded synthetic cache), but the tile SIZE must stay realistic
    # relative to a 224px query patch so a query hole physically removes
    # only its own tile, not the entire visible tile set.
    dense_tile_size = 256.0
    for sid in sample_ids:
        coords = coords_by_sample[sid]
        features = rng.normal(size=(coords.shape[0], GIGAPATH_FEAT_DIM)).astype(np.float32)
        np.savez(
            cache_dir / "gigapath_slide_cache" / f"{sid}.npz",
            features=features, coords=coords.astype(np.float32), level0_coords=coords.astype(np.float32),
            tile_size=np.asarray(dense_tile_size, dtype=np.float32),
            level0_tile_size=np.asarray(dense_tile_size, dtype=np.float32),
            coords_are_centers=np.asarray(True),
            tile_encoder_hf_repo_id=np.asarray(provenance["hf_repo_id"]),
            tile_encoder_hf_revision=np.asarray(provenance["hf_revision"]),
            tile_encoder_timm_version=np.asarray(str(provenance["timm_version"])),
            tile_encoder_preprocessing_spec=np.asarray(provenance["preprocessing_spec"]),
            tile_encoder_state_dict_sha256=np.asarray(provenance["state_dict_sha256"]),
            tile_encoder_schema_version=np.asarray(provenance["schema_version"]),
        )
        patches, patch_barcodes = loaders.load_hest_patches(hest_dir, sid)
        raw_adata = ad.read_h5ad(hest_dir / "st" / f"{sid}.h5ad")
        _, aligned_patches, image_source_available = loaders.align_patches_to_adata(
            raw_adata, patches, patch_barcodes,
        )
        build_gen3_spot_feature_cache(
            cfg, sid, np.asarray(raw_adata.obs_names), aligned_patches, image_source_available,
            tile_encoder_revision=tile_encoder_revision, device="cpu",
        )

    return cfg, manifest


def prepare_step6_experiment(tmp_path: Path, monkeypatch, **experiment_kwargs) -> tuple["OmegaConf", dict, Path]:
    """build_synthetic_gen3_experiment + persist the manifest to disk (the
    real trainer/scripts always read a manifest from a path, never an
    in-memory dict) -- shared by test_train.py and test_step6_scripts.py."""
    cfg, manifest = build_synthetic_gen3_experiment(tmp_path, monkeypatch, **experiment_kwargs)
    manifest_path = tmp_path / "manifest.json"
    save_dataset_manifest(manifest, manifest_path)
    return cfg, manifest, manifest_path


def step6_model_params(architecture: str, **overrides) -> dict:
    """Small, CPU-fast model.params for a given architecture -- real
    architecture-specific constructor requirements (e.g. Architecture4 has
    no use_anchor_blend/use_global_gex; see models/architectures.py::
    Architecture4.__init__) are already accounted for."""
    params = {
        "image_feature_dim": 1536, "hidden_dim": 16, "n_heads": 2, "n_blocks": 1,
        "dense_threshold": 256, "sparse_k": 10, "chunk_size": 1024, "max_boundary_size": None,
        "transport_heads": 2, "transport_temperature": 1.0, "gene_gate_mode": "per_gene",
        "use_query_gate": True, "use_residual": False, "residual_rank": 4,
        "use_anchor_blend": False, "use_regional_he": False, "use_global_gex": False,
        "use_global_slide": False, "global_slide_dim": 16, "n_gex_inducing": 4,
        "harmonic_k_neighbors": 4, "gene_encoder_type": "weighted_linear", "init_seed": 0,
    }
    if architecture == "4":
        del params["use_anchor_blend"]
        del params["use_global_gex"]
        params.update({"n_flow_blocks": 1, "n_flow_samples": 2, "n_ode_steps": 2, "gene_basis_rank": 4})
    params.update(overrides)
    return params


def write_step6_train_config(
    cfg, manifest_path: Path, config_path: Path, *, architecture: str, checkpoint_dir: Path,
    model_param_overrides: dict | None = None, synchronized_init_dir: str | None = None,
    gene_residual_basis_path: str | None = None, checkpoint_every_n_steps: int = 1,
) -> dict:
    """Write a real, small, CPU-runnable Step 6 train.py config to
    `config_path` -- shared by test_train.py and test_step6_scripts.py so
    both exercise the exact same config shape."""
    required_fingerprints = {
        "gigapath_checkpoint": None, "gene_vocabulary": None,
        "train_mask_bank": None, "validation_mask_bank": None, "test_mask_bank": None,
    }
    if architecture == "4":
        required_fingerprints["gene_residual_basis"] = gene_residual_basis_path
    config = {
        "model": {"architecture": architecture, "params": step6_model_params(architecture, **(model_param_overrides or {}))},
        "loss": {"gradient_weight": 0.05, "k_neighbors": 4},
        "masking": {"strata": STEP6_TRAIN_STRATA},
        "data": {
            "hest_data_dir": str(cfg.data.hest_data_dir), "hest_cache_dir": str(cfg.data.hest_cache_dir),
            "slide_context_source": "dense_wsi_cache",
            "gen3_manifest_path": str(manifest_path), "tile_encoder_revision": VALID_HF_REVISION,
            "gex_feature_dim": 8, "n_training_masks_per_sample": 3, "n_validation_masks": 2,
            "novae": {"enabled": False},
        },
        "required_fingerprints": required_fingerprints,
        "training": {
            "device": "cpu", "seed": 0, "lr": 1.0e-3, "total_steps": 100000000,
            "max_wall_clock_hours": 1, "gradient_clip_val": 1.0, "log_every_n_steps": 1,
            "checkpoint_every_n_steps": checkpoint_every_n_steps, "checkpoint_dir": str(checkpoint_dir),
            "eval_every_n_steps": 1, "synchronized_init_dir": synchronized_init_dir,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config
