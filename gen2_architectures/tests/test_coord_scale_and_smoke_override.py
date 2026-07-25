import numpy as np
import anndata as ad
from omegaconf import OmegaConf

from gen2_architectures.training import data_prep


def _fake_adata(n_spots: int, xy_std: float, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = np.zeros((n_spots, 3), dtype=np.float32)
    adata = ad.AnnData(X=x)
    adata.obsm["spatial"] = (rng.standard_normal((n_spots, 2)) * xy_std).astype(np.float32)
    return adata


def test_derive_coord_scale_matches_real_coordinate_spread():
    adatas = [_fake_adata(200, xy_std=500.0, seed=0), _fake_adata(200, xy_std=1500.0, seed=1)]
    coord_scale = data_prep.derive_coord_scale(adatas)
    # mean of per-sample std should land somewhere between the two inputs' real stds
    assert 500.0 < coord_scale < 1500.0


def test_apply_coord_scale_injects_for_architecture1():
    cfg = OmegaConf.create({"model": {"architecture": "1", "params": {}}})
    adatas = [_fake_adata(200, xy_std=800.0, seed=0)]
    data_prep.apply_coord_scale(cfg, adatas)
    assert cfg.model.params.coord_scale is not None
    assert cfg.model.params.coord_scale > 0.0


def test_apply_coord_scale_injects_for_stage_b_style_config():
    cfg = OmegaConf.create({"model": {"stage_a_checkpoint_dir": "/fake", "params": {}}})
    adatas = [_fake_adata(200, xy_std=800.0, seed=0)]
    data_prep.apply_coord_scale(cfg, adatas)
    assert cfg.model.params.coord_scale is not None


def test_apply_coord_scale_does_not_inject_for_architecture4():
    """Real bug class this guards against (same as apply_sample_selection's
    organ_vocab guard): Architecture4's constructor has no coord_scale
    param -- injecting it would crash with an unexpected keyword argument."""
    cfg = OmegaConf.create({"model": {"architecture": "4", "params": {"organ_type": "Lung"}}})
    adatas = [_fake_adata(200, xy_std=800.0, seed=0)]
    data_prep.apply_coord_scale(cfg, adatas)
    assert "coord_scale" not in cfg.model.params


def test_apply_coord_scale_does_not_inject_for_stage_a_style_config():
    cfg = OmegaConf.create({"model": {"params": {}}})  # no architecture, no stage_a_checkpoint_dir -> Stage A
    adatas = [_fake_adata(200, xy_std=800.0, seed=0)]
    data_prep.apply_coord_scale(cfg, adatas)
    assert "coord_scale" not in cfg.model.params


def test_explicit_coord_scale_is_not_overwritten():
    cfg = OmegaConf.create({"model": {"architecture": "1", "params": {"coord_scale": 42.0}}})
    adatas = [_fake_adata(200, xy_std=800.0, seed=0)]
    data_prep.apply_coord_scale(cfg, adatas)
    assert cfg.model.params.coord_scale == 42.0


def test_apply_smoke_override_scales_intervals_and_is_noop_when_absent():
    cfg = OmegaConf.create({"training": {
        "total_steps": 100000, "checkpoint_every_n_steps": 2000,
        "eval_every_n_steps": 5000, "log_every_n_steps": 50,
    }})
    data_prep.apply_smoke_override(cfg, None)
    assert cfg.training.total_steps == 100000  # untouched

    data_prep.apply_smoke_override(cfg, 400)
    assert cfg.training.total_steps == 400
    assert cfg.training.checkpoint_every_n_steps == 100
    assert cfg.training.eval_every_n_steps == 200
    assert cfg.training.log_every_n_steps == 40


def test_apply_smoke_override_never_produces_zero_intervals_for_tiny_step_counts():
    cfg = OmegaConf.create({"training": {
        "total_steps": 100000, "checkpoint_every_n_steps": 2000,
        "eval_every_n_steps": 5000, "log_every_n_steps": 50,
    }})
    data_prep.apply_smoke_override(cfg, 3)
    assert cfg.training.checkpoint_every_n_steps >= 1
    assert cfg.training.eval_every_n_steps >= 1
    assert cfg.training.log_every_n_steps >= 1
