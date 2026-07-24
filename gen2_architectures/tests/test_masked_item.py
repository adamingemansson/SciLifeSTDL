import numpy as np
import torch
from omegaconf import OmegaConf

from gen2_architectures.data.masked_item import build_masked_item


def _synthetic_slide(n_spots=200, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    coords3d = np.concatenate([rng.uniform(0, 2000, size=(n_spots, 2)), np.zeros((n_spots, 1))], axis=1)
    expr = rng.normal(size=(n_spots, n_genes)).astype(np.float32)
    images = rng.normal(size=(n_spots, 1536)).astype(np.float32)
    slice_ids = np.array(["s1"] * n_spots)
    return coords3d, expr, images, slice_ids


def _masking_cfg(max_context_points=40):
    return OmegaConf.create({
        "strategy": "random_dropout_patches",
        "max_context_points": max_context_points,
        "context_selection": "nearest_query",
        "params": {"n_patches": 1, "radius_range": [80, 150], "radius_unit": "coordinate", "shape": "mixed"},
    })


def test_build_masked_item_basic_shapes():
    coords3d, expr, images, slice_ids = _synthetic_slide()
    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              image_mode="target_zero", context_gex_mode="full")
    assert item["context"]["coords"].shape[1] == 3
    assert item["context"]["expression"].shape[1] == expr.shape[1]
    assert item["target_expression"].shape[1] == expr.shape[1]
    assert item["query"]["coords"].shape[0] == item["target_expression"].shape[0]
    assert item["context"]["expression"].shape[0] <= 40, "context must respect max_context_points"


def test_context_gex_mode_zero_removes_expression_signal():
    coords3d, expr, images, slice_ids = _synthetic_slide()
    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              image_mode="target_zero", context_gex_mode="zero")
    assert torch.equal(item["context"]["expression"], torch.zeros_like(item["context"]["expression"]))


def test_image_mode_all_zero_removes_context_and_query_images():
    coords3d, expr, images, slice_ids = _synthetic_slide()
    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              image_mode="all_zero", context_gex_mode="full")
    assert torch.equal(item["context"]["images"], torch.zeros_like(item["context"]["images"]))
    assert torch.equal(item["query"]["images"], torch.zeros_like(item["query"]["images"]))
    assert not item["context"]["image_available"].any()
    assert not item["query"]["image_available"].any()


def test_image_mode_target_zero_keeps_context_images_only():
    coords3d, expr, images, slice_ids = _synthetic_slide()
    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              image_mode="target_zero", context_gex_mode="full")
    assert item["context"]["image_available"].all()
    assert not item["query"]["image_available"].any()


def test_context_gene_feature_provider_replaces_expression():
    """Architecture 2's replacement channel: context["expression"] must
    come from the provider, not raw expr, when a provider is given."""
    coords3d, expr, images, slice_ids = _synthetic_slide(n_genes=30)
    replacement_width = 8

    def provider(context_mask):
        return np.ones((int(context_mask.sum()), replacement_width), dtype=np.float32) * 7.0

    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              context_gene_feature_provider=provider,
                              image_mode="target_zero", context_gex_mode="full")
    assert item["context"]["expression"].shape[1] == replacement_width
    assert torch.allclose(item["context"]["expression"], torch.full_like(item["context"]["expression"], 7.0))
    # target_expression must still come from the REAL raw expr, never the replacement
    assert item["target_expression"].shape[1] == expr.shape[1]


def test_context_extra_feature_provider_is_additive_not_a_replacement():
    """Architecture 4's additive residual channel: context["expression"]
    must stay the REAL raw expression, with extra_features stashed
    separately."""
    coords3d, expr, images, slice_ids = _synthetic_slide(n_genes=30)
    extra_width = 5

    def provider(context_mask):
        return np.ones((int(context_mask.sum()), extra_width), dtype=np.float32) * 3.0

    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              context_extra_feature_provider=provider,
                              image_mode="target_zero", context_gex_mode="full")
    assert item["context"]["expression"].shape[1] == expr.shape[1]
    assert item["context"]["extra_features"].shape[1] == extra_width


def test_context_gex_mode_zero_also_zeros_extra_features():
    """The GEX ablation must cover EVERY GEX-derived channel, not just raw
    expression -- otherwise a 'zero' ablation would still leak real signal
    through the additive channel."""
    coords3d, expr, images, slice_ids = _synthetic_slide(n_genes=30)

    def provider(context_mask):
        return np.ones((int(context_mask.sum()), 5), dtype=np.float32) * 3.0

    item = build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                              context_extra_feature_provider=provider,
                              image_mode="target_zero", context_gex_mode="zero")
    assert torch.equal(item["context"]["extra_features"], torch.zeros_like(item["context"]["extra_features"]))


def test_fixed_mask_and_augment_are_mutually_exclusive():
    coords3d, expr, images, slice_ids = _synthetic_slide()
    fixed_mask = np.zeros(coords3d.shape[0], dtype=bool)
    fixed_mask[:50] = True
    try:
        build_masked_item(coords3d, expr, slice_ids, _masking_cfg(), images, seed=0,
                           fixed_context_mask=fixed_mask, fixed_query_mask=~fixed_mask, augment=True)
        assert False, "expected a ValueError combining fixed masks with augmentation"
    except ValueError:
        pass
