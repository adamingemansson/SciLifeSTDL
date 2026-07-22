import numpy as np
import torch

from src.data.slide_context import (
    nonoverlapping_context_patch_mask,
    visible_slide_context,
)
from src.models.registry import build_model


def _slide_context():
    # Three 256px tiles at x=0,256,512.  Query at x=384 intersects only
    # the middle tile when represented by a 224px missing patch.
    return {
        "features": np.ones((3, 1536), dtype=np.float32),
        "coords": np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32),
        "mask_coords": np.asarray([[0, 0], [256, 0], [512, 0]], dtype=np.float32),
        "tile_size": 256.0,
        "mask_tile_size": 256.0,
        "coords_are_centers": False,
        "context_id": "unit-test",
        "source": "dense_wsi_cache",
    }


def test_target_zero_removes_every_wsi_tile_intersecting_hole():
    visible = visible_slide_context(
        _slide_context(), np.asarray([[384, 128, 0]], dtype=np.float32),
        image_mode="target_zero", query_patch_size=224.0,
    )
    assert visible["available"]
    assert visible["n_total"] == 3
    assert visible["n_visible"] == 2
    assert np.array_equal(visible["coords"][:, 0], np.asarray([0, 512]))


def test_all_zero_removes_complete_slide_context():
    visible = visible_slide_context(
        _slide_context(), np.asarray([[384, 128, 0]], dtype=np.float32),
        image_mode="all_zero", query_patch_size=224.0,
    )
    assert visible == {"available": False}


def test_local_context_patch_overlap_is_removed():
    context = np.asarray([[0, 0, 0], [200, 0, 0], [600, 0, 0]], dtype=np.float32)
    query = np.asarray([[300, 0, 0]], dtype=np.float32)
    safe = nonoverlapping_context_patch_mask(context, query, patch_size=224.0)
    assert safe.tolist() == [True, False, True]


def test_hierarchical_model_needs_only_observed_inputs_for_queries():
    model = build_model({
        "name": "hierarchical_missing_tissue_regressor",
        "params": {
            "n_genes": 5,
            "hidden_dim": 16,
            "decoder_hidden_dim": 24,
            "n_heads": 4,
            "context_layers": 1,
            "cross_layers": 1,
            "query_layers": 1,
            "local_k": 2,
            "dropout": 0.0,
            "use_novae": False,
            "use_local_images": True,
            "use_slide_context": False,
            "gene_encoder_type": "weighted_linear",
            "target_gene_mean": [0.0] * 5,
            "target_gene_scale": [1.0] * 5,
        },
    }).eval()
    context = {
        "coords": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "expression": torch.rand(3, 5),
        "images": torch.rand(3, 1536),
        "image_available": torch.ones(3, dtype=torch.bool),
        "slide_available": False,
    }
    # Query has coordinates only: no query GEX and no query H&E key exists.
    query = {"coords": torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])}
    with torch.inference_mode():
        output = model.sample(context, query)["expression"]
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
