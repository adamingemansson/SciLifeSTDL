"""
Smoke test for the H&E wiring end-to-end (task #17/#20): does
MaskedContextQueryDataset actually produce context["images"]/
query["images"], and does a real generator model (WAE-GAN here,
representative of all three — they all go through
BaseGenerativeModel._encode_context) consume them correctly via
image_encoder_type="cnn"? Synthetic data throughout — no real HEST-1k
download needed. Gigapath itself is covered separately (and skips
cleanly without external access) in tests/test_conditioning.py.

Run with: python -m tests.test_he_wiring
"""
import numpy as np
import torch

from src.training.train import MaskedContextQueryDataset
from src.models.registry import WAEGAN


class _FakeMaskingCfg:
    strategy = "random_dropout_patches"
    params = {"n_patches": 2, "radius_range": (5, 15)}


def test_dataset_produces_images():
    rng = np.random.default_rng(0)
    n, n_genes, patch_size = 60, 20, 16
    coords3d = rng.uniform(0, 100, size=(n, 3))
    expr = rng.rand(n, n_genes).astype(np.float32)
    slice_ids = np.zeros(n, dtype=int)
    images = rng.integers(0, 255, size=(n, patch_size, patch_size, 3), dtype=np.uint8)

    dataset = MaskedContextQueryDataset(
        coords3d, expr, slice_ids, _FakeMaskingCfg(), n_items=1, base_seed=0, images=images
    )
    item = dataset[0]
    assert "images" in item["context"] and "images" in item["query"]
    assert item["context"]["images"].dtype == torch.float32
    assert item["context"]["images"].max() <= 1.0 and item["context"]["images"].min() >= 0.0
    assert item["context"]["images"].shape[1:] == (3, patch_size, patch_size)
    print(f"[dataset images] OK — context {tuple(item['context']['images'].shape)}, "
          f"query {tuple(item['query']['images'].shape)}")


def test_model_consumes_images():
    torch.manual_seed(0)
    n_genes, patch_size = 20, 16
    model = WAEGAN(n_genes=n_genes, coord_dim=3, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16,
                    image_encoder_type="cnn", image_feat_dim=8, image_patch_size=patch_size)

    context = {
        "coords": torch.randn(30, 3),
        "expression": torch.rand(30, n_genes),
        "images": torch.rand(30, 3, patch_size, patch_size),
    }
    query = {
        "coords": torch.randn(10, 3),
        "images": torch.rand(10, 3, patch_size, patch_size),
    }
    out = model.sample(context, query)
    assert out["expression"].shape == (10, n_genes)
    assert torch.isfinite(out["expression"]).all()
    print(f"[model consumes images] OK — output shape {tuple(out['expression'].shape)}")

    # image_encoder_type="none" (default) models must NOT accept an
    # "images" key silently changing behavior — they should just ignore it
    # (SpatialContextEncoder.use_images=False means forward() never looks
    # at context_images/query_images even if passed)
    model_no_images = WAEGAN(n_genes=n_genes, coord_dim=3, latent_dim=8, hidden_dim=32,
                              cond_hidden_dim=32, disc_hidden_dim=16)
    out2 = model_no_images.sample(context, query)  # images present in dict but unused
    assert out2["expression"].shape == (10, n_genes)
    print("[model ignores images when image_encoder_type='none'] OK")


if __name__ == "__main__":
    test_dataset_produces_images()
    test_model_consumes_images()
    print("\nAll H&E wiring smoke tests passed.")
