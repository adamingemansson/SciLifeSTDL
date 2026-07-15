"""
Smoke test for the conditioning encoder (src/models/conditioning.py) —
tiny synthetic data, no real dataset needed. Checks shapes, no crashes, no
NaNs, and that it handles both Track A (2D) and Track B (3D) coordinate
dimensionality with the same module.

Run with: python -m tests.test_conditioning
(needs the st3d conda env active — docs/environment_setup.md; must be run
as a module with -m from the repo root, not as a plain script, so `src` is
importable — same convention as src/training/train.py)
"""
import torch

from src.models.conditioning import SpatialContextEncoder


def _run_case(coord_dim: int, n_context: int, n_query: int, n_genes: int, label: str):
    torch.manual_seed(0)
    encoder = SpatialContextEncoder(n_genes=n_genes, coord_dim=coord_dim, hidden_dim=64,
                                     n_message_layers=2, k_neighbors=5, rff_features=16)

    context_coords = torch.randn(n_context, coord_dim)
    context_expression = torch.rand(n_context, n_genes)
    query_coords = torch.randn(n_query, coord_dim)

    c = encoder(context_coords, context_expression, query_coords)

    assert c.shape == (n_query, 64), f"[{label}] expected shape ({n_query}, 64), got {tuple(c.shape)}"
    assert torch.isfinite(c).all(), f"[{label}] output contains NaN/Inf"
    print(f"[{label}] OK — output shape {tuple(c.shape)}, "
          f"mean={c.mean().item():.4f}, std={c.std().item():.4f}")


def _run_image_branch_case():
    """Task #17: image_encoder_type="cnn" fuses an H&E patch branch;
    image_encoder_type="none" (default, everywhere else in this file) must
    stay completely unchanged — that's the whole point of keeping an
    expression-only variant. "gigapath" is covered separately below since
    it needs external access this sandbox doesn't have."""
    torch.manual_seed(0)
    n_context, n_query, n_genes, patch_size = 40, 10, 30, 32
    encoder = SpatialContextEncoder(n_genes=n_genes, coord_dim=3, hidden_dim=64,
                                     k_neighbors=5, rff_features=16,
                                     image_encoder_type="cnn", image_feat_dim=16,
                                     image_patch_size=patch_size)

    context_coords = torch.randn(n_context, 3)
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.rand(n_context, 3, patch_size, patch_size)
    query_coords = torch.randn(n_query, 3)
    query_images = torch.rand(n_query, 3, patch_size, patch_size)

    c = encoder(context_coords, context_expression, query_coords,
                context_images=context_images, query_images=query_images)
    assert c.shape == (n_query, 64)
    assert torch.isfinite(c).all()

    # required-args check: image branch enabled without images must raise,
    # not silently ignore the image branch
    try:
        encoder(context_coords, context_expression, query_coords)
        raised = False
    except ValueError:
        raised = True
    assert raised, "image_encoder_type='cnn' should require context_images/query_images"

    print(f"[image branch: cnn] OK — output shape {tuple(c.shape)}, "
          f"correctly raises when images are missing")


def _run_gigapath_case():
    """image_encoder_type="gigapath" needs `timm` installed AND a
    HuggingFace account with access granted to the gated
    prov-gigapath/prov-gigapath repo (research-use-only license) — not
    available in every environment. Skip cleanly rather than fail the
    whole suite if either is missing; this is the one case that actually
    needs a real network fetch of pretrained weights, unlike every other
    test in this file."""
    try:
        import timm  # noqa: F401
    except ImportError:
        print("[image branch: gigapath] SKIPPED — timm not installed (pip install timm)")
        return

    torch.manual_seed(0)
    n_context, n_query, n_genes, patch_size = 6, 3, 30, 256
    try:
        encoder = SpatialContextEncoder(n_genes=n_genes, coord_dim=3, hidden_dim=64,
                                         k_neighbors=3, rff_features=16,
                                         image_encoder_type="gigapath", image_feat_dim=16,
                                         image_patch_size=patch_size)
    except Exception as e:  # gated HF repo access not granted, no token, network, etc.
        print(f"[image branch: gigapath] SKIPPED — could not load Gigapath ({e})")
        return

    context_coords = torch.randn(n_context, 3)
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.rand(n_context, 3, patch_size, patch_size)
    query_coords = torch.randn(n_query, 3)
    query_images = torch.rand(n_query, 3, patch_size, patch_size)

    c = encoder(context_coords, context_expression, query_coords,
                context_images=context_images, query_images=query_images)
    assert c.shape == (n_query, 64)
    assert torch.isfinite(c).all()
    print(f"[image branch: gigapath] OK — output shape {tuple(c.shape)}")


def _run_edge_cases():
    # fewer context points than k_neighbors — _knn_indices should clip k, not crash
    encoder = SpatialContextEncoder(n_genes=10, coord_dim=2, hidden_dim=32, k_neighbors=20)
    context_coords = torch.randn(3, 2)
    context_expression = torch.rand(3, 10)
    query_coords = torch.randn(2, 2)
    c = encoder(context_coords, context_expression, query_coords)
    assert c.shape == (2, 32)
    assert torch.isfinite(c).all()
    print("[edge: fewer context points than k] OK")

    # single query point
    encoder2 = SpatialContextEncoder(n_genes=10, coord_dim=3, hidden_dim=32, k_neighbors=5)
    c2 = encoder2(torch.randn(20, 3), torch.rand(20, 10), torch.randn(1, 3))
    assert c2.shape == (1, 32)
    print("[edge: single query point] OK")


if __name__ == "__main__":
    _run_case(coord_dim=2, n_context=200, n_query=30, n_genes=50, label="Track A (2D, intra-slice)")
    _run_case(coord_dim=3, n_context=500, n_query=80, n_genes=2000, label="Track B (3D, inter-slice)")
    _run_image_branch_case()
    _run_gigapath_case()
    _run_edge_cases()
    print("\nAll conditioning encoder smoke tests passed.")
