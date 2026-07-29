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
import torch.nn as nn

from src.models.conditioning import SpatialContextEncoder, gigapath_tile_encoder_provenance


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
    encoder = SpatialContextEncoder(n_genes=n_genes, coord_dim=3, hidden_dim=64,
                                     k_neighbors=3, rff_features=16,
                                     image_encoder_type="gigapath", image_feat_dim=16,
                                     image_patch_size=patch_size)
    # Construction alone no longer touches Gigapath at all (2026-07-15 fix:
    # GigapathPatchEncoder used to probe the tile encoder's output dim via a
    # real forward pass at __init__ time, unconditionally — moved to only
    # happen lazily, on first RAW-patch use, since real training never uses
    # raw patches and that eager probe caused real HuggingFace network hangs
    # even when features were fully cached locally). So the try/except that
    # used to wrap construction now has to wrap this first raw-patch forward
    # call instead — that's the actual first point Gigapath's real weights
    # get loaded.
    context_coords = torch.randn(n_context, 3)
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.rand(n_context, 3, patch_size, patch_size)
    query_coords = torch.randn(n_query, 3)
    query_images = torch.rand(n_query, 3, patch_size, patch_size)

    try:
        c = encoder(context_coords, context_expression, query_coords,
                    context_images=context_images, query_images=query_images)
    except Exception as e:  # gated HF repo access not granted, no token, network, etc.
        print(f"[image branch: gigapath] SKIPPED — could not load Gigapath ({e})")
        return
    assert c.shape == (n_query, 64)
    assert torch.isfinite(c).all()
    print(f"[image branch: gigapath] OK — output shape {tuple(c.shape)}")

    # precomputed-features (cached) path — task #18/#20's real fix,
    # 2026-07-15: real training must use precomputed features, not raw
    # patches recomputed every step. Must give the SAME result as the raw
    # path above, since both go through _gigapath_preprocess_and_encode.
    from src.models.conditioning import precompute_gigapath_features
    context_np = (context_images.permute(0, 2, 3, 1) * 255).byte().numpy()
    query_np = (query_images.permute(0, 2, 3, 1) * 255).byte().numpy()
    context_feats = torch.from_numpy(precompute_gigapath_features(context_np))
    query_feats = torch.from_numpy(precompute_gigapath_features(query_np))
    c_cached = encoder(context_coords, context_expression, query_coords,
                        context_images=context_feats, query_images=query_feats)
    assert c_cached.shape == (n_query, 64)
    assert torch.isfinite(c_cached).all()
    print(f"[image branch: gigapath, cached] OK — output shape {tuple(c_cached.shape)}")


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


def test_gigapath_tile_encoder_provenance_is_real_and_content_sensitive():
    """18th Codex re-audit (Step 5 Part 2, "the dense WSI cache... stores
    no resolved Hugging Face revision, model identifier, preprocessing
    version, library versions, or tile-encoder fingerprint"): a real
    tile-encoder identity, computed from actual loaded weights rather
    than assumed from the repo id/revision string alone. Uses a small
    stub nn.Module rather than the real (gated, network-dependent)
    GigaPath weights -- gigapath_tile_encoder_provenance only needs
    something with a real state_dict()."""
    encoder_a = nn.Linear(4, 4)
    encoder_b = nn.Linear(4, 4)
    with torch.no_grad():
        encoder_a.weight.fill_(1.0)
        encoder_b.weight.fill_(2.0)  # genuinely different real weights

    prov_a = gigapath_tile_encoder_provenance(encoder_a, revision="deadbeef")
    prov_a_again = gigapath_tile_encoder_provenance(encoder_a, revision="deadbeef")
    prov_b = gigapath_tile_encoder_provenance(encoder_b, revision="deadbeef")

    assert prov_a["state_dict_sha256"] == prov_a_again["state_dict_sha256"]  # stable/deterministic
    assert prov_a["state_dict_sha256"] != prov_b["state_dict_sha256"]  # real content-sensitivity
    assert prov_a["hf_revision"] == "deadbeef"  # explicit revision passed through, never overridden
    assert prov_a["hf_repo_id"] == "prov-gigapath/prov-gigapath"
    assert "timm_version" in prov_a  # None here (timm not installed in this sandbox) is the documented best-effort fallback
    assert prov_a["preprocessing_spec"]
    assert prov_a["schema_version"] == 1


if __name__ == "__main__":
    _run_case(coord_dim=2, n_context=200, n_query=30, n_genes=50, label="Track A (2D, intra-slice)")
    _run_case(coord_dim=3, n_context=500, n_query=80, n_genes=2000, label="Track B (3D, inter-slice)")
    _run_image_branch_case()
    _run_gigapath_case()
    _run_edge_cases()
    print("\nAll conditioning encoder smoke tests passed.")
