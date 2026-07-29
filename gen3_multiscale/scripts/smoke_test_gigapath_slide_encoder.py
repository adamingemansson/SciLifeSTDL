#!/usr/bin/env python3
"""Real A100 smoke test for FrozenGigaPathSlideEncoder AND a real
Architecture3/Architecture4 forward pass using it -- Step 5's explicit
acceptance criterion: "Real A100 smoke testing verifies frozen/eval-mode
LongNet, FP16/FlashAttention and bounded memory."

NOT RUN by the agent that wrote this script -- this environment has no
GPU and no Prov-GigaPath checkpoint. Adam (or whoever has A100 access
and the real checkpoint) must run this directly:

    python -m gen3_multiscale.scripts.smoke_test_gigapath_slide_encoder \
        --checkpoint /path/to/slide_encoder.pth --n-tiles 4096

Exits 0 and prints a PASS summary only if every check below succeeds;
raises/exits non-zero with a specific message on the first failure.
Nothing here is a substitute for a real training run -- it only proves
(1) the frozen slide encoder itself behaves as this codebase requires:
eval-mode and non-trainable even after an explicit .train(True) call
(Lightning calls this recursively), FP16 + FlashAttention actually
engaged on an A100-class device, peak CUDA memory bounded for a
realistic tile count, and its own internal cache genuinely avoids a
second real forward pass (not merely coincidentally-deterministic
output); and (2) a real Architecture3 and Architecture4 forward pass,
with this real encoder wired in via use_regional_he/use_global_slide,
completes end to end on CUDA within a memory bound -- not just the
isolated LongNet call in isolation (17th Codex re-audit, Step 5 Part 2,
"Important before Step 6/7").
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.data.example import SpatialFieldInputs, SpatialFieldTargets, validate_spatial_field_example
from gen3_multiscale.models.architectures import Architecture3, Architecture4
from gen3_multiscale.models.gene_basis import fit_gene_residual_basis
from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder


def _fail(message: str) -> "typing.NoReturn":  # noqa: F821 -- string annotation only
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def _check_frozen_encoder(encoder: FrozenGigaPathSlideEncoder) -> None:
    # Frozen/eval-mode check #1: immediately after construction.
    if encoder.training:
        _fail("encoder.training is True immediately after construction -- must be eval-mode by default")
    non_frozen = [name for name, p in encoder.named_parameters() if p.requires_grad]
    if non_frozen:
        _fail(f"{len(non_frozen)} parameter(s) have requires_grad=True after construction: {non_frozen[:5]}")

    # Frozen/eval-mode check #2: Lightning calls .train(True) recursively on
    # every child module during a training loop -- the encoder's overridden
    # train() must force itself back to eval() regardless.
    encoder.train(True)
    if encoder.training or encoder.model.training:
        _fail("encoder (or its child model) left eval mode after encoder.train(True) -- must stay in eval mode unconditionally")
    still_frozen = [name for name, p in encoder.named_parameters() if p.requires_grad]
    if still_frozen:
        _fail(f"{len(still_frozen)} parameter(s) became trainable after encoder.train(True): {still_frozen[:5]}")
    print("PASS: encoder stays eval-mode and fully frozen, including after an explicit train(True) call")

    if not isinstance(encoder.checkpoint_sha256, str) or len(encoder.checkpoint_sha256) != 64:
        _fail(f"encoder.checkpoint_sha256 is not a real 64-hex-char SHA256: {encoder.checkpoint_sha256!r}")
    print(f"PASS: encoder exposes its real checkpoint SHA256 ({encoder.checkpoint_sha256[:12]}...)")


def run_slide_encoder_smoke_test(
    checkpoint_path: str,
    n_tiles: int = 4096,
    tile_feature_dim: int = 1536,
    output_dim: int = 768,
    max_memory_gb: float = 20.0,
    seed: int = 0,
) -> FrozenGigaPathSlideEncoder:
    if not torch.cuda.is_available():
        _fail("CUDA is not available -- this smoke test must run on a real A100-class GPU")
    device_capability = torch.cuda.get_device_capability()
    print(f"CUDA device: {torch.cuda.get_device_name()} (compute capability {device_capability})")

    print(f"Constructing FrozenGigaPathSlideEncoder from {checkpoint_path!r} ...")
    encoder = FrozenGigaPathSlideEncoder(checkpoint_path=checkpoint_path, tile_feature_dim=tile_feature_dim, output_dim=output_dim)
    _check_frozen_encoder(encoder)

    encoder = encoder.to(device="cuda")
    rng = np.random.default_rng(seed)
    tile_features = torch.as_tensor(rng.normal(size=(n_tiles, tile_feature_dim)), dtype=torch.float32, device="cuda")
    tile_coords = torch.as_tensor(rng.uniform(0.0, 100_000.0, size=(n_tiles, 2)), dtype=torch.float32, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    embedding = encoder(tile_features, tile_coords, cache_namespace="smoke-test:001")
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024 ** 3)

    if embedding.shape != (output_dim,):
        _fail(f"expected embedding shape ({output_dim},), got {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        _fail("embedding contains non-finite values")
    print(f"PASS: forward pass produced a finite [{output_dim}] embedding for {n_tiles} tiles")

    # FP16/FlashAttention check: the class itself only promotes model +
    # tile-feature inputs to FP16 on CUDA -- verify that actually happened,
    # rather than trusting the code path ran silently in FP32.
    model_param = next(encoder.model.parameters())
    if model_param.dtype != torch.float16:
        _fail(f"frozen LongNet parameters are {model_param.dtype}, expected torch.float16 on CUDA")
    print("PASS: frozen LongNet ran in FP16 on CUDA")
    if device_capability[0] > 7:
        # FrozenGigaPathSlideEncoder.__init__ already raises at
        # construction time if FlashAttention's compiled kernel is
        # unavailable on an A100-class device -- reaching this line at
        # all is itself the FlashAttention-engaged confirmation.
        print("PASS: FlashAttention kernel was available and engaged (construction would have raised otherwise)")
    else:
        print("NOTE: device compute capability <= 7 -- FlashAttention gate not exercised on this GPU")

    print(f"Peak CUDA memory for {n_tiles} tiles: {peak_gb:.2f} GB (bound: {max_memory_gb:.2f} GB)")
    if peak_gb > max_memory_gb:
        _fail(f"peak CUDA memory {peak_gb:.2f} GB exceeded the {max_memory_gb:.2f} GB bound")
    print("PASS: peak CUDA memory stayed within bound")

    # 17th Codex re-audit (Step 5 Part 2 launch blocker #3), CONFIRMED
    # real gap in an earlier version of this script: identical outputs
    # across two calls can simply result from DETERMINISTIC inference,
    # not from the cache actually being hit -- that proves nothing about
    # whether a second real (expensive) forward pass happened. Force the
    # frozen LongNet's own forward to raise on any call AFTER the first,
    # real one -- a repeated call with the identical
    # cache_namespace/coords must therefore either raise (cache miss,
    # real bug) or succeed via the cache path alone.
    real_model_forward = encoder.model.forward

    def _raise_if_called_again(*args, **kwargs):
        _fail("encoder.model.forward() was called a SECOND time for an identical cache_namespace/coords -- the cache was not hit")

    encoder.model.forward = _raise_if_called_again
    try:
        embedding_again = encoder(tile_features, tile_coords, cache_namespace="smoke-test:001")
    finally:
        encoder.model.forward = real_model_forward
    if not torch.equal(embedding, embedding_again.to(embedding.device, embedding.dtype)):
        _fail("a repeated call with the identical cache_namespace/coords returned a different embedding")
    print("PASS: repeated call with the same cache_namespace/coords hit the cache (the real LongNet forward was never called again)")

    print("\nALL FrozenGigaPathSlideEncoder CHECKS PASSED\n")
    return encoder


def _synthetic_inputs_with_wsi_context(n_genes: int, gex_dim: int, image_dim: int, n_wsi_tiles: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 15
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n), np.arange(lo, lo + n))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    dist = np.linalg.norm(grid, axis=1)
    observed_coords, query_coords = grid[dist > 2.5], grid[dist <= 2.5]
    n_observed, n_query = observed_coords.shape[0], query_coords.shape[0]
    boundary = extract_boundary_and_local_context(observed_coords, query_coords, k_neighbors=6, local_k=6, max_rings=3)

    grid_bound = 10.0
    inputs = SpatialFieldInputs(
        sample_id="smoke-test", patient_id="smoke-test-patient",
        observed_barcodes=np.array([f"o{i}" for i in range(n_observed)]),
        query_barcodes=np.array([f"q{i}" for i in range(n_query)]),
        observed_coords=observed_coords.astype(np.float32),
        query_coords=query_coords.astype(np.float32),
        observed_full_gene_expression=rng.normal(size=(n_observed, n_genes)).astype(np.float32),
        observed_gigapath_features=rng.normal(size=(n_observed, image_dim)).astype(np.float32),
        observed_image_available=np.ones(n_observed, dtype=bool),
        query_local_neighbor_idx=boundary.query_local_neighbor_idx,
        boundary_idx=boundary.boundary_idx,
        boundary_ring=boundary.boundary_ring,
        query_depth_to_boundary=boundary.query_depth_to_boundary,
        wsi_tile_longnet_coords=rng.uniform(10_000.0, 20_000.0, size=(n_wsi_tiles, 2)).astype(np.float32),
        wsi_tile_regional_coords=rng.uniform(-grid_bound + 0.5, grid_bound - 0.5, size=(n_wsi_tiles, 2)).astype(np.float32),
        wsi_tile_features=rng.normal(size=(n_wsi_tiles, image_dim)).astype(np.float32),
        full_slide_coord_bounds=(-grid_bound, grid_bound, -grid_bound, grid_bound),
        slide_cache_namespace="smoke-test-slide-namespace",
    )
    targets = SpatialFieldTargets(query_expression=rng.normal(size=(n_query, n_genes)).astype(np.float32))
    validate_spatial_field_example(inputs, targets)
    return inputs, targets


def run_architecture_smoke_test(
    slide_encoder: FrozenGigaPathSlideEncoder,
    n_genes: int = 32,
    gex_dim: int = 16,
    image_dim: int = 1536,
    hidden_dim: int = 256,
    n_wsi_tiles: int = 512,
    max_memory_gb: float = 20.0,
) -> None:
    """17th Codex re-audit (Step 5 Part 2, "Important before Step 6/7"):
    the isolated FrozenGigaPathSlideEncoder call above proves the
    encoder itself behaves correctly, but says nothing about a real
    Architecture3/Architecture4 forward pass with use_regional_he=True/
    use_global_slide=True actually wired to it -- run one of each here,
    on CUDA, with a real checkpoint SHA256 cross-check."""
    torch.manual_seed(0)
    inputs, targets = _synthetic_inputs_with_wsi_context(n_genes, gex_dim, image_dim, n_wsi_tiles)

    model3 = Architecture3(
        n_genes=n_genes, gex_feature_dim=gex_dim, image_feature_dim=image_dim, hidden_dim=hidden_dim,
        n_heads=4, n_blocks=2, dense_threshold=100,
        use_regional_he=True, use_global_slide=True, global_slide_dim=slide_encoder.output_dim,
        regional_grid_size=4, slide_encoder=slide_encoder, gigapath_checkpoint_sha256=slide_encoder.checkpoint_sha256,
    ).to(device="cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out3 = model3(inputs)
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if out3["expression"].shape != tuple(targets.query_expression.shape):
        _fail(f"Architecture3 output shape {tuple(out3['expression'].shape)} != expected {targets.query_expression.shape}")
    if not torch.isfinite(out3["expression"]).all():
        _fail("Architecture3 produced non-finite predictions")
    if peak_gb > max_memory_gb:
        _fail(f"Architecture3 forward pass peak CUDA memory {peak_gb:.2f} GB exceeded the {max_memory_gb:.2f} GB bound")
    print(f"PASS: real Architecture3 forward pass (use_regional_he=True, use_global_slide=True) succeeded on CUDA, peak {peak_gb:.2f} GB")

    gene_names = [f"g{i}" for i in range(n_genes)]
    gene_basis = fit_gene_residual_basis(
        np.random.default_rng(1).normal(size=(64, n_genes)), gene_names, rank=8,
    )
    model4 = Architecture4(
        n_genes=n_genes, gex_feature_dim=gex_dim, gene_basis=gene_basis, gene_names=gene_names,
        image_feature_dim=image_dim, hidden_dim=hidden_dim, n_heads=4, n_blocks=2, dense_threshold=100,
        use_regional_he=True, use_global_slide=True, global_slide_dim=slide_encoder.output_dim,
        regional_grid_size=4, slide_encoder=slide_encoder, gigapath_checkpoint_sha256=slide_encoder.checkpoint_sha256,
    ).to(device="cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    out4 = model4(inputs)
    torch.cuda.synchronize()
    peak_gb4 = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if not torch.isfinite(out4["expression"]).all():
        _fail("Architecture4 produced non-finite predictions")
    if peak_gb4 > max_memory_gb:
        _fail(f"Architecture4 forward pass peak CUDA memory {peak_gb4:.2f} GB exceeded the {max_memory_gb:.2f} GB bound")
    print(f"PASS: real Architecture4 forward pass (conditioner use_regional_he=True, use_global_slide=True) succeeded on CUDA, peak {peak_gb4:.2f} GB")

    print("\nALL Architecture3/Architecture4 CHECKS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the official Prov-GigaPath slide_encoder.pth")
    parser.add_argument("--n-tiles", type=int, default=4096, help="Number of synthetic tiles for the isolated LongNet call")
    parser.add_argument("--tile-feature-dim", type=int, default=1536)
    parser.add_argument("--output-dim", type=int, default=768)
    parser.add_argument("--max-memory-gb", type=float, default=20.0, help="Peak CUDA memory bound to enforce")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-architecture-smoke", action="store_true",
        help="Skip the real Architecture3/Architecture4 end-to-end smoke test (isolated LongNet checks only)",
    )
    args = parser.parse_args()
    encoder = run_slide_encoder_smoke_test(
        checkpoint_path=args.checkpoint, n_tiles=args.n_tiles, tile_feature_dim=args.tile_feature_dim,
        output_dim=args.output_dim, max_memory_gb=args.max_memory_gb, seed=args.seed,
    )
    if not args.skip_architecture_smoke:
        run_architecture_smoke_test(encoder, image_dim=args.tile_feature_dim, max_memory_gb=args.max_memory_gb)


if __name__ == "__main__":
    main()
