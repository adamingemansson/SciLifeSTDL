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
completes end to end on CUDA within a memory bound -- built through the
REAL data pipeline (a real, on-disk dense_wsi_cache ->
slide_context.load_slide_context -> example_builder.build_spatial_field_example),
not a hand-rolled SpatialFieldInputs, and with each architecture proven
to independently exercise a REAL LongNet forward pass (not merely serve
the other architecture's cached result) via an explicit call counter
(17th/18th Codex re-audits, Step 5 Part 2, "Important before Step 6/7" /
"Other real gaps").
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

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


def _build_real_inputs_through_the_data_pipeline(
    n_genes: int, image_dim: int, tmp_dir: Path, seed: int = 0,
):
    """18th Codex re-audit (Step 5 Part 2, "Other real gaps"): a prior
    version hand-built a SpatialFieldInputs directly, bypassing
    load_slide_context/build_spatial_field_example entirely -- this
    smoke test is supposed to catch real data-layer bugs too, not just
    exercise the model forward pass on inputs that were never actually
    produced by the real pipeline. Writes a real, on-disk synthetic
    dense_wsi_cache .npz (the same schema
    scripts/precompute_gigapath_wsi_tiles.py writes, tile-encoder
    provenance included), loads it through the REAL
    slide_context.load_slide_context, and builds the example through the
    REAL example_builder.build_spatial_field_example -- the actual
    functions Step 6's trainer will call, not a hand-rolled substitute."""
    import anndata as ad
    import pandas as pd
    from omegaconf import OmegaConf

    from gen3_multiscale.data import example_builder
    from gen3_multiscale.data.slide_context import load_slide_context

    rng = np.random.default_rng(seed)
    n_side, spacing = 15, 300.0  # real-Visium-scale pixel spacing, not a tiny synthetic unit
    coords = np.asarray([[x * spacing, y * spacing] for x in range(n_side) for y in range(n_side)], dtype=np.float64)
    n_spots = coords.shape[0]
    barcodes = [f"SPOT{i}-1" for i in range(n_spots)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    counts = rng.poisson(5, size=(n_spots, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=counts, obs=pd.DataFrame(index=pd.Index(barcodes)), var=pd.DataFrame(index=pd.Index(gene_names)),
    )
    adata.obsm["spatial"] = coords
    patches = np.zeros((n_spots, 4, 4, 3), dtype=np.uint8)  # content irrelevant -- image_feature_fn below is a stub

    query_idx = n_spots // 2
    query_barcodes = [barcodes[query_idx]]
    context_barcodes = [b for b in barcodes if b != barcodes[query_idx]]

    # A REAL, regularly-gridded (not random-scatter) dense WSI tile
    # cache -- matches scripts/precompute_gigapath_wsi_tiles.py's actual
    # output structure (a real, non-overlapping tile grid), guaranteeing
    # full ST-spot/WSI-tile coverage deterministically rather than
    # depending on a random scatter happening to align.
    tile_step = 256.0
    tile_xs = np.arange(0.0, (n_side - 1) * spacing + tile_step, tile_step)
    tile_ys = np.arange(0.0, (n_side - 1) * spacing + tile_step, tile_step)
    tile_gx, tile_gy = np.meshgrid(tile_xs, tile_ys)
    tile_xy = np.stack([tile_gx.ravel(), tile_gy.ravel()], axis=1).astype(np.float32)
    n_wsi_tiles = tile_xy.shape[0]
    tile_features = rng.normal(size=(n_wsi_tiles, image_dim)).astype(np.float32)

    cache_dir = tmp_dir / "gigapath_slide_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "smoke-test.npz",
        features=tile_features, coords=tile_xy.copy(), level0_coords=tile_xy.copy(),
        tile_size=np.asarray(tile_step, dtype=np.float32), level0_tile_size=np.asarray(tile_step, dtype=np.float32),
        coords_are_centers=np.asarray(True),
        tile_encoder_hf_repo_id=np.asarray("prov-gigapath/prov-gigapath"),
        tile_encoder_hf_revision=np.asarray("smoke-test"),
        tile_encoder_timm_version=np.asarray("smoke-test"),
        tile_encoder_preprocessing_spec=np.asarray("smoke-test"),
        tile_encoder_state_dict_sha256=np.asarray("f" * 64),
        tile_encoder_schema_version=np.asarray(1),
    )
    cfg = OmegaConf.create({
        "data": {
            "slide_context_source": "dense_wsi_cache", "hest_data_dir": str(tmp_dir),
            "slide_context_cache_dir": str(cache_dir),
        },
    })
    slide_ctx = load_slide_context(cfg, "smoke-test", None, coords)

    def _stub_image_feature_fn(patch_batch: np.ndarray) -> np.ndarray:
        return rng.normal(size=(patch_batch.shape[0], image_dim)).astype(np.float32)

    return example_builder.build_spatial_field_example(
        adata, patches, context_barcodes, query_barcodes, _stub_image_feature_fn,
        sample_id="smoke-test", patient_id="smoke-test-patient", patch_size_fullres=1.0,
        full_sample_coords=coords, slide_context=slide_ctx, expected_feature_width=image_dim,
    )


def run_architecture_smoke_test(
    slide_encoder: FrozenGigaPathSlideEncoder,
    n_genes: int = 32,
    gex_dim: int = 16,
    image_dim: int = 1536,
    hidden_dim: int = 256,
    max_memory_gb: float = 20.0,
) -> None:
    """17th/18th Codex re-audits (Step 5 Part 2, "Important before Step
    6/7" / "Other real gaps"): the isolated FrozenGigaPathSlideEncoder
    call above proves the encoder itself behaves correctly, but says
    nothing about (1) a real Architecture3/Architecture4 forward pass
    with use_regional_he=True/use_global_slide=True actually wired to
    it, built through the REAL data pipeline (load_slide_context ->
    build_spatial_field_example), and (2) whether LongNet actually runs
    TWICE, independently, for Architecture3 and Architecture4 -- a prior
    version reused the identical inputs/cache_namespace for both calls,
    so Architecture4 silently served Architecture3's CACHED LongNet
    result and never independently exercised the real forward pass at
    all."""
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="gen3_smoke_test_") as tmp_dir_str:
        inputs, targets = _build_real_inputs_through_the_data_pipeline(n_genes, image_dim, Path(tmp_dir_str))

    # Count REAL LongNet forward passes (not cache hits) by wrapping the
    # frozen model's own forward -- proves each architecture below
    # genuinely exercises LongNet, rather than one silently reusing the
    # other's cached result.
    real_longnet_forward = slide_encoder.model.forward
    longnet_call_count = {"n": 0}

    def _counting_longnet_forward(*args, **kwargs):
        longnet_call_count["n"] += 1
        return real_longnet_forward(*args, **kwargs)

    slide_encoder.model.forward = _counting_longnet_forward

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
    if longnet_call_count["n"] != 1:
        _fail(f"expected exactly 1 real LongNet forward pass for Architecture3, got {longnet_call_count['n']}")
    print(f"PASS: real Architecture3 forward pass (use_regional_he=True, use_global_slide=True) succeeded on CUDA, peak {peak_gb:.2f} GB, LongNet called for real")

    # 18th Codex re-audit, CONFIRMED real: clear the encoder's own cache
    # before the Architecture4 call -- with the SAME slide_encoder,
    # inputs, and cache_namespace as Architecture3 just used, this call
    # would otherwise be served entirely from cache, never actually
    # re-running LongNet under Architecture4's own code path.
    slide_encoder._cache.clear()

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
    if longnet_call_count["n"] != 2:
        _fail(
            f"expected exactly 2 TOTAL real LongNet forward passes after Architecture4 (1 from "
            f"Architecture3 + 1 independently from Architecture4), got {longnet_call_count['n']} -- "
            "Architecture4 may have silently served Architecture3's cached LongNet result instead "
            "of independently exercising the real forward pass"
        )
    slide_encoder.model.forward = real_longnet_forward
    print(f"PASS: real Architecture4 forward pass (conditioner use_regional_he=True, use_global_slide=True) succeeded on CUDA, peak {peak_gb4:.2f} GB, LongNet called for real (independently of Architecture3's cache)")

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
