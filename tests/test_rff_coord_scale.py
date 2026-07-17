"""
Regression test for the real 2026-07-17 RandomFourierFeatures scale bug
(src/models/conditioning.py) — the second of two coordinate-scale bugs
found this session from real A100 results (the first was
RelativePositionBias, see tests/test_storm_lite_encoder.py's own
scale-invariance test).

sigma=1.0 (every config's untouched default) only makes sense if
coordinates are already roughly unit-scale. Real HEST-1k coordinates are
pixel-scale (thousands — see this project's masking configs' own
radius_range, e.g. [250, 450]). At that scale, sin(2*pi*x@B) wraps around
(aliases) so many times between nearby points that the encoding becomes
essentially random noise with respect to real spatial locality — verified
directly below: two points 5 pixel-units apart (basically the same spot)
got cosine similarity ~0.001 with the OLD code, indistinguishable from
points 2000 units apart. Every config using "builtin"
(SpatialContextEncoder) or "storm_lite" (StormLiteContextEncoder) was
silently getting a near-useless absolute-position signal on real data.

Fixed via a FIXED (construction-time, not per-call) coord_scale divisor —
see RandomFourierFeatures' own docstring for why per-call auto-scaling
(the fix used for RelativePositionBias) would be WRONG here specifically:
SpatialContextEncoder calls this module separately for context_coords and
query_coords, so per-call normalization would give the same real position
a different encoding depending on which call computed it.

Run with: python -m tests.test_rff_coord_scale
"""
import torch

from src.models.conditioning import RandomFourierFeatures


def test_coord_scale_default_preserves_original_behavior():
    """coord_scale=1.0 (default) must be byte-identical to not having the
    parameter at all — every existing test/config that doesn't opt in
    keeps working unchanged."""
    torch.manual_seed(0)
    rff_default = RandomFourierFeatures(3, num_features=16, sigma=1.0)
    torch.manual_seed(0)
    rff_explicit = RandomFourierFeatures(3, num_features=16, sigma=1.0, coord_scale=1.0)

    x = torch.rand(10, 3) * 100.0
    assert torch.allclose(rff_default(x), rff_explicit(x)), (
        "coord_scale=1.0 must produce identical output to omitting it"
    )
    print("[RandomFourierFeatures] OK — coord_scale=1.0 preserves original behavior exactly")


def test_coord_scale_fixes_real_pixel_scale_aliasing():
    """The actual bug: at real HEST-1k pixel scale (coord_scale=1.0, the
    unfixed default), nearby points should NOT be indistinguishable from
    far points. With an appropriately-set coord_scale, they must be."""
    torch.manual_seed(0)
    base = torch.rand(1, 3) * 5000.0
    near = base + torch.tensor([[5.0, 0.0, 0.0]])     # ~same location
    far = base + torch.tensor([[2000.0, 0.0, 0.0]])   # genuinely far

    # unfixed (coord_scale=1.0): must reproduce the real aliasing failure
    torch.manual_seed(1)
    rff_broken = RandomFourierFeatures(3, num_features=64, sigma=1.0, coord_scale=1.0)
    cs_near_broken = torch.cosine_similarity(rff_broken(base), rff_broken(near)).item()
    assert abs(cs_near_broken) < 0.3, (
        f"expected the UNFIXED (coord_scale=1.0) encoder to alias badly on near points "
        f"(cos_sim={cs_near_broken:.4f}) — if this assertion fails, the bug this test "
        f"guards against may no longer reproduce, which would need re-verification"
    )

    # fixed (coord_scale matched to real coordinate spread): nearby points
    # must now be highly similar, far points must not be
    torch.manual_seed(1)
    rff_fixed = RandomFourierFeatures(3, num_features=64, sigma=1.0, coord_scale=1000.0)
    cs_near_fixed = torch.cosine_similarity(rff_fixed(base), rff_fixed(near)).item()
    cs_far_fixed = torch.cosine_similarity(rff_fixed(base), rff_fixed(far)).item()
    assert cs_near_fixed > 0.99, (
        f"with a correctly-set coord_scale, nearby points (5 units apart) should be "
        f"nearly identically encoded, got cos_sim={cs_near_fixed:.4f}"
    )
    assert abs(cs_far_fixed) < 0.3, (
        f"far points (2000 units apart) should NOT be highly correlated, "
        f"got cos_sim={cs_far_fixed:.4f}"
    )
    print(f"[RandomFourierFeatures] OK — coord_scale=1.0 (broken) cos_sim(near)={cs_near_broken:+.4f}; "
          f"coord_scale=1000 (fixed) cos_sim(near)={cs_near_fixed:+.4f}, cos_sim(far)={cs_far_fixed:+.4f}")


if __name__ == "__main__":
    test_coord_scale_default_preserves_original_behavior()
    test_coord_scale_fixes_real_pixel_scale_aliasing()
    print("\nAll RandomFourierFeatures coord_scale regression tests passed.")
