"""
Tests for FrameAveragingBias (src/models/conditioning.py, added
2026-07-17) — a real, VERIFIED reproduction of STPath's own relative-
position attention bias mechanism, directly confirmed by cloning and
reading github.com/Graph-and-Geometric-Learning/STPath's real source
(stpath/model/nn_utils/fa.py + stpath/model/encoder/spatial_transformer.py).

The defining property of frame averaging (Puny et al. 2022) is exact
invariance to the symmetry group averaged over — here, rotation +
reflection of the whole coordinate system. This test file's central check
is exactly that property, verified numerically with a real random
rotation+reflection+translation, not just a shape/finiteness smoke test.

Run with: python -m tests.test_frame_averaging_bias
"""
import torch

from src.models.conditioning import FrameAveragingBias


def _random_isometry(coords_xy: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply a random rotation + reflection + translation to coords_xy's
    xy plane — the exact symmetry group FrameAveragingBias is supposed to
    be invariant to."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(1, generator=g).item() * 2 * 3.14159265
    cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    rotation = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    reflect = torch.tensor([[-1.0, 0.0], [0.0, 1.0]]) if torch.rand(1, generator=g).item() < 0.5 \
        else torch.eye(2)
    transform = reflect @ rotation
    translation = torch.rand(2, generator=g) * 100.0 - 50.0
    out = coords_xy.clone()
    out[:, :2] = coords_xy[:, :2] @ transform.T + translation
    return out


def test_shape_and_finiteness():
    torch.manual_seed(0)
    n_heads, n = 4, 8
    mod = FrameAveragingBias(n_heads=n_heads)
    coords = torch.randn(n, 3) * 3
    bias = mod(coords)
    assert bias.shape == (n_heads, n, n), bias.shape
    assert torch.isfinite(bias).all()
    print(f"[FrameAveragingBias] OK — shape {tuple(bias.shape)}, all finite")


def test_rotation_reflection_translation_invariance():
    """The core mathematical guarantee frame averaging exists to provide —
    verified with an ACTUAL random rotation+reflection+translation applied
    to real coordinates, not just asserted from theory."""
    torch.manual_seed(0)
    n_heads, n = 4, 10
    mod = FrameAveragingBias(n_heads=n_heads)
    coords = torch.randn(n, 3) * 5

    bias_orig = mod(coords)
    for seed in range(5):
        coords_transformed = _random_isometry(coords, seed=seed)
        bias_transformed = mod(coords_transformed)
        max_diff = (bias_orig - bias_transformed).abs().max().item()
        assert max_diff < 1e-4, (
            f"seed={seed}: bias changed under a rotation+reflection+translation "
            f"(max abs diff={max_diff}) — frame averaging's core invariance guarantee is broken"
        )
    print("[FrameAveragingBias] OK — provably invariant to 5 different random "
          "rotation+reflection+translations")


def test_not_invariant_to_nonrigid_change():
    """Sanity check the invariance test above isn't vacuous (e.g. the bias
    isn't just a constant regardless of input) — a genuine SHAPE change
    (not a rigid transform) must change the bias."""
    torch.manual_seed(0)
    n_heads, n = 4, 10
    mod = FrameAveragingBias(n_heads=n_heads)
    coords = torch.randn(n, 3) * 5
    bias_orig = mod(coords)

    coords_deformed = coords.clone()
    coords_deformed[0, :2] += 20.0  # move a single point non-rigidly
    bias_deformed = mod(coords_deformed)
    assert not torch.allclose(bias_orig, bias_deformed, atol=1e-3), (
        "bias should change under a genuine non-rigid deformation — "
        "if this fails, the invariance test above may be vacuously trivial"
    )
    print("[FrameAveragingBias] OK — genuinely responds to non-rigid changes (invariance test isn't vacuous)")


def test_gradient_flow():
    torch.manual_seed(0)
    mod = FrameAveragingBias(n_heads=4)
    coords = torch.randn(8, 3) * 3
    bias = mod(coords)
    loss = bias.sum()
    loss.backward()
    assert mod.edge_bias.weight.grad is not None
    assert torch.isfinite(mod.edge_bias.weight.grad).all()
    print("[FrameAveragingBias] OK — gradient flows to edge_bias")


def test_coord_scale_default_preserves_behavior():
    torch.manual_seed(0)
    mod_default = FrameAveragingBias(n_heads=4)
    torch.manual_seed(0)
    mod_explicit = FrameAveragingBias(n_heads=4, coord_scale=1.0)
    coords = torch.randn(6, 3) * 10
    assert torch.allclose(mod_default(coords), mod_explicit(coords)), (
        "coord_scale=1.0 must be the implicit default"
    )
    print("[FrameAveragingBias] OK — coord_scale=1.0 is the implicit default")


def test_only_xy_used():
    """dim=2 (matches STPath's own real Attention class) — changing z must
    not change the bias at all."""
    torch.manual_seed(0)
    mod = FrameAveragingBias(n_heads=4)
    coords = torch.randn(8, 3) * 3
    coords_diff_z = coords.clone()
    coords_diff_z[:, 2] = torch.randn(8) * 1000.0  # wildly different z
    assert torch.allclose(mod(coords), mod(coords_diff_z), atol=1e-5), (
        "z should have zero effect on the bias — this class only ever uses xy, matching STPath's own real usage"
    )
    print("[FrameAveragingBias] OK — z has no effect (only xy used, matches STPath's real dim=2)")


if __name__ == "__main__":
    test_shape_and_finiteness()
    test_rotation_reflection_translation_invariance()
    test_not_invariant_to_nonrigid_change()
    test_gradient_flow()
    test_coord_scale_default_preserves_behavior()
    test_only_xy_used()
    print("\nAll FrameAveragingBias tests passed.")
