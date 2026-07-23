"""Regression test for a real bug (2026-07-23): DINOv2 preprocessing
hardcoded a resize to 224x224 (matching Gigapath's own convention), but
timm's vit_large_patch14_dinov2.lvd142m checkpoint actually expects
518x518 -- crashed on real hardware with "Input height (224) doesn't
match model (518)." Fixed by reading the loaded model's own
patch_embed.img_size instead of assuming a constant. Uses a fake
minimal model object (no real DINOv2 download needed) to verify the
preprocessing step resizes to whatever the model actually reports.
"""
import torch
import torch.nn as nn

from src.models.conditioning import _dinov2_expected_size, _dinov2_preprocess_and_encode


class _FakePatchEmbed:
    def __init__(self, size):
        self.img_size = (size, size)


class _FakeDinov2(nn.Module):
    def __init__(self, expected_size, feat_dim=1024):
        super().__init__()
        self.patch_embed = _FakePatchEmbed(expected_size)
        self.feat_dim = feat_dim
        self.seen_shapes = []

    def forward(self, x):
        self.seen_shapes.append(tuple(x.shape))
        return torch.randn(x.shape[0], self.feat_dim)


def test_dinov2_expected_size_reads_the_real_model_not_a_constant():
    model = _FakeDinov2(expected_size=518)
    assert _dinov2_expected_size(model) == 518

    model_224 = _FakeDinov2(expected_size=224)
    assert _dinov2_expected_size(model_224) == 224
    print("[dinov2_encoder] OK — _dinov2_expected_size reads the real loaded model's "
          "patch_embed.img_size, not a hardcoded constant")


def test_dinov2_preprocess_resizes_to_the_models_real_expected_size():
    model = _FakeDinov2(expected_size=518)
    patches = torch.rand(3, 3, 224, 224)  # arbitrary input resolution (e.g. Gigapath's own 224 patches)
    out = _dinov2_preprocess_and_encode(model, patches)
    assert out.shape == (3, 1024)
    assert model.seen_shapes[-1][-2:] == (518, 518), (
        f"expected the model to be called with 518x518 input, got {model.seen_shapes[-1]}"
    )
    print("[dinov2_encoder] OK — preprocessing resizes to the model's real expected "
          "size (518x518), reproducing and fixing the exact real crash "
          "('Input height (224) doesn't match model (518)')")


def test_dinov2_preprocess_adapts_to_a_224_variant_too():
    model = _FakeDinov2(expected_size=224)
    patches = torch.rand(2, 3, 518, 518)
    out = _dinov2_preprocess_and_encode(model, patches)
    assert out.shape == (2, 1024)
    assert model.seen_shapes[-1][-2:] == (224, 224)
    print("[dinov2_encoder] OK — also adapts correctly for a smaller-resolution DINOv2 variant")


if __name__ == "__main__":
    test_dinov2_expected_size_reads_the_real_model_not_a_constant()
    test_dinov2_preprocess_resizes_to_the_models_real_expected_size()
    test_dinov2_preprocess_adapts_to_a_224_variant_too()
    print("\nAll dinov2_encoder tests passed.")
