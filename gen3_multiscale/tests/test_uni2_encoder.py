"""Integration audit finding #6 (six-launch-blocker follow-up):
FrozenUNI2TileEncoder.device was accepted but never used, and dense-WSI
tiles (256x256) were never resized to the 224x224 UNI2-h expects.
`__init__` needs the real `timm` package (not installed in this sandbox
-- GEN4_CONTRACT.md section 13's documented gap), so
`encode_available_patches`'s real preprocessing logic (the actual bug)
is exercised directly against a bypassed-__init__ instance carrying a
tiny, real (not stubbed-away) nn.Module standing in for `self.model` --
proves the resize/normalize/device-placement pipeline itself, not
UNI2-h's real weights."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.uni2_encoder import FrozenUNI2TileEncoder


class _RecordingModel(nn.Module):
    """Records the exact tensor shape/device it was called with, and
    returns a fixed-width embedding per row -- lets tests assert on the
    REAL input the model received, not merely on the final output."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.last_input_shape = None
        self.last_input_device = None
        self.proj = nn.Linear(3, output_dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(tensor.shape)
        self.last_input_device = tensor.device
        return self.proj(tensor.mean(dim=(2, 3)))


def _bare_encoder(output_dim: int = 8, device: str = "cpu") -> FrozenUNI2TileEncoder:
    encoder = FrozenUNI2TileEncoder.__new__(FrozenUNI2TileEncoder)
    nn.Module.__init__(encoder)
    encoder.device = torch.device(device)
    encoder.model = _RecordingModel(output_dim).to(encoder.device)
    encoder.output_dim = output_dim
    return encoder


def test_encode_available_patches_resizes_256_tiles_to_224():
    """CONFIRMED real bug: HEST's dense-WSI tiles are 256x256
    (scripts/precompute_gigapath_wsi_tiles.py's own output_size=256), but
    were previously fed to UNI2-h unresized -- dynamic_img_size=True let
    timm silently accept the wrong size instead of raising."""
    encoder = _bare_encoder()
    patches = np.random.default_rng(0).integers(0, 255, size=(3, 256, 256, 3)).astype(np.uint8)
    encoder.encode_available_patches(patches)
    assert encoder.model.last_input_shape == (3, 3, 224, 224)


def test_encode_available_patches_leaves_already_224_tiles_unresized_in_shape():
    encoder = _bare_encoder()
    patches = np.random.default_rng(0).integers(0, 255, size=(2, 224, 224, 3)).astype(np.uint8)
    encoder.encode_available_patches(patches)
    assert encoder.model.last_input_shape == (2, 3, 224, 224)


def test_encode_available_patches_moves_model_and_inputs_to_the_configured_device():
    """CONFIRMED real bug: `device` was accepted at construction but
    never used anywhere -- model and input tensors always stayed on
    whatever timm.create_model defaulted to (CPU)."""
    encoder = _bare_encoder(device="cpu")
    assert next(encoder.model.parameters()).device == torch.device("cpu")
    patches = np.random.default_rng(0).integers(0, 255, size=(2, 256, 256, 3)).astype(np.uint8)
    encoder.encode_available_patches(patches)
    assert encoder.model.last_input_device == torch.device("cpu")


def test_encode_available_patches_returns_finite_cpu_numpy_output():
    encoder = _bare_encoder(output_dim=8)
    patches = np.random.default_rng(0).integers(0, 255, size=(4, 256, 256, 3)).astype(np.uint8)
    out = encoder.encode_available_patches(patches)
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
