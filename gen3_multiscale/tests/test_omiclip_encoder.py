"""Tests for gen3_multiscale/gen4/omiclip_encoder.py. `__init__` needs the
real `open_clip_torch` package (not installed in this sandbox), so
`encode_available_patches`'s real preprocessing/encode pipeline is
exercised directly against a bypassed-__init__ instance carrying a stub
`self.preprocess` (a real, deterministic PIL->tensor transform, standing
in for open_clip's own eval-time transform) and a tiny, real nn.Module
standing in for `self.model` (with an `encode_image` method, matching
open_clip's own CoCa model interface) -- proves the real batching/device-
placement/output-validation logic, not OmiCLIP's real pretrained weights.
Also covers FrozenOmiCLIPTileEncoder's fail-closed construction checks
(missing checkpoint / wrong model name / malformed revision), all of which
happen before `open_clip` is ever imported."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from gen3_multiscale.gen4.omiclip_encoder import FrozenOmiCLIPTileEncoder

_VALID_REVISION = "d072f48609bec7ec4d2c43889262b3029bb1279f"  # 40 lowercase hex chars


class _RecordingModel(nn.Module):
    """Records the exact tensor shape/device `encode_image` was called
    with, and returns a fixed-width embedding per row."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.last_input_shape = None
        self.last_input_device = None
        self.proj = nn.Linear(3, output_dim)

    def encode_image(self, tensor: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(tensor.shape)
        self.last_input_device = tensor.device
        return self.proj(tensor.mean(dim=(2, 3)))


def _stub_preprocess(image: Image.Image) -> torch.Tensor:
    """Deterministic PIL->tensor stand-in for open_clip's own eval
    transform: channel-first float tensor in [0, 1], no resizing (this
    test's patches are already a fixed small size)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _bare_encoder(output_dim: int = 8, device: str = "cpu") -> FrozenOmiCLIPTileEncoder:
    encoder = FrozenOmiCLIPTileEncoder.__new__(FrozenOmiCLIPTileEncoder)
    nn.Module.__init__(encoder)
    encoder.device = torch.device(device)
    encoder.model = _RecordingModel(output_dim).to(encoder.device)
    encoder.preprocess = _stub_preprocess
    encoder.output_dim = output_dim
    return encoder


def test_encode_available_patches_runs_every_patch_through_the_preprocess_transform():
    encoder = _bare_encoder(output_dim=8)
    patches = np.random.default_rng(0).integers(0, 255, size=(3, 16, 16, 3)).astype(np.uint8)
    encoder.encode_available_patches(patches)
    assert encoder.model.last_input_shape == (3, 3, 16, 16)


def test_encode_available_patches_moves_inputs_to_the_configured_device():
    encoder = _bare_encoder(output_dim=8, device="cpu")
    assert next(encoder.model.parameters()).device == torch.device("cpu")
    patches = np.random.default_rng(0).integers(0, 255, size=(2, 8, 8, 3)).astype(np.uint8)
    encoder.encode_available_patches(patches)
    assert encoder.model.last_input_device == torch.device("cpu")


def test_encode_available_patches_returns_finite_cpu_numpy_output_with_correct_shape():
    encoder = _bare_encoder(output_dim=8)
    patches = np.random.default_rng(0).integers(0, 255, size=(4, 8, 8, 3)).astype(np.uint8)
    out = encoder.encode_available_patches(patches)
    assert out.shape == (4, 8)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_encode_available_patches_accepts_float_zero_one_patches():
    encoder = _bare_encoder(output_dim=8)
    patches = np.random.default_rng(0).random(size=(2, 8, 8, 3)).astype(np.float32)
    out = encoder.encode_available_patches(patches)
    assert out.shape == (2, 8)
    assert np.isfinite(out).all()


def test_encode_available_patches_rejects_wrong_shape():
    encoder = _bare_encoder(output_dim=8)
    with pytest.raises(ValueError, match="H, W, 3"):
        encoder.encode_available_patches(np.zeros((3, 8, 8, 4), dtype=np.uint8))


def test_encode_available_patches_raises_on_unexpected_model_output_shape():
    encoder = _bare_encoder(output_dim=8)

    class _WrongShapeModel(nn.Module):
        def encode_image(self, tensor):
            return torch.zeros(tensor.shape[0], 4)  # wrong width

    encoder.model = _WrongShapeModel()
    patches = np.random.default_rng(0).integers(0, 255, size=(2, 8, 8, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="expected"):
        encoder.encode_available_patches(patches)


# -- fail-closed construction checks (no open_clip import required) ----------

def test_init_rejects_a_missing_checkpoint_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        FrozenOmiCLIPTileEncoder(str(tmp_path / "missing.pt"), _VALID_REVISION)


def test_init_rejects_a_non_default_model_name(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub")
    with pytest.raises(ValueError, match="coca_ViT-L-14"):
        FrozenOmiCLIPTileEncoder(str(checkpoint), _VALID_REVISION, model_name="ViT-B-32")


def test_init_rejects_a_malformed_revision(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub")
    with pytest.raises(ValueError, match="revision"):
        FrozenOmiCLIPTileEncoder(str(checkpoint), "not-a-real-revision")


def test_init_rejects_a_short_revision(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub")
    with pytest.raises(ValueError, match="revision"):
        FrozenOmiCLIPTileEncoder(str(checkpoint), _VALID_REVISION[:39])
