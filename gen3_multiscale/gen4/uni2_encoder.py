"""Frozen UNI2 tile encoder -- GEN4_CONTRACT.md section 6.

Same fail-closed-on-missing-checkpoint discipline as
`models/slide_encoder.py::FrozenGigaPathSlideEncoder`: a real, local
checkpoint file and a pinned, immutable revision string are both mandatory
at construction time; nothing here ever downloads a checkpoint or silently
constructs a randomly-initialized model. `timm` is imported lazily (like
every other optional-heavy-dependency import in this codebase) so importing
this module does not require `timm` to be installed unless a caller
actually constructs a `FrozenUNI2TileEncoder`.

No real UNI2 weights are available in this environment (downloading large
weights is explicitly out of scope for this suite -- GEN4_CONTRACT.md
section 13). This class is structurally complete and exercised in tests
only via a stub satisfying the same public interface
(`tests/_gen4_fixtures.py::stub_uni2`); real-weight validation is listed as
an explicit gap in the runbook.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.providers import EncoderIdentity

_VALID_REVISION_LENGTH = 40


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_immutable_revision(revision: str) -> str:
    revision = str(revision).strip()
    if len(revision) != _VALID_REVISION_LENGTH or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(
            f"UNI2 revision must be a 40-character lowercase hex commit SHA, got {revision!r} -- "
            "an unpinned/mutable revision (e.g. a branch name) is not permitted"
        )
    return revision


class FrozenUNI2TileEncoder(nn.Module):
    """Frozen official UNI2 tile encoder (per-tile ViT). Encodes independent
    H&E tile patches -- no spatial mixing across tiles happens inside this
    class (that is `uni2_global_pool.py`'s job, run separately on this
    encoder's already-produced per-tile features)."""

    # "uni2-h" is not a registered timm architecture -- Codex audit finding #2
    # (of the first Gen4 push), confirmed real: `timm.create_model("uni2-h")`
    # raises immediately, since UNI2-h is not a named timm model but a
    # specific parameterization of `vit_giant_patch14_224`. This is the
    # real, official construction (MahmoodLab/UNI2-h model card / the
    # mahmoodlab/UNI GitHub repository's own loading example) -- a plain
    # ViT-Giant/14 with UNI2-h's documented non-default arguments
    # (register tokens, SwiGLU MLP, no class-token positional embedding).
    _TIMM_BASE_MODEL = "vit_giant_patch14_224"
    _TIMM_KWARGS = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        reg_tokens=8, dynamic_img_size=True,
    )
    # Official MahmoodLab/UNI2-h model card: 224x224 input, ImageNet
    # normalization -- see encode_available_patches's own docstring
    # (Integration audit finding #6) for why this must be enforced
    # explicitly rather than trusted to dynamic_img_size.
    _INPUT_SIZE = 224

    def __init__(
        self,
        checkpoint_path: str,
        revision: str,
        model_name: str = _TIMM_BASE_MODEL,
        output_dim: int = 1536,
        device: str = "cpu",
    ):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"UNI2 checkpoint not found: {path}. Download the official pinned "
                "MahmoodLab/UNI2-h checkpoint to this exact path -- a randomly "
                "initialized model is never permitted."
            )
        if model_name != self._TIMM_BASE_MODEL:
            raise ValueError(
                f"FrozenUNI2TileEncoder only ever constructs {self._TIMM_BASE_MODEL!r} with UNI2-h's "
                f"documented non-default arguments -- never fall back to a different UNI model or a "
                f"caller-supplied architecture name (got {model_name!r})"
            )
        revision = _validate_immutable_revision(revision)
        try:
            import timm
            from timm.layers import SwiGLUPacked
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise ImportError(
                "The `timm` package (with timm.layers.SwiGLUPacked) is required to construct "
                "FrozenUNI2TileEncoder. Install it in the training environment."
            ) from exc

        self.checkpoint_sha256 = _sha256_file(path)
        self.model = timm.create_model(
            self._TIMM_BASE_MODEL, pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
            **self._TIMM_KWARGS,
        )
        state_dict = torch.load(path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"UNI2 checkpoint {path} does not match model_name={model_name!r} exactly "
                f"(missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}) -- refusing to "
                "silently proceed with a partially-loaded model"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        # Integration audit finding #6 (CONFIRMED real): `device` was
        # accepted but never used -- the model (and every input tensor
        # below) always stayed on whatever device timm.create_model
        # defaulted to (CPU), silently ignoring device="cuda".
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.output_dim = int(output_dim)
        self.identity = EncoderIdentity(
            encoder_name="uni2",
            checkpoint_sha256=self.checkpoint_sha256,
            pinned_revision=revision,
            package_version=str(getattr(timm, "__version__", "unknown")),
            preprocessing_spec=f"uni2_tile_v1:{model_name}:resize224:imagenet_norm",
            output_dim=self.output_dim,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        """Integration audit finding #6 (CONFIRMED real): the official
        MahmoodLab/UNI2-h model card specifies 224x224 input with
        ImageNet normalization -- this class's own `preprocessing_spec`
        already CLAIMED "resize224", but the prior version never actually
        resized: HEST's dense-WSI tiles are produced at 256x256 (see
        `scripts/precompute_gigapath_wsi_tiles.py`'s own `_tile_grid`,
        `output_size=256`) and were fed straight through. `dynamic_img_size
        =True` (self._TIMM_KWARGS) let timm silently accept the wrong
        input size without erroring -- a different effective patch grid
        than UNI2-h was actually trained/documented for, not a crash a
        caller would notice. Patches are now resized to exactly
        `_INPUT_SIZE` (224) via bilinear interpolation before
        normalization, on whichever spatial size they arrive at, so the
        code matches what `preprocessing_spec` already claimed."""
        if patches.ndim != 4 or patches.shape[-1] != 3:
            raise ValueError(f"patches must be [N, H, W, 3], got shape {patches.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(patches)).permute(0, 3, 1, 2).float().to(self.device)
        if tensor.max() > 1.5:  # heuristically 0-255 range, matches spot_feature_cache's own convention
            tensor = tensor.div(255.0)
        if tensor.shape[-2:] != (self._INPUT_SIZE, self._INPUT_SIZE):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self._INPUT_SIZE, self._INPUT_SIZE), mode="bilinear", align_corners=False,
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        out = self.model(tensor).detach().to("cpu").numpy().astype(np.float32)
        if out.shape != (patches.shape[0], self.output_dim):
            raise RuntimeError(
                f"UNI2 encoder returned features with shape {out.shape}, expected "
                f"({patches.shape[0]}, {self.output_dim})"
            )
        if not np.isfinite(out).all():
            raise RuntimeError("UNI2 encoder returned non-finite feature values")
        return out
