"""Frozen OmiCLIP tile encoder.

OmiCLIP (Wang et al., "A visual-omics foundation model to bridge
histopathology with spatial transcriptomics", Nature Methods 2026) is a
CoCa (Contrastive Captioner) model with a ViT-L/14 vision tower, pretrained
on 2.2M paired (H&E patch, spatial-transcriptomics expression) tiles across
32 organs -- unlike GigaPath/UNI2, its own pretraining objective was never
image-only; the image tower's representations are shaped by real,
coupled expression supervision (confirmed via the real Loki source,
github.com/GuangyuWangLab2021/Loki/blob/main/src/loki/utils.py:
`create_model_from_pretrained("coca_ViT-L-14", device=device,
pretrained=model_path)` from the `open_clip` package). Same fail-closed-
on-missing-checkpoint discipline as `uni2_encoder.py`/
`models/slide_encoder.py::FrozenGigaPathSlideEncoder`: a real, local
checkpoint file and a pinned, immutable revision string are both mandatory
at construction time; nothing here ever downloads a checkpoint or silently
constructs a randomly-initialized model. `open_clip` is imported lazily so
importing this module does not require it to be installed unless a caller
actually constructs a `FrozenOmiCLIPTileEncoder`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.providers import EncoderIdentity

_VALID_REVISION_LENGTH = 40
_MODEL_NAME = "coca_ViT-L-14"
_OUTPUT_DIM = 768  # coca_ViT-L-14's projected image-embedding width (open_clip model_configs)


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
            f"OmiCLIP revision must be a 40-character lowercase hex commit SHA, got {revision!r} -- "
            "an unpinned/mutable revision (e.g. a branch name) is not permitted"
        )
    return revision


class FrozenOmiCLIPTileEncoder(nn.Module):
    """Frozen OmiCLIP (CoCa ViT-L/14) tile encoder. Encodes independent
    H&E tile patches through the vision tower's own projected image
    embedding (`model.encode_image`) -- no spatial mixing across tiles
    happens inside this class, mirroring `FrozenUNI2TileEncoder`'s own
    scope."""

    def __init__(
        self,
        checkpoint_path: str,
        revision: str,
        model_name: str = _MODEL_NAME,
        output_dim: int = _OUTPUT_DIM,
        device: str = "cpu",
    ):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"OmiCLIP checkpoint not found: {path}. Download the official pinned "
                "WangGuangyuLab/Loki checkpoint.pt to this exact path -- a randomly "
                "initialized model is never permitted."
            )
        if model_name != _MODEL_NAME:
            raise ValueError(
                f"FrozenOmiCLIPTileEncoder only ever constructs {_MODEL_NAME!r} -- never fall back "
                f"to a different architecture name (got {model_name!r})"
            )
        revision = _validate_immutable_revision(revision)
        try:
            import open_clip
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise ImportError(
                "The `open_clip_torch` package is required to construct FrozenOmiCLIPTileEncoder. "
                "Install it in the training environment."
            ) from exc

        self.checkpoint_sha256 = _sha256_file(path)
        # Mirrors the real, confirmed Loki source
        # (github.com/GuangyuWangLab2021/Loki/blob/main/src/loki/utils.py) exactly --
        # `create_model_from_pretrained` (2-tuple: model, eval-only preprocess), not
        # `create_model_and_transforms` (3-tuple incl. a train-time transform we never use).
        # weights_only=False: this checkpoint predates PyTorch 2.6's weights_only=True
        # default and contains a plain numpy.core.multiarray.scalar the default
        # safe-globals allowlist rejects. Safe here -- checkpoint_path is already
        # required to be a real, already-downloaded local file (never fetched by this
        # class), and its sha256 is recorded in this encoder's own identity below.
        model, eval_preprocess = open_clip.create_model_from_pretrained(
            model_name, pretrained=str(path), device="cpu", weights_only=False,
        )
        if not hasattr(model, "encode_image"):
            raise RuntimeError(
                f"OmiCLIP checkpoint {path} loaded a model with no encode_image method -- "
                f"refusing to proceed with an unexpected architecture"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        self.model = model
        self.preprocess = eval_preprocess
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.output_dim = int(output_dim)
        self.identity = EncoderIdentity(
            encoder_name="omiclip",
            checkpoint_sha256=self.checkpoint_sha256,
            pinned_revision=revision,
            package_version=str(getattr(open_clip, "__version__", "unknown")),
            preprocessing_spec=f"omiclip_tile_v1:{model_name}:open_clip_eval_transform",
            output_dim=self.output_dim,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def encode_available_patches(self, patches: np.ndarray) -> np.ndarray:
        """`patches` is `[N, H, W, 3]` uint8 or float in `[0, 1]`/`[0, 255]`
        (matching every other tile encoder's own convention in this
        codebase). Each patch is converted to a PIL image and run through
        OmiCLIP's own `open_clip` eval transform (its real resize/crop/
        normalization, not a hand-reconstructed approximation), then
        encoded via `model.encode_image` -- the same projected embedding
        space OmiCLIP's own paired image-expression contrastive
        pretraining directly optimized."""
        if patches.ndim != 4 or patches.shape[-1] != 3:
            raise ValueError(f"patches must be [N, H, W, 3], got shape {patches.shape}")
        from PIL import Image

        array = np.ascontiguousarray(patches)
        if array.dtype != np.uint8:
            if array.max() <= 1.5:
                array = (array * 255.0).round()
            array = array.astype(np.uint8)
        tensors = [self.preprocess(Image.fromarray(array[i])) for i in range(array.shape[0])]
        batch = torch.stack(tensors, dim=0).to(self.device)
        out = self.model.encode_image(batch).detach().to("cpu").numpy().astype(np.float32)
        if out.shape != (patches.shape[0], self.output_dim):
            raise RuntimeError(
                f"OmiCLIP encoder returned features with shape {out.shape}, expected "
                f"({patches.shape[0]}, {self.output_dim})"
            )
        if not np.isfinite(out).all():
            raise RuntimeError("OmiCLIP encoder returned non-finite feature values")
        return out
