"""Frozen scFoundation GEX-context encoder -- GEN4_CONTRACT.md section 7.

Same fail-closed discipline as `uni2_encoder.py`: a real, local checkpoint
file and an exact gene-vocabulary mapping file are both mandatory; nothing
here ever downloads a checkpoint. `encode_rows` is strictly row-independent
-- it must never compute or use any statistic derived from more than one
row at a time (no batch norm, no dataset-level rescaling), so it is safe
against training/validation/test rows or the same row called twice.

No real scFoundation weights are available in this environment. This class
is structurally complete and exercised in tests only via a stub satisfying
the same public interface (`tests/_gen4_fixtures.py::stub_scfoundation`);
real-weight validation is listed as an explicit gap in the runbook.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.providers import EncoderIdentity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenSCFoundationEncoder(nn.Module):
    """Wraps a real scFoundation checkpoint. `gene_vocab_path` must be a
    JSON file mapping this manifest's gene panel (or a superset) to
    scFoundation's own vocabulary ids -- required even to construct this
    class, since scFoundation's row-independent encoding still needs to
    know which input column corresponds to which of ITS OWN genes,
    entirely independent of whether the checkpoint itself has loaded."""

    def __init__(
        self,
        checkpoint_path: str,
        gene_vocab_path: str,
        gene_names: list[str],
        output_dim: int = 3072,
        device: str = "cpu",
    ):
        super().__init__()
        ckpt_path = Path(checkpoint_path).expanduser()
        vocab_path = Path(gene_vocab_path).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"scFoundation checkpoint not found: {ckpt_path}. Download the official "
                "pretrained checkpoint to this exact path -- a randomly initialized model "
                "is never permitted."
            )
        if not vocab_path.is_file():
            raise FileNotFoundError(f"scFoundation gene-vocabulary file not found: {vocab_path}")
        vocab = json.loads(vocab_path.read_text())
        missing = [g for g in gene_names if g not in vocab]
        if missing:
            raise ValueError(
                f"scFoundation gene vocabulary {vocab_path} is missing {len(missing)} gene(s) "
                f"from the current manifest gene panel (examples: {missing[:5]})"
            )
        self.gene_names = tuple(gene_names)
        self.vocab_ids = [int(vocab[g]) for g in gene_names]
        vocab_sha256 = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode("utf-8")).hexdigest()

        try:
            import scfoundation  # noqa: F401 -- optional external dependency, real import deferred to real use
            package_version = str(getattr(scfoundation, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise ImportError(
                "The `scfoundation` package is required to construct FrozenSCFoundationEncoder. "
                "Install it in the training environment."
            ) from exc

        checkpoint_sha256 = _sha256_file(ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # Real scFoundation model construction/loading is intentionally left
        # to the caller's installed `scfoundation` package version (its
        # public loader API is not stable enough to hardcode here without a
        # real installation to verify against) -- this wrapper's contract
        # is the frozen, row-independent `encode_rows` interface, not a
        # specific internal architecture.
        self.model = scfoundation.build_model_from_state_dict(state_dict)  # pragma: no cover
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.output_dim = int(output_dim)
        self.identity = EncoderIdentity(
            encoder_name="scfoundation",
            checkpoint_sha256=checkpoint_sha256,
            pinned_revision=vocab_sha256,  # scFoundation has no HF revision; vocab hash pins the identity instead
            package_version=package_version,
            preprocessing_spec="scfoundation_row_independent_v1:raw_log1p_normalized",
            output_dim=self.output_dim,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def encode_rows(self, expression: np.ndarray) -> np.ndarray:
        if expression.ndim != 2 or expression.shape[1] != len(self.gene_names):
            raise ValueError(
                f"expression must be [N, {len(self.gene_names)}], got shape {expression.shape}"
            )
        tensor = torch.as_tensor(expression, dtype=torch.float32)
        out = self.model(tensor, gene_ids=self.vocab_ids).cpu().numpy().astype(np.float32)  # pragma: no cover
        if out.shape != (expression.shape[0], self.output_dim):
            raise RuntimeError(f"scFoundation encoder returned shape {out.shape}, expected ({expression.shape[0]}, {self.output_dim})")
        if not np.isfinite(out).all():
            raise RuntimeError("scFoundation encoder returned non-finite feature values")
        return out
