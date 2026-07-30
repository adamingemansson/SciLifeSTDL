"""Frozen scFoundation GEX-context encoder -- GEN4_CONTRACT.md section 7.

Same fail-closed discipline as `uni2_encoder.py`: a real, local checkpoint
file and an exact gene-vocabulary mapping file are both mandatory; nothing
here ever downloads a checkpoint. `encode_rows` is strictly row-independent
-- it must never compute or use any statistic derived from more than one
row at a time (no batch norm, no dataset-level rescaling), so it is safe
against training/validation/test rows or the same row called twice.

Codex audit finding #2 (of the first Gen4 push), confirmed real: the
original version of this file assumed an arbitrary `model(tensor,
gene_ids=...)` call shape that does not match the official scFoundation
API (biomap-research/scFoundation, `model/load.py` /
`model/pretrainmodels.py`). The real, documented preprocessing this class
now reflects, from the official repository's own `main_gene_selection`
helper and the scFoundation paper's own description of its read-depth
tokens:

1. scFoundation has ONE fixed, ~19264-gene vocabulary, in a fixed order --
   `scfoundation_gene_list` (loaded from `gene_vocab_path`, a JSON array
   of gene symbols in that exact fixed order) is that vocabulary, NOT an
   arbitrary caller-supplied gene->id mapping. This manifest's own
   `gene_names` are re-indexed INTO that fixed vocabulary (zero-filled
   for any scFoundation vocabulary gene absent from this manifest's
   panel) -- never the other way around.
2. Two extra "total count" tokens (log1p of each row's own total raw
   count, and a target total-count value) are appended as the final two
   input positions before encoding -- scFoundation's own read-depth-aware
   token design, not a per-dataset-statistic (each row's own total count
   is a row-local, row-independent quantity, so this does not violate the
   row-independence contract above).
3. `load_model_frommmf(checkpoint_path, key="cell")` loads a
   cell-embedding-mode checkpoint; the real forward call requests
   `output_type="cell"` to get one pooled per-row embedding (scFoundation
   also supports a "gene" output mode returning per-gene embeddings --
   NOT what a per-spot GEX-context embedding needs here).

HONEST LIMIT (GEN4_CONTRACT.md section 13 / RUNBOOK.md section 4): no real
`scfoundation` package is installed in this environment, so the exact
`load_model_frommmf`/forward() call signature below has never been run
against the real package and MUST be verified (and corrected if it has
drifted from what is reflected here) against a real installation before
trusting this path for real inference. What IS structurally guaranteed by
this file regardless of that: fixed vocabulary order, row-independent
encoding, and fail-closed checkpoint/vocabulary identity.
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
    JSON file containing scFoundation's own FIXED, ordered gene vocabulary
    (a list of gene symbols, in the exact order the real checkpoint was
    trained with) -- this class re-indexes the manifest's `gene_names`
    into that fixed vocabulary, not the reverse."""

    def __init__(
        self,
        checkpoint_path: str,
        gene_vocab_path: str,
        gene_names: list[str],
        output_dim: int = 3072,
        target_total_count: float = 1e4,
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
        scfoundation_vocab = json.loads(vocab_path.read_text())
        if not isinstance(scfoundation_vocab, list) or not scfoundation_vocab:
            raise ValueError(
                f"scFoundation gene vocabulary {vocab_path} must be a non-empty JSON array of gene "
                "symbols in scFoundation's own fixed vocabulary order, not a name->id mapping"
            )
        self.scfoundation_vocab = list(scfoundation_vocab)
        self.gene_names = tuple(gene_names)
        # Position in scfoundation_vocab for each manifest gene that scFoundation
        # actually has; manifest genes absent from scFoundation's vocabulary are
        # simply not represented in the scFoundation-order input (real gap, not
        # silently invented) -- and scFoundation vocabulary genes absent from the
        # manifest panel stay at their input's default zero.
        vocab_position = {gene: idx for idx, gene in enumerate(self.scfoundation_vocab)}
        self._manifest_to_vocab_pos = [
            (row, vocab_position[gene]) for row, gene in enumerate(gene_names) if gene in vocab_position
        ]
        if not self._manifest_to_vocab_pos:
            raise ValueError(
                f"none of the {len(gene_names)} manifest genes appear in scFoundation's own "
                f"{len(self.scfoundation_vocab)}-gene vocabulary {vocab_path} -- refusing to encode "
                "an all-zero input"
            )
        self.target_total_count = float(target_total_count)
        vocab_sha256 = hashlib.sha256(json.dumps(scfoundation_vocab).encode("utf-8")).hexdigest()

        try:
            import scfoundation  # noqa: F401 -- optional external dependency, real import deferred to real use
            package_version = str(getattr(scfoundation, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - optional external dependency
            raise ImportError(
                "The `scfoundation` package is required to construct FrozenSCFoundationEncoder. "
                "Install it in the training environment."
            ) from exc

        checkpoint_sha256 = _sha256_file(ckpt_path)
        # Real API per the official repository's model/load.py -- see this
        # class's own docstring for the HONEST LIMIT on this call's
        # verification status in this environment.
        self.model = scfoundation.load_model_frommmf(str(ckpt_path), key="cell")  # pragma: no cover
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.output_dim = int(output_dim)
        self.identity = EncoderIdentity(
            encoder_name="scfoundation",
            checkpoint_sha256=checkpoint_sha256,
            pinned_revision=vocab_sha256,  # scFoundation has no HF revision; vocab hash pins the identity instead
            package_version=package_version,
            preprocessing_spec="scfoundation_row_independent_v2:fixed_vocab_reindex+read_depth_tokens",
            output_dim=self.output_dim,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _to_scfoundation_input(self, expression: np.ndarray) -> np.ndarray:
        """Re-index `expression` (aligned to `self.gene_names`) into
        scFoundation's own fixed vocabulary order, then append the two
        read-depth tokens. Each row's total count is that row's OWN sum
        -- a row-local quantity, never a batch/dataset statistic."""
        n_rows = expression.shape[0]
        vocab_input = np.zeros((n_rows, len(self.scfoundation_vocab)), dtype=np.float32)
        manifest_rows = [r for r, _v in self._manifest_to_vocab_pos]
        vocab_cols = [v for _r, v in self._manifest_to_vocab_pos]
        vocab_input[:, vocab_cols] = expression[:, manifest_rows]
        total_count_token = np.log1p(expression.sum(axis=1, keepdims=True)).astype(np.float32)
        target_token = np.full((n_rows, 1), np.log1p(self.target_total_count), dtype=np.float32)
        return np.concatenate([vocab_input, total_count_token, target_token], axis=1)

    @torch.inference_mode()
    def encode_rows(self, expression: np.ndarray) -> np.ndarray:
        if expression.ndim != 2 or expression.shape[1] != len(self.gene_names):
            raise ValueError(
                f"expression must be [N, {len(self.gene_names)}], got shape {expression.shape}"
            )
        model_input = self._to_scfoundation_input(np.asarray(expression, dtype=np.float32))
        tensor = torch.as_tensor(model_input, dtype=torch.float32)
        out = self.model(tensor, output_type="cell").cpu().numpy().astype(np.float32)  # pragma: no cover
        if out.shape != (expression.shape[0], self.output_dim):
            raise RuntimeError(f"scFoundation encoder returned shape {out.shape}, expected ({expression.shape[0]}, {self.output_dim})")
        if not np.isfinite(out).all():
            raise RuntimeError("scFoundation encoder returned non-finite feature values")
        return out
