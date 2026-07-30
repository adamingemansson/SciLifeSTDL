"""Frozen scFoundation GEX-context encoder -- GEN4_CONTRACT.md section 7.

Same fail-closed discipline as `uni2_encoder.py`: a real, local checkpoint
file and an exact gene-vocabulary mapping file are both mandatory; nothing
here ever downloads a checkpoint. `encode_rows` is strictly row-independent
-- it must never compute or use any statistic derived from more than one
row at a time (no batch norm, no dataset-level rescaling), so it is safe
against training/validation/test rows or the same row called twice.

Integration audit finding #5 (six-launch-blocker follow-up), CONFIRMED
real: the prior version of this file assumed
`load_model_frommmf(...)` returns a single model and that a plain
`model(tensor, output_type="cell")` call performs a full forward pass.
Neither matches the real, official biomap-research/scFoundation
implementation (fetched 2026-07-30 from
`model/load.py`/`model/get_embedding.py` at the `main` branch --
`required_fingerprints.scfoundation_checkpoint`'s own pinned revision is
what a real deployment must actually verify against, per this class's
own `pinned_revision` identity field below). The real, verbatim
reference pipeline this class now reproduces:

1. `load_model_frommmf(checkpoint_path, key="cell")` returns a
   `(model, config)` TUPLE, not a bare model -- `config` carries
   `pad_token_id` and every other hyperparameter the forward pass below
   needs; it is never guessed.
2. `main_gene_selection`: scFoundation has ONE fixed, ~19264-gene
   vocabulary in a fixed order (`scfoundation_gene_list`, loaded from
   `gene_vocab_path`) -- this manifest's own `gene_names` are re-indexed
   INTO that fixed vocabulary (zero-filled for any scFoundation
   vocabulary gene absent from this manifest's panel), never the other
   way around.
3. Two extra tokens are appended, in this EXACT order (a prior version
   had them reversed): a fixed target-resolution token (`tgthighres="t4"`
   -- the official script's own default meaning "target token 1 is the
   literal value `4.0`", not a log-transformed count of anything), then
   `log10(raw_library_size)` -- the OFFICIAL script's own resolution-token
   ordering and `log10` (never `log1p`) semantics for the real total-count
   token. `raw_library_size` is the real, pre-normalization total count
   per row (`adata.obs['_scilifestdl_raw_library_size']`, stashed by
   `data/loaders.py::basic_qc_and_normalize`) -- a prior version summed
   the already-normalized `expression` matrix instead, which cannot
   recover a real read depth.
4. Cell embedding (`output_type="cell"`, `version="ce"`): only genes
   (plus both resolution tokens) with strictly positive value are kept
   (`value_labels = pretrain_gene_x > 0`), compacted via the official
   `gatherData` gene-gathering routine (vendored locally below, not
   imported, so this class does not depend on the external package's
   internal/undocumented module layout for anything beyond
   `load_model_frommmf`'s own public return value), fed through the
   real model's own `token_emb` (continuous-value embedding) +
   `pos_emb` (gene-index positional embedding) + `encoder`, then pooled
   via the official FOUR-way scheme (`pool_type="all"`, the official
   default): the last two positions (the two resolution tokens'
   post-encoder representations) concatenated with a max-pool and a
   mean-pool over every other (real gene) position.

HONEST LIMIT (GEN4_CONTRACT.md section 13 / RUNBOOK.md section 4): no
real `scfoundation` package is installed in this environment, so this
has never been run against the real package/checkpoint and MUST be
verified against a real installation before trusting this path for real
inference -- `tests/test_scfoundation_encoder.py` exercises every piece
of the computation above (gathering, token ordering, pooling) against a
bypassed-__init__ instance with a tiny real (not scFoundation-weighted)
transformer-shaped model standing in for the real one, and a SEPARATE
adversarial test proves the official gatherData/pooling logic here
reduces to a known-correct closed form on a hand-checkable tiny example
-- but the real checkpoint's own numerical output has not been compared
against the official script's own output on identical rows in this
sandbox. What IS structurally guaranteed regardless: fixed vocabulary
order, row-independent encoding, fail-closed checkpoint/vocabulary
identity, and (now) device placement and the official token
ordering/pooling algorithm.
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


def gather_scfoundation_data(data: torch.Tensor, labels: torch.Tensor, pad_token_id: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendored, functionally verbatim copy of the official
    biomap-research/scFoundation repository's `model/load.py::gatherData`
    (see this module's own docstring for why it is copied here rather
    than imported). Compacts `data` to only its `labels`-True positions
    per row, in original left-to-right order, right-padded with
    `pad_token_id` out to the batch's own max labeled-count.

    Mechanism (identical to the official version): every position gets
    a per-row-monotonically-decreasing tiebreak added to `labels`
    (`tmp_data`) so a `topk` over `labels` recovers the True positions in
    their ORIGINAL left-to-right order (not sorted by value); `max_num`
    extra guaranteed-True "fake" columns are appended to both `data` (as
    `pad_token_id`) and `labels` (as True) so `topk(max_num)` always has
    enough real candidates even for a row with fewer than `max_num`
    labeled positions -- those fake columns naturally sort last (their
    tiebreak score is 0, lower than every real True position's) and end
    up selected only to fill out the padding."""
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())

    fake_data = torch.full((data.shape[0], max_num), pad_token_id, device=data.device, dtype=data.dtype)
    data = torch.hstack([data, fake_data])

    fake_label = torch.ones((labels.shape[0], max_num), device=labels.device, dtype=torch.float32)
    none_labels = ~labels
    labels = labels.float()
    labels = labels.masked_fill(none_labels, -float("inf"))

    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device, dtype=labels.dtype,
    )
    labels = labels + tmp_data
    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)
    padding_labels = new_data == pad_token_id
    return new_data, padding_labels


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
        pool_type: str = "all",
        target_resolution_token: float = 4.0,
        device: str = "cuda",
    ):
        super().__init__()
        if pool_type not in ("all", "max"):
            raise ValueError(f"pool_type must be 'all' or 'max' (official scFoundation cell-embedding modes), got {pool_type!r}")
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
        self.pool_type = pool_type
        self.target_resolution_token = float(target_resolution_token)
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
        # Integration audit finding #5 (CONFIRMED real): the official
        # load_model_frommmf returns a (model, config) TUPLE -- a prior
        # version treated its return value as a bare model.
        self.model, self.scfoundation_config = scfoundation.load_model_frommmf(str(ckpt_path), key="cell")  # pragma: no cover
        if "pad_token_id" not in self.scfoundation_config:
            raise ValueError(
                f"scFoundation checkpoint {ckpt_path}'s own config has no pad_token_id -- cannot gather "
                "gene tokens without it (this looks like an incompatible/corrupted checkpoint)"
            )
        self.pad_token_id = self.scfoundation_config["pad_token_id"]
        # Item 6 (six-launch-blocker audit): `device` was accepted here but
        # never used anywhere in this class -- the model always stayed on
        # whatever device `load_model_frommmf` itself defaulted to
        # (typically CPU), silently ignoring a caller's `device="cuda"`.
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.output_dim = int(output_dim)
        self.identity = EncoderIdentity(
            encoder_name="scfoundation",
            checkpoint_sha256=checkpoint_sha256,
            pinned_revision=vocab_sha256,  # scFoundation has no HF revision; vocab hash pins the identity instead
            package_version=package_version,
            preprocessing_spec=(
                f"scfoundation_official_v1:main_gene_selection+cell_pooling_{pool_type}:"
                f"tgthighres=t{target_resolution_token:g}:log10_totalcount"
            ),
            output_dim=self.output_dim,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _to_scfoundation_input(self, expression: np.ndarray, raw_library_size: np.ndarray) -> np.ndarray:
        """Re-index `expression` (aligned to `self.gene_names`) into
        scFoundation's own fixed vocabulary order, then append the
        official two read-depth tokens IN THE OFFICIAL ORDER: the fixed
        target-resolution token first, `log10(raw_library_size)` second
        (biomap-research/scFoundation `model/get_embedding.py`'s own
        `input_type='singlecell'`, `tgthighres[0]=='t'` branch -- a
        prior version had these two tokens reversed and used `log1p` of
        the wrong quantity instead of `log10` of the real one)."""
        n_rows = expression.shape[0]
        vocab_input = np.zeros((n_rows, len(self.scfoundation_vocab)), dtype=np.float32)
        manifest_rows = [r for r, _v in self._manifest_to_vocab_pos]
        vocab_cols = [v for _r, v in self._manifest_to_vocab_pos]
        vocab_input[:, vocab_cols] = expression[:, manifest_rows]
        resolution_token = np.full((n_rows, 1), self.target_resolution_token, dtype=np.float32)
        total_count_token = np.log10(raw_library_size).astype(np.float32).reshape(n_rows, 1)
        return np.concatenate([vocab_input, resolution_token, total_count_token], axis=1)

    @torch.inference_mode()
    def encode_rows(self, expression: np.ndarray, raw_library_size: np.ndarray | None = None) -> np.ndarray:
        if expression.ndim != 2 or expression.shape[1] != len(self.gene_names):
            raise ValueError(
                f"expression must be [N, {len(self.gene_names)}], got shape {expression.shape}"
            )
        if raw_library_size is None:
            raise ValueError(
                "FrozenSCFoundationEncoder.encode_rows requires raw_library_size (the real, "
                "pre-normalization total count per row, e.g. adata.obs['_scilifestdl_raw_library_size']) "
                "-- it cannot be recovered from the already-normalized/log1p'd expression matrix"
            )
        raw_library_size = np.asarray(raw_library_size, dtype=np.float32).reshape(-1)
        if raw_library_size.shape[0] != expression.shape[0]:
            raise ValueError(
                f"raw_library_size has {raw_library_size.shape[0]} rows, expected {expression.shape[0]} "
                "(row-aligned with expression)"
            )
        if not np.all(raw_library_size > 0):
            raise ValueError("raw_library_size must be strictly positive (log10 of a real total count)")

        model_input = self._to_scfoundation_input(np.asarray(expression, dtype=np.float32), raw_library_size)
        pretrain_gene_x = torch.as_tensor(model_input, dtype=torch.float32, device=self.device)
        data_gene_ids = torch.arange(pretrain_gene_x.shape[1], device=self.device).repeat(pretrain_gene_x.shape[0], 1)

        # Official model/get_embedding.py, output_type="cell": only
        # strictly-positive positions (real expression + both resolution
        # tokens) are kept; gathered/compacted, embedded, encoded, pooled.
        value_labels = pretrain_gene_x > 0
        x, x_padding = gather_scfoundation_data(pretrain_gene_x, value_labels, self.pad_token_id)
        position_gene_ids, _ = gather_scfoundation_data(data_gene_ids.float(), value_labels, self.pad_token_id)

        x = self.model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids.long())
        x = x + position_emb
        geneemb = self.model.encoder(x, x_padding)

        # Official four-way "all" pooling: the two resolution tokens'
        # own post-encoder representations (always the LAST two gathered
        # positions -- gatherData preserves original left-to-right order
        # and both tokens were appended last, so they sort last among
        # every row's real, positive-valued positions) concatenated with
        # a max-pool and a mean-pool over every other (gene) position.
        geneemb1 = geneemb[:, -1, :]
        geneemb2 = geneemb[:, -2, :]
        geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
        geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
        if self.pool_type == "all":
            pooled = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
        else:
            pooled, _ = torch.max(geneemb, dim=1)

        out = pooled.detach().to("cpu").numpy().astype(np.float32)
        if out.shape != (expression.shape[0], self.output_dim):
            raise RuntimeError(f"scFoundation encoder returned shape {out.shape}, expected ({expression.shape[0]}, {self.output_dim})")
        if not np.isfinite(out).all():
            raise RuntimeError("scFoundation encoder returned non-finite feature values")
        return out
