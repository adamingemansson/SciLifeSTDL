"""Alternative gene-coexpression basis source for the MK conditional WAE:
scFoundation's pretrained per-gene positional embeddings, instead of a
from-scratch SVD fit on this project's own training expression
(coexpression.py::fit_conditional_wae_gene_coexpression_basis).

Grounded directly in `gen3_multiscale.gen4.scfoundation_encoder.
FrozenSCFoundationEncoder`'s real, verified-checkpoint forward pass:
`self.model.pos_emb` is an `nn.Embedding` indexed by each gene's fixed
integer position in scFoundation's own ~19,264-gene vocabulary (see
`scfoundation_encoder.py`'s own docstring and its `data_gene_ids =
torch.arange(...)` / `position_gene_ids` construction) -- row `i` of
`pos_emb.weight` (for `i < len(scfoundation_vocab)`; the LAST TWO rows
are the two appended resolution/depth tokens, never a real gene) is
that specific gene's own pretrained embedding vector, independent of
any particular cell/spot's expression values.

Genes in our panel with no match in scFoundation's vocabulary get an
all-zero column (contribute nothing through this source for that gene,
never a fabricated pretrained value) -- reported explicitly via
`extract_scfoundation_gene_embedding_table`'s `report`, mirroring
`src/models/stpath_gene_table.py::extract_stpath_gene_embedding_table`'s
established fail-open-but-audited pattern for exactly this situation.

The resulting `[embedding_dim, n_genes]` table is fed directly into
`models.gene_basis.fit_gene_residual_basis` UNCHANGED -- that function
only ever cares that its input is some `[n_rows, n_genes]` matrix to
rank-reduce into an orthonormal `[rank, n_genes]` basis; it has no
opinion on what the rows semantically represent (real training spots
for the from-scratch source, pretrained embedding dimensions here).
The output `GeneResidualBasis` and `GeneCoexpressionRefinement` are
therefore drop-in identical regardless of source -- `_build_model`/
`contract.py` need no source-specific branching at all.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from gen3_multiscale.gen4.providers import EncoderIdentity
from gen3_multiscale.models.gene_basis import GeneResidualBasis, fit_gene_residual_basis

_REQUIRED_METADATA_FIELDS_SCFOUNDATION = {
    "kind", "rank", "fit_seed", "embedding_table_sha256",
    "scfoundation_checkpoint_sha256", "scfoundation_pinned_revision",
    "scfoundation_package_version", "scfoundation_preprocessing_spec",
    "n_genes_found_in_scfoundation_vocab", "n_genes_missing_from_scfoundation_vocab",
    "missing_genes",
}


def extract_scfoundation_gene_embedding_table(encoder) -> tuple[np.ndarray, dict]:
    """`encoder` is a real, already-constructed, already-checkpoint-
    verified `gen4.scfoundation_encoder.FrozenSCFoundationEncoder` (its
    own `__init__` refuses a missing/unverified checkpoint, repository,
    or vocabulary file -- this function never loads anything itself).

    Returns `(table, report)`: `table` is `[embedding_dim,
    len(encoder.gene_names)]` float32 -- the SAME `[n_rows, n_genes]`
    orientation `fit_gene_residual_basis` already expects, so it can be
    passed straight through with no transpose. `report` records exactly
    which of our genes matched scFoundation's vocabulary and which
    didn't (`n_found`/`n_missing`/`missing_genes`), mirroring
    `stpath_gene_table.py`'s own coverage-report convention."""
    gene_names = list(encoder.gene_names)
    scfoundation_vocab = list(encoder.scfoundation_vocab)
    vocab_position = {gene: idx for idx, gene in enumerate(scfoundation_vocab)}

    pos_emb_weight = encoder.model.pos_emb.weight.detach().to("cpu")
    n_vocab_genes = len(scfoundation_vocab)
    if pos_emb_weight.shape[0] < n_vocab_genes:
        raise ValueError(
            f"scFoundation pos_emb has {pos_emb_weight.shape[0]} rows, fewer than its own "
            f"{n_vocab_genes}-gene vocabulary -- checkpoint/vocabulary mismatch"
        )
    embedding_dim = int(pos_emb_weight.shape[1])

    table = torch.zeros(embedding_dim, len(gene_names), dtype=torch.float32)
    found, missing = [], []
    for column, gene in enumerate(gene_names):
        vocab_index = vocab_position.get(gene)
        # The last two pos_emb rows are the appended resolution/depth
        # tokens (see this module's docstring), never a real gene --
        # exclude them even if a gene name were pathologically identical.
        if vocab_index is None or vocab_index >= n_vocab_genes:
            missing.append(gene)
            continue
        table[:, column] = pos_emb_weight[vocab_index].float()
        found.append(gene)

    if not found:
        raise ValueError(
            f"none of the {len(gene_names)} panel genes appear in scFoundation's own "
            f"{n_vocab_genes}-gene vocabulary -- refusing to fit a basis from an all-zero table"
        )
    report = {
        "n_genes": len(gene_names), "n_found": len(found), "n_missing": len(missing),
        "missing_genes": missing, "embedding_dim": embedding_dim,
    }
    return table.numpy().astype(np.float32), report


def fit_conditional_wae_gene_coexpression_basis_from_scfoundation(
    embedding_table: np.ndarray,
    gene_names: list[str],
    encoder_identity: EncoderIdentity,
    report: dict,
    *,
    rank: int = 64,
    seed: int = 0,
    svd_device: str = "cpu",
) -> tuple[GeneResidualBasis, dict]:
    """Rank-reduce `embedding_table` (`[embedding_dim, n_genes]`, from
    `extract_scfoundation_gene_embedding_table`) into a `GeneResidualBasis`
    via the SAME `fit_gene_residual_basis` the from-scratch source uses --
    only the input matrix's origin differs, not the fitting math, the
    output type, or anything downstream that consumes it."""
    matrix = np.asarray(embedding_table, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(gene_names):
        raise ValueError(f"embedding_table has shape {matrix.shape}, expected [*, {len(gene_names)}]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scFoundation gene embedding table contains non-finite values")

    basis = fit_gene_residual_basis(matrix, gene_names, rank=rank, random_state=seed, svd_device=svd_device)
    content_hash = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    metadata = {
        "kind": "conditional_wae_gene_coexpression_basis_scfoundation",
        "rank": int(basis.rank),
        "fit_seed": int(seed),
        "embedding_table_sha256": content_hash,
        "scfoundation_checkpoint_sha256": encoder_identity.checkpoint_sha256,
        "scfoundation_pinned_revision": encoder_identity.pinned_revision,
        "scfoundation_package_version": encoder_identity.package_version,
        "scfoundation_preprocessing_spec": encoder_identity.preprocessing_spec,
        "n_genes_found_in_scfoundation_vocab": int(report["n_found"]),
        "n_genes_missing_from_scfoundation_vocab": int(report["n_missing"]),
        "missing_genes": list(report["missing_genes"]),
    }
    missing_fields = _REQUIRED_METADATA_FIELDS_SCFOUNDATION - set(metadata)
    if missing_fields:  # pragma: no cover - internal consistency guard
        raise RuntimeError(f"internal error: metadata is missing field(s) {sorted(missing_fields)}")
    return basis, metadata
