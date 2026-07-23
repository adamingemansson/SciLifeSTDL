"""Frozen STPath gene-embedding table, spliced in ISOLATION from STPath's
whole architecture -- isolates "does STPath's PRETRAINING help" from "does
STPath's whole spatial-transformer fusion help," a distinction this
project's own internal ablation (STPath pretrained vs. STPath unfrozen/
from-scratch, mean PCC 0.503 vs. 0.4706, see docs/results_log.md) cannot
make: that ablation always uses STPath's whole architecture, so a ~0.03-0.04
PCC gap could be explained by either "pretraining helped" or "the
unfrozen/from-scratch run just needed more steps to converge the same
architecture" -- it can't distinguish the two. This module lets ONLY the
pretrained gene_embed weights transfer into our own architecture, decoupled
from STPath's transformer/image tower entirely.

Grounded directly in STPath's real source (Huang et al. 2025,
github.com/Graph-and-Geometric-Learning/STPath), not guessed:
  - stpath/model/model.py, class EncodeInputs:
        self.gene_embed = nn.Linear(config.n_genes, config.d_model, bias=False)
    called as gene_embed(ge_tokens) where ge_tokens is REAL observed
    expression scattered into STPath's fixed gene-vocabulary indices (see
    stpath/tokenization/ge_tokenizer.py GeneExpTokenizer.convert_gene_exp_to_one_hot_tensor)
    -- i.e. gene_embed.weight has shape [d_model, n_tokens] and column
    gene2id[ensembl_id] is that specific gene's own pretrained weight
    vector. Top-level checkpoint key: "input_encoder.gene_embed.weight"
    (stpath/model/model.py class STFM: self.input_encoder = EncodeInputs(config)).
  - stpath/tokenization/ge_tokenizer.py, class GeneExpTokenizer.__init__:
    gene2id is built by sorting the DEDUPLICATED set of Ensembl IDs from
    symbol2ensembl.json and assigning integer ids 0..N-1, shifted by +2
    (reserved pad/mask token slots at 0 and 1). Fully deterministic given
    that JSON file alone -- reimplemented here (_stpath_gene2id) rather
    than importing the real stpath package, which pulls in scanpy/
    torch_geometric at import time even though this lookup needs neither.

Genes in OUR panel with no match in STPath's ~138k-entry vocabulary get a
ZERO frozen column (contribute nothing through this pathway for that gene,
rather than a fabricated pretrained value) -- reported explicitly, never
silently guessed.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn


def _stpath_gene2id(symbol2ensembl_path: str) -> dict[str, int]:
    """Reconstruct STPath's own gene_embed column index for every Ensembl
    ID in its vocabulary file -- mirrors GeneExpTokenizer.__init__ exactly
    (see this module's docstring)."""
    symbol2gene = json.load(open(symbol2ensembl_path, "r"))
    gene_ids = sorted(set(symbol2gene.values()))
    return {gene_id: i + 2 for i, gene_id in enumerate(gene_ids)}


def extract_stpath_gene_embedding_table(
    gene_names: list[str],
    symbol2ensembl_path: str,
    model_weight_path: str,
    d_model: int = 512,
    state_dict_key: str = "input_encoder.gene_embed.weight",
) -> tuple[np.ndarray, dict]:
    """Gather the frozen, pretrained STPath columns for OUR gene panel, in
    OUR panel's order.

    Returns (table, report): table is [d_model, len(gene_names)] float32 --
    same shape/orientation as WeightedGeneExpressionEncoder's own
    nn.Linear(n_genes, hidden_dim, bias=False).weight would be if
    hidden_dim==d_model, so it composes with this project's existing
    "real expression @ frozen/trainable per-gene matrix" gene-encoder
    contract (see STPathFrozenGeneEncoder below). report records exactly
    which of our genes matched STPath's vocabulary and which didn't, so a
    caller can audit coverage rather than trust it silently.
    """
    symbol2gene = json.load(open(symbol2ensembl_path, "r"))
    gene2id = _stpath_gene2id(symbol2ensembl_path)

    state_dict = torch.load(model_weight_path, map_location="cpu")
    if state_dict_key not in state_dict:
        raise KeyError(
            f"{state_dict_key!r} not found in STPath checkpoint at {model_weight_path}. "
            f"Available top-level keys with 'gene_embed': "
            f"{[k for k in state_dict if 'gene_embed' in k]}"
        )
    full_weight = state_dict[state_dict_key]
    if full_weight.shape[0] != d_model:
        raise ValueError(
            f"STPath checkpoint's gene_embed weight has d_model={full_weight.shape[0]}, "
            f"expected {d_model} -- pass the real d_model explicitly if this checkpoint differs."
        )

    table = torch.zeros(d_model, len(gene_names), dtype=full_weight.dtype)
    found, missing = [], []
    for col, gene in enumerate(gene_names):
        ensembl_id = symbol2gene.get(gene)
        token_id = gene2id.get(ensembl_id) if ensembl_id is not None else None
        if token_id is None or token_id >= full_weight.shape[1]:
            missing.append(gene)
            continue
        table[:, col] = full_weight[:, token_id]
        found.append(gene)

    report = {
        "n_genes": len(gene_names),
        "n_found": len(found),
        "n_missing": len(missing),
        "missing_genes": missing,
    }
    return table.numpy().astype(np.float32), report


class STPathFrozenGeneEncoder(nn.Module):
    """Real observed expression -> STPath's own FROZEN pretrained
    gene_embed computation (real_expression @ frozen_table, exactly
    STPath's gene_embed(ge_tokens) with STPath's own trained weights,
    never updated by our training) -> a small TRAINABLE projection down to
    this model's hidden_dim -- same "frozen big representation + small
    trainable head" (RAE) pattern as GigapathPatchEncoder/NovaeGeneEncoder/
    DINOv2PatchEncoder. Real observed expression is the ONLY input; this
    never substitutes for or removes real GEX (see this module's own
    docstring and HierarchicalGeneTransportRegressor's docstring for the
    scoring-only role of every gene_encoder_type option -- the actual
    transport candidates/IDW anchor always use raw expression directly,
    regardless of gene encoder choice)."""

    def __init__(self, frozen_table: np.ndarray | torch.Tensor, hidden_dim: int):
        super().__init__()
        table = torch.as_tensor(frozen_table, dtype=torch.float32)
        if table.dim() != 2:
            raise ValueError(f"frozen_table must be [d_model, n_genes], got shape {tuple(table.shape)}")
        self.register_buffer("frozen_table", table)  # [d_model, n_genes], never trained
        self.proj = nn.Linear(table.shape[0], hidden_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        if expression.shape[-1] != self.frozen_table.shape[1]:
            raise ValueError(
                f"expression has {expression.shape[-1]} genes, frozen_table was built for "
                f"{self.frozen_table.shape[1]} -- gene panel mismatch"
            )
        pretrained = expression @ self.frozen_table.T  # [B, d_model], STPath's own frozen computation
        return self.proj(pretrained)
