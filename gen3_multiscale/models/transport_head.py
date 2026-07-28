"""Gene-value-preserving transport head -- Phase 4 of the multiscale
spatial-field handoff ("Reuse the audited gene-value-preserving transport
implementation").

The math and initialization below are extracted from
src/models/registry.py::HierarchicalGeneTransportRegressor's `sample()`
method (commit e0c91ce) -- NOT copied verbatim as a whole class, because
that class hard-constructs `HierarchicalMissingTissueEncoder` (the OLDER
fusion conditioner) inside its own __init__, and the handoff is explicit:
"Do not copy the older fusion design blindly." What IS extracted
verbatim is the actually-audited part: the transport SCORER (geometry ->
per-head logits -> softmax spot weights), the per-gene head GATE
(optionally query-dependent, low-rank), and the zero-initialized
low-rank RESIDUAL -- exactly the same tensor shapes, the same
initialization scales (`* 0.02` for score parameters, `1/sqrt(rank)` for
the residual gene embedding, exact-zero for the residual query
projection), the same softmax/entropy formulas. This module is
deliberately ENCODER-AGNOSTIC: it takes query/candidate hidden states as
plain tensors, never owns or constructs an encoder itself, so gen3's own
NEW spatial-field backbone (Phase 5/6, ring-boundary-aware, not yet
built) can feed it -- unlike the old class, which only ever received
hidden states from `HierarchicalMissingTissueEncoder.forward_with_neighbors()`.

Deliberate NEW divergence from the old class, required by the handoff's
per-architecture fairness matrix: the old class ALWAYS blends against an
IDW anchor (`transport = anchor + blend * (candidate - anchor)`) -- every
config paid that architectural cost whether or not anchoring was the
axis being tested. The handoff instead requires "Architectures 1, 3, and
4... contain no computational path from any interpolation output to
their predictions or losses" -- a stricter requirement than "blend
weight near zero," it means the anchor computation must not exist in the
graph at all for those three. `use_anchor_blend` (default False) governs
this: when False, `blend_logit` is never constructed and forward()
rejects an `anchor_expression` argument outright (structurally
impossible to leak an anchor in); when True (Architecture 2 only),
`anchor_expression` is REQUIRED and must be computed OUTSIDE this
module by a harmonic solver with no learned parameters, per the
handoff's "The harmonic solver must be outside the trainable neural
input path."
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GeneValueTransportHead(nn.Module):
    """candidate(q, g) = sum_h gene_gate(g, h) * sum_j spot_weight(q, h, j) * candidate_expression(q, j, g)

    spot_weight(q, h, ·) is a convex (softmax) distribution over this
    query's own C candidates for head h -- never over the whole observed
    set, and never shared across queries (each query scores its own
    candidate pool). gene_gate(g, h) is a per-gene distribution over
    heads, shared across queries unless use_query_gate/use_query_gene_gate
    make it (partially) query-dependent. Both are always non-negative and
    sum to 1 along their respective softmax dimension -- verified by test.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        score_hidden_dim: int = 128,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        gene_gate_mode: str = "per_gene",
        use_query_gate: bool = True,
        use_query_gene_gate: bool = False,
        query_gene_gate_rank: int = 16,
        use_residual: bool = False,
        residual_rank: int = 32,
        target_gene_scale: torch.Tensor | None = None,
        target_scale_floor: float = 0.05,
        use_anchor_blend: bool = False,
        blend_logit_init: float = -2.9444389791664403,  # logit(0.05), matches the audited default
    ):
        super().__init__()
        if gene_gate_mode not in {"per_gene", "shared"}:
            raise ValueError("gene_gate_mode must be 'per_gene' or 'shared'")
        if transport_heads < 1:
            raise ValueError("transport_heads must be positive")
        if transport_temperature <= 0 or target_scale_floor <= 0:
            raise ValueError("transport_temperature and target_scale_floor must be positive")
        if use_residual and residual_rank < 1:
            raise ValueError("residual_rank must be positive when use_residual=True")

        self.n_genes = int(n_genes)
        self.transport_heads = int(transport_heads)
        self.transport_temperature = float(transport_temperature)
        self.gene_gate_mode = str(gene_gate_mode)
        self.use_query_gate = bool(use_query_gate)
        self.use_query_gene_gate = bool(use_query_gene_gate)
        self.use_residual = bool(use_residual)
        self.use_anchor_blend = bool(use_anchor_blend)

        scale = torch.ones(n_genes) if target_gene_scale is None else torch.as_tensor(
            target_gene_scale, dtype=torch.float32
        )
        if scale.shape != (n_genes,):
            raise ValueError(f"target_gene_scale must have shape ({n_genes},), got {tuple(scale.shape)}")
        if not torch.isfinite(scale).all():
            raise ValueError("target_gene_scale contains non-finite values")
        self.register_buffer("target_gene_scale", scale.clamp_min(float(target_scale_floor)))

        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, score_hidden_dim), nn.GELU(),
            nn.Linear(score_hidden_dim, score_hidden_dim),
        )
        self.head_embedding = nn.Parameter(torch.randn(self.transport_heads, score_hidden_dim) * 0.02)
        self.head_score_vector = nn.Parameter(torch.randn(self.transport_heads, score_hidden_dim) * 0.02)
        self.score_norm = nn.LayerNorm(score_hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, score_hidden_dim)
        self.query_score_projection = nn.Linear(hidden_dim, score_hidden_dim)

        gate_genes = 1 if self.gene_gate_mode == "shared" else n_genes
        self.gene_head_logits = nn.Parameter(torch.zeros(gate_genes, self.transport_heads))
        self.query_gate_projection = (
            nn.Linear(hidden_dim, self.transport_heads) if self.use_query_gate else None
        )
        self.query_gene_gate_query = None
        self.query_gene_gate_table = None
        if self.use_query_gene_gate:
            self.query_gene_gate_query = nn.Linear(hidden_dim, query_gene_gate_rank)
            self.query_gene_gate_table = nn.Parameter(
                torch.randn(n_genes, query_gene_gate_rank, self.transport_heads) * 0.02
            )

        self.blend_logit = None
        if self.use_anchor_blend:
            self.blend_logit = nn.Parameter(torch.full((n_genes,), float(blend_logit_init)))

        self.residual_rank = int(residual_rank) if use_residual else 0
        self.residual_query_projection = None
        self.residual_gene_embedding = None
        if self.use_residual:
            self.residual_query_projection = nn.Linear(hidden_dim, self.residual_rank)
            # Zero-initialized: the residual contributes exactly nothing
            # until training moves this weight away from zero.
            nn.init.zeros_(self.residual_query_projection.weight)
            nn.init.zeros_(self.residual_query_projection.bias)
            self.residual_gene_embedding = nn.Parameter(
                torch.randn(n_genes, self.residual_rank) * (1.0 / math.sqrt(self.residual_rank))
            )

    def forward(
        self,
        query_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        candidate_relative_geometry: torch.Tensor,
        candidate_expression: torch.Tensor,
        anchor_expression: torch.Tensor | None = None,
    ) -> dict:
        n_query, n_candidates, hidden_dim = candidate_hidden.shape
        if query_hidden.shape[0] != n_query:
            raise ValueError(f"query_hidden has {query_hidden.shape[0]} rows, expected {n_query}")
        if candidate_relative_geometry.shape != (n_query, n_candidates, 3):
            raise ValueError(
                f"candidate_relative_geometry must be [{n_query}, {n_candidates}, 3], got "
                f"{tuple(candidate_relative_geometry.shape)}"
            )
        if candidate_expression.shape != (n_query, n_candidates, self.n_genes):
            raise ValueError(
                f"candidate_expression must be [{n_query}, {n_candidates}, {self.n_genes}], got "
                f"{tuple(candidate_expression.shape)}"
            )
        if self.use_anchor_blend and anchor_expression is None:
            raise ValueError(
                "use_anchor_blend=True requires anchor_expression -- compute it OUTSIDE this "
                "module with a non-learned harmonic solver, per the handoff's requirement that "
                "the anchor stay outside the trainable neural input path"
            )
        if not self.use_anchor_blend and anchor_expression is not None:
            raise ValueError(
                "anchor_expression was provided but use_anchor_blend=False -- this head was "
                "constructed as anchor-free (Architecture 1/3/4 contract: no computational path "
                "from any interpolation output to the prediction). Construct with "
                "use_anchor_blend=True (Architecture 2 only) to use an anchor."
            )

        hidden = self.geometry_encoder(candidate_relative_geometry)[:, :, None, :]  # [Nq, C, 1, S]
        hidden = hidden + self.head_embedding[None, None, :, :]  # [1, 1, heads, S]
        hidden = hidden + self.neighbor_projection(candidate_hidden)[:, :, None, :]
        hidden = hidden + self.query_score_projection(query_hidden)[:, None, None, :]
        hidden = torch.nn.functional.gelu(self.score_norm(hidden))
        logits = torch.einsum("qchd,hd->qch", hidden, self.head_score_vector)  # d==S here, reused as contraction dim
        head_weights = torch.softmax(
            logits.transpose(1, 2) / self.transport_temperature, dim=-1
        )  # [Nq, heads, C], convex per (query, head)
        head_expression = torch.einsum(
            "qhc,qcg->qhg", head_weights, candidate_expression
        )  # [Nq, heads, G]

        base_gate = (
            self.gene_head_logits.expand(self.n_genes, -1)
            if self.gene_gate_mode == "shared" else self.gene_head_logits
        )
        gate_logits = base_gate[None, :, :].expand(n_query, -1, -1)  # [Nq, G, heads]
        if self.query_gate_projection is not None:
            gate_logits = gate_logits + self.query_gate_projection(query_hidden)[:, None, :]
        if self.query_gene_gate_table is not None:
            query_low = self.query_gene_gate_query(query_hidden)  # [Nq, rank]
            gate_logits = gate_logits + torch.einsum(
                "qr,grh->qgh", query_low, self.query_gene_gate_table
            )
        gene_gates = torch.softmax(gate_logits, dim=-1)  # [Nq, G, heads], convex per (query, gene)
        candidate = torch.einsum("qhg,qgh->qg", head_expression, gene_gates)  # the transport candidate

        blend = None
        if self.use_anchor_blend:
            blend = torch.sigmoid(self.blend_logit)[None, :]  # [1, G]
            transport = anchor_expression + blend * (candidate - anchor_expression)
        else:
            transport = candidate

        residual = torch.zeros_like(transport)
        if self.use_residual:
            query_factor = self.residual_query_projection(query_hidden)  # [Nq, rank]
            residual = torch.einsum(
                "qr,gr->qg", query_factor, self.residual_gene_embedding
            ) * self.target_gene_scale[None, :]

        expression = transport + residual
        head_entropy = -(head_weights * head_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
        gate_entropy = -(gene_gates * gene_gates.clamp_min(1e-12).log()).sum(dim=-1).mean()

        return {
            "expression": expression,
            "candidate_expression": candidate,
            "anchor_expression": anchor_expression,
            "transport_expression": transport,
            "residual_expression": residual,
            "blend": blend,
            "transport_head_entropy": head_entropy,
            "gene_gate_entropy": gate_entropy,
            "head_weights": head_weights,
            "gene_gates": gene_gates,
        }
