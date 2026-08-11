"""Iterative spatial refinement that lets NEIGHBOURING EXPRESSION steer attention.

Motivation (measured, Aug 2026). Our conditional WAE's spatial attention runs
over image features only: expression never participates in the spatial
reasoning, and the latent's contribution was measured to be orthogonal noise
(``corr(residual, conditional-mean error) ~= 0.002-0.025`` across four trained
arms). Meanwhile STFlow (ICML 2025) reports that the whole competitive gap in
H&E->ST prediction comes from spatial context, not from generative modelling:
its own ablation puts flow matching at +0.010 of 0.415 (2.4% relative), while
"STFlow w/o FM" at 0.405 already beats every published baseline (TRIPLEX
0.395, BLEEP 0.368, spot-only foundation-model heads 0.344).

The mechanism behind that gap is STFlow's Equation 7, whose attention logit
between spots i and j includes the GENE EXPRESSION DIFFERENCE ``(Y_i - Y_j)``
alongside image features and relative geometry, with representation and
expression both updated at every layer (Eqs. 8-9). This module implements the
same idea for this project, with one necessary deviation: STFlow's benchmark
panel is small, whereas this project predicts 17,189 genes, so a raw
``[N, K, n_genes]`` difference tensor is not affordable (2,000 spots x 10
neighbours x 17,189 genes x 4 bytes ~= 1.4 GB per layer). Expression is
therefore projected through the same ``WeightedGeneExpressionEncoder`` the
image conditioner already uses for observed GEX, and the DIFFERENCE IS TAKEN
IN THAT EMBEDDING, which is affordable and keeps one gene-encoding convention
across the codebase.

Applied iteratively, this is refinement over the model's own prediction: no
observed target expression is ever read, so the task contract (query GEX is
never visible) is preserved exactly. ``refine_expression`` is a pure function
of (prediction, image context, coordinates) and is used identically in
training and inference.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.boundary_graph import build_knn_adjacency
from gen3_multiscale.models.gene_encoder import WeightedGeneExpressionEncoder


def padded_neighbor_graph(
    coords: np.ndarray | torch.Tensor, k_neighbors: int, *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``[N, K]`` neighbour indices + validity mask from the project's
    own symmetric k-NN union graph.

    ``build_knn_adjacency`` returns a RAGGED adjacency (the undirected union
    of directed selections, so a node may exceed ``k_neighbors``), which
    cannot be gathered in one batched op. This pads to the observed maximum
    degree and returns the mask that keeps padding out of the softmax. The
    graph itself is deliberately the identical one used by the spatial
    gradient loss and by boundary extraction -- never a second, ad hoc notion
    of adjacency.
    """
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive")
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    if coords_np.ndim != 2 or coords_np.shape[1] < 2:
        raise ValueError(f"coords must be [N, >=2], got {coords_np.shape}")
    n = coords_np.shape[0]
    adjacency = build_knn_adjacency(coords_np[:, :2], k_neighbors=k_neighbors)
    width = max(1, max((len(row) for row in adjacency), default=1))
    indices = np.zeros((n, width), dtype=np.int64)
    mask = np.zeros((n, width), dtype=bool)
    for row, neighbors in enumerate(adjacency):
        if len(neighbors) == 0:
            continue
        indices[row, : len(neighbors)] = np.asarray(neighbors, dtype=np.int64)
        mask[row, : len(neighbors)] = True
    return (
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.as_tensor(mask, dtype=torch.bool, device=device),
    )


class SpatialExpressionRefiner(nn.Module):
    """One refinement pass: attention over neighbours whose weights depend on
    image context, relative geometry AND the neighbour's expression.

    The output is an ADDITIVE update to the incoming expression estimate, so
    a freshly-initialised refiner starts near the identity and the first
    training steps cannot destroy an already-reasonable base prediction.
    """

    def __init__(self, n_genes: int, context_dim: int, *,
                 gex_feature_dim: int = 256, hidden_dim: int = 256,
                 geometry_dim: int = 32, k_neighbors: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        if n_genes < 1 or context_dim < 1:
            raise ValueError("n_genes and context_dim must be positive")
        for name, value in (
            ("gex_feature_dim", gex_feature_dim), ("hidden_dim", hidden_dim),
            ("geometry_dim", geometry_dim), ("k_neighbors", k_neighbors),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        self.n_genes = int(n_genes)
        self.k_neighbors = int(k_neighbors)
        # Bounded memo of coordinate-derived neighbour graphs (see
        # cached_neighbor_graph). Not a registered buffer: it is a pure
        # recomputable cache and must never enter a checkpoint.
        self._neighbor_graph_cache: dict = {}
        self._neighbor_graph_cache_size = 512
        self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, gex_feature_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.expression_proj = nn.Linear(gex_feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.geometry_mlp = nn.Sequential(
            nn.Linear(3, geometry_dim), nn.GELU(), nn.Linear(geometry_dim, geometry_dim),
        )
        # Equation-7 analogue: the logit sees the pair's hidden states, their
        # relative geometry, and their expression-embedding difference.
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + geometry_dim + gex_feature_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_genes),
        )
        # Zero-initialised final layer: the module is exactly the identity at
        # step 0, so enabling refinement never degrades a loaded checkpoint
        # before any refinement gradient has been taken.
        nn.init.zeros_(self.update_head[-1].weight)
        nn.init.zeros_(self.update_head[-1].bias)

    def cached_neighbor_graph(self, coords: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Memoised ``padded_neighbor_graph`` keyed by the coordinates' bytes.

        The graph is a pure function of the coordinates, and this project's
        mask schedule is deterministic and finite, so the same coordinate sets
        recur every epoch. Hashing the coordinate bytes is O(n) against the
        k-NN build's O(n log n) plus a Python-level padding loop, so the cache
        pays for itself immediately and removes the per-forward-pass cost.
        The cache is bounded so a long run over many distinct masks cannot
        grow it without limit.
        """
        coords_np = np.ascontiguousarray(
            coords.detach().cpu().numpy()[:, :2].astype(np.float64)
        )
        key = (coords_np.shape[0], hashlib.blake2b(coords_np.tobytes(), digest_size=16).digest())
        cached = self._neighbor_graph_cache.get(key)
        if cached is not None:
            indices, mask = cached
            return indices.to(coords.device), mask.to(coords.device)
        indices, mask = padded_neighbor_graph(coords_np, self.k_neighbors)
        if len(self._neighbor_graph_cache) >= self._neighbor_graph_cache_size:
            self._neighbor_graph_cache.pop(next(iter(self._neighbor_graph_cache)))
        self._neighbor_graph_cache[key] = (indices, mask)
        return indices.to(coords.device), mask.to(coords.device)

    def forward(self, expression: torch.Tensor, context: torch.Tensor,
                coords: torch.Tensor, neighbor_indices: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        if expression.ndim != 2 or expression.shape[1] != self.n_genes:
            raise ValueError(
                f"expression must be [N, {self.n_genes}], got {tuple(expression.shape)}"
            )
        if context.shape[0] != expression.shape[0]:
            raise ValueError("context and expression must have the same number of rows")
        if neighbor_indices.shape != neighbor_mask.shape:
            raise ValueError("neighbor_indices and neighbor_mask must have the same shape")
        if neighbor_indices.shape[0] != expression.shape[0]:
            raise ValueError("neighbour graph must have one row per spot")

        gene_features = self.gene_encoder(expression)
        hidden = self.norm(self.context_proj(context) + self.expression_proj(gene_features))

        gathered_hidden = hidden[neighbor_indices]
        gathered_genes = gene_features[neighbor_indices]
        delta = coords[neighbor_indices][..., :2] - coords[:, None, :2]
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
        geometry = self.geometry_mlp(torch.cat([delta, distance], dim=-1))

        pair = torch.cat(
            [
                hidden[:, None, :].expand_as(gathered_hidden),
                gathered_hidden,
                geometry,
                gene_features[:, None, :] - gathered_genes,
            ],
            dim=-1,
        )
        logits = self.attention_mlp(pair).squeeze(-1)
        logits = logits.masked_fill(~neighbor_mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        aggregated = torch.einsum("nk,nkh->nh", weights, self.value_proj(gathered_hidden))
        return expression + self.update_head(hidden + aggregated)


def refine_expression(refiner: SpatialExpressionRefiner, expression: torch.Tensor,
                      context: torch.Tensor, coords: torch.Tensor, *,
                      n_steps: int) -> torch.Tensor:
    """Apply ``refiner`` ``n_steps`` times over one shared neighbour graph.

    STFlow finds performance rising from one-step prediction through about
    five refinement steps and plateauing or declining beyond that, so a small
    step count is the intended operating point.

    The graph depends ONLY on coordinates, so it is memoised across calls, not
    merely across the steps of one call. Rebuilding it per forward pass was
    measured at 3.1 s/step against a 0.41 s/step baseline -- a 7.6x slowdown
    that would have left these arms at ~7% of the baseline's training within
    the same wall-clock budget, turning a cost bug into a false negative about
    the mechanism. The mask schedule is deterministic and finite, so the same
    coordinate sets recur every epoch and the cache hits almost always after
    the first pass over the data.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_steps == 0:
        return expression
    neighbor_indices, neighbor_mask = refiner.cached_neighbor_graph(coords)
    for _ in range(n_steps):
        expression = refiner(
            expression, context, coords, neighbor_indices, neighbor_mask,
        )
    return expression
