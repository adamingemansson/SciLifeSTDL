"""Architecture 1/2/3/4 wrappers -- Phase 6 of the multiscale spatial-
field handoff. Architectures 1/2/3 share ONE base class
(_SharedFieldArchitecture) with explicit feature flags, per Phase 6's
own instruction: "Avoid four copy-pasted models. Use shared modules with
explicit feature flags so accidental differences are visible in resolved
configs, while preserving clear architecture names and checkpoints."
Architecture 4 wraps a full Architecture3 instance as its frozen-at-the-
flow-loss conditioner (see Architecture4's own docstring below) rather
than being a fourth _SharedFieldArchitecture flag combination, since its
extra apparatus (velocity network, gene-basis projection, ODE sampling)
is qualitatively different from a token/attention feature flag.

KNOWN SIMPLIFICATION (documented here and in CONTRACT.md, not silently
assumed away): the handoff specifies the transport candidate pool as
"the deduplicated union of local and complete-boundary spots." This
module instead CONCATENATES local (per-query, local_k candidates) and
boundary (shared, n_boundary candidates) without deduplicating overlap
between them (a query's true-nearest local neighbor is very often ALSO a
Ring-1 boundary spot). A concatenated, non-deduplicated pool is strictly
easier to batch (uniform per-query candidate count = local_k +
n_boundary, no ragged/masked tensor needed) and is NOT a leakage or
correctness bug -- transport weights still sum to 1 over the (possibly
duplicated) pool -- but a spot appearing twice can receive correlated
extra influence in the transport gate's softmax competition versus the
doc's literal "deduplicated" wording. Proper deduplication would need
per-query masked attention (a real, larger change); flagged as a
follow-up, not implemented in this pass.

Also documented: modality-availability flags are hardcoded to
"available" (no real per-spot H&E-missing signal is wired in yet -- the
data-builder that would produce that flag from SpatialFieldInputs is not
built, see CONTRACT.md's Phase 3/6 notes), so SpotTokenProjection's
modality-flag branch is currently a fixed input, not yet doing real work.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.example import SpatialFieldInputs
from gen3_multiscale.models.backbone import SpatialFieldBackbone
from gen3_multiscale.models.flow import VelocityNetwork, flow_matching_loss, sample_residual_coefficients
from gen3_multiscale.models.gene_basis import GeneResidualBasis, verify_gene_residual_basis
from gen3_multiscale.models.geometry_utils import compute_hole_geometry, compute_relative_geometry, scatter_boundary_ring
from gen3_multiscale.models.global_context import InducedGlobalGEXPool
from gen3_multiscale.models.harmonic import harmonic_interpolation
from gen3_multiscale.models.tokens import QueryTokenProjection, SpotTokenProjection
from gen3_multiscale.models.transport_head import GeneValueTransportHead


class _SharedFieldArchitecture(nn.Module):
    """Common assembly for Architectures 1/2/3: token projections, the
    anchor-free SpatialFieldBackbone conditioner (Architecture 2 adds its
    anchor only inside the transport head, never in the conditioner
    itself), and the transport head. Subclasses set use_anchor_blend,
    use_regional_he, use_global_gex, use_global_slide via __init__ and
    otherwise share every module and code path -- a resolved config diff
    between architectures is exactly the set of flags passed here, not a
    hidden divergence in duplicated code."""

    def __init__(
        self,
        n_genes: int,
        gex_feature_dim: int,
        image_feature_dim: int = 1536,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        max_boundary_size: int | None = None,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        gene_gate_mode: str = "per_gene",
        use_query_gate: bool = True,
        use_residual: bool = False,
        residual_rank: int = 32,
        target_gene_scale: torch.Tensor | None = None,
        use_anchor_blend: bool = False,
        use_regional_he: bool = False,
        use_global_gex: bool = False,
        use_global_slide: bool = False,
        global_slide_dim: int = 768,
        n_gex_inducing: int = 16,
        harmonic_k_neighbors: int = 6,
    ):
        super().__init__()
        self.use_anchor_blend = use_anchor_blend
        self.use_regional_he = use_regional_he
        self.use_global_gex = use_global_gex
        self.use_global_slide = use_global_slide
        self.harmonic_k_neighbors = harmonic_k_neighbors

        self.spot_token = SpotTokenProjection(
            hidden_dim=hidden_dim, image_feature_dim=image_feature_dim, gex_feature_dim=gex_feature_dim,
        )
        self.query_token = QueryTokenProjection(hidden_dim=hidden_dim, use_hole_geometry=True)
        self.backbone = SpatialFieldBackbone(
            n_blocks=n_blocks, hidden_dim=hidden_dim, n_heads=n_heads,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size,
            use_regional_he=use_regional_he, use_global_gex=use_global_gex,
            use_global_slide=use_global_slide, global_slide_dim=global_slide_dim,
        )
        self.gex_pool = InducedGlobalGEXPool(hidden_dim=hidden_dim, n_inducing=n_gex_inducing, n_heads=n_heads) \
            if use_global_gex else None
        self.transport_head = GeneValueTransportHead(
            n_genes=n_genes, hidden_dim=hidden_dim, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale, use_anchor_blend=use_anchor_blend,
        )

    def _observed_tokens(self, inputs: SpatialFieldInputs) -> torch.Tensor:
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed, torch.as_tensor(inputs.boundary_idx), torch.as_tensor(inputs.boundary_ring),
        )
        modality_flags = torch.ones(n_observed, 1)  # see module docstring: not yet real per-spot availability
        return self.spot_token(
            image_features=torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32),
            gex_features=torch.as_tensor(inputs.observed_gex_conditioning, dtype=torch.float32),
            coords=torch.as_tensor(inputs.observed_coords, dtype=torch.float32),
            boundary_ring=full_ring,
            modality_flags=modality_flags,
        )

    def _candidate_pool(self, inputs: SpatialFieldInputs, observed_tokens: torch.Tensor, query_coords: torch.Tensor):
        observed_coords = torch.as_tensor(inputs.observed_coords, dtype=torch.float32)
        observed_expr = torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32)
        local_idx = torch.as_tensor(inputs.query_local_neighbor_idx, dtype=torch.long)
        boundary_idx = torch.as_tensor(inputs.boundary_idx, dtype=torch.long)

        local_hidden = observed_tokens[local_idx]  # [Nq, local_k, H]
        local_coords = observed_coords[local_idx]  # [Nq, local_k, 2]
        local_geometry = compute_relative_geometry(query_coords, local_coords)
        local_expr = observed_expr[local_idx]  # [Nq, local_k, G]

        boundary_hidden = observed_tokens[boundary_idx]  # [n_boundary, H]
        boundary_coords = observed_coords[boundary_idx]  # [n_boundary, 2]
        boundary_geometry = compute_relative_geometry(query_coords, boundary_coords)  # [Nq, n_boundary, 3]
        n_query = query_coords.shape[0]
        boundary_expr = observed_expr[boundary_idx][None].expand(n_query, -1, -1)  # [Nq, n_boundary, G]
        boundary_hidden_broadcast = boundary_hidden[None].expand(n_query, -1, -1)

        candidate_hidden = torch.cat([local_hidden, boundary_hidden_broadcast], dim=1)
        candidate_geometry = torch.cat([local_geometry, boundary_geometry], dim=1)
        candidate_expression = torch.cat([local_expr, boundary_expr], dim=1)
        return {
            "candidate_hidden": candidate_hidden,
            "candidate_geometry": candidate_geometry,
            "candidate_expression": candidate_expression,
            "local_hidden": local_hidden,
            "local_geometry": local_geometry,
            "boundary_hidden": boundary_hidden,
            "boundary_geometry": boundary_geometry,
        }

    def _harmonic_anchor(self, inputs: SpatialFieldInputs, query_coords: torch.Tensor) -> torch.Tensor:
        anchor = harmonic_interpolation(
            observed_coords=np.asarray(inputs.observed_coords, dtype=np.float64),
            observed_expression=np.asarray(inputs.observed_full_gene_expression, dtype=np.float64),
            query_coords=np.asarray(inputs.query_coords, dtype=np.float64),
            k_neighbors=self.harmonic_k_neighbors,
        )
        return torch.from_numpy(anchor).to(dtype=query_coords.dtype)

    def forward(self, inputs: SpatialFieldInputs) -> dict:
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
        depth = torch.as_tensor(inputs.query_depth_to_boundary, dtype=torch.long)
        hole_geometry = compute_hole_geometry(query_coords)
        query_hidden = self.query_token(query_coords, depth, hole_geometry)

        observed_tokens = self._observed_tokens(inputs)
        pool = self._candidate_pool(inputs, observed_tokens, query_coords)

        block_kwargs = dict(
            local_hidden=pool["local_hidden"], local_geometry=pool["local_geometry"],
            boundary_hidden=pool["boundary_hidden"], boundary_geometry=pool["boundary_geometry"],
        )
        if self.use_global_gex:
            gex_out = self.gex_pool(observed_tokens, torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32))
            n_inducing = gex_out["hidden"].shape[0]
            # inducing tokens have no real position -- a sentinel relative
            # geometry (zero direction, a fixed "far" distance) marks them
            # as categorically distinct from real spatial candidates,
            # mirroring HierarchicalGeneTransportRegressor's own
            # use_global_candidate sentinel (src/models/registry.py).
            sentinel = torch.zeros(query_coords.shape[0], n_inducing, 3)
            sentinel[..., 2] = 3.0
            block_kwargs["gex_inducing_hidden"] = gex_out["hidden"]
            block_kwargs["gex_inducing_geometry"] = sentinel
        if self.use_regional_he:
            raise NotImplementedError(
                "use_regional_he=True requires regional H&E tokens from Phase 3's WSI path, "
                "not yet wired into this forward() -- see CONTRACT.md"
            )
        if self.use_global_slide:
            raise NotImplementedError(
                "use_global_slide=True requires a real LongNet global token from Phase 3's "
                "FrozenGigaPathSlideEncoder, not yet wired into this forward() -- see CONTRACT.md"
            )

        final_query_hidden = self.backbone(query_hidden, query_coords, **block_kwargs)

        anchor_expression = self._harmonic_anchor(inputs, query_coords) if self.use_anchor_blend else None
        out = self.transport_head(
            final_query_hidden, pool["candidate_hidden"], pool["candidate_geometry"],
            pool["candidate_expression"], anchor_expression=anchor_expression,
        )
        out["query_hidden"] = final_query_hidden
        return out


class Architecture1(_SharedFieldArchitecture):
    """Anchor-Free Full Boundary Field. No interpolation anchor, no
    global slide/GEX branches."""

    def __init__(self, **kwargs):
        kwargs.setdefault("use_anchor_blend", False)
        kwargs.setdefault("use_regional_he", False)
        kwargs.setdefault("use_global_gex", False)
        kwargs.setdefault("use_global_slide", False)
        super().__init__(**kwargs)


class Architecture2(_SharedFieldArchitecture):
    """Harmonic-Residual Full Boundary Field. Exactly Architecture 1's
    conditioner; the ONLY difference is use_anchor_blend=True, adding the
    non-learned harmonic anchor inside the transport head."""

    def __init__(self, **kwargs):
        kwargs.setdefault("use_anchor_blend", True)
        kwargs.setdefault("use_regional_he", False)
        kwargs.setdefault("use_global_gex", False)
        kwargs.setdefault("use_global_slide", False)
        super().__init__(**kwargs)


class Architecture3(_SharedFieldArchitecture):
    """Hierarchical Slide-Boundary Field. Exactly Architecture 1 plus
    global observed-GEX inducing tokens and (once Phase 3's WSI path is
    wired in, not yet done here) global/regional H&E -- still anchor-free."""

    def __init__(self, **kwargs):
        kwargs.setdefault("use_anchor_blend", False)
        kwargs.setdefault("use_regional_he", False)  # NotImplementedError until wired -- see CONTRACT.md
        kwargs.setdefault("use_global_gex", True)
        kwargs.setdefault("use_global_slide", False)  # NotImplementedError until wired -- see CONTRACT.md
        super().__init__(**kwargs)


class Architecture4(nn.Module):
    """Hierarchical Residual Flow Field.

    "Use exactly the Architecture 3 conditioner and deterministic
    transport mean. Do not change its width, number of blocks, boundary
    selection, slide inputs, gene encoder, or transport head. Architecture
    4 remains anchor-free. Its deterministic mean is Architecture 3's
    learned transport prediction, not harmonic or IDW." -- satisfied by
    literally constructing a full Architecture3 instance as self.conditioner
    and calling it unmodified, rather than re-deriving an equivalent
    conditioner from scratch.

    "Initially stop gradients from the flow loss into the deterministic
    conditioner" -- self.conditioner's forward pass is always run WITHOUT
    torch.no_grad() (so its own deterministic-loss gradients, computed
    separately by a caller, are unaffected), but the query_hidden and
    deterministic mean handed to the flow apparatus are .detach()'d
    before use, every time, unconditionally -- there is no flag to turn
    this off in this first implementation, matching the handoff's
    "initially" framing (a later experiment could relax it, not this one).

    gene_basis must be a GeneResidualBasis already fit on TRAINING-split
    residuals (gene_basis.py, fit offline, outside this class -- this
    class only ever calls verify_gene_residual_basis, never fits one
    itself, so it can never accidentally fit on validation/test data).
    """

    def __init__(
        self,
        n_genes: int,
        gex_feature_dim: int,
        gene_basis: GeneResidualBasis,
        gene_names: list[str],
        image_feature_dim: int = 1536,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_blocks: int = 4,
        n_flow_blocks: int = 2,
        dense_threshold: int = 256,
        sparse_k: int = 10,
        chunk_size: int = 1024,
        max_boundary_size: int | None = None,
        transport_heads: int = 8,
        transport_temperature: float = 1.0,
        gene_gate_mode: str = "per_gene",
        use_query_gate: bool = True,
        use_residual: bool = False,
        residual_rank: int = 32,
        target_gene_scale: torch.Tensor | None = None,
        n_gex_inducing: int = 16,
        harmonic_k_neighbors: int = 6,
        n_flow_samples: int = 8,
        n_ode_steps: int = 20,
    ):
        super().__init__()
        verify_gene_residual_basis(gene_basis, gene_names)
        self.gene_basis = gene_basis
        self.n_flow_samples = n_flow_samples
        self.n_ode_steps = n_ode_steps

        self.conditioner = Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale, n_gex_inducing=n_gex_inducing,
            harmonic_k_neighbors=harmonic_k_neighbors,
        )
        self.velocity_network = VelocityNetwork(
            residual_rank=gene_basis.rank, hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_flow_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
        )

    def forward(self, inputs: SpatialFieldInputs) -> dict:
        """Runs ONLY the deterministic conditioner -- the same contract
        Architecture 3 itself has, so a caller computing the shared
        deterministic reconstruction/gradient losses (Phase 7) never
        needs to know it's holding an Architecture4 instance rather than
        an Architecture3 one."""
        return self.conditioner(inputs)

    def compute_flow_matching_loss(self, inputs: SpatialFieldInputs, target_expression: torch.Tensor) -> torch.Tensor:
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        deterministic_mean = conditioner_out["expression"].detach()
        target_residual = target_expression - deterministic_mean
        target_coefficients = self.gene_basis.to_coefficients(target_residual)
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
        return flow_matching_loss(self.velocity_network, target_coefficients, query_coords, query_hidden)

    @torch.no_grad()
    def sample_predictive_distribution(
        self, inputs: SpatialFieldInputs, n_samples: int | None = None, n_steps: int | None = None,
    ) -> dict:
        """Draws multiple low-rank residual-field samples and adds the
        corresponding full-gene residuals to the deterministic transport
        mean. "Primary PCC/RMSE comparison should use the predictive mean
        across samples" -- returned as predictive_mean (and also as
        "expression", so this dict is drop-in compatible with the
        conditioner-only forward()'s output for anything that only reads
        "expression"). Also reports predictive_std (uncertainty) and the
        raw per-sample field for diversity diagnostics (Phase 7)."""
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"]
        deterministic_mean = conditioner_out["expression"]
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32)
        n_query = query_coords.shape[0]

        coefficient_samples = sample_residual_coefficients(
            self.velocity_network, n_query, query_coords, query_hidden,
            n_samples=n_samples or self.n_flow_samples, n_steps=n_steps or self.n_ode_steps,
        )
        residual_samples = self.gene_basis.from_coefficients(coefficient_samples)  # [S, Nq, G]
        predictive_samples = deterministic_mean[None] + residual_samples
        predictive_mean = predictive_samples.mean(dim=0)
        predictive_std = predictive_samples.std(dim=0)
        return {
            "expression": predictive_mean,
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "predictive_samples": predictive_samples,
            "deterministic_mean": deterministic_mean,
        }
