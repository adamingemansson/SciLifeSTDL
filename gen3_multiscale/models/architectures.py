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

Modality-availability flags (16th Codex re-audit, Step 5 Part 2): now
read from SpatialFieldInputs.observed_image_available, the real
per-spot H&E-availability signal example_builder.py produces --
SpotTokenProjection's modality-flag branch does real work.

Regional H&E / global LongNet wiring (Step 5 Part 2): use_regional_he
pools the item's visible WSI tiles into a fixed grid_size x grid_size
of regional tokens (models.slide_encoder.pool_regional_tokens, stable
across every hole on a slide since it pools from full_slide_coord_bounds),
projects them into hidden_dim, and cross-attends the query field to
only the grid cells that actually had a visible tile (unavailable cells
are excluded from the attended set entirely, not zero-valued-but-still-
attended, matching this codebase's established index-selection pattern
for excluded content elsewhere -- boundary_idx/local_idx are index sets,
never masks). use_global_slide runs the injected, frozen
FrozenGigaPathSlideEncoder once per item over the item's visible tiles
in their REAL, unnormalized GigaPath LongNet target-MPP coordinates
(never the normalized regional-attention coordinates, nor the level-0/
HEST-aligned frame those are derived from -- see SpatialFieldInputs'
own docstring for why two separate coordinate fields exist) and FiLM-
modulates the query field with the resulting global vector. Both
branches read ONLY from SpatialFieldInputs' WSI fields and ONLY feed
the backbone's HIDDEN-state attention/FiLM path -- neither ever touches
`shared_expression_parts`/`local_expression` (the real, untouched GEX
value-candidate pool the transport head draws predictions from), so
regional/global H&E can influence what the model attends to but can
never become a literal predicted gene value itself.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.data.example import SpatialFieldInputs
from gen3_multiscale.models.backbone import SpatialFieldBackbone
from gen3_multiscale.models.flow import VelocityNetwork, flow_matching_loss, sample_residual_coefficients
from gen3_multiscale.models.gene_encoder import WeightedGeneExpressionEncoder
from gen3_multiscale.models.gene_basis import GeneResidualBasis, verify_gene_residual_basis
from gen3_multiscale.models.geometry_utils import compute_hole_geometry, compute_relative_geometry, scatter_boundary_ring
from gen3_multiscale.models.global_context import InducedGlobalGEXPool
from gen3_multiscale.models.harmonic import harmonic_interpolation
from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder, pool_regional_tokens, regional_grid_cell_centers
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
        regional_grid_size: int = 4,
        n_refinement_steps: int = 0,
        refinement_k_neighbors: int = 6,
        refinement_hidden_dim: int = 256,
        refinement_gex_feature_dim: int = 256,
        slide_encoder: FrozenGigaPathSlideEncoder | None = None,
        gigapath_checkpoint_sha256: str | None = None,
        model_architecture_version: str = "gen3-multiscale-shared-field-v1",
    ):
        super().__init__()
        self.use_anchor_blend = use_anchor_blend
        self.use_regional_he = use_regional_he
        self.use_global_gex = use_global_gex
        self.use_global_slide = use_global_slide
        self.harmonic_k_neighbors = harmonic_k_neighbors
        self.regional_grid_size = regional_grid_size
        self.model_architecture_version = model_architecture_version

        # 16th Codex re-audit (Step 5 Part 2): the complete LongNet cache
        # namespace must bind tile-cache content + visible-tile identity
        # (SpatialFieldInputs.slide_cache_namespace, already real) PLUS
        # the checkpoint's own SHA256 PLUS this model's architecture/
        # version -- properties of WHICH MODEL is running, not of the
        # data, so they are supplied here at construction time, never by
        # the data layer. Required (fail-closed, not a silent default)
        # whenever use_global_slide=True, mirroring gen3_multiscale's
        # established "make provenance required, not optional" discipline
        # (Step 4's checkpoint_provenance is the same pattern).
        if use_global_slide:
            if slide_encoder is None:
                raise ValueError("use_global_slide=True requires a real slide_encoder (FrozenGigaPathSlideEncoder)")
            if not gigapath_checkpoint_sha256 or not str(gigapath_checkpoint_sha256).strip():
                raise ValueError("use_global_slide=True requires a non-empty gigapath_checkpoint_sha256")
            # 17th Codex re-audit (Step 5 Part 2, "Important before Step
            # 6/7"), CONFIRMED real: gigapath_checkpoint_sha256 was only
            # ever a caller-supplied string, trusted blindly -- nothing
            # verified it actually matched the checkpoint slide_encoder
            # loaded, so a caller could pass any unrelated string and
            # silently poison the cache namespace with a false identity.
            # FrozenGigaPathSlideEncoder now exposes checkpoint_sha256,
            # computed from the real file bytes at construction; verify
            # the caller's claim against it, fail closed on mismatch or
            # on a slide_encoder that doesn't expose it at all (a
            # correctly-typed real encoder always does).
            if not hasattr(slide_encoder, "checkpoint_sha256"):
                raise ValueError(
                    "use_global_slide=True requires slide_encoder to expose checkpoint_sha256 (the "
                    "real checkpoint's own SHA256, computed by FrozenGigaPathSlideEncoder at "
                    "construction) so gigapath_checkpoint_sha256 can be verified against it"
                )
            if str(slide_encoder.checkpoint_sha256) != str(gigapath_checkpoint_sha256):
                raise ValueError(
                    f"gigapath_checkpoint_sha256 {gigapath_checkpoint_sha256!r} does not match "
                    f"slide_encoder's actual loaded checkpoint SHA256 "
                    f"{slide_encoder.checkpoint_sha256!r} -- a caller cannot supply an unrelated string"
                )
        self.slide_encoder = slide_encoder
        self.gigapath_checkpoint_sha256 = gigapath_checkpoint_sha256

        if use_regional_he:
            self.regional_token_proj = nn.Sequential(
                nn.LayerNorm(image_feature_dim), nn.Linear(image_feature_dim, hidden_dim),
            )
        else:
            self.regional_token_proj = None

        # The real, trainable "weighted_linear" gene conditioning encoder
        # (CONTRACT.md section 10's frozen choice) -- fixes a real,
        # confirmed gap (2nd Codex re-audit of commit 547f51e): the
        # module existed (models/gene_encoder.py) but nothing called it.
        # Sourced from observed_full_gene_expression (the untouched real
        # values, always-populated) -- SpatialFieldInputs never had a
        # separate precomputed conditioning field to begin with (a 3rd-
        # round Codex re-audit of commit ca7cf53 flagged the interim
        # version of that field as a dangerous unused input; removed
        # entirely rather than kept-but-optional).
        self.gene_encoder = WeightedGeneExpressionEncoder(n_genes, gex_feature_dim)
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
        # Iterative refinement over the PREDICTED expression field, applied
        # after the transport head. Query spots already exchange hidden states
        # through the backbone, but never predicted EXPRESSION -- and the
        # transport head's prediction is a convex combination of observed
        # spots' values, so a spot deep inside a hole, far from any observed
        # neighbour, receives almost nothing. Refinement propagates the
        # boundary inward one step at a time.
        #
        # 0 steps is a strict no-op: no module is constructed and the forward
        # path is unchanged, which matters because every architecture in this
        # project inherits this class.
        #
        # Constructed LAST, after every other submodule. Building it earlier
        # consumes draws from the global RNG, which changes the initialisation
        # of everything constructed after it -- so a refinement arm and its
        # control would silently differ in weights they are supposed to share.
        # This project synchronises initialisations across arms on purpose;
        # ordering is what keeps that true here.
        if n_refinement_steps < 0:
            raise ValueError("n_refinement_steps must be non-negative")
        self.n_refinement_steps = int(n_refinement_steps)
        self.expression_refiner = None
        if n_refinement_steps > 0:
            # Imported lazily: the module lives under conditional_wae, whose
            # package __init__ pulls in the whole WAE stack, and importing that
            # from models/ would risk an import cycle. Nothing is imported when
            # refinement is off, which is every existing architecture.
            from gen3_multiscale.conditional_wae.spatial_refinement import (
                SpatialExpressionRefiner,
            )

            self.expression_refiner = SpatialExpressionRefiner(
                n_genes, hidden_dim,
                gex_feature_dim=refinement_gex_feature_dim,
                hidden_dim=refinement_hidden_dim,
                k_neighbors=refinement_k_neighbors,
            )

    def _observed_tokens(self, inputs: SpatialFieldInputs, device: torch.device) -> torch.Tensor:
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed,
            torch.as_tensor(inputs.boundary_idx, device=device),
            torch.as_tensor(inputs.boundary_ring, device=device),
        )
        # 16th Codex re-audit (Step 5 Part 2), CONFIRMED real: this used
        # to be hardcoded to "always available" -- example_builder.py now
        # produces a genuine per-spot flag (observed_image_available),
        # threaded through here for real.
        modality_flags = torch.as_tensor(
            inputs.observed_image_available, dtype=torch.float32, device=device,
        ).unsqueeze(-1)
        # gex_features is computed HERE by the real trainable gene
        # encoder from the untouched observed_full_gene_expression (see
        # the module-level note on self.gene_encoder above).
        observed_expr = torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32, device=device)
        return self.spot_token(
            image_features=torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32, device=device),
            gex_features=self.gene_encoder(observed_expr),
            coords=torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device),
            boundary_ring=full_ring,
            modality_flags=modality_flags,
        )

    def _candidate_pool(
        self, inputs: SpatialFieldInputs, observed_tokens: torch.Tensor, query_coords: torch.Tensor, device: torch.device,
    ):
        """Local candidates are genuinely PER-QUERY (each query's own
        true local_k nearest observed spots) and stay a dense
        [Nq, local_k, G] tensor -- local_k is small (32) by construction,
        so this is cheap. Boundary candidates are the SAME set for every
        query in the item -- boundary_expr/boundary_hidden are kept
        UN-broadcast ([n_boundary, G]/[n_boundary, H]) here and handed to
        GeneValueTransportHead's shared_candidate_* path (models/
        transport_head.py), which combines them with the per-query local
        pool via one joint softmax without ever materializing a
        [Nq, n_boundary, G] tensor. This was a real, confirmed memory bug
        in an earlier version of this method (broadcasting boundary
        expression per query before concatenating with local expression --
        for 500 queries x 500 boundary candidates x ~17,000 genes x 4
        bytes, roughly 17 GB for that one tensor) caught by an external
        Codex audit against commit 386bcf4, verified against the actual
        code before fixing, not accepted on faith. boundary_geometry
        below IS still [Nq, n_boundary, 3] -- relative geometry
        legitimately differs per query even though candidate IDENTITY is
        shared, and 3 floats/candidate is not the memory problem."""
        observed_coords = torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device)
        observed_expr = torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32, device=device)
        local_idx = torch.as_tensor(inputs.query_local_neighbor_idx, dtype=torch.long, device=device)
        boundary_idx = torch.as_tensor(inputs.boundary_idx, dtype=torch.long, device=device)

        local_hidden = observed_tokens[local_idx]  # [Nq, local_k, H]
        local_coords = observed_coords[local_idx]  # [Nq, local_k, 2]
        local_geometry = compute_relative_geometry(query_coords, local_coords)
        local_expr = observed_expr[local_idx]  # [Nq, local_k, G]

        boundary_hidden = observed_tokens[boundary_idx]  # [n_boundary, H] -- SHARED, not broadcast
        boundary_coords = observed_coords[boundary_idx]  # [n_boundary, 2]
        boundary_geometry = compute_relative_geometry(query_coords, boundary_coords)  # [Nq, n_boundary, 3]
        boundary_expr = observed_expr[boundary_idx]  # [n_boundary, G] -- SHARED, not broadcast

        return {
            "local_hidden": local_hidden,
            "local_geometry": local_geometry,
            "local_expression": local_expr,
            "boundary_hidden": boundary_hidden,
            "boundary_geometry": boundary_geometry,
            "boundary_expression": boundary_expr,
        }

    def _harmonic_anchor(self, inputs: SpatialFieldInputs, query_coords: torch.Tensor, device: torch.device) -> torch.Tensor:
        anchor = harmonic_interpolation(
            observed_coords=np.asarray(inputs.observed_coords, dtype=np.float64),
            observed_expression=np.asarray(inputs.observed_full_gene_expression, dtype=np.float64),
            query_coords=np.asarray(inputs.query_coords, dtype=np.float64),
            k_neighbors=self.harmonic_k_neighbors,
        )
        return torch.from_numpy(anchor).to(dtype=query_coords.dtype, device=device)

    def _require_wsi_context(self, inputs: SpatialFieldInputs, requiring_flag: str) -> None:
        if inputs.wsi_tile_features is None:
            raise ValueError(
                f"{requiring_flag}=True requires wsi_tile_features/wsi_tile_longnet_coords/"
                "wsi_tile_regional_coords/full_slide_coord_bounds/slide_cache_namespace on "
                "SpatialFieldInputs -- build the example with a real slide_context "
                "(gen3_multiscale.data.slide_context.load_slide_context) passed to "
                "example_builder.build_spatial_field_example"
            )

    def _regional_he_tokens(
        self, inputs: SpatialFieldInputs, query_coords: torch.Tensor, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pools this item's visible WSI tiles into a stable
        regional_grid_size x regional_grid_size grid (models.slide_encoder.
        pool_regional_tokens, using full_slide_coord_bounds so grid cell
        (i, j) refers to the SAME physical region regardless of which
        hole this item's mask cut), projects the pooled tile FEATURES
        into hidden_dim, and computes each remaining query's relative
        geometry to the grid cells' real centers. Cells with zero
        visible tiles are EXCLUDED from the returned tensors entirely
        (index-selected out, never zero-valued-but-still-attended) --
        the same "excluded means absent from the tensor, not masked"
        discipline this module already uses for boundary_idx/local_idx."""
        self._require_wsi_context(inputs, "use_regional_he")
        tile_features = torch.as_tensor(inputs.wsi_tile_features, dtype=torch.float32, device=device)
        tile_coords = torch.as_tensor(inputs.wsi_tile_regional_coords, dtype=torch.float32, device=device)
        tokens, available = pool_regional_tokens(
            tile_features.cpu().numpy(), tile_coords.cpu().numpy(), inputs.full_slide_coord_bounds,
            grid_size=self.regional_grid_size,
        )
        available_idx = available.nonzero(as_tuple=True)[0]
        if available_idx.numel() == 0:
            raise ValueError(
                f"{inputs.sample_id}: no regional grid cell has any visible WSI tile -- "
                "use_regional_he=True cannot proceed with zero regional tokens"
            )
        centers = regional_grid_cell_centers(inputs.full_slide_coord_bounds, self.regional_grid_size)
        regional_features = tokens[available_idx].to(device=device, dtype=torch.float32)
        regional_centers = centers[available_idx].to(device=device, dtype=query_coords.dtype)
        regional_hidden = self.regional_token_proj(regional_features)
        regional_geometry = compute_relative_geometry(query_coords, regional_centers)
        return regional_hidden, regional_geometry

    def _global_slide_vector(self, inputs: SpatialFieldInputs, device: torch.device) -> torch.Tensor:
        """Runs the injected, frozen FrozenGigaPathSlideEncoder ONCE over
        this item's visible WSI tiles in their REAL, unnormalized
        GigaPath LongNet target-MPP coordinates (wsi_tile_longnet_coords
        -- NEVER wsi_tile_regional_coords, which would silently corrupt
        LongNet's real positional encoding, see SpatialFieldInputs' own
        docstring). The cache namespace combines the data layer's own
        real cache material (content hash + visible-tile-set identity,
        already bound into inputs.slide_cache_namespace) with THIS
        model's checkpoint SHA256 and architecture/version -- properties
        of which model is running, supplied at construction, never
        assumed by the data layer (16th Codex re-audit's complete
        cache-key requirement)."""
        self._require_wsi_context(inputs, "use_global_slide")
        cache_namespace = (
            f"{inputs.slide_cache_namespace}:checkpoint={self.gigapath_checkpoint_sha256}:"
            f"model={self.model_architecture_version}"
        )
        tile_features = torch.as_tensor(inputs.wsi_tile_features, dtype=torch.float32, device=device)
        longnet_coords = torch.as_tensor(inputs.wsi_tile_longnet_coords, dtype=torch.float32, device=device)
        return self.slide_encoder(tile_features, longnet_coords, cache_namespace)

    def forward(self, inputs: SpatialFieldInputs) -> dict:
        # 17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
        # before Step 6/7"), CONFIRMED real: `next(self.parameters())`
        # picks whatever parameter happens to be registered FIRST --
        # self.slide_encoder (when given) is registered before
        # self.gene_encoder/spot_token/backbone/transport_head, AND
        # FrozenGigaPathSlideEncoder.forward() independently moves ITS
        # OWN frozen submodule onto CUDA lazily, per call, regardless of
        # where the rest of this model lives. A CPU-resident learned
        # model (e.g. during CPU-only evaluation) that had already run
        # one use_global_slide forward pass would then silently pick
        # CUDA as `device` on the NEXT item, while its actually-trainable
        # layers remained on CPU -- a device mismatch. self.gene_encoder
        # always exists, is always genuinely trainable, and is never
        # independently relocated by any other code path, so it is the
        # correct, stable device source.
        device = next(self.gene_encoder.parameters()).device
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        depth = torch.as_tensor(inputs.query_depth_to_boundary, dtype=torch.long, device=device)
        hole_geometry = compute_hole_geometry(query_coords)
        query_hidden = self.query_token(query_coords, depth, hole_geometry)

        observed_tokens = self._observed_tokens(inputs, device)
        pool = self._candidate_pool(inputs, observed_tokens, query_coords, device)

        block_kwargs = dict(
            local_hidden=pool["local_hidden"], local_geometry=pool["local_geometry"],
            boundary_hidden=pool["boundary_hidden"], boundary_geometry=pool["boundary_geometry"],
        )
        # Transport candidates shared across every query in the item
        # (boundary, and global-GEX inducing candidates below) -- kept
        # UN-broadcast ([S, G], never [Nq, S, G]) and handed to
        # GeneValueTransportHead's shared_candidate_* path, which combines
        # them with the per-query local pool via one joint softmax
        # without ever materializing a dense [Nq, S, G] tensor. See
        # _candidate_pool's docstring for the memory-scaling bug this
        # replaced (confirmed by an external Codex audit, fixed after
        # verifying against the actual code).
        shared_hidden_parts = [pool["boundary_hidden"]]
        shared_geometry_parts = [pool["boundary_geometry"]]
        shared_expression_parts = [pool["boundary_expression"]]
        if self.use_global_gex:
            gex_out = self.gex_pool(
                observed_tokens, torch.as_tensor(inputs.observed_full_gene_expression, dtype=torch.float32, device=device),
            )
            n_inducing = gex_out["hidden"].shape[0]
            # inducing tokens have no real position -- a sentinel relative
            # geometry (zero direction, a fixed "far" distance) marks them
            # as categorically distinct from real spatial candidates,
            # mirroring HierarchicalGeneTransportRegressor's own
            # use_global_candidate sentinel (src/models/registry.py).
            sentinel = torch.zeros(query_coords.shape[0], n_inducing, 3, device=device)
            sentinel[..., 2] = 3.0
            block_kwargs["gex_inducing_hidden"] = gex_out["hidden"]
            block_kwargs["gex_inducing_geometry"] = sentinel
            # Real bug fixed here (Codex audit finding #6, confirmed):
            # InducedGlobalGEXPool computes a genuine value-preserving
            # convex candidate per inducing token (gex_out["expression"]),
            # but an earlier version of this method only ever forwarded
            # gex_out["hidden"] into the backbone's attention and never
            # added gex_out["expression"] to the transport head's own
            # candidate pool -- the real GEX-mixture candidates were
            # computed and then silently discarded. Now included as a
            # third shared-candidate group alongside boundary spots.
            shared_hidden_parts.append(gex_out["hidden"])
            shared_geometry_parts.append(sentinel)
            shared_expression_parts.append(gex_out["expression"])
        if self.use_regional_he:
            regional_hidden, regional_geometry = self._regional_he_tokens(inputs, query_coords, device)
            block_kwargs["regional_hidden"] = regional_hidden
            block_kwargs["regional_geometry"] = regional_geometry
            # Regional H&E feeds ONLY the backbone's hidden-state
            # cross-attention branch above -- structurally never appended
            # to shared_hidden_parts/shared_expression_parts, so it can
            # never become a literal GEX value candidate the transport
            # head could select as a prediction (16th Codex re-audit's
            # "regional/global H&E enters hidden conditioning only"
            # requirement -- true by construction, not by a runtime
            # check, exactly like gex_inducing_expression's structural
            # separation from the backbone-only regional/boundary path
            # above).
        if self.use_global_slide:
            block_kwargs["global_slide_vector"] = self._global_slide_vector(inputs, device)

        # end-of-branch invariant, verifiable by any caller/test: neither
        # branch above ever touched shared_hidden_parts/shared_expression_parts.

        final_query_hidden = self.backbone(query_hidden, query_coords, **block_kwargs)

        anchor_expression = self._harmonic_anchor(inputs, query_coords, device) if self.use_anchor_blend else None
        out = self.transport_head(
            final_query_hidden, pool["local_hidden"], pool["local_geometry"], pool["local_expression"],
            anchor_expression=anchor_expression,
            shared_candidate_hidden=torch.cat(shared_hidden_parts, dim=0),
            shared_candidate_relative_geometry=torch.cat(shared_geometry_parts, dim=1),
            shared_candidate_expression=torch.cat(shared_expression_parts, dim=0),
        )
        if self.expression_refiner is not None and self.n_refinement_steps > 0:
            from gen3_multiscale.conditional_wae.spatial_refinement import refine_expression

            # Refines the model's OWN prediction over the query spots only, so
            # no observed or target expression is read and the leakage contract
            # is unchanged. query_coords already spans exactly the predicted
            # spots here, unlike the conditional-WAE path where it spans the
            # whole slide.
            out["expression"] = refine_expression(
                self.expression_refiner, out["expression"], final_query_hidden,
                query_coords, n_steps=self.n_refinement_steps,
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
        use_regional_he: bool = False,
        use_global_slide: bool = False,
        global_slide_dim: int = 768,
        regional_grid_size: int = 4,
        slide_encoder: FrozenGigaPathSlideEncoder | None = None,
        gigapath_checkpoint_sha256: str | None = None,
        model_architecture_version: str = "gen3-multiscale-shared-field-v1",
    ):
        super().__init__()
        verify_gene_residual_basis(gene_basis, gene_names)
        self.gene_basis = gene_basis  # metadata only (rank, gene_names, hash) -- see _gene_basis_matrix below
        # GeneResidualBasis is a plain frozen dataclass, not an nn.Module,
        # so `gene_basis.basis` would NOT move when this Architecture4
        # instance is sent to a CUDA device (a real bug, confirmed by an
        # external Codex audit: a CUDA-resident model would keep calling
        # self.gene_basis.to_coefficients/from_coefficients against a
        # CPU-resident tensor). Registering a CLONE as a buffer makes
        # `.to(device)` move it along with every other tensor in this
        # module; to_coefficients/from_coefficients below use this buffer
        # directly (the same `residual @ basis.T` / `coefficients @ basis`
        # math GeneResidualBasis itself uses) rather than the dataclass's
        # own (still CPU/original-device) tensor.
        self.register_buffer("_gene_basis_matrix", gene_basis.basis.clone())
        self.n_flow_samples = n_flow_samples
        self.n_ode_steps = n_ode_steps

        # 16th Codex re-audit (Step 5 Part 2), CONFIRMED real: this used
        # to hardcode-omit use_regional_he/use_global_slide/global_slide_dim
        # and every slide-encoder param entirely -- Architecture3's OWN
        # kwargs.setdefault(False) (or its global_slide_dim default) would
        # then silently apply even if a caller asked THIS Architecture4
        # for regional/global H&E, meaning "Architecture 4 reuses
        # Architecture 3's exact conditioner" was never actually true for
        # those flags. Now forwarded explicitly, so self.conditioner
        # really is what a standalone Architecture3(**these same kwargs)
        # would be.
        self.conditioner = Architecture3(
            n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale, n_gex_inducing=n_gex_inducing,
            harmonic_k_neighbors=harmonic_k_neighbors,
            use_regional_he=use_regional_he, use_global_slide=use_global_slide,
            global_slide_dim=global_slide_dim,
            regional_grid_size=regional_grid_size, slide_encoder=slide_encoder,
            gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
            model_architecture_version=model_architecture_version,
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

    def _prepare_target_expression(self, target_expression: torch.Tensor | np.ndarray, deterministic_mean: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Move `target_expression` onto the model's own device/dtype and
        validate its shape/finiteness against `deterministic_mean` --
        real, confirmed gap (6th Codex re-audit of commit 06f5cce): "Both
        flow-loss methods should also move and validate target_expression
        against the model's actual device/dtype and check shape/
        finiteness." Without this, a caller passing a target on the wrong
        device would previously hit an opaque CUDA/CPU mismatch error
        deep inside the einsum in `to_coefficients`, a shape mismatch
        would silently broadcast into a much larger tensor instead of
        failing at the actual point of the mistake, and a non-finite
        target (a real, plausible upstream data bug) would silently
        poison the flow loss with NaN/Inf rather than failing loudly at
        the boundary where it enters this model.

        Uses `torch.as_tensor(...)`, not a bare `.to(...)` call (real,
        confirmed gap -- 7th Codex re-audit of commit 2782ff0):
        `SpatialFieldTargets.query_expression` -- the natural, real
        source of this argument -- is typed and documented as a plain
        `np.ndarray` throughout `data/example.py`, and a numpy array has
        no `.to()` method at all; a caller passing that natural target
        object directly (exactly what a real trainer would do) would
        have hit an `AttributeError` here instead of the validation this
        method exists to provide. `torch.as_tensor` accepts both a numpy
        array and an existing tensor uniformly."""
        target_expression = torch.as_tensor(target_expression, device=device, dtype=deterministic_mean.dtype)
        if target_expression.shape != deterministic_mean.shape:
            raise ValueError(
                f"target_expression {tuple(target_expression.shape)} must match the conditioner's "
                f"deterministic_mean shape {tuple(deterministic_mean.shape)}"
            )
        if not torch.isfinite(target_expression).all():
            raise ValueError("target_expression contains non-finite (NaN/Inf) values")
        return target_expression

    def compute_flow_matching_loss(
        self, inputs: SpatialFieldInputs, target_expression: torch.Tensor | np.ndarray,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        # 17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
        # before Step 6/7"): same device-selection bug as
        # _SharedFieldArchitecture.forward() -- self.conditioner's own
        # slide_encoder (when given) can independently move itself to
        # CUDA per call, and next(self.parameters()) would pick it up
        # first (registered before self.velocity_network). velocity_network
        # always exists, is always genuinely trainable, and is never
        # independently relocated by any other code path.
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        deterministic_mean = conditioner_out["expression"].detach()
        target_expression = self._prepare_target_expression(target_expression, deterministic_mean, device)
        target_residual = target_expression - deterministic_mean
        target_coefficients = target_residual @ self._gene_basis_matrix.T  # GeneResidualBasis.to_coefficients, device-correct
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        return flow_matching_loss(self.velocity_network, target_coefficients, query_coords, query_hidden, generator=generator)

    def compute_losses(
        self, inputs: SpatialFieldInputs, target_expression: torch.Tensor | np.ndarray,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Runs self.conditioner exactly ONCE and derives both the
        deterministic conditioner output and the flow-matching loss from
        that single pass. Calling forward() and compute_flow_matching_loss()
        separately in the same training step runs self.conditioner twice;
        with dropout active (the default everywhere in this package) the
        two calls draw different dropout masks, so the flow loss's
        deterministic_mean would silently disagree with the mean the
        reconstruction loss was computed against (Codex audit finding
        against commit c02a5d1). This method is the one callers needing
        both losses per step should use; forward() and
        compute_flow_matching_loss() are kept unchanged for callers that
        only need one or the other and for the existing tests exercising
        them independently.

        `generator` (added for Step 6's real trainer, Codex audit of
        commit 27e1232): flow_matching_loss draws a random t and x0 from
        the GLOBAL torch RNG when no generator is given, making the flow
        loss non-deterministic across repeated calls with the same
        inputs -- fine for training, but wrong for VALIDATION, where a
        caller computing model-selection metrics needs reproducible
        numbers. Passing a caller-owned `torch.Generator` here makes the
        flow loss (and only the flow loss) reproducible without touching
        the global RNG stream at all."""
        # 17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
        # before Step 6/7"): same device-selection bug as
        # _SharedFieldArchitecture.forward() -- self.conditioner's own
        # slide_encoder (when given) can independently move itself to
        # CUDA per call, and next(self.parameters()) would pick it up
        # first (registered before self.velocity_network). velocity_network
        # always exists, is always genuinely trainable, and is never
        # independently relocated by any other code path.
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        deterministic_mean = conditioner_out["expression"].detach()
        target_expression = self._prepare_target_expression(target_expression, deterministic_mean, device)
        target_residual = target_expression - deterministic_mean
        target_coefficients = target_residual @ self._gene_basis_matrix.T  # GeneResidualBasis.to_coefficients, device-correct
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        flow_loss = flow_matching_loss(self.velocity_network, target_coefficients, query_coords, query_hidden, generator=generator)
        return {**conditioner_out, "flow_loss": flow_loss}

    @torch.no_grad()
    def sample_predictive_distribution(
        self, inputs: SpatialFieldInputs, n_samples: int | None = None, n_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Draws multiple low-rank residual-field samples and adds the
        corresponding full-gene residuals to the deterministic transport
        mean. "Primary PCC/RMSE comparison should use the predictive mean
        across samples" -- returned as predictive_mean (and also as
        "expression", so this dict is drop-in compatible with the
        conditioner-only forward()'s output for anything that only reads
        "expression"). Also reports predictive_std (uncertainty) and the
        raw per-sample field for diversity diagnostics (Phase 7).

        `generator` (Adam's Step 6 audit #1 of commit a32051b): threaded
        through to `sample_residual_coefficients`'s own x0 draws so that
        validation/evaluation/overfit-gate callers get IDENTICAL sampled
        predictions across a resume or repeated evaluation, the same
        reproducibility guarantee `compute_losses`/`compute_flow_matching_loss`
        already give the training-time flow loss via their own
        `generator` argument."""
        # 17th Codex re-audit (Step 5 Part 2 launch blocker, "Important
        # before Step 6/7"): same device-selection bug as
        # _SharedFieldArchitecture.forward() -- self.conditioner's own
        # slide_encoder (when given) can independently move itself to
        # CUDA per call, and next(self.parameters()) would pick it up
        # first (registered before self.velocity_network). velocity_network
        # always exists, is always genuinely trainable, and is never
        # independently relocated by any other code path.
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"]
        deterministic_mean = conditioner_out["expression"]
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        n_query = query_coords.shape[0]

        coefficient_samples = sample_residual_coefficients(
            self.velocity_network, n_query, query_coords, query_hidden,
            n_samples=n_samples or self.n_flow_samples, n_steps=n_steps or self.n_ode_steps,
            generator=generator,
        )
        residual_samples = coefficient_samples @ self._gene_basis_matrix  # GeneResidualBasis.from_coefficients, device-correct  # [S, Nq, G]
        predictive_samples = deterministic_mean[None] + residual_samples
        predictive_mean = predictive_samples.mean(dim=0)
        # unbiased=False: torch's default (unbiased=True, i.e. dividing by
        # n-1) returns all-NaN with a UserWarning whenever n_samples == 1,
        # since there are then zero degrees of freedom (confirmed directly:
        # torch.randn(1, 5).std(dim=0) -> NaN). A single-sample draw has a
        # well-defined population std of exactly 0, which unbiased=False
        # (dividing by n) reports correctly for every n_samples >= 1.
        predictive_std = predictive_samples.std(dim=0, unbiased=False)
        return {
            "expression": predictive_mean,
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "predictive_samples": predictive_samples,
            "deterministic_mean": deterministic_mean,
        }
