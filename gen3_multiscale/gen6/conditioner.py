"""Deterministic Gen6 component-screen conditioners."""
from __future__ import annotations

import torch
import torch.nn as nn

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen6.contract import Gen6ArmSpec
from gen3_multiscale.gen6.fusion import (
    BidirectionalCrossAttentionFusion, MoMEFusion,
)
from gen3_multiscale.gen6.geometry import install_query_geometry
from gen3_multiscale.models.geometry_utils import scatter_boundary_ring


class Gen6Conditioner(Gen4Conditioner):
    """Gen4's audited spatial field with one explicit Gen6 component swap."""
    def __init__(self, *, arm_spec: Gen6ArmSpec, fusion_heads: int = 4,
                 coord_scale: float = 0.0, **kwargs):
        self.arm_spec = arm_spec
        super().__init__(**kwargs)
        self.gen6_fusion = None
        self.fused_to_image = None
        self.fused_to_gene = None
        if arm_spec.fusion in {"mome", "bidirectional_cross_attention"}:
            image_dim = int(kwargs.get("image_feature_dim", 1536))
            gene_dim = int(kwargs["gex_feature_dim"])
            hidden_dim = int(kwargs.get("hidden_dim", 512))
            fusion_cls = MoMEFusion if arm_spec.fusion == "mome" else BidirectionalCrossAttentionFusion
            self.gen6_fusion = fusion_cls(image_dim, gene_dim, hidden_dim, n_heads=fusion_heads)
            self.fused_to_image = nn.Linear(hidden_dim, image_dim)
            self.fused_to_gene = nn.Linear(hidden_dim, gene_dim)
        install_query_geometry(self, arm_spec.geometry, coord_scale=coord_scale)
        self.model_architecture_version = f"gen6-{arm_spec.arm}-v1"

    def _observed_tokens(self, inputs, device: torch.device) -> torch.Tensor:
        if self.gen6_fusion is None:
            return super()._observed_tokens(inputs, device)
        context_embedding = getattr(inputs, "context_gex_embedding", None)
        if context_embedding is None:
            raise ValueError(f"{self.arm_spec.arm} requires context_gex_embedding")
        n_observed = inputs.observed_coords.shape[0]
        full_ring = scatter_boundary_ring(
            n_observed, torch.as_tensor(inputs.boundary_idx, device=device),
            torch.as_tensor(inputs.boundary_ring, device=device),
        )
        available = torch.as_tensor(inputs.observed_image_available, dtype=torch.bool, device=device)
        image = torch.as_tensor(inputs.observed_gigapath_features, dtype=torch.float32, device=device)
        gene = self.gex_context_proj(
            torch.as_tensor(context_embedding, dtype=torch.float32, device=device)
        )
        fused = self.gen6_fusion(image, gene, available)
        return self.spot_token(
            image_features=self.fused_to_image(fused),
            gex_features=self.fused_to_gene(fused),
            coords=torch.as_tensor(inputs.observed_coords, dtype=torch.float32, device=device),
            boundary_ring=full_ring,
            modality_flags=available.float().unsqueeze(-1),
        )
