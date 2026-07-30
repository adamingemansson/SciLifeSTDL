"""Gen5LatentFlowModel -- GEN5_CONTRACT.md sections 2, 4.

Wraps a `Gen4Conditioner` (reused unmodified) and `models.flow.VelocityNetwork`
(reused unmodified, operating in the full autoencoder latent space rather
than Gen4's rank-64 residual-basis space) plus a frozen, shared
`ExpressionAutoencoder`. `compute_losses`/`sample_predictive_distribution`
never read `conditioner_out["expression"]` -- the deterministic
conditioner's own predicted mean is NEVER added to, or otherwise mixed
into, the returned prediction. This is the one structural rule that makes
Gen5 a genuinely different generative mechanism from Gen4, not merely a
relabeled residual flow -- see GEN5_CONTRACT.md section 1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from gen3_multiscale.gen4.conditioner import Gen4Conditioner
from gen3_multiscale.gen5.autoencoder import ExpressionAutoencoder, verify_expression_autoencoder_gene_names
from gen3_multiscale.models.flow import VelocityNetwork, flow_matching_loss, sample_residual_coefficients


class Gen5LatentFlowModel(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        gene_names: list[str],
        gex_feature_dim: int,
        autoencoder: ExpressionAutoencoder,
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
        use_regional_he: bool = False,
        use_global_gex: bool = True,
        regional_grid_size: int = 4,
        n_gex_inducing: int = 16,
        harmonic_k_neighbors: int = 6,
        n_flow_samples: int = 8,
        n_ode_steps: int = 20,
        gex_feature_source: str = "weighted_linear",
        gex_context_embedding_dim: int | None = None,
        image_feature_source: str = "precomputed",
        stpath_encoder: nn.Module | None = None,
        global_context_source: str = "none",
        global_slide_dim: int = 768,
        slide_encoder=None,
        gigapath_checkpoint_sha256: str | None = None,
        uni2_global_pool: nn.Module | None = None,
        model_architecture_version: str = "gen5-latent-flow-v1",
    ):
        super().__init__()
        verify_expression_autoencoder_gene_names(autoencoder, gene_names)
        if autoencoder.n_genes != n_genes:
            raise ValueError(f"autoencoder.n_genes ({autoencoder.n_genes}) must equal n_genes ({n_genes})")
        self.autoencoder = autoencoder
        for parameter in self.autoencoder.parameters():
            parameter.requires_grad_(False)
        self.autoencoder.eval()
        self.latent_dim = autoencoder.latent_dim
        self.n_flow_samples = n_flow_samples
        self.n_ode_steps = n_ode_steps
        self._conditioner_frozen = False

        self.conditioner = Gen4Conditioner(
            n_genes=n_genes, gex_feature_dim=gex_feature_dim, image_feature_dim=image_feature_dim,
            hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
            max_boundary_size=max_boundary_size, transport_heads=transport_heads,
            transport_temperature=transport_temperature, gene_gate_mode=gene_gate_mode,
            use_query_gate=use_query_gate, use_residual=use_residual, residual_rank=residual_rank,
            target_gene_scale=target_gene_scale,
            use_regional_he=use_regional_he, use_global_gex=use_global_gex, regional_grid_size=regional_grid_size,
            n_gex_inducing=n_gex_inducing, harmonic_k_neighbors=harmonic_k_neighbors,
            gex_feature_source=gex_feature_source, gex_context_embedding_dim=gex_context_embedding_dim,
            image_feature_source=image_feature_source, stpath_encoder=stpath_encoder,
            global_context_source=global_context_source, global_slide_dim=global_slide_dim,
            slide_encoder=slide_encoder, gigapath_checkpoint_sha256=gigapath_checkpoint_sha256,
            uni2_global_pool=uni2_global_pool, model_architecture_version=model_architecture_version,
        )
        self.velocity_network = VelocityNetwork(
            residual_rank=self.latent_dim, hidden_dim=hidden_dim, n_heads=n_heads, n_blocks=n_flow_blocks,
            dense_threshold=dense_threshold, sparse_k=sparse_k, chunk_size=chunk_size,
        )

    def freeze_conditioner(self) -> None:
        """Explicit, named freeze step -- mirrors Gen4/Architecture4's own
        `freeze_conditioner_initially` trainer-side discipline
        (configs/gen4/*_flow.yaml); not called automatically at
        construction, since a caller loading a real trained checkpoint
        must load weights BEFORE freezing."""
        for parameter in self.conditioner.parameters():
            parameter.requires_grad_(False)
        self.conditioner.eval()
        self._conditioner_frozen = True

    def train(self, mode: bool = True):
        """Codex audit finding, confirmed real (reproduced): calling
        `model.train()` on the WHOLE Gen5LatentFlowModel -- the ordinary
        start-of-epoch call any real training loop makes -- recursively
        calls `.train(mode)` on every submodule, including
        `self.conditioner`, silently RE-ENABLING dropout inside a
        conditioner `freeze_conditioner()` had just pinned to eval. The
        frozen conditioner must never leave eval mode once frozen,
        regardless of what mode is requested for the rest of the model --
        the same discipline `FrozenGigaPathSlideEncoder.train()` and
        `Gen4STPathContextEncoder.train()` already use for their own
        frozen backbones. The autoencoder is ALWAYS frozen (never
        trainable, by construction above) and is pinned the same way
        unconditionally."""
        super().train(mode)
        self.autoencoder.eval()
        if self._conditioner_frozen:
            self.conditioner.eval()
        return self

    def _prepare_target_expression(self, target_expression, device: torch.device) -> torch.Tensor:
        target_expression = torch.as_tensor(target_expression, device=device, dtype=torch.float32)
        if target_expression.shape[-1] != self.autoencoder.n_genes:
            raise ValueError(
                f"target_expression last dim ({target_expression.shape[-1]}) must equal "
                f"autoencoder.n_genes ({self.autoencoder.n_genes})"
            )
        if not torch.isfinite(target_expression).all():
            raise ValueError("target_expression contains non-finite (NaN/Inf) values")
        return target_expression

    def compute_losses(
        self, inputs, target_expression: torch.Tensor | np.ndarray, generator: torch.Generator | None = None,
    ) -> dict:
        """Runs self.conditioner exactly once; encodes the TRUE query
        expression through the frozen autoencoder to obtain z_target
        (training-only -- see GEN5_CONTRACT.md section 5, item 1/2).
        conditioner_out["expression"] (the deterministic mean) is read
        into the returned dict for diagnostic/logging purposes ONLY -- it
        is never combined with the flow loss or the generated latent."""
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"].detach()
        target_expression = self._prepare_target_expression(target_expression, device)
        with torch.no_grad():
            z_target = self.autoencoder.encode(target_expression)
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        flow_loss = flow_matching_loss(self.velocity_network, z_target, query_coords, query_hidden, generator=generator)
        return {
            "flow_loss": flow_loss,
            "query_hidden": conditioner_out["query_hidden"],
            "conditioner_expression_diagnostic_only": conditioner_out["expression"],
        }

    @torch.no_grad()
    def sample_predictive_distribution(
        self, inputs, n_samples: int | None = None, n_steps: int | None = None, generator: torch.Generator | None = None,
    ) -> dict:
        """`prediction = expression_decoder(generated_latent)` -- the ONLY
        allowed form (GEN5_CONTRACT.md section 1). `conditioner_out["expression"]`
        is computed (the conditioner must still run its forward pass to
        produce `query_hidden`) but is not read anywhere below."""
        device = next(self.velocity_network.parameters()).device
        conditioner_out = self.conditioner(inputs)
        query_hidden = conditioner_out["query_hidden"]
        query_coords = torch.as_tensor(inputs.query_coords, dtype=torch.float32, device=device)
        n_query = query_coords.shape[0]

        z_samples = sample_residual_coefficients(
            self.velocity_network, n_query, query_coords, query_hidden,
            n_samples=n_samples or self.n_flow_samples, n_steps=n_steps or self.n_ode_steps, generator=generator,
        )  # [S, Nq, latent_dim] -- pure Gaussian-noise-initialized ODE integration, never conditioner-seeded
        n_s = z_samples.shape[0]
        decoded = self.autoencoder.decode(z_samples.reshape(n_s * n_query, self.latent_dim))
        decoded_samples = decoded.reshape(n_s, n_query, self.autoencoder.n_genes)
        predictive_mean = decoded_samples.mean(dim=0)
        predictive_std = decoded_samples.std(dim=0, unbiased=False)
        return {
            "expression": predictive_mean,
            "predictive_mean": predictive_mean,
            "predictive_std": predictive_std,
            "predictive_samples": decoded_samples,
            "latent_samples": z_samples,
        }
