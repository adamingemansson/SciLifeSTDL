"""Context-only STPath encoder -- GEN4_CONTRACT.md section 8.

Wraps the real, already-verified `src.models.stpath_encoder.STPathContextEncoder`
but adds `encode_context_only`, a NEW method whose signature has no
query-shaped parameter at all -- structurally incapable of leaking query
H&E/GEX into STPath's spatial transformer, unlike that class's own
`forward()` (built for a different, pilot-study use case that concatenates
context+query rows; `forward()` is never called by Gen4 code).

No real STPath package/weights are available in this environment. This
class is structurally complete and exercised in tests only via
`Gen4STPathStub` (tests/_gen4_fixtures.py), a tiny deterministic CPU-only
stand-in satisfying the same public interface; real-weight validation is
listed as an explicit gap in the runbook.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Gen4STPathContextEncoder(nn.Module):
    """`hidden_dim` is the encoder's own trainable projection width (the
    dimension `encode_context_only` returns) -- kept separate from
    STPath's internal `d_model` (its frozen pretrained backbone width),
    exactly like the base `STPathContextEncoder` already separates the
    two via its own `proj`/`embedding_norm`."""

    def __init__(
        self,
        gene_names: list[str],
        gene_voc_path: str,
        model_weight_path: str,
        organ_type: str = "Kidney",
        tech_type: str = "Visium",
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        from src.models.stpath_encoder import STPathContextEncoder

        # pretrained=True (mandatory -- Gen4 never trains STPath's own
        # backbone from scratch; only proj/embedding_norm are trainable,
        # matching the base class's own frozen-when-pretrained contract).
        self._base = STPathContextEncoder(
            gene_names=gene_names, gene_voc_path=gene_voc_path, model_weight_path=model_weight_path,
            organ_type=organ_type, tech_type=tech_type, hidden_dim=hidden_dim, device=device,
            new_gene_encoder_type="none", pretrained=True, input_already_log1p=True,
        )
        self.hidden_dim = hidden_dim

    def train(self, mode: bool = True):
        # STPath's own backbone stays frozen/eval regardless -- only
        # proj/embedding_norm (inside self._base) are ever trainable, and
        # nn.Module's default train() already reaches them correctly via
        # self._base's own child modules; this override exists only to
        # keep the frozen backbone explicitly pinned to eval(), mirroring
        # FrozenGigaPathSlideEncoder's identical discipline.
        super().train(mode)
        self._base.model.eval()
        return self

    def encode_context_only(
        self,
        context_coords: torch.Tensor,
        context_expression: torch.Tensor,
        context_image_features: torch.Tensor,
        context_image_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """context_coords: [n_context, 2]. context_expression: [n_context,
        n_genes], REAL observed values (every row here is context -- unlike
        the base class's mixed context+query batch, nothing here is ever
        masked). context_image_features: [n_context, 1536] precomputed
        GigaPath tile features (STPath's own image tokenizer input, see
        base class docstring) -- never raw pixels, never a query row.
        Returns [n_context, hidden_dim]. There is no query_* parameter
        anywhere in this signature."""
        n_context = context_coords.shape[0]
        if context_expression.shape[0] != n_context or context_image_features.shape[0] != n_context:
            raise ValueError("context_coords/context_expression/context_image_features must be row-aligned")
        device = context_coords.device
        base = self._base

        coords = context_coords[:, :2].clone()
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        coords = base._rescale_coords(coords)

        img_feats = context_image_features
        if context_image_available is not None:
            available = context_image_available.to(device).bool()
            img_feats = torch.where(available[:, None], img_feats, base.missing_image_token[None, :])

        ge_tokens = base.tokenizer.ge_tokenizer.mask_token.float().to(device).repeat(n_context, 1)
        processed_expression = context_expression  # input_already_log1p=True
        expr = processed_expression[:, base._valid_gene_pos]
        ge_tokens[:] = base.tokenizer.ge_tokenizer.convert_gene_exp_to_one_hot_tensor(
            base.tokenizer.ge_tokenizer.n_tokens, expr, base._context_gene_ids.to(device),
        )

        organ = base.tokenizer.organ_tokenizer.encode(base.organ_type, align_first=True)
        organ_ids = torch.full((n_context,), organ, dtype=torch.long, device=device)
        tech = base.tokenizer.tech_tokenizer.encode(base.tech_type, align_first=True)
        tech_ids = torch.full((n_context,), tech, dtype=torch.long, device=device)

        with torch.no_grad():  # STPath's own backbone is frozen
            _pred, x = base.model.prediction_head(
                img_tokens=img_feats, coords=coords, ge_tokens=ge_tokens,
                batch_idx=torch.zeros(n_context, dtype=torch.long, device=device),
                tech_tokens=tech_ids, organ_tokens=organ_ids, return_all=True,
            )
        x = base.embedding_norm(x)  # every row is context -- no query slice needed
        return base.proj(x)
