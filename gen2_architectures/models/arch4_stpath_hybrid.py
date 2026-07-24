"""Architecture 4 — STPath-grounded hybrid: frozen real pretrained STPath
weights + a small scFoundation additive residual.

Hypothesis: rather than a GPT-only design, this leans on what our OWN
lung_round experiments already showed — STPath's real frozen pretrained
backbone consistently outperformed every from-scratch alternative tried so
far — corrected for the one concrete, verified bug found this session
(library-size-normalization mismatch: STPath's real pretrained weights and
the reference notebook only ever see log1p(RAW counts), never library-size-
normalized log1p), with scFoundation layered in as a small ADDITIVE
residual (not a replacement for STPath's own gene pathway).

GPT review's #6 concern (this is the riskiest of the 4 — STPath's frozen
weights may be domain-mismatched outside the organs it was pretrained on)
is handled at the CONFIG/training-budget level (half compute budget, see
gen2_architectures/README.md's Architecture 4 section), not in this module.

Deliberately the smallest-trainable-parameter architecture of the 4: STPath
itself stays fully frozen (pretrained=True), so the ONLY trainable
parameters are the scFoundation residual projection head
(ScFoundationGeneEncoder) and its injection into STPath's hidden state
(residual_proj) — both already built inside STPathContextEncoder
(gen2_architectures/models/stpath_encoder.py, new_gene_encoder_type=
"scfoundation", extended there 2026-07-25 for this architecture).

Known limitation, inherited from STPathContextEncoder and explicitly NOT
worked around here: organ_type/tech_type are FIXED at construction (STPath's
own real IDTokenizer vocabulary, not a per-sample runtime value) — this
architecture is not naturally multi-organ the way Architectures 1-3 are with
their OrganTechEmbedding. A full-HEST-1k run of this architecture should
either (a) restrict training to one organ/tech, matching what
STPathContextEncoder can actually condition on, or (b) train several
per-organ instances. See README's open-risk note.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gen2_architectures.models.stpath_encoder import STPathContextEncoder
from gen2_architectures.models.eval_compat import DeterministicSampleMixin


class Architecture4(DeterministicSampleMixin, nn.Module):
    def __init__(
        self, gene_names: list[str], gene_voc_path: str, model_weight_path: str,
        organ_type: str, tech_type: str, scfoundation_dim: int,
        hidden_dim: int = 256, device: str = "cpu",
    ):
        super().__init__()
        self.stpath = STPathContextEncoder(
            gene_names=gene_names, gene_voc_path=gene_voc_path, model_weight_path=model_weight_path,
            organ_type=organ_type, tech_type=tech_type, hidden_dim=hidden_dim, device=device,
            new_gene_encoder_type="scfoundation", scfoundation_dim=scfoundation_dim,
            pretrained=True,
            # STPath's real pretrained weights only ever see
            # log1p(raw_counts) -- context["expression"] must already be
            # RAW-COUNT log1p by the time it reaches this model (see
            # gen2_architectures/training/data_prep.py's
            # "stpath_native_log1p" preprocessing mode), so no further
            # log1p is applied here.
            input_already_log1p=True,
        )
        # Exposed so the shared evaluation harness
        # (gen2_architectures/evaluation/audit_evaluation.py's
        # _target_for_model) automatically slices the target expression to
        # the same STPath-vocabulary-overlapping gene subset this model
        # actually predicts -- STPath's released vocabulary does not cover
        # every gene in our shared training panel, and this is a real,
        # expected restriction (documented in the original codebase's own
        # n_evaluated_genes handling), not a bug to work around.
        self.register_buffer(
            "_decoder_target_col_idx",
            torch.tensor(self.stpath._valid_gene_pos, dtype=torch.long),
        )

    def forward(self, context: dict, query: dict) -> torch.Tensor:
        """context/query: see gen2_architectures/data/masked_item.py's
        build_masked_item output. context["images"]/query["images"] must be
        precomputed GigaPath features [N, 1536] (STPath's own image
        tokenizer contract). context["extra_features"]: precomputed
        scFoundation cell embeddings for the scFoundation residual (see
        arch2_scfoundation.py's docstring for why scFoundation, unlike
        STPath, does NOT need special raw-count preprocessing).

        Returns predicted expression [N_query, n_valid_genes] where
        n_valid_genes = len(self._decoder_target_col_idx) — narrower than
        the full training gene panel by construction (STPath's vocabulary
        does not cover every gene)."""
        return self.stpath(
            context_coords=context["coords"], context_expression=context["expression"],
            query_coords=query["coords"],
            context_images=context["images"], query_images=query["images"],
            context_image_available=context.get("image_available"),
            query_image_available=query.get("image_available"),
            context_scfoundation_features=context.get("extra_features"),
            return_official_predictions=True,
        )
