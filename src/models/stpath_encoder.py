"""
STPath as a pretrained H&E+expression conditioning encoder (task #18).

Wraps the real STPath model (Huang et al. 2025, bioRxiv, "STPath: A
Generative Foundation Model for Integrating Spatial Transcriptomics and
Whole Slide Images" — github.com/Graph-and-Geometric-Learning/STPath,
weights at huggingface.co/tlhuang/STPath). Verified 2026-07-15 by cloning
the actual repo and reading its source, not trusting the README alone
(the lesson from task #16/Mimyr) — unlike Mimyr, STPath's real code IS a
genuinely reusable, generic inference API (`STPathInference` in
`stpath/app/pipeline/inference.py`), not hardcoded to the authors' own
cluster paths/datasets. Its documented "in-context learning" mode (feed
real expression for some spots as context, get predictions for the rest)
maps almost exactly onto our own context/query masking setup.

This class replaces our ENTIRE SpatialContextEncoder for this arm, not a
branch fused into it — STPath's own spatial transformer already does
joint image+expression+organ+tech reasoning via k-NN attention over ALL
spots at once, so there's nothing for our own encoder to add on top. This
is the "does STPath's whole architecture help" arm of the task #19
three-way comparison, distinct from task #20 (Gigapath alone, fused into
our own encoder instead).

We call `model.prediction_head(..., return_all=True)` directly rather
than going through `STPathInference.inference()` (which only returns
final gene-expression predictions) — `return_all=True` additionally
returns the pre-head hidden state `x` [N, d_model=512], the actual
embedding we want as conditioning `c`. Token construction (coordinate
rescaling, masked/context gene tokens, organ/tech tokens) mirrors
`STPathInference`'s own internal logic as closely as possible, so this
stays faithful to how the real class calls the model — just exposing the
embedding instead of the final prediction.

SETUP (neither is a default dependency/step of this repo):
  1. Clone github.com/Graph-and-Geometric-Learning/STPath and run
     `pip install -e .` from that directory. Its own setup.py has no
     install_requires — additionally install `einops==0.8.0` (its README
     also lists `torch_geometric==2.6.1`, but inspecting
     stpath/tokenization/ge_tokenizer.py shows the only use of
     torch_geometric — `coalesce`, in `.encode()` — is behind a soft
     try/except ImportError at import time and is NOT on the code path
     this class actually uses (symbol2id / convert_gene_exp_to_one_hot_tensor),
     so it's likely not required for this specific usage — verify this
     yourself if you hit an ImportError).
  2. Download the pretrained weight from huggingface.co/tlhuang/STPath.
     No gated-access requirement was found when checked 2026-07-15
     (unlike Gigapath) — verify this yourself, HF repo settings change.
  3. The bundled gene vocabulary file at
     <cloned STPath repo>/utils_data/symbol2ensembl.json — pass its path
     as gene_voc_path below.
  4. Gigapath itself (task #20's GigapathPatchEncoder setup — STPath's
     image tokens are Gigapath tile-encoder features, feature_dim=1536).

Structural smoke-testing: tests/test_stpath_encoder.py, skips cleanly
without the above. CONFIRMED running end-to-end on real hardware
(2026-07-15, user's Mac, MPS backend, exp_hest1k_wae_gan_stpath.yaml,
50/50 masking draws in ~7s) after fixing two real bugs found along the
way: HEST-1k's real patch .h5 format (src/data/loaders.py) and
PYTORCH_ENABLE_MPS_FALLBACK timing (must be set before ANY MPS op runs
in the process — see src/training/train.py's top-of-file comment, not
the os.environ.setdefault below, which alone was NOT sufficient).
"""
from __future__ import annotations

import os

# STPath's own SpatialTransformer (stpath/model/nn_utils/fa.py create_frame,
# its "frame averaging" geometry step) calls torch.linalg.eigh, which isn't
# implemented on MPS (Apple Silicon) as of this writing - confirmed
# 2026-07-15 via a real crash (NotImplementedError: aten::_linalg_eigh...).
# This is inside the external stpath package's own code, not ours, so we
# can't fix it the way we fixed our own bicubic-interpolate MPS gap
# (moving that one op to CPU manually) without patching code outside this
# repo. PYTORCH_ENABLE_MPS_FALLBACK is PyTorch's own documented workaround
# for exactly this situation - it falls back to CPU only for the specific
# unimplemented op, not the whole model. Set here too (setdefault, so an
# explicit user setting always wins) as a belt-and-suspenders default for
# anyone importing this module directly — but this alone is NOT
# sufficient. Confirmed empirically 2026-07-15: PyTorch checks this env
# var once, early (not lazily per-op as an earlier version of this
# comment assumed) — setting it here, after src/training/train.py's
# _load_images had already dispatched MPS ops via the Gigapath precompute
# step earlier in the same process, produced the identical
# torch.linalg.eigh crash. The real fix is setting it before `import
# torch` at the top of the process's actual entry point
# (src/training/train.py / src/evaluation/run_comparison.py) — see the
# comment there.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

from src.models.conditioning import _load_gigapath_tile_encoder, _gigapath_preprocess_and_encode


class STPathContextEncoder(nn.Module):
    def __init__(self, gene_names: list[str], gene_voc_path: str, model_weight_path: str,
                 organ_type: str = "Kidney", tech_type: str = "Visium",
                 hidden_dim: int = 256, device: str = "cpu"):
        super().__init__()
        from stpath.model.model import STFM
        from stpath.model.nn_utils.config import ModelConfig
        from stpath.tokenization import (
            GeneExpTokenizer, ImageTokenizer, IDTokenizer, TokenizerTools, AnnotationTokenizer,
        )
        from stpath.data.dataset import rescale_coords

        self._rescale_coords = rescale_coords
        self.organ_type = organ_type
        self.tech_type = tech_type
        self.device_str = device

        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        config = ModelConfig.get_default_config()
        config.feature_dim = 1536
        config.activation = "gelu"
        config.n_genes = self.tokenizer.ge_tokenizer.n_tokens
        config.n_tech = self.tokenizer.tech_tokenizer.n_tokens
        config.n_species = self.tokenizer.specie_tokenizer.n_tokens
        config.n_organs = self.tokenizer.organ_tokenizer.n_tokens
        config.backbone = "spatial_transformer"
        self.d_model = config.d_model

        self.model = STFM(config).to(device)
        self.model.load_state_dict(torch.load(model_weight_path, map_location=device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Gigapath tile encoder (frozen) turns our raw H&E patches into the
        # 1536-dim features STPath's image tokenizer expects — shares the
        # same loader as GigapathPatchEncoder (task #20), no separate
        # trainable projection here since STPath wants the raw feature.
        # Only actually used for raw-patch input (see _gigapath_features
        # below) — the real training pipeline passes precomputed features
        # (src/models/conditioning.py precompute_gigapath_features) and
        # never touches this.
        self.tile_encoder = _load_gigapath_tile_encoder()
        self.tile_encoder.eval()
        for p in self.tile_encoder.parameters():
            p.requires_grad_(False)

        self.proj = nn.Linear(self.d_model, hidden_dim)

        context_gene_ids, valid_gene_pos = self.tokenizer.ge_tokenizer.symbol2id(
            gene_names, return_valid_positions=True
        )
        self._valid_gene_pos = valid_gene_pos
        self.register_buffer("_context_gene_ids", torch.tensor(context_gene_ids, dtype=torch.long))

    def _gigapath_features(self, patches_or_features: torch.Tensor) -> torch.Tensor:
        # accepts EITHER raw patches [B, 3, H, W] float in [0,1] (encodes
        # from scratch — slow, uncached path) OR already-precomputed
        # Gigapath features [B, 1536] (the fast path the real training
        # pipeline uses, see precompute_gigapath_features) - same
        # dispatch-on-rank pattern as GigapathPatchEncoder.forward
        if patches_or_features.dim() == 4:
            return _gigapath_preprocess_and_encode(self.tile_encoder, patches_or_features)
        return patches_or_features

    @torch.no_grad()
    def forward(self, context_coords: torch.Tensor, context_expression: torch.Tensor,
                query_coords: torch.Tensor, context_images: torch.Tensor,
                query_images: torch.Tensor) -> torch.Tensor:
        """context_images/query_images: raw H&E patches [N, 3, H, W] float
        in [0,1] — required (STPath has no meaningful expression-only
        mode). Returns c [N_query, hidden_dim]."""
        n_context = context_coords.shape[0]
        n_query = query_coords.shape[0]
        device = context_coords.device

        coords = torch.cat([context_coords[:, :2], query_coords[:, :2]], dim=0)
        coords = coords.clone()
        coords[:, 0] -= coords[:, 0].min()
        coords[:, 1] -= coords[:, 1].min()
        coords = self._rescale_coords(coords)

        img_feats = torch.cat([
            self._gigapath_features(context_images), self._gigapath_features(query_images),
        ], dim=0)

        n_total = n_context + n_query
        ge_tokens = self.tokenizer.ge_tokenizer.mask_token.float().to(device).repeat(n_total, 1)
        expr = torch.log1p(context_expression)[:, self._valid_gene_pos]
        context_one_hot = self.tokenizer.ge_tokenizer.convert_gene_exp_to_one_hot_tensor(
            self.tokenizer.ge_tokenizer.n_tokens, expr, self._context_gene_ids.to(device)
        )
        ge_tokens[:n_context] = context_one_hot  # real expression for context, mask token for query

        organ = self.tokenizer.organ_tokenizer.encode(self.organ_type, align_first=True)
        organ_ids = torch.full((n_total,), organ, dtype=torch.long, device=device)
        tech = self.tokenizer.tech_tokenizer.encode(self.tech_type, align_first=True)
        tech_ids = torch.full((n_total,), tech, dtype=torch.long, device=device)

        _, x = self.model.prediction_head(
            img_tokens=img_feats,
            coords=coords,
            ge_tokens=ge_tokens,
            batch_idx=torch.zeros(n_total, dtype=torch.long, device=device),
            tech_tokens=tech_ids,
            organ_tokens=organ_ids,
            return_all=True,
        )
        return self.proj(x[n_context:])  # query positions only
