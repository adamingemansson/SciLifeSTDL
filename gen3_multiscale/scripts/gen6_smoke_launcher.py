#!/usr/bin/env python3
"""CPU-only one-step smoke for every package-independent Gen6 conditioner."""
from __future__ import annotations

import torch

from gen3_multiscale.scripts.gen4_smoke_launcher import (
    CONTEXT_DIM, GEX_DIM, IMAGE_DIM, N_GENES, _StubSlideEncoder,
    _synthetic_example, _with_wsi_context,
)
from gen3_multiscale.gen6.model_factory import build_gen6_model
from gen3_multiscale.models.losses import combined_reconstruction_loss


def _config(arm: str) -> dict:
    return {
        "model": {"arm": arm, "kind": "conditioner", "params": {
            "image_feature_dim": IMAGE_DIM, "gex_context_embedding_dim": CONTEXT_DIM,
            "hidden_dim": 32, "n_heads": 4, "n_blocks": 2,
            "dense_threshold": 100, "n_gex_inducing": 4,
            "harmonic_k_neighbors": 4, "global_slide_dim": 16,
            "regional_grid_size": 2, "fusion_heads": 4,
        }},
        "data": {"gex_feature_dim": GEX_DIM}, "training": {"seed": 0},
        "required_fingerprints": {},
    }


def smoke_arm(arm: str) -> dict:
    inputs, targets = _synthetic_example(seed=2, gex_context_dim=CONTEXT_DIM)
    slide_encoder = None
    slide_sha = None
    if arm in {"gen6d", "gen6e"}:
        slide_encoder = _StubSlideEncoder()
        slide_sha = slide_encoder.checkpoint_sha256
        inputs = _with_wsi_context(inputs)
    elif arm == "gen6j":
        inputs = _with_wsi_context(inputs, wsi_tile_feature_provenance="uni2")
    model = build_gen6_model(
        _config(arm), gene_names=[f"g{i}" for i in range(N_GENES)],
        gex_feature_dim=GEX_DIM, image_feature_dim=IMAGE_DIM,
        gex_context_embedding_dim=CONTEXT_DIM, slide_encoder=slide_encoder,
        gigapath_checkpoint_sha256=slide_sha, seed=0,
    )
    target = torch.as_tensor(targets.query_expression)
    output = model(inputs)
    losses = combined_reconstruction_loss(
        output["expression"], target, torch.as_tensor(inputs.query_coords),
        primary_mode="rmse_pcc", pcc_weight=0.1,
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    return {
        "arm": arm, "loss": float(losses["total"]),
        "shape": list(output["expression"].shape),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def main() -> None:
    for arm in ("gen6b", "gen6c", "gen6d", "gen6e", "gen6f", "gen6g", "gen6h", "gen6i", "gen6j"):
        print(smoke_arm(arm))
    print({"arm": "gen6a", "status": "requires real STPath package/checkpoint smoke"})
    print({"arms": ["gen6k", "gen6l"], "status": "require staged conditioner/generator artifacts"})


if __name__ == "__main__":
    main()
