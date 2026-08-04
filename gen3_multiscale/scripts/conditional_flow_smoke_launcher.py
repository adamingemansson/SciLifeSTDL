#!/usr/bin/env python3
"""CPU one-step smoke for the four matched MK conditional-flow arms."""
from __future__ import annotations

import json

import numpy as np
import torch

from gen3_multiscale.conditional_flow import (
    Architecture1ImageConditioner,
    ConditionalLatentFlow,
    FullImageExpressionInputs,
)


def _model(coupling: str) -> ConditionalLatentFlow:
    conditioner = Architecture1ImageConditioner(
        6, image_feature_dim=32, gex_feature_dim=8,
        hidden_dim=32, n_heads=4, n_blocks=1,
        dense_threshold=64, sparse_k=4, dropout=0.0,
    )
    return ConditionalLatentFlow(
        6, conditioner, coupling=coupling, latent_dim=8,
        autoencoder_hidden_dim=32, n_flow_blocks=1,
        n_flow_samples=3, n_ode_steps=2, ot_sinkhorn_iters=4,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(size=(21, 32)).astype(np.float32)
    coords = rng.normal(size=(21, 2)).astype(np.float32)
    target = torch.as_tensor(rng.normal(size=(21, 6)), dtype=torch.float32)
    task_i = FullImageExpressionInputs(
        sample_id="synthetic", image_features=image, coords=coords,
        image_available=np.ones(21, dtype=bool), query_mask=np.ones(21, dtype=bool),
    )
    query_mask = np.zeros(21, dtype=bool)
    query_mask[:8] = True
    observed_indices = np.flatnonzero(~query_mask)
    task_ii = FullImageExpressionInputs(
        sample_id="synthetic", image_features=image, coords=coords,
        image_available=np.ones(21, dtype=bool), query_mask=query_mask,
        observed_expression=target.numpy()[observed_indices].copy(),
        observed_expression_indices=observed_indices,
        expression_available=~query_mask,
    )
    for task, inputs, task_target in (
        ("he_to_st", task_i, target),
        ("he_plus_st_to_st", task_ii, target[query_mask]),
    ):
        for coupling in ("independent", "sinkhorn_ot"):
            torch.manual_seed(0)
            model = _model(coupling)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_losses(
                inputs, task_target, generator=torch.Generator().manual_seed(2),
            )
            losses["total"].backward()
            optimizer.step()
            prediction = model.sample_predictive_distribution(
                inputs, generator=torch.Generator().manual_seed(3),
            )
            print(json.dumps({
                "task": task,
                "coupling": coupling,
                "total_loss": float(losses["total"].detach()),
                "flow_loss": float(losses["flow_loss"].detach()),
                "predictive_mean_shape": list(prediction["predictive_mean"].shape),
                "predictive_samples_shape": list(prediction["predictive_samples"].shape),
            }, sort_keys=True))


if __name__ == "__main__":
    main()
