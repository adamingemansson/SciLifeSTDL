#!/usr/bin/env python3
"""CPU one-step smoke for the matched conditional WAE variants."""
from __future__ import annotations

import json

import numpy as np
import torch

from gen3_multiscale.conditional_wae import (
    Architecture1ImageConditioner,
    ConditionalWAE,
    FullImageExpressionInputs,
)


def _model(regularizer: str) -> ConditionalWAE:
    conditioner = Architecture1ImageConditioner(
        6, image_feature_dim=32, gex_feature_dim=8,
        hidden_dim=32, n_heads=4, n_blocks=1,
        dense_threshold=64, sparse_k=4, dropout=0.0,
    )
    return ConditionalWAE(
        6, conditioner, regularizer=regularizer, latent_dim=8,
        autoencoder_hidden_dim=32, discriminator_hidden_dim=16,
        n_inference_samples=3,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    image = rng.normal(size=(21, 32)).astype(np.float32)
    coords = rng.normal(size=(21, 2)).astype(np.float32)
    target = torch.as_tensor(rng.normal(size=(21, 6)), dtype=torch.float32)
    task_i = FullImageExpressionInputs(
        sample_id="synthetic",
        image_features=image,
        coords=coords,
        image_available=np.ones(21, dtype=bool),
        query_mask=np.ones(21, dtype=bool),
    )
    query_mask = np.zeros(21, dtype=bool)
    query_mask[:8] = True
    observed_expression_indices = np.flatnonzero(~query_mask)
    observed_expression = target.numpy()[observed_expression_indices].copy()
    task_ii = FullImageExpressionInputs(
        sample_id="synthetic",
        image_features=image,
        coords=coords,
        image_available=np.ones(21, dtype=bool),
        query_mask=query_mask,
        observed_expression=observed_expression,
        observed_expression_indices=observed_expression_indices,
        expression_available=~query_mask,
    )
    for task_name, inputs, task_target in (
        ("he_to_st", task_i, target),
        ("he_plus_st_to_st", task_ii, target[query_mask]),
    ):
        for regularizer in ("mmd", "gan"):
            torch.manual_seed(0)
            model = _model(regularizer)
            generator_parameters = [
                parameter for name, parameter in model.named_parameters()
                if not name.startswith("discriminator.")
            ]
            generator_optimizer = torch.optim.AdamW(generator_parameters, lr=1e-3)
            if regularizer == "gan":
                discriminator_optimizer = torch.optim.AdamW(model.discriminator.parameters(), lr=1e-3)
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss = model.compute_discriminator_loss(
                    task_target, generator=torch.Generator().manual_seed(1),
                )
                discriminator_loss.backward()
                discriminator_optimizer.step()
            else:
                discriminator_loss = None
            generator_optimizer.zero_grad(set_to_none=True)
            losses = model.compute_generator_losses(
                inputs, task_target, generator=torch.Generator().manual_seed(2),
            )
            losses["total"].backward()
            generator_optimizer.step()
            prediction = model.sample_predictive_distribution(
                inputs, generator=torch.Generator().manual_seed(3),
            )
            print(json.dumps({
                "task": task_name,
                "regularizer": regularizer,
                "generator_loss": float(losses["total"].detach()),
                "prior_loss": float(losses["prior_loss"].detach()),
                "discriminator_loss": (
                    float(discriminator_loss.detach()) if discriminator_loss is not None else None
                ),
                "predictive_mean_shape": list(prediction["predictive_mean"].shape),
                "predictive_samples_shape": list(prediction["predictive_samples"].shape),
            }, sort_keys=True))


if __name__ == "__main__":
    main()
