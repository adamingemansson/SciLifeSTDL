"""Compatibility shim so gen2_architectures' deterministic regression
models can be evaluated through the copied audit_evaluation.py harness.

The original codebase's evaluation (evaluate_model_on_mask_bank ->
predictive_samples, gen2_architectures/training/validation.py) was built
for genuinely STOCHASTIC generative models (FM-OT and friends), where
drawing the model n_samples times and averaging IS the predictive mean,
and the spread across draws IS the model's own uncertainty estimate. Every
gen2 architecture is a plain deterministic regressor instead (GPT's
suggestion #5, predicting a real (mu, sigma) with an NLL loss, was
explicitly deferred out of v1 scope per the curated-subset decision) — so
model.sample(context, query) here just returns the SAME deterministic
prediction every call. This is an honest reflection of "this model has no
predictive uncertainty mechanism," not a bug: reusing the shared harness
still gives correct PCC/RMSE/nonzero_auc/ST-FID/pcc_raw_log1p (the metrics
that actually matter for comparing these 4 architectures against each
other and against the notebook), while interval90_coverage/predictive_std
will trivially read ~1.0/~0.0 for every gen2 config — expected, not
informative, and should be read accordingly rather than compared against
the old stochastic models' values for those two specific metrics.
"""
from __future__ import annotations

import torch.nn as nn


class DeterministicSampleMixin:
    """Mix into any nn.Module whose forward(context, query) returns the
    predicted expression tensor directly."""

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, context: dict, query: dict) -> dict:
        return {"expression": self.forward(context, query)}


class DeterministicLatentSampleMixin:
    """Mix into any nn.Module whose forward(context, query) returns a dict
    containing a "predicted_expression" key (Architecture 3 Stage B)."""

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(self, context: dict, query: dict) -> dict:
        return {"expression": self.forward(context, query)["predicted_expression"]}
