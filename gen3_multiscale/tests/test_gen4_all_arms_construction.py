"""Integration-10 (real-path integration tests): construct every one of
Gen4's four INTENDED arms (gen4b/c/d/e -- gen4a is the baseline, see
gen4/model_factory.py's own INTENDED_ARM_NAMES) through the real,
shared `training/train.py::build_model_for_inference` entry point --
never gen4.trainer_adapter directly, never a hand-built model, matching
exactly what the trainer/evaluator/overfit-gate themselves call.

Integration audit finding #3 (CONFIRMED real, fixed): gen4b/c/e all
failed BEFORE this fix with "requires gex_context_embedding_dim" --
purely a config-plumbing bug, not a missing real dependency. Only
gen4c has no real external encoder as a live submodule (its
scFoundation contribution is a precomputed embedding + a DIMENSION, not
a constructed encoder; its UNI2 global-context pool is a plain,
self-contained nn.Module needing no real UNI2 weights) and so
constructs successfully with NO real external package/checkpoint.
gen4b (real GigaPath slide encoder) and gen4d/gen4e (real STPath
encoder) genuinely cannot construct without the real `gigapath`
package/`timm` package and real checkpoints -- neither is installed in
this sandbox (GEN4_CONTRACT.md section 13's documented, unavoidable
gap) -- so this test proves those three arms correctly FAIL CLOSED on
the real-encoder-required guard specifically (never on the
gex_context_embedding_dim bug that predated this fix, and never by
silently constructing from a stub)."""
from __future__ import annotations

import pathlib

import pytest
import torch
import yaml

from gen3_multiscale.training.train import build_model_for_inference

_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "configs" / "gen4"

N_GENES = 5
GENE_NAMES = [f"g{i}" for i in range(N_GENES)]

_TINY_PARAMS = dict(
    n_genes=N_GENES, gex_feature_dim=4, image_feature_dim=8, hidden_dim=16, n_heads=2, n_blocks=1,
    dense_threshold=100, n_gex_inducing=4, harmonic_k_neighbors=4, global_slide_dim=8,
    gex_context_embedding_dim=6,
)


def _tiny_conditioner_config(arm: str):
    config = yaml.safe_load((_CONFIG_DIR / f"{arm}_conditioner.yaml").read_text())
    config["model"]["params"].update(_TINY_PARAMS)
    return config


def test_gen4c_arm1_uni2_scfoundation_constructs_with_no_real_external_package():
    from gen3_multiscale.gen4.conditioner import Gen4Conditioner

    config = _tiny_conditioner_config("gen4c")
    model, info = build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True)
    assert isinstance(model, Gen4Conditioner)
    assert info["kind"] == "conditioner"


@pytest.mark.parametrize("arm,missing_encoder", [("gen4b", "slide_encoder"), ("gen4d", "stpath_encoder"), ("gen4e", "stpath_encoder")])
def test_intended_arms_needing_real_external_weights_fail_closed_on_the_right_guard(arm, missing_encoder):
    """These three arms genuinely cannot construct in this sandbox (no
    real gigapath/timm package or checkpoints) -- the real, correct
    outcome is a fail-closed error naming the SPECIFIC missing real
    encoder, never the gex_context_embedding_dim bug this fix closed."""
    config = _tiny_conditioner_config(arm)
    with pytest.raises(ValueError, match=missing_encoder):
        build_model_for_inference(config, gene_names=GENE_NAMES, device=torch.device("cpu"), smoke=True)
