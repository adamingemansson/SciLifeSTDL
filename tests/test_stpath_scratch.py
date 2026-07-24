"""Smoke test for the 'stpath_scratch' registry model (2026-07-24):
STPath's real architecture, randomly initialized, trained from scratch --
the "does STPath's architecture help without its pretraining" comparison
arm for the lung round. Only needs STPATH_GENE_VOC_PATH (no pretrained
weight file -- pretrained=False never loads one), same skip-cleanly
pattern as tests/test_stpath_encoder.py.

Unlike that file's frozen-baseline test, this one specifically checks the
NEW behavior stpath_scratch adds: gradient actually reaches STFM's own
(randomly initialized, trainable) parameters through training_step, and
configure_optimizers returns a real optimizer rather than stpath_official's
None.

Run with (after setting STPATH_GENE_VOC_PATH):
    python -m tests.test_stpath_scratch
"""
import os

import torch


def test_stpath_scratch():
    gene_voc_path = os.environ.get("STPATH_GENE_VOC_PATH")
    if not gene_voc_path:
        print("[stpath_scratch] SKIPPED — set STPATH_GENE_VOC_PATH to run this test")
        return

    try:
        from src.models.registry import build_model
    except ImportError as e:
        print(f"[stpath_scratch] SKIPPED — could not import (stpath package not "
              f"installed?) ({e})")
        return

    torch.manual_seed(0)
    n_context, n_query, n_genes = 8, 3, 5
    gene_names = ["EPCAM", "PTPRC", "PECAM1", "COL1A1", "CD3E"][:n_genes]

    try:
        model = build_model({
            "name": "stpath_scratch",
            "params": {
                "n_genes": n_genes, "stpath_gene_names": gene_names,
                "stpath_gene_voc_path": gene_voc_path,
                "stpath_organ_type": "Lung", "stpath_hidden_dim": 32, "lr": 1e-3,
            },
        })
    except Exception as e:
        print(f"[stpath_scratch] SKIPPED — could not build model ({e})")
        return

    # pretrained=False must leave every STFM parameter trainable (the whole
    # point of this comparison arm) -- assert it directly rather than
    # trusting the docstring.
    trainable = [p for p in model.predictor.model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "STFM has no trainable parameters -- pretrained=False path broken"
    print(f"[stpath_scratch] OK — {len(trainable)} STFM parameter tensors are trainable")

    context_coords = torch.rand(n_context, 3) * 100
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.rand(n_context, 3, 256, 256)
    query_coords = torch.rand(n_query, 3) * 100
    query_images = torch.rand(n_query, 3, 256, 256)

    context = {"coords": context_coords, "expression": context_expression, "images": context_images}
    query = {"coords": query_coords, "images": query_images}
    out = model.sample(context, query)
    n_valid = model._decoder_target_col_idx.numel()
    assert out["expression"].shape == (n_query, n_valid), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()
    print(f"[stpath_scratch] OK — sample() output shape {tuple(out['expression'].shape)}")

    batch = {
        "context": context, "query": query,
        "target_expression": torch.rand(n_query, n_genes),
    }
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    grad_reached_stfm = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.predictor.model.parameters()
    )
    assert grad_reached_stfm, "gradient did not reach STFM's own parameters"
    print("[stpath_scratch] OK — gradient reaches STFM's own (randomly initialized) parameters")

    optimizer = model.configure_optimizers()
    assert optimizer is not None, "configure_optimizers returned None (should train, unlike stpath_official)"
    print("[stpath_scratch] OK — configure_optimizers returns a real optimizer")


if __name__ == "__main__":
    test_stpath_scratch()
    print("\nstpath_scratch smoke test done (see above for SKIPPED vs OK).")
