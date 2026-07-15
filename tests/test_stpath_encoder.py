"""
Smoke test for STPathContextEncoder (task #18, src/models/stpath_encoder.py).

Needs the external `stpath` package installed (git clone
Graph-and-Geometric-Learning/STPath + `pip install -e .`), the pretrained
weight downloaded from huggingface.co/tlhuang/STPath, the bundled
utils_data/symbol2ensembl.json gene vocabulary from that same clone, and
Gigapath access (task #20) — none of which are available in every
environment. Skips cleanly if the STPATH_MODEL_WEIGHT_PATH and
STPATH_GENE_VOC_PATH environment variables aren't set, or if any required
package/weight/access is missing, same pattern as the Gigapath case in
tests/test_conditioning.py.

Run with (after setting the two env vars):
    python -m tests.test_stpath_encoder
"""
import os

import torch


def test_stpath_context_encoder():
    gene_voc_path = os.environ.get("STPATH_GENE_VOC_PATH")
    model_weight_path = os.environ.get("STPATH_MODEL_WEIGHT_PATH")
    if not gene_voc_path or not model_weight_path:
        print("[stpath_encoder] SKIPPED — set STPATH_GENE_VOC_PATH and "
              "STPATH_MODEL_WEIGHT_PATH to run this test")
        return

    try:
        from src.models.stpath_encoder import STPathContextEncoder
    except ImportError as e:
        print(f"[stpath_encoder] SKIPPED — could not import (stpath package not "
              f"installed? pip install timm?) ({e})")
        return

    torch.manual_seed(0)
    n_context, n_query, n_genes, patch_size = 8, 3, 5, 256
    gene_names = ["EPCAM", "PTPRC", "PECAM1", "COL1A1", "CD3E"][:n_genes]

    try:
        encoder = STPathContextEncoder(
            gene_names=gene_names, gene_voc_path=gene_voc_path,
            model_weight_path=model_weight_path,
            organ_type="Kidney", tech_type="Visium", hidden_dim=32, device="cpu",
        )
    except Exception as e:
        print(f"[stpath_encoder] SKIPPED — could not load STPath ({e})")
        return

    context_coords = torch.rand(n_context, 3) * 100
    context_expression = torch.rand(n_context, n_genes)
    context_images = torch.rand(n_context, 3, patch_size, patch_size)
    query_coords = torch.rand(n_query, 3) * 100
    query_images = torch.rand(n_query, 3, patch_size, patch_size)

    c = encoder(context_coords, context_expression, query_coords,
                context_images=context_images, query_images=query_images)
    assert c.shape == (n_query, 32), c.shape
    assert torch.isfinite(c).all()
    print(f"[stpath_encoder] OK — output shape {tuple(c.shape)}")


if __name__ == "__main__":
    test_stpath_context_encoder()
    print("\nSTPath encoder smoke test done (see above for SKIPPED vs OK).")
