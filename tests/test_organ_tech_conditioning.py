"""
Smoke tests for organ/technology conditioning (2026-07-16,
src/models/conditioning.py OrganTechEmbedding/build_organ_tech_vocab, plus
the SpatialContextEncoder/StormLiteContextEncoder forward-path wiring) —
item 1 of the "multi-sample training + organ/tech conditioning" follow-up
(see docstrings on those classes for the real caveat: only meaningful once
training spans genuinely different organs/platforms; currently-available
INT1-INT24 samples are all-Visium/same-cohort, so this is infrastructure
built ahead of real multi-organ data, not yet validated against it).

No external dependency needed — uses precomputed feature tensors, same
pattern as test_storm_lite_encoder.py.

Run with:
    python -m tests.test_organ_tech_conditioning
"""
import torch

from src.models.conditioning import (
    OrganTechEmbedding, build_organ_tech_vocab, SpatialContextEncoder,
    _GIGAPATH_FEAT_DIM,
)
from src.models.storm_lite_encoder import StormLiteContextEncoder


def test_build_organ_tech_vocab():
    organs = ["Kidney", "Lung", "Kidney", "Brain"]
    techs = ["Visium", "Visium", "Xenium", "Visium"]
    organ_vocab, tech_vocab = build_organ_tech_vocab(organs, techs)
    assert organ_vocab == sorted(set(organs)), organ_vocab
    assert tech_vocab == sorted(set(techs)), tech_vocab
    # deterministic regardless of input order
    organ_vocab2, tech_vocab2 = build_organ_tech_vocab(list(reversed(organs)), list(reversed(techs)))
    assert organ_vocab2 == organ_vocab and tech_vocab2 == tech_vocab
    print(f"[build_organ_tech_vocab] OK — organ_vocab={organ_vocab}, tech_vocab={tech_vocab}")


def test_organ_tech_embedding():
    torch.manual_seed(0)
    organ_vocab, tech_vocab = ["Kidney", "Lung"], ["Visium", "Xenium"]
    embed = OrganTechEmbedding(organ_vocab, tech_vocab, hidden_dim=8)

    out = embed("Kidney", "Visium", n=5, device="cpu")
    assert out.shape == (5, 8), out.shape
    # broadcast — every row identical
    assert torch.allclose(out, out[0].expand_as(out))

    # different (organ, tech) pairs must produce different embeddings —
    # real check this isn't accidentally a no-op / constant lookup
    out2 = embed("Lung", "Xenium", n=5, device="cpu")
    assert not torch.allclose(out, out2), "different organ/tech produced identical embedding"

    # out-of-vocab lookups must raise loudly, not silently default
    for bad_call in [lambda: embed("Liver", "Visium", 1, "cpu"),
                      lambda: embed("Kidney", "MERFISH", 1, "cpu")]:
        raised = False
        try:
            bad_call()
        except KeyError:
            raised = True
        assert raised, "out-of-vocab organ/tech must raise KeyError, not default silently"

    print("[OrganTechEmbedding] OK — broadcasts correctly, organ/tech both contribute, "
          "out-of-vocab raises")


def test_spatial_context_encoder_organ_tech():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 10, 4, 12, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_expression = torch.rand(n_context, n_genes)
    organ_vocab, tech_vocab = ["Kidney", "Lung"], ["Visium"]

    encoder = SpatialContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="raw",
        organ_vocab=organ_vocab, tech_vocab=tech_vocab,
    )
    # without organ/tech: identical to the pre-conditioning behavior
    c_no_cond = encoder(context_coords, context_expression, query_coords)
    c_with_cond = encoder(context_coords, context_expression, query_coords,
                           organ="Kidney", tech="Visium")
    assert c_no_cond.shape == c_with_cond.shape == (n_query, hidden_dim)
    assert torch.isfinite(c_with_cond).all()
    assert not torch.allclose(c_no_cond, c_with_cond), (
        "organ/tech conditioning had no effect on output — may be a no-op"
    )

    # different organ -> different output, same everything else
    c_other_organ = encoder(context_coords, context_expression, query_coords,
                             organ="Lung", tech="Visium")
    assert not torch.allclose(c_with_cond, c_other_organ), (
        "changing organ produced identical output"
    )

    # gradient-flow check: OrganTechEmbedding's params must receive real
    # gradient (same failure class this project has hit before with
    # silently-frozen new components — see stpath_encoder.py's docstring)
    loss = c_with_cond.sum()
    loss.backward()
    organ_tech_params = dict(encoder.organ_tech_embed.named_parameters())
    assert organ_tech_params, "no params found on organ_tech_embed"
    no_grad = [name for name, p in organ_tech_params.items()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not no_grad, f"organ_tech_embed params with no/invalid gradient: {no_grad}"

    # constructing without organ_vocab/tech_vocab: organ_tech_embed is
    # None, and passing organ/tech at forward time is safely ignored
    # (backward-compatible — every pre-2026-07-16 config keeps working)
    plain_encoder = SpatialContextEncoder(n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="raw")
    assert plain_encoder.organ_tech_embed is None
    c_plain = plain_encoder(context_coords, context_expression, query_coords,
                             organ="Kidney", tech="Visium")
    assert c_plain.shape == (n_query, hidden_dim)
    assert torch.isfinite(c_plain).all()

    print("[SpatialContextEncoder organ/tech] OK — conditioning changes output, "
          "gradient flows, no-vocab construction stays a safe no-op")


def test_storm_lite_context_encoder_organ_tech():
    torch.manual_seed(0)
    n_context, n_query, n_genes, hidden_dim = 8, 3, 10, 16
    context_coords = torch.rand(n_context, 3) * 100
    query_coords = torch.rand(n_query, 3) * 100
    context_images = torch.rand(n_context, _GIGAPATH_FEAT_DIM)
    query_images = torch.rand(n_query, _GIGAPATH_FEAT_DIM)
    context_expression = torch.rand(n_context, n_genes)
    organ_vocab, tech_vocab = ["Kidney", "Lung"], ["Visium"]

    encoder = StormLiteContextEncoder(
        n_genes=n_genes, hidden_dim=hidden_dim, gene_encoder_type="mlp",
        organ_vocab=organ_vocab, tech_vocab=tech_vocab,
    )
    c_no_cond = encoder(context_coords, context_expression, query_coords,
                         context_images, query_images)
    c_with_cond = encoder(context_coords, context_expression, query_coords,
                           context_images, query_images, organ="Kidney", tech="Visium")
    assert c_no_cond.shape == c_with_cond.shape == (n_query, hidden_dim)
    assert torch.isfinite(c_with_cond).all()
    assert not torch.allclose(c_no_cond, c_with_cond), (
        "organ/tech conditioning had no effect on StormLiteContextEncoder output"
    )

    loss = c_with_cond.sum()
    loss.backward()
    no_grad = [name for name, p in encoder.organ_tech_embed.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not no_grad, f"organ_tech_embed params with no/invalid gradient: {no_grad}"
    print("[StormLiteContextEncoder organ/tech] OK — conditioning changes output, gradient flows")


if __name__ == "__main__":
    test_build_organ_tech_vocab()
    test_organ_tech_embedding()
    test_spatial_context_encoder_organ_tech()
    test_storm_lite_context_encoder_organ_tech()
    print("\nAll organ/tech conditioning smoke tests done.")
