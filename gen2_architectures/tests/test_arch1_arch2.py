import torch

from gen2_architectures.models.arch1_gpt_baseline import Architecture1
from gen2_architectures.models.arch2_scfoundation import Architecture2
from gen2_architectures.models.local_neighborhood_transformer import nearest_context_neighbors


def _synthetic_context(n_context: int, feature_width: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "coords": torch.randn(n_context, 2, generator=g) * 500,
        "expression": torch.randn(n_context, feature_width, generator=g),
        "images": torch.randn(n_context, 1536, generator=g),
        "image_available": torch.ones(n_context, dtype=torch.bool),
    }


def test_nearest_context_neighbors_shape_and_correctness():
    context_xy = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [100.0, 100.0]])
    query_xy = torch.tensor([[1.0, 0.0]])
    idx = nearest_context_neighbors(context_xy, query_xy, k=2)
    assert idx.shape == (1, 2)
    assert set(idx[0].tolist()) == {0, 1}, "the two nearest points to (1,0) should be indices 0 and 1"


def test_nearest_context_neighbors_caps_at_available_context():
    context_xy = torch.randn(3, 2)
    query_xy = torch.randn(5, 2)
    idx = nearest_context_neighbors(context_xy, query_xy, k=100)
    assert idx.shape == (5, 3), "k should silently cap at the number of available context points"


def test_architecture1_forward_shape_and_gradient():
    n_genes = 50
    model = Architecture1(n_genes=n_genes, feat_dim=32, coord_dim=16, conf_dim=8,
                           hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20, decoder_hidden_dim=128)
    context = _synthetic_context(40, n_genes)
    query = {"coords": torch.randn(5, 2) * 500}
    pred = model(context, query)
    assert pred.shape == (5, n_genes)
    pred.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_architecture1_handles_context_smaller_than_max_neighbors():
    n_genes = 20
    model = Architecture1(n_genes=n_genes, feat_dim=16, coord_dim=8, conf_dim=4,
                           hidden_dim=32, n_layers=1, n_heads=2, max_neighbors=50, decoder_hidden_dim=64)
    context = _synthetic_context(5, n_genes)  # fewer than max_neighbors=50
    query = {"coords": torch.randn(3, 2) * 500}
    pred = model(context, query)
    assert pred.shape == (3, n_genes)


def test_architecture1_organ_tech_conditioning_changes_output():
    n_genes = 20
    torch.manual_seed(0)
    model = Architecture1(n_genes=n_genes, feat_dim=16, coord_dim=8, conf_dim=4,
                           hidden_dim=32, n_layers=1, n_heads=2, max_neighbors=20, decoder_hidden_dim=64,
                           organ_vocab=["Lung", "Kidney"], tech_vocab=["Visium"])
    context = _synthetic_context(20, n_genes)
    query = {"coords": torch.randn(3, 2) * 500}
    model.eval()
    context["organ"], context["tech"] = "Lung", "Visium"
    pred_lung = model(context, query)
    context["organ"] = "Kidney"
    pred_kidney = model(context, query)
    assert not torch.allclose(pred_lung, pred_kidney), "different organ conditioning should change the prediction"


def test_architecture1_unknown_organ_raises():
    n_genes = 20
    model = Architecture1(n_genes=n_genes, feat_dim=16, coord_dim=8, conf_dim=4,
                           hidden_dim=32, n_layers=1, n_heads=2, max_neighbors=20, decoder_hidden_dim=64,
                           organ_vocab=["Lung"], tech_vocab=["Visium"])
    context = _synthetic_context(20, n_genes)
    context["organ"], context["tech"] = "Brain", "Visium"
    query = {"coords": torch.randn(3, 2) * 500}
    try:
        model(context, query)
        assert False, "expected a KeyError for an organ outside the fixed vocabulary"
    except KeyError:
        pass


def test_architecture2_forward_shape_and_gradient_scfoundation_features():
    n_genes = 50
    scf_dim = 128
    model = Architecture2(n_genes=n_genes, scfoundation_dim=scf_dim, feat_dim=32, coord_dim=16, conf_dim=8,
                           hidden_dim=64, n_layers=2, n_heads=4, max_neighbors=20, decoder_hidden_dim=128)
    context = _synthetic_context(40, scf_dim)  # context["expression"] holds precomputed scFoundation features here
    query = {"coords": torch.randn(5, 2) * 500}
    pred = model(context, query)
    assert pred.shape == (5, n_genes)
    pred.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_architecture2_requires_scfoundation_dim():
    try:
        Architecture2(n_genes=20, scfoundation_dim=None)
        assert False, "expected a ValueError when scfoundation_dim is not provided"
    except ValueError:
        pass


def test_architecture1_and_architecture2_share_non_gene_encoder_parameter_count():
    """The whole point of the shared LocalNeighborhoodTransformer base is
    that Architectures 1 and 2 differ ONLY in their gene encoder -- verify
    that by construction rather than just by code inspection."""
    n_genes, scf_dim = 50, 128
    kwargs = dict(feat_dim=32, coord_dim=16, conf_dim=8, hidden_dim=64, n_layers=2, n_heads=4,
                  max_neighbors=20, decoder_hidden_dim=128)
    m1 = Architecture1(n_genes=n_genes, **kwargs)
    m2 = Architecture2(n_genes=n_genes, scfoundation_dim=scf_dim, **kwargs)
    n1 = sum(p.numel() for n, p in m1.named_parameters() if "gene_encoder" not in n)
    n2 = sum(p.numel() for n, p in m2.named_parameters() if "gene_encoder" not in n)
    assert n1 == n2, "everything except the gene encoder should have identical parameter counts"
