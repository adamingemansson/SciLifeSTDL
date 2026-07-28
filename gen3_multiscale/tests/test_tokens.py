"""Phase 5 items 1-2 (multiscale spatial-field handoff): shared token
modules. Tests the "concatenate then project, never sum unrelated
modalities before normalization" fusion contract, the query token's lack
of any target-carrying field, and basic shape/gradient sanity."""
import torch

from gen3_multiscale.models.tokens import FourierCoordinateEncoding, QueryTokenProjection, SpotTokenProjection


def test_fourier_coordinate_encoding_output_shape_and_distinguishes_positions():
    enc = FourierCoordinateEncoding(output_dim=64, num_frequencies=8)
    coords = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, -5.0]])
    out = enc(coords)
    assert out.shape == (3, 64)
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_fourier_coordinate_encoding_rejects_wrong_last_dim():
    enc = FourierCoordinateEncoding(output_dim=16)
    try:
        enc(torch.zeros(3, 3))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "coords" in str(exc)


def test_spot_token_projection_produces_hidden_dim_tokens():
    proj = SpotTokenProjection(hidden_dim=512, image_feature_dim=1536, gex_feature_dim=256)
    n = 10
    image_features = torch.randn(n, 1536)
    gex_features = torch.randn(n, 256)
    coords = torch.randn(n, 2)
    boundary_ring = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 0])
    modality_flags = torch.ones(n, 1)
    tokens = proj(image_features, gex_features, coords, boundary_ring, modality_flags)
    assert tokens.shape == (n, 512)
    assert torch.isfinite(tokens).all()


def test_spot_token_projection_rejects_out_of_range_boundary_ring():
    proj = SpotTokenProjection(hidden_dim=64, image_feature_dim=8, gex_feature_dim=4)
    n = 3
    try:
        proj(
            torch.randn(n, 8), torch.randn(n, 4), torch.randn(n, 2),
            torch.tensor([0, 1, 99]), torch.ones(n, 1),
        )
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "boundary_ring" in str(exc)


def test_spot_token_projection_boundary_ring_changes_the_token():
    """Different ring identity for otherwise-identical spots must produce
    a different token -- proves ring identity is actually used, not
    silently dropped by the concatenation/projection."""
    proj = SpotTokenProjection(hidden_dim=64, image_feature_dim=8, gex_feature_dim=4)
    image_features = torch.zeros(1, 8)
    gex_features = torch.zeros(1, 4)
    coords = torch.zeros(1, 2)
    modality_flags = torch.ones(1, 1)
    tok_ring0 = proj(image_features, gex_features, coords, torch.tensor([0]), modality_flags)
    tok_ring1 = proj(image_features, gex_features, coords, torch.tensor([1]), modality_flags)
    assert not torch.allclose(tok_ring0, tok_ring1)


def test_spot_token_projection_gradients_flow_through_every_modality_branch():
    proj = SpotTokenProjection(hidden_dim=64, image_feature_dim=8, gex_feature_dim=4)
    n = 3
    image_features = torch.randn(n, 8, requires_grad=True)
    gex_features = torch.randn(n, 4, requires_grad=True)
    coords = torch.randn(n, 2, requires_grad=True)
    modality_flags = torch.ones(n, 1, requires_grad=True)
    tokens = proj(image_features, gex_features, coords, torch.zeros(n, dtype=torch.long), modality_flags)
    tokens.sum().backward()
    for name, tensor in (
        ("image_features", image_features), ("gex_features", gex_features),
        ("coords", coords), ("modality_flags", modality_flags),
    ):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), f"{name} got no gradient"


def test_query_token_projection_produces_hidden_dim_tokens():
    proj = QueryTokenProjection(hidden_dim=512, use_hole_geometry=True)
    n_query = 7
    coords = torch.randn(n_query, 2)
    depth = torch.tensor([0, 1, 1, 2, 3, 3, 5])
    hole_geometry = torch.tensor([[12.5, 0.8]])  # [1, 2] -- one hole shared by every query in this item
    tokens = proj(coords, depth, hole_geometry)
    assert tokens.shape == (n_query, 512)


def test_query_token_projection_depth_changes_the_token():
    proj = QueryTokenProjection(hidden_dim=64, use_hole_geometry=False)
    coords = torch.zeros(1, 2)
    tok_shallow = proj(coords, torch.tensor([0]))
    tok_deep = proj(coords, torch.tensor([5]))
    assert not torch.allclose(tok_shallow, tok_deep)


def test_query_token_projection_has_no_field_that_could_carry_a_target():
    """QueryTokenProjection.forward's signature is (coords,
    depth_to_boundary, hole_geometry) -- there is no argument through
    which target GEX or target H&E could ever pass."""
    import inspect
    params = list(inspect.signature(QueryTokenProjection.forward).parameters)
    assert "target" not in " ".join(params).lower()
    assert "expression" not in " ".join(params).lower()
    assert "image" not in " ".join(params).lower()


def test_query_token_projection_rejects_negative_depth():
    proj = QueryTokenProjection(hidden_dim=32, use_hole_geometry=False)
    try:
        proj(torch.zeros(1, 2), torch.tensor([-1]))
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "non-negative" in str(exc)
