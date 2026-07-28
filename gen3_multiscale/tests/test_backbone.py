"""Phase 6: the shared MultiscaleBlock/SpatialFieldBackbone all four
architecture wrappers assemble from via feature flags -- never four
copy-pasted implementations."""
import torch

from gen3_multiscale.models.backbone import MultiscaleBlock, SpatialFieldBackbone


def _local_boundary_inputs(n_query=6, n_local=8, n_boundary=15, hidden_dim=32):
    query_hidden = torch.randn(n_query, hidden_dim)
    query_coords = torch.randn(n_query, 2)
    # Local candidates are PER-QUERY (each query has its own local_k
    # nearest observed neighbors) -- [n_query, n_local, hidden_dim].
    local_hidden = torch.randn(n_query, n_local, hidden_dim)
    local_geometry = torch.randn(n_query, n_local, 3)
    # Boundary is a SHARED context every query in the item attends to --
    # [n_boundary, hidden_dim] (2D, not per-query), with per-query
    # relative geometry to those same shared spots.
    boundary_hidden = torch.randn(n_boundary, hidden_dim)
    boundary_geometry = torch.randn(n_query, n_boundary, 3)
    return query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry


def test_minimal_block_forward_shape_local_and_boundary_only():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100)
    args = _local_boundary_inputs(hidden_dim=32)
    out = block(*args)
    assert out.shape == args[0].shape
    assert torch.isfinite(out).all()


def test_full_block_with_regional_global_gex_and_global_slide():
    block = MultiscaleBlock(
        hidden_dim=32, n_heads=4, dense_threshold=100,
        use_regional_he=True, use_global_gex=True, use_global_slide=True, global_slide_dim=8,
    )
    query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry = _local_boundary_inputs(hidden_dim=32)
    out = block(
        query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry,
        regional_hidden=torch.randn(16, 32), regional_geometry=torch.randn(6, 16, 3),
        gex_inducing_hidden=torch.randn(16, 32), gex_inducing_geometry=torch.randn(6, 16, 3),
        global_slide_vector=torch.randn(8),
    )
    assert out.shape == query_hidden.shape
    assert torch.isfinite(out).all()
    assert block.n_branches == 4


def test_missing_regional_inputs_raises_when_enabled():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100, use_regional_he=True)
    args = _local_boundary_inputs(hidden_dim=32)
    try:
        block(*args)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "use_regional_he" in str(exc)


def test_missing_global_gex_inputs_raises_when_enabled():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100, use_global_gex=True)
    args = _local_boundary_inputs(hidden_dim=32)
    try:
        block(*args)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "use_global_gex" in str(exc)


def test_missing_global_slide_vector_raises_when_enabled():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100, use_global_slide=True, global_slide_dim=8)
    args = _local_boundary_inputs(hidden_dim=32)
    try:
        block(*args)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "use_global_slide" in str(exc)


def test_gradients_flow_through_local_and_boundary_branches():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100)
    query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry = _local_boundary_inputs(hidden_dim=32)
    query_hidden.requires_grad_(True)
    local_hidden.requires_grad_(True)
    boundary_hidden.requires_grad_(True)
    out = block(query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry)
    out.sum().backward()
    assert query_hidden.grad is not None and torch.isfinite(query_hidden.grad).all()
    assert local_hidden.grad is not None and torch.isfinite(local_hidden.grad).all()
    assert boundary_hidden.grad is not None and torch.isfinite(boundary_hidden.grad).all()


def test_backbone_stacks_blocks_and_changes_the_representation():
    torch.manual_seed(0)
    backbone_shallow = SpatialFieldBackbone(n_blocks=1, hidden_dim=32, n_heads=4, dense_threshold=100)
    torch.manual_seed(0)
    backbone_deep = SpatialFieldBackbone(n_blocks=4, hidden_dim=32, n_heads=4, dense_threshold=100)

    query_hidden, query_coords, local_hidden, local_geometry, boundary_hidden, boundary_geometry = _local_boundary_inputs(hidden_dim=32)
    out_shallow = backbone_shallow(
        query_hidden, query_coords, local_hidden=local_hidden, local_geometry=local_geometry,
        boundary_hidden=boundary_hidden, boundary_geometry=boundary_geometry,
    )
    out_deep = backbone_deep(
        query_hidden, query_coords, local_hidden=local_hidden, local_geometry=local_geometry,
        boundary_hidden=boundary_hidden, boundary_geometry=boundary_geometry,
    )
    assert out_shallow.shape == out_deep.shape == query_hidden.shape
    assert not torch.allclose(out_shallow, out_deep)


def test_same_seed_and_config_gives_identical_backbone_parameters():
    """"Confirm common state-dict modules initialize identically across
    arms for the same seed" (Phase 6 item 4) -- two backbones built with
    identical feature-flag configuration and the same seed must have
    byte-identical parameters, so a config difference (not accidental
    nondeterminism) is the only thing that can make two architecture arms
    diverge at initialization."""
    torch.manual_seed(42)
    backbone_a = SpatialFieldBackbone(n_blocks=2, hidden_dim=32, n_heads=4, dense_threshold=100)
    torch.manual_seed(42)
    backbone_b = SpatialFieldBackbone(n_blocks=2, hidden_dim=32, n_heads=4, dense_threshold=100)

    for (name_a, p_a), (name_b, p_b) in zip(backbone_a.named_parameters(), backbone_b.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), f"parameter {name_a} differs despite identical seed and config"


def test_branch_gate_weights_are_convex():
    block = MultiscaleBlock(hidden_dim=32, n_heads=4, dense_threshold=100)
    normed = torch.randn(5, 32)
    gate_weights = torch.softmax(block.branch_gate(normed), dim=-1)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (gate_weights >= 0).all()
