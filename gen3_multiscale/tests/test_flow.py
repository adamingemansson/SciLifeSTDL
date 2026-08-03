"""Architecture 4's velocity network and flow-matching mechanics. Tests
directly exercise the handoff's own learning gate: "Architecture 4's
velocity output must depend on both time and conditioning, and multiple
samples must not be identical." """
import torch

from gen3_multiscale.models.flow import (
    VelocityNetwork, flow_matching_loss, sample_residual_coefficients, sinusoidal_time_embedding,
)


def test_sinusoidal_time_embedding_shape_and_distinguishes_times():
    a = sinusoidal_time_embedding(torch.tensor(0.0), dim=16)
    b = sinusoidal_time_embedding(torch.tensor(0.7), dim=16)
    assert a.shape == (16,)
    assert not torch.allclose(a, b)


def test_sinusoidal_time_embedding_rejects_non_scalar():
    try:
        sinusoidal_time_embedding(torch.zeros(3), dim=16)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "0-D" in str(exc)


def test_velocity_network_output_shape():
    net = VelocityNetwork(residual_rank=8, hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)
    n_query = 6
    noisy = torch.randn(n_query, 8)
    t = torch.tensor(0.3)
    coords = torch.randn(n_query, 2)
    conditioning = torch.randn(n_query, 32)
    out = net(noisy, t, coords, conditioning)
    assert out.shape == (n_query, 8)
    assert torch.isfinite(out).all()


def test_velocity_depends_on_time():
    torch.manual_seed(0)
    net = VelocityNetwork(residual_rank=8, hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)
    noisy = torch.randn(5, 8)
    coords = torch.randn(5, 2)
    conditioning = torch.randn(5, 32)
    out_t0 = net(noisy, torch.tensor(0.0), coords, conditioning)
    out_t1 = net(noisy, torch.tensor(1.0), coords, conditioning)
    assert not torch.allclose(out_t0, out_t1)


def test_velocity_depends_on_conditioning():
    torch.manual_seed(0)
    net = VelocityNetwork(residual_rank=8, hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)
    noisy = torch.randn(5, 8)
    coords = torch.randn(5, 2)
    out_a = net(noisy, torch.tensor(0.5), coords, torch.randn(5, 32))
    out_b = net(noisy, torch.tensor(0.5), coords, torch.randn(5, 32))
    assert not torch.allclose(out_a, out_b)


def test_flow_matching_loss_is_finite_and_gradients_flow_into_the_velocity_network():
    net = VelocityNetwork(residual_rank=8, hidden_dim=32, n_heads=4, n_blocks=2, dense_threshold=100)
    target_coefficients = torch.randn(6, 8)
    coords = torch.randn(6, 2)
    conditioning = torch.randn(6, 32)  # NOT requires_grad -- simulates the caller's stop-gradient
    loss = flow_matching_loss(net, target_coefficients, coords, conditioning)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.out_proj.weight.grad is not None
    assert torch.isfinite(net.out_proj.weight.grad).all()


def test_flow_matching_loss_does_not_require_conditioning_to_have_gradients():
    """The caller detaches conditioning_hidden before this call (see
    Architecture 4's stop-gradient contract) -- the loss function itself
    must not assume or require it to be part of the graph."""
    net = VelocityNetwork(residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1, dense_threshold=100)
    target_coefficients = torch.randn(3, 4)
    coords = torch.randn(3, 2)
    conditioning = torch.randn(3, 16).detach()
    loss = flow_matching_loss(net, target_coefficients, coords, conditioning)
    loss.backward()  # must not raise


def test_multiple_samples_are_not_identical():
    """Directly tests the handoff's "multiple samples must not be
    identical" gate."""
    net = VelocityNetwork(residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1, dense_threshold=100)
    coords = torch.randn(5, 2)
    conditioning = torch.randn(5, 16)
    samples = sample_residual_coefficients(net, n_query=5, query_coords=coords, conditioning_hidden=conditioning, n_samples=4, n_steps=5)
    assert samples.shape == (4, 5, 4)
    assert not torch.allclose(samples[0], samples[1])
    assert not torch.allclose(samples[2], samples[3])


def test_sampling_is_reproducible_with_a_fixed_generator():
    net = VelocityNetwork(residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1, dense_threshold=100)
    coords = torch.randn(5, 2)
    conditioning = torch.randn(5, 16)
    gen_a = torch.Generator().manual_seed(0)
    gen_b = torch.Generator().manual_seed(0)
    samples_a = sample_residual_coefficients(net, 5, coords, conditioning, n_samples=2, n_steps=4, generator=gen_a)
    samples_b = sample_residual_coefficients(net, 5, coords, conditioning, n_samples=2, n_steps=4, generator=gen_b)
    assert torch.allclose(samples_a, samples_b)


def test_vectorized_sampling_matches_independent_serial_fields():
    """Vectorizing the sample axis must not allow attention to mix fields."""
    torch.manual_seed(4)
    net = VelocityNetwork(
        residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1,
        dense_threshold=4, sparse_k=3,
    ).eval()
    coords = torch.randn(7, 2)
    conditioning = torch.randn(7, 16)
    n_samples, n_steps = 3, 4
    seed = 19

    vectorized = sample_residual_coefficients(
        net, 7, coords, conditioning, n_samples=n_samples, n_steps=n_steps,
        generator=torch.Generator().manual_seed(seed),
    )

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(n_samples, 7, 4, generator=generator)
    serial = []
    for sample in initial:
        x = sample
        t = torch.zeros(())
        for _ in range(n_steps):
            x = x + net(x, t, coords, conditioning) / n_steps
            t = t + 1.0 / n_steps
        serial.append(x)

    assert torch.allclose(vectorized, torch.stack(serial), atol=1e-5, rtol=1e-5)


def test_sampling_runs_in_eval_mode_and_restores_the_callers_training_mode():
    """Real bug caught while testing: dropout was active during sampling,
    which draws from the global torch RNG on every forward call and
    silently broke reproducibility even under a fixed `generator`. Fixed
    by forcing eval() during sampling and restoring the caller's original
    mode afterward -- verified here so it can't silently regress."""
    net = VelocityNetwork(residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1, dense_threshold=100)
    net.train()
    coords = torch.randn(4, 2)
    conditioning = torch.randn(4, 16)
    sample_residual_coefficients(net, 4, coords, conditioning, n_samples=2, n_steps=3)
    assert net.training is True  # restored, not left in eval()

    net.eval()
    sample_residual_coefficients(net, 4, coords, conditioning, n_samples=2, n_steps=3)
    assert net.training is False  # restored to its original (eval) state too


def test_sampling_rejects_nonpositive_steps_or_samples():
    net = VelocityNetwork(residual_rank=4, hidden_dim=16, n_heads=2, n_blocks=1, dense_threshold=100)
    coords = torch.randn(3, 2)
    conditioning = torch.randn(3, 16)
    try:
        sample_residual_coefficients(net, 3, coords, conditioning, n_samples=0, n_steps=5)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "n_samples" in str(exc)
    try:
        sample_residual_coefficients(net, 3, coords, conditioning, n_samples=2, n_steps=0)
        assert False, "expected a ValueError"
    except ValueError as exc:
        assert "n_steps" in str(exc)
