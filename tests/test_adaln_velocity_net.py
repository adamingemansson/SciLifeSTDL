"""
Regression tests for the AdaLN-residual velocity_net (2026-07-19) — see
docs/results_log.md's architecture audit. The default velocity_net (a
plain 2-hidden-layer feedforward MLP, no residual connections, context
injected via one-time input concatenation) is >100x smaller than the
context_encoder feeding it and structurally unlike every comparable
published flow-matching/diffusion architecture (DiT, SiT, SD3's MM-DiT),
which use residual blocks with conditioning re-injected at every layer
via AdaLN. velocity_net_type="adaln_residual" (registry.py's
_AdaLNVelocityNet/_AdaLNResidualBlock) is that fix, opt-in — default
"mlp" preserves the exact old architecture, byte-for-byte.

Run with: python -m tests.test_adaln_velocity_net
"""
import torch

from src.models.registry import FlowMatchingOT, _AdaLNResidualBlock, _AdaLNVelocityNet


def _ctx_query(n_context=15, n_query=6, n_genes=10, coord_dim=3):
    context = {"coords": torch.randn(n_context, coord_dim), "expression": torch.rand(n_context, n_genes)}
    query = {"coords": torch.randn(n_query, coord_dim)}
    return context, query


def test_default_velocity_net_type_is_plain_mlp_unchanged():
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=3)
    assert isinstance(model.velocity_net, torch.nn.Sequential), "default must stay the plain MLP — zero behavior change"
    print("[adaln_velocity_net] OK — default velocity_net_type='mlp' is architecturally unchanged")


def test_adaln_zero_block_is_identity_at_init():
    """The AdaLN-Zero trick: the modulation projection's weight AND bias
    are zero-initialized, so gate=0 at construction — the block must be
    an exact residual identity (block(x, cond) == x) before any training,
    regardless of what cond is. This is what makes a deep stack trainable
    from scratch without a separate warmup schedule."""
    torch.manual_seed(0)
    block = _AdaLNResidualBlock(dim=16, cond_dim=8)
    x = torch.randn(5, 16)
    cond = torch.randn(5, 8)
    out = block(x, cond)
    assert torch.equal(out, x), "AdaLN-Zero block must be an exact identity at initialization"
    print("[adaln_velocity_net] OK — AdaLN-Zero residual block is identity at init")


def test_adaln_velocity_net_runs_end_to_end_and_produces_finite_output():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3,
                            velocity_net_type="adaln_residual", velocity_net_n_layers=3)
    assert isinstance(model.velocity_net, _AdaLNVelocityNet)
    context, query = _ctx_query()
    out = model.sample(context, query)
    assert torch.isfinite(out["expression"]).all(), (
        "adaln_residual velocity_net must produce finite output on a real sample() call"
    )
    print("[adaln_velocity_net] OK — adaln_residual runs end-to-end via sample(), finite output")


def test_adaln_velocity_net_gradients_flow_through_training_step():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3,
                            velocity_net_type="adaln_residual", velocity_net_n_layers=2)
    model.log_dict = lambda *args, **kwargs: None  # no attached Trainer in this direct-call test
    context, query = _ctx_query()
    batch = {"context": context, "query": query, "target_expression": torch.rand(6, 10)}
    loss = model.training_step(batch, 0)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.velocity_net.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and any(g > 0 for g in grad_norms), (
        "adaln_residual velocity_net params must receive real gradients from training_step"
    )
    print("[adaln_velocity_net] OK — gradients flow through every adaln_residual param during training_step")


def test_adaln_velocity_net_accepts_the_same_concatenated_input_as_mlp():
    """_velocity/_edm_denoise call sites do torch.cat([z_t, t_embed, c])
    and call self.velocity_net(...) on the result unconditionally — the
    AdaLN net must accept that exact same single-tensor calling
    convention (splitting it back apart internally), so no call site
    needs to know or care which velocity_net_type is active."""
    torch.manual_seed(0)
    net = _AdaLNVelocityNet(latent_dim=4, time_embed_dim=6, cond_hidden_dim=10, hidden_dim=16, n_layers=2)
    x = torch.randn(3, 4 + 6 + 10)  # exactly what torch.cat([z_t, t_embed, c], dim=-1) produces
    out = net(x)
    assert out.shape == (3, 4)
    print("[adaln_velocity_net] OK — accepts the same concatenated-tensor calling convention as the plain MLP")


def test_unknown_velocity_net_type_raises_clearly():
    try:
        FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                        time_embed_dim=16, n_ode_steps=3, velocity_net_type="bogus")
        assert False, "must raise on an unrecognized velocity_net_type"
    except AssertionError as e:
        assert "velocity_net_type" in str(e)
    print("[adaln_velocity_net] OK — unknown velocity_net_type raises a clear error")


if __name__ == "__main__":
    test_default_velocity_net_type_is_plain_mlp_unchanged()
    test_adaln_zero_block_is_identity_at_init()
    test_adaln_velocity_net_runs_end_to_end_and_produces_finite_output()
    test_adaln_velocity_net_gradients_flow_through_training_step()
    test_adaln_velocity_net_accepts_the_same_concatenated_input_as_mlp()
    test_unknown_velocity_net_type_raises_clearly()
    print("\nAll AdaLN velocity_net tests passed.")
