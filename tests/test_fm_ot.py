"""
Smoke test for FM-OT (flow matching, optimal-transport path) in
src/models/registry.py. Tiny synthetic data, no real dataset or Lightning
Trainer needed — calls sample() and training_step() directly to check
shapes/gradients/no-NaN.

Run with: python -m tests.test_fm_ot
"""
import torch

from src.models.registry import FlowMatchingOT
from src.training.validation import predictive_samples


def _make_batch(n_context, n_query, n_genes, coord_dim):
    context = {
        "coords": torch.randn(n_context, coord_dim),
        "expression": torch.rand(n_context, n_genes),
    }
    query = {"coords": torch.randn(n_query, coord_dim)}
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


def test_sample():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=50, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5)
    batch = _make_batch(n_context=100, n_query=20, n_genes=50, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (20, 50), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()
    print(f"[sample] OK — output shape {tuple(out['expression'].shape)}")


def test_predictive_sampling_reuses_flow_conditioning_without_changing_draws():
    torch.manual_seed(0)
    model = FlowMatchingOT(
        n_genes=12, coord_dim=3, cond_hidden_dim=16,
        hidden_dim=32, time_embed_dim=8, n_ode_steps=2,
    ).eval()
    batch = _make_batch(n_context=10, n_query=4, n_genes=12, coord_dim=3)

    # Reproduce predictive_samples' old one-shot loop under the same isolated
    # RNG seed. The cached path must be numerically identical in eval mode.
    with torch.random.fork_rng():
        torch.manual_seed(123)
        expected = torch.stack([
            model.sample(batch["context"], batch["query"])["expression"]
            for _ in range(3)
        ])

    encode_calls = 0
    original_encode = model._encode_context

    def counted_encode(context, query):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(context, query)

    model._encode_context = counted_encode
    actual = predictive_samples(
        model, batch["context"], batch["query"], n_samples=3, seed=123
    )

    assert encode_calls == 1
    assert torch.equal(actual, expected)


def test_training_step_updates_weights():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=50, coord_dim=2, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5)
    opt = model.configure_optimizers()
    model.optimizers = lambda: opt
    model.log_dict = lambda *args, **kwargs: None

    batch = _make_batch(n_context=60, n_query=15, n_genes=50, coord_dim=2)
    before = model.velocity_net[0].weight.clone()

    before_enc = model.encoder[0].weight.clone()

    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    after = model.velocity_net[0].weight
    after_enc = model.encoder[0].weight
    assert not torch.allclose(before, after), "velocity_net weights did not change — optimizer step had no effect"
    assert not torch.allclose(before_enc, after_enc), "encoder weights did not change — optimizer step had no effect"
    assert torch.isfinite(after).all(), "velocity_net weights contain NaN/Inf after one training step"
    print("[training_step] OK — velocity_net + encoder weights updated, no NaNs")


def test_sample_edm():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=50, coord_dim=3, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            path_type="edm")
    batch = _make_batch(n_context=100, n_query=20, n_genes=50, coord_dim=3)

    out = model.sample(batch["context"], batch["query"])
    assert out["expression"].shape == (20, 50), out["expression"].shape
    assert torch.isfinite(out["expression"]).all()
    print(f"[sample_edm] OK — output shape {tuple(out['expression'].shape)}")


def test_training_step_updates_weights_edm():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=50, coord_dim=2, cond_hidden_dim=32,
                            hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                            path_type="edm")
    opt = model.configure_optimizers()
    model.log_dict = lambda *args, **kwargs: None

    batch = _make_batch(n_context=60, n_query=15, n_genes=50, coord_dim=2)
    before = model.velocity_net[0].weight.clone()
    before_enc = model.encoder[0].weight.clone()

    loss = model.training_step(batch, batch_idx=0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    after = model.velocity_net[0].weight
    after_enc = model.encoder[0].weight
    assert not torch.allclose(before, after), "velocity_net weights did not change — optimizer step had no effect"
    assert not torch.allclose(before_enc, after_enc), "encoder weights did not change — optimizer step had no effect"
    assert torch.isfinite(after).all(), "velocity_net weights contain NaN/Inf after one training step"
    print("[training_step_edm] OK — velocity_net + encoder weights updated, no NaNs")


if __name__ == "__main__":
    test_sample()
    test_training_step_updates_weights()
    test_sample_edm()
    test_training_step_updates_weights_edm()
    print("\nAll FM-OT smoke tests passed.")
