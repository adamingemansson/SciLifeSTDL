"""
Unit test for _ResidualEncodeInputs (src/models/stpath_encoder.py, 2026-07-16
"Route B" GEX-encoder-bottleneck follow-up — residual gene-encoder patch
into STPath's real pretrained transformer, verified feasible via STPath's
actual source, stpath/model/model.py's EncodeInputs class).

Self-contained — no dependency on the real `stpath` package, unlike
tests/test_stpath_encoder.py. _ResidualEncodeInputs wraps ANY nn.Module,
so this verifies its two critical correctness properties directly:

1. Zero-init safety: at construction, forward() returns EXACTLY the
   wrapped original module's output — the residual contributes nothing
   until trained. Real bug this guards against: if residual_proj weren't
   zero-initialized, training would start already perturbing STPath's
   pretrained fusion with a random/untrained signal, discarding the
   "safe starting point" the whole design is meant to guarantee.

2. Gradient flow: after a backward pass, residual_proj's parameters
   receive real, nonzero gradients. Real bug this guards against: this
   project already hit the EXACT failure mode once before in this same
   file (STPathContextEncoder.forward's blanket @torch.no_grad() silently
   zeroing gradients for its only trainable params) — this test exists
   specifically so the same mistake, if reintroduced here, fails loudly
   instead of silently training a component that never actually learns.

Run with:
    python -m tests.test_stpath_residual_encoder
"""
import torch
import torch.nn as nn

from src.models.stpath_encoder import _ResidualEncodeInputs


class _FakeEncodeInputs(nn.Module):
    """Minimal stand-in for STPath's real EncodeInputs — same call
    signature (img_tokens, ge_tokens, tech_tokens, organ_tokens), returns
    a deterministic, gradient-free tensor (mimics being frozen/pretrained,
    same as the real thing after requires_grad_(False))."""

    def forward(self, img_tokens, ge_tokens, tech_tokens, organ_tokens):
        return img_tokens + ge_tokens  # arbitrary deterministic combination


def test_zero_init_safety():
    torch.manual_seed(0)
    n, d_model = 5, 16
    original = _FakeEncodeInputs()
    wrapped = _ResidualEncodeInputs(original, d_model)

    img_tokens = torch.randn(n, d_model)
    ge_tokens = torch.randn(n, d_model)
    extra_embed = torch.randn(n, d_model)  # new gene encoder's (nonzero) output

    wrapped.set_extra_embed(extra_embed)
    out = wrapped(img_tokens, ge_tokens, None, None)
    expected = original(img_tokens, ge_tokens, None, None)
    assert torch.allclose(out, expected), (
        "residual_proj must contribute exactly zero at initialization, "
        "regardless of extra_embed's value"
    )
    print("[zero_init_safety] OK — wrapped output matches unwrapped original exactly at init")


def test_gradient_flows_to_residual_proj():
    """Two backward passes, not one — a real, non-obvious property of
    zero-initialized layers used as residual connections (the same
    "zero convolution" pattern ControlNet uses): a zero-initialized
    Linear's gradient w.r.t. ITS WEIGHT is nonzero from the very first
    step (outer product of grad_output and input, unaffected by the
    weight's own value), but its gradient w.r.t. its INPUT is the weight
    matrix itself — so with weight still exactly zero, gradient
    flowing further upstream (e.g. to a new gene encoder feeding this
    residual) is also exactly zero on step one. That's correct, expected
    behavior, not a bug: residual_proj's own weights must move off zero
    first (via their own direct gradient) before gradient can propagate
    further back. Verified explicitly here so this non-obvious property
    is documented and doesn't get "fixed" into a real bug later."""
    torch.manual_seed(0)
    n, d_model = 5, 16
    original = _FakeEncodeInputs()
    wrapped = _ResidualEncodeInputs(original, d_model)
    opt = torch.optim.SGD(wrapped.parameters(), lr=1.0)

    img_tokens = torch.randn(n, d_model)
    ge_tokens = torch.randn(n, d_model)
    extra_embed_1 = torch.randn(n, d_model, requires_grad=True)

    # Step 1: residual_proj is still exactly zero-initialized.
    wrapped.set_extra_embed(extra_embed_1)
    out = wrapped(img_tokens, ge_tokens, None, None)
    out.sum().backward()

    assert wrapped.residual_proj.weight.grad is not None, (
        "residual_proj.weight received no gradient at all — the exact "
        "silent-no_grad failure mode this project already hit once "
        "(STPathContextEncoder's old blanket @torch.no_grad())"
    )
    assert torch.any(wrapped.residual_proj.weight.grad != 0), (
        "residual_proj.weight's gradient is all zeros — it would never "
        "move away from its zero-init, and the residual would stay "
        "permanently inactive regardless of how long training runs"
    )
    assert extra_embed_1.grad is not None and torch.all(extra_embed_1.grad == 0), (
        "expected exactly zero gradient reaching extra_embed on step 1 "
        "(residual_proj's weight is still zero) — got something else, "
        "meaning the zero-init property doesn't hold as expected"
    )

    # Move residual_proj off zero (one real optimizer step), THEN check
    # that gradient reaches further upstream too.
    opt.step()
    opt.zero_grad()
    extra_embed_2 = torch.randn(n, d_model, requires_grad=True)
    wrapped.set_extra_embed(extra_embed_2)
    out2 = wrapped(img_tokens, ge_tokens, None, None)
    out2.sum().backward()

    assert extra_embed_2.grad is not None and torch.any(extra_embed_2.grad != 0), (
        "gradient still isn't reaching extra_embed after residual_proj "
        "moved off zero — would mean a new gene encoder feeding this "
        "residual could never learn, at any point in training"
    )
    print("[gradient_flows_to_residual_proj] OK — residual_proj's own weights get "
          "real gradient from step 1; gradient reaches further upstream (to the new "
          "gene encoder) once residual_proj has moved off its zero-init")


def test_forward_without_set_extra_embed_raises():
    d_model = 8
    wrapped = _ResidualEncodeInputs(_FakeEncodeInputs(), d_model)
    raised = False
    try:
        wrapped(torch.randn(1, d_model), torch.randn(1, d_model), None, None)
    except AssertionError:
        raised = True
    assert raised, "forward() without set_extra_embed() first must raise, not silently misuse None"
    print("[forward_without_set_extra_embed_raises] OK")


if __name__ == "__main__":
    test_zero_init_safety()
    test_gradient_flows_to_residual_proj()
    test_forward_without_set_extra_embed_raises()
    print("\nAll _ResidualEncodeInputs tests passed.")
