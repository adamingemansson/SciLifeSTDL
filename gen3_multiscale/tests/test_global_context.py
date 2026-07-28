"""Phase 5 items 7-8 (multiscale spatial-field handoff): the global
observed-GEX inducing pool (16 hidden/value candidate pairs) and
global-slide FiLM conditioning."""
import torch

from gen3_multiscale.models.global_context import GlobalConditioningFiLM, InducedGlobalGEXPool


class TestInducedGlobalGEXPool:
    def test_output_shapes(self):
        pool = InducedGlobalGEXPool(hidden_dim=32, n_inducing=16, n_heads=4)
        n_observed, n_genes = 50, 10
        out = pool(torch.randn(n_observed, 32), torch.randn(n_observed, n_genes))
        assert out["hidden"].shape == (16, 32)
        assert out["expression"].shape == (16, n_genes)
        assert out["value_weights"].shape == (16, n_observed)

    def test_value_weights_are_convex_over_observed_spots(self):
        pool = InducedGlobalGEXPool(hidden_dim=32, n_inducing=8, n_heads=4)
        out = pool(torch.randn(20, 32), torch.randn(20, 5))
        w = out["value_weights"]
        assert torch.isfinite(w).all()
        assert (w >= 0).all()
        assert torch.allclose(w.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_untouched_observed_values_are_what_gets_mixed(self):
        pool = InducedGlobalGEXPool(hidden_dim=32, n_inducing=8, n_heads=4)
        observed_hidden = torch.randn(20, 32)
        expr_a = torch.randn(20, 5)
        expr_b = expr_a + 10.0  # a different, unambiguous real-expression matrix
        out_a = pool(observed_hidden, expr_a)
        out_b = pool(observed_hidden, expr_b)
        assert not torch.allclose(out_a["expression"], out_b["expression"])
        # a uniform expression matrix must be reproduced exactly by a convex mixture
        uniform = torch.full((20, 5), 2.5)
        out_uniform = pool(observed_hidden, uniform)
        assert torch.allclose(out_uniform["expression"], torch.full((8, 5), 2.5), atol=1e-4)

    def test_query_spots_cannot_be_passed_in_the_first_place(self):
        """Enforces exclusion structurally: forward()'s signature only
        accepts observed_hidden/observed_expression."""
        import inspect
        params = list(inspect.signature(InducedGlobalGEXPool.forward).parameters)
        assert params == ["self", "observed_hidden", "observed_expression"]

    def test_gradients_flow(self):
        pool = InducedGlobalGEXPool(hidden_dim=32, n_inducing=8, n_heads=4)
        observed_hidden = torch.randn(20, 32, requires_grad=True)
        out = pool(observed_hidden, torch.randn(20, 5))
        out["hidden"].sum().backward()
        assert observed_hidden.grad is not None and torch.isfinite(observed_hidden.grad).all()

    def test_rejects_empty_observed_set(self):
        pool = InducedGlobalGEXPool(hidden_dim=16, n_inducing=4, n_heads=2)
        try:
            pool(torch.zeros(0, 16), torch.zeros(0, 3))
            assert False, "expected a ValueError"
        except ValueError as exc:
            assert "empty" in str(exc)


class TestGlobalConditioningFiLM:
    def test_is_the_identity_function_at_initialization(self):
        """Zero-initialized scale/shift -- a fresh module must not alter
        token_hidden at all, so any later dependence a model shows on the
        global vector is something training actually learned."""
        film = GlobalConditioningFiLM(hidden_dim=16, global_dim=8)
        token_hidden = torch.randn(5, 16)
        global_vector = torch.randn(8)
        out = film(token_hidden, global_vector)
        assert torch.allclose(out, token_hidden)

    def test_output_changes_once_weights_are_perturbed(self):
        film = GlobalConditioningFiLM(hidden_dim=16, global_dim=8)
        with torch.no_grad():
            film.scale_proj.weight.add_(0.1)
            film.shift_proj.weight.add_(0.1)
        token_hidden = torch.randn(5, 16)
        global_vector = torch.randn(8)
        out = film(token_hidden, global_vector)
        assert not torch.allclose(out, token_hidden)

    def test_different_global_vectors_give_different_outputs_after_training(self):
        film = GlobalConditioningFiLM(hidden_dim=16, global_dim=8)
        with torch.no_grad():
            film.scale_proj.weight.add_(0.1)
            film.shift_proj.weight.add_(0.1)
        token_hidden = torch.randn(5, 16)
        out_a = film(token_hidden, torch.randn(8))
        out_b = film(token_hidden, torch.randn(8))
        assert not torch.allclose(out_a, out_b)

    def test_rejects_a_batched_global_vector(self):
        film = GlobalConditioningFiLM(hidden_dim=16, global_dim=8)
        try:
            film(torch.randn(5, 16), torch.randn(2, 8))
            assert False, "expected a ValueError"
        except ValueError as exc:
            assert "1-D" in str(exc)
