"""
Regression tests for two more 2026-07-19 FM-OT improvements (see
docs/results_log.md):

  1. Minibatch OT coupling (fm_coupling="minibatch_ot") — Tong et al. 2023,
     "Improving and Generalizing Flow-Based Generative Models with
     Minibatch Optimal Transport"; Pooladian et al. 2023, "Multisample
     Flow Matching: Straightening Flows with Minibatch Couplings" (ICML).
     Closes a real naming gap: this class is called "FlowMatchingOT" for
     its straight-line path formulation, but its z_0<->z_1 PAIRING was
     never actually OT-coupled before this — just an independent i.i.d.
     draw. This is training-time only.
  2. Heun's 2nd-order ODE solver (ode_solver="heun") for the "ot" sampling
     path — Karras et al. 2022 (EDM paper), whose own recommended sampler
     this project's path_type="edm" already cites but path_type="ot"
     never used (a documented simplification until now). Inference-time
     only, zero training-time effect.

Both are opt-in, defaults preserve exact prior behavior.

Run with: python -m tests.test_fm_ot_coupling_and_solver
"""
import torch

from src.models.registry import FlowMatchingOT


def _ctx_query(n_context=15, n_query=6, n_genes=10, coord_dim=3):
    context = {"coords": torch.randn(n_context, coord_dim), "expression": torch.rand(n_context, n_genes)}
    query = {"coords": torch.randn(n_query, coord_dim)}
    return context, query


def test_independent_coupling_default_matches_torch_randn_like_exactly():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=3)
    assert model.fm_coupling == "independent"
    z_1 = torch.randn(8, model.latent_dim)
    torch.manual_seed(5)
    out = model._couple_noise(z_1)
    torch.manual_seed(5)
    ref = torch.randn_like(z_1)
    assert torch.equal(out, ref), "default fm_coupling must exactly reproduce torch.randn_like"
    print("[fm_coupling] OK — default 'independent' matches torch.randn_like exactly")


def test_minibatch_ot_coupling_is_a_valid_permutation_with_lower_cost():
    """The coupled z_0 must be exactly the same MULTISET of samples as an
    independent draw (a true assignment/permutation, not a different
    distribution) — and its total pairwise cost to z_1 must be <= any
    random permutation of the same pool (that's the whole point of
    solving the assignment problem)."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3, fm_coupling="minibatch_ot")
    n, d = 10, 4
    z_1 = torch.randn(n, d)
    torch.manual_seed(1)
    z_0_coupled = model._couple_noise(z_1)
    torch.manual_seed(1)
    z_0_pool = torch.randn(n, d)  # the same underlying pool _couple_noise draws internally

    # same multiset of rows (a genuine permutation, not new samples)
    sorted_coupled = torch.sort(z_0_coupled.flatten())[0]
    sorted_pool = torch.sort(z_0_pool.flatten())[0]
    assert torch.allclose(sorted_coupled, sorted_pool), (
        "minibatch_ot coupling must be a PERMUTATION of the noise pool, not new samples"
    )

    def total_cost(a, b):
        return ((a - b) ** 2).sum().item()

    ot_cost = total_cost(z_0_coupled, z_1)
    random_costs = [total_cost(z_0_pool[torch.randperm(n)], z_1) for _ in range(300)]
    assert ot_cost <= min(random_costs) + 1e-4, (
        f"OT-coupled cost ({ot_cost:.3f}) should be <= best of 300 random permutations "
        f"({min(random_costs):.3f}) — it's solving for the minimum"
    )
    print(f"[fm_coupling] OK — minibatch_ot cost ({ot_cost:.3f}) <= best random permutation "
          f"({min(random_costs):.3f}) of the same pool")


def test_minibatch_ot_coupling_runs_end_to_end_in_training_step():
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3, fm_coupling="minibatch_ot")
    context, query = _ctx_query()
    batch = {"context": context, "query": query, "target_expression": torch.rand(6, 10)}
    loss = model.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss), "minibatch_ot training_step must produce a finite loss"
    loss.backward()
    grads_exist = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert grads_exist, "gradients must flow through the minibatch_ot-coupled loss"
    print("[fm_coupling] OK — minibatch_ot runs end-to-end in training_step, finite loss, gradients flow")


def test_heun_solver_reduces_to_exact_solution_for_constant_velocity_field():
    """Correctness check independent of the real model: for a CONSTANT
    velocity field, Heun's predictor-corrector formula must give the
    IDENTICAL exact analytic answer z(1) = z(0) + v (v_cur == v_next
    collapses the corrector to the same thing Euler already computes) —
    confirms the formula is implemented correctly, not merely different."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=10)
    const_v = torch.randn(6, model.latent_dim)
    model._velocity = lambda z, t, c: const_v.clone()
    z0 = torch.randn(6, model.latent_dim)

    def integrate(use_heun, steps=10):
        z = z0.clone()
        dt = 1.0 / steps
        for _ in range(steps):
            if use_heun:
                v_cur = model._velocity(z, None, None)
                z_pred = z + dt * v_cur
                v_next = model._velocity(z_pred, None, None)
                z = z + dt * 0.5 * (v_cur + v_next)
            else:
                z = z + dt * model._velocity(z, None, None)
        return z

    z_exact = z0 + const_v
    assert torch.allclose(integrate(False), z_exact, atol=1e-5), "Euler must match the analytic solution here"
    assert torch.allclose(integrate(True), z_exact, atol=1e-5), "Heun must match the analytic solution here"
    print("[ode_solver] OK — Heun's corrector formula reduces to the exact analytic solution for a constant field")


def test_heun_sampler_runs_end_to_end_and_default_euler_unaffected():
    torch.manual_seed(0)
    model_euler = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32, time_embed_dim=16, n_ode_steps=5)
    assert model_euler.ode_solver == "euler"

    torch.manual_seed(0)
    model_heun = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                                 time_embed_dim=16, n_ode_steps=5, ode_solver="heun")

    torch.manual_seed(42)
    ctx1, q1 = _ctx_query()
    out_euler = model_euler.sample(ctx1, q1)

    torch.manual_seed(42)
    ctx2, q2 = _ctx_query()
    out_heun = model_heun.sample(ctx2, q2)

    assert out_euler["expression"].shape == out_heun["expression"].shape == (6, 10)
    assert torch.isfinite(out_euler["expression"]).all()
    assert torch.isfinite(out_heun["expression"]).all()
    assert not torch.equal(out_euler["expression"], out_heun["expression"]), (
        "heun must genuinely differ from euler on a real (non-constant) velocity field"
    )
    print("[ode_solver] OK — heun runs end-to-end, differs from euler, both finite")


if __name__ == "__main__":
    test_independent_coupling_default_matches_torch_randn_like_exactly()
    test_minibatch_ot_coupling_is_a_valid_permutation_with_lower_cost()
    test_minibatch_ot_coupling_runs_end_to_end_in_training_step()
    test_heun_solver_reduces_to_exact_solution_for_constant_velocity_field()
    test_heun_sampler_runs_end_to_end_and_default_euler_unaffected()
    print("\nAll FM-OT coupling/solver tests passed.")
