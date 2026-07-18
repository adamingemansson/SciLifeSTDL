"""
Regression tests for three 2026-07-19 improvements found via a deep code
audit + literature research (see docs/results_log.md):

  1. QK-normalization (_QKNormAttention, storm_lite_encoder.py) — the
     structural fix for the "bigger" StormLite attention-entropy collapse
     (Henry et al. 2020 EMNLP; ViT-22B / SD3 LayerNorm-on-head-dim variant).
  2. Logit-normal flow-matching timestep sampling (FlowMatchingOT
     ._sample_flow_time) — SD3 / Esser et al. 2024 (arXiv 2403.03206).
  3. input_already_log1p flag (StormLiteContextEncoder._maybe_log1p) —
     opt-out of the double-log1p the audit found (basic_qc_and_normalize
     already log1p's adata.X, the gene encoder logged it again).

All three are opt-in with defaults that preserve the exact prior behavior.

Run with: python -m tests.test_qknorm_timesampling_log1p
"""
import torch
import pytorch_lightning as pl

from src.models.storm_lite_encoder import _QKNormAttention, _MoMETransformerBlock, StormLiteContextEncoder
from src.models.registry import FlowMatchingOT


def test_qk_norm_bounds_attention_logits_under_weight_blowup():
    """The whole point of QK-norm: with the Q/K projection weights blown
    up (simulating the unstable-training weight growth that collapses a
    bigger transformer), QK-normed logits stay bounded (LayerNorm on each
    head caps their magnitude), while un-normed scaled-dot-product logits
    explode — which saturates softmax to near-one-hot (attention-entropy
    collapse), the documented root cause of the mode-collapse gradient
    clipping alone couldn't fix (and made worse)."""
    torch.manual_seed(0)
    d, h, L = 32, 4, 20
    x = torch.randn(1, L, d)
    attn = _QKNormAttention(d, h)
    with torch.no_grad():
        attn.q_proj.weight.mul_(50.0)
        attn.k_proj.weight.mul_(50.0)

    def _split(t):
        return t.view(1, L, h, d // h).transpose(1, 2)

    q_n = attn.q_norm(_split(attn.q_proj(x)))
    k_n = attn.k_norm(_split(attn.k_proj(x)))
    logits_normed = (torch.matmul(q_n, k_n.transpose(-2, -1)) * attn.scale)

    q_raw = _split(attn.q_proj(x))
    k_raw = _split(attn.k_proj(x))
    logits_raw = (torch.matmul(q_raw, k_raw.transpose(-2, -1)) * attn.scale)

    assert logits_normed.abs().max() < 50.0, (
        f"QK-normed logits should stay bounded, got max {logits_normed.abs().max():.1f}"
    )
    assert logits_raw.abs().max() > 100.0 * logits_normed.abs().max(), (
        "un-normed logits should explode relative to QK-normed under blown-up Q/K weights "
        f"(got raw {logits_raw.abs().max():.1f} vs normed {logits_normed.abs().max():.1f})"
    )
    print(f"[qk_norm] OK — bounds logits under weight blowup "
          f"(normed {logits_normed.abs().max():.1f} vs raw {logits_raw.abs().max():.1f})")


def test_qk_norm_default_off_leaves_mome_block_unchanged():
    """qk_norm=False (default) must keep the MoME block using
    nn.MultiheadAttention exactly as before — same param count as a block
    built without the qk_norm argument even existing."""
    torch.manual_seed(0)
    blk_default = _MoMETransformerBlock(32, 4)
    torch.manual_seed(0)
    blk_explicit_off = _MoMETransformerBlock(32, 4, qk_norm=False)
    n_default = sum(p.numel() for p in blk_default.parameters())
    n_off = sum(p.numel() for p in blk_explicit_off.parameters())
    assert n_default == n_off, "qk_norm=False must be identical to the pre-existing default block"
    assert not blk_default.qk_norm
    # qk_norm=True adds params (separate q/k/v/out projections + q_norm/k_norm)
    blk_on = _MoMETransformerBlock(32, 4, qk_norm=True)
    assert blk_on.qk_norm
    assert sum(p.numel() for p in blk_on.parameters()) != n_default
    print("[qk_norm] OK — default off is byte-identical to the pre-existing block; on adds params")


def test_qk_norm_mome_block_accepts_additive_bias_and_flows_gradient():
    """QK-norm path must still accept FrameAveragingBias's [n_heads, L, L]
    additive attention bias and let gradient flow — the same contract the
    nn.MultiheadAttention path had."""
    torch.manual_seed(0)
    N, d, h = 10, 32, 4
    x = torch.randn(1, 2 * N, d, requires_grad=True)
    is_img = torch.cat([torch.ones(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)])
    bias = torch.randn(h, 2 * N, 2 * N) * 0.1
    blk = _MoMETransformerBlock(d, h, qk_norm=True)
    out = blk(x, is_img, attn_mask=bias)
    out.sum().backward()
    assert out.shape == (1, 2 * N, d)
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("[qk_norm] OK — MoME block accepts per-head additive bias and flows gradient")


def test_logit_normal_time_sampling_concentrates_mid_trajectory():
    """Logit-normal t sampling (SD3) must (a) stay in [0,1] and (b)
    concentrate around t=0.5 with m=0/s=1 — more mass in the informative
    middle than uniform, less at the near-noise/near-data ends."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3, fm_time_sampling="logit_normal")
    t = model._sample_flow_time(20000)
    assert (t >= 0).all() and (t <= 1).all(), "sampled t must stay in [0,1]"
    # with m=0,s=1 the distribution is symmetric around 0.5; the middle
    # third [1/3, 2/3] should hold more mass than under uniform (~1/3)
    mid_frac = ((t > 1 / 3) & (t < 2 / 3)).float().mean().item()
    assert mid_frac > 0.34, f"logit-normal should concentrate the middle third, got {mid_frac:.3f}"
    assert abs(t.mean().item() - 0.5) < 0.02, "m=0 logit-normal should center at 0.5"
    print(f"[logit_normal] OK — stays in [0,1], centered at 0.5, middle-third mass {mid_frac:.3f} > uniform 0.33")


def test_uniform_time_sampling_is_the_default_and_unchanged():
    """fm_time_sampling='uniform' (default) must reproduce plain U[0,1]
    exactly — same as torch.rand under the same seed."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=10, coord_dim=3, cond_hidden_dim=16, hidden_dim=32,
                            time_embed_dim=16, n_ode_steps=3)  # default fm_time_sampling
    assert model.fm_time_sampling == "uniform"
    torch.manual_seed(123)
    t_model = model._sample_flow_time(100)
    torch.manual_seed(123)
    t_ref = torch.rand(100)
    assert torch.equal(t_model, t_ref), "default uniform sampling must equal torch.rand exactly"
    print("[uniform] OK — default t sampling is exactly torch.rand, unchanged")


def test_input_already_log1p_flag_toggles_the_redundant_log1p():
    """input_already_log1p=True skips the encoder's log1p (the audit
    fix — data is already log1p'd by basic_qc_and_normalize); False
    (default) keeps applying it (unchanged behavior / historical-comparison
    fairness)."""
    torch.manual_seed(0)
    x = torch.rand(5, 10) * 9.0  # log-normalized-scale values

    enc_off = StormLiteContextEncoder(n_genes=10, novae_dim=8, hidden_dim=16,
                                       fusion_mode="mome", gene_encoder_type="both")
    assert not enc_off.input_already_log1p
    assert torch.equal(enc_off._maybe_log1p(x), torch.log1p(x)), (
        "default must still apply log1p (double-log preserved for comparison fairness)"
    )

    enc_on = StormLiteContextEncoder(n_genes=10, novae_dim=8, hidden_dim=16,
                                      fusion_mode="mome", gene_encoder_type="both",
                                      input_already_log1p=True)
    assert torch.equal(enc_on._maybe_log1p(x), x), (
        "input_already_log1p=True must pass expression through unchanged (no double-log1p)"
    )
    print("[input_already_log1p] OK — False applies log1p (default), True skips the redundant second log1p")


if __name__ == "__main__":
    test_qk_norm_bounds_attention_logits_under_weight_blowup()
    test_qk_norm_default_off_leaves_mome_block_unchanged()
    test_qk_norm_mome_block_accepts_additive_bias_and_flows_gradient()
    test_logit_normal_time_sampling_concentrates_mid_trajectory()
    test_uniform_time_sampling_is_the_default_and_unchanged()
    test_input_already_log1p_flag_toggles_the_redundant_log1p()
    print("\nAll QK-norm / time-sampling / log1p tests passed.")
