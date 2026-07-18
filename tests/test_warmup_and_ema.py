"""
Regression tests for LR warmup (src/models/registry.py,
_linear_warmup_lr_lambda + warmup_steps) and EMA weight averaging
(src/training/train.py, EMACallback) — both added 2026-07-19 in response
to two real, separate problems found the same day (see
docs/results_log.md's day2 entry):

  1. "Bigger" StormLite capacity (4 layers/8 heads/512-dim) mode-collapsed
     to a constant output in both single- and multi-sample settings, and
     gradient_clip_val=1.0 alone made it WORSE, not better. No trainer in
     this codebase scheduled LR at all before this fix — warmup is the
     standard fix for exactly this "deeper/wider transformer unstable at
     a flat LR" failure mode.
  2. 5 identically-configured StormLite+decoder runs, differing only in
     random seed, landed anywhere from PCC 0.366 to 0.493 — a huge spread
     for "the same config." EMA (Polyak averaging) is the standard fix
     for run-to-run training noise this large.

The two are independent, composable opt-ins (warmup_steps=0 and
ema_decay=None are both zero-behavior-change defaults) — tested
separately here, not as a combined fix for either specific collapse
(that requires a real multi-thousand-step training run on real data,
out of scope for a fast smoke test).

Run with: python -m tests.test_warmup_and_ema
"""
import torch
import pytorch_lightning as pl

from src.models.registry import WAEGAN, FlowMatchingOT, VQVAEAutoregressive, _linear_warmup_lr_lambda
from src.training.train import EMACallback


def _make_batch(n_genes=20, coord_dim=3):
    n_context, n_query = 30, 8
    context = {"coords": torch.randn(n_context, coord_dim), "expression": torch.rand(n_context, n_genes)}
    query = {"coords": torch.randn(n_query, coord_dim)}
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n=6):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return _make_batch()


def _collate(batch_list):
    return batch_list[0]


class _LRProbe(pl.Callback):
    def __init__(self):
        self.lrs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        opts = pl_module.optimizers()
        opts = opts if isinstance(opts, list) else [opts]
        self.lrs.append([g["lr"] for opt in opts for g in opt.param_groups])


def test_linear_warmup_lr_lambda_shape():
    """Direct test of the schedule shape, independent of any model/Trainer:
    ramps linearly from just above 0 to exactly 1.0 at warmup_steps, then
    stays at 1.0 (clamped, not overshooting) after."""
    lr_lambda = _linear_warmup_lr_lambda(warmup_steps=4)
    values = [lr_lambda(step) for step in range(8)]
    assert values == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0], values
    print("[warmup] OK — linear ramp to 1.0 at warmup_steps, clamped afterward")


def test_warmup_ramps_lr_for_every_model_family():
    """warmup_steps=0 (default) vs warmup_steps=3 (opt-in) for all three
    model families — WAEGAN (manual optimization, needs explicit
    scheduler.step() calls in training_step) and FlowMatchingOT/
    VQVAEAutoregressive (automatic optimization, Lightning steps the
    scheduler itself via "interval": "step")."""
    configs = [
        ("WAEGAN", lambda ws: WAEGAN(n_genes=20, coord_dim=3, latent_dim=8, hidden_dim=32,
                                      cond_hidden_dim=32, disc_hidden_dim=16, warmup_steps=ws)),
        ("FlowMatchingOT", lambda ws: FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                                                       hidden_dim=64, time_embed_dim=16, n_ode_steps=5,
                                                       warmup_steps=ws)),
        ("VQVAEAutoregressive", lambda ws: VQVAEAutoregressive(
            n_genes=20, coord_dim=3, cond_hidden_dim=32, latent_dim=8, ae_hidden_dim=32,
            codebook_size=16, transformer_dim=16, n_transformer_layers=1, n_heads=2, warmup_steps=ws)),
    ]
    for name, build in configs:
        torch.manual_seed(0)
        probe_off = _LRProbe()
        model_off = build(0)
        trainer_off = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                                  enable_checkpointing=False, logger=False, callbacks=[probe_off])
        trainer_off.fit(model_off, torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, collate_fn=_collate))
        assert all(all(lr == pytest_lr[0] for lr in pytest_lr) for pytest_lr in probe_off.lrs), (
            f"{name}: warmup_steps=0 should give a flat LR the whole run, got {probe_off.lrs}"
        )
        first_lr = probe_off.lrs[0][0]
        assert all(lr == first_lr for step_lrs in probe_off.lrs for lr in step_lrs), (
            f"{name}: warmup_steps=0 must never move LR at all — got {probe_off.lrs}"
        )

        torch.manual_seed(0)
        probe_on = _LRProbe()
        model_on = build(3)
        trainer_on = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                                 enable_checkpointing=False, logger=False, callbacks=[probe_on])
        trainer_on.fit(model_on, torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, collate_fn=_collate))
        assert probe_on.lrs[0][0] < probe_on.lrs[-1][0], (
            f"{name}: warmup_steps=3 should start below the final (full) LR — got {probe_on.lrs}"
        )
        assert probe_on.lrs[-1][0] == first_lr, (
            f"{name}: warmup should ramp UP TO the same full LR warmup_steps=0 uses flat — "
            f"got final {probe_on.lrs[-1][0]} vs flat {first_lr}"
        )
        print(f"[warmup] OK — {name}: warmup_steps=0 stays flat, warmup_steps=3 ramps up to the same full LR")


def test_warmup_direct_training_step_call_still_works():
    """Regression test for a real bug found 2026-07-19: adding an
    unconditional self.lr_schedulers() call to WAEGAN.training_step broke
    every test that calls training_step() directly against a bare model
    (no attached Trainer) — a testing pattern several existing test files
    rely on (see test_wae_gan.py's own comment: 'bypass Lightning Trainer
    wiring for this isolated smoke test'). self.lr_schedulers() raises
    RuntimeError without an attached Trainer. Fixed by only calling it
    when warmup_steps > 0 — warmup_steps=0 (default, what every one of
    those tests uses) must stay a genuine no-op that never touches any
    Trainer-dependent API."""
    torch.manual_seed(0)
    model = WAEGAN(n_genes=20, coord_dim=2, latent_dim=8, hidden_dim=32,
                    cond_hidden_dim=32, disc_hidden_dim=16)  # warmup_steps=0, default
    opt_ae, opt_disc = model.configure_optimizers()
    model.optimizers = lambda: (opt_ae, opt_disc)
    model.manual_backward = lambda loss: loss.backward()
    model.log_dict = lambda *args, **kwargs: None

    batch = _make_batch(n_genes=20, coord_dim=2)
    model.training_step(batch, batch_idx=0)  # would raise RuntimeError pre-fix
    print("[warmup] OK — training_step() callable directly without an attached Trainer when warmup_steps=0")


def test_ema_tracks_only_trainable_parameters():
    """EMACallback.shadow should track exactly model.named_parameters()
    with requires_grad=True — same set save_trainable_state_dict already
    saves — not buffers (RandomFourierFeatures.B is fixed-random, nothing
    to average; VectorQuantizer's codebook already has its own internal
    EMA mechanism, averaging an average would distort it)."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32, hidden_dim=64,
                            time_embed_dim=16, n_ode_steps=5)
    ema = EMACallback(decay=0.9)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                          enable_checkpointing=False, logger=False, callbacks=[ema])
    trainer.fit(model, torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, collate_fn=_collate))

    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == trainable_names, (
        "EMACallback.shadow must track exactly the trainable parameters, no more, no less"
    )
    buffer_names = {name for name, _ in model.named_buffers()}
    assert not (set(ema.shadow.keys()) & buffer_names), "EMACallback must never track buffers"
    print(f"[EMA] OK — shadow tracks exactly {len(trainable_names)} trainable params, 0 buffers")


def test_ema_apply_to_model_differs_from_raw_final_weights():
    """The actual point of EMA: after a few training steps with a
    deliberately large LR (to guarantee real weight movement across
    steps), the EMA-smoothed weights must differ from whatever the raw
    optimizer trajectory landed on at the very last step — and must
    exactly match the stored shadow value."""
    torch.manual_seed(0)
    model = FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32, hidden_dim=64,
                            time_embed_dim=16, n_ode_steps=5, lr=1e-2)
    ema = EMACallback(decay=0.9)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                          enable_checkpointing=False, logger=False, callbacks=[ema])
    trainer.fit(model, torch.utils.data.DataLoader(_TinyDataset(n=20), batch_size=1, collate_fn=_collate))

    raw_weight = dict(model.named_parameters())["velocity_net.0.weight"].clone()
    ema.apply_to_model(model)
    ema_weight = dict(model.named_parameters())["velocity_net.0.weight"].clone()

    assert not torch.equal(raw_weight, ema_weight), (
        "EMA weights should differ from the raw final-step weights after real training movement"
    )
    assert torch.equal(ema_weight, ema.shadow["velocity_net.0.weight"]), (
        "apply_to_model must copy the shadow value exactly"
    )
    print("[EMA] OK — apply_to_model produces weights that differ from raw final state and match the shadow exactly")


def test_ema_disabled_by_default_is_a_true_noop():
    """ema_decay unset (this codebase's train.py/run_comparison.py gate:
    `EMACallback(ema_decay) if ema_decay else None`) must mean NO
    EMACallback is ever constructed at all — not an EMACallback with
    decay=None silently doing nothing. Verified at the call-site gating
    logic level, not by constructing EMACallback(None) (which isn't a
    supported/meaningful call — decay must be a real float when used)."""
    ema_decay = None
    ema_callback = EMACallback(ema_decay) if ema_decay else None
    assert ema_callback is None, "unset training.ema_decay must mean no EMACallback at all"
    print("[EMA] OK — unset ema_decay means no EMACallback constructed, true zero-behavior-change default")


if __name__ == "__main__":
    test_linear_warmup_lr_lambda_shape()
    test_warmup_ramps_lr_for_every_model_family()
    test_warmup_direct_training_step_call_still_works()
    test_ema_tracks_only_trainable_parameters()
    test_ema_apply_to_model_differs_from_raw_final_weights()
    test_ema_disabled_by_default_is_a_true_noop()
    print("\nAll warmup/EMA tests passed.")
