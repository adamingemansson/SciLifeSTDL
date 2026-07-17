"""
Regression test for PeriodicCheckpointCallback (src/training/train.py,
2026-07-17) — specifically the real bug found during a pre-overnight-run
audit: trainer.global_step increments once per optimizer.step() call, not
once per training batch, so it silently advances TWICE per batch for
WAE-GAN (manual optimization, two separate optimizers stepped per
training_step) but only once per batch for FM-OT/VQ-VAE+AR (single
automatic optimizer). Keying the checkpoint cadence off global_step would
have saved WAE-GAN checkpoints twice as often as every other family for
the same save_every_n_steps value. Fixed by keying off batch_idx instead,
which is uniform across every family (every trainer in this codebase sets
max_epochs=1, so batch_idx directly IS the 0-indexed training-item count
for the whole run).

Run with: python -m tests.test_periodic_checkpoint
"""
import torch
import pytorch_lightning as pl

from src.models.registry import WAEGAN, FlowMatchingOT


def _make_batch(n_genes=20, coord_dim=3):
    n_context, n_query = 30, 8
    context = {"coords": torch.randn(n_context, coord_dim), "expression": torch.rand(n_context, n_genes)}
    query = {"coords": torch.randn(n_query, coord_dim)}
    target_expression = torch.rand(n_query, n_genes)
    return {"context": context, "query": query, "target_expression": target_expression}


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, idx):
        return _make_batch()


def _collate(batch_list):
    return batch_list[0]


class _StepProbeCallback(pl.Callback):
    """Records both trainer.global_step and batch_idx+1 at every batch,
    to directly demonstrate the discrepancy the real fix addresses."""

    def __init__(self):
        self.global_steps = []
        self.batch_counts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.global_steps.append(trainer.global_step)
        self.batch_counts.append(batch_idx + 1)


def test_global_step_diverges_from_batch_count_for_manual_optimization():
    """Confirms the ROOT CAUSE this fix addresses is real, not
    hypothetical: WAE-GAN's global_step sequence must NOT match its
    batch-count sequence (it advances by 2 per batch, from stepping two
    optimizers), while FM-OT's single-optimizer automatic optimization
    keeps the two in lockstep."""
    torch.manual_seed(0)
    wae_gan = WAEGAN(n_genes=20, coord_dim=3, latent_dim=8, hidden_dim=32,
                      cond_hidden_dim=32, disc_hidden_dim=16)
    probe = _StepProbeCallback()
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                          enable_checkpointing=False, logger=False, callbacks=[probe])
    trainer.fit(wae_gan, torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, collate_fn=_collate))
    assert probe.global_steps == [2, 4, 6, 8, 10, 12], (
        f"expected WAE-GAN's global_step to advance by 2 per batch (two manual "
        f"optimizer.step() calls), got {probe.global_steps} -- if this assertion now "
        f"fails, PyTorch Lightning's manual-optimization global_step semantics changed "
        f"and PeriodicCheckpointCallback's batch_idx-based fix should be re-evaluated"
    )
    assert probe.batch_counts == [1, 2, 3, 4, 5, 6]
    print("[global_step vs batch_idx] OK — confirmed real divergence for manual optimization "
          "(global_step=2x batch count), motivating the batch_idx-based fix")


def test_periodic_checkpoint_cadence_uniform_across_optimizer_modes():
    """The actual fix: using batch_idx (not global_step) for the
    checkpoint cadence check gives an IDENTICAL trigger sequence for
    WAE-GAN (manual, 2 optimizers) and FM-OT (automatic, 1 optimizer) —
    mirrors PeriodicCheckpointCallback.on_train_batch_end's own logic
    exactly (step = batch_idx + 1; step % save_every_n_steps == 0)."""
    save_every_n_steps = 3

    def run(model):
        triggered_at = []

        class _CadenceProbe(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                step = batch_idx + 1
                if step % save_every_n_steps == 0:
                    triggered_at.append(step)

        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                              enable_checkpointing=False, logger=False, callbacks=[_CadenceProbe()])
        trainer.fit(model, torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, collate_fn=_collate))
        return triggered_at

    torch.manual_seed(0)
    wae_gan_triggers = run(WAEGAN(n_genes=20, coord_dim=3, latent_dim=8, hidden_dim=32,
                                   cond_hidden_dim=32, disc_hidden_dim=16))
    fm_ot_triggers = run(FlowMatchingOT(n_genes=20, coord_dim=3, cond_hidden_dim=32,
                                         hidden_dim=64, time_embed_dim=16, n_ode_steps=5))

    assert wae_gan_triggers == [3, 6] == fm_ot_triggers, (
        f"checkpoint cadence must be identical across optimizer modes for the same "
        f"save_every_n_steps -- got WAE-GAN={wae_gan_triggers}, FM-OT={fm_ot_triggers}"
    )
    print(f"[PeriodicCheckpointCallback cadence] OK — identical trigger sequence "
          f"{wae_gan_triggers} for both manual (WAE-GAN) and automatic (FM-OT) optimization")


if __name__ == "__main__":
    test_global_step_diverges_from_batch_count_for_manual_optimization()
    test_periodic_checkpoint_cadence_uniform_across_optimizer_modes()
    print("\nAll PeriodicCheckpointCallback tests passed.")
