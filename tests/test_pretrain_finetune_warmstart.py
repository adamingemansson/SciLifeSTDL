"""
Regression tests for load_pretrained_weights_into (src/training/train.py,
2026-07-19) — the pretrain->finetune warm-start mechanism. Motivated by a
real, verified finding (see docs/results_log.md): STPath's own
pretrained-vs-unfrozen ablation showed pretraining alone is worth ~0.086
PCC, architecture held constant — StormLite never had an actual
pretraining stage before this; it always trained directly on the target
task. This is the mechanism for a genuine two-stage recipe: pretrain
(e.g. multi-sample across many samples) -> save via save_trained_model ->
finetune (build a fresh model for the target task, warm-start via this
function, then train normally).

Run with: python -m tests.test_pretrain_finetune_warmstart
"""
import tempfile

import torch

from src.models.registry import build_model
from src.training.train import save_trained_model, load_pretrained_weights_into


def test_warmstart_transfers_weights_exactly_for_matching_architecture():
    torch.manual_seed(0)
    cfg = {"name": "fm_ot", "params": {"n_genes": 20, "coord_dim": 3, "cond_hidden_dim": 16,
                                        "hidden_dim": 32, "time_embed_dim": 16, "n_ode_steps": 3}}
    pretrained = build_model(cfg)
    with torch.no_grad():
        pretrained.velocity_net[0].weight.add_(5.0)  # simulate real training movement
    expected = pretrained.velocity_net[0].weight.clone()

    with tempfile.TemporaryDirectory() as tmp:
        save_trained_model(pretrained, cfg, [f"G{i}" for i in range(20)], tmp)

        torch.manual_seed(99)
        finetune_model = build_model(cfg)
        before = finetune_model.velocity_net[0].weight.clone()
        result = load_pretrained_weights_into(finetune_model, tmp)
        after = finetune_model.velocity_net[0].weight

        assert torch.equal(after, expected), "warm-started weight must exactly match the pretrained checkpoint"
        assert not torch.equal(before, after), "warm-start must actually change the fresh-init weights"
        assert len(result["skipped_shape_mismatch"]) == 0
        assert len(result["skipped_missing_from_checkpoint"]) == 0
        assert len(result["loaded"]) == sum(1 for _ in finetune_model.named_parameters())
        print("[pretrain_finetune] OK — matching architecture transfers every trainable param exactly")


def test_warmstart_gracefully_skips_shape_mismatches_from_different_gene_panel():
    """A different sample's post-QC gene panel changes n_genes, which
    changes the encoder/decoder's first/last layer shapes — those rows
    must be SKIPPED (kept at fresh init), not crash, while panel-agnostic
    layers (e.g. velocity_net, which only ever sees latent_dim, never
    n_genes directly) still warm-start normally."""
    torch.manual_seed(0)
    cfg_pretrain = {"name": "fm_ot", "params": {"n_genes": 20, "coord_dim": 3, "cond_hidden_dim": 16,
                                                 "hidden_dim": 32, "time_embed_dim": 16, "n_ode_steps": 3}}
    pretrained = build_model(cfg_pretrain)

    with tempfile.TemporaryDirectory() as tmp:
        save_trained_model(pretrained, cfg_pretrain, [f"G{i}" for i in range(20)], tmp)

        cfg_finetune = {"name": "fm_ot", "params": {"n_genes": 30, "coord_dim": 3, "cond_hidden_dim": 16,
                                                      "hidden_dim": 32, "time_embed_dim": 16, "n_ode_steps": 3}}
        finetune_model = build_model(cfg_finetune)
        result = load_pretrained_weights_into(finetune_model, tmp)

        assert len(result["skipped_shape_mismatch"]) > 0, "n_genes-dependent layers must be skipped, not crashed on"
        assert "velocity_net.0.weight" in result["loaded"], "panel-agnostic layers must still warm-start"
        assert len(result["loaded"]) > 0
        print(f"[pretrain_finetune] OK — different gene panel: {len(result['skipped_shape_mismatch'])} "
              f"shape-mismatched layers skipped, {len(result['loaded'])} panel-agnostic layers still warm-started")


def test_warmstart_does_not_touch_missing_checkpoint_keys():
    """A finetune model with EXTRA trainable params the pretrain checkpoint
    never had (e.g. a decoder_type the pretrain run didn't use) must keep
    those at fresh init, not crash — 'skipped_missing_from_checkpoint'
    reports them."""
    torch.manual_seed(0)
    cfg_pretrain = {"name": "fm_ot", "params": {"n_genes": 20, "coord_dim": 3, "cond_hidden_dim": 16,
                                                 "hidden_dim": 32, "time_embed_dim": 16, "n_ode_steps": 3,
                                                 "decoder_type": "dense"}}
    pretrained = build_model(cfg_pretrain)

    with tempfile.TemporaryDirectory() as tmp:
        save_trained_model(pretrained, cfg_pretrain, [f"G{i}" for i in range(20)], tmp)

        gene_names = [f"G{i}" for i in range(20)]
        cfg_finetune = {"name": "fm_ot", "params": {
            "n_genes": 20, "coord_dim": 3, "cond_hidden_dim": 16, "hidden_dim": 32,
            "time_embed_dim": 16, "n_ode_steps": 3,
            "decoder_type": "panel_invariant", "decoder_gene_names": gene_names,
        }}
        finetune_model = build_model(cfg_finetune)
        before_gene_embed = finetune_model.decoder.gene_embed.weight.clone()
        result = load_pretrained_weights_into(finetune_model, tmp)

        assert any("gene_embed" in n for n in result["skipped_missing_from_checkpoint"]), (
            "panel_invariant's gene_embed table (never in a dense-decoder checkpoint) must be reported as missing"
        )
        assert torch.equal(finetune_model.decoder.gene_embed.weight, before_gene_embed), (
            "params missing from the checkpoint must stay at their fresh init, untouched"
        )
        assert "velocity_net.0.weight" in result["loaded"], "shared architecture parts still warm-start"
        print("[pretrain_finetune] OK — params absent from the checkpoint (new decoder_type) "
              "stay at fresh init, shared parts still warm-start")


def test_init_checkpoint_dir_unset_is_a_true_noop():
    """cfg.training.get('init_checkpoint_dir') unset (None) — the gating
    logic every entry point (main(), _main_multi_sample(),
    run_comparison.py's _train_model()) uses — must mean
    load_pretrained_weights_into is never called at all."""
    init_checkpoint_dir = None
    called = False

    def _fake_load(*a, **kw):
        nonlocal called
        called = True

    if init_checkpoint_dir:
        _fake_load()
    assert not called, "unset init_checkpoint_dir must never call load_pretrained_weights_into"
    print("[pretrain_finetune] OK — unset init_checkpoint_dir is a true no-op")


if __name__ == "__main__":
    test_warmstart_transfers_weights_exactly_for_matching_architecture()
    test_warmstart_gracefully_skips_shape_mismatches_from_different_gene_panel()
    test_warmstart_does_not_touch_missing_checkpoint_keys()
    test_init_checkpoint_dir_unset_is_a_true_noop()
    print("\nAll pretrain/finetune warm-start tests passed.")
