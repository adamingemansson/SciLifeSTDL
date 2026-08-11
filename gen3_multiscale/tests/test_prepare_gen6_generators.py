import hashlib
import json
from types import SimpleNamespace

import yaml

from gen3_multiscale.gen5.autoencoder import (
    ExpressionAutoencoder,
    save_expression_autoencoder_checkpoint,
)
from gen3_multiscale.scripts import prepare_gen6_generators as module
from gen3_multiscale.training.train import dataset_manifest_fingerprint


def _staged_artifacts(tmp_path, monkeypatch):
    """Write a verified gen6c bundle + expression autoencoder and return the
    autoencoder path, so several tests can drive ``prepare`` for real."""
    genes = ["g0", "g1", "g2"]
    manifest = {
        "gene_panel": genes,
        "train_sample_ids": ["train"],
        "validation_sample_ids": ["validation"],
        "samples": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    fingerprints = {
        "uni2_checkpoint": "uni2.bin",
        "uni2_revision": "a" * 40,
        "uni2_package_version": "1",
        "uni2_preprocessing_spec": "spec",
        "scfoundation_checkpoint": "scf.ckpt",
        "scfoundation_vocab": "vocab.tsv",
        "scfoundation_package_version": "1",
        "scfoundation_preprocessing_spec": "spec",
    }
    base = {
        "experiment_name": "gen6c",
        "model": {"arm": "gen6c", "kind": "conditioner", "params": {
            "hidden_dim": 16, "n_heads": 2, "gex_context_embedding_dim": 6,
        }},
        "masking": {"strata": [{"name": "small"}]},
        "data": {"gen3_manifest_path": str(manifest_path)},
        "evaluation": {},
        "required_fingerprints": fingerprints,
        "training": {"seed": 0},
    }
    bundle = tmp_path / "conditioner_bundle"
    bundle.mkdir()
    (bundle / "model_config.json").write_text(json.dumps(base))
    monkeypatch.setattr(
        module.checkpoint_module,
        "resolve_checkpoint_identity",
        lambda _: SimpleNamespace(resolved_dir=bundle, weights_sha256="conditioner-sha"),
    )

    manifest_fp = dataset_manifest_fingerprint(manifest)
    autoencoder_path = tmp_path / "autoencoder.pt"
    autoencoder = ExpressionAutoencoder(3, genes, latent_dim=2, hidden_dim=8)
    save_expression_autoencoder_checkpoint(
        autoencoder, autoencoder_path,
        dataset_manifest_fingerprint=manifest_fp,
        preprocessing_spec="spec", code_identity="test",
    )
    autoencoder_sha = hashlib.sha256(autoencoder_path.read_bytes()).hexdigest()
    report = {
        "kind": "gen5_expression_autoencoder_training_report",
        "checkpoint_sha256": autoencoder_sha,
        "dataset_manifest_fingerprint": manifest_fp,
        "train_sample_ids": ["train"],
        "n_genes": 3,
        "latent_dim": 2,
        "hidden_dim": 8,
        "validation_reconstruction": {
            "rmse": 0.2, "pcc_mean": 0.7, "sample_ids": ["validation"],
        },
    }
    (tmp_path / "autoencoder.pt.report.json").write_text(json.dumps(report))
    return autoencoder_path


def test_prepare_gen6_generators_binds_same_gen6c_and_verified_autoencoder(
    tmp_path, monkeypatch,
):
    autoencoder_path = _staged_artifacts(tmp_path, monkeypatch)
    root = tmp_path / "generators"
    plan = module.prepare(
        conditioner_checkpoint=str(tmp_path / "best"),
        autoencoder_checkpoint=str(autoencoder_path),
        output_root=str(root), hours=0.01,
    )
    k = yaml.safe_load((root / "configs" / "gen6k.yaml").read_text())
    l = yaml.safe_load((root / "configs" / "gen6l.yaml").read_text())
    assert plan["conditioner_arm"] == "gen6c"
    assert k["model"]["kind"] == "latent_flow"
    assert l["model"]["kind"] == "wae_gan"
    assert k["model"]["params"]["conditioner_arm"] == "gen6c"
    assert l["model"]["params"]["conditioner_arm"] == "gen6c"
    assert k["required_fingerprints"]["expression_autoencoder_checkpoint"] == str(
        autoencoder_path.resolve()
    )
    assert "gene_residual_basis" not in k["required_fingerprints"]


def test_gen6p_differs_from_gen6k_only_in_ot_assignment(tmp_path, monkeypatch):
    """gen6p exists to test hard-versus-barycentric OT pairing. Any second
    differing parameter would make its delta uninterpretable, and gen6k must
    stay pinned to the barycentric behaviour it was originally trained with
    rather than silently inheriting the new default."""
    autoencoder_path = _staged_artifacts(tmp_path, monkeypatch)
    root = tmp_path / "generators"
    module.prepare(
        conditioner_checkpoint=str(tmp_path / "best"),
        autoencoder_checkpoint=str(autoencoder_path),
        output_root=str(root), hours=0.01,
    )
    k = yaml.safe_load((root / "configs" / "gen6k.yaml").read_text())["model"]
    p = yaml.safe_load((root / "configs" / "gen6p.yaml").read_text())["model"]
    assert k["kind"] == p["kind"] == "latent_flow"
    assert k["params"]["ot_assignment"] == "barycentric"
    assert p["params"]["ot_assignment"] == "hard"
    differing = {
        name for name in set(k["params"]) | set(p["params"])
        if k["params"].get(name) != p["params"].get(name)
    }
    assert differing == {"ot_assignment"}


def test_gen6o_differs_from_gen6l_only_in_the_conditional_mean_head_and_rho(
    tmp_path, monkeypatch,
):
    autoencoder_path = _staged_artifacts(tmp_path, monkeypatch)
    root = tmp_path / "generators"
    module.prepare(
        conditioner_checkpoint=str(tmp_path / "best"),
        autoencoder_checkpoint=str(autoencoder_path),
        output_root=str(root), hours=0.01,
    )
    l = yaml.safe_load((root / "configs" / "gen6l.yaml").read_text())["model"]
    o = yaml.safe_load((root / "configs" / "gen6o.yaml").read_text())["model"]
    assert l["kind"] == o["kind"] == "wae_gan"
    # Weight 0 leaves the head without gradient, so gen6l reproduces the
    # original arm exactly even though the module now exists on both.
    assert l["params"]["conditional_mean_weight"] == 0.0
    assert o["params"]["conditional_mean_weight"] > 0
    assert l["params"]["latent_spatial_correlation"] == 0.0
    assert 0.0 < o["params"]["latent_spatial_correlation"] <= 1.0
    differing = {
        name for name in set(l["params"]) | set(o["params"])
        if l["params"].get(name) != o["params"].get(name)
    }
    assert differing == {"conditional_mean_weight", "latent_spatial_correlation"}


def test_wae_gan_arms_do_not_demand_an_autoencoder_fingerprint(tmp_path, monkeypatch):
    autoencoder_path = _staged_artifacts(tmp_path, monkeypatch)
    root = tmp_path / "generators"
    module.prepare(
        conditioner_checkpoint=str(tmp_path / "best"),
        autoencoder_checkpoint=str(autoencoder_path),
        output_root=str(root), hours=0.01, arms=("gen6o",),
    )
    assert not (root / "configs" / "gen6l.yaml").exists()
    o = yaml.safe_load((root / "configs" / "gen6o.yaml").read_text())
    assert "expression_autoencoder_checkpoint" not in o["required_fingerprints"]
    assert "autoencoder_hidden_dim" not in o["model"]["params"]
