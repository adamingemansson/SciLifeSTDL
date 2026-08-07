"""Tests for the organ-stratified, mean-variance-trend-corrected
dispersion panels added to train_gene_panels.py.

Motivating real-world bug: a gene like ALB (Albumin, a liver-specific
marker) can dominate the pooled/global RAW-VARIANCE top panel purely
because of liver samples elsewhere in a multi-organ training cohort --
its expression swings from ~0 (every non-liver organ) to high (liver),
which inflates pooled variance even though, on any single non-liver
slide, ALB is just near-zero background/dropout noise with no real
spatial signal. Organ-specific, mean-variance-trend-corrected dispersion
ranking should NOT surface ALB as informative for a non-liver organ."""
from types import SimpleNamespace

import numpy as np
import pytest

from gen3_multiscale.evaluation import train_gene_panels as panels_module


def test_binned_dispersion_ranks_excess_variance_above_same_mean_peers():
    rng = np.random.default_rng(0)
    n_genes = 100
    mean = rng.uniform(1.0, 5.0, size=n_genes)
    # Baseline dispersion proportional to mean (the normal mean-variance
    # trend); gene 7 gets 5x the dispersion of its same-mean peers.
    variance = mean * 0.2
    variance[7] = mean[7] * 1.0
    gene_names = [f"g{i:03d}" for i in range(n_genes)]
    ranking = panels_module._binned_dispersion_ranking(gene_names, mean, variance)
    top_gene = ranking[0]["gene"]
    assert top_gene == "g007"
    # A gene with dispersion exactly matching its bin's trend should sit
    # near the middle of the ranking, not the top.
    ranked_genes = [row["gene"] for row in ranking]
    assert ranked_genes.index("g007") < ranked_genes.index(gene_names[0])


def _multi_organ_manifest():
    n_filler = 22
    gene_names = ["ALB", "G1"] + [f"filler{i:02d}" for i in range(n_filler)]
    return {
        "gene_panel": gene_names,
        "train_sample_ids": ["L1", "L2", "K1", "K2"],
        "validation_sample_ids": [],
        "test_sample_ids": [],
        "build_args": {"expression_transform": "log1p", "expression_target_sum": 10000},
        "samples": {
            "L1": {"organ": "Liver"}, "L2": {"organ": "Liver"},
            "K1": {"organ": "Kidney"}, "K2": {"organ": "Kidney"},
        },
    }


def _multi_organ_fake_loader():
    rng = np.random.default_rng(1)
    n_spots = 30
    n_filler = 22
    # Filler genes: identical distribution in every sample/organ (real,
    # organ-agnostic variance) so pooled and per-organ dispersion agree
    # on them -- they exist purely to give the mean-variance binning
    # enough members per bin to be meaningful.
    filler = {
        f"filler{i:02d}": rng.uniform(0.5, 4.0, size=n_spots) for i in range(n_filler)
    }

    def fake_load(_manifest, sample_id):
        organ = "Liver" if sample_id.startswith("L") else "Kidney"
        alb = (
            rng.uniform(4.5, 6.0, size=n_spots) if organ == "Liver"
            else np.zeros(n_spots)
        )
        g1 = rng.uniform(0.5, 4.0, size=n_spots)  # real variance on every organ
        columns = [alb, g1] + [filler[name] + rng.normal(0, 1e-6, n_spots) for name in sorted(filler)]
        matrix = np.stack(columns, axis=1)
        gene_names = ["ALB", "G1"] + sorted(filler)
        return SimpleNamespace(X=matrix, var_names=gene_names)

    return fake_load


def test_pooled_panel_surfaces_alb_but_organ_panel_does_not(monkeypatch):
    manifest = _multi_organ_manifest()
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    artifact = panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))

    # Pooled ranking mixes Liver (ALB ~5) with Kidney (ALB ~0): the raw
    # cross-organ swing inflates ALB's pooled variance to the top.
    assert artifact["panels"]["train_log1p_variance_top1"] == ["ALB"]

    # Within Kidney alone, ALB is a flat zero (no real signal) -- the
    # organ-specific dispersion panel must not lead with it.
    kidney_top1 = artifact["panels_by_organ"]["Kidney"]["train_dispersion_top1"]
    assert kidney_top1 != ["ALB"]


def test_organs_are_computed_from_only_their_own_training_samples(monkeypatch):
    manifest = _multi_organ_manifest()
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    artifact = panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))

    kidney_ranking = {row["gene"]: row for row in artifact["ranking_by_organ"]["Kidney"]}
    liver_ranking = {row["gene"]: row for row in artifact["ranking_by_organ"]["Liver"]}
    # ALB's organ-local mean must reflect ONLY that organ's samples.
    assert kidney_ranking["ALB"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert liver_ranking["ALB"]["mean"] > 1.0


def test_panels_by_organ_are_nested_prefixes(monkeypatch):
    manifest = _multi_organ_manifest()
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    artifact = panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5, 10))
    for organ_panels in artifact["panels_by_organ"].values():
        top1, top5, top10 = (
            organ_panels["train_dispersion_top1"], organ_panels["train_dispersion_top5"],
            organ_panels["train_dispersion_top10"],
        )
        assert top5[:1] == top1
        assert top10[:5] == top5


def test_build_rejects_an_organ_with_fewer_than_two_training_samples(monkeypatch):
    manifest = _multi_organ_manifest()
    manifest["train_sample_ids"] = ["L1", "K1", "K2"]  # Liver has only one
    manifest["samples"] = {
        "L1": {"organ": "Liver"}, "K1": {"organ": "Kidney"}, "K2": {"organ": "Kidney"},
    }
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    with pytest.raises(ValueError, match="Liver.*only 1 training sample"):
        panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))


def test_build_rejects_a_train_sample_with_no_organ_field(monkeypatch):
    manifest = _multi_organ_manifest()
    del manifest["samples"]["K2"]["organ"]
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    with pytest.raises(ValueError, match="K2.*organ"):
        panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))


def test_save_load_validate_round_trips_organ_panels(tmp_path, monkeypatch):
    manifest = _multi_organ_manifest()
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    artifact = panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))
    path = panels_module.save_train_derived_gene_panels(artifact, tmp_path / "panels.json")
    loaded = panels_module.load_train_derived_gene_panels(path, manifest)
    assert loaded["panels_by_organ"].keys() == {"Liver", "Kidney"}


def test_validate_rejects_an_artifact_missing_panels_by_organ(monkeypatch):
    """Simulates a pre-organ-stratification (schema v1) artifact still on
    disk -- must fail closed rather than silently skip organ-aware
    selection."""
    manifest = _multi_organ_manifest()
    monkeypatch.setattr(panels_module, "load_expression_for_model_target_space", _multi_organ_fake_loader())
    artifact = panels_module.build_train_derived_gene_panels(manifest, panel_sizes=(1, 5))
    del artifact["panels_by_organ"]
    del artifact["ranking_by_organ"]
    del artifact["method_by_organ"]
    artifact["version"] = 1
    artifact["artifact_sha256"] = panels_module._canonical_hash(artifact)
    with pytest.raises(ValueError, match="unsupported"):
        panels_module.validate_train_derived_gene_panels(artifact, manifest)
