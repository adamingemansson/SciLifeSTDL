"""Tests for the self-supervised spatial-expression prior (Architecture 4).

The gates that actually matter here, in order of how badly a silent failure
would mislead the study:

1. **Leakage.** A masked spot must not see its own expression. If it did, the
   task would be trivially solvable by copying, the loss would collapse, and
   the pretrained weights would encode an identity map that helps nothing.
2. **Panel identity.** Loading a prior fit on a different gene panel would
   misalign 17,189 columns silently and produce plausible-looking garbage.
3. **Initialisation-only semantics.** The prior sets where training starts;
   a resume or evaluation must still take its weights from the checkpoint.
4. **It actually learns.** A module that runs without reducing the masked-spot
   loss is indistinguishable from one that was never trained.
"""
import json

import numpy as np
import pytest
import torch

from gen3_multiscale.conditional_wae.contract import static_audit_conditional_wae_config
from gen3_multiscale.conditional_wae.spatial_prior import (
    SPATIAL_PRIOR_FORMAT,
    load_spatial_prior_into,
    mask_spots,
    masked_spot_loss,
    masked_spot_pearson,
    pretrain_spatial_prior,
    pretrain_spatial_prior_streaming,
    save_spatial_prior,
)
from gen3_multiscale.conditional_wae.spatial_refinement import SpatialExpressionRefiner
from gen3_multiscale.training import train_conditional_wae

N_GENES = 6
CONTEXT_DIM = 12
GENE_NAMES = [f"GENE{i}" for i in range(N_GENES)]


def _refiner(seed: int = 0) -> SpatialExpressionRefiner:
    # geometry_dim is left at the default because that is the only value
    # _build_model can construct (it is not a config field), and several tests
    # below load these artifacts into a _build_model-produced refiner.
    torch.manual_seed(seed)
    return SpatialExpressionRefiner(
        N_GENES, CONTEXT_DIM, gex_feature_dim=8, hidden_dim=16, k_neighbors=3,
    )


def _slide(n_spots: int = 24, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic slide with REAL spatial structure: expression is a smooth
    function of position, so a neighbour-attending module can genuinely beat
    the per-gene mean and the learning gate below is meaningful."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_spots)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), axis=-1).reshape(-1, 2)
    coords = grid[:n_spots].astype(np.float32)
    frequencies = rng.normal(size=(N_GENES, 2)) * 0.6
    phases = rng.uniform(0, 2 * np.pi, size=N_GENES)
    expression = np.sin(coords @ frequencies.T + phases).astype(np.float32)
    return expression, coords


def test_masked_spot_loss_hides_the_spot_from_itself():
    """The single most important property: changing ONLY the masked spots'
    own expression must not change what the refiner predicts for them."""
    refiner = _refiner()
    with torch.no_grad():
        for parameter in refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    expression_np, coords_np = _slide()
    expression = torch.as_tensor(expression_np)
    coords = torch.as_tensor(coords_np)
    spot_mask = mask_spots(expression.shape[0], 0.25, torch.Generator().manual_seed(0))

    def predict(values: torch.Tensor) -> torch.Tensor:
        visible = values.clone()
        visible[spot_mask] = 0.0
        context = torch.zeros(values.shape[0], CONTEXT_DIM)
        indices, mask = refiner.cached_neighbor_graph(coords)
        return refiner(visible, context, coords, indices, mask)[spot_mask]

    tampered = expression.clone()
    tampered[spot_mask] += 100.0
    torch.testing.assert_close(predict(expression), predict(tampered))


def test_masked_spot_loss_does_use_the_neighbours():
    """The complement of the leakage test: perturbing the UNMASKED neighbours
    must change the prediction, or the module is ignoring its only input."""
    refiner = _refiner()
    with torch.no_grad():
        for parameter in refiner.update_head[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    expression_np, coords_np = _slide()
    expression = torch.as_tensor(expression_np)
    coords = torch.as_tensor(coords_np)
    spot_mask = mask_spots(expression.shape[0], 0.25, torch.Generator().manual_seed(0))
    baseline = masked_spot_loss(refiner, expression, coords, spot_mask)
    perturbed = expression.clone()
    perturbed[~spot_mask] += 3.0
    assert not torch.allclose(
        baseline, masked_spot_loss(refiner, perturbed, coords, spot_mask),
    )


def test_mask_spots_always_leaves_neighbours_to_predict_from():
    generator = torch.Generator().manual_seed(0)
    for fraction in (0.01, 0.25, 0.99):
        mask = mask_spots(50, fraction, generator)
        assert 1 <= int(mask.sum()) <= 49
    with pytest.raises(ValueError, match="mask_fraction"):
        mask_spots(50, 0.0, generator)
    with pytest.raises(ValueError, match="mask_fraction"):
        mask_spots(50, 1.0, generator)
    with pytest.raises(ValueError, match="at least two spots"):
        mask_spots(1, 0.25, generator)


def test_pretraining_actually_reduces_the_masked_spot_loss():
    refiner = _refiner()
    slides = [_slide(seed=seed) for seed in range(3)]
    history = pretrain_spatial_prior(
        refiner, slides, steps=120, learning_rate=3e-3, seed=0, log_every=20,
    )
    assert history[-1] < history[0], (
        f"masked-spot loss did not fall ({history[0]:.4f} -> {history[-1]:.4f}); "
        "the prior would be an untrained module wearing a trained module's name"
    )


def test_pretraining_improves_the_projects_own_metric_not_just_mse():
    """A module that learns each gene's slide mean drives MSE down while
    scoring ~0 on within-slide per-gene Pearson, which is what we actually
    report. Gate on the metric we report."""
    refiner = _refiner()
    expression_np, coords_np = _slide(n_spots=36)
    expression = torch.as_tensor(expression_np)
    coords = torch.as_tensor(coords_np)
    spot_mask = mask_spots(expression.shape[0], 0.25, torch.Generator().manual_seed(3))
    # Before training the refiner is exactly the identity (zero-initialised
    # update head), so it returns the masked spots' ZEROED input: a constant
    # prediction, whose correlation is undefined for every gene. A NaN in the
    # first log line of a real pretraining run is this, not a bug.
    assert np.isnan(masked_spot_pearson(refiner, expression, coords, spot_mask))
    pretrain_spatial_prior(
        refiner, [(expression_np, coords_np)], steps=300, learning_rate=3e-3, seed=0,
        log_every=100,
    )
    after = masked_spot_pearson(refiner, expression, coords, spot_mask)
    assert after > 0.3, f"masked-spot PCC only reached {after:.4f}"


def test_streaming_pretraining_holds_one_slide_and_visits_every_sample():
    refiner = _refiner()
    slides = {f"S{i}": _slide(seed=i) for i in range(4)}
    resident = []

    def load(sample_id):
        resident.append(sample_id)
        return slides[sample_id]

    history = pretrain_spatial_prior_streaming(
        refiner, sorted(slides), load, rounds=2, steps_per_slide=2,
        learning_rate=3e-3, seed=0,
    )
    assert len(history) == 8
    assert sorted(record["sample_id"] for record in history if record["round"] == 1) == sorted(slides)
    assert len(resident) == 8  # loaded once per visit, never held across visits
    assert all(np.isfinite(record["last_loss"]) for record in history)


def test_save_and_load_round_trips_the_weights(tmp_path):
    trained = _refiner()
    pretrain_spatial_prior(
        trained, [_slide()], steps=20, learning_rate=3e-3, seed=0, log_every=50,
    )
    path = save_spatial_prior(
        trained, tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={"rounds": 1},
    )
    fresh = _refiner(seed=99)
    identity = load_spatial_prior_into(fresh, path, gene_names=GENE_NAMES)
    assert identity["format"] == SPATIAL_PRIOR_FORMAT
    assert identity["provenance"] == {"rounds": 1}
    for (name, expected), (_name, actual) in zip(
        trained.state_dict().items(), fresh.state_dict().items(),
    ):
        torch.testing.assert_close(expected, actual, msg=f"{name} did not round-trip")


def test_loading_a_prior_from_a_different_gene_panel_fails_closed(tmp_path):
    path = save_spatial_prior(
        _refiner(), tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={},
    )
    other_panel = list(GENE_NAMES)
    other_panel[0] = "SOMETHING_ELSE"
    with pytest.raises(ValueError, match="gene panel differs"):
        load_spatial_prior_into(_refiner(), path, gene_names=other_panel)
    # Same names, different ORDER is also a real misalignment.
    with pytest.raises(ValueError, match="gene panel differs"):
        load_spatial_prior_into(_refiner(), path, gene_names=list(reversed(GENE_NAMES)))


def test_loading_a_prior_with_mismatched_dimensions_fails_closed(tmp_path):
    path = save_spatial_prior(
        _refiner(), tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={},
    )
    torch.manual_seed(0)
    wider = SpatialExpressionRefiner(
        N_GENES, CONTEXT_DIM, gex_feature_dim=8, hidden_dim=32, k_neighbors=3,
    )
    with pytest.raises(ValueError, match="hidden_dim"):
        load_spatial_prior_into(wider, path, gene_names=GENE_NAMES)
    torch.manual_seed(0)
    other_k = SpatialExpressionRefiner(
        N_GENES, CONTEXT_DIM, gex_feature_dim=8, hidden_dim=16, k_neighbors=5,
    )
    with pytest.raises(ValueError, match="k_neighbors"):
        load_spatial_prior_into(other_k, path, gene_names=GENE_NAMES)
    # geometry_dim shapes real parameters too; without it in the identity
    # record this surfaced as load_state_dict's opaque size-mismatch dump.
    torch.manual_seed(0)
    other_geometry = SpatialExpressionRefiner(
        N_GENES, CONTEXT_DIM, gex_feature_dim=8, hidden_dim=16,
        geometry_dim=8, k_neighbors=3,
    )
    with pytest.raises(ValueError, match="geometry_dim"):
        load_spatial_prior_into(other_geometry, path, gene_names=GENE_NAMES)


def test_loading_a_non_prior_artifact_fails_closed(tmp_path):
    path = tmp_path / "not_a_prior.pt"
    torch.save({"format": "something_else", "state_dict": {}}, path)
    with pytest.raises(ValueError, match=SPATIAL_PRIOR_FORMAT):
        load_spatial_prior_into(_refiner(), path, gene_names=GENE_NAMES)


def _prior_config(spatial_prior_path=None, n_refinement_steps=2):
    params = {
        "image_feature_dim": 16, "latent_dim": 8, "hidden_dim": CONTEXT_DIM,
        "gex_feature_dim": 6, "autoencoder_hidden_dim": 20, "discriminator_hidden_dim": 12,
        "gene_encoder_source": "linear", "encoder_conditioning": "none",
        "n_heads": 4, "n_blocks": 1, "dense_threshold": 20, "sparse_k": 3,
        "n_inference_samples": 3,
        "n_refinement_steps": n_refinement_steps, "refinement_k_neighbors": 3,
        "refinement_hidden_dim": 16, "refinement_gex_feature_dim": 8,
    }
    data = {"gen3_manifest_path": "manifest.json", "tile_encoder_revision": "abc"}
    if spatial_prior_path is not None:
        data["spatial_prior_path"] = str(spatial_prior_path)
    return {
        "model": {
            "arm": "wae_he_mmd_geneencoder_mlp_nofilm", "kind": "conditional_wae",
            "task": "he_to_st", "regularizer": "mmd", "include_observed_gex": False,
            "image_mode": "full_visible", "params": params,
        },
        "data": data,
        "training": {"checkpoint_dir": "checkpoints"},
        "loss": {"pcc_weight": 0.1, "regularizer_weight": 0.1, "conditional_mean_weight": 1.0},
    }


def test_static_contract_reports_the_prior_and_rejects_it_without_a_refiner():
    assert static_audit_conditional_wae_config(_prior_config())["spatial_prior_path"] is None
    report = static_audit_conditional_wae_config(_prior_config("/tmp/prior.pt"))
    assert report["spatial_prior_path"] == "/tmp/prior.pt"
    with pytest.raises(ValueError, match="no spatial refiner"):
        static_audit_conditional_wae_config(
            _prior_config("/tmp/prior.pt", n_refinement_steps=0)
        )


def test_build_model_initialises_the_refiner_from_the_configured_prior(tmp_path):
    """The end-to-end wiring: a config naming a prior must produce a model
    whose refiner carries the pretrained weights, not fresh random ones."""
    trained = _refiner()
    pretrain_spatial_prior(
        trained, [_slide()], steps=40, learning_rate=3e-3, seed=0, log_every=100,
    )
    path = save_spatial_prior(
        trained, tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={},
    )
    without = train_conditional_wae._build_model(
        _prior_config(), n_genes=N_GENES, gene_names=GENE_NAMES,
    )
    with_prior = train_conditional_wae._build_model(
        _prior_config(path), n_genes=N_GENES, gene_names=GENE_NAMES,
    )
    reference = trained.state_dict()["update_head.3.weight"]
    torch.testing.assert_close(
        with_prior.spatial_refiner.state_dict()["update_head.3.weight"], reference,
    )
    assert not torch.allclose(
        without.spatial_refiner.state_dict()["update_head.3.weight"], reference,
    )


def test_build_model_requires_gene_names_when_a_prior_is_configured(tmp_path):
    path = save_spatial_prior(
        _refiner(), tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={},
    )
    with pytest.raises(ValueError, match="gene_names"):
        train_conditional_wae._build_model(_prior_config(path), n_genes=N_GENES)


def test_the_prior_is_initialisation_only_and_a_checkpoint_overrides_it(tmp_path):
    """Loading trained state after construction must win. Otherwise a resume
    would silently rewind the refiner to its pretrained starting point."""
    path = save_spatial_prior(
        _refiner(), tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance={},
    )
    model = train_conditional_wae._build_model(
        _prior_config(path), n_genes=N_GENES, gene_names=GENE_NAMES,
    )
    checkpoint_state = {
        name: torch.full_like(tensor, 0.5)
        for name, tensor in model.state_dict().items()
    }
    model.load_state_dict(checkpoint_state)
    torch.testing.assert_close(
        model.spatial_refiner.state_dict()["update_head.3.weight"],
        torch.full_like(model.spatial_refiner.state_dict()["update_head.3.weight"], 0.5),
    )


def test_pretraining_cli_runs_end_to_end_and_writes_a_loadable_prior(tmp_path, monkeypatch, capsys):
    """The whole stage-1 command, from a YAML on disk to an artifact the
    trainer can consume. Only the HEST reader is stubbed; config resolution,
    the static audit, model construction, the streaming loop and the save all
    run for real, because this is the path that executes on the server where
    I cannot iterate on a traceback."""
    import types

    import yaml

    from gen3_multiscale.data import example_builder
    from gen3_multiscale.scripts import pretrain_conditional_wae_spatial_prior as cli

    sample_ids = ["S0", "S1", "S2"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "gene_panel": GENE_NAMES,
        "train_sample_ids": sample_ids,
        "validation_sample_ids": ["V0"],
        "test_sample_ids": ["T0"],
    }))
    config = _prior_config()
    config["data"]["gen3_manifest_path"] = str(manifest_path)
    config_path = tmp_path / "arm.yaml"
    config_path.write_text(yaml.safe_dump(config))

    opened = []

    def fake_load(manifest, sample_id, hest_data_dir=None):
        assert sample_id in sample_ids, "a non-TRAIN slide was opened during pretraining"
        opened.append(sample_id)
        expression, coords = _slide(n_spots=25, seed=sample_ids.index(sample_id))
        return types.SimpleNamespace(X=expression, obsm={"spatial": coords})

    monkeypatch.setattr(example_builder, "load_expression_for_model_target_space", fake_load)
    output = tmp_path / "prior.pt"
    monkeypatch.setattr("sys.argv", [
        "pretrain_conditional_wae_spatial_prior",
        "--config", str(config_path), "--output", str(output),
        "--rounds", "2", "--steps-per-slide", "2", "--learning-rate", "3e-3",
    ])
    cli.main()

    assert sorted(set(opened)) == sample_ids
    assert output.exists()
    identity = load_spatial_prior_into(
        train_conditional_wae._build_model(
            _prior_config(), n_genes=N_GENES, gene_names=GENE_NAMES,
        ).spatial_refiner,
        output, gene_names=GENE_NAMES,
    )
    assert identity["provenance"]["n_train_samples"] == 3
    assert identity["provenance"]["train_sample_ids"] == sample_ids
    assert json.loads((tmp_path / "prior.pt.provenance.json").read_text())["rounds"] == 2
    assert "spatial prior saved to" in capsys.readouterr().out


def test_pretraining_cli_refuses_a_config_with_no_refiner(tmp_path, monkeypatch):
    import yaml

    from gen3_multiscale.scripts import pretrain_conditional_wae_spatial_prior as cli

    config = _prior_config(n_refinement_steps=0)
    config["data"]["gen3_manifest_path"] = str(tmp_path / "manifest.json")
    config_path = tmp_path / "arm.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("sys.argv", [
        "pretrain_conditional_wae_spatial_prior",
        "--config", str(config_path), "--output", str(tmp_path / "prior.pt"),
    ])
    with pytest.raises(ValueError, match="no spatial refiner to pretrain"):
        cli.main()


def test_pretraining_cli_ignores_a_prior_already_named_by_the_config(tmp_path, monkeypatch):
    """Stage 1 PRODUCES the prior. If the config already names one -- e.g. the
    command is re-run against a suite config that was prepared with
    --spatial-prior-path -- consuming it would make the run a continuation of
    the old fit while its provenance still claimed a fresh one.

    The named prior here does not exist, so if the CLI failed to strip it,
    _build_model would raise trying to load it and this run could not finish.
    """
    import types

    import yaml

    from gen3_multiscale.data import example_builder
    from gen3_multiscale.scripts import pretrain_conditional_wae_spatial_prior as cli

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "gene_panel": GENE_NAMES, "train_sample_ids": ["S0"],
        "validation_sample_ids": [], "test_sample_ids": [],
    }))
    config = _prior_config(spatial_prior_path=tmp_path / "does_not_exist.pt")
    config["data"]["gen3_manifest_path"] = str(manifest_path)
    config_path = tmp_path / "arm.yaml"
    config_path.write_text(yaml.safe_dump(config))

    monkeypatch.setattr(
        example_builder, "load_expression_for_model_target_space",
        lambda manifest, sample_id, hest_data_dir=None: types.SimpleNamespace(
            X=_slide(n_spots=16)[0], obsm={"spatial": _slide(n_spots=16)[1]},
        ),
    )
    output = tmp_path / "prior.pt"
    monkeypatch.setattr("sys.argv", [
        "pretrain_conditional_wae_spatial_prior",
        "--config", str(config_path), "--output", str(output),
        "--rounds", "1", "--steps-per-slide", "1",
    ])
    cli.main()
    assert output.exists()


def test_saved_prior_records_enough_provenance_to_reproduce_it(tmp_path):
    provenance = {
        "config_path": "arm.yaml", "n_train_samples": 3, "rounds": 2,
        "mask_fraction": 0.25, "seed": 7,
    }
    path = save_spatial_prior(
        _refiner(), tmp_path / "prior.pt", gene_names=GENE_NAMES, provenance=provenance,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["provenance"] == provenance
    assert payload["n_genes"] == N_GENES
    assert json.loads(json.dumps(payload["provenance"])) == provenance
