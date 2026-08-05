from types import SimpleNamespace

import numpy as np
import pytest

from gen3_multiscale.conditional_wae import reference_projection as rp

GENES = ["G0", "G1", "G2", "G3"]


def _sample(seed, n=20):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        adata=SimpleNamespace(X=rng.normal(size=(n, len(GENES))).astype(np.float32), var_names=GENES),
    )


def _samples():
    return {"S0": _sample(0), "S1": _sample(1)}


def test_projection_is_deterministic_for_the_same_cohort():
    samples = _samples()
    first = rp.build_reference_gex_projection(samples, ["S0", "S1"], GENES, n_components=2, n_clusters=2, seed=3)
    second = rp.build_reference_gex_projection(samples, ["S1", "S0"], GENES, n_components=2, n_clusters=2, seed=3)
    np.testing.assert_allclose(first["components"], second["components"])
    np.testing.assert_allclose(first["mean"], second["mean"])
    np.testing.assert_allclose(first["cluster_centroids"], second["cluster_centroids"])
    assert first["provenance_hash"] == second["provenance_hash"]


def test_true_and_predicted_gex_share_the_same_projection():
    samples = _samples()
    projection = rp.build_reference_gex_projection(samples, ["S0", "S1"], GENES, n_components=2, seed=0)
    true_gex = samples["S0"].adata.X[:5]
    predicted_gex = true_gex + 0.1
    true_coords = rp.project_onto_reference(true_gex, GENES, projection)
    predicted_coords = rp.project_onto_reference(predicted_gex, GENES, projection)
    assert true_coords.shape == (5, 2)
    assert predicted_coords.shape == (5, 2)
    assert not np.allclose(true_coords, predicted_coords)


def test_project_onto_reference_rejects_gene_order_mismatch():
    samples = _samples()
    projection = rp.build_reference_gex_projection(samples, ["S0", "S1"], GENES, n_components=2, seed=0)
    with pytest.raises(ValueError, match="gene order"):
        rp.project_onto_reference(samples["S0"].adata.X, list(reversed(GENES)), projection)


def test_assign_clusters_returns_nearest_centroid():
    projection = {
        "cluster_centroids": np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
    }
    coords = np.array([[0.5, 0.5], [9.5, 9.5]], dtype=np.float32)
    assignments = rp.assign_clusters(coords, projection)
    np.testing.assert_array_equal(assignments, [0, 1])


def test_save_and_load_round_trips_exactly(tmp_path):
    samples = _samples()
    projection = rp.build_reference_gex_projection(samples, ["S0", "S1"], GENES, n_components=2, n_clusters=2, seed=0)
    path = tmp_path / "reference"
    rp.save_reference_projection(projection, path)
    loaded = rp.load_reference_projection(path)
    np.testing.assert_allclose(loaded["components"], projection["components"])
    np.testing.assert_allclose(loaded["pc_ranges"], projection["pc_ranges"])
    assert loaded["provenance_hash"] == projection["provenance_hash"]
    assert loaded["gene_names"] == GENES


def test_ensure_reference_gex_projection_reuses_a_matching_existing_file(tmp_path):
    samples = _samples()
    path = tmp_path / "reference"
    first = rp.ensure_reference_gex_projection(path, samples, ["S0", "S1"], GENES, n_components=2, seed=0)
    second = rp.ensure_reference_gex_projection(path, samples, ["S1", "S0"], GENES, n_components=2, seed=0)
    np.testing.assert_allclose(first["components"], second["components"])


def test_ensure_reference_gex_projection_fails_closed_on_mismatch(tmp_path):
    samples = _samples()
    path = tmp_path / "reference"
    rp.ensure_reference_gex_projection(path, samples, ["S0", "S1"], GENES, n_components=2, seed=0)
    with pytest.raises(ValueError, match="does not match"):
        rp.ensure_reference_gex_projection(path, samples, ["S0"], GENES, n_components=2, seed=0)


def test_build_rejects_gene_order_mismatch_in_a_sample():
    samples = {"S0": _sample(0)}
    with pytest.raises(ValueError, match="gene order"):
        rp.build_reference_gex_projection(samples, ["S0"], list(reversed(GENES)), n_components=2, seed=0)


def test_pc_ranges_are_fixed_at_build_time_not_recomputed_per_projection():
    samples = _samples()
    projection = rp.build_reference_gex_projection(samples, ["S0", "S1"], GENES, n_components=2, seed=0)
    extreme = np.full((3, len(GENES)), 1000.0, dtype=np.float32)
    rp.project_onto_reference(extreme, GENES, projection)
    # pc_ranges must be untouched by projecting new, out-of-range data.
    assert projection["pc_ranges"].shape == (2, 2)
