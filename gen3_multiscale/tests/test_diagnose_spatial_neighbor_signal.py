"""Tests for the spatial-neighbour-signal premise check.

This diagnostic exists to decide whether a whole architectural direction is
founded, so the ways it could lie matter more than usual:

* If hidden spots leaked into their own neighbour averages it would report
  signal that no real inference could use.
* If it scored spots with no observed neighbours it would mix a graph fact
  into a prediction quality number.
* If it could not separate genuine spatial structure from the marginal
  statistics of expression, a positive result would mean nothing -- hence the
  shuffled-coordinate control.
"""
import numpy as np
import pytest

from gen3_multiscale.scripts.diagnose_spatial_neighbor_signal import (
    diagnose_slide,
    neighbor_mean_prediction,
)


def _grid(side: int) -> np.ndarray:
    return np.stack(
        np.meshgrid(np.arange(side), np.arange(side)), axis=-1,
    ).reshape(-1, 2).astype(np.float64)


def _smooth_field(coords: np.ndarray, n_genes: int, seed: int = 0) -> np.ndarray:
    """Expression as a smooth function of position: neighbours are genuinely
    informative, so the diagnostic must detect strong signal here."""
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(size=(n_genes, 2)) * 0.5
    phases = rng.uniform(0, 2 * np.pi, size=n_genes)
    return np.sin(coords @ frequencies.T + phases).astype(np.float32)


def test_hidden_spots_never_enter_their_own_neighbour_average():
    """The leakage gate: changing the HIDDEN spots' expression must not change
    a single predicted value."""
    coords = _grid(6)
    expression = _smooth_field(coords, 5)
    observed = np.ones(coords.shape[0], dtype=bool)
    observed[[3, 7, 11, 19]] = False

    baseline, _ = neighbor_mean_prediction(expression, coords, observed, k_neighbors=4)
    tampered = expression.copy()
    tampered[~observed] += 1000.0
    perturbed, _ = neighbor_mean_prediction(tampered, coords, observed, k_neighbors=4)
    np.testing.assert_allclose(baseline, perturbed)


def test_prediction_is_the_plain_mean_of_observed_neighbours():
    coords = _grid(5)
    rng = np.random.default_rng(0)
    expression = rng.normal(size=(coords.shape[0], 3)).astype(np.float32)
    observed = np.ones(coords.shape[0], dtype=bool)
    observed[2] = False

    from gen3_multiscale.conditional_wae.spatial_refinement import padded_neighbor_graph

    indices, valid = padded_neighbor_graph(coords, 4)
    indices, valid = indices.numpy(), valid.numpy()
    prediction, has_neighbor = neighbor_mean_prediction(
        expression, coords, observed, k_neighbors=4,
    )
    for row in range(coords.shape[0]):
        neighbours = [j for j, ok in zip(indices[row], valid[row]) if ok and observed[j]]
        if not neighbours:
            assert not has_neighbor[row]
            continue
        np.testing.assert_allclose(
            prediction[row], expression[neighbours].astype(np.float64).mean(axis=0),
            rtol=1e-6, atol=1e-9,
        )


def test_spots_with_no_observed_neighbour_are_flagged_not_fabricated():
    coords = _grid(5)
    expression = _smooth_field(coords, 4)
    observed = np.zeros(coords.shape[0], dtype=bool)
    observed[0] = True
    _prediction, has_neighbor = neighbor_mean_prediction(
        expression, coords, observed, k_neighbors=4,
    )
    assert not has_neighbor.all(), "a spot with no observed neighbour must be flagged"
    assert has_neighbor.any(), "spot 0's neighbours should still see it"


def test_detects_strong_signal_in_a_spatially_smooth_field():
    coords = _grid(9)
    expression = _smooth_field(coords, 12)
    record = diagnose_slide(
        expression, coords, mask_fraction=0.25, k_neighbors=6, n_top_genes=5,
        rng=np.random.default_rng(0),
    )
    assert record["neighbor_mean"]["all_genes_pcc"] > 0.5
    # The control must not: a constant prediction scores 0 for every gene that
    # actually varies, which is the whole point of using pearson_per_gene.
    assert abs(record["slide_mean"]["all_genes_pcc"]) < 1e-9


def test_reports_no_signal_when_expression_is_spatially_random():
    """Same marginal distribution, no spatial relationship. Both the
    neighbour-mean and the shuffled control must land near zero, or the
    diagnostic would manufacture a positive result from nothing."""
    coords = _grid(9)
    rng = np.random.default_rng(1)
    expression = rng.normal(size=(coords.shape[0], 40)).astype(np.float32)
    record = diagnose_slide(
        expression, coords, mask_fraction=0.25, k_neighbors=6, n_top_genes=10,
        rng=np.random.default_rng(0),
    )
    assert abs(record["neighbor_mean"]["all_genes_pcc"]) < 0.2


def test_shuffling_coordinates_destroys_the_signal_it_measures():
    coords = _grid(9)
    expression = _smooth_field(coords, 12)
    record = diagnose_slide(
        expression, coords, mask_fraction=0.25, k_neighbors=6, n_top_genes=5,
        rng=np.random.default_rng(0),
    )
    assert (
        record["neighbor_mean"]["all_genes_pcc"]
        > record["shuffled_neighbor_mean"]["all_genes_pcc"] + 0.3
    ), "the spatial control did not separate space from expression's marginals"


def test_refuses_a_slide_with_too_few_scorable_spots():
    coords = _grid(2)
    expression = _smooth_field(coords, 3)
    with pytest.raises(ValueError, match="too few scorable hidden spots"):
        diagnose_slide(
            expression, coords, mask_fraction=0.25, k_neighbors=3, n_top_genes=2,
            rng=np.random.default_rng(0),
        )
