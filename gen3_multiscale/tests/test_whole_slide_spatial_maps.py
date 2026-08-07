import copy
from types import SimpleNamespace

import numpy as np
import pytest

from gen3_multiscale.conditional_wae import reference_projection as rp
from gen3_multiscale.conditional_wae.tensorboard import ConditionalWAETensorBoardLogger, _fixed_pca_rgb

GENES = ["G0", "G1", "G2", "G3"]


def _sample(seed, n=20):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        adata=SimpleNamespace(X=rng.normal(size=(n, len(GENES))).astype(np.float32), var_names=GENES),
    )


class FakeWriter:
    def __init__(self):
        self.figures = []
        self.scalars = []
        self.histograms = []

    def add_figure(self, tag, _fig, step, close=True):
        self.figures.append((tag, step))

    def add_scalar(self, *args):
        self.scalars.append(args)

    def add_histogram(self, tag, values, step):
        self.histograms.append((tag, step, np.asarray(values).shape))

    def flush(self):
        pass

    def close(self):
        pass


def _projection():
    samples = {"S0": _sample(0), "S1": _sample(1)}
    return rp.build_reference_gex_projection(
        samples, ["S0", "S1"], GENES, n_components=3, n_clusters=3, seed=0,
    )


def test_fixed_pca_rgb_ignores_outliers_and_uses_the_given_ranges():
    pc_ranges = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    coords = np.array([[5.0, 5.0, 5.0], [1000.0, -1000.0, 5.0]], dtype=np.float32)
    rgb = _fixed_pca_rgb(coords, pc_ranges)
    # Row 0 sits at the midpoint of every fixed range.
    np.testing.assert_allclose(rgb[0], [0.5, 0.5, 0.5], atol=1e-5)
    # Row 1's extreme outlier is clipped to the fixed range's edges, not
    # used to rescale the whole mapping the way percentile-based scaling would.
    np.testing.assert_allclose(rgb[1], [1.0, 0.0, 0.5], atol=1e-5)


def test_add_whole_slide_spatial_maps_logs_pc_composite_and_cluster_figures_for_both_arms(tmp_path):
    projection = _projection()
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    true_gex = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    predicted_gex = true_gex + 0.1

    logger.add_whole_slide_spatial_maps(
        10, "slideA", coords, true_gex, predicted_gex, GENES, projection,
    )
    tags = {tag for tag, _step in writer.figures}
    for label in ("true", "predicted"):
        for suffix in ("pc1", "pc2", "pc3", "pca_rgb_composite", "clusters"):
            assert f"whole_slide/slideA/{label}/{suffix}" in tags
    assert all(step == 10 for _tag, step in writer.figures)


def test_add_whole_slide_spatial_maps_logs_per_gene_figures_when_gene_indices_given(tmp_path):
    projection = _projection()
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    true_gex = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    predicted_gex = true_gex + 0.1

    logger.add_whole_slide_spatial_maps(
        10, "slideA", coords, true_gex, predicted_gex, GENES, projection, gene_indices=[0, 2],
    )
    tags = {tag for tag, _step in writer.figures}
    assert "whole_slide/slideA/genes/G0" in tags
    assert "whole_slide/slideA/genes/G2" in tags
    assert "whole_slide/slideA/genes/G1" not in tags


def test_add_whole_slide_spatial_maps_logs_a_real_pcc_scalar_per_gene(tmp_path):
    projection = _projection()
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    true_gex = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    predicted_gex = true_gex + 0.1  # a fixed offset -- perfect correlation, PCC == 1

    logger.add_whole_slide_spatial_maps(
        10, "slideA", coords, true_gex, predicted_gex, GENES, projection, gene_indices=[0, 2],
    )
    scalars_by_tag = {tag: (value, step) for tag, value, step in writer.scalars}
    assert "whole_slide/slideA/genes/G0/pcc" in scalars_by_tag
    assert "whole_slide/slideA/genes/G2/pcc" in scalars_by_tag
    assert "whole_slide/slideA/genes/G1/pcc" not in scalars_by_tag  # G1 wasn't in gene_indices
    value, step = scalars_by_tag["whole_slide/slideA/genes/G0/pcc"]
    assert value == pytest.approx(1.0, abs=1e-4)
    assert step == 10


def test_add_whole_slide_spatial_maps_skips_pcc_for_a_constant_truth_gene(tmp_path):
    """Matches evaluation.metrics.pearson_per_gene's convention: a
    truth-constant gene has undefined correlation and must not silently
    log a fabricated value (e.g. 0 or NaN plotted as if real)."""
    projection = _projection()
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    true_gex = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    true_gex[:, 0] = 3.0  # G0 is constant across every spot
    predicted_gex = true_gex + 0.1

    logger.add_whole_slide_spatial_maps(
        10, "slideA", coords, true_gex, predicted_gex, GENES, projection, gene_indices=[0],
    )
    tags = {tag for tag, *_rest in writer.scalars}
    assert "whole_slide/slideA/genes/G0/pcc" not in tags


def test_add_whole_slide_spatial_maps_omits_gene_figures_by_default(tmp_path):
    projection = _projection()
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    true_gex = np.random.default_rng(2).normal(size=(6, 4)).astype(np.float32)
    predicted_gex = true_gex + 0.1

    logger.add_whole_slide_spatial_maps(10, "slideA", coords, true_gex, predicted_gex, GENES, projection)
    tags = {tag for tag, _step in writer.figures}
    assert not any("/genes/" in tag for tag in tags)


def test_add_whole_slide_spatial_maps_never_mutates_the_frozen_projection(tmp_path):
    projection = _projection()
    original = copy.deepcopy({
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in projection.items()
    })
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    coords = np.stack([np.arange(6), np.arange(6)], axis=1).astype(np.float32)
    extreme_prediction = np.full((6, 4), 1e6, dtype=np.float32)
    true_gex = np.random.default_rng(3).normal(size=(6, 4)).astype(np.float32)

    logger.add_whole_slide_spatial_maps(
        1, "slideA", coords, true_gex, extreme_prediction, GENES, projection,
    )
    np.testing.assert_array_equal(projection["pc_ranges"], original["pc_ranges"])
    np.testing.assert_array_equal(projection["components"], original["components"])
    np.testing.assert_array_equal(projection["mean"], original["mean"])
    np.testing.assert_array_equal(projection["cluster_centroids"], original["cluster_centroids"])


def test_add_film_diagnostics_logs_scalars_and_histograms(tmp_path):
    writer = FakeWriter()
    logger = ConditionalWAETensorBoardLogger(tmp_path, writer=writer)
    diagnostics = {
        "effective_rank": 3.2,
        "active_dimensions": 4,
        "predictive_std_mean": 0.5,
        "stochastic_vs_conditional_mean_diff": 0.8,
        "latent_dim_mean": np.zeros(5),
        "latent_dim_std": np.ones(5),
        "gamma_beta": {
            "first": (np.ones((6, 8)), np.zeros((6, 8))),
            "second": (np.ones((6, 8)) * 2, np.zeros((6, 8))),
        },
    }
    logger.add_film_diagnostics(7, diagnostics)
    scalar_tags = {tag for tag, *_rest in writer.scalars}
    assert scalar_tags >= {
        "film/posterior_effective_rank", "film/posterior_active_dimensions",
        "film/predictive_std_mean", "film/stochastic_vs_conditional_mean_diff",
    }
    histogram_tags = {tag for tag, _step, _shape in writer.histograms}
    assert histogram_tags == {
        "film/latent_dim_mean", "film/latent_dim_std",
        "film/gamma_first", "film/beta_first", "film/gamma_second", "film/beta_second",
    }
    assert all(step == 7 for _tag, step, _shape in writer.histograms)
