"""Phase 3 (multiscale spatial-field handoff): FrozenGigaPathSlideEncoder's
fail-closed checkpoint validation, and pool_regional_tokens' spatial
pooling into fixed, slide-stable grid cells."""
import numpy as np
import pytest
import torch

from gen3_multiscale.models.slide_encoder import FrozenGigaPathSlideEncoder, pool_regional_tokens


def test_frozen_gigapath_slide_encoder_fails_closed_on_a_missing_checkpoint(tmp_path):
    """The handoff and the class's own docstring are explicit: a missing
    checkpoint must never fall through to a randomly-initialized LongNet
    that could silently invalidate an experiment while still running."""
    missing = tmp_path / "does_not_exist.pth"
    with pytest.raises(FileNotFoundError, match="GigaPath slide checkpoint not found"):
        FrozenGigaPathSlideEncoder(checkpoint_path=str(missing))


def test_pool_regional_tokens_assigns_tiles_to_the_correct_grid_cell():
    # a 2x2 grid over a 0..100 x 0..100 slide -- one tile per quadrant
    features = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
    coords = np.array([[10, 10], [90, 10], [10, 90], [90, 90]], dtype=np.float32)
    tokens, available = pool_regional_tokens(features, coords, full_slide_coord_bounds=(0, 100, 0, 100), grid_size=2)
    assert tokens.shape == (4, 2)
    assert available.all()
    # row=0 (y<50): col=0 (x<50) -> cell 0 = tile [1,1]; col=1 -> cell 1 = tile [2,2]
    # row=1 (y>=50): col=0 -> cell 2 = tile [3,3]; col=1 -> cell 3 = tile [4,4]
    assert torch.allclose(tokens[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(tokens[1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(tokens[2], torch.tensor([3.0, 3.0]))
    assert torch.allclose(tokens[3], torch.tensor([4.0, 4.0]))


def test_pool_regional_tokens_averages_multiple_tiles_in_the_same_cell():
    features = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    coords = np.array([[5, 5], [15, 15]], dtype=np.float32)  # both in the same cell of a 1x1 grid
    tokens, available = pool_regional_tokens(features, coords, full_slide_coord_bounds=(0, 20, 0, 20), grid_size=1)
    assert torch.allclose(tokens[0], torch.tensor([1.0, 1.0]))
    assert available[0]


def test_pool_regional_tokens_marks_empty_cells_unavailable_not_a_fabricated_zero():
    """A hole-covered region should be visibly "no data", not silently
    indistinguishable from a real all-zero-content region."""
    features = np.array([[5.0, 5.0]], dtype=np.float32)
    coords = np.array([[10, 10]], dtype=np.float32)  # only cell (0,0) of a 2x2 grid has a tile
    tokens, available = pool_regional_tokens(features, coords, full_slide_coord_bounds=(0, 100, 0, 100), grid_size=2)
    assert available[0] and not available[1] and not available[2] and not available[3]
    assert torch.allclose(tokens[1], torch.zeros(2))


def test_pool_regional_tokens_handles_zero_visible_tiles():
    features = np.zeros((0, 8), dtype=np.float32)
    coords = np.zeros((0, 2), dtype=np.float32)
    tokens, available = pool_regional_tokens(features, coords, full_slide_coord_bounds=(0, 10, 0, 10), grid_size=3)
    assert tokens.shape == (9, 8)
    assert not available.any()


def test_pool_regional_tokens_grid_is_stable_regardless_of_which_tiles_are_visible():
    """The same physical location must map to the same grid cell whether
    or not a hole happens to remove nearby tiles for a given training
    item -- computed from full_slide_coord_bounds, not the visible
    subset's own (hole-shrunk) extent."""
    full_bounds = (0, 100, 0, 100)
    features_a = np.array([[9.0, 9.0]], dtype=np.float32)
    coords_a = np.array([[95, 95]], dtype=np.float32)
    tokens_a, _ = pool_regional_tokens(features_a, coords_a, full_slide_coord_bounds=full_bounds, grid_size=4)

    # same physical tile, but now other tiles near the slide's edge have
    # been removed by a hole -- the bounds passed in stay the FULL-slide
    # bounds, so this tile must land in the identical cell as above.
    features_b = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    coords_b = np.array([[50, 50], [95, 95]], dtype=np.float32)
    tokens_b, _ = pool_regional_tokens(features_b, coords_b, full_slide_coord_bounds=full_bounds, grid_size=4)

    cell_a = torch.nonzero(tokens_a.any(dim=1)).flatten()
    assert torch.allclose(tokens_b[cell_a], torch.tensor([[9.0, 9.0]]))


def test_pool_regional_tokens_rejects_invalid_bounds():
    features = np.zeros((1, 4), dtype=np.float32)
    coords = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="full_slide_coord_bounds"):
        pool_regional_tokens(features, coords, full_slide_coord_bounds=(10, 10, 0, 10), grid_size=2)
