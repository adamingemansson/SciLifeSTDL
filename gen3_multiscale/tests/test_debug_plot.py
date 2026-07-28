"""Phase 3 item 5: the mask/WSI/boundary debug artifact must actually run
end-to-end on real boundary_graph.py output and produce a file -- not a
visual regression test (no pixel comparison), just proof the plotting
code path executes cleanly against the real data shapes it will be fed."""
import numpy as np

from gen3_multiscale.data.boundary_graph import extract_boundary_and_local_context
from gen3_multiscale.evaluation.debug_plot import plot_mask_wsi_boundary_debug


def _square_grid(n=15, spacing=1.0):
    lo = -(n // 2)
    xs, ys = np.meshgrid(np.arange(lo, lo + n) * spacing, np.arange(lo, lo + n) * spacing)
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def test_plot_runs_end_to_end_on_real_boundary_extraction_output(tmp_path):
    grid = _square_grid(n=15)
    dist = np.linalg.norm(grid, axis=1)
    observed, query = grid[dist > 2.5], grid[dist <= 2.5]
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=6, max_rings=3)

    out = tmp_path / "debug.png"
    written = plot_mask_wsi_boundary_debug(
        out, observed_coords=observed, query_coords=query,
        boundary_idx=result.boundary_idx, boundary_ring=result.boundary_ring,
        title="test slide",
    )
    assert written == out
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_runs_with_wsi_tile_overlays(tmp_path):
    grid = _square_grid(n=15)
    dist = np.linalg.norm(grid, axis=1)
    observed, query = grid[dist > 2.5], grid[dist <= 2.5]
    result = extract_boundary_and_local_context(observed, query, k_neighbors=6, local_k=6, max_rings=3)

    retained = observed[:5]
    rejected = query[:3]  # stand-in tiles for "rejected" (near/inside the hole)
    out = tmp_path / "debug_with_tiles.png"
    written = plot_mask_wsi_boundary_debug(
        out, observed_coords=observed, query_coords=query,
        boundary_idx=result.boundary_idx, boundary_ring=result.boundary_ring,
        retained_tile_coords=retained, rejected_tile_coords=rejected, tile_size=1.0,
    )
    assert written.is_file() and written.stat().st_size > 0
