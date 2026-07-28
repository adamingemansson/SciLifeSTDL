"""Phase 3 item 5 of the multiscale spatial-field handoff: "Add a
visual/debug artifact showing the WSI, query polygon/spots, rejected
tiles, retained tiles, ST spots, and selected boundary rings."

Not a metric, not consumed by training -- a human-auditable sanity check
that the mask-aware WSI tile filtering and boundary-ring extraction
(boundary_graph.py, Phase 2) actually agree with each other on a real or
synthetic example, before any of it is trusted inside a model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_mask_wsi_boundary_debug(
    output_path: str | Path,
    observed_coords: np.ndarray,
    query_coords: np.ndarray,
    boundary_idx: np.ndarray,
    boundary_ring: np.ndarray,
    retained_tile_coords: np.ndarray | None = None,
    rejected_tile_coords: np.ndarray | None = None,
    tile_size: float | None = None,
    title: str | None = None,
) -> Path:
    """Render one PNG showing, in the same coordinate frame:

    - every observed ST spot (light gray);
    - every query (hole) ST spot (red);
    - the boundary-ring spots, colored by ring (Ring 1 darkest, Ring 3
      lightest) -- lets a reviewer visually confirm Rings 1-3 actually
      trace the hole's true rim rather than something disconnected from
      it;
    - retained WSI tiles (green squares, drawn at tile_size if given) --
      the tiles GigaPath LongNet will actually see;
    - rejected WSI tiles (orange squares) -- tiles whose footprint
      overlapped the hole and were correctly excluded.

    retained_tile_coords/rejected_tile_coords/tile_size are all optional
    so this also works for boundary-only debugging (Phase 2, no WSI tiles
    involved yet) without a separate code path.

    Returns the written path. Import matplotlib lazily so this module
    (and everything that transitively imports it) stays importable in
    headless/CI environments that never call this function.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    if retained_tile_coords is not None and retained_tile_coords.shape[0] and tile_size:
        for x, y in retained_tile_coords[:, :2]:
            ax.add_patch(Rectangle(
                (x - tile_size / 2, y - tile_size / 2), tile_size, tile_size,
                facecolor="tab:green", edgecolor="none", alpha=0.15, zorder=0,
            ))
    if rejected_tile_coords is not None and rejected_tile_coords.shape[0] and tile_size:
        for x, y in rejected_tile_coords[:, :2]:
            ax.add_patch(Rectangle(
                (x - tile_size / 2, y - tile_size / 2), tile_size, tile_size,
                facecolor="tab:orange", edgecolor="none", alpha=0.25, zorder=0,
            ))

    ax.scatter(
        observed_coords[:, 0], observed_coords[:, 1],
        s=6, c="lightgray", label="observed ST spots", zorder=1,
    )

    ring_colors = {1: "#08306b", 2: "#4292c6", 3: "#9ecae1"}
    for ring, color in ring_colors.items():
        in_ring = boundary_ring == ring
        if in_ring.any():
            pts = observed_coords[boundary_idx[in_ring]]
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c=color, label=f"boundary ring {ring}", zorder=3)

    ax.scatter(
        query_coords[:, 0], query_coords[:, 1],
        s=10, c="red", label="query (hole) ST spots", zorder=2,
    )

    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.5)
    ax.set_title(title or "Mask / WSI / boundary-ring debug view")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
