"""Stitch, crop, reproject and exaggerate elevation tiles into a heightmap.

Not implemented yet — see Task 2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_heightmap"]


def build_heightmap(
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int | None = None,
) -> np.ndarray:
    """Assemble a cropped, reprojected heightmap for a bounding box.

    Returns:
        A 2D float32 array of elevations in metres.
    """
    raise NotImplementedError("heightmap stitching is not implemented yet")
