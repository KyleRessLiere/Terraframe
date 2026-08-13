"""Turn a heightmap into a watertight, printable solid and write it as STL.

Not implemented yet — see Task 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["heightmap_to_mesh", "write_stl"]


def heightmap_to_mesh(heightmap: np.ndarray) -> object:
    """Build a watertight mesh (surface plus walls and base) from a heightmap."""
    raise NotImplementedError("mesh generation is not implemented yet")


def write_stl(mesh: object, path: str | Path) -> None:
    """Write a mesh to ``path`` as binary STL."""
    raise NotImplementedError("STL export is not implemented yet")
