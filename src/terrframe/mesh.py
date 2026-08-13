"""Turn a heightmap into a watertight, printable solid and write it out.

The solid is built in one pass from a single vertex array and a single face
array: a triangulated terrain surface on top, four vertical skirt walls down
to ``z = 0``, and a flat bottom. Watertightness comes from construction --
walls reuse the terrain's own edge vertices, so there is nothing to weld and
no boolean union or mesh repair anywhere in this module.

Orientation: ``+x`` is east, ``+y`` is north, ``+z`` is up. Every face is
wound counter-clockwise seen from outside, so normals point out of the solid
and the reported volume is positive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import zoom as ndimage_zoom

from .heightmap import Heightmap

__all__ = [
    "AUTO_EXAGGERATION_MAX",
    "AUTO_EXAGGERATION_MIN",
    "TARGET_RELIEF_RATIO",
    "auto_exaggeration",
    "export",
    "heightmap_to_mesh",
]

# --- Tuning constants -------------------------------------------------------
# These are deliberate, hand-tunable style choices, not derived quantities.

#: Printed relief (terrain only, excluding the base) that ``auto_exaggeration``
#: aims for, as a fraction of the print's width. Around 18% reads as dramatic
#: without looking like a spike field.
TARGET_RELIEF_RATIO = 0.18

#: Never de-emphasise real terrain: already-dramatic country prints as-is.
AUTO_EXAGGERATION_MIN = 1.0

#: Beyond this, plains stop looking like terrain and start looking like noise.
AUTO_EXAGGERATION_MAX = 12.5

#: Supported output containers, mapped to trimesh's exporter names.
_EXPORT_FORMATS = {".stl": "stl", ".3mf": "3mf"}


def auto_exaggeration(relief_m: float, width_m: float) -> float:
    """Suggest a vertical exaggeration from terrain relief and bbox width.

    Aims to land the printed relief at :data:`TARGET_RELIEF_RATIO` of the
    print's width, so flat country gets pushed up and alpine country is left
    close to alone.

    Args:
        relief_m: Elevation range of the terrain, in metres.
        width_m: Ground width of the bounding box, in metres.

    Returns:
        A factor clamped to ``[AUTO_EXAGGERATION_MIN, AUTO_EXAGGERATION_MAX]``.
        Terrain with no relief at all gets the maximum.
    """
    if width_m <= 0.0:
        raise ValueError(f"width_m must be positive, got {width_m}")
    if relief_m <= 0.0:
        # Perfectly flat ground has no relief to scale; nothing to lose by
        # asking for the most.
        return AUTO_EXAGGERATION_MAX

    ideal = TARGET_RELIEF_RATIO * width_m / relief_m
    return float(np.clip(ideal, AUTO_EXAGGERATION_MIN, AUTO_EXAGGERATION_MAX))


def _downsample(elevation: np.ndarray, max_vertices: int) -> np.ndarray:
    """Shrink a heightmap until its mesh fits inside ``max_vertices``.

    Slicers bog down past a few million vertices and no FDM printer resolves
    them anyway, so this trades detail nobody can print for a usable file.
    """
    rows, cols = elevation.shape
    if _vertex_count(rows, cols) <= max_vertices:
        return elevation

    # Budget is dominated by the rows*cols terrain grid; solve for the linear
    # scale that fits, then verify and step down if rounding overshoots.
    scale = float(np.sqrt(max_vertices / (rows * cols)))
    for _ in range(64):
        new_rows = max(2, int(rows * scale))
        new_cols = max(2, int(cols * scale))
        if _vertex_count(new_rows, new_cols) <= max_vertices or (new_rows, new_cols) == (2, 2):
            break
        scale *= 0.98
    else:  # pragma: no cover - the loop above converges long before this
        raise RuntimeError("could not shrink the heightmap under max_vertices")

    zoom_factors = (new_rows / rows, new_cols / cols)
    return ndimage_zoom(elevation.astype(np.float64), zoom_factors, order=1).astype(np.float32)


def _vertex_count(rows: int, cols: int) -> int:
    """Total vertices in the finished solid for an ``rows x cols`` terrain grid."""
    perimeter = 2 * rows + 2 * cols - 4
    return rows * cols + perimeter + 1  # terrain + bottom ring + bottom centre


def _perimeter_indices(rows: int, cols: int) -> np.ndarray:
    """Terrain-grid vertex indices around the boundary, CCW seen from above.

    Order matters: the walls and the bottom fan both rely on it to get their
    winding right. Starts at the south-west corner and runs east, north, west,
    then south, with no corner repeated.
    """

    def flat(i: int, j: int) -> int:
        return i * cols + j

    south = [flat(rows - 1, j) for j in range(cols)]
    east = [flat(i, cols - 1) for i in range(rows - 2, -1, -1)]
    north = [flat(0, j) for j in range(cols - 2, -1, -1)]
    west = [flat(i, 0) for i in range(1, rows - 1)]
    return np.array(south + east + north + west, dtype=np.int64)


def _terrain_faces(rows: int, cols: int) -> np.ndarray:
    """Triangulate the terrain grid with upward-facing normals."""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    top_left = idx[:-1, :-1].ravel()
    top_right = idx[:-1, 1:].ravel()
    bottom_left = idx[1:, :-1].ravel()
    bottom_right = idx[1:, 1:].ravel()

    # Row index grows southward, so "bottom_*" is the lower-y pair. This
    # ordering is CCW seen from +z, giving outward (upward) normals.
    return np.concatenate(
        [
            np.stack([top_left, bottom_left, bottom_right], axis=1),
            np.stack([top_left, bottom_right, top_right], axis=1),
        ]
    )


def _wall_faces(perimeter: np.ndarray, ring_start: int) -> np.ndarray:
    """Stitch the terrain boundary down to the bottom ring.

    Args:
        perimeter: Terrain vertex indices, CCW from above.
        ring_start: Index of the first bottom-ring vertex; the ring runs in
            the same order as ``perimeter``.

    Returns:
        Triangles wound so their normals point away from the solid.
    """
    count = len(perimeter)
    nxt = np.roll(np.arange(count), -1)

    top_a = perimeter
    top_b = perimeter[nxt]
    bot_a = ring_start + np.arange(count, dtype=np.int64)
    bot_b = ring_start + nxt.astype(np.int64)

    return np.concatenate(
        [
            np.stack([top_a, bot_a, bot_b], axis=1),
            np.stack([top_a, bot_b, top_b], axis=1),
        ]
    )


def _bottom_faces(count: int, ring_start: int, centre: int) -> np.ndarray:
    """Fan the flat bottom from a centre vertex, with downward normals.

    The fan runs from a centre point rather than a corner because the ring has
    many collinear vertices along each side; a corner fan would emit zero-area
    triangles down every edge it sits on.
    """
    ring = ring_start + np.arange(count, dtype=np.int64)
    nxt = ring_start + np.roll(np.arange(count), -1).astype(np.int64)
    centres = np.full(count, centre, dtype=np.int64)
    # Reversed relative to the CCW-from-above ring, so normals point at -z.
    return np.stack([centres, nxt, ring], axis=1)


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    """Number of edges used by exactly one face -- zero for a closed solid."""
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return int((counts == 1).sum())


def heightmap_to_mesh(
    hm: Heightmap,
    width_mm: float = 200.0,
    base_mm: float = 6.0,
    max_vertices: int = 4_000_000,
) -> trimesh.Trimesh:
    """Build a watertight printable solid from a heightmap.

    The print is ``width_mm`` across in x; depth follows the heightmap's aspect
    ratio, and the vertical scale is derived from ``hm.meters_per_px`` so the
    result is geometrically truthful -- including whatever vertical
    exaggeration was already baked in upstream.

    The terrain's lowest point sits at ``z = base_mm``, so there is always
    solid material under the deepest valley, and the bottom is flat at
    ``z = 0``.

    Args:
        hm: The heightmap to model.
        width_mm: Printed width (x extent) in millimetres.
        base_mm: Solid base thickness under the lowest terrain point.
        max_vertices: Vertex ceiling; the heightmap is downsampled to fit.

    Returns:
        A watertight, consistently wound :class:`trimesh.Trimesh`.

    Raises:
        ValueError: The heightmap is too small to mesh, or the parameters are
            not positive.
        RuntimeError: The assembled mesh is not watertight or not consistently
            wound. Includes the open-edge count for diagnosis.
    """
    if width_mm <= 0.0:
        raise ValueError(f"width_mm must be positive, got {width_mm}")
    if base_mm < 0.0:
        raise ValueError(f"base_mm must be non-negative, got {base_mm}")
    if max_vertices < 16:
        raise ValueError(f"max_vertices must leave room for a mesh, got {max_vertices}")

    source = np.asarray(hm.elevation, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"expected a 2D elevation array, got shape {source.shape}")
    if source.shape[0] < 2 or source.shape[1] < 2:
        raise ValueError(f"need at least a 2x2 heightmap to mesh, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("elevation contains NaN or infinity; run fill_nodata first")

    # Ground width is fixed by the *original* grid, so the vertical scale below
    # stays truthful no matter how much the grid is decimated afterwards.
    ground_width_m = hm.meters_per_px * (source.shape[1] - 1)
    if ground_width_m <= 0.0:
        raise ValueError("heightmap has zero ground width")

    elevation = _downsample(source, max_vertices)
    rows, cols = elevation.shape

    pitch_mm = width_mm / (cols - 1)
    # Millimetres of model per metre of ground, applied to both axes equally.
    z_scale = width_mm / ground_width_m

    floor_m = float(elevation.min())
    heights = base_mm + (elevation.astype(np.float64) - floor_m) * z_scale

    # --- Vertices ---------------------------------------------------------
    xs = np.arange(cols, dtype=np.float64) * pitch_mm
    # Row 0 is the northern edge, so it takes the largest y.
    ys = (rows - 1 - np.arange(rows, dtype=np.float64)) * pitch_mm
    grid_x, grid_y = np.meshgrid(xs, ys)

    terrain = np.stack([grid_x.ravel(), grid_y.ravel(), heights.ravel()], axis=1)

    perimeter = _perimeter_indices(rows, cols)
    ring = terrain[perimeter].copy()
    ring[:, 2] = 0.0

    centre = np.array([[xs.mean(), ys.mean(), 0.0]])

    vertices = np.concatenate([terrain, ring, centre])

    # --- Faces ------------------------------------------------------------
    ring_start = len(terrain)
    centre_index = ring_start + len(ring)

    faces = np.concatenate(
        [
            _terrain_faces(rows, cols),
            _wall_faces(perimeter, ring_start),
            _bottom_faces(len(perimeter), ring_start, centre_index),
        ]
    )

    # process=False keeps exactly the topology built here: no welding, no
    # dropped faces, nothing silently "fixed" behind the assertions below.
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(
            "constructed mesh is not a closed solid "
            f"(watertight={mesh.is_watertight}, "
            f"winding_consistent={mesh.is_winding_consistent}, "
            f"open_edges={_boundary_edge_count(mesh)}, "
            f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
            f"grid={rows}x{cols})"
        )

    return mesh


def export(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write a mesh to disk as binary STL or 3MF, chosen by file extension.

    Args:
        mesh: The mesh to write.
        path: Destination ending in ``.stl`` or ``.3mf``.

    Returns:
        The path written.

    Raises:
        ValueError: The extension is not a supported output format.
    """
    destination = Path(path)
    suffix = destination.suffix.lower()

    if suffix not in _EXPORT_FORMATS:
        supported = ", ".join(sorted(_EXPORT_FORMATS))
        raise ValueError(f"unsupported output format {suffix!r}; expected one of {supported}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    data = mesh.export(file_type=_EXPORT_FORMATS[suffix])
    if isinstance(data, str):  # pragma: no cover - binary STL/3MF return bytes
        data = data.encode()
    destination.write_bytes(data)
    return destination
