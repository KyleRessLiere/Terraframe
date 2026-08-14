"""Turn a bounding box into a single clean, print-ready elevation array.

The pipeline is: pick a zoom, fetch the covering tiles (with a one-tile
margin), stitch them into one raster, crop to the requested bbox, resample
onto a grid whose pixels are square in real-world metres, fill nodata holes,
flatten water, and apply vertical exaggeration.

A note on projections, because it drives :func:`resample_to_meters`.
Terrarium tiles are Web Mercator, which is *conformal*: it already stretches
the y axis by exactly ``1 / cos(lat)``, so a raw tile crop's pixels are
already locally square on the ground. The correction that is genuinely
needed is not per-axis (that would double-apply the stretch and squash the
result north-south) but the *drift* of scale with latitude across a tall
bbox, which Web Mercator does not handle at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from affine import Affine
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label,
    map_coordinates,
    median_filter,
)

from .tiles import (
    TILE_SIZE,
    fetch_tile,
    mercator_x_norm,
    mercator_y_norm,
    tiles_for_bbox,
    zoom_for_bbox,
)

__all__ = [
    "DESPIKE_THRESHOLD",
    "DESPIKE_WINDOW_PX",
    "EARTH_RADIUS_M",
    "MAX_SPIKE_CLUSTER_PX",
    "SMOOTH_GROUND_METERS",
    "SMOOTH_SIGMA_MAX",
    "SMOOTH_SIGMA_MIN",
    "Heightmap",
    "auto_smooth_sigma",
    "bounds_to_bbox",
    "build_heightmap",
    "crop_bounds",
    "crop_to_bbox",
    "despike",
    "exaggerate",
    "fill_nodata",
    "flatten_water",
    "ground_extent",
    "meters_per_pixel",
    "resample_to_meters",
    "smooth",
    "stitch",
    "tile_origin",
]

#: Web Mercator's sphere radius (metres). Matches the projection the tiles use.
EARTH_RADIUS_M = 6378137.0

#: Terrarium's encoding floor. A pixel decoding to this is missing, not -32768 m deep.
NODATA_FLOOR_M = -32000.0

#: Pixel-boundary tolerance used when rounding crop windows outward.
_PIXEL_EPS = 1e-6

# --- Cleanup tuning ---------------------------------------------------------
# Hand-tunable style constants, kept together rather than buried in the calls.

#: Neighbourhood the spike test compares each pixel against.
DESPIKE_WINDOW_PX = 5

#: Deviation from the local median, in interquartile ranges, that counts as a
#: spike. Lower removes more.
DESPIKE_THRESHOLD = 3.0

#: Largest connected blob of flagged pixels still treated as a spike. Above
#: this it is assumed to be real terrain -- a ridgeline, not a needle.
MAX_SPIKE_CLUSTER_PX = 4

#: Ground radius the automatic smoothing aims for, in metres. Roughly the
#: scale of the trees and buildings that need to come off.
SMOOTH_GROUND_METERS = 40.0

#: Reference print width the millimetre-based constants assume, matching the
#: CLI's default and :func:`terrframe.mesh.heightmap_to_mesh`.
PRINT_WIDTH_MM = 200.0

#: Ceiling on that blur expressed in printed millimetres.
#:
#: Clutter removal is a *ground*-scale job -- a tree is a tree regardless of
#: framing -- but legibility is a *print*-scale one, and the print is always
#: the same width. On a 28 km bbox 40 m of ground is 0.29 mm of print, which is
#: nothing; on a 6 km bbox it is 1.32 mm, which erases the street grid, the
#: Mall and the terraced riverbanks. Tight framings were being blurred roughly
#: 4.5x harder than wide ones for no reason anyone chose.
#:
#: So the ground target still drives the blur, capped so it can never eat more
#: than this much of the printed surface.
SMOOTH_PRINT_MM_MAX = 0.5

#: Sigma floor: below this, blurring does nothing useful.
SMOOTH_SIGMA_MIN = 0.5

#: Sigma ceiling: above this, real landforms start dissolving.
SMOOTH_SIGMA_MAX = 6.0


@dataclass(frozen=True)
class Heightmap:
    """A finished elevation grid plus the georeferencing a mesh needs.

    Attributes:
        elevation: 2D float32 array of metres, row 0 at the northern edge.
        meters_per_px: Ground distance between adjacent pixels, both axes.
        bbox: ``(south, west, north, east)`` the array actually covers. Cropping
            rounds outward, so this is a superset of what was requested.
        zoom: Source tile zoom level.
        exaggeration: Vertical factor already baked into ``elevation``.
        requested_bbox: The ``(south, west, north, east)`` originally asked for.
        water_mask: Pixels stamped as water from OSM polygons, when features
            were applied. Renderers should prefer this over guessing water from
            flatness -- a city's water is many small polygons, each too small to
            clear an area threshold on its own.
    """

    elevation: np.ndarray
    meters_per_px: float
    bbox: tuple[float, float, float, float]
    zoom: int
    exaggeration: float = 1.0
    requested_bbox: tuple[float, float, float, float] | None = field(default=None)
    water_mask: np.ndarray | None = field(default=None)

    @property
    def shape(self) -> tuple[int, int]:
        """The elevation grid's ``(rows, cols)``."""
        return self.elevation.shape  # type: ignore[return-value]

    @property
    def transform(self) -> Affine:
        """Affine mapping pixel coordinates to lon/lat.

        The grid is equidistant in latitude and longitude across ``bbox`` (see
        :func:`resample_to_meters`), so a plain north-up affine describes it
        exactly. Rasterising OSM geometry against the grid needs this.
        """
        south, west, north, east = self.bbox
        rows, cols = self.elevation.shape
        return Affine(
            (east - west) / cols,
            0.0,
            west,
            0.0,
            -(north - south) / rows,
            north,
        )

    @property
    def size_meters(self) -> tuple[float, float]:
        """Printed ground footprint as ``(width_m, height_m)``."""
        rows, cols = self.elevation.shape
        return cols * self.meters_per_px, rows * self.meters_per_px


def _validate_bbox(south: float, west: float, north: float, east: float) -> None:
    if south > north:
        raise ValueError(f"south ({south}) must be <= north ({north})")
    if west > east:
        raise ValueError(f"west ({west}) must be <= east ({east})")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def meters_per_pixel(zoom: int, lat: float) -> float:
    """Ground distance covered by one Web Mercator pixel at ``lat``.

    Args:
        zoom: Tile zoom level.
        lat: Latitude in degrees. Resolution gets finer toward the poles.

    Returns:
        Metres per pixel along both axes (Web Mercator is locally isotropic).
    """
    world_px = TILE_SIZE * (1 << zoom)
    return (2.0 * math.pi * EARTH_RADIUS_M * math.cos(math.radians(lat))) / world_px


def ground_extent(
    south: float,
    west: float,
    north: float,
    east: float,
) -> tuple[float, float]:
    """Real-world size of a bbox as ``(width_m, height_m)``.

    Width is measured along the parallel through the bbox's centre latitude,
    height along a meridian. Uses Web Mercator's sphere, so this sits within
    about 0.3% of a WGS84 geodesic.

    Args:
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.

    Returns:
        East-west and north-south spans in metres.
    """
    _validate_bbox(south, west, north, east)
    center_lat = (south + north) / 2.0
    width = EARTH_RADIUS_M * math.radians(east - west) * math.cos(math.radians(center_lat))
    height = EARTH_RADIUS_M * math.radians(north - south)
    return width, height


# ---------------------------------------------------------------------------
# Stitch
# ---------------------------------------------------------------------------


def stitch(tiles_dict: dict[tuple[int, int], np.ndarray], zoom: int) -> np.ndarray:
    """Assemble fetched tiles into one raster.

    Pure: the grid extent is derived from the dict's own keys, so this never
    touches the network and needs no extra bookkeeping from the caller. Use
    :func:`tile_origin` to recover the north-west tile for cropping.

    Args:
        tiles_dict: Maps ``(x, y)`` tile coordinates to equally-shaped 2D arrays.
        zoom: Zoom the tiles came from; used to validate the coordinates.

    Returns:
        One 2D array, north-west tile at the top-left.

    Raises:
        ValueError: The tiles do not form a complete rectangular grid, have
            mismatched shapes, or fall outside the grid for ``zoom``.
    """
    if not tiles_dict:
        raise ValueError("tiles_dict is empty; nothing to stitch")

    n = 1 << zoom
    for x, y in tiles_dict:
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"tile ({x}, {y}) is outside the 0..{n - 1} grid for zoom {zoom}")

    xs = sorted({x for x, _ in tiles_dict})
    ys = sorted({y for _, y in tiles_dict})

    expected = len(xs) * len(ys)
    if len(tiles_dict) != expected:
        raise ValueError(
            f"tiles do not form a complete grid: got {len(tiles_dict)} tiles, "
            f"need {expected} for a {len(xs)}x{len(ys)} rectangle"
        )
    if xs != list(range(xs[0], xs[0] + len(xs))) or ys != list(range(ys[0], ys[0] + len(ys))):
        raise ValueError("tile coordinates are not contiguous")

    shapes = {arr.shape for arr in tiles_dict.values()}
    if len(shapes) != 1:
        raise ValueError(f"all tiles must share one shape, got {sorted(shapes)}")

    # Row-major: y outer (north to south), x inner (west to east) -- the order
    # tiles_for_bbox already returns, and the order np.block wants.
    rows = [[tiles_dict[(x, y)] for x in xs] for y in ys]
    return np.block(rows).astype(np.float32, copy=False)


def tile_origin(tiles_dict: dict[tuple[int, int], np.ndarray]) -> tuple[int, int]:
    """Return the ``(x, y)`` of the north-west tile in a stitched grid."""
    if not tiles_dict:
        raise ValueError("tiles_dict is empty; no origin")
    return min(x for x, _ in tiles_dict), min(y for _, y in tiles_dict)


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


def crop_bounds(
    tile_origin: tuple[int, int],
    zoom: int,
    south: float,
    west: float,
    north: float,
    east: float,
) -> tuple[int, int, int, int]:
    """Pixel window of a bbox within a stitched grid, rounded outward.

    Sub-pixel edges always expand: losing requested area is worse than
    including a few extra metres.

    Args:
        tile_origin: ``(x, y)`` of the stitched grid's north-west tile.
        zoom: Zoom the grid was built at.
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.

    Returns:
        ``(col_start, row_start, col_stop, row_stop)`` as a half-open window.
    """
    _validate_bbox(south, west, north, east)

    world_px = TILE_SIZE * (1 << zoom)
    origin_col = tile_origin[0] * TILE_SIZE
    origin_row = tile_origin[1] * TILE_SIZE

    left = mercator_x_norm(west) * world_px - origin_col
    right = mercator_x_norm(east) * world_px - origin_col
    # North is the smaller Mercator y, so it maps to the smaller row.
    top = mercator_y_norm(north) * world_px - origin_row
    bottom = mercator_y_norm(south) * world_px - origin_row

    # Nudge by an epsilon before rounding so an edge that is mathematically on
    # a pixel boundary is not pushed out a whole pixel by float noise in the
    # lat/lon -> pixel round trip. At 1e-6 px this cannot mask a real sub-pixel
    # offset (that would be sub-nanometre on the ground).
    col_start = math.floor(left + _PIXEL_EPS)
    col_stop = math.ceil(right - _PIXEL_EPS)
    row_start = math.floor(top + _PIXEL_EPS)
    row_stop = math.ceil(bottom - _PIXEL_EPS)

    # A bbox thinner than a pixel still has to yield one.
    col_stop = max(col_stop, col_start + 1)
    row_stop = max(row_stop, row_start + 1)

    return col_start, row_start, col_stop, row_stop


def crop_to_bbox(
    stitched: np.ndarray,
    tile_origin: tuple[int, int],
    zoom: int,
    south: float,
    west: float,
    north: float,
    east: float,
) -> np.ndarray:
    """Cut a stitched grid down to the requested bbox.

    Sub-pixel edges round outward, so the result covers at least the bbox.
    Use :func:`bounds_to_bbox` on the same window to learn what it actually
    covers.

    Args:
        stitched: Raster produced by :func:`stitch`.
        tile_origin: ``(x, y)`` of that raster's north-west tile.
        zoom: Zoom the raster was built at.
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.

    Returns:
        A cropped view-shaped copy covering the bbox.

    Raises:
        ValueError: The bbox does not overlap the stitched grid.
    """
    col_start, row_start, col_stop, row_stop = crop_bounds(
        tile_origin, zoom, south, west, north, east
    )

    rows, cols = stitched.shape[:2]
    c0, c1 = max(0, col_start), min(cols, col_stop)
    r0, r1 = max(0, row_start), min(rows, row_stop)

    if c0 >= c1 or r0 >= r1:
        raise ValueError(
            f"bbox window ({col_start}:{col_stop}, {row_start}:{row_stop}) does not "
            f"overlap the stitched grid of shape {stitched.shape}"
        )

    return np.array(stitched[r0:r1, c0:c1], dtype=np.float32)


def bounds_to_bbox(
    bounds: tuple[int, int, int, int],
    tile_origin: tuple[int, int],
    zoom: int,
) -> tuple[float, float, float, float]:
    """Invert :func:`crop_bounds` to the ``(south, west, north, east)`` covered.

    Args:
        bounds: ``(col_start, row_start, col_stop, row_stop)``.
        tile_origin: ``(x, y)`` of the stitched grid's north-west tile.
        zoom: Zoom the grid was built at.

    Returns:
        The bbox the pixel window actually spans, in degrees.
    """
    col_start, row_start, col_stop, row_stop = bounds
    world_px = TILE_SIZE * (1 << zoom)
    origin_col = tile_origin[0] * TILE_SIZE
    origin_row = tile_origin[1] * TILE_SIZE

    west = (col_start + origin_col) / world_px * 360.0 - 180.0
    east = (col_stop + origin_col) / world_px * 360.0 - 180.0
    north = _inverse_mercator_y((row_start + origin_row) / world_px)
    south = _inverse_mercator_y((row_stop + origin_row) / world_px)
    return south, west, north, east


def _inverse_mercator_y(y_norm: float) -> float:
    """Latitude in degrees for a normalised Web Mercator y."""
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_norm))))


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def resample_to_meters(
    arr: np.ndarray,
    south: float,
    west: float,
    north: float,
    east: float,
) -> np.ndarray:
    """Resample onto a grid whose pixels are square in real-world metres.

    ``arr`` is assumed to span exactly the given bbox in Web Mercator. Because
    that projection is conformal, its pixels are *already* locally square on
    the ground -- applying a per-axis ``cos(lat)`` correction here would
    double-apply Mercator's own stretch and squash the terrain north-south.
    What Mercator does not do is hold scale constant: metres per pixel shrinks
    as ``cos(lat)`` toward the poles. So the real work is rebuilding rows at a
    constant ground spacing (equidistant in latitude) rather than at constant
    Mercator spacing.

    Output resolution is the finer of the input's two axes, so detail is never
    thrown away.

    Residual error: scale is pinned at the bbox's centre latitude, so a tall
    bbox is slightly off at its top and bottom edges, by roughly
    ``1 - cos(dlat / 2) / cos(lat_center)``. That is under 0.1% for a 1-degree
    span at mid-latitudes and under 1% for 5 degrees, but grows sharply above
    about 70 degrees latitude, where a tall box should be split instead.

    Args:
        arr: 2D elevation array spanning the bbox, row 0 at the north edge.
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.

    Returns:
        A float32 array with equal ground spacing on both axes.
    """
    _validate_bbox(south, west, north, east)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("cannot resample an empty array")

    src_rows, src_cols = arr.shape
    width_m, height_m = ground_extent(south, west, north, east)
    if width_m <= 0.0 or height_m <= 0.0:
        raise ValueError("bbox has zero ground extent")

    # Keep the finest detail present in the source.
    mpp = min(width_m / src_cols, height_m / src_rows)
    out_cols = max(1, int(round(width_m / mpp)))
    out_rows = max(1, int(round(height_m / mpp)))

    # Output rows are equally spaced in latitude (constant ground distance);
    # map each back to the Mercator row it came from.
    y_top = mercator_y_norm(north)
    y_bottom = mercator_y_norm(south)
    lats = north - (np.arange(out_rows, dtype=np.float64) + 0.5) / out_rows * (north - south)
    merc = np.array([mercator_y_norm(float(v)) for v in lats])
    src_r = (merc - y_top) / (y_bottom - y_top) * src_rows - 0.5

    # Longitude is already linear in Mercator x, so columns map straight across.
    src_c = (np.arange(out_cols, dtype=np.float64) + 0.5) / out_cols * src_cols - 0.5

    grid_r, grid_c = np.meshgrid(src_r, src_c, indexing="ij")
    out = map_coordinates(
        arr.astype(np.float64, copy=False),
        np.stack([grid_r, grid_c]),
        order=1,
        mode="nearest",
    )
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Clean-up
# ---------------------------------------------------------------------------


def fill_nodata(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Fill NaN/nodata holes by nearest neighbour, softened at the borders.

    Holes are filled with their nearest valid neighbour, then a Gaussian blend
    weighted by proximity to the hole smooths the seam. Every filled value is
    a convex combination of real samples, so nothing is invented outside the
    surrounding data's range.

    Terrarium rarely has true holes, but USGS and other sources do.

    Args:
        arr: 2D array that may contain NaN, infinities, or values at
            Terrarium's encoding floor.
        sigma: Gaussian radius, in pixels, for the border blend.

    Returns:
        A float32 array with no holes left.

    Raises:
        ValueError: Every pixel is nodata, so there is nothing to fill from.
    """
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {data.shape}")

    holes = ~np.isfinite(data) | (data <= NODATA_FLOOR_M)
    if not holes.any():
        return data.copy()
    if holes.all():
        raise ValueError("every pixel is nodata; nothing to fill from")

    # distance_transform_edt measures distance to the nearest zero, so feeding
    # it the hole mask gives, for each hole pixel, the nearest valid pixel.
    indices = distance_transform_edt(holes, return_distances=False, return_indices=True)
    filled = data[tuple(indices)]

    if sigma <= 0.0:
        return filled.astype(np.float32, copy=False)

    # Blend strength tapers off with distance from the holes, so untouched
    # terrain keeps its original detail.
    weight = gaussian_filter(holes.astype(np.float32), sigma=sigma)
    peak = float(weight.max())
    if peak > 0.0:
        weight = np.clip(weight / peak, 0.0, 1.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    blended = filled * (1.0 - weight) + smoothed * weight
    return blended.astype(np.float32, copy=False)


def despike(arr: np.ndarray, threshold: float = DESPIKE_THRESHOLD) -> np.ndarray:
    """Replace isolated outlier pixels with their local median.

    Each pixel is compared against a 5x5 median of its neighbourhood. Where the
    difference exceeds ``threshold`` times the global interquartile range of
    those differences, the pixel is treated as a spike and replaced.

    Ridgelines survive this. A one-pixel-wide ridge deviates from its own 5x5
    median just as hard as a needle does -- only 5 of 25 window samples sit on
    the crest, so the median falls off it -- which is exactly how a naive
    median test erodes real terrain. So flagged pixels are additionally
    required to be *isolated*: candidates are grouped into connected
    components, and anything larger than :data:`MAX_SPIKE_CLUSTER_PX` is left
    alone. A ridge is one long component; a needle is one pixel.

    Args:
        arr: 2D elevation array, already free of nodata.
        threshold: Multiples of the interquartile range beyond which a
            deviation counts as a spike. Lower removes more.

    Returns:
        A float32 array with isolated spikes flattened to local median.
    """
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {data.shape}")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if data.size == 0:
        return data.copy()

    local_median = median_filter(data, size=DESPIKE_WINDOW_PX, mode="reflect")
    residual = data - local_median

    q25, q75 = np.percentile(residual, [25.0, 75.0])
    scale = float(q75 - q25)
    if scale <= 0.0:
        # Degenerate: the residual field is flat enough that three quarters of
        # it sits at one value (a synthetic ramp, say). Fall back to standard
        # deviation, which the spikes themselves inflate, so they still stand
        # out without every rounding wobble counting as one.
        scale = float(residual.std())
    if scale <= 0.0:
        return data.copy()  # perfectly uniform; nothing can be an outlier

    candidates = np.abs(residual) > threshold * scale
    if not candidates.any():
        return data.copy()

    # Keep only isolated clusters: connected ridgelines are real terrain.
    labels, count = label(candidates, structure=np.ones((3, 3), dtype=int))
    if count:
        sizes = np.bincount(labels.ravel())
        isolated = sizes <= MAX_SPIKE_CLUSTER_PX
        # Label 0 is background, whose bincount entry is meaningless here; it
        # would otherwise compare as "small" and flag the entire image.
        isolated[0] = False
        candidates = isolated[labels]

    out = data.copy()
    out[candidates] = local_median[candidates]
    return out


def auto_smooth_sigma(meters_per_px: float, mm_per_px: float | None = None) -> float:
    """Pick a Gaussian sigma that removes clutter without erasing the print.

    The blur is aimed at :data:`SMOOTH_GROUND_METERS` -- clutter is a physical
    size, so that target is in ground metres -- but capped at
    :data:`SMOOTH_PRINT_MM_MAX` of the finished surface. Without the cap the
    same 40 m target costs 0.29 mm of print on a wide framing and 1.32 mm on a
    tight one, so tight framings came out blank while wide ones were untouched.

    Args:
        meters_per_px: Ground resolution of the grid to be smoothed.
        mm_per_px: Printed millimetres per pixel. When omitted the print cap is
            not applied, which is only correct if the caller has already sized
            the blur itself.

    Returns:
        Sigma in pixels, clamped to ``[SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX]``.
    """
    if meters_per_px <= 0.0:
        raise ValueError(f"meters_per_px must be positive, got {meters_per_px}")

    ideal = SMOOTH_GROUND_METERS / meters_per_px
    if mm_per_px is not None:
        if mm_per_px <= 0.0:
            raise ValueError(f"mm_per_px must be positive, got {mm_per_px}")
        ideal = min(ideal, SMOOTH_PRINT_MM_MAX / mm_per_px)

    return float(np.clip(ideal, SMOOTH_SIGMA_MIN, SMOOTH_SIGMA_MAX))


def smooth(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur the terrain to take surface clutter down.

    Trees and buildings sit on top of the ground as high-frequency texture.
    Blurring before exaggeration means the vertical stretch amplifies landforms
    rather than clutter.

    Args:
        arr: 2D elevation array.
        sigma: Blur radius in pixels. Zero or negative is a no-op.

    Returns:
        A float32 array.
    """
    data = np.asarray(arr, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {data.shape}")
    if sigma <= 0.0 or data.size == 0:
        return data.copy()

    return gaussian_filter(data, sigma=sigma).astype(np.float32, copy=False)


#: Module-level handle so :func:`build_heightmap` can take a ``despike`` flag
#: parameter without shadowing the function it needs to call.
_despike = despike


def flatten_water(arr: np.ndarray, level: float | None = None) -> np.ndarray:
    """Clamp everything below ``level`` up to it, so water prints flat.

    Terrarium decodes ocean as real bathymetry, which would otherwise print as
    a chasm under the coastline. This is a styling choice, not a correction.

    Args:
        arr: 2D elevation array in metres.
        level: Surface to clamp to. When ``None``, sea level is used if the
            array dips below zero, and the array is left alone otherwise.

    Returns:
        A float32 array with no values below the chosen level.
    """
    data = np.asarray(arr, dtype=np.float32)

    if level is None:
        if data.size == 0 or float(np.nanmin(data)) >= 0.0:
            return data.copy()
        level = 0.0

    return np.maximum(data, np.float32(level))


def exaggerate(arr: np.ndarray, factor: float) -> np.ndarray:
    """Scale relief vertically about the array's minimum.

    Anchoring at the minimum keeps the model's base at the same height, so
    only the terrain above it grows.

    Args:
        arr: 2D elevation array in metres.
        factor: Multiplier for relief. ``1.0`` is a no-op.

    Returns:
        A float32 array with ``min + (arr - min) * factor``.
    """
    data = np.asarray(arr, dtype=np.float32)
    if factor <= 0.0:
        raise ValueError(f"exaggeration factor must be positive, got {factor}")
    if data.size == 0:
        return data.copy()

    # Compute in float64: in float32, (x - min) * 1.0 + min can drift off x by
    # an ulp, so a factor of 1.0 would not be exactly the no-op it claims to be.
    wide = data.astype(np.float64)
    floor = float(np.nanmin(wide))
    return (floor + (wide - floor) * float(factor)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_heightmap(
    south: float,
    west: float,
    north: float,
    east: float,
    target_px: int = 800,
    exaggeration: float = 1.0,
    flatten_water_level: float | str | None = "auto",
    smooth_px: float | str | None = "auto",
    despike: bool = True,
    despike_threshold: float = DESPIKE_THRESHOLD,
    cache_dir: str = ".tile_cache",
    pre_clean: Callable[[Heightmap], np.ndarray] | None = None,
) -> Heightmap:
    """Build a finished, print-ready heightmap for a bounding box.

    Runs the whole pipeline: zoom selection, tile fetch (with a one-tile
    margin), stitch, crop, resample to square ground pixels, nodata fill,
    despike, smooth, water flattening, and vertical exaggeration.

    Cleanup runs *before* exaggeration on purpose, so the vertical stretch
    amplifies landforms rather than tree canopy and outlier needles.

    Args:
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.
        target_px: Desired pixel span of the bbox's longer side.
        exaggeration: Vertical relief multiplier.
        flatten_water_level: ``"auto"`` clamps to sea level when the bbox dips
            below it, a number clamps to that height, and ``None`` skips it.
        smooth_px: Gaussian sigma in pixels, ``"auto"`` to derive one from the
            grid's ground resolution, or ``None``/``0`` for no smoothing.
        despike: Whether to remove isolated outlier pixels.
        despike_threshold: Interquartile ranges beyond which a deviation from
            the local median counts as a spike.
        cache_dir: Directory for the raw tile PNG cache.
        pre_clean: Optional hook run on the filled grid *before* despike and
            smooth, receiving a provisional un-exaggerated :class:`Heightmap`
            and returning a replacement elevation array. Building removal uses
            this, so exaggeration is later sized off cleaned ground rather than
            off a rooftop.

    Returns:
        A :class:`Heightmap` carrying the elevation grid and its scale.
    """
    _validate_bbox(south, west, north, east)

    zoom = zoom_for_bbox(south, west, north, east, target_px=target_px)
    coords = tiles_for_bbox(south, west, north, east, zoom)
    tiles_dict = {(x, y): fetch_tile(x, y, zoom, cache_dir=cache_dir) for x, y in coords}

    stitched = stitch(tiles_dict, zoom)
    origin = tile_origin(tiles_dict)

    bounds = crop_bounds(origin, zoom, south, west, north, east)
    cropped = crop_to_bbox(stitched, origin, zoom, south, west, north, east)

    # Resample against the window actually cut, not the one asked for, so the
    # outward rounding does not shift everything by up to a pixel.
    covered = bounds_to_bbox(bounds, origin, zoom)
    resampled = resample_to_meters(cropped, *covered)

    # Nodata is filled after resampling, per the pipeline order; bilinear
    # sampling can smear a hole by at most one pixel first.
    cleaned = fill_nodata(resampled)

    width_m, _ = ground_extent(*covered)
    meters_per_px = width_m / resampled.shape[1]

    if pre_clean is not None:
        cleaned = np.asarray(
            pre_clean(
                Heightmap(
                    elevation=cleaned,
                    meters_per_px=meters_per_px,
                    bbox=covered,
                    zoom=zoom,
                    requested_bbox=(south, west, north, east),
                )
            ),
            dtype=np.float32,
        )

    if despike:
        cleaned = _despike(cleaned, despike_threshold)

    # The print is always PRINT_WIDTH_MM across, so pixels map to millimetres
    # through the grid's own column count.
    mm_per_px = PRINT_WIDTH_MM / max(resampled.shape[1] - 1, 1)
    sigma = (
        auto_smooth_sigma(meters_per_px, mm_per_px)
        if smooth_px == "auto"
        else float(smooth_px or 0.0)
    )
    # Always call through the stage, even at sigma 0: it is a pipeline step,
    # and a no-op smooth is cheaper to reason about than a conditional one.
    cleaned = smooth(cleaned, sigma)

    if flatten_water_level == "auto":
        cleaned = flatten_water(cleaned, None)
    elif flatten_water_level is not None:
        cleaned = flatten_water(cleaned, float(flatten_water_level))

    final = exaggerate(cleaned, exaggeration)

    return Heightmap(
        elevation=final,
        meters_per_px=meters_per_px,
        bbox=covered,
        zoom=zoom,
        exaggeration=exaggeration,
        requested_bbox=(south, west, north, east),
    )
