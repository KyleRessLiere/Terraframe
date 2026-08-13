"""Slippy-map tile math and Terrarium elevation tile fetching.

Elevation data comes from the AWS-hosted Terrarium tileset (Mapzen /
Nextzen terrain tiles), which encodes metres above sea level into the RGB
channels of an ordinary 256x256 PNG:

    elevation_m = (R * 256 + G + B / 256) - 32768

Tiles are addressed with the standard Web Mercator ("slippy map") scheme:
x increases eastward from 0 at longitude -180, y increases *southward*
from 0 at the top of the Mercator projection (~85.05 degrees north).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

__all__ = [
    "TILE_SIZE",
    "MIN_ZOOM",
    "MAX_ZOOM",
    "TileFetchError",
    "latlon_to_tile",
    "tile_to_latlon",
    "tiles_for_bbox",
    "fetch_tile",
    "zoom_for_bbox",
    "mercator_x_norm",
    "mercator_y_norm",
]

#: Edge length in pixels of a single tile.
TILE_SIZE = 256

#: Zoom levels the Terrarium tileset is served at (and that we clamp to).
MIN_ZOOM = 1
MAX_ZOOM = 15

#: Latitude at which Web Mercator is truncated to keep the world square.
MAX_MERCATOR_LAT = 85.05112877980659

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

USER_AGENT = "terrframe/0.1.0 (+https://github.com/KyleRessLiere/Terraframe)"

#: Seconds before a tile request is considered failed.
REQUEST_TIMEOUT = 30.0


class TileFetchError(RuntimeError):
    """Raised when an elevation tile cannot be downloaded or decoded."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Return the ``(x, y)`` tile containing a latitude/longitude at ``zoom``.

    Args:
        lat: Latitude in degrees. Clamped to the Web Mercator limits.
        lon: Longitude in degrees, normalised into ``[-180, 180)``.
        zoom: Zoom level; the world is ``2 ** zoom`` tiles on a side.

    Returns:
        The tile column and row, both in ``[0, 2 ** zoom - 1]``.
    """
    if zoom < 0:
        raise ValueError(f"zoom must be non-negative, got {zoom}")

    n = 1 << zoom
    lat = _clamp(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT)
    # Wrap longitude so that e.g. 181 -> -179 rather than falling off the grid.
    lon = (lon + 180.0) % 360.0 - 180.0

    lat_rad = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

    # A point exactly on the antimeridian/pole edge can land one past the end.
    return min(x, n - 1), min(y, n - 1)


def tile_to_latlon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Return the ``(lat, lon)`` of the north-west corner of tile ``(x, y)``.

    This is the inverse of :func:`latlon_to_tile` up to the truncation that
    maps a whole tile's worth of coordinates onto a single tile index.

    Args:
        x: Tile column.
        y: Tile row.
        zoom: Zoom level.

    Returns:
        Latitude and longitude in degrees of the tile's NW corner.
    """
    if zoom < 0:
        raise ValueError(f"zoom must be non-negative, got {zoom}")

    n = 1 << zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def tiles_for_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int,
) -> list[tuple[int, int]]:
    """List every tile covering a bounding box, plus a one-tile margin.

    The margin matters downstream: stitching and resampling near the edge of
    the requested area needs real elevation samples just outside it, or the
    boundary picks up seams and clamped-gradient artefacts.

    Args:
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.
        zoom: Zoom level to enumerate tiles at.

    Returns:
        Tiles as ``(x, y)`` pairs in row-major order (y outer, x inner),
        clamped to the valid tile grid for ``zoom``.
    """
    if south > north:
        raise ValueError(f"south ({south}) must be <= north ({north})")
    if west > east:
        raise ValueError(f"west ({west}) must be <= east ({east})")

    n = 1 << zoom
    # North latitude gives the smaller y, so the NW corner is (x_min, y_min).
    x_min, y_min = latlon_to_tile(north, west, zoom)
    x_max, y_max = latlon_to_tile(south, east, zoom)

    x_min = int(max(0, x_min - 1))
    y_min = int(max(0, y_min - 1))
    x_max = int(min(n - 1, x_max + 1))
    y_max = int(min(n - 1, y_max + 1))

    return [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]


def zoom_for_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    target_px: int = 800,
) -> int:
    """Pick the zoom whose pixel grid spans roughly ``target_px`` across the bbox.

    Resolution is chosen from the *longer* side of the box so the target acts
    as a ceiling on the largest dimension rather than a floor on the smallest.

    Args:
        south: Southern latitude bound, in degrees.
        west: Western longitude bound, in degrees.
        north: Northern latitude bound, in degrees.
        east: Eastern longitude bound, in degrees.
        target_px: Desired pixel span of the bbox's longer side.

    Returns:
        A zoom level clamped to ``[MIN_ZOOM, MAX_ZOOM]``.
    """
    if south > north:
        raise ValueError(f"south ({south}) must be <= north ({north})")
    if west > east:
        raise ValueError(f"west ({west}) must be <= east ({east})")
    if target_px <= 0:
        raise ValueError(f"target_px must be positive, got {target_px}")

    # Fraction of the whole world the bbox covers, in normalised (0..1)
    # Web Mercator space, where the y axis is already distorted by latitude.
    lon_frac = (east - west) / 360.0
    lat_frac = abs(_mercator_y(north) - _mercator_y(south))
    span_frac = max(lon_frac, lat_frac)

    if span_frac <= 0.0:
        # Degenerate (zero-area) box: nothing to scale to, so go max detail.
        return MAX_ZOOM

    # world_px(z) = TILE_SIZE * 2**z, and we want world_px * span_frac ~ target_px.
    zoom = math.log2(target_px / (TILE_SIZE * span_frac))
    return int(_clamp(round(zoom), MIN_ZOOM, MAX_ZOOM))


def mercator_y_norm(lat: float) -> float:
    """Normalised Web Mercator y in ``[0, 1]`` (0 at the north edge).

    Multiply by ``TILE_SIZE * 2 ** zoom`` to get a global pixel row.

    Args:
        lat: Latitude in degrees, clamped to the Web Mercator limits.

    Returns:
        Position down the projected world, 0 at the north edge and 1 at the south.
    """
    lat = _clamp(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT)
    return (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0


def mercator_x_norm(lon: float) -> float:
    """Normalised Web Mercator x in ``[0, 1]`` (0 at longitude -180).

    Multiply by ``TILE_SIZE * 2 ** zoom`` to get a global pixel column.

    Args:
        lon: Longitude in degrees.

    Returns:
        Position across the projected world, 0 at the west edge and 1 at the east.
    """
    return (lon + 180.0) / 360.0


#: Backwards-compatible alias for the pre-public spelling.
_mercator_y = mercator_y_norm


def _cache_path(x: int, y: int, zoom: int, cache_dir: str | os.PathLike[str]) -> Path:
    return Path(cache_dir) / str(zoom) / str(x) / f"{y}.png"


def _download_tile(x: int, y: int, zoom: int, path: Path) -> None:
    """Download one tile to ``path``, writing atomically."""
    url = TERRARIUM_URL.format(z=zoom, x=x, y=y)
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise TileFetchError(f"failed to download tile {zoom}/{x}/{y} from {url}: {exc}") from exc

    if not response.content:
        raise TileFetchError(f"tile {zoom}/{x}/{y} came back empty from {url}")

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file first so an interrupted run can never leave
    # a truncated PNG behind in the cache.
    tmp = path.with_suffix(".png.part")
    tmp.write_bytes(response.content)
    tmp.replace(path)


def _decode_terrarium(path: Path) -> np.ndarray:
    """Decode a Terrarium PNG on disk into a float32 array of metres."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {rgb.shape}")

    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (red * 256.0 + green + blue / 256.0) - 32768.0


def fetch_tile(
    x: int,
    y: int,
    zoom: int,
    cache_dir: str | os.PathLike[str] = ".tile_cache",
) -> np.ndarray:
    """Fetch one Terrarium tile and decode it to elevations in metres.

    The raw PNG is cached on disk under ``cache_dir/{z}/{x}/{y}.png`` and a
    cached tile is never re-downloaded. If a cached file turns out to be
    unreadable it is discarded and fetched again once.

    Args:
        x: Tile column.
        y: Tile row.
        zoom: Zoom level.
        cache_dir: Directory holding the raw PNG cache.

    Returns:
        A ``(TILE_SIZE, TILE_SIZE)`` float32 array of metres above sea level,
        row 0 being the northern edge of the tile.

    Raises:
        TileFetchError: The tile could not be downloaded, or the freshly
            downloaded PNG could not be decoded.
    """
    n = 1 << zoom
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"tile ({x}, {y}) is outside the 0..{n - 1} grid for zoom {zoom}")

    path = _cache_path(x, y, zoom, cache_dir)

    if path.exists():
        try:
            return _decode_terrarium(path)
        except (OSError, ValueError, UnidentifiedImageError):
            # Corrupt or truncated cache entry — drop it and fetch fresh.
            path.unlink(missing_ok=True)

    _download_tile(x, y, zoom, path)

    try:
        return _decode_terrarium(path)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        path.unlink(missing_ok=True)
        raise TileFetchError(f"tile {zoom}/{x}/{y} downloaded but could not be decoded: {exc}") from exc
