#!/usr/bin/env python
"""Render a heightmap as a shaded relief PNG for human review.

    python scripts/preview.py --bbox 46.75,-121.95,46.95,-121.55 -o preview.png

Produces a north-west-lit hillshade blended with a hypsometric elevation tint,
which is the fastest way to eyeball whether a bbox is worth printing and
whether the pipeline mangled anything.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from terrframe.heightmap import Heightmap, build_heightmap

#: Hypsometric ramp: relative height -> RGB. Low ground reads green, mid
#: ground tan, peaks rock-grey into snow.
TERRAIN_RAMP: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (56, 94, 68)),
    (0.15, (94, 128, 72)),
    (0.35, (156, 158, 88)),
    (0.55, (176, 140, 92)),
    (0.75, (150, 118, 98)),
    (0.90, (190, 185, 180)),
    (1.00, (250, 250, 252)),
]

#: Flat water is drawn as water, not as the bottom of the land ramp.
WATER_RGB = (58, 92, 124)

#: Fraction of the image that must sit exactly at the minimum before we treat
#: that level as a flattened water surface rather than an incidental low point.
WATER_MIN_FRACTION = 0.005


def hillshade(
    elevation: np.ndarray,
    meters_per_px: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Compute shaded relief in ``[0, 1]``.

    Args:
        elevation: 2D array of metres, row 0 at the north edge.
        meters_per_px: Ground spacing, used so slope is a true gradient.
        azimuth: Light direction in compass degrees; 315 is the cartographic
            north-west convention.
        altitude: Light elevation above the horizon, in degrees.
        z_factor: Extra vertical emphasis applied to the shading only.

    Returns:
        A float32 array of illumination, 0 fully shadowed and 1 fully lit.
    """
    # np.gradient gives d/drow and d/dcol; rows increase southward, so the
    # northward derivative is the negated row gradient.
    d_row, d_col = np.gradient(elevation.astype(np.float64), meters_per_px)
    dz_dx = d_col * z_factor  # eastward
    dz_dy = -d_row * z_factor  # northward

    # Shade by dotting the surface normal with the light direction directly.
    # The surface normal of z = f(x, y) is (-dz_dx, -dz_dy, 1), and compass
    # azimuth puts the light at (sin A, cos A) in (east, north).
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    light_e = math.sin(az) * math.cos(alt)
    light_n = math.cos(az) * math.cos(alt)
    light_up = math.sin(alt)

    shaded = (-dz_dx * light_e - dz_dy * light_n + light_up) / np.sqrt(
        dz_dx**2 + dz_dy**2 + 1.0
    )
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


def elevation_tint(elevation: np.ndarray) -> np.ndarray:
    """Map elevations onto the hypsometric ramp.

    Args:
        elevation: 2D array of metres.

    Returns:
        A ``(rows, cols, 3)`` float32 array of RGB in ``[0, 1]``.
    """
    low = float(np.nanmin(elevation))
    high = float(np.nanmax(elevation))
    span = high - low
    # A perfectly flat tile would divide by zero; render it as mid-ramp.
    norm = np.full_like(elevation, 0.5, dtype=np.float64) if span <= 0 else (elevation - low) / span

    stops = np.array([s for s, _ in TERRAIN_RAMP])
    colors = np.array([c for _, c in TERRAIN_RAMP], dtype=np.float64) / 255.0

    rgb = np.empty(elevation.shape + (3,), dtype=np.float64)
    for channel in range(3):
        rgb[..., channel] = np.interp(norm, stops, colors[:, channel])
    return rgb.astype(np.float32)


def _water_mask(elevation: np.ndarray) -> np.ndarray:
    """Find a flattened water surface, if the heightmap has one."""
    low = float(np.nanmin(elevation))
    if low > 0.0:
        return np.zeros(elevation.shape, dtype=bool)
    mask = elevation <= low
    if mask.mean() < WATER_MIN_FRACTION:
        return np.zeros(elevation.shape, dtype=bool)
    return mask


def render(heightmap: Heightmap, z_factor: float = 1.5) -> Image.Image:
    """Blend hillshade and tint into a finished preview image.

    Args:
        heightmap: The heightmap to draw.
        z_factor: Vertical emphasis for the shading only; leaves data untouched.

    Returns:
        An RGB Pillow image.
    """
    elevation = heightmap.elevation
    shade = hillshade(elevation, heightmap.meters_per_px, z_factor=z_factor)
    rgb = elevation_tint(elevation)

    water = _water_mask(elevation)
    if water.any():
        rgb[water] = np.array(WATER_RGB, dtype=np.float32) / 255.0

    # Keep ambient light in the shadows so relief stays readable instead of
    # crushing to black, and lift the midtones slightly.
    lit = rgb * (0.30 + 0.85 * shade[..., None])
    lit = np.clip(lit, 0.0, 1.0) ** (1 / 1.08)

    return Image.fromarray((lit * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """Parse a ``S,W,N,E`` string into a bbox tuple."""
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"--bbox needs 4 comma-separated numbers (S,W,N,E), got {len(parts)}"
        )
    try:
        south, west, north, east = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--bbox values must be numbers: {exc}") from exc

    if south > north:
        raise argparse.ArgumentTypeError(f"south ({south}) must be <= north ({north})")
    if west > east:
        raise argparse.ArgumentTypeError(f"west ({west}) must be <= east ({east})")
    return south, west, north, east


def main(argv: list[str] | None = None) -> int:
    """Render a preview PNG for a bbox. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Render a hillshaded preview of a terrframe heightmap.",
    )
    parser.add_argument(
        "--bbox",
        required=True,
        type=parse_bbox,
        help="bounding box as S,W,N,E in degrees",
    )
    parser.add_argument(
        "--exaggeration", type=float, default=1.0, help="vertical relief multiplier"
    )
    parser.add_argument(
        "--target-px", type=int, default=800, help="pixel span of the bbox's longer side"
    )
    parser.add_argument(
        "--z-factor",
        type=float,
        default=1.5,
        help="shading-only vertical emphasis; does not alter the data",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("preview.png"), help="output PNG path"
    )
    args = parser.parse_args(argv)

    south, west, north, east = args.bbox
    print(f"bbox      {south}, {west}, {north}, {east}", file=sys.stderr)

    heightmap = build_heightmap(
        south,
        west,
        north,
        east,
        target_px=args.target_px,
        exaggeration=args.exaggeration,
    )

    elevation = heightmap.elevation
    width_m, height_m = heightmap.size_meters
    print(f"zoom      {heightmap.zoom}", file=sys.stderr)
    print(f"grid      {elevation.shape[1]} x {elevation.shape[0]} px", file=sys.stderr)
    print(f"scale     {heightmap.meters_per_px:.1f} m/px", file=sys.stderr)
    print(f"ground    {width_m / 1000:.1f} x {height_m / 1000:.1f} km", file=sys.stderr)
    print(
        f"relief    {elevation.min():.0f} to {elevation.max():.0f} m"
        f" (x{heightmap.exaggeration:g})",
        file=sys.stderr,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(heightmap, z_factor=args.z_factor).save(args.output)
    print(f"wrote     {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
