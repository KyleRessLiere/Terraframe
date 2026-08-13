#!/usr/bin/env python
"""Render an STL/3MF to a PNG so the printed solid can be eyeballed.

Hillshaded heightmaps show what the *data* looks like; this shows what the
*model* looks like -- base, skirt walls, silhouette and all. It loads the
exported file back off disk, so what gets rendered is the artefact a slicer
would see, not an in-memory approximation.

    python scripts/render3d.py rainier.stl -o rainier_3d.png

Deliberately dependency-free: a painter's-algorithm point splat over numpy,
because pulling in an OpenGL stack for an occasional sanity render is a poor
trade.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

__all__ = ["render_mesh", "load_mesh"]

#: Camera azimuth in compass degrees; 315 matches the hillshade convention.
VIEW_AZIMUTH = 315.0

#: Camera elevation above the horizon.
VIEW_ELEVATION = 32.0

#: Light direction, also north-west and slightly higher than the camera.
LIGHT_AZIMUTH = 315.0
LIGHT_ELEVATION = 50.0

#: Neutral filament-like material, so form reads without hypsometric tinting.
MATERIAL_RGB = (232, 228, 220)
BACKGROUND_RGB = (24, 25, 28)

#: Shadow floor, so downward faces stay readable instead of going black.
AMBIENT = 0.28


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file, concatenating scenes into one body."""
    loaded = trimesh.load(Path(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} did not load as a single mesh (got {type(loaded).__name__})")
    return loaded


def _basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right/up/forward vectors for a compass azimuth and elevation."""
    az, el = math.radians(azimuth), math.radians(elevation)
    right = np.array([math.cos(az), -math.sin(az), 0.0])
    up = np.array([math.sin(az) * math.sin(el), math.cos(az) * math.sin(el), math.cos(el)])
    forward = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), -math.sin(el)])
    return right, up, forward


def _surface_samples(mesh: trimesh.Trimesh, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Points covering the surface, with a normal each.

    Per-face sampling (centroids, midpoints) allocates by triangle *count*,
    which starves the skirt walls: they are a handful of very tall triangles
    next to hundreds of thousands of small terrain ones, so the base renders as
    a few thin lines instead of a solid wall. Area-weighted sampling allocates
    by surface area instead, which is what actually determines screen coverage.

    Vertices are included too, to keep ridges and the silhouette crisp.
    """
    points, face_index = trimesh.sample.sample_surface(mesh, budget)
    normals = mesh.face_normals[face_index]

    return (
        np.concatenate([mesh.vertices, points]),
        np.concatenate([mesh.vertex_normals, normals]),
    )


def render_mesh(
    mesh: trimesh.Trimesh,
    width: int = 900,
    azimuth: float = VIEW_AZIMUTH,
    elevation: float = VIEW_ELEVATION,
    margin: float = 0.04,
    oversample: float = 6.0,
) -> Image.Image:
    """Render a mesh to an RGB image with a painter's-algorithm splat.

    Args:
        mesh: The solid to draw.
        width: Output width in pixels; height follows the projected aspect.
        azimuth: Camera compass bearing.
        elevation: Camera height above the horizon.
        margin: Fraction of the frame left as padding.
        oversample: Surface samples per output pixel; higher fills gaps.

    Returns:
        An RGB Pillow image.
    """
    points, normals = _surface_samples(mesh, int(width * width * oversample))

    right, up, forward = _basis(azimuth, elevation)
    screen_x = points @ right
    screen_y = points @ up
    depth = -(points @ forward)  # larger is nearer the camera

    span_x = float(screen_x.max() - screen_x.min())
    span_y = float(screen_y.max() - screen_y.min())
    if span_x <= 0 or span_y <= 0:
        raise ValueError("mesh projects to zero area; nothing to render")

    usable = 1.0 - 2.0 * margin
    scale = (width * usable) / span_x
    height = max(1, int(round(span_y * scale + 2.0 * margin * width)))

    px = (screen_x - screen_x.min()) * scale + margin * width
    # Screen rows grow downward, so the projected y axis is flipped.
    py = (screen_y.max() - screen_y) * scale + margin * width

    cols = np.clip(px.astype(np.int32), 0, width - 1)
    rows = np.clip(py.astype(np.int32), 0, height - 1)

    lr, lu, lf = _basis(LIGHT_AZIMUTH, LIGHT_ELEVATION)
    light = -lf  # points from the surface toward the light
    shade = np.clip(normals @ light, 0.0, 1.0)
    intensity = AMBIENT + (1.0 - AMBIENT) * shade

    material = np.array(MATERIAL_RGB, dtype=np.float64)
    colors = np.clip(intensity[:, None] * material, 0, 255).astype(np.uint8)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = BACKGROUND_RGB

    # Painter's algorithm: draw far to near so the nearest sample wins each
    # pixel. numpy fancy indexing keeps the last write for duplicate targets.
    order = np.argsort(depth)
    canvas[rows[order], cols[order]] = colors[order]

    return Image.fromarray(canvas, mode="RGB")


def main(argv: list[str] | None = None) -> int:
    """Render a mesh file to a PNG. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="render3d.py", description="Render an STL or 3MF to a PNG for visual review."
    )
    parser.add_argument("model", type=Path, help="path to a .stl or .3mf file")
    parser.add_argument("-o", "--output", type=Path, help="output PNG (default: alongside input)")
    parser.add_argument("--width", type=int, default=900, help="output width in pixels")
    parser.add_argument("--azimuth", type=float, default=VIEW_AZIMUTH, help="camera bearing")
    parser.add_argument("--elevation", type=float, default=VIEW_ELEVATION, help="camera height")
    args = parser.parse_args(argv)

    mesh = load_mesh(args.model)
    destination = args.output or args.model.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)

    image = render_mesh(mesh, width=args.width, azimuth=args.azimuth, elevation=args.elevation)
    image.save(destination)

    print(f"{args.model.name}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"  watertight {mesh.is_watertight}   extents {mesh.extents.round(1)}")
    print(f"  wrote {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
