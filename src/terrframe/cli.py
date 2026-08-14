"""Command-line entry point for terrframe.

    terrframe --bbox 46.75,-121.95,46.95,-121.55 -o rainier.stl
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

from .features import (
    SHORELINE_HEIGHT_MM,
    STYLES,
    WATER_DEPTH_MM,
    apply_features,
    building_remover,
    fetch_osm_or_warn,
    layers_for_style,
    should_remove_buildings,
)
from .heightmap import (
    DESPIKE_THRESHOLD,
    Heightmap,
    build_heightmap,
    exaggerate,
    ground_extent,
)
from .mesh import auto_exaggeration, export, heightmap_to_mesh

__all__ = ["main"]


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
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
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise argparse.ArgumentTypeError("latitudes must lie within -90..90")
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise argparse.ArgumentTypeError("longitudes must lie within -180..180")
    return south, west, north, east


def _parse_exaggeration(text: str) -> float | str:
    """Parse ``--exaggeration``: either ``auto`` or a positive number."""
    if text.strip().lower() == "auto":
        return "auto"
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--exaggeration takes a number or 'auto', got {text!r}"
        ) from exc
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"--exaggeration must be positive, got {value}")
    return value


def _parse_water(text: str) -> float | str | None:
    """Parse ``--flatten-water``: ``auto``, ``none``, or an elevation in metres."""
    lowered = text.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"none", "off"}:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--flatten-water takes a number, 'auto', or 'none', got {text!r}"
        ) from exc


def _parse_smooth(text: str) -> float | str | None:
    """Parse ``--smooth``: ``auto``, ``none``, or a sigma in pixels."""
    lowered = text.strip().lower()
    if lowered == "auto":
        return "auto"
    if lowered in {"none", "off"}:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--smooth takes a number, 'auto', or 'none', got {text!r}"
        ) from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"--smooth must be non-negative, got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terrframe",
        description="Convert a geographic bounding box into a 3D-printable terrain model.",
    )
    parser.add_argument(
        "--bbox",
        required=True,
        type=_parse_bbox,
        metavar="S,W,N,E",
        help="bounding box in degrees, e.g. 46.75,-121.95,46.95,-121.55",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="output model path; .stl or .3mf",
    )
    parser.add_argument(
        "--width-mm", type=float, default=200.0, help="printed width in mm (default: 200)"
    )
    parser.add_argument(
        "--base-mm",
        type=float,
        default=6.0,
        help="solid base thickness under the lowest valley, in mm (default: 6)",
    )
    parser.add_argument(
        "--exaggeration",
        type=_parse_exaggeration,
        default="auto",
        metavar="FACTOR",
        help="vertical exaggeration, or 'auto' to pick one from the terrain (default: auto)",
    )
    parser.add_argument(
        "--flatten-water",
        type=_parse_water,
        default="auto",
        metavar="LEVEL",
        help="flatten water to this elevation, 'auto' for sea level, or 'none' (default: auto)",
    )
    parser.add_argument(
        "--smooth",
        type=_parse_smooth,
        default=None,
        metavar="SIGMA",
        help=(
            "gaussian smoothing radius in pixels, or 'auto' to scale it to the "
            "grid's resolution (default: none -- building removal declutters "
            "better, and blurring only costs detail)"
        ),
    )
    parser.add_argument(
        "--despike",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove isolated outlier pixels before smoothing (default: enabled)",
    )
    parser.add_argument(
        "--despike-threshold",
        type=float,
        default=DESPIKE_THRESHOLD,
        metavar="IQR",
        help=f"spike sensitivity in interquartile ranges (default: {DESPIKE_THRESHOLD})",
    )
    parser.add_argument(
        "--style",
        choices=STYLES,
        default="minimal",
        help=(
            "minimal = terrain only; natural = + OSM water; detailed = + water "
            "(roads/trails pending) (default: minimal)"
        ),
    )
    parser.add_argument(
        "--remove-buildings",
        default="auto",
        metavar="on|off|auto",
        help=(
            "cut OSM building footprints out of the terrain; auto enables it "
            "when buildings exist and the source is not bare earth (default: auto)"
        ),
    )
    parser.add_argument(
        "--water-depth",
        "--water-depth-mm",
        dest="water_depth",
        type=float,
        default=WATER_DEPTH_MM,
        metavar="MM",
        help=f"recess depth for stamped water, in mm (default: {WATER_DEPTH_MM})",
    )
    parser.add_argument(
        "--shoreline",
        type=float,
        default=SHORELINE_HEIGHT_MM,
        metavar="MM",
        help=(
            "height of the raised bead outlining water, in mm; 0 disables "
            f"(default: {SHORELINE_HEIGHT_MM})"
        ),
    )
    parser.add_argument(
        "--target-px",
        type=int,
        default=800,
        help="sampling detail: pixel span of the bbox's longer side (default: 800)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=4_000_000,
        help="vertex ceiling; the terrain is downsampled to fit (default: 4000000)",
    )
    return parser


def _resolve_exaggeration(hm: Heightmap, requested: float | str) -> tuple[Heightmap, float]:
    """Apply the requested exaggeration, choosing one automatically if asked.

    ``hm`` must be un-exaggerated. Returns the exaggerated heightmap and the
    factor used.
    """
    if requested != "auto":
        factor = float(requested)
    else:
        relief_m = float(hm.elevation.max() - hm.elevation.min())
        width_m, _ = ground_extent(*hm.bbox)
        factor = auto_exaggeration(relief_m, width_m)
        print(
            f"auto exaggeration: x{factor:.2f} "
            f"({relief_m:.0f} m relief over {width_m / 1000:.1f} km)",
            file=sys.stderr,
        )

    if factor == 1.0:
        return hm, factor

    return (
        dataclasses.replace(
            hm, elevation=exaggerate(hm.elevation, factor), exaggeration=factor
        ),
        factor,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the terrframe CLI. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    south, west, north, east = args.bbox

    try:
        # OSM first: building removal has to happen before despike/smooth so
        # auto exaggeration sizes off cleaned ground, not off a rooftop.
        wanted = layers_for_style(args.style, args.remove_buildings)
        features = (
            fetch_osm_or_warn((south, west, north, east), wanted) if wanted else None
        )

        buildings = features["buildings"] if features else []
        # is_bare_earth is False for every source today; sources.py will supply
        # the real answer once 3DEP routing lands.
        removing = should_remove_buildings(args.remove_buildings, buildings, is_bare_earth=False)
        if removing:
            print(f"removing {len(buildings):,} building footprints", file=sys.stderr)

        # Always build unexaggerated first: 'auto' needs the true relief, and
        # applying the factor afterwards avoids a second fetch-and-resample.
        heightmap = build_heightmap(
            south,
            west,
            north,
            east,
            target_px=args.target_px,
            exaggeration=1.0,
            flatten_water_level=args.flatten_water,
            smooth_px=args.smooth,
            despike=args.despike,
            despike_threshold=args.despike_threshold,
            pre_clean=building_remover(buildings) if removing else None,
        )
        heightmap, factor = _resolve_exaggeration(heightmap, args.exaggeration)

        # Stamps go on after exaggeration, so their printed size is fixed.
        if features is not None:
            heightmap = apply_features(
                heightmap,
                args.style,
                features,
                water_depth_mm=args.water_depth,
                shoreline_mm=args.shoreline,
            )

        mesh = heightmap_to_mesh(
            heightmap,
            width_mm=args.width_mm,
            base_mm=args.base_mm,
            max_vertices=args.max_vertices,
        )
        destination = export(mesh, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"terrframe: {exc}", file=sys.stderr)
        return 1

    _print_summary(heightmap, mesh, destination, factor)
    return 0


def _print_summary(
    heightmap: Heightmap,
    mesh: object,
    destination: Path,
    factor: float,
) -> None:
    """Report what was built, in the units the user thinks in."""
    elevation = heightmap.elevation
    size_x, size_y, size_z = mesh.extents  # type: ignore[attr-defined]

    # Undo the exaggeration for reporting, so elevations stay geographic.
    floor_m = float(elevation.min())
    true_ceiling = floor_m + float(elevation.max() - floor_m) / factor

    print(f"wrote        {destination}")
    print(f"size         {size_x:.1f} x {size_y:.1f} x {size_z:.1f} mm")
    print(f"geometry     {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")  # type: ignore[attr-defined]
    print(f"elevation    {floor_m:.0f} to {true_ceiling:.0f} m")
    print(f"exaggeration x{factor:.2f}")
    print(f"scale        {heightmap.meters_per_px:.1f} m/px, zoom {heightmap.zoom}")
    print(f"watertight   {'yes' if mesh.is_watertight else 'NO'}")  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
