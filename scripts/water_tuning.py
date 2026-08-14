#!/usr/bin/env python
"""Render a depth x bead sweep of one scene, for picking defaults by eye.

    python scripts/water_tuning.py

Writes gallery/water_tuning.png: a grid of 3D renders, one per parameter pair,
each labelled. The point is that water definition is a judgement call about how
a single-colour print reads in the hand, which no metric settles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render3d  # noqa: E402

from terrframe.features import apply_features, fetch_osm_or_warn, layers_for_style  # noqa: E402
from terrframe.heightmap import build_heightmap, exaggerate, ground_extent  # noqa: E402
from terrframe.mesh import auto_exaggeration, heightmap_to_mesh  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The acceptance scene: flat metro whose identity is entirely water.
DC_BBOX = (38.7800, -77.2000, 38.9900, -76.8800)

DEPTHS_MM = (0.8, 1.0, 1.4)
BEADS_MM = (0.3, 0.5)


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main(argv: list[str] | None = None) -> int:
    """Render the sweep. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="water_tuning.py", description=__doc__)
    parser.add_argument("--target-px", type=int, default=700, help="sampling detail")
    parser.add_argument("--width", type=int, default=620, help="pixel width per panel")
    parser.add_argument("--view", default="top", choices=sorted(render3d.VIEWS))
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "gallery" / "water_tuning.png"
    )
    args = parser.parse_args(argv)

    south, west, north, east = DC_BBOX
    base = build_heightmap(south, west, north, east, target_px=args.target_px)
    ground_width_m, _ = ground_extent(*base.bbox)
    factor = auto_exaggeration(float(np.ptp(base.elevation)), ground_width_m)
    exaggerated = type(base)(
        exaggerate(base.elevation, factor),
        base.meters_per_px,
        base.bbox,
        base.zoom,
        factor,
        base.requested_bbox,
    )
    features = fetch_osm_or_warn(base.bbox, layers_for_style("natural", "off"))
    print(f"dc_natural: exag x{factor:.2f}, {len(features['water'])} water polygons", flush=True)

    azimuth, elevation = render3d.VIEWS[args.view]
    panels: list[list[Image.Image]] = []

    for bead_mm in BEADS_MM:
        row: list[Image.Image] = []
        for depth_mm in DEPTHS_MM:
            stamped = apply_features(
                exaggerated, "natural", features, water_depth_mm=depth_mm, shoreline_mm=bead_mm
            )
            mesh = heightmap_to_mesh(stamped)
            # Stamps are array ops, but the invariant is asserted, not assumed.
            if not mesh.is_watertight:
                raise RuntimeError(f"depth {depth_mm} bead {bead_mm} broke watertightness")
            row.append(render3d.render_mesh(mesh, width=args.width, azimuth=azimuth, elevation=elevation))
            print(
                f"  depth {depth_mm} mm  bead {bead_mm} mm  ->  "
                f"{mesh.extents[2]:.1f} mm tall, watertight",
                flush=True,
            )
        panels.append(row)

    title_font, cell_font = _font(26), _font(19)
    cell_w = max(img.width for row in panels for img in row)
    cell_h = max(img.height for row in panels for img in row)
    gap, header, caption = 12, 52, 30

    sheet = Image.new(
        "RGB",
        (
            len(DEPTHS_MM) * cell_w + (len(DEPTHS_MM) + 1) * gap,
            header + len(BEADS_MM) * (cell_h + caption + gap),
        ),
        (16, 16, 18),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (gap, 14),
        f"DC water tuning — recess depth x shoreline bead ({args.view} view)",
        font=title_font,
        fill=(242, 242, 247),
    )

    for r, bead_mm in enumerate(BEADS_MM):
        for c, depth_mm in enumerate(DEPTHS_MM):
            x = gap + c * (cell_w + gap)
            y = header + r * (cell_h + caption + gap)
            img = panels[r][c]
            sheet.paste(img, (x + (cell_w - img.width) // 2, y))
            draw.text(
                (x, y + cell_h + 5),
                f"depth {depth_mm} mm    bead {bead_mm} mm",
                font=cell_font,
                fill=(228, 228, 235),
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"\nwrote {args.out}  {sheet.size}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
