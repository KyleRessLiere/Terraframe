#!/usr/bin/env python
"""Visual regression harness: render a fixed scene suite and measure it.

The scene list below is deliberately frozen. It spans terrain classes that
stress different parts of the pipeline, so a change that helps alpine terrain
and quietly ruins flat terrain shows up immediately.

    python scripts/gallery.py                       # all scenes
    python scripts/gallery.py --scene tahoe
    python scripts/gallery.py --params exaggeration=3,smooth=2

Any pipeline or constant change should be validated by rerunning this and
comparing ``gallery/contact_sheet.png`` against the last committed one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preview  # noqa: E402

from terrframe.heightmap import (  # noqa: E402
    DESPIKE_THRESHOLD,
    DESPIKE_WINDOW_PX,
    MAX_SPIKE_CLUSTER_PX,
    auto_smooth_sigma,
    build_heightmap,
    exaggerate,
    ground_extent,
)
from terrframe.mesh import auto_exaggeration  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "gallery"

#: Print width the printed-relief metric is quoted against, matching the CLI.
REFERENCE_WIDTH_MM = 200.0

#: Border trimmed before measuring roughness; gaussian reflection at the edge
#: fakes gradient variance that grows with sigma.
ROUGHNESS_BORDER_PX = 12

#: Gradient magnitude below which ground counts as dead flat.
FLAT_EPS = 1e-3


@dataclass(frozen=True)
class Scene:
    """One fixed test area."""

    name: str
    bbox: tuple[float, float, float, float]
    terrain: str


#: Frozen suite. Do not reorder or edit without regenerating the baseline.
SCENES: list[Scene] = [
    Scene("tahoe", (38.85, -120.25, 39.35, -119.85), "alpine + big lake"),
    Scene("stoneybrooke", (38.7500, -77.1244, 38.7860, -77.0782), "flat suburb"),
    Scene("rainier", (46.75, -121.95, 46.95, -121.55), "single dramatic peak"),
    Scene("sf_coast", (37.70, -122.60, 37.85, -122.35), "coastline + ocean + city"),
    Scene("kansas", (38.80, -97.75, 39.00, -97.45), "near-zero relief, worst case"),
]


def _interior(arr: np.ndarray, border: int = ROUGHNESS_BORDER_PX) -> np.ndarray:
    trimmed = arr[border:-border, border:-border] if border else arr
    return trimmed if trimmed.size else arr


def roughness(elevation: np.ndarray) -> float:
    """Spread of the surface's slopes -- how jagged it would print."""
    return float(np.std(np.gradient(_interior(elevation))))


def spike_counts(elevation: np.ndarray) -> tuple[int, int]:
    """Count pixels deviating past ``DESPIKE_THRESHOLD`` IQRs of a local median.

    Returns both the literal count and the count restricted to *isolated*
    clusters. The literal figure never reaches zero on real terrain: a sharp
    ridge crest deviates from its own 5x5 median exactly as hard as a needle
    does, because only 5 of 25 window samples sit on the crest. The isolated
    figure is the one that answers "are there spikes left", since that is the
    population :func:`terrframe.heightmap.despike` targets.

    Returns:
        ``(literal_count, isolated_count)``.
    """
    data = elevation.astype(np.float32)
    residual = data - median_filter(data, size=DESPIKE_WINDOW_PX, mode="reflect")
    q25, q75 = np.percentile(residual, [25.0, 75.0])
    scale = float(q75 - q25) or float(residual.std())
    if scale <= 0.0:
        return 0, 0

    flagged = np.abs(residual) > DESPIKE_THRESHOLD * scale
    literal = int(flagged.sum())
    if not literal:
        return 0, 0

    labels, count = label(flagged, structure=np.ones((3, 3), dtype=int))
    if not count:
        return literal, 0
    sizes = np.bincount(labels.ravel())
    isolated = sizes <= MAX_SPIKE_CLUSTER_PX
    isolated[0] = False
    return literal, int(isolated[labels].sum())


def pct_flat(elevation: np.ndarray) -> float:
    """Percentage of the surface with essentially no slope (water, slabs)."""
    d_row, d_col = np.gradient(elevation.astype(np.float64))
    return float(100.0 * np.mean(np.hypot(d_row, d_col) < FLAT_EPS))


def measure(
    heightmap: object,
    exaggeration: float,
    sigma: float,
    width_mm: float = REFERENCE_WIDTH_MM,
) -> dict[str, object]:
    """Compute the full metric set for one rendered scene."""
    elevation = heightmap.elevation  # type: ignore[attr-defined]
    printed_span = float(np.ptp(elevation))
    true_relief = printed_span / exaggeration if exaggeration else printed_span

    ground_width_m, _ = ground_extent(*heightmap.bbox)  # type: ignore[attr-defined]
    z_scale = width_mm / ground_width_m
    printed_relief_mm = printed_span * z_scale

    literal, isolated = spike_counts(elevation)
    return {
        "relief_m": round(true_relief, 1),
        "printed_relief_mm": round(printed_relief_mm, 2),
        "printed_relief_pct_of_width": round(100.0 * printed_relief_mm / width_mm, 2),
        "exaggeration_used": round(float(exaggeration), 3),
        "smooth_sigma_used": round(float(sigma), 3),
        "smooth_ground_m": round(float(sigma) * heightmap.meters_per_px, 1),  # type: ignore[attr-defined]
        "roughness": round(roughness(elevation), 4),
        "spike_count": literal,
        "isolated_spike_count": isolated,
        "pct_flat": round(pct_flat(elevation), 2),
        "meters_per_px": round(heightmap.meters_per_px, 2),  # type: ignore[attr-defined]
        "zoom": heightmap.zoom,  # type: ignore[attr-defined]
        "grid": [elevation.shape[1], elevation.shape[0]],
        "reference_width_mm": width_mm,
    }


def render_scene(
    scene: Scene,
    out_dir: Path,
    overrides: dict[str, float] | None = None,
    target_px: int = 800,
) -> dict[str, object]:
    """Render one scene through the full pipeline and write PNG + JSON."""
    overrides = overrides or {}
    south, west, north, east = scene.bbox

    forced_sigma = overrides.get("smooth")
    heightmap = build_heightmap(
        south,
        west,
        north,
        east,
        target_px=target_px,
        exaggeration=1.0,
        smooth_px=("auto" if forced_sigma is None else (forced_sigma or None)),
    )
    sigma = auto_smooth_sigma(heightmap.meters_per_px) if forced_sigma is None else forced_sigma

    if "exaggeration" in overrides:
        factor = float(overrides["exaggeration"])
    else:
        ground_width_m, _ = ground_extent(*heightmap.bbox)
        factor = auto_exaggeration(float(np.ptp(heightmap.elevation)), ground_width_m)

    if factor != 1.0:
        heightmap = type(heightmap)(
            elevation=exaggerate(heightmap.elevation, factor),
            meters_per_px=heightmap.meters_per_px,
            bbox=heightmap.bbox,
            zoom=heightmap.zoom,
            exaggeration=factor,
            requested_bbox=heightmap.requested_bbox,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    preview.render(heightmap).save(out_dir / f"{scene.name}.png")

    metrics = measure(heightmap, factor, sigma)
    metrics["scene"] = scene.name
    metrics["terrain"] = scene.terrain
    metrics["bbox"] = list(scene.bbox)
    (out_dir / f"{scene.name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(
    results: list[tuple[Scene, dict[str, object]]],
    out_dir: Path,
    panel: int = 430,
) -> Path:
    """Composite every scene into one sheet, captioned with its parameters."""
    title_font, body_font = _font(19), _font(15)
    caption_h, header_h, gap = 122, 34, 10

    frames: list[Image.Image] = []
    for scene, _ in results:
        with Image.open(out_dir / f"{scene.name}.png") as src:
            frame = src.copy()
        frame.thumbnail((panel, panel), Image.LANCZOS)
        frames.append(frame)

    sheet_w = len(frames) * panel + (len(frames) + 1) * gap
    sheet_h = header_h + panel + caption_h + gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 9), "terrframe gallery — full pipeline, current constants", font=title_font, fill=(240, 240, 245))

    for index, (frame, (scene, m)) in enumerate(zip(frames, results)):
        left = gap + index * (panel + gap)
        # Centre each frame inside its square cell.
        sheet.paste(frame, (left + (panel - frame.width) // 2, header_h + (panel - frame.height) // 2))

        y = header_h + panel + 6
        draw.text((left, y), scene.name, font=title_font, fill=(255, 255, 255))
        draw.text((left + 150, y + 3), scene.terrain, font=body_font, fill=(150, 150, 160))
        lines = [
            f"exag x{m['exaggeration_used']}   sigma {m['smooth_sigma_used']} px ({m['smooth_ground_m']} m)",
            f"relief {m['relief_m']} m -> {m['printed_relief_mm']} mm  ({m['printed_relief_pct_of_width']}% of width)",
            f"roughness {m['roughness']}   flat {m['pct_flat']}%",
            f"spikes {m['spike_count']} literal / {m['isolated_spike_count']} isolated",
        ]
        for offset, line in enumerate(lines):
            band = m["printed_relief_pct_of_width"]
            colour = (255, 255, 255)
            if offset == 1:
                colour = (130, 230, 150) if 12.0 <= band <= 25.0 else (245, 160, 120)
            if offset == 3 and m["isolated_spike_count"]:
                colour = (245, 160, 120)
            draw.text((left, y + 26 + offset * 20), line, font=body_font, fill=colour)

    destination = out_dir / "contact_sheet.png"
    sheet.save(destination)
    return destination


def _parse_params(text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise argparse.ArgumentTypeError(f"--params wants key=value, got {chunk!r}")
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in {"exaggeration", "smooth"}:
            raise argparse.ArgumentTypeError(f"unknown param {key!r}; expected exaggeration or smooth")
        try:
            params[key] = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{key} must be a number: {exc}") from exc
    return params


def main(argv: list[str] | None = None) -> int:
    """Render the scene suite. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="gallery.py", description="Render the fixed scene suite and measure it."
    )
    parser.add_argument("--scene", choices=[s.name for s in SCENES], help="render just one scene")
    parser.add_argument("--params", type=_parse_params, default={}, metavar="K=V,...",
                        help="override exaggeration and/or smooth for a sweep")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--target-px", type=int, default=800, help="sampling detail")
    args = parser.parse_args(argv)

    scenes = [s for s in SCENES if args.scene is None or s.name == args.scene]
    if args.params:
        print(f"overrides: {args.params}", flush=True)

    header = f"{'scene':14}{'exag':>7}{'sigma':>7}{'relief_m':>10}{'print_mm':>10}{'%width':>8}{'rough':>9}{'flat%':>7}{'spikes':>13}"
    print(header)
    print("-" * len(header))

    results: list[tuple[Scene, dict[str, object]]] = []
    for scene in scenes:
        metrics = render_scene(scene, args.out, args.params, args.target_px)
        results.append((scene, metrics))
        flag = "" if 12.0 <= metrics["printed_relief_pct_of_width"] <= 25.0 else "  <-- outside 12-25%"
        print(
            f"{scene.name:14}{metrics['exaggeration_used']:>7}{metrics['smooth_sigma_used']:>7}"
            f"{metrics['relief_m']:>10}{metrics['printed_relief_mm']:>10}"
            f"{metrics['printed_relief_pct_of_width']:>8}{metrics['roughness']:>9}"
            f"{metrics['pct_flat']:>7}"
            f"{str(metrics['spike_count']) + '/' + str(metrics['isolated_spike_count']):>13}{flag}",
            flush=True,
        )

    if len(results) > 1:
        sheet = contact_sheet(results, args.out)
        print(f"\nwrote {sheet}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
