#!/usr/bin/env python
"""Render preview sweeps into a timestamped run directory.

Each invocation creates ``runs/<timestamp>/`` holding one subdirectory per
site, the rendered PNGs, a contact sheet per site, and a ``manifest.json``
recording every parameter and measurement. Nothing is ever overwritten, so
runs stay comparable after the tuning constants change.

    python scripts/batch.py                        # every preset
    python scripts/batch.py --preset tahoe
    python scripts/batch.py --bbox 38.85,-120.25,39.35,-119.85 \
        --smooth 0,1,2 --exaggeration 2 --name tahoe-x2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# preview.py is a sibling script, not an installed module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preview  # noqa: E402

from terrframe.heightmap import auto_smooth_sigma, build_heightmap  # noqa: E402
from terrframe.runstamp import run_stamp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"

#: Border trimmed before measuring roughness. gaussian_filter reflects at the
#: edges, which injects gradient variance that grows with sigma; that is a
#: boundary artefact, not terrain, and it is the one place the metric lies.
ROUGHNESS_BORDER_PX = 12


@dataclass
class Preset:
    """A named sweep: one bbox rendered at several smoothing levels."""

    name: str
    bbox: tuple[float, float, float, float]
    smooth: list[float]
    exaggeration: float = 1.0
    target_px: int = 800
    despike: bool = True
    note: str = ""


PRESETS: dict[str, Preset] = {
    "stoneybrooke": Preset(
        name="stoneybrooke",
        bbox=(38.7500, -77.1244, 38.7860, -77.0782),
        smooth=[1, 2, 3, 4],
        exaggeration=3.0,
        note="flat suburb; zoom 15, ~3.7 m/px. Auto sigma clamps at the ceiling here.",
    ),
    "tahoe": Preset(
        name="tahoe",
        bbox=(38.85, -120.25, 39.35, -119.85),
        smooth=[0, 1, 2],
        exaggeration=1.0,
        note="alpine; zoom 11, ~59 m/px. One pixel is already 59 m, so sigma 1 blurs hard.",
    ),
}


@dataclass
class RenderResult:
    """Everything measured about one rendered frame."""

    smooth: float
    despike: bool
    exaggeration: float
    image: str
    zoom: int
    meters_per_px: float
    grid: tuple[int, int]
    smooth_ground_m: float
    relief_m: float
    elevation_min_m: float
    elevation_max_m: float
    roughness: float
    roughness_vs_raw_pct: float | None = None


@dataclass
class SiteResult:
    """A whole sweep over one site."""

    name: str
    bbox: tuple[float, float, float, float]
    note: str
    target_px: int
    auto_smooth_sigma: float
    raw_roughness: float
    contact_sheet: str
    renders: list[RenderResult] = field(default_factory=list)


def roughness(elevation: np.ndarray, border: int = ROUGHNESS_BORDER_PX) -> float:
    """Spread of the surface's slopes -- how jagged it would print."""
    interior = elevation[border:-border, border:-border] if border else elevation
    if interior.size == 0:
        interior = elevation
    return float(np.std(np.gradient(interior)))


def _git_commit() -> str | None:
    """Current commit, so a run can be traced back to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _contact_sheet(
    images: list[Path],
    labels: list[str],
    destination: Path,
    max_width: int = 1600,
) -> Path:
    """Lay the sweep out side by side so the frames can be compared at a glance."""
    frames = [Image.open(path) for path in images]
    try:
        gap, header = 12, 26
        count = len(frames)
        budget = (max_width - gap * (count - 1)) / count
        scale = min(1.0, budget / frames[0].width)
        width = max(1, int(frames[0].width * scale))
        height = max(1, int(frames[0].height * scale))

        sheet = Image.new("RGB", (width * count + gap * (count - 1), height + header), (18, 18, 20))
        draw = ImageDraw.Draw(sheet)
        for index, (frame, text) in enumerate(zip(frames, labels)):
            left = index * (width + gap)
            sheet.paste(frame.resize((width, height), Image.LANCZOS), (left, header))
            draw.text((left + 6, 8), text, fill=(255, 255, 255))

        sheet.save(destination)
    finally:
        for frame in frames:
            frame.close()
    return destination


def run_preset(preset: Preset, run_dir: Path) -> SiteResult:
    """Render one preset's whole sweep into ``run_dir/<name>/``."""
    site_dir = run_dir / preset.name
    site_dir.mkdir(parents=True, exist_ok=True)

    south, west, north, east = preset.bbox

    # A raw, uncleaned build is the baseline every frame is scored against.
    raw = build_heightmap(
        south,
        west,
        north,
        east,
        target_px=preset.target_px,
        exaggeration=preset.exaggeration,
        smooth_px=None,
        despike=False,
    )
    raw_roughness = roughness(raw.elevation)

    site = SiteResult(
        name=preset.name,
        bbox=preset.bbox,
        note=preset.note,
        target_px=preset.target_px,
        auto_smooth_sigma=auto_smooth_sigma(raw.meters_per_px),
        raw_roughness=raw_roughness,
        contact_sheet="",
    )

    images: list[Path] = []
    labels: list[str] = []

    for sigma in preset.smooth:
        heightmap = build_heightmap(
            south,
            west,
            north,
            east,
            target_px=preset.target_px,
            exaggeration=preset.exaggeration,
            smooth_px=sigma or None,
            despike=preset.despike,
        )

        image_path = site_dir / f"smooth{sigma:g}.png"
        preview.render(heightmap).save(image_path)

        elevation = heightmap.elevation
        this_roughness = roughness(elevation)
        result = RenderResult(
            smooth=float(sigma),
            despike=preset.despike,
            exaggeration=preset.exaggeration,
            image=image_path.relative_to(run_dir).as_posix(),
            zoom=heightmap.zoom,
            meters_per_px=round(heightmap.meters_per_px, 3),
            grid=(elevation.shape[1], elevation.shape[0]),
            smooth_ground_m=round(sigma * heightmap.meters_per_px, 1),
            relief_m=round(float(np.ptp(elevation)), 1),
            elevation_min_m=round(float(elevation.min()), 1),
            elevation_max_m=round(float(elevation.max()), 1),
            roughness=round(this_roughness, 4),
            roughness_vs_raw_pct=(
                round((1.0 - this_roughness / raw_roughness) * 100.0, 1)
                if raw_roughness > 0
                else None
            ),
        )
        site.renders.append(result)
        images.append(image_path)
        labels.append(f"smooth {sigma:g} ({result.smooth_ground_m:.0f} m)")

        print(
            f"  smooth {sigma:<4g} {result.smooth_ground_m:6.0f} m  "
            f"relief {result.relief_m:8.1f} m  roughness {result.roughness:8.4f}  "
            f"({result.roughness_vs_raw_pct:+.1f}% vs raw)",
            flush=True,
        )

    sheet = _contact_sheet(images, labels, site_dir / "_contact.png")
    site.contact_sheet = sheet.relative_to(run_dir).as_posix()
    return site


def _parse_bbox(text: str) -> tuple[float, float, float, float]:
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--bbox needs 4 numbers (S,W,N,E), got {len(parts)}")
    try:
        south, west, north, east = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--bbox values must be numbers: {exc}") from exc
    if south > north or west > east:
        raise argparse.ArgumentTypeError("--bbox must be ordered S,W,N,E")
    return south, west, north, east


def _parse_sigmas(text: str) -> list[float]:
    try:
        values = [float(p) for p in text.split(",") if p.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--smooth takes comma-separated numbers: {exc}") from exc
    if not values:
        raise argparse.ArgumentTypeError("--smooth needs at least one value")
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--smooth values must be non-negative")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch.py",
        description="Render preview sweeps into a timestamped run directory.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="preset to run; repeatable. Default: all of them.",
    )
    parser.add_argument("--bbox", type=_parse_bbox, help="ad-hoc site as S,W,N,E")
    parser.add_argument(
        "--smooth",
        type=_parse_sigmas,
        default=[0, 1, 2, 3],
        help="comma-separated sigmas for --bbox (default: 0,1,2,3)",
    )
    parser.add_argument("--exaggeration", type=float, default=1.0, help="vertical relief multiplier")
    parser.add_argument("--target-px", type=int, default=800, help="pixel span of the longer side")
    parser.add_argument(
        "--despike",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="remove isolated outlier pixels (default: enabled)",
    )
    parser.add_argument("--name", default="site", help="folder name for an ad-hoc --bbox run")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"where run directories are created (default: {DEFAULT_RUNS_DIR.name}/)",
    )
    parser.add_argument("--label", default="", help="suffix appended to the run directory name")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render the requested sweeps. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    if args.bbox is not None:
        presets = [
            Preset(
                name=args.name,
                bbox=args.bbox,
                smooth=args.smooth,
                exaggeration=args.exaggeration,
                target_px=args.target_px,
                despike=args.despike,
                note="ad-hoc",
            )
        ]
    else:
        chosen = args.preset or sorted(PRESETS)
        presets = [PRESETS[name] for name in chosen]

    run_dir = args.runs_dir / run_stamp(args.label)
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"run {run_dir}", flush=True)

    sites: list[SiteResult] = []
    for preset in presets:
        print(f"\n{preset.name}  {preset.bbox}", flush=True)
        site = run_preset(preset, run_dir)
        print(
            f"  auto sigma would be {site.auto_smooth_sigma:g} px; "
            f"raw roughness {site.raw_roughness:.4f}",
            flush=True,
        )
        sites.append(site)

    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "git_commit": _git_commit(),
        "command": ["batch.py", *(argv if argv is not None else sys.argv[1:])],
        "sites": [asdict(site) for site in sites],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    frames = sum(len(site.renders) for site in sites)
    print(f"\nwrote {frames} frames across {len(sites)} site(s)")
    print(f"      {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
