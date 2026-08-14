#!/usr/bin/env python
"""Render finished models from several angles into a timestamped run folder.

Each model gets one shot per view plus a side-by-side strip, and the run gets a
contact sheet of every model. Nothing is overwritten, so a run can be compared
against an earlier one after a pipeline or styling change.

Top-down is rendered by default alongside the isometric because that is how a
finished plaque is usually looked at: it shows footprint, water shapes and
outline, where the isometric shows relief and the base.

    python scripts/shots.py                       # every .stl in the repo root
    python scripts/shots.py dc_core_natural.stl dc_district.stl
    python scripts/shots.py --views top --width 1400 --label tidal
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render3d  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHOTS_DIR = REPO_ROOT / "shots"

#: Views rendered when none are named. Order is the order they appear in strips.
DEFAULT_VIEWS = ("iso", "top")

MODEL_SUFFIXES = (".stl", ".3mf")


@dataclass
class ShotResult:
    """What was rendered for one model."""

    model: str
    views: dict[str, str]
    strip: str
    vertices: int
    faces: int
    extents_mm: list[float]
    watertight: bool
    file_mb: float


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(image: Image.Image, box: int) -> Image.Image:
    """Scale an image to fit inside a ``box``-square, preserving aspect."""
    copy = image.copy()
    copy.thumbnail((box, box), Image.LANCZOS)
    return copy


def _strip(
    panels: list[tuple[str, Image.Image]],
    title: str,
    subtitle: str,
    destination: Path,
) -> Path:
    """Lay panels side by side under a title."""
    title_font, body_font, tag_font = _font(24), _font(15), _font(17)
    gap, header, caption = 12, 62, 30

    # Views have wildly different aspects -- a top-down shot of a tall bbox is
    # portrait while the isometric is landscape -- so fit each into a common
    # box rather than padding everything to the tallest, which leaves the
    # strip mostly empty.
    box = max(max(img.width, img.height) for _, img in panels)
    box = min(box, 900)
    panels = [(label, _fit(img, box)) for label, img in panels]

    width = max(img.width for _, img in panels)
    height = max(img.height for _, img in panels)
    count = len(panels)

    sheet = Image.new(
        "RGB", (count * width + (count + 1) * gap, header + height + caption + gap), (16, 16, 18)
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), title, font=title_font, fill=(242, 242, 247))
    draw.text((gap, 38), subtitle, font=body_font, fill=(160, 160, 170))

    for index, (label, img) in enumerate(panels):
        left = gap + index * (width + gap)
        # Centre both ways: views differ in aspect, and top-aligning a short
        # isometric next to a tall top-down leaves the strip looking broken.
        sheet.paste(img, (left + (width - img.width) // 2, header + (height - img.height) // 2))
        draw.text((left, header + height + 6), label, font=tag_font, fill=(225, 225, 232))

    sheet.save(destination)
    return destination


def render_model(
    path: Path,
    run_dir: Path,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    width: int = 900,
) -> ShotResult:
    """Render one model in every requested view, plus a combined strip."""
    mesh = render3d.load_mesh(path)
    stem = path.stem
    model_dir = run_dir / stem
    model_dir.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Image.Image]] = []
    written: dict[str, str] = {}

    for view in views:
        azimuth, elevation = render3d.VIEWS[view]
        image = render3d.render_mesh(mesh, width=width, azimuth=azimuth, elevation=elevation)
        out = model_dir / f"{stem}_{view}.png"
        image.save(out)
        written[view] = out.relative_to(run_dir).as_posix()
        panels.append((view, image))

    extents = [round(float(v), 1) for v in mesh.extents]
    subtitle = (
        f"{extents[0]} x {extents[1]} x {extents[2]} mm   "
        f"{len(mesh.faces):,} faces   watertight {mesh.is_watertight}"
    )
    strip = _strip(panels, stem, subtitle, model_dir / f"{stem}.png")

    return ShotResult(
        model=path.name,
        views=written,
        strip=strip.relative_to(run_dir).as_posix(),
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        extents_mm=extents,
        watertight=bool(mesh.is_watertight),
        file_mb=round(path.stat().st_size / 1e6, 1),
    )


def contact_sheet(
    results: list[ShotResult],
    run_dir: Path,
    view: str,
    panel: int = 420,
) -> Path:
    """One panel per model, all in the same view, for scanning a whole run."""
    title_font, body_font = _font(22), _font(14)
    gap, header, caption = 10, 40, 44

    frames: list[tuple[str, Image.Image]] = []
    for result in results:
        if view not in result.views:
            continue
        with Image.open(run_dir / result.views[view]) as src:
            frame = src.copy()
        frame.thumbnail((panel, panel), Image.LANCZOS)
        frames.append((result.model, frame))

    if not frames:
        raise ValueError(f"no panels rendered for view {view!r}")

    sheet = Image.new(
        "RGB",
        (len(frames) * panel + (len(frames) + 1) * gap, header + panel + caption),
        (16, 16, 18),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 12), f"terrframe models — {view} view", font=title_font, fill=(242, 242, 247))

    for index, (name, frame) in enumerate(frames):
        left = gap + index * (panel + gap)
        sheet.paste(frame, (left + (panel - frame.width) // 2, header + (panel - frame.height) // 2))
        draw.text((left, header + panel + 8), name, font=body_font, fill=(225, 225, 232))

    destination = run_dir / f"contact_sheet_{view}.png"
    sheet.save(destination)
    return destination


def discover_models(paths: list[Path]) -> list[Path]:
    """Expand the given paths, or find every model in the repo root."""
    if not paths:
        return sorted(
            p for p in REPO_ROOT.iterdir() if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
        )

    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in MODEL_SUFFIXES))
        else:
            found.append(path)
    return found


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shots.py",
        description="Render models from several angles into a timestamped run folder.",
    )
    parser.add_argument("models", nargs="*", type=Path, help="model files or a directory")
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help=f"comma-separated views from {sorted(render3d.VIEWS)} (default: iso,top)",
    )
    parser.add_argument("--width", type=int, default=900, help="pixel width per shot")
    parser.add_argument(
        "--shots-dir", type=Path, default=DEFAULT_SHOTS_DIR, help="where run folders are created"
    )
    parser.add_argument("--label", default="", help="suffix appended to the run folder name")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render every model. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    unknown = [v for v in views if v not in render3d.VIEWS]
    if unknown:
        print(
            f"shots: unknown view(s) {unknown}; expected {sorted(render3d.VIEWS)}", file=sys.stderr
        )
        return 2

    models = discover_models(args.models)
    if not models:
        print("shots: no .stl or .3mf files found", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.shots_dir / (f"{stamp}_{args.label}" if args.label else stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"run {run_dir}", flush=True)

    results: list[ShotResult] = []
    for path in models:
        if not path.exists():
            print(f"  {path.name}: missing, skipped", file=sys.stderr)
            continue
        result = render_model(path, run_dir, views, args.width)
        results.append(result)
        print(
            f"  {result.model:24} {result.extents_mm[0]:>6.1f} x {result.extents_mm[1]:>6.1f} x"
            f" {result.extents_mm[2]:>5.1f} mm   {result.faces:>9,} faces   "
            f"watertight {str(result.watertight):5}  {result.file_mb:>5.1f} MB",
            flush=True,
        )

    if not results:
        # Every named model was missing. Don't leave an empty run folder behind
        # to be mistaken for a real run later.
        for leftover in sorted(run_dir.rglob("*"), reverse=True):
            leftover.rmdir() if leftover.is_dir() else leftover.unlink()
        run_dir.rmdir()
        print("shots: nothing rendered", file=sys.stderr)
        return 1

    sheets = [contact_sheet(results, run_dir, view).name for view in views]

    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "views": list(views),
        "width": args.width,
        "contact_sheets": sheets,
        "models": [asdict(r) for r in results],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{len(results)} model(s), {len(views)} view(s) each")
    for name in sheets:
        print(f"      {run_dir / name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
