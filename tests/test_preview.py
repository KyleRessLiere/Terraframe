"""Tests for the preview renderer. Offline: heightmaps are constructed directly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from terrframe.heightmap import Heightmap

# scripts/ is not an installed package, so load the module by path.
_PREVIEW_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preview.py"
_spec = importlib.util.spec_from_file_location("preview", _PREVIEW_PATH)
assert _spec is not None and _spec.loader is not None
preview = importlib.util.module_from_spec(_spec)
sys.modules["preview"] = preview
_spec.loader.exec_module(preview)


def _ridge_heightmap(rows: int = 64, cols: int = 96, water: bool = False) -> Heightmap:
    """A smooth ridge, optionally with a flat sea in one corner."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    elevation = (
        800.0
        + 600.0 * np.sin(xx / cols * np.pi)
        + 300.0 * np.cos(yy / rows * np.pi * 2.0)
    ).astype(np.float32)
    if water:
        elevation[:, : cols // 3] = 0.0
    return Heightmap(
        elevation=elevation,
        meters_per_px=30.0,
        bbox=(46.75, -121.95, 46.95, -121.55),
        zoom=12,
    )


def test_hillshade_is_normalised_and_lit_from_the_north_west() -> None:
    """A slope facing north-west is brighter than one facing south-east."""
    rows = cols = 32
    yy, xx = np.mgrid[0:rows, 0:cols]

    # A cone: north-west flank faces the light, south-east flank faces away.
    elevation = (-np.hypot(xx - cols / 2, yy - rows / 2) * 20.0).astype(np.float32)
    shade = preview.hillshade(elevation, meters_per_px=30.0)

    assert shade.shape == elevation.shape
    assert shade.min() >= 0.0 and shade.max() <= 1.0

    nw = shade[4:12, 4:12].mean()
    se = shade[-12:-4, -12:-4].mean()
    assert nw > se, "north-west light must brighten the NW flank"


def test_hillshade_of_flat_ground_is_uniform() -> None:
    flat = np.full((16, 16), 250.0, dtype=np.float32)
    shade = preview.hillshade(flat, meters_per_px=30.0)
    assert np.allclose(shade, shade[0, 0])


def test_elevation_tint_spans_the_ramp() -> None:
    """Low ground reads green, peaks read pale; output stays in range."""
    elevation = np.linspace(0.0, 4000.0, 256, dtype=np.float32)[None, :]
    rgb = preview.elevation_tint(elevation)

    assert rgb.shape == (1, 256, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    low, high = rgb[0, 0], rgb[0, -1]
    assert low[1] > low[2], "low ground is green-dominant"
    assert high.mean() > low.mean(), "peaks are brighter than valleys"


def test_elevation_tint_handles_perfectly_flat_input() -> None:
    """A flat array must not divide by zero."""
    rgb = preview.elevation_tint(np.full((8, 8), 100.0, dtype=np.float32))
    assert np.isfinite(rgb).all()


def test_render_produces_a_correctly_sized_rgb_image() -> None:
    hm = _ridge_heightmap()
    img = preview.render(hm)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (hm.elevation.shape[1], hm.elevation.shape[0])

    # Shaded relief should actually vary, not come out as a flat wash.
    arr = np.asarray(img, dtype=np.float32)
    assert arr.std() > 5.0


def test_render_colours_flat_water_as_water() -> None:
    """A flattened sea is drawn blue, not as the bottom of the land ramp."""
    hm = _ridge_heightmap(water=True)
    arr = np.asarray(preview.render(hm), dtype=np.float32)

    sea = arr[:, :10]
    land = arr[:, -10:]
    assert sea[..., 2].mean() > sea[..., 0].mean(), "water is blue-dominant"
    assert sea[..., 2].mean() > land[..., 2].mean()


def test_parse_bbox_accepts_and_rejects() -> None:
    assert preview.parse_bbox("46.75,-121.95,46.95,-121.55") == (
        46.75,
        -121.95,
        46.95,
        -121.55,
    )
    for bad in ["1,2,3", "a,b,c,d", "48,-122,47,-121", "47,-121,48,-122"]:
        with pytest.raises(Exception):
            preview.parse_bbox(bad)


def test_main_writes_a_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI wires argument parsing through to a written file."""
    monkeypatch.setattr(preview, "build_heightmap", lambda *a, **k: _ridge_heightmap())

    out = tmp_path / "nested" / "preview.png"
    code = preview.main(["--bbox", "46.75,-121.95,46.95,-121.55", "-o", str(out)])

    assert code == 0
    assert out.is_file()
    with Image.open(out) as img:
        assert img.size == (96, 64)
