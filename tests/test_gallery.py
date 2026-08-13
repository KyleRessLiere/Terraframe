"""Tests for the visual regression harness. Offline: tile fetching is stubbed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from terrframe import heightmap as hm
from terrframe.tiles import TILE_SIZE

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gallery.py"
_spec = importlib.util.spec_from_file_location("gallery", _PATH)
assert _spec is not None and _spec.loader is not None
gallery = importlib.util.module_from_spec(_spec)
sys.modules["gallery"] = gallery
_spec.loader.exec_module(gallery)


@pytest.fixture
def stub_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((x, y, zoom))) % (2**32))
        rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
        dome = np.sin(rows * np.pi) * np.sin(cols * np.pi)
        return (200.0 + 800.0 * dome + rng.normal(0.0, 5.0, (TILE_SIZE, TILE_SIZE))).astype(
            np.float32
        )

    monkeypatch.setattr(hm, "fetch_tile", _fetch)


# ---------------------------------------------------------------------------
# The frozen suite
# ---------------------------------------------------------------------------


def test_scene_suite_is_the_documented_five() -> None:
    """The suite is a baseline; changing it silently invalidates comparisons."""
    assert [s.name for s in gallery.SCENES] == [
        "tahoe",
        "stoneybrooke",
        "rainier",
        "sf_coast",
        "kansas",
    ]
    for scene in gallery.SCENES:
        south, west, north, east = scene.bbox
        assert south < north and west < east
        assert scene.terrain


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_roughness_responds_to_smoothing() -> None:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 20.0, (128, 128)).astype(np.float32)
    assert gallery.roughness(gaussian_filter(arr, 4.0)) < gallery.roughness(arr)


def test_spike_counts_separates_isolated_from_literal() -> None:
    """A ridge trips the literal test but is not an isolated spike."""
    yy, xx = np.mgrid[0:64, 0:64]
    ramp = (2.0 * xx + 1.0 * yy).astype(np.float32)

    ridge = ramp.copy()
    ridge[:, 30] += 300.0
    literal, isolated = gallery.spike_counts(ridge)
    assert literal > 0, "a sharp crest does deviate from its own local median"
    assert isolated == 0, "but a connected ridge is not a needle"

    needles = ramp.copy()
    for row, col in [(10, 12), (20, 40), (33, 7)]:
        needles[row, col] += 500.0
    assert gallery.spike_counts(needles)[1] > 0


def test_spike_counts_is_a_relative_measure_with_a_noise_floor() -> None:
    """Documents why ``isolated_spike_count`` never reaches 0 on real terrain.

    The threshold is 3x the IQR of the residual, so it rescales itself to
    whatever variation is present. On a perfectly analytic dome the count is
    a handful of pixels rather than a clean zero -- the apex, where curvature
    departs from a 5x5 median while the IQR has shrunk to match a noiseless
    surface. That floor is geometric, not numerical: float64 gives the same
    answer as float32. Add real texture and the count climbs by orders of
    magnitude, which is what the metric is actually good for.
    """
    yy, xx = np.mgrid[0:200, 0:200]
    dome = 1000.0 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / 2 / 50.0**2)

    clean = gallery.spike_counts(dome.astype(np.float32))[1]
    assert clean < 0.001 * dome.size, "an analytic surface is essentially spike-free"
    assert gallery.spike_counts(dome)[1] == clean, "the floor is geometric, not precision"

    rng = np.random.default_rng(0)
    noisy = gallery.spike_counts((dome + rng.normal(0.0, 1.0, dome.shape)).astype(np.float32))[1]
    assert noisy > 20 * max(clean, 1), "real texture dominates the count"


def test_pct_flat_finds_dead_flat_ground() -> None:
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[:, 32:] = np.arange(32, dtype=np.float32) * 10.0
    flat = gallery.pct_flat(arr)
    assert 30.0 < flat < 70.0

    yy, xx = np.mgrid[0:64, 0:64]
    assert gallery.pct_flat((xx * 5.0).astype(np.float32)) == pytest.approx(0.0)


def test_measure_reports_true_relief_not_exaggerated(stub_tiles: None) -> None:
    """relief_m is geographic; printed_relief_mm is what comes off the printer."""
    from terrframe.heightmap import build_heightmap, exaggerate

    base = build_heightmap(46.75, -121.95, 46.95, -121.55, target_px=200)
    tall = type(base)(
        elevation=exaggerate(base.elevation, 3.0),
        meters_per_px=base.meters_per_px,
        bbox=base.bbox,
        zoom=base.zoom,
        exaggeration=3.0,
    )

    plain_m = gallery.measure(base, 1.0, 0.5)
    tall_m = gallery.measure(tall, 3.0, 0.5)

    assert tall_m["relief_m"] == pytest.approx(plain_m["relief_m"], rel=1e-3)
    assert tall_m["printed_relief_mm"] == pytest.approx(3.0 * plain_m["printed_relief_mm"], rel=1e-3)
    assert tall_m["printed_relief_pct_of_width"] == pytest.approx(
        100.0 * tall_m["printed_relief_mm"] / gallery.REFERENCE_WIDTH_MM
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_scene_writes_png_and_json(stub_tiles: None, tmp_path: Path) -> None:
    scene = gallery.SCENES[2]  # rainier
    metrics = gallery.render_scene(scene, tmp_path, target_px=200)

    assert (tmp_path / "rainier.png").is_file()
    saved = json.loads((tmp_path / "rainier.json").read_text(encoding="utf-8"))
    assert saved == metrics
    for key in (
        "relief_m",
        "printed_relief_mm",
        "exaggeration_used",
        "smooth_sigma_used",
        "roughness",
        "spike_count",
        "pct_flat",
    ):
        assert key in saved


def test_params_override_exaggeration_and_smooth(stub_tiles: None, tmp_path: Path) -> None:
    scene = gallery.SCENES[2]
    metrics = gallery.render_scene(
        scene, tmp_path, overrides={"exaggeration": 2.5, "smooth": 3.0}, target_px=200
    )
    assert metrics["exaggeration_used"] == 2.5
    assert metrics["smooth_sigma_used"] == 3.0


def test_contact_sheet_covers_every_scene(stub_tiles: None, tmp_path: Path) -> None:
    results = [(s, gallery.render_scene(s, tmp_path, target_px=150)) for s in gallery.SCENES[:3]]
    sheet = gallery.contact_sheet(results, tmp_path)

    assert sheet.is_file()
    with Image.open(sheet) as img:
        assert img.width > img.height, "panels sit in a row"
        assert img.width > 3 * 400


def test_main_runs_one_scene(stub_tiles: None, tmp_path: Path) -> None:
    code = gallery.main(["--scene", "kansas", "--out", str(tmp_path), "--target-px", "150"])
    assert code == 0
    assert (tmp_path / "kansas.png").is_file()
    assert (tmp_path / "kansas.json").is_file()


def test_parse_params() -> None:
    assert gallery._parse_params("exaggeration=3,smooth=2") == {
        "exaggeration": 3.0,
        "smooth": 2.0,
    }
    assert gallery._parse_params("smooth=0") == {"smooth": 0.0}
    for bad in ["exaggeration", "gamma=2", "smooth=x"]:
        with pytest.raises(Exception):
            gallery._parse_params(bad)
