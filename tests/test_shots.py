"""Tests for the multi-view shot runner. Offline: synthetic meshes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "shots.py"
_spec = importlib.util.spec_from_file_location("shots", _PATH)
assert _spec is not None and _spec.loader is not None
shots = importlib.util.module_from_spec(_spec)
sys.modules["shots"] = shots
_spec.loader.exec_module(shots)

render3d = shots.render3d

from terrframe.heightmap import Heightmap  # noqa: E402
from terrframe.mesh import export, heightmap_to_mesh  # noqa: E402


def _model(tmp_path: Path, name: str = "peak", rows: int = 30, cols: int = 50) -> Path:
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist = np.hypot(yy - rows / 2, xx - cols / 2)
    elevation = (1200.0 * np.exp(-(dist**2) / (2 * (rows / 4) ** 2)) + 150.0).astype(np.float32)
    hm = Heightmap(elevation, 50.0, (46.75, -121.95, 46.95, -121.55), 12)
    return export(heightmap_to_mesh(hm), tmp_path / f"{name}.stl")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def test_view_presets_include_iso_and_top() -> None:
    assert "iso" in render3d.VIEWS and "top" in render3d.VIEWS
    assert render3d.VIEWS["top"] == (0.0, 90.0), "top-down must be north-up"


def test_top_view_basis_is_valid_at_90_degrees() -> None:
    """Elevation 90 is the gimbal case; the basis must stay orthonormal."""
    right, up, forward = render3d._basis(*render3d.VIEWS["top"])
    for vector in (right, up, forward):
        assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert right @ up == pytest.approx(0.0, abs=1e-9)
    assert forward == pytest.approx(np.array([0.0, 0.0, -1.0]))


def test_top_view_matches_the_model_footprint(tmp_path: Path) -> None:
    """Straight down, the rendered aspect should track the x/y extents."""
    mesh = render3d.load_mesh(_model(tmp_path))
    img = render3d.render_mesh(mesh, width=300, **dict(zip(("azimuth", "elevation"), render3d.VIEWS["top"])))

    extents = mesh.extents
    drawn = np.any(np.asarray(img) != np.array(render3d.BACKGROUND_RGB), axis=-1)
    rows = np.where(drawn.any(axis=1))[0]
    cols = np.where(drawn.any(axis=0))[0]
    aspect = (rows[-1] - rows[0]) / (cols[-1] - cols[0])
    assert aspect == pytest.approx(extents[1] / extents[0], rel=0.08)


def test_speckle_fill_closes_gaps_without_touching_background() -> None:
    """Point splatting leaves pinholes; only enclosed ones may be filled."""
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    canvas[:, :] = render3d.BACKGROUND_RGB
    drawn = np.zeros((20, 20), dtype=bool)
    drawn[5:15, 5:15] = True
    canvas[drawn] = (200, 200, 200)

    drawn[9, 9] = False  # a pinhole inside the shape
    canvas[9, 9] = render3d.BACKGROUND_RGB

    filled = render3d._fill_speckle(canvas, drawn)

    assert tuple(filled[9, 9]) != tuple(render3d.BACKGROUND_RGB), "pinhole should close"
    assert tuple(filled[0, 0]) == tuple(render3d.BACKGROUND_RGB), "background must stay"
    assert tuple(filled[19, 19]) == tuple(render3d.BACKGROUND_RGB)


def test_sample_budget_scales_with_output_pixels(tmp_path: Path) -> None:
    """A tall top-down frame needs more samples than a short isometric one."""
    mesh = render3d.load_mesh(_model(tmp_path, rows=60, cols=20))
    top = render3d.render_mesh(mesh, width=200, azimuth=0.0, elevation=90.0)
    arr = np.asarray(top)
    drawn = np.any(arr != np.array(render3d.BACKGROUND_RGB), axis=-1)

    # Inside the silhouette's bounding box, coverage should be near-total.
    rows = np.where(drawn.any(axis=1))[0]
    cols = np.where(drawn.any(axis=0))[0]
    inner = drawn[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    assert inner.mean() > 0.95, f"only {inner.mean():.1%} covered; speckled"


# ---------------------------------------------------------------------------
# Run folders
# ---------------------------------------------------------------------------


def test_run_folder_is_timestamped(tmp_path: Path) -> None:
    """Folder names are for humans: dashed date, 12-hour clock, zone."""
    import re

    model = _model(tmp_path)
    assert shots.main([str(model), "--shots-dir", str(tmp_path / "shots"), "--width", "150"]) == 0

    runs = list((tmp_path / "shots").iterdir())
    assert len(runs) == 1
    # e.g. 2026-08-13_11-27-00PM-EDT
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[AP]M-\w+", runs[0].name), runs[0].name


def test_each_model_gets_every_view_plus_a_strip(tmp_path: Path) -> None:
    model = _model(tmp_path)
    shots.main([str(model), "--shots-dir", str(tmp_path / "s"), "--width", "150"])
    run = next((tmp_path / "s").iterdir())

    assert (run / "peak" / "peak_iso.png").is_file()
    assert (run / "peak" / "peak_top.png").is_file()
    assert (run / "peak" / "peak.png").is_file(), "side-by-side strip"
    assert (run / "contact_sheet_iso.png").is_file()
    assert (run / "contact_sheet_top.png").is_file()


def test_strip_places_views_side_by_side(tmp_path: Path) -> None:
    model = _model(tmp_path)
    shots.main([str(model), "--shots-dir", str(tmp_path / "s"), "--width", "150"])
    run = next((tmp_path / "s").iterdir())

    with Image.open(run / "peak" / "peak.png") as strip, Image.open(
        run / "peak" / "peak_iso.png"
    ) as single:
        assert strip.width > single.width, "two panels across"


def test_manifest_records_geometry_and_paths(tmp_path: Path) -> None:
    model = _model(tmp_path)
    shots.main([str(model), "--shots-dir", str(tmp_path / "s"), "--width", "150", "--label", "x"])
    run = next((tmp_path / "s").iterdir())
    assert run.name.endswith("_x")

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["views"] == ["iso", "top"]
    entry = manifest["models"][0]
    assert entry["model"] == "peak.stl"
    assert entry["watertight"] is True
    assert entry["faces"] > 0 and len(entry["extents_mm"]) == 3
    for relative in entry["views"].values():
        assert (run / relative).is_file()


def test_runs_never_collide(tmp_path: Path) -> None:
    model = _model(tmp_path)
    for label in ("a", "b"):
        shots.main(
            [str(model), "--shots-dir", str(tmp_path / "s"), "--width", "120", "--label", label]
        )
    assert len(list((tmp_path / "s").iterdir())) == 2


def test_single_view_selection(tmp_path: Path) -> None:
    model = _model(tmp_path)
    shots.main(
        [str(model), "--shots-dir", str(tmp_path / "s"), "--width", "120", "--views", "top"]
    )
    run = next((tmp_path / "s").iterdir())
    assert (run / "peak" / "peak_top.png").is_file()
    assert not (run / "peak" / "peak_iso.png").exists()


def test_unknown_view_is_rejected(tmp_path: Path) -> None:
    model = _model(tmp_path)
    code = shots.main(
        [str(model), "--shots-dir", str(tmp_path / "s"), "--views", "sideways"]
    )
    assert code == 2


def test_no_models_reports_cleanly(tmp_path: Path) -> None:
    assert shots.main([str(tmp_path / "empty"), "--shots-dir", str(tmp_path / "s")]) != 0


def test_discover_models_expands_a_directory(tmp_path: Path) -> None:
    _model(tmp_path, "a")
    _model(tmp_path, "b")
    found = shots.discover_models([tmp_path])
    assert {p.name for p in found} == {"a.stl", "b.stl"}
