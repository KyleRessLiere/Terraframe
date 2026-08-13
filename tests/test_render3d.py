"""Tests for the software mesh renderer. Offline: meshes are synthetic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render3d.py"
_spec = importlib.util.spec_from_file_location("render3d", _PATH)
assert _spec is not None and _spec.loader is not None
render3d = importlib.util.module_from_spec(_spec)
sys.modules["render3d"] = render3d
_spec.loader.exec_module(render3d)

from terrframe.heightmap import Heightmap  # noqa: E402
from terrframe.mesh import export, heightmap_to_mesh  # noqa: E402


def _peak_mesh(rows: int = 40, cols: int = 60) -> trimesh.Trimesh:
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist = np.hypot(yy - rows / 2, xx - cols / 2)
    elevation = (1500.0 * np.exp(-(dist**2) / (2 * (rows / 4) ** 2)) + 200.0).astype(np.float32)
    hm = Heightmap(
        elevation=elevation,
        meters_per_px=50.0,
        bbox=(46.75, -121.95, 46.95, -121.55),
        zoom=12,
    )
    return heightmap_to_mesh(hm)


def test_render_returns_an_image_of_the_requested_width() -> None:
    img = render3d.render_mesh(_peak_mesh(), width=240)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.width == 240
    assert img.height > 0


def test_render_draws_the_mesh_not_just_background() -> None:
    """A rendered solid must cover a healthy share of the frame."""
    img = render3d.render_mesh(_peak_mesh(), width=240)
    arr = np.asarray(img)

    background = np.array(render3d.BACKGROUND_RGB, dtype=np.uint8)
    drawn = np.any(arr != background, axis=-1)
    assert drawn.mean() > 0.25, "mesh should fill much of the frame"


def test_render_shades_by_orientation() -> None:
    """Faces turned toward the light come out brighter than those away."""
    img = render3d.render_mesh(_peak_mesh(), width=240)
    arr = np.asarray(img).astype(np.float64)
    background = np.array(render3d.BACKGROUND_RGB, dtype=np.float64)
    drawn = np.any(arr != background, axis=-1)

    lit = arr[drawn].mean(axis=-1)
    assert lit.std() > 8.0, "a flat wash means shading is not working"
    assert lit.max() <= 255.0 and lit.min() >= 0.0


def test_area_weighted_sampling_covers_the_base_walls() -> None:
    """Skirt walls are few but large; per-face sampling would starve them.

    This is the bug the renderer was written around: sampling by triangle
    count draws the base as a few thin lines instead of a solid slab.
    """
    mesh = _peak_mesh()
    points, normals = render3d._surface_samples(mesh, 200_000)

    assert len(points) == len(normals)
    assert len(points) > len(mesh.vertices)

    # Wall samples are the ones with near-horizontal normals.
    horizontal = np.abs(normals[:, 2]) < 0.1
    assert horizontal.sum() > 100, "the skirt walls must receive real coverage"

    # And they must span the full base height, not cling to the terrain edge.
    wall_z = points[horizontal][:, 2]
    assert wall_z.min() < 0.5, "walls should reach the z=0 floor"


def test_basis_vectors_are_orthonormal() -> None:
    for azimuth in (0.0, 90.0, 315.0):
        for elevation in (10.0, 32.0, 80.0):
            right, up, forward = render3d._basis(azimuth, elevation)
            for vector in (right, up, forward):
                assert np.linalg.norm(vector) == pytest.approx(1.0)
            assert right @ up == pytest.approx(0.0, abs=1e-9)
            assert right @ forward == pytest.approx(0.0, abs=1e-9)
            assert up @ forward == pytest.approx(0.0, abs=1e-9)


def test_different_viewpoints_give_different_images() -> None:
    mesh = _peak_mesh()
    a = np.asarray(render3d.render_mesh(mesh, width=200, azimuth=315.0))
    b = np.asarray(render3d.render_mesh(mesh, width=200, azimuth=135.0))
    assert not np.array_equal(a, b)


def test_round_trip_through_an_exported_stl(tmp_path: Path) -> None:
    """The renderer reads the artefact a slicer would, not an in-memory mesh."""
    path = export(_peak_mesh(), tmp_path / "peak.stl")
    mesh = render3d.load_mesh(path)

    assert mesh.is_watertight
    img = render3d.render_mesh(mesh, width=200)
    assert img.width == 200


def test_main_writes_a_png(tmp_path: Path) -> None:
    stl = export(_peak_mesh(), tmp_path / "peak.stl")
    out = tmp_path / "shots" / "peak.png"

    assert render3d.main([str(stl), "-o", str(out), "--width", "180"]) == 0
    assert out.is_file()
    with Image.open(out) as img:
        assert img.width == 180


def test_main_defaults_output_next_to_input(tmp_path: Path) -> None:
    stl = export(_peak_mesh(), tmp_path / "peak.stl")
    assert render3d.main([str(stl), "--width", "150"]) == 0
    assert (tmp_path / "peak.png").is_file()
