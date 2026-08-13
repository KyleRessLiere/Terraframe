"""Tests for mesh construction and export. Offline: heightmaps are synthetic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from terrframe.heightmap import Heightmap
from terrframe.mesh import (
    AUTO_EXAGGERATION_MAX,
    AUTO_EXAGGERATION_MIN,
    TARGET_RELIEF_RATIO,
    auto_exaggeration,
    export,
    heightmap_to_mesh,
)

BBOX = (46.75, -121.95, 46.95, -121.55)


def make_heightmap(
    elevation: np.ndarray,
    meters_per_px: float = 50.0,
    exaggeration: float = 1.0,
) -> Heightmap:
    """Wrap a raw array as a Heightmap for meshing."""
    return Heightmap(
        elevation=np.asarray(elevation, dtype=np.float32),
        meters_per_px=meters_per_px,
        bbox=BBOX,
        zoom=12,
        exaggeration=exaggeration,
    )


# --- Terrain shapes used across the invariant tests -------------------------


def flat_plane(rows: int = 24, cols: int = 32) -> np.ndarray:
    return np.full((rows, cols), 500.0, dtype=np.float32)


def single_peak(rows: int = 24, cols: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist = np.hypot(yy - rows / 2, xx - cols / 2)
    return (2000.0 * np.exp(-(dist**2) / (2 * (rows / 5) ** 2)) + 300.0).astype(np.float32)


def random_noise(rows: int = 24, cols: int = 32) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.uniform(100.0, 1800.0, (rows, cols)).astype(np.float32)


def with_flat_water(rows: int = 24, cols: int = 32) -> np.ndarray:
    arr = single_peak(rows, cols)
    arr[:, : cols // 3] = 0.0  # a flattened sea along the west edge
    return arr


TERRAIN_CASES = {
    "flat_plane": flat_plane,
    "single_peak": single_peak,
    "random_noise": random_noise,
    "flattened_water": with_flat_water,
}


# ---------------------------------------------------------------------------
# Watertight invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TERRAIN_CASES))
def test_mesh_is_a_closed_solid(name: str) -> None:
    """Every terrain shape must produce a sealed, correctly wound solid."""
    mesh = heightmap_to_mesh(make_heightmap(TERRAIN_CASES[name]()))

    assert mesh.is_watertight, f"{name} produced an open mesh"
    assert mesh.is_winding_consistent, f"{name} has inconsistent winding"
    assert mesh.volume > 0.0, f"{name} has non-positive volume (inverted normals?)"
    assert mesh.euler_number == 2, f"{name} is not genus 0 (euler={mesh.euler_number})"


@pytest.mark.parametrize("name", sorted(TERRAIN_CASES))
def test_mesh_has_outward_normals_and_no_degenerate_faces(name: str) -> None:
    """Zero-area faces would make normals undefined and slicers unhappy."""
    mesh = heightmap_to_mesh(make_heightmap(TERRAIN_CASES[name]()))

    areas = mesh.area_faces
    assert (areas > 0.0).all(), f"{name} has {(areas <= 0).sum()} degenerate faces"
    # A closed solid with outward normals encloses exactly its convex-hull-free
    # volume; trimesh reports that as positive only when winding is outward.
    assert mesh.volume == pytest.approx(abs(mesh.volume))


def test_mesh_bottom_is_flat_and_terrain_sits_on_the_base() -> None:
    """Bottom at z=0, and the lowest valley floor exactly base_mm above it."""
    rows, cols = 24, 32
    base_mm = 6.0
    hm = make_heightmap(single_peak(rows, cols))
    mesh = heightmap_to_mesh(hm, base_mm=base_mm)

    assert mesh.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-9)

    # The terrain grid is the first rows*cols vertices, by construction.
    terrain_z = mesh.vertices[: rows * cols, 2]
    assert terrain_z.min() == pytest.approx(base_mm, abs=1e-9)


def test_zero_base_still_closes() -> None:
    """base_mm=0 pinches the lowest point onto the floor but stays watertight."""
    mesh = heightmap_to_mesh(make_heightmap(single_peak()), base_mm=0.0)
    assert mesh.is_watertight
    assert mesh.euler_number == 2


# ---------------------------------------------------------------------------
# Dimensions and scale
# ---------------------------------------------------------------------------


def test_requested_width_is_honoured() -> None:
    """200 mm asked for is 200 mm delivered, to well under a slicer's tolerance."""
    mesh = heightmap_to_mesh(make_heightmap(single_peak()), width_mm=200.0)
    assert mesh.extents[0] == pytest.approx(200.0, abs=0.1)


def test_z_extent_is_base_plus_scaled_relief() -> None:
    """Vertical size follows meters_per_px, not an arbitrary normalisation."""
    rows, cols = 24, 32
    mpp = 50.0
    width_mm = 200.0
    base_mm = 6.0

    elevation = single_peak(rows, cols)
    mesh = heightmap_to_mesh(
        make_heightmap(elevation, meters_per_px=mpp), width_mm=width_mm, base_mm=base_mm
    )

    ground_width_m = mpp * (cols - 1)
    z_scale = width_mm / ground_width_m
    relief_m = float(elevation.max() - elevation.min())
    expected_z = base_mm + relief_m * z_scale

    assert mesh.extents[2] == pytest.approx(expected_z, abs=0.1)


def test_vertical_scale_is_truthful_against_horizontal() -> None:
    """1 m of climb prints as the same mm as 1 m of ground travel."""
    rows, cols = 20, 40
    mpp = 25.0
    width_mm = 100.0

    elevation = np.zeros((rows, cols), dtype=np.float32)
    elevation[0, 0] = 1000.0  # a 1000 m spike

    mesh = heightmap_to_mesh(
        make_heightmap(elevation, meters_per_px=mpp), width_mm=width_mm, base_mm=0.0
    )

    mm_per_ground_m = width_mm / (mpp * (cols - 1))
    assert mesh.extents[2] == pytest.approx(1000.0 * mm_per_ground_m, abs=1e-6)


def test_exaggeration_baked_upstream_shows_up_in_z() -> None:
    """Exaggeration already applied to the array scales the print vertically."""
    rows, cols = 20, 30
    plain = single_peak(rows, cols)
    floor = plain.min()
    doubled = (floor + (plain - floor) * 2.0).astype(np.float32)

    a = heightmap_to_mesh(make_heightmap(plain), base_mm=0.0)
    b = heightmap_to_mesh(make_heightmap(doubled, exaggeration=2.0), base_mm=0.0)

    assert b.extents[2] == pytest.approx(2.0 * a.extents[2], rel=1e-6)
    assert b.extents[0] == pytest.approx(a.extents[0], rel=1e-9), "footprint must not change"


def test_footprint_follows_heightmap_aspect() -> None:
    """A 2:1 heightmap prints as a 2:1 footprint.

    Sized (201, 101) so the vertex span (rows-1)/(cols-1) is exactly 2; a
    (200, 100) grid spans 199/99 pitches, which is 2:1 only to within 1%.
    """
    elevation = single_peak(201, 101)
    mesh = heightmap_to_mesh(make_heightmap(elevation))

    size_x, size_y = mesh.extents[0], mesh.extents[1]
    assert size_y / size_x == pytest.approx(2.0, rel=1e-9)


def test_north_is_up_in_y() -> None:
    """Row 0 is the northern edge and must land at maximum y."""
    rows, cols = 16, 16
    elevation = np.zeros((rows, cols), dtype=np.float32)
    elevation[0, :] = 1000.0  # a ridge along the northern edge

    mesh = heightmap_to_mesh(make_heightmap(elevation), base_mm=1.0)

    terrain = mesh.vertices[: rows * cols]
    highest = terrain[terrain[:, 2] > terrain[:, 2].mean()]
    assert highest[:, 1].min() == pytest.approx(terrain[:, 1].max())


def test_rejects_bad_parameters() -> None:
    hm = make_heightmap(single_peak())
    with pytest.raises(ValueError):
        heightmap_to_mesh(hm, width_mm=0.0)
    with pytest.raises(ValueError):
        heightmap_to_mesh(hm, base_mm=-1.0)
    with pytest.raises(ValueError):
        heightmap_to_mesh(make_heightmap(np.zeros((1, 5), dtype=np.float32)))
    with pytest.raises(ValueError, match="fill_nodata"):
        bad = np.full((8, 8), 100.0, dtype=np.float32)
        bad[2, 2] = np.nan
        heightmap_to_mesh(make_heightmap(bad))


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_oversized_heightmap_is_downsampled_but_still_closed() -> None:
    """Past the vertex ceiling the grid is decimated, not truncated or dropped.

    Uses a small ceiling so the same code path runs in milliseconds; the
    default 4,000,000 differs only in arithmetic.
    """
    elevation = single_peak(400, 300)
    assert elevation.size > 5_000

    mesh = heightmap_to_mesh(make_heightmap(elevation), max_vertices=5_000)

    assert len(mesh.vertices) <= 5_000
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.euler_number == 2
    assert mesh.volume > 0.0


def test_downsampling_preserves_footprint_and_vertical_scale() -> None:
    """Decimation must not quietly change how big the model is."""
    elevation = single_peak(400, 300)
    hm = make_heightmap(elevation)

    full = heightmap_to_mesh(hm, max_vertices=4_000_000)
    small = heightmap_to_mesh(hm, max_vertices=5_000)

    assert len(full.vertices) > len(small.vertices)
    assert small.extents[0] == pytest.approx(full.extents[0], abs=0.1)
    assert small.extents[1] == pytest.approx(full.extents[1], rel=0.02)
    # Bilinear decimation clips the very peak slightly; scale, not shape.
    assert small.extents[2] == pytest.approx(full.extents[2], rel=0.10)


def test_small_heightmap_is_left_alone() -> None:
    """Under the ceiling nothing is resampled."""
    rows, cols = 24, 32
    mesh = heightmap_to_mesh(make_heightmap(single_peak(rows, cols)))
    expected = rows * cols + (2 * rows + 2 * cols - 4) + 1
    assert len(mesh.vertices) == expected


# ---------------------------------------------------------------------------
# auto_exaggeration
# ---------------------------------------------------------------------------


def test_auto_exaggeration_follows_the_documented_formula() -> None:
    """The factor is exactly target_ratio * width / relief, then clamped."""
    assert auto_exaggeration(2000.0, 30_000.0) == pytest.approx(
        TARGET_RELIEF_RATIO * 30_000.0 / 2000.0
    )


def test_auto_exaggeration_alpine_terrain_needs_little_help() -> None:
    """Steep country lands near the bottom of the range and never below 1."""
    # Relief at or above TARGET_RELIEF_RATIO of the width already prints well.
    steep = auto_exaggeration(0.18 * 30_000.0, 30_000.0)
    assert steep == pytest.approx(AUTO_EXAGGERATION_MIN)

    very_steep = auto_exaggeration(9000.0, 30_000.0)
    assert very_steep == AUTO_EXAGGERATION_MIN, "must never flatten real terrain"


def test_auto_exaggeration_plains_hit_the_ceiling() -> None:
    """80 m of relief over 20 km is hopeless without the maximum push."""
    assert auto_exaggeration(80.0, 20_000.0) == pytest.approx(AUTO_EXAGGERATION_MAX)


def test_auto_exaggeration_is_monotonic_between_the_clamps() -> None:
    """More relief always means less exaggeration."""
    width = 30_000.0
    reliefs = np.linspace(1200.0, 5400.0, 25)
    factors = [auto_exaggeration(float(r), width) for r in reliefs]

    assert all(b <= a for a, b in zip(factors, factors[1:])), "must not increase with relief"
    assert factors[0] > factors[-1], "and must actually vary in between"
    assert all(AUTO_EXAGGERATION_MIN <= f <= AUTO_EXAGGERATION_MAX for f in factors)


def test_auto_exaggeration_targets_the_relief_ratio() -> None:
    """Where unclamped, the printed relief really does land on the target."""
    relief_m, width_m = 1500.0, 25_000.0
    factor = auto_exaggeration(relief_m, width_m)
    assert AUTO_EXAGGERATION_MIN < factor < AUTO_EXAGGERATION_MAX
    assert (relief_m * factor) / width_m == pytest.approx(TARGET_RELIEF_RATIO)


def test_auto_exaggeration_edge_cases() -> None:
    assert auto_exaggeration(0.0, 10_000.0) == AUTO_EXAGGERATION_MAX
    assert auto_exaggeration(-5.0, 10_000.0) == AUTO_EXAGGERATION_MAX
    with pytest.raises(ValueError):
        auto_exaggeration(100.0, 0.0)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_writes_a_loadable_binary_stl(tmp_path: Path) -> None:
    mesh = heightmap_to_mesh(make_heightmap(single_peak()))
    path = export(mesh, tmp_path / "out.stl")

    assert path.is_file() and path.stat().st_size > 0
    # Binary STL: 80-byte header, then a uint32 triangle count.
    header = path.read_bytes()[:84]
    assert not header.lstrip().startswith(b"solid "), "should be binary, not ASCII"
    assert int(np.frombuffer(header[80:84], dtype="<u4")[0]) == len(mesh.faces)

    reloaded = trimesh.load(path)
    assert reloaded.is_watertight
    assert len(reloaded.faces) == len(mesh.faces)


def test_export_writes_3mf(tmp_path: Path) -> None:
    mesh = heightmap_to_mesh(make_heightmap(single_peak()))
    path = export(mesh, tmp_path / "out.3mf")

    assert path.is_file() and path.stat().st_size > 0
    reloaded = trimesh.load(path)
    geometry = (
        list(reloaded.geometry.values())[0]
        if isinstance(reloaded, trimesh.Scene)
        else reloaded
    )
    assert len(geometry.faces) == len(mesh.faces)


def test_export_creates_missing_directories(tmp_path: Path) -> None:
    mesh = heightmap_to_mesh(make_heightmap(flat_plane()))
    path = export(mesh, tmp_path / "deep" / "nested" / "out.stl")
    assert path.is_file()


def test_export_rejects_unknown_extensions(tmp_path: Path) -> None:
    mesh = heightmap_to_mesh(make_heightmap(flat_plane()))
    with pytest.raises(ValueError, match="unsupported output format"):
        export(mesh, tmp_path / "out.obj")
