"""Tests for heightmap assembly. Everything here runs offline."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pyproj import Geod

from terrframe import heightmap as hm
from terrframe.heightmap import (
    Heightmap,
    bounds_to_bbox,
    build_heightmap,
    crop_bounds,
    crop_to_bbox,
    exaggerate,
    fill_nodata,
    flatten_water,
    ground_extent,
    meters_per_pixel,
    resample_to_meters,
    stitch,
    tile_origin,
)
from terrframe.tiles import TILE_SIZE, latlon_to_tile, mercator_y_norm, zoom_for_bbox

GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# stitch
# ---------------------------------------------------------------------------


def test_stitch_2x2_grid_places_each_quadrant() -> None:
    """A 2x2 grid of constant tiles lands in the right quadrants."""
    size = 4
    tiles = {
        (10, 20): np.full((size, size), 1.0, dtype=np.float32),  # NW
        (11, 20): np.full((size, size), 2.0, dtype=np.float32),  # NE
        (10, 21): np.full((size, size), 3.0, dtype=np.float32),  # SW
        (11, 21): np.full((size, size), 4.0, dtype=np.float32),  # SE
    }

    out = stitch(tiles, zoom=8)

    assert out.shape == (2 * size, 2 * size)
    assert np.all(out[:size, :size] == 1.0), "north-west quadrant"
    assert np.all(out[:size, size:] == 2.0), "north-east quadrant"
    assert np.all(out[size:, :size] == 3.0), "south-west quadrant"
    assert np.all(out[size:, size:] == 4.0), "south-east quadrant"


def test_stitch_is_row_major_over_tiles_for_bbox_order() -> None:
    """The order tiles_for_bbox returns feeds stitch without rearranging."""
    tiles = {
        (x, y): np.full((2, 2), float(x * 10 + y), dtype=np.float32)
        for y in (5, 6, 7)
        for x in (3, 4)
    }
    out = stitch(tiles, zoom=6)
    assert out.shape == (6, 4)
    # Top-left cell comes from tile (3, 5).
    assert out[0, 0] == pytest.approx(35.0)
    # Bottom-right cell comes from tile (4, 7).
    assert out[-1, -1] == pytest.approx(47.0)


def test_tile_origin_is_north_west_corner() -> None:
    tiles = {(11, 21): np.zeros((2, 2)), (10, 20): np.zeros((2, 2))}
    tiles[(10, 21)] = np.zeros((2, 2))
    tiles[(11, 20)] = np.zeros((2, 2))
    assert tile_origin(tiles) == (10, 20)


def test_stitch_rejects_incomplete_or_ragged_grids() -> None:
    with pytest.raises(ValueError, match="empty"):
        stitch({}, zoom=8)

    with pytest.raises(ValueError, match="complete grid"):
        stitch(
            {
                (0, 0): np.zeros((2, 2)),
                (1, 0): np.zeros((2, 2)),
                (0, 1): np.zeros((2, 2)),
            },
            zoom=8,
        )

    with pytest.raises(ValueError, match="one shape"):
        stitch({(0, 0): np.zeros((2, 2)), (1, 0): np.zeros((3, 3))}, zoom=8)

    with pytest.raises(ValueError, match="contiguous"):
        stitch({(0, 0): np.zeros((2, 2)), (5, 0): np.zeros((2, 2))}, zoom=8)

    with pytest.raises(ValueError, match="outside"):
        stitch({(99, 0): np.zeros((2, 2))}, zoom=1)


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


def test_crop_rounds_outward_on_sub_pixel_edges() -> None:
    """Fractional bbox edges expand the window; requested area is never lost."""
    zoom = 10
    origin = (0, 0)
    world_px = TILE_SIZE * (1 << zoom)

    # Pick a bbox whose edges land at x.5 pixels by construction.
    west = (100.5 / world_px) * 360.0 - 180.0
    east = (200.5 / world_px) * 360.0 - 180.0
    north = _lat_for_row(50.5, world_px)
    south = _lat_for_row(150.5, world_px)

    col_start, row_start, col_stop, row_stop = crop_bounds(
        origin, zoom, south, west, north, east
    )

    assert (col_start, col_stop) == (100, 201), "columns must round out both ways"
    assert (row_start, row_stop) == (50, 151), "rows must round out both ways"


def test_crop_never_shrinks_below_the_request() -> None:
    """The window always covers at least the requested bbox, at many offsets."""
    zoom = 11
    origin = (0, 0)
    world_px = TILE_SIZE * (1 << zoom)

    for offset in (0.0, 0.1, 0.49, 0.5, 0.51, 0.99):
        west = ((300 + offset) / world_px) * 360.0 - 180.0
        east = ((360 + offset) / world_px) * 360.0 - 180.0
        north = _lat_for_row(200 + offset, world_px)
        south = _lat_for_row(260 + offset, world_px)

        c0, r0, c1, r1 = crop_bounds(origin, zoom, south, west, north, east)

        assert c0 <= 300 + offset and c1 >= 360 + offset
        assert r0 <= 200 + offset and r1 >= 260 + offset


def test_crop_extracts_the_expected_gradient_window() -> None:
    """Cropping a known gradient returns exactly the right sub-rectangle."""
    zoom = 8
    origin = (0, 0)
    world_px = TILE_SIZE * (1 << zoom)

    rows = cols = 256
    # Value encodes its own position so the window is self-identifying.
    grid = (np.arange(rows)[:, None] * 1000 + np.arange(cols)[None, :]).astype(np.float32)

    west = (10.0 / world_px) * 360.0 - 180.0
    east = (20.0 / world_px) * 360.0 - 180.0
    north = _lat_for_row(30.0, world_px)
    south = _lat_for_row(40.0, world_px)

    out = crop_to_bbox(grid, origin, zoom, south, west, north, east)

    assert out.shape == (10, 10)
    assert out[0, 0] == pytest.approx(30 * 1000 + 10)
    assert out[-1, -1] == pytest.approx(39 * 1000 + 19)


def test_crop_of_degenerate_bbox_still_yields_a_pixel() -> None:
    zoom = 10
    grid = np.zeros((512, 512), dtype=np.float32)
    lat, lon = 46.85, -121.76
    out = crop_to_bbox(grid, latlon_to_tile(lat, lon, zoom), zoom, lat, lon, lat, lon)
    assert out.size >= 1


def test_bounds_to_bbox_inverts_crop_bounds() -> None:
    """The reported covered bbox really does contain what was asked for."""
    zoom = 12
    south, west, north, east = 46.75, -121.95, 46.95, -121.55
    origin = latlon_to_tile(north, west, zoom)

    bounds = crop_bounds(origin, zoom, south, west, north, east)
    got_s, got_w, got_n, got_e = bounds_to_bbox(bounds, origin, zoom)

    assert got_s <= south and got_n >= north
    assert got_w <= west and got_e >= east
    # Outward rounding costs at most a pixel per edge.
    per_px_deg = 360.0 / (TILE_SIZE * (1 << zoom))
    assert got_w == pytest.approx(west, abs=per_px_deg)
    assert got_e == pytest.approx(east, abs=per_px_deg)


def _lat_for_row(row: float, world_px: float) -> float:
    """Latitude whose Mercator pixel row is ``row``."""
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * row / world_px))))


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


def test_resample_square_degrees_at_lat60_is_twice_as_tall_as_wide() -> None:
    """A 1x1 degree bbox at lat 60 is ~2x taller than wide on the ground.

    Note this is the opposite of a plate-carree intuition: Web Mercator is
    conformal, so the source crop is *already* isotropic. cos(60) = 0.5 makes
    the east-west ground span half the north-south one, and a metres-correct
    grid has to show that.
    """
    south, west, north, east = 59.5, 0.0, 60.5, 1.0

    # Source shaped like a real Mercator crop of this bbox.
    src_cols = 256
    y_span = abs(mercator_y_norm(north) - mercator_y_norm(south))
    src_rows = int(round(src_cols * y_span / ((east - west) / 360.0)))
    src = np.random.default_rng(0).normal(500.0, 50.0, (src_rows, src_cols)).astype(np.float32)

    out = resample_to_meters(src, south, west, north, east)

    aspect = out.shape[0] / out.shape[1]
    assert aspect == pytest.approx(2.0, rel=0.02), "height/width should track 1/cos(60)"

    # And it matches real geodesic distances, not just the sphere model.
    ew = GEOD.inv(west, (south + north) / 2, east, (south + north) / 2)[2]
    ns = GEOD.inv((west + east) / 2, south, (west + east) / 2, north)[2]
    assert aspect == pytest.approx(ns / ew, rel=0.02)


def test_resample_produces_square_ground_pixels() -> None:
    """Metres per pixel comes out equal on both axes, which is the whole point."""
    south, west, north, east = 59.5, 0.0, 60.5, 1.0
    src = np.zeros((512, 256), dtype=np.float32)

    out = resample_to_meters(src, south, west, north, east)
    width_m, height_m = ground_extent(south, west, north, east)

    mpp_x = width_m / out.shape[1]
    mpp_y = height_m / out.shape[0]
    assert mpp_x == pytest.approx(mpp_y, rel=0.01)


def test_resample_at_equator_is_near_identity() -> None:
    """With cos(lat) = 1 there is nothing to correct, so the shape holds."""
    src = np.zeros((200, 200), dtype=np.float32)
    out = resample_to_meters(src, -0.5, -0.5, 0.5, 0.5)
    assert out.shape[0] == pytest.approx(out.shape[1], abs=2)
    assert abs(out.shape[0] - 200) <= 2


def test_resample_preserves_values_and_never_downsamples() -> None:
    """Resampling a constant field is exact, and detail is not thrown away."""
    src = np.full((300, 200), 123.5, dtype=np.float32)
    out = resample_to_meters(src, 44.5, -70.5, 45.5, -69.5)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 123.5, rtol=1e-5)
    assert out.shape[0] >= src.shape[0] or out.shape[1] >= src.shape[1]


def test_resample_corrects_latitude_scale_drift() -> None:
    """Output rows are equally spaced in latitude, unlike the Mercator source.

    A linear-in-latitude ramp fed through a Mercator-spaced source must come
    back linear; if rows were copied straight across it would bow.
    """
    south, north = 60.0, 70.0
    west, east = 0.0, 1.0

    src_rows = 400
    y_top, y_bottom = mercator_y_norm(north), mercator_y_norm(south)
    y = y_top + (np.arange(src_rows) + 0.5) / src_rows * (y_bottom - y_top)
    lats = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y))))
    src = np.repeat(lats[:, None], 64, axis=1).astype(np.float32)

    out = resample_to_meters(src, south, west, north, east)

    col = out[:, out.shape[1] // 2]
    expected = north - (np.arange(out.shape[0]) + 0.5) / out.shape[0] * (north - south)
    np.testing.assert_allclose(col, expected, atol=0.02)


def test_resample_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        resample_to_meters(np.zeros((4, 4, 4)), 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        resample_to_meters(np.zeros((0, 0)), 0.0, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# fill_nodata
# ---------------------------------------------------------------------------


def test_fill_nodata_removes_nan_blob_within_neighbour_range() -> None:
    """A NaN blob is filled from its surroundings and invents nothing."""
    rng = np.random.default_rng(7)
    arr = rng.uniform(100.0, 900.0, (64, 64)).astype(np.float32)
    valid_before = arr.copy()

    arr[20:30, 25:35] = np.nan
    holes = ~np.isfinite(arr)

    out = fill_nodata(arr)

    assert np.isfinite(out).all(), "no NaNs may survive"
    assert out.dtype == np.float32
    assert out.shape == arr.shape

    neighbours = valid_before[~holes]
    lo, hi = float(neighbours.min()), float(neighbours.max())
    filled = out[holes]
    assert filled.min() >= lo - 1e-3
    assert filled.max() <= hi + 1e-3


def test_fill_nodata_treats_encoding_floor_as_a_hole() -> None:
    """Terrarium's -32768 floor means 'missing', not '32 km below sea level'."""
    arr = np.full((16, 16), 250.0, dtype=np.float32)
    arr[4:8, 4:8] = -32768.0

    out = fill_nodata(arr)

    assert out.min() > 0.0
    np.testing.assert_allclose(out, 250.0, atol=1e-3)


def test_fill_nodata_leaves_clean_arrays_alone() -> None:
    """With no holes the data must come back untouched, not merely close."""
    rng = np.random.default_rng(3)
    arr = rng.uniform(0.0, 1000.0, (32, 32)).astype(np.float32)
    out = fill_nodata(arr)
    np.testing.assert_array_equal(out, arr)
    assert out is not arr


def test_fill_nodata_blends_the_hole_border() -> None:
    """The seam is softened: a filled hole is not a hard-edged nearest copy."""
    arr = np.zeros((48, 48), dtype=np.float32)
    arr[:, 24:] = 100.0
    arr[20:28, 20:28] = np.nan

    hard = fill_nodata(arr, sigma=0.0)
    soft = fill_nodata(arr, sigma=2.0)

    assert np.isfinite(soft).all()
    # Nearest-fill alone reproduces the hard step; blending must not.
    assert not np.allclose(hard, soft)
    assert soft.min() >= -1e-3 and soft.max() <= 100.0 + 1e-3


def test_fill_nodata_rejects_an_all_hole_array() -> None:
    with pytest.raises(ValueError, match="nothing to fill from"):
        fill_nodata(np.full((8, 8), np.nan, dtype=np.float32))


# ---------------------------------------------------------------------------
# flatten_water
# ---------------------------------------------------------------------------


def test_flatten_water_auto_clamps_negative_elevations() -> None:
    """A mixed-sign array comes back with nothing below sea level."""
    arr = np.array([[-500.0, -1.0], [0.0, 250.0]], dtype=np.float32)
    out = flatten_water(arr)

    assert out.min() >= 0.0
    assert not (out < 0.0).any()
    # Land is untouched.
    assert out[1, 1] == pytest.approx(250.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_flatten_water_auto_leaves_all_positive_arrays_unchanged() -> None:
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = flatten_water(arr)
    np.testing.assert_array_equal(out, arr)


def test_flatten_water_explicit_level_clamps_above_sea_level() -> None:
    """An explicit level flattens lakes that sit well above zero."""
    arr = np.array([[1200.0, 1250.0], [1300.0, 1400.0]], dtype=np.float32)
    out = flatten_water(arr, level=1275.0)

    assert out.min() == pytest.approx(1275.0)
    assert out[1, 1] == pytest.approx(1400.0)


def test_flatten_water_explicit_zero_is_not_treated_as_auto() -> None:
    """level=0.0 must clamp, even though 0.0 is falsy."""
    arr = np.array([[-5.0, 5.0]], dtype=np.float32)
    out = flatten_water(arr, level=0.0)
    assert out.min() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# exaggerate
# ---------------------------------------------------------------------------


def test_exaggerate_doubles_relief_about_the_minimum() -> None:
    arr = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    out = exaggerate(arr, 2.0)

    assert out.min() == pytest.approx(arr.min()), "the base must not move"
    relief_before = float(arr.max() - arr.min())
    relief_after = float(out.max() - out.min())
    assert relief_after == pytest.approx(2.0 * relief_before)
    np.testing.assert_allclose(out, [[100.0, 300.0], [500.0, 700.0]])


def test_exaggerate_factor_one_is_identity() -> None:
    rng = np.random.default_rng(11)
    arr = rng.uniform(-100.0, 4000.0, (32, 32)).astype(np.float32)
    np.testing.assert_allclose(exaggerate(arr, 1.0), arr, rtol=1e-6)


def test_exaggerate_rejects_non_positive_factors() -> None:
    arr = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        exaggerate(arr, 0.0)
    with pytest.raises(ValueError):
        exaggerate(arr, -2.0)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def test_meters_per_pixel_matches_known_web_mercator_values() -> None:
    """The textbook figure is ~156.5 km/px at zoom 0 on the equator."""
    assert meters_per_pixel(0, 0.0) == pytest.approx(156543.03, rel=1e-4)
    # Each zoom halves it, and cos(lat) shrinks it toward the poles.
    assert meters_per_pixel(10, 0.0) == pytest.approx(156543.03 / 1024, rel=1e-4)
    assert meters_per_pixel(10, 60.0) == pytest.approx(meters_per_pixel(10, 0.0) * 0.5, rel=1e-3)


def test_ground_extent_matches_geodesic_distance() -> None:
    """The sphere model stays within a fraction of a percent of WGS84."""
    south, west, north, east = 46.75, -121.95, 46.95, -121.55
    width_m, height_m = ground_extent(south, west, north, east)

    mid_lat = (south + north) / 2
    mid_lon = (west + east) / 2
    assert width_m == pytest.approx(GEOD.inv(west, mid_lat, east, mid_lat)[2], rel=0.005)
    assert height_m == pytest.approx(GEOD.inv(mid_lon, south, mid_lon, north)[2], rel=0.005)


# ---------------------------------------------------------------------------
# build_heightmap (end to end, monkeypatched fetch)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_tiles(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int, int]]:
    """Serve synthetic tiles so the orchestrator never touches the network."""
    fetched: list[tuple[int, int, int]] = []

    def _fake_fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        fetched.append((x, y, zoom))
        # A smooth ramp that differs per tile, so stitching errors show up.
        rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
        return (500.0 + 200.0 * (rows + cols) + 10.0 * ((x % 5) + (y % 5))).astype(np.float32)

    monkeypatch.setattr(hm, "fetch_tile", _fake_fetch)
    return fetched


RAINIER_BBOX = (46.75, -121.95, 46.95, -121.55)


def test_build_heightmap_end_to_end(fake_tiles: list[tuple[int, int, int]]) -> None:
    """The full pipeline yields a clean grid with truthful scale."""
    south, west, north, east = RAINIER_BBOX
    result = build_heightmap(south, west, north, east, target_px=800)

    assert isinstance(result, Heightmap)
    assert result.elevation.dtype == np.float32
    assert result.elevation.ndim == 2
    assert np.isfinite(result.elevation).all(), "no NaNs may reach the mesh stage"
    assert fake_tiles, "the orchestrator must actually fetch tiles"

    # Zoom matches what the tile layer would pick on its own.
    assert result.zoom == zoom_for_bbox(south, west, north, east, target_px=800)

    # meters_per_px within 10% of the Web Mercator resolution at this latitude.
    expected_mpp = meters_per_pixel(result.zoom, (south + north) / 2)
    assert result.meters_per_px == pytest.approx(expected_mpp, rel=0.10)

    # The longer side lands near the requested pixel target.
    assert max(result.elevation.shape) == pytest.approx(800, rel=0.35)


def test_build_heightmap_shape_matches_ground_aspect(
    fake_tiles: list[tuple[int, int, int]],
) -> None:
    """Pixel aspect tracks the true geodesic aspect, so prints aren't stretched."""
    south, west, north, east = RAINIER_BBOX
    result = build_heightmap(south, west, north, east)

    rows, cols = result.elevation.shape
    mid_lat, mid_lon = (south + north) / 2, (west + east) / 2
    ew = GEOD.inv(west, mid_lat, east, mid_lat)[2]
    ns = GEOD.inv(mid_lon, south, mid_lon, north)[2]

    assert rows / cols == pytest.approx(ns / ew, rel=0.05)


def test_build_heightmap_covers_at_least_the_request(
    fake_tiles: list[tuple[int, int, int]],
) -> None:
    """Outward cropping means the covered bbox is a superset of the request."""
    south, west, north, east = RAINIER_BBOX
    result = build_heightmap(south, west, north, east)

    got_s, got_w, got_n, got_e = result.bbox
    assert got_s <= south and got_n >= north
    assert got_w <= west and got_e >= east
    assert result.requested_bbox == (south, west, north, east)


def test_build_heightmap_exaggeration_scales_relief(
    fake_tiles: list[tuple[int, int, int]],
) -> None:
    """Exaggeration multiplies relief without moving the base or the scale."""
    south, west, north, east = RAINIER_BBOX
    plain = build_heightmap(south, west, north, east, exaggeration=1.0)
    tall = build_heightmap(south, west, north, east, exaggeration=3.0)

    plain_relief = float(plain.elevation.max() - plain.elevation.min())
    tall_relief = float(tall.elevation.max() - tall.elevation.min())

    assert tall_relief == pytest.approx(3.0 * plain_relief, rel=1e-4)
    assert tall.elevation.min() == pytest.approx(plain.elevation.min(), rel=1e-4)
    assert tall.meters_per_px == pytest.approx(plain.meters_per_px)
    assert tall.exaggeration == 3.0


def test_build_heightmap_flatten_water_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """'auto' clamps ocean, None keeps bathymetry, a number sets the level."""

    def _ocean_fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        # Half the tile is below sea level.
        arr = np.linspace(-300.0, 300.0, TILE_SIZE, dtype=np.float32)
        return np.repeat(arr[:, None], TILE_SIZE, axis=1)

    monkeypatch.setattr(hm, "fetch_tile", _ocean_fetch)
    bbox = (10.0, 10.0, 10.2, 10.2)

    auto = build_heightmap(*bbox, flatten_water_level="auto")
    assert auto.elevation.min() >= 0.0

    raw = build_heightmap(*bbox, flatten_water_level=None)
    assert raw.elevation.min() < 0.0

    explicit = build_heightmap(*bbox, flatten_water_level=-100.0)
    assert explicit.elevation.min() == pytest.approx(-100.0, abs=1e-3)


def test_build_heightmap_size_meters_is_consistent(
    fake_tiles: list[tuple[int, int, int]],
) -> None:
    """The reported footprint agrees with the covered bbox's real extent."""
    result = build_heightmap(*RAINIER_BBOX)
    width_m, height_m = result.size_meters
    expected_w, expected_h = ground_extent(*result.bbox)

    assert width_m == pytest.approx(expected_w, rel=0.01)
    assert height_m == pytest.approx(expected_h, rel=0.01)
    assert result.shape == result.elevation.shape


def test_build_heightmap_rejects_inverted_bbox() -> None:
    with pytest.raises(ValueError):
        build_heightmap(48.0, -122.0, 47.0, -121.0)
