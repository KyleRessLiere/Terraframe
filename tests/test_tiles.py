"""Tests for slippy-map tile math and Terrarium tile fetching."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

from terrframe import tiles
from terrframe.tiles import (
    MAX_ZOOM,
    MIN_ZOOM,
    TILE_SIZE,
    TileFetchError,
    fetch_tile,
    latlon_to_tile,
    tile_to_latlon,
    tiles_for_bbox,
    zoom_for_bbox,
)

SEATTLE = (47.6, -122.3)
RAINIER = (46.8523, -121.7603)


# --------------------------------------------------------------------------
# tile <-> lat/lon
# --------------------------------------------------------------------------


def test_latlon_tile_round_trip_within_one_tile() -> None:
    """Decoding a tile index back to lat/lon lands within that tile's span."""
    zoom = 12
    lat, lon = SEATTLE

    x, y = latlon_to_tile(lat, lon, zoom)
    nw_lat, nw_lon = tile_to_latlon(x, y, zoom)
    se_lat, se_lon = tile_to_latlon(x + 1, y + 1, zoom)

    # tile_to_latlon gives the NW corner, so the input sits inside the one-tile
    # box between that corner and the next tile's corner.
    assert nw_lon <= lon < se_lon
    assert se_lat < lat <= nw_lat
    assert abs(nw_lon - lon) < se_lon - nw_lon
    assert abs(nw_lat - lat) < nw_lat - se_lat


def test_latlon_to_tile_known_values() -> None:
    """Anchor the projection against hand-checkable reference points."""
    # Zoom 0 is a single tile covering the world.
    assert latlon_to_tile(0.0, 0.0, 0) == (0, 0)
    # At zoom 1 the world is 2x2; (0,0) is the NE... actually the origin corner
    # of the SE quadrant of the northern half, i.e. tile (1, 1).
    assert latlon_to_tile(0.0, 0.0, 1) == (1, 1)
    assert latlon_to_tile(45.0, -90.0, 2) == (1, 1)


def test_tile_to_latlon_corners() -> None:
    """Tile (0, 0) starts at the NW corner of the Mercator world."""
    lat, lon = tile_to_latlon(0, 0, 0)
    assert lon == pytest.approx(-180.0)
    assert lat == pytest.approx(85.0511287798, abs=1e-6)


def test_latlon_to_tile_clamps_and_wraps() -> None:
    """Out-of-range inputs stay on the grid instead of falling off it."""
    n = 1 << 4
    for lat, lon in [(90.0, 0.0), (-90.0, 0.0), (0.0, 180.0), (0.0, -180.0)]:
        x, y = latlon_to_tile(lat, lon, 4)
        assert 0 <= x < n
        assert 0 <= y < n


# --------------------------------------------------------------------------
# tiles_for_bbox
# --------------------------------------------------------------------------


def test_tiles_for_bbox_single_tile_gives_3x3_with_margin() -> None:
    """A bbox inside one tile still returns that tile plus a one-tile ring."""
    zoom = 12
    lat, lon = SEATTLE
    x, y = latlon_to_tile(lat, lon, zoom)

    # A tiny box comfortably inside tile (x, y).
    nw_lat, nw_lon = tile_to_latlon(x, y, zoom)
    se_lat, se_lon = tile_to_latlon(x + 1, y + 1, zoom)
    mid_lat = (nw_lat + se_lat) / 2.0
    mid_lon = (nw_lon + se_lon) / 2.0
    eps_lat = (nw_lat - se_lat) / 100.0
    eps_lon = (se_lon - nw_lon) / 100.0

    result = tiles_for_bbox(
        south=mid_lat - eps_lat,
        west=mid_lon - eps_lon,
        north=mid_lat + eps_lat,
        east=mid_lon + eps_lon,
        zoom=zoom,
    )

    assert len(result) == 9
    expected = {(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    assert set(result) == expected


def test_tiles_for_bbox_covers_every_tile_in_range() -> None:
    """A bbox spanning several tiles yields a full rectangle plus the margin."""
    zoom = 10
    result = tiles_for_bbox(south=46.6, west=-122.1, north=47.1, east=-121.4, zoom=zoom)

    x_min, y_min = latlon_to_tile(47.1, -122.1, zoom)
    x_max, y_max = latlon_to_tile(46.6, -121.4, zoom)

    width = (x_max - x_min + 1) + 2
    height = (y_max - y_min + 1) + 2
    assert len(result) == width * height
    assert len(set(result)) == len(result)
    assert (x_min, y_min) in result
    assert (x_min - 1, y_min - 1) in result
    assert (x_max + 1, y_max + 1) in result


def test_tiles_for_bbox_clamps_at_world_edge() -> None:
    """The margin never produces negative or out-of-range tile indices."""
    zoom = 2
    n = 1 << zoom
    result = tiles_for_bbox(south=-85.0, west=-180.0, north=85.0, east=180.0, zoom=zoom)
    assert result
    for x, y in result:
        assert 0 <= x < n
        assert 0 <= y < n


def test_tiles_for_bbox_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        tiles_for_bbox(south=48.0, west=-122.0, north=47.0, east=-121.0, zoom=10)
    with pytest.raises(ValueError):
        tiles_for_bbox(south=47.0, west=-121.0, north=48.0, east=-122.0, zoom=10)


# --------------------------------------------------------------------------
# zoom_for_bbox
# --------------------------------------------------------------------------


def test_zoom_for_bbox_hits_target_pixel_span() -> None:
    """The chosen zoom puts the bbox's long side near the pixel target."""
    south, west, north, east = 46.6, -122.1, 47.1, -121.4
    target = 800

    zoom = zoom_for_bbox(south, west, north, east, target_px=target)

    lon_frac = (east - west) / 360.0
    lat_frac = abs(tiles._mercator_y(north) - tiles._mercator_y(south))
    span_px = max(lon_frac, lat_frac) * TILE_SIZE * (1 << zoom)

    # Rounding to an integer zoom can only put us within a factor of sqrt(2).
    assert target / 1.5 <= span_px <= target * 1.5


def test_zoom_for_bbox_smaller_area_means_higher_zoom() -> None:
    big = zoom_for_bbox(46.0, -123.0, 48.0, -121.0)
    small = zoom_for_bbox(46.90, -121.80, 46.95, -121.75)
    assert small > big


def test_zoom_for_bbox_doubling_target_adds_one_zoom() -> None:
    a = zoom_for_bbox(46.6, -122.1, 47.1, -121.4, target_px=400)
    b = zoom_for_bbox(46.6, -122.1, 47.1, -121.4, target_px=800)
    assert b == a + 1


def test_zoom_for_bbox_clamped_to_supported_range() -> None:
    # 800px across the whole world genuinely rounds to zoom 2; only a coarser
    # target pushes past the floor.
    assert zoom_for_bbox(-85.0, -180.0, 85.0, 180.0, target_px=800) == 2
    assert zoom_for_bbox(-85.0, -180.0, 85.0, 180.0, target_px=100) == MIN_ZOOM

    pinpoint = zoom_for_bbox(46.85230, -121.76030, 46.85231, -121.76029)
    assert pinpoint == MAX_ZOOM


def test_zoom_for_bbox_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        zoom_for_bbox(48.0, -122.0, 47.0, -121.0)
    with pytest.raises(ValueError):
        zoom_for_bbox(47.0, -122.0, 48.0, -121.0, target_px=0)


# --------------------------------------------------------------------------
# Terrarium decoding
# --------------------------------------------------------------------------

# (R, G, B) -> exact metres, per elevation_m = (R * 256 + G + B / 256) - 32768.
SYNTHETIC_PIXELS: list[tuple[tuple[int, int, int], float]] = [
    ((0, 0, 0), -32768.0),  # floor of the encoding
    ((128, 0, 0), 0.0),  # sea level
    ((128, 10, 128), 10.5),  # blue channel is the 1/256 m fraction
    ((145, 44, 0), 4396.0),  # roughly Mt. Rainier
]


def _write_synthetic_tile(path: Path) -> None:
    """Write a 2x2 Terrarium PNG with the known pixels above."""
    data = np.array(
        [
            [SYNTHETIC_PIXELS[0][0], SYNTHETIC_PIXELS[1][0]],
            [SYNTHETIC_PIXELS[2][0], SYNTHETIC_PIXELS[3][0]],
        ],
        dtype=np.uint8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(data, mode="RGB").save(path, format="PNG")


def test_terrarium_decoding_is_exact(tmp_path: Path) -> None:
    """A synthetic PNG with known RGB decodes to exactly the expected metres."""
    cache_dir = tmp_path / "cache"
    _write_synthetic_tile(cache_dir / "0" / "0" / "0.png")

    # Cached, so this must not touch the network.
    elevations = fetch_tile(0, 0, 0, cache_dir=cache_dir)

    assert elevations.dtype == np.float32
    assert elevations.shape == (2, 2)

    expected = np.array(
        [
            [SYNTHETIC_PIXELS[0][1], SYNTHETIC_PIXELS[1][1]],
            [SYNTHETIC_PIXELS[2][1], SYNTHETIC_PIXELS[3][1]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(elevations, expected)


def test_cached_tile_is_never_redownloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A tile already on disk is decoded without any HTTP request."""
    cache_dir = tmp_path / "cache"
    _write_synthetic_tile(cache_dir / "0" / "0" / "0.png")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("fetch_tile hit the network for a cached tile")

    monkeypatch.setattr(tiles.requests, "get", _boom)

    first = fetch_tile(0, 0, 0, cache_dir=cache_dir)
    second = fetch_tile(0, 0, 0, cache_dir=cache_dir)
    np.testing.assert_array_equal(first, second)


def test_corrupt_cache_is_deleted_and_refetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable cache entry is thrown away and fetched again."""
    cache_dir = tmp_path / "cache"
    cached = cache_dir / "0" / "0" / "0.png"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"this is not a png")

    calls: list[tuple[int, int, int]] = []

    def _fake_download(x: int, y: int, zoom: int, path: Path) -> None:
        calls.append((x, y, zoom))
        _write_synthetic_tile(path)

    monkeypatch.setattr(tiles, "_download_tile", _fake_download)

    elevations = fetch_tile(0, 0, 0, cache_dir=cache_dir)

    assert calls == [(0, 0, 0)], "corrupt cache entry should trigger exactly one re-fetch"
    assert elevations.shape == (2, 2)
    assert elevations[0, 1] == pytest.approx(0.0)


def test_network_failure_raises_tile_fetch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport error surfaces as TileFetchError, not a raw requests error."""

    def _fail(*args: object, **kwargs: object) -> None:
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(tiles.requests, "get", _fail)

    with pytest.raises(TileFetchError, match="failed to download tile"):
        fetch_tile(0, 0, 0, cache_dir=tmp_path / "cache")


def test_request_sets_user_agent_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tile request is polite: identified and time-limited."""
    seen: dict[str, object] = {}

    buf = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class _Response:
        content = png_bytes

        def raise_for_status(self) -> None:
            return None

    def _capture(url: str, **kwargs: object) -> _Response:
        seen["url"] = url
        seen.update(kwargs)
        return _Response()

    monkeypatch.setattr(tiles.requests, "get", _capture)

    fetch_tile(3, 5, 4, cache_dir=tmp_path / "cache")

    assert seen["url"] == "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/4/3/5.png"
    assert seen["timeout"] is not None and seen["timeout"] > 0
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert "terrframe" in headers["User-Agent"]


def test_fetch_tile_rejects_out_of_range_tiles(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fetch_tile(4, 0, 2, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError):
        fetch_tile(-1, 0, 2, cache_dir=tmp_path / "cache")


# --------------------------------------------------------------------------
# Live network
# --------------------------------------------------------------------------


@pytest.mark.network
def test_fetch_real_tile_containing_mount_rainier(tmp_path: Path) -> None:
    """The real zoom-10 tile over Mt. Rainier tops out near its true 4392 m."""
    zoom = 10
    x, y = latlon_to_tile(*RAINIER, zoom)

    elevations = fetch_tile(x, y, zoom, cache_dir=tmp_path / "cache")

    assert elevations.shape == (TILE_SIZE, TILE_SIZE)
    assert elevations.dtype == np.float32
    assert 4300.0 < float(elevations.max()) < 4450.0

    # And the cache is populated for a second, offline call.
    cached = tmp_path / "cache" / str(zoom) / str(x) / f"{y}.png"
    assert cached.is_file()
    np.testing.assert_array_equal(elevations, fetch_tile(x, y, zoom, cache_dir=tmp_path / "cache"))
