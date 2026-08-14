"""Tests for OSM feature stamping. Offline: synthetic geometry plus a fixture."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import requests
from shapely.geometry import LineString, Polygon, box

from terrframe import features as F
from terrframe.heightmap import Heightmap, build_heightmap
from terrframe.tiles import TILE_SIZE

FIXTURE = Path(__file__).parent / "fixtures" / "overpass_tidal_basin_water.json"

# A 1-degree grid with 1 px == 0.01 degrees, so geometry maps to pixels exactly.
BBOX = (38.0, -77.0, 39.0, -76.0)


def make_hm(rows: int = 100, cols: int = 100, base: float = 100.0) -> Heightmap:
    return Heightmap(
        elevation=np.full((rows, cols), base, dtype=np.float32),
        meters_per_px=10.0,
        bbox=BBOX,
        zoom=12,
    )


# ---------------------------------------------------------------------------
# Transform and rasterisation
# ---------------------------------------------------------------------------


def test_heightmap_transform_maps_corners() -> None:
    hm = make_hm()
    t = hm.transform
    assert t * (0, 0) == pytest.approx((-77.0, 39.0))          # NW corner
    assert t * (100, 100) == pytest.approx((-76.0, 38.0))      # SE corner
    assert t.a == pytest.approx(0.01) and t.e == pytest.approx(-0.01)


def test_rasterize_mask_is_pixel_exact_on_a_known_square() -> None:
    """A square from lon -76.8..-76.6, lat 38.6..38.8 lands on an exact block."""
    hm = make_hm()
    square = box(-76.8, 38.6, -76.6, 38.8)

    mask = F.rasterize_mask([square], hm)

    # x: (-76.8 - -77.0)/0.01 = 20 .. 40 ; y: (39.0-38.8)/0.01 = 20 .. 40
    expected = np.zeros((100, 100), dtype=bool)
    expected[20:40, 20:40] = True
    assert mask[21:39, 21:39].all(), "interior must be filled"
    assert not mask[:19, :].any(), "nothing above the square"
    assert not mask[41:, :].any(), "nothing below the square"
    # all_touched adds at most the boundary pixel ring.
    assert abs(int(mask.sum()) - int(expected.sum())) <= 4 * 40


def test_rasterize_mask_empty_geometry_is_all_false() -> None:
    assert not F.rasterize_mask([], make_hm()).any()


def test_rasterize_mask_buffers_lines_by_width() -> None:
    hm = make_hm()
    line = LineString([(-76.9, 38.5), (-76.1, 38.5)])
    thin = F.rasterize_mask([line], hm, width_px=1)
    thick = F.rasterize_mask([line], hm, width_px=5)
    assert thick.sum() > thin.sum()


# ---------------------------------------------------------------------------
# Water stamping
# ---------------------------------------------------------------------------


def test_water_interior_is_flat_at_edge_min_minus_depth() -> None:
    """Exact spec: interior == (minimum on the bank ring) - depth, dead flat."""
    hm = make_hm()
    elevation = np.array(hm.elevation)
    # Slope the terrain so the bank minimum is well defined.
    elevation += np.arange(100, dtype=np.float32)[None, :]
    hm = Heightmap(elevation, hm.meters_per_px, hm.bbox, hm.zoom)

    poly = box(-76.8, 38.6, -76.6, 38.8)
    depth_mm = 0.3
    scale = F._mm_per_meter(hm)

    out = F.stamp_water(hm, [poly], depth_mm=depth_mm)
    mask = F.rasterize_mask([poly], hm)

    values = np.unique(out[mask])
    assert values.size == 1, "water must be dead flat, one value"

    expected_level = F._edge_minimum(elevation, mask) - depth_mm / scale
    assert float(values[0]) == pytest.approx(expected_level, abs=1e-3)


def test_water_bank_is_crisp() -> None:
    """The polygon edge is the bank: full drop within a pixel, never blurred."""
    hm = make_hm()
    poly = box(-76.8, 38.6, -76.6, 38.8)

    out = F.stamp_water(hm, [poly])
    mask = F.rasterize_mask([poly], hm)

    from scipy.ndimage import binary_dilation

    ring1 = binary_dilation(mask, iterations=1) & ~mask
    ring2 = binary_dilation(mask, iterations=2) & ~binary_dilation(mask, iterations=1)

    # Terrain one pixel outside the bank is untouched -- no gradient ramp.
    assert out[ring1] == pytest.approx(100.0)
    assert out[ring2] == pytest.approx(100.0)
    assert out[mask].max() < 100.0


def test_water_depth_is_constant_in_printed_mm() -> None:
    """Doubling exaggeration must not change how deep the water looks."""
    hm = make_hm()
    poly = box(-76.8, 38.6, -76.6, 38.8)
    scale = F._mm_per_meter(hm)

    plain = F.stamp_water(hm, [poly], depth_mm=0.3)
    drop_mm = (100.0 - plain[F.rasterize_mask([poly], hm)].max()) * scale
    assert drop_mm == pytest.approx(0.3, abs=1e-6)


def test_adjacent_water_polygons_do_not_cascade() -> None:
    """Each polygon reads bank height from pristine terrain, not from a neighbour.

    Touching polygons used to step down one after another; across a city that
    compounded to hundreds of metres below sea level.
    """
    hm = make_hm()
    left = box(-76.80, 38.60, -76.70, 38.80)
    right = box(-76.70, 38.60, -76.60, 38.80)

    out = F.stamp_water(hm, [left, right])
    levels = np.unique(np.round(out[out < 100.0], 4))
    assert levels.size == 1, f"both polygons should reach the same level, got {levels}"


def test_stamp_water_without_geometry_is_a_no_op() -> None:
    hm = make_hm()
    np.testing.assert_array_equal(F.stamp_water(hm, []), hm.elevation)


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def test_building_removal_fills_within_neighbour_range() -> None:
    hm = make_hm()
    elevation = np.array(hm.elevation)
    elevation += np.arange(100, dtype=np.float32)[None, :]
    poly = box(-76.8, 38.6, -76.6, 38.8)
    mask = F.rasterize_mask([poly], Heightmap(elevation, 10.0, BBOX, 12))
    elevation[mask] += 300.0  # a tower block
    hm = Heightmap(elevation, 10.0, BBOX, 12)

    out = F.remove_buildings(hm, [poly])

    assert np.isfinite(out).all()
    assert out[mask].max() < 300.0, "tower must be gone"
    surrounding = elevation[~mask]
    assert out[mask].min() >= surrounding.min() - 1e-3
    assert out[mask].max() <= surrounding.max() + 1e-3


def test_building_removal_dilates_the_footprint() -> None:
    """A bare footprint leaves a rim of roof height from resampling bleed."""
    hm = make_hm()
    poly = box(-76.8, 38.6, -76.6, 38.8)
    mask = F.rasterize_mask([poly], hm)

    elevation = np.array(hm.elevation)
    elevation[mask] += 300.0
    hm = Heightmap(elevation, 10.0, BBOX, 12)

    undilated = F.remove_buildings(hm, [poly], dilation_px=0)
    dilated = F.remove_buildings(hm, [poly], dilation_px=1)
    assert dilated.max() <= undilated.max()


def test_building_removal_without_geometry_is_a_no_op() -> None:
    hm = make_hm()
    np.testing.assert_array_equal(F.remove_buildings(hm, []), hm.elevation)


@pytest.mark.parametrize(
    ("setting", "geoms", "bare_earth", "expected"),
    [
        ("auto", ["g"], False, True),    # Terrarium + buildings -> on
        ("auto", ["g"], True, False),    # 3DEP bare earth -> off
        ("auto", [], False, False),      # wilderness -> off
        ("on", [], True, True),          # manual overrides both
        ("off", ["g"], False, False),
        (True, [], True, True),
        (False, ["g"], False, False),
    ],
)
def test_auto_building_removal_logic(setting, geoms, bare_earth, expected) -> None:
    assert F.should_remove_buildings(setting, geoms, bare_earth) is expected


def test_should_remove_buildings_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        F.should_remove_buildings("maybe", [])


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------


def test_query_is_one_combined_request() -> None:
    query = F.build_query(BBOX, ("water", "buildings"))
    assert query.count("out geom") == 1
    assert 'natural"="water' in query and 'way["building"]' in query
    assert f"timeout:{F.OVERPASS_TIMEOUT_S}" in query


def test_query_rejects_unknown_layer() -> None:
    with pytest.raises(ValueError):
        F.build_query(BBOX, ("elevation",))


def test_fixture_parses_into_water_polygons() -> None:
    """A recorded real Overpass response yields usable geometry."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parsed = F._parse_elements(payload, ("water",))

    assert parsed["water"], "fixture should contain water"
    for geom in parsed["water"]:
        assert geom.geom_type in {"Polygon", "MultiPolygon"}
        assert geom.is_valid


def test_relations_are_parsed_not_dropped() -> None:
    """Big rivers are relations; dropping them loses the main channel."""
    relation = {
        "type": "relation",
        "tags": {"natural": "water"},
        "members": [
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lon": -76.8, "lat": 38.6},
                    {"lon": -76.6, "lat": 38.6},
                    {"lon": -76.6, "lat": 38.8},
                ],
            },
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lon": -76.6, "lat": 38.8},
                    {"lon": -76.8, "lat": 38.8},
                    {"lon": -76.8, "lat": 38.6},
                ],
            },
        ],
    }
    parsed = F._parse_elements({"elements": [relation]}, ("water",))
    assert len(parsed["water"]) == 1
    assert parsed["water"][0].area > 0


def test_cache_avoids_a_second_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls: list[str] = []

    def _fake(query: str) -> dict:
        calls.append(query)
        return payload

    monkeypatch.setattr(F, "_request_overpass", _fake)

    first = F.fetch_osm(BBOX, ("water",), cache_dir=tmp_path)
    second = F.fetch_osm(BBOX, ("water",), cache_dir=tmp_path)

    assert len(calls) == 1, "second call must be served from disk"
    assert len(first["water"]) == len(second["water"])


def test_overpass_failure_degrades_to_terrain_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead Overpass must warn and yield no features, never break a build."""

    def _boom(*args: object, **kwargs: object):
        raise requests.ConnectionError("overpass is down")

    monkeypatch.setattr(F.requests, "post", _boom)

    with pytest.warns(RuntimeWarning, match="Overpass"):
        result = F.fetch_osm_or_warn(BBOX, ("water",), cache_dir=tmp_path)

    assert not result
    assert result["water"] == []


def test_overpass_retries_once_on_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    codes = [429, 200]
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code

        def json(self) -> dict:
            return payload

    monkeypatch.setattr(F.time, "sleep", lambda _s: None)
    monkeypatch.setattr(F, "_post", lambda url, q: _Resp(codes.pop(0)))

    assert F._request_overpass("q") == payload
    assert not codes, "both the 429 and the retry should have been consumed"


def test_error_message_names_overpass(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 500

        def json(self) -> dict:  # pragma: no cover - never reached
            return {}

    monkeypatch.setattr(F, "_post", lambda url, q: _Resp())
    with pytest.raises(F.OverpassError, match="Overpass"):
        F._request_overpass("q")


# ---------------------------------------------------------------------------
# Styles, z-order and pipeline position
# ---------------------------------------------------------------------------


def test_style_layers() -> None:
    assert F.STYLE_LAYERS["minimal"] == ()
    assert "water" in F.STYLE_LAYERS["natural"]
    assert "water" in F.STYLE_LAYERS["detailed"]


def test_layers_for_style_includes_buildings_when_removal_possible() -> None:
    assert "buildings" in F.layers_for_style("natural", "auto")
    assert "buildings" in F.layers_for_style("minimal", "on")
    assert "buildings" not in F.layers_for_style("natural", "off")


def test_apply_features_records_the_water_mask() -> None:
    hm = make_hm()
    poly = box(-76.8, 38.6, -76.6, 38.8)
    out = F.apply_features(hm, "natural", F.FeatureSet({"water": [poly]}))

    assert out.water_mask is not None and out.water_mask.any()
    assert out.elevation.min() < hm.elevation.min()
    # minimal stamps nothing.
    assert F.apply_features(hm, "minimal", F.FeatureSet({"water": [poly]})).water_mask is None


def test_apply_features_rejects_unknown_style() -> None:
    with pytest.raises(ValueError):
        F.apply_features(make_hm(), "fancy", F.FeatureSet({}))


def test_building_removal_runs_before_despike_and_smooth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Z-order spy: removal must precede cleanup so exaggeration sizes on ground."""
    from terrframe import heightmap as hm_mod

    calls: list[str] = []

    def _fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        return np.full((TILE_SIZE, TILE_SIZE), 300.0, dtype=np.float32)

    monkeypatch.setattr(hm_mod, "fetch_tile", _fetch)
    for name in ("despike", "smooth", "exaggerate"):
        real = getattr(hm_mod, name)

        def _spy(*a, _n=name, _r=real, **k):
            calls.append(_n)
            return _r(*a, **k)

        monkeypatch.setattr(hm_mod, name, _spy)
    monkeypatch.setattr(hm_mod, "_despike", hm_mod.despike)

    def _hook(hm: Heightmap) -> np.ndarray:
        calls.append("remove_buildings")
        return hm.elevation

    build_heightmap(38.75, -77.13, 38.79, -77.08, target_px=200, pre_clean=_hook)

    assert calls.index("remove_buildings") < calls.index("despike"), calls
    assert calls.index("remove_buildings") < calls.index("exaggerate"), calls


def test_stamps_are_applied_after_exaggeration() -> None:
    """A stamp's printed depth must not scale with terrain exaggeration."""
    poly = box(-76.8, 38.6, -76.6, 38.8)
    scale = F._mm_per_meter(make_hm())

    depths = []
    for factor in (1.0, 5.0):
        elevation = np.full((100, 100), 100.0 * factor, dtype=np.float32)
        hm = Heightmap(elevation, 10.0, BBOX, 12, exaggeration=factor)
        out = F.apply_features(hm, "natural", F.FeatureSet({"water": [poly]}))
        mask = F.rasterize_mask([poly], hm)
        depths.append((elevation.max() - out.elevation[mask].max()) * scale)

    # 1e-4 mm is a tenth of a micron -- orders below any printer, and the
    # tightest float32 can express when subtracting ~40 from ~500.
    assert depths[0] == pytest.approx(depths[1], abs=1e-4)
    assert depths[0] == pytest.approx(F.WATER_DEPTH_MM, abs=1e-4)


def test_features_never_break_watertightness() -> None:
    """Stamps are array ops, but the invariant is asserted, not assumed."""
    from terrframe.mesh import heightmap_to_mesh

    yy, xx = np.mgrid[0:60, 0:60]
    elevation = (500.0 + 200.0 * np.sin(xx / 12.0) * np.cos(yy / 12.0)).astype(np.float32)
    hm = Heightmap(elevation, 30.0, BBOX, 12)

    stamped = F.apply_features(hm, "natural", F.FeatureSet({"water": [box(-76.9, 38.2, -76.3, 38.7)]}))
    mesh = heightmap_to_mesh(stamped)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.euler_number == 2
