"""Tests for deep water recess and shoreline beads.

Offline throughout, built on a synthetic circular lake with a central island,
so interior rings and MultiPolygon handling are exercised without a network.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, box

from terrframe import features as F
from terrframe.heightmap import Heightmap

# 1 px == 0.01 degrees, so geometry maps to pixels predictably.
BBOX = (38.0, -77.0, 39.0, -76.0)
TERRAIN_M = 100.0


def make_hm(rows: int = 200, cols: int = 200) -> Heightmap:
    return Heightmap(
        elevation=np.full((rows, cols), TERRAIN_M, dtype=np.float32),
        meters_per_px=10.0,
        bbox=BBOX,
        zoom=12,
    )


def lake_with_island():
    """A circular lake centred in the grid with a smaller island inside it."""
    centre = Point(-76.5, 38.5)
    lake = centre.buffer(0.25)
    island = centre.buffer(0.06)
    return lake.difference(island)


def scale_of(hm: Heightmap) -> float:
    return F._mm_per_meter(hm)


# ---------------------------------------------------------------------------
# Deep recess
# ---------------------------------------------------------------------------


def test_default_depth_is_one_millimetre() -> None:
    assert F.WATER_DEPTH_MM == 1.0


def test_water_plane_is_flat_at_exactly_the_requested_depth() -> None:
    hm = make_hm()
    lake = lake_with_island()
    depth_mm = 1.0

    out = F.stamp_water(hm, [lake], depth_mm=depth_mm)
    mask = F.rasterize_mask([lake], hm)

    values = np.unique(out[mask])
    assert values.size == 1, "water must be a dead-flat plane"
    drop_mm = (TERRAIN_M - float(values[0])) * scale_of(hm)
    assert drop_mm == pytest.approx(depth_mm, abs=1e-4)


def test_depth_is_honoured_exactly_across_values() -> None:
    hm = make_hm()
    lake = lake_with_island()
    mask = F.rasterize_mask([lake], hm)

    for depth_mm in (0.8, 1.0, 1.4):
        out = F.stamp_water(hm, [lake], depth_mm=depth_mm)
        drop_mm = (TERRAIN_M - float(out[mask].max())) * scale_of(hm)
        assert drop_mm == pytest.approx(depth_mm, abs=1e-4)


# ---------------------------------------------------------------------------
# Cross-section
# ---------------------------------------------------------------------------


def _centre_row_profile(array: np.ndarray) -> np.ndarray:
    return array[array.shape[0] // 2, :]


def test_cross_section_order_is_water_then_terrain_then_crown() -> None:
    """Walking the centre row: flat water < terrain < bead crown."""
    hm = make_hm()
    lake = lake_with_island()

    recessed = F.stamp_water(hm, [lake], depth_mm=1.0)
    final = F.stamp_shoreline(hm, [lake], height_mm=0.4, elevation=recessed)

    water = F.rasterize_mask([lake], hm)
    row = final.shape[0] // 2
    water_row = water[row]

    water_values = final[row][water_row]
    assert np.unique(water_values).size == 1, "water stays flat under the bead pass"

    water_level = float(water_values[0])
    crown = float(_centre_row_profile(final).max())

    assert water_level < TERRAIN_M < crown, (water_level, TERRAIN_M, crown)


def test_bead_crown_is_local_terrain_plus_height() -> None:
    hm = make_hm()
    lake = lake_with_island()
    height_mm = 0.4

    out = F.stamp_shoreline(hm, [lake], height_mm=height_mm)
    rise_mm = (float(out.max()) - TERRAIN_M) * scale_of(hm)
    assert rise_mm == pytest.approx(height_mm, abs=0.02)


def test_bead_does_not_touch_the_water_plane() -> None:
    """The bead may not bleed inward; water is flat to its very edge."""
    hm = make_hm()
    lake = lake_with_island()

    recessed = F.stamp_water(hm, [lake], depth_mm=1.0)
    beaded = F.stamp_shoreline(hm, [lake], height_mm=0.4, elevation=recessed)
    mask = F.rasterize_mask([lake], hm)

    np.testing.assert_array_equal(beaded[mask], recessed[mask])


def test_bank_wall_is_a_hard_step_within_budget() -> None:
    """Crown to water plane happens inside MAX_BANK_STEP_MM horizontally.

    Run at a production-like grid. The wall is always exactly one pixel wide --
    that is the hardest a heightfield can be -- so whether it clears 0.3 mm is
    a statement about resolution, not about the stamping. At 200 mm across, the
    pipeline's 800-933 px grids give 0.21-0.25 mm per pixel; the 200 px grid
    used elsewhere in this file is ~1 mm per pixel and physically cannot.
    """
    hm = make_hm(rows=900, cols=900)
    assert F._mm_per_pixel(hm) < F.MAX_BANK_STEP_MM, "fixture must be fine enough to test this"
    lake = lake_with_island()

    recessed = F.stamp_water(hm, [lake], depth_mm=1.0)
    final = F.stamp_shoreline(hm, [lake], height_mm=0.4, elevation=recessed)
    water = F.rasterize_mask([lake], hm)

    row = final.shape[0] // 2
    profile = final[row]
    wet = np.where(water[row])[0]
    first_wet = int(wet[0])

    mm_per_px = F._mm_per_pixel(hm)
    crown_index = int(np.argmax(profile[:first_wet]))
    step_mm = (first_wet - crown_index) * mm_per_px

    assert step_mm <= F.MAX_BANK_STEP_MM + 1e-9, f"bank ramps over {step_mm:.3f} mm"
    assert first_wet - crown_index == 1, "the wall should be exactly one pixel"
    # And the wall descends monotonically -- a heightfield cannot undercut,
    # but it can still ramp, which is what this rules out.
    wall = profile[crown_index : first_wet + 1]
    assert np.all(np.diff(wall) <= 1e-4), wall


def test_profile_down_the_wall_is_monotonic_on_both_shores() -> None:
    hm = make_hm()
    lake = lake_with_island()
    final = F.stamp_shoreline(
        hm, [lake], height_mm=0.4, elevation=F.stamp_water(hm, [lake], depth_mm=1.0)
    )
    water = F.rasterize_mask([lake], hm)

    row = final.shape[0] // 2
    wet = np.where(water[row])[0]

    left = final[row][: int(wet[0]) + 1]
    right = final[row][int(wet[-1]) :][::-1]
    for name, wall in (("left", left), ("right", right)):
        crown = int(np.argmax(wall))
        assert np.all(np.diff(wall[crown:]) <= 1e-4), f"{name} wall not monotonic"


# ---------------------------------------------------------------------------
# Islands and multipolygons
# ---------------------------------------------------------------------------


def test_island_interior_ring_gets_a_bead() -> None:
    """Islands are half the charm of a river scene; they must be outlined."""
    hm = make_hm()
    lake = lake_with_island()

    out = F.stamp_shoreline(hm, [lake], height_mm=0.4)
    water = F.rasterize_mask([lake], hm)

    # The island is dry ground fully enclosed by water.
    from scipy.ndimage import binary_fill_holes

    island = binary_fill_holes(water) & ~water
    assert island.any(), "test fixture should have an island"
    assert float(out[island].max()) > TERRAIN_M + 1e-3, "island shore has no bead"


def test_multipolygon_water_is_handled() -> None:
    hm = make_hm()
    multi = MultiPolygon([Point(-76.7, 38.5).buffer(0.1), Point(-76.3, 38.5).buffer(0.1)])

    recessed = F.stamp_water(hm, [multi], depth_mm=1.0)
    final = F.stamp_shoreline(hm, [multi], height_mm=0.4, elevation=recessed)

    mask = F.rasterize_mask([multi], hm)
    assert mask.any()
    assert float(final.max()) > TERRAIN_M
    assert np.unique(recessed[mask]).size == 1


# ---------------------------------------------------------------------------
# Max semantics
# ---------------------------------------------------------------------------


def test_touching_shorelines_do_not_stack() -> None:
    """Two near-touching lakes: the shared band is one bead, not two."""
    hm = make_hm()
    gap = 0.03
    left = box(-76.7, 38.4, -76.5 - gap / 2, 38.6)
    right = box(-76.5 + gap / 2, 38.4, -76.3, 38.6)

    both = F.stamp_shoreline(hm, [left, right], height_mm=0.4)
    single = F.stamp_shoreline(hm, [left], height_mm=0.4)

    assert float(both.max()) == pytest.approx(float(single.max()), abs=1e-4), (
        "overlapping shorelines stacked"
    )


# ---------------------------------------------------------------------------
# Disabling
# ---------------------------------------------------------------------------


def test_shoreline_zero_is_a_strict_no_op() -> None:
    hm = make_hm()
    lake = lake_with_island()
    recessed = F.stamp_water(hm, [lake], depth_mm=1.0)

    np.testing.assert_array_equal(
        F.stamp_shoreline(hm, [lake], height_mm=0.0, elevation=recessed), recessed
    )


def test_shoreline_without_geometry_is_a_no_op() -> None:
    hm = make_hm()
    np.testing.assert_array_equal(F.stamp_shoreline(hm, []), hm.elevation)


def test_apply_features_wires_depth_and_bead_through() -> None:
    hm = make_hm()
    lake = lake_with_island()
    features = F.FeatureSet({"water": [lake]})

    plain = F.apply_features(hm, "natural", features, water_depth_mm=1.0, shoreline_mm=0.0)
    beaded = F.apply_features(hm, "natural", features, water_depth_mm=1.0, shoreline_mm=0.4)

    assert float(plain.elevation.max()) == pytest.approx(TERRAIN_M)
    assert float(beaded.elevation.max()) > TERRAIN_M
    # Depth is unaffected by the bead.
    assert float(plain.elevation.min()) == pytest.approx(float(beaded.elevation.min()))


def test_stamps_stay_constant_under_exaggeration() -> None:
    """Depth and bead height are printed millimetres, not terrain metres."""
    lake = lake_with_island()
    measured = []
    for factor in (1.0, 6.0):
        hm = Heightmap(
            np.full((200, 200), TERRAIN_M * factor, dtype=np.float32),
            10.0,
            BBOX,
            12,
            exaggeration=factor,
        )
        out = F.apply_features(
            hm, "natural", F.FeatureSet({"water": [lake]}), water_depth_mm=1.0, shoreline_mm=0.4
        )
        scale = scale_of(hm)
        base = TERRAIN_M * factor
        measured.append(
            (
                (base - float(out.elevation.min())) * scale,
                (float(out.elevation.max()) - base) * scale,
            )
        )

    assert measured[0][0] == pytest.approx(measured[1][0], abs=1e-3)
    assert measured[0][1] == pytest.approx(measured[1][1], abs=1e-3)


def test_water_features_keep_the_mesh_watertight() -> None:
    from terrframe.mesh import heightmap_to_mesh

    yy, xx = np.mgrid[0:80, 0:80]
    elevation = (400.0 + 120.0 * np.sin(xx / 15.0) * np.cos(yy / 15.0)).astype(np.float32)
    hm = Heightmap(elevation, 30.0, BBOX, 12)

    out = F.apply_features(
        hm, "natural", F.FeatureSet({"water": [lake_with_island()]}),
        water_depth_mm=1.4, shoreline_mm=0.5,
    )
    mesh = heightmap_to_mesh(out)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.euler_number == 2
