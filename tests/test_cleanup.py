"""Tests for the despike and smooth cleanup stages. All offline."""

from __future__ import annotations

import numpy as np
import pytest

from terrframe import heightmap as hm
from terrframe.heightmap import (
    MAX_SPIKE_CLUSTER_PX,
    SMOOTH_GROUND_METERS,
    SMOOTH_SIGMA_MAX,
    SMOOTH_SIGMA_MIN,
    auto_smooth_sigma,
    build_heightmap,
    despike,
    smooth,
)
from terrframe.tiles import TILE_SIZE

SPIKE_SITES = [(10, 12), (20, 40), (33, 7), (48, 55), (55, 30)]


def _gradient(rows: int = 64, cols: int = 64) -> np.ndarray:
    """A smooth diagonal ramp -- no terrain features to confuse the test."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    return (100.0 + 3.0 * xx + 2.0 * yy).astype(np.float32)


# ---------------------------------------------------------------------------
# despike
# ---------------------------------------------------------------------------


def test_despike_removes_isolated_spikes() -> None:
    """Injected single-pixel needles go, and the ramp under them is untouched."""
    clean = _gradient()
    spiked = clean.copy()
    for row, col in SPIKE_SITES:
        spiked[row, col] += 500.0

    out = despike(spiked)

    mask = np.zeros(clean.shape, dtype=bool)
    for row, col in SPIKE_SITES:
        mask[row, col] = True

    # Each spike is back inside the range of the neighbours around it.
    for row, col in SPIKE_SITES:
        window = clean[row - 2 : row + 3, col - 2 : col + 3]
        assert window.min() <= out[row, col] <= window.max(), f"spike at {(row, col)} survived"

    # Everything else is bit-for-bit the input.
    np.testing.assert_allclose(out[~mask], spiked[~mask])


def test_despike_handles_negative_spikes() -> None:
    """Sinkholes are outliers too, not just towers."""
    clean = _gradient()
    spiked = clean.copy()
    spiked[25, 25] -= 400.0

    out = despike(spiked)

    window = clean[23:28, 23:28]
    assert window.min() <= out[25, 25] <= window.max()


def test_despike_preserves_a_sharp_ridgeline() -> None:
    """A real crest must survive: it is connected, so it is not a needle."""
    rows, cols = 64, 64
    arr = _gradient(rows, cols)
    ridge_col = 30
    arr[:, ridge_col] += 300.0  # a one-pixel-wide crest, the hardest case
    peak_before = float(arr[:, ridge_col].max())

    out = despike(arr)

    peak_after = float(out[:, ridge_col].max())
    assert peak_after == pytest.approx(peak_before, rel=0.05), "ridge crest was eroded"
    # And not just the peak: the whole crest holds up.
    assert out[:, ridge_col].mean() == pytest.approx(arr[:, ridge_col].mean(), rel=0.05)


def test_despike_preserves_a_diagonal_ridge() -> None:
    """Connectivity is 8-way, so diagonal crests count as connected too."""
    rows = cols = 64
    arr = _gradient(rows, cols)
    for i in range(rows):
        arr[i, i] += 300.0
    before = arr.diagonal().copy()

    out = despike(arr)

    np.testing.assert_allclose(out.diagonal(), before, rtol=0.05)


def test_despike_removes_a_spike_next_to_a_ridge() -> None:
    """Preserving ridges must not amount to preserving everything."""
    arr = _gradient()
    arr[:, 30] += 300.0
    arr[15, 10] += 500.0  # an isolated needle well clear of the crest

    out = despike(arr)

    assert out[15, 10] < arr[15, 10] - 100.0, "needle should have been cut down"
    assert out[15, 30] == pytest.approx(arr[15, 30], rel=0.05), "ridge should remain"


def test_despike_cluster_limit_is_respected() -> None:
    """A blob at the size limit is a spike; one above it is terrain."""
    small = _gradient()
    small[20:21, 20:23] += 500.0  # 3 px, at or under the limit
    assert MAX_SPIKE_CLUSTER_PX >= 3
    assert despike(small)[20, 21] < small[20, 21] - 100.0

    big = _gradient()
    big[20:30, 20:30] += 500.0  # 100 px, a plateau
    assert despike(big)[25, 25] == pytest.approx(big[25, 25])


def test_despike_leaves_clean_terrain_alone() -> None:
    """Ordinary rough terrain has no outliers, so nothing should move much."""
    rng = np.random.default_rng(5)
    arr = _gradient() + rng.normal(0.0, 2.0, (64, 64)).astype(np.float32)

    out = despike(arr)

    changed = int((out != arr).sum())
    assert changed < arr.size * 0.02, f"{changed} pixels altered on clean terrain"


def test_despike_threshold_controls_sensitivity() -> None:
    """A lower threshold removes at least as much as a higher one."""
    rng = np.random.default_rng(9)
    arr = _gradient() + rng.normal(0.0, 3.0, (64, 64)).astype(np.float32)
    arr[30, 30] += 200.0

    aggressive = int((despike(arr, threshold=1.5) != arr).sum())
    relaxed = int((despike(arr, threshold=6.0) != arr).sum())
    assert aggressive >= relaxed


def test_despike_edge_cases() -> None:
    flat = np.full((16, 16), 300.0, dtype=np.float32)
    np.testing.assert_array_equal(despike(flat), flat)

    with pytest.raises(ValueError):
        despike(np.zeros((4, 4)), threshold=0.0)
    with pytest.raises(ValueError):
        despike(np.zeros((4, 4, 4)))


# ---------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------


def _gradient_roughness(arr: np.ndarray, border: int = 12) -> float:
    """How jagged a surface is, as the spread of its slopes.

    Measured on the interior: ``gaussian_filter`` reflects at the edges, which
    flattens a sloping surface there and injects gradient variance that grows
    with sigma. That is an artefact of the boundary, not of the smoothing, and
    it is the only place this metric is non-monotonic.
    """
    interior = arr[border:-border, border:-border] if border else arr
    return float(np.std(np.gradient(interior)))


def test_smooth_sigma_zero_is_identity() -> None:
    rng = np.random.default_rng(1)
    arr = rng.uniform(0.0, 1000.0, (32, 32)).astype(np.float32)

    np.testing.assert_array_equal(smooth(arr, 0.0), arr)
    np.testing.assert_array_equal(smooth(arr, -1.0), arr)


def test_smooth_monotonically_reduces_roughness() -> None:
    """More blur is always less jagged, never more."""
    rng = np.random.default_rng(2)
    arr = (_gradient() + rng.normal(0.0, 25.0, (64, 64))).astype(np.float32)

    roughness = [_gradient_roughness(smooth(arr, s)) for s in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]]

    assert all(b <= a for a, b in zip(roughness, roughness[1:])), roughness
    assert roughness[-1] < roughness[0] * 0.5, "heavy blur should visibly calm the surface"


def test_smooth_preserves_the_overall_surface() -> None:
    """Blurring is not supposed to move the terrain up or down."""
    rng = np.random.default_rng(4)
    arr = (_gradient() + rng.normal(0.0, 10.0, (64, 64))).astype(np.float32)

    out = smooth(arr, 2.0)

    assert out.mean() == pytest.approx(arr.mean(), rel=0.01)
    assert out.dtype == np.float32


def test_smooth_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        smooth(np.zeros((4, 4, 4)), 1.0)


# ---------------------------------------------------------------------------
# auto sigma
# ---------------------------------------------------------------------------


def test_auto_smooth_sigma_clamps_at_both_ends() -> None:
    """Fine grids hit the ceiling, coarse grids hit the floor.

    Derived from the constants rather than frozen numbers: these are tuning
    knobs, and a hardcoded expectation here just breaks every time they move
    without saying anything about whether clamping still works.
    """
    # A resolution fine enough that the ideal sigma overshoots the ceiling.
    fine = SMOOTH_GROUND_METERS / (SMOOTH_SIGMA_MAX * 2.0)
    assert SMOOTH_GROUND_METERS / fine > SMOOTH_SIGMA_MAX, "test's own premise"
    assert auto_smooth_sigma(fine) == pytest.approx(SMOOTH_SIGMA_MAX)

    # And one coarse enough that it undershoots the floor.
    coarse = SMOOTH_GROUND_METERS / (SMOOTH_SIGMA_MIN / 2.0)
    assert SMOOTH_GROUND_METERS / coarse < SMOOTH_SIGMA_MIN, "test's own premise"
    assert auto_smooth_sigma(coarse) == pytest.approx(SMOOTH_SIGMA_MIN)


def test_auto_smooth_sigma_targets_a_constant_ground_radius() -> None:
    """Between the clamps, sigma x resolution is the target ground distance."""
    for mpp in (10.0, 20.0, 30.0):
        sigma = auto_smooth_sigma(mpp)
        assert SMOOTH_SIGMA_MIN < sigma < SMOOTH_SIGMA_MAX
        assert sigma * mpp == pytest.approx(SMOOTH_GROUND_METERS)


def test_print_cap_only_bites_on_tight_framings() -> None:
    """Blur is aimed in ground metres but capped in printed millimetres.

    Without the cap the same 40 m target costs 0.29 mm of print on a 28 km
    bbox and 1.32 mm on a 6 km one, so tight framings came out blank while
    wide ones were untouched. The cap must change the former and not the
    latter.
    """
    from terrframe.heightmap import PRINT_WIDTH_MM, SMOOTH_PRINT_MM_MAX

    # A tight framing: fine ground resolution, so the ground target overshoots.
    tight_mpp, tight_cols = 7.4, 817
    tight_mm_px = PRINT_WIDTH_MM / (tight_cols - 1)
    uncapped = auto_smooth_sigma(tight_mpp)
    capped = auto_smooth_sigma(tight_mpp, tight_mm_px)
    assert capped < uncapped
    assert capped * tight_mm_px == pytest.approx(SMOOTH_PRINT_MM_MAX, rel=1e-6)

    # A wide framing: the ground target already fits inside the print budget.
    wide_mpp, wide_cols = 29.7, 933
    wide_mm_px = PRINT_WIDTH_MM / (wide_cols - 1)
    assert auto_smooth_sigma(wide_mpp, wide_mm_px) == pytest.approx(
        auto_smooth_sigma(wide_mpp)
    ), "the cap must not touch framings that were already within budget"


def test_print_cap_rejects_bad_scale() -> None:
    with pytest.raises(ValueError):
        auto_smooth_sigma(10.0, 0.0)


def test_auto_smooth_sigma_is_monotonic() -> None:
    resolutions = np.linspace(5.0, 150.0, 30)
    sigmas = [auto_smooth_sigma(float(m)) for m in resolutions]
    assert all(b <= a for a, b in zip(sigmas, sigmas[1:]))


def test_auto_smooth_sigma_rejects_bad_resolution() -> None:
    with pytest.raises(ValueError):
        auto_smooth_sigma(0.0)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

BBOX = (46.75, -121.95, 46.95, -121.55)


@pytest.fixture
def stub_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Noisy suburban-style terrain: gentle relief plus clutter and needles."""

    def _fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((x, y, zoom))) % (2**32))
        rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
        base = 200.0 + 60.0 * (rows + cols)
        clutter = rng.normal(0.0, 4.0, (TILE_SIZE, TILE_SIZE))
        arr = (base + clutter).astype(np.float32)
        arr[::37, ::41] += 120.0  # scattered needles
        return arr

    monkeypatch.setattr(hm, "fetch_tile", _fetch)


def test_pipeline_smooths_before_exaggerating(
    stub_tiles: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order matters: exaggerating first would amplify the noise, not the land."""
    calls: list[str] = []

    for name in ("fill_nodata", "despike", "smooth", "flatten_water", "exaggerate"):
        real = getattr(hm, name)

        def _spy(*args: object, _name: str = name, _real: object = real, **kwargs: object):
            calls.append(_name)
            return _real(*args, **kwargs)  # type: ignore[operator]

        monkeypatch.setattr(hm, name, _spy)

    # build_heightmap reaches despike through this alias, so it needs spying too.
    monkeypatch.setattr(hm, "_despike", getattr(hm, "despike"))

    build_heightmap(*BBOX, exaggeration=2.0)

    assert "smooth" in calls and "exaggerate" in calls
    assert calls.index("smooth") < calls.index("exaggerate"), calls
    assert calls.index("despike") < calls.index("smooth"), calls
    assert calls.index("fill_nodata") < calls.index("despike"), calls
    assert calls.index("smooth") < calls.index("flatten_water"), calls


def test_cleanup_reduces_roughness_end_to_end(stub_tiles: None) -> None:
    """The whole point: cleaned output is calmer than raw output."""
    raw = build_heightmap(*BBOX, smooth_px=None, despike=False)
    cleaned = build_heightmap(*BBOX)

    assert _gradient_roughness(cleaned.elevation) < _gradient_roughness(raw.elevation)
    assert cleaned.elevation.shape == raw.elevation.shape
    assert np.isfinite(cleaned.elevation).all()


def test_cleanup_survives_exaggeration(stub_tiles: None) -> None:
    """Exaggerating cleaned terrain stays calmer than exaggerating raw terrain."""
    raw = build_heightmap(*BBOX, smooth_px=None, despike=False, exaggeration=4.0)
    cleaned = build_heightmap(*BBOX, exaggeration=4.0)

    assert _gradient_roughness(cleaned.elevation) < _gradient_roughness(raw.elevation)


def test_despike_flag_is_honoured(stub_tiles: None) -> None:
    """Turning despiking off leaves the needles standing."""
    with_spikes = build_heightmap(*BBOX, smooth_px=None, despike=False)
    without = build_heightmap(*BBOX, smooth_px=None, despike=True)

    assert without.elevation.max() < with_spikes.elevation.max()


def test_smooth_px_none_disables_smoothing(stub_tiles: None) -> None:
    sharp = build_heightmap(*BBOX, smooth_px=None, despike=False)
    blurred = build_heightmap(*BBOX, smooth_px=3.0, despike=False)

    assert _gradient_roughness(blurred.elevation) < _gradient_roughness(sharp.elevation)


def test_auto_smoothing_matches_the_grids_resolution(stub_tiles: None) -> None:
    """'auto' resolves against the finished grid's own metres-per-pixel."""
    result = build_heightmap(*BBOX)
    expected_sigma = auto_smooth_sigma(result.meters_per_px)

    explicit = build_heightmap(*BBOX, smooth_px=expected_sigma)
    np.testing.assert_allclose(result.elevation, explicit.elevation, rtol=1e-6)
