"""Tests for the terrframe CLI. Offline: tile fetching is stubbed out."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import trimesh

from terrframe import cli
from terrframe import heightmap as hm
from terrframe.tiles import TILE_SIZE

BBOX_ARG = "46.75,-121.95,46.95,-121.55"


def _synthetic_tile(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
    """A tile with a broad dome, so the model has real relief to scale."""
    rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
    dome = np.sin(rows * np.pi) * np.sin(cols * np.pi)
    return (400.0 + 1600.0 * dome + 40.0 * ((x % 7) + (y % 7))).astype(np.float32)


@pytest.fixture
def stub_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hm, "fetch_tile", _synthetic_tile)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_bbox_round_trip() -> None:
    assert cli._parse_bbox(BBOX_ARG) == (46.75, -121.95, 46.95, -121.55)


@pytest.mark.parametrize(
    "bad",
    ["1,2,3", "a,b,c,d", "48,-122,47,-121", "47,-121,48,-122", "47,-121,200,-120"],
)
def test_parse_bbox_rejects_bad_input(bad: str) -> None:
    with pytest.raises(Exception):
        cli._parse_bbox(bad)


def test_parse_exaggeration_accepts_auto_and_numbers() -> None:
    assert cli._parse_exaggeration("auto") == "auto"
    assert cli._parse_exaggeration("AUTO") == "auto"
    assert cli._parse_exaggeration("2.5") == 2.5
    for bad in ["0", "-1", "high"]:
        with pytest.raises(Exception):
            cli._parse_exaggeration(bad)


def test_parse_water_modes() -> None:
    assert cli._parse_water("auto") == "auto"
    assert cli._parse_water("none") is None
    assert cli._parse_water("off") is None
    assert cli._parse_water("-50") == -50.0
    with pytest.raises(Exception):
        cli._parse_water("wet")


# ---------------------------------------------------------------------------
# End to end, in process
# ---------------------------------------------------------------------------


def test_main_writes_a_watertight_stl(stub_tiles: None, tmp_path: Path) -> None:
    """The default invocation produces a sealed, loadable solid."""
    out = tmp_path / "model.stl"
    code = cli.main(["--bbox", BBOX_ARG, "-o", str(out)])

    assert code == 0
    assert out.is_file() and out.stat().st_size > 0

    mesh = trimesh.load(out)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0


def test_auto_exaggeration_is_reported_on_stderr(
    stub_tiles: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user must be told what 'auto' picked, not just handed a file."""
    cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "m.stl")])
    captured = capsys.readouterr()

    assert "auto exaggeration" in captured.err
    assert "x" in captured.err
    # And the summary lands on stdout, separately.
    assert "watertight   yes" in captured.out


def test_summary_reports_the_requested_dimensions(
    stub_tiles: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(
        ["--bbox", BBOX_ARG, "-o", str(tmp_path / "m.stl"), "--width-mm", "150", "--base-mm", "4"]
    )
    out = capsys.readouterr().out

    assert "size         150.0 x" in out
    for field in ("wrote", "geometry", "elevation", "exaggeration", "scale", "watertight"):
        assert field in out


def test_explicit_exaggeration_skips_auto(
    stub_tiles: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "m.stl"), "--exaggeration", "2.0"])
    captured = capsys.readouterr()

    assert "auto exaggeration" not in captured.err
    assert "exaggeration x2.00" in captured.out


def test_exaggeration_changes_model_height(stub_tiles: None, tmp_path: Path) -> None:
    """A bigger factor really does make a taller print."""
    flat = tmp_path / "flat.stl"
    tall = tmp_path / "tall.stl"
    cli.main(["--bbox", BBOX_ARG, "-o", str(flat), "--exaggeration", "1.0"])
    cli.main(["--bbox", BBOX_ARG, "-o", str(tall), "--exaggeration", "3.0"])

    a, b = trimesh.load(flat), trimesh.load(tall)
    base_mm = 6.0
    # Relief above the base triples; the base itself does not.
    assert (b.extents[2] - base_mm) == pytest.approx(3.0 * (a.extents[2] - base_mm), rel=1e-3)


def test_reported_elevation_is_geographic_not_exaggerated(
    stub_tiles: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The summary quotes real metres, so x3 does not invent a 6 km mountain."""
    cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "a.stl"), "--exaggeration", "1.0"])
    plain = _elevation_line(capsys.readouterr().out)

    cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "b.stl"), "--exaggeration", "3.0"])
    tripled = _elevation_line(capsys.readouterr().out)

    assert plain == tripled


def _elevation_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("elevation"))


def test_output_format_follows_the_extension(stub_tiles: None, tmp_path: Path) -> None:
    out = tmp_path / "model.3mf"
    assert cli.main(["--bbox", BBOX_ARG, "-o", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 0


def test_unsupported_extension_exits_nonzero(
    stub_tiles: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "model.obj")])
    assert code == 1
    assert "unsupported output format" in capsys.readouterr().err


def test_max_vertices_is_respected(stub_tiles: None, tmp_path: Path) -> None:
    out = tmp_path / "small.stl"
    cli.main(["--bbox", BBOX_ARG, "-o", str(out), "--max-vertices", "3000"])
    mesh = trimesh.load(out)
    assert len(mesh.vertices) <= 3000
    assert mesh.is_watertight


def test_flatten_water_none_is_honoured(stub_tiles: None, tmp_path: Path) -> None:
    """'none' must reach build_heightmap as None, not the string."""
    seen: dict[str, object] = {}
    real = cli.build_heightmap

    def _spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)

    cli.build_heightmap = _spy  # type: ignore[assignment]
    try:
        cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "m.stl"), "--flatten-water", "none"])
    finally:
        cli.build_heightmap = real  # type: ignore[assignment]

    assert seen["flatten_water_level"] is None


def test_parse_smooth_modes() -> None:
    assert cli._parse_smooth("auto") == "auto"
    assert cli._parse_smooth("none") is None
    assert cli._parse_smooth("off") is None
    assert cli._parse_smooth("2.5") == 2.5
    assert cli._parse_smooth("0") == 0.0
    for bad in ["-1", "blurry"]:
        with pytest.raises(Exception):
            cli._parse_smooth(bad)


def test_cleanup_flags_reach_the_pipeline(stub_tiles: None, tmp_path: Path) -> None:
    """--smooth and --no-despike are forwarded, not silently dropped."""
    seen: dict[str, object] = {}
    real = cli.build_heightmap

    def _spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)

    cli.build_heightmap = _spy  # type: ignore[assignment]
    try:
        cli.main(
            [
                "--bbox", BBOX_ARG,
                "-o", str(tmp_path / "m.stl"),
                "--smooth", "2.5",
                "--no-despike",
                "--despike-threshold", "4.5",
            ]
        )
    finally:
        cli.build_heightmap = real  # type: ignore[assignment]

    assert seen["smooth_px"] == 2.5
    assert seen["despike"] is False
    assert seen["despike_threshold"] == 4.5


def test_defaults_enable_cleanup(stub_tiles: None, tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    real = cli.build_heightmap

    def _spy(*args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        return real(*args, **kwargs)

    cli.build_heightmap = _spy  # type: ignore[assignment]
    try:
        cli.main(["--bbox", BBOX_ARG, "-o", str(tmp_path / "m.stl")])
    finally:
        cli.build_heightmap = real  # type: ignore[assignment]

    assert seen["smooth_px"] is None, "blur is off by default; removal declutters"
    assert seen["despike"] is True


def test_smoothing_produces_a_calmer_model(stub_tiles: None, tmp_path: Path) -> None:
    """Heavier smoothing means less surface area on the printed terrain."""
    rough = tmp_path / "rough.stl"
    calm = tmp_path / "calm.stl"
    cli.main(["--bbox", BBOX_ARG, "-o", str(rough), "--smooth", "none", "--no-despike"])
    cli.main(["--bbox", BBOX_ARG, "-o", str(calm), "--smooth", "4"])

    assert trimesh.load(calm).area < trimesh.load(rough).area


def test_bad_bbox_exits_with_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--bbox", "48,-122,47,-121", "-o", str(tmp_path / "m.stl")])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# Subprocess smoke test
# ---------------------------------------------------------------------------

_RUNNER = """
import sys
import numpy as np
from terrframe import heightmap as hm
from terrframe.tiles import TILE_SIZE

def fake_fetch(x, y, zoom, cache_dir=None):
    rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
    dome = np.sin(rows * np.pi) * np.sin(cols * np.pi)
    return (400.0 + 1600.0 * dome + 40.0 * ((x % 7) + (y % 7))).astype(np.float32)

hm.fetch_tile = fake_fetch

from terrframe.cli import main
sys.exit(main())
"""


def test_cli_subprocess_smoke(tmp_path: Path) -> None:
    """Run the real entry point in a fresh interpreter, no network."""
    runner = tmp_path / "runner.py"
    runner.write_text(textwrap.dedent(_RUNNER))
    out = tmp_path / "tahoe.stl"

    result = subprocess.run(
        [sys.executable, str(runner), "--bbox", "38.85,-120.25,39.35,-119.85", "-o", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out.is_file() and out.stat().st_size > 0
    assert "watertight   yes" in result.stdout
    assert "auto exaggeration" in result.stderr

    mesh = trimesh.load(out)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0


def test_console_script_is_installed() -> None:
    """`terrframe --help` must work as an installed command."""
    result = subprocess.run(
        [sys.executable, "-m", "terrframe.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "--bbox" in result.stdout
