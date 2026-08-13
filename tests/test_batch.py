"""Tests for the batch preview runner. Offline: tile fetching is stubbed."""

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

_BATCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "batch.py"
_spec = importlib.util.spec_from_file_location("batch", _BATCH_PATH)
assert _spec is not None and _spec.loader is not None
batch = importlib.util.module_from_spec(_spec)
sys.modules["batch"] = batch
_spec.loader.exec_module(batch)

BBOX_ARG = "46.75,-121.95,46.95,-121.55"


@pytest.fixture
def stub_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terrain with relief plus clutter, so smoothing has something to do."""

    def _fetch(x: int, y: int, zoom: int, cache_dir: object = None) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((x, y, zoom))) % (2**32))
        rows = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, TILE_SIZE, dtype=np.float32)[None, :]
        dome = np.sin(rows * np.pi) * np.sin(cols * np.pi)
        clutter = rng.normal(0.0, 6.0, (TILE_SIZE, TILE_SIZE))
        return (300.0 + 900.0 * dome + clutter).astype(np.float32)

    monkeypatch.setattr(hm, "fetch_tile", _fetch)


def _run(tmp_path: Path, *extra: str) -> Path:
    """Run batch.py into a temp runs dir and return the run directory."""
    code = batch.main(
        ["--bbox", BBOX_ARG, "--smooth", "0,2", "--target-px", "200",
         "--runs-dir", str(tmp_path), *extra]
    )
    assert code == 0
    runs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(runs) >= 1
    return runs[-1]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_run_creates_a_timestamped_directory(stub_tiles: None, tmp_path: Path) -> None:
    """The run folder is named for when it happened."""
    run_dir = _run(tmp_path, "--name", "site")

    stamp = run_dir.name
    assert len(stamp) == 15 and stamp[8] == "-", f"unexpected stamp {stamp!r}"
    assert stamp[:8].isdigit() and stamp[9:].isdigit()


def test_label_is_appended_to_the_directory_name(stub_tiles: None, tmp_path: Path) -> None:
    run_dir = _run(tmp_path, "--label", "tuning")
    assert run_dir.name.endswith("_tuning")


def test_frames_and_contact_sheet_are_written(stub_tiles: None, tmp_path: Path) -> None:
    """One PNG per sigma, plus a contact sheet, under a per-site folder."""
    run_dir = _run(tmp_path, "--name", "rainier")
    site_dir = run_dir / "rainier"

    assert (site_dir / "smooth0.png").is_file()
    assert (site_dir / "smooth2.png").is_file()
    assert (site_dir / "_contact.png").is_file()

    with Image.open(site_dir / "smooth0.png") as frame, Image.open(
        site_dir / "_contact.png"
    ) as sheet:
        # The sheet holds both frames side by side, so it is wider than one.
        assert sheet.width > frame.width
        assert sheet.height >= frame.height


def test_successive_runs_never_collide(stub_tiles: None, tmp_path: Path) -> None:
    """Two runs land in separate folders; nothing is overwritten."""
    first = _run(tmp_path, "--label", "a")
    second = _run(tmp_path, "--label", "b")

    assert first != second
    assert first.is_dir() and second.is_dir()
    assert (first / "site" / "smooth0.png").is_file()
    assert (second / "site" / "smooth0.png").is_file()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_records_parameters_and_measurements(
    stub_tiles: None, tmp_path: Path
) -> None:
    """The manifest is the point: a run must be readable without the images."""
    run_dir = _run(tmp_path, "--name", "rainier", "--exaggeration", "2")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["run"] == run_dir.name
    assert manifest["timestamp"]
    assert manifest["command"][0] == "batch.py"

    site = manifest["sites"][0]
    assert site["name"] == "rainier"
    assert site["bbox"] == [46.75, -121.95, 46.95, -121.55]
    assert site["auto_smooth_sigma"] > 0
    assert site["raw_roughness"] > 0
    assert site["contact_sheet"] == "rainier/_contact.png"

    assert [r["smooth"] for r in site["renders"]] == [0.0, 2.0]
    for render in site["renders"]:
        assert render["exaggeration"] == 2.0
        assert render["zoom"] > 0
        assert render["meters_per_px"] > 0
        assert render["relief_m"] > 0
        assert len(render["grid"]) == 2
        # Every recorded image path really exists, relative to the run dir.
        assert (run_dir / render["image"]).is_file()


def test_manifest_smoothing_reduces_roughness(stub_tiles: None, tmp_path: Path) -> None:
    """The numbers in the manifest reflect what smoothing actually did."""
    run_dir = _run(tmp_path, "--name", "site")
    site = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["sites"][0]

    unsmoothed, smoothed = site["renders"]
    assert smoothed["roughness"] < unsmoothed["roughness"]
    assert smoothed["roughness_vs_raw_pct"] > unsmoothed["roughness_vs_raw_pct"]
    assert smoothed["smooth_ground_m"] == pytest.approx(2 * smoothed["meters_per_px"], rel=0.01)


def test_manifest_is_valid_json_utf8(stub_tiles: None, tmp_path: Path) -> None:
    run_dir = _run(tmp_path)
    raw = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(raw)


# ---------------------------------------------------------------------------
# Presets and parsing
# ---------------------------------------------------------------------------


def test_presets_are_well_formed() -> None:
    """Both shipped presets describe a usable sweep."""
    assert set(batch.PRESETS) == {"stoneybrooke", "tahoe"}
    for preset in batch.PRESETS.values():
        south, west, north, east = preset.bbox
        assert south < north and west < east
        assert preset.smooth
        assert all(s >= 0 for s in preset.smooth)
        assert preset.exaggeration > 0
        assert preset.note


def test_preset_run_uses_the_preset_name(stub_tiles: None, tmp_path: Path) -> None:
    """Selecting a preset writes into a folder named for it."""
    code = batch.main(["--preset", "tahoe", "--runs-dir", str(tmp_path)])
    assert code == 0

    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert (run_dir / "tahoe").is_dir()

    site = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["sites"][0]
    assert site["name"] == "tahoe"
    assert [r["smooth"] for r in site["renders"]] == [0.0, 1.0, 2.0]


def test_parse_helpers_reject_bad_input() -> None:
    assert batch._parse_bbox("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)
    assert batch._parse_sigmas("0,1.5,3") == [0.0, 1.5, 3.0]

    for bad in ["1,2,3", "a,b,c,d", "4,2,3,1"]:
        with pytest.raises(Exception):
            batch._parse_bbox(bad)
    for bad in ["", "x", "-1"]:
        with pytest.raises(Exception):
            batch._parse_sigmas(bad)


def test_roughness_ignores_the_border(stub_tiles: None) -> None:
    """The metric trims edges, where gaussian reflection fakes extra variance."""
    rng = np.random.default_rng(0)
    arr = rng.normal(0.0, 10.0, (64, 64)).astype(np.float32)

    assert batch.roughness(arr) > 0
    # A tiny array has no interior to trim; it must still return a number.
    assert batch.roughness(np.zeros((4, 4), dtype=np.float32)) == pytest.approx(0.0)
