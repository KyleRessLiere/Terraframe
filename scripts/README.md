# scripts/

Two review tools that sit on top of the `terrframe` pipeline. Neither is needed
to produce an STL — they exist so you can *look* at terrain and tune settings
before committing to a print.

| Script | What it does |
| --- | --- |
| `preview.py` | Renders one hillshaded PNG of a bbox |
| `batch.py` | Renders a sweep of settings into a timestamped run directory |

Run everything from the repo root. Every command below is on one line so it
pastes into PowerShell and bash alike — PowerShell continues lines with a
backtick, not a backslash.

---

## preview.py

One bbox, one PNG. Use it to check whether an area is worth printing.

```bash
python scripts/preview.py --bbox 46.75,-121.95,46.95,-121.55 -o rainier.png
```

Bbox is `S,W,N,E` in degrees. Some that are known to look good:

```bash
# Mount Rainier
python scripts/preview.py --bbox 46.75,-121.95,46.95,-121.55 -o rainier.png

# Lake Tahoe
python scripts/preview.py --bbox 38.85,-120.25,39.35,-119.85 -o tahoe.png

# Stoneybrooke, VA -- flat suburb, the hard case for cleanup
python scripts/preview.py --bbox 38.7500,-77.1244,38.7860,-77.0782 --exaggeration 3 -o stoneybrooke.png
```

### Options

| Flag | Default | Notes |
| --- | --- | --- |
| `--bbox S,W,N,E` | required | Degrees |
| `-o`, `--output` | `preview.png` | Output PNG |
| `--exaggeration` | `1.0` | A number only — **no `auto` here** (see below) |
| `--target-px` | `800` | Detail: pixel span of the longer side |
| `--smooth` | `auto` | Sigma in pixels, `auto`, or `none` |
| `--despike` / `--no-despike` | on | Remove isolated outlier pixels |
| `--z-factor` | `1.5` | Shading emphasis only; does not touch the data |

> **`--exaggeration auto` does not work here.** Auto exaggeration lives only in
> the `terrframe` CLI. If you want the preview to match what will print, run
> `terrframe` first, read the factor it announces, then pass that number.

### Tuning smoothing

`--smooth` is in **pixels**, so its real-world effect depends on zoom. The
script prints `scale` (m/px) — multiply to get the ground radius.

```bash
# compare a few by hand
python scripts/preview.py --bbox 38.85,-120.25,39.35,-119.85 --smooth none -o s_none.png
python scripts/preview.py --bbox 38.85,-120.25,39.35,-119.85 --smooth 1    -o s_1.png
python scripts/preview.py --bbox 38.85,-120.25,39.35,-119.85 --smooth 2    -o s_2.png
```

`auto` targets ~20 m of ground blur at any zoom, clamped to 0.5–4.0 px. What
that means in practice:

- **Alpine at zoom 11** (~59 m/px): auto lands on 0.5, the floor. One pixel is
  already 59 m, so `--smooth 1` blurs three times harder than auto, and
  `--smooth 2` visibly softens ridgelines and costs ~5% of relief.
- **Suburb at zoom 15** (~3.7 m/px): auto lands on 4.0, the *ceiling* — it
  wants 5.4 and gets clamped. If output still reads rough, raise
  `SMOOTH_SIGMA_MAX` in `src/terrframe/heightmap.py`; lowering
  `SMOOTH_GROUND_METERS` will not help, because the clamp is what binds.

---

## batch.py

Renders a sweep and files it under `runs/<timestamp>/`, so results stay
comparable after you change the tuning constants.

```bash
python scripts/batch.py                   # every preset
python scripts/batch.py --preset tahoe    # just one
```

### Output

```
runs/20260813-135640/
  manifest.json
  stoneybrooke/  smooth1.png  smooth2.png  smooth3.png  smooth4.png  _contact.png
  tahoe/         smooth0.png  smooth1.png  smooth2.png                _contact.png
```

`_contact.png` lays the sweep out side by side — usually the only file you need
to open. `manifest.json` records the timestamp, git commit, exact command, and
per-frame measurements (zoom, m/px, relief, roughness, and roughness against an
uncleaned baseline), so two runs can be compared without reopening images.

Run directories never collide: a second run in the same second fails rather
than overwriting.

### Presets

| Name | Area | Sweep |
| --- | --- | --- |
| `stoneybrooke` | Flat suburb, VA | smooth 1–4 at ×3 exaggeration |
| `tahoe` | Alpine, CA/NV | smooth 0–2 at ×1 |

### Ad-hoc sites

```bash
python scripts/batch.py --bbox 46.75,-121.95,46.95,-121.55 --smooth 0,1,2,3 --exaggeration 1.5 --name rainier --label tuning
```

Creates `runs/<timestamp>_tuning/rainier/`.

| Flag | Default | Notes |
| --- | --- | --- |
| `--preset` | all | Repeatable |
| `--bbox S,W,N,E` | — | Ad-hoc site; overrides presets |
| `--smooth` | `0,1,2,3` | Comma-separated sigmas |
| `--exaggeration` | `1.0` | |
| `--target-px` | `800` | |
| `--despike` / `--no-despike` | on | |
| `--name` | `site` | Folder name for an ad-hoc run |
| `--label` | — | Suffix on the run directory |
| `--runs-dir` | `runs/` | |

> The baseline every frame is scored against is built with **despiking off**,
> so a `smooth 0` row reports despike's own contribution rather than `0.0%`.

### Reading the newest manifest without opening it

```bash
python -c "import json,glob;d=json.load(open(sorted(glob.glob('runs/*/manifest.json'))[-1]));[print(s['name'], 'sigma', r['smooth'], 'rough', r['roughness'], r['roughness_vs_raw_pct']) for s in d['sites'] for r in s['renders']]"
```

```
tahoe sigma 0.0 rough 11.3785 0.0
tahoe sigma 1.0 rough 10.3371 9.2
tahoe sigma 2.0 rough 9.1565 19.6
```

Only single quotes inside the double-quoted argument — nesting escaped double
quotes works in bash but is a parse error in PowerShell.

---

## Producing an actual model

The scripts only make pictures. To get something printable:

```bash
terrframe --bbox 46.75,-121.95,46.95,-121.55 -o rainier.stl
```

```
auto exaggeration: x1.44 (3817 m relief over 30.5 km)
wrote        rainier.stl
size         200.0 x 146.4 x 42.1 mm
geometry     250,958 vertices, 501,912 faces
elevation    566 to 4383 m
exaggeration x1.44
scale        52.3 m/px, zoom 11
watertight   yes
```

`terrframe --help` lists the rest (`--width-mm`, `--base-mm`, `--flatten-water`,
`--max-vertices`, and the same `--smooth` / `--despike` cleanup flags). Output
format follows the extension: `.stl` or `.3mf`.

---

## Notes

- **Tiles are cached** under `.tile_cache/`. The first render of a new area hits
  the network; everything after is local and near-instant. Re-running a sweep is
  cheap.
- **Both scripts import `terrframe`**, so the package must be installed
  (`pip install -e ".[dev]"` from the repo root). It is an editable install, so
  source edits take effect with no reinstall.
- `runs/`, `.tile_cache/`, `*.stl` and `*.3mf` are gitignored.
