# terrframe

Turn a geographic bounding box into a 3D-printable STL terrain model.

## Status

Working end to end: a bounding box in, a printable STL out.

| Stage | Module | State |
| --- | --- | --- |
| Elevation tiles | `terrframe.tiles` | ✅ implemented |
| Heightmap stitch / crop / resample | `terrframe.heightmap` | ✅ implemented |
| Preview renderer | `scripts/preview.py` | ✅ implemented |
| Mesh generation + STL/3MF export | `terrframe.mesh` | ✅ implemented |
| CLI | `terrframe.cli` | ✅ implemented |

## Quick start

```bash
terrframe --bbox 38.85,-120.25,39.35,-119.85 -o tahoe.stl
```

```
auto exaggeration: x3.49 (1785 m relief over 34.6 km)
wrote        tahoe.stl
size         200.0 x 322.3 x 42.1 mm
geometry     550,478 vertices, 1,100,952 faces
elevation    1528 to 3313 m
exaggeration x3.49
scale        59.3 m/px, zoom 11
watertight   yes
```

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
# .venv/bin/python -m pip install -e ".[dev]"     # macOS / Linux
```

## Elevation data

Heights come from the [Terrarium](https://github.com/tilezen/joerd/blob/master/docs/formats.md#terrarium)
tileset hosted on S3, which packs metres above sea level into a normal PNG:

```
elevation_m = (R * 256 + G + B / 256) - 32768
```

Ocean tiles carry bathymetry, so values below zero are expected and real.

## Usage

```python
from terrframe.tiles import zoom_for_bbox, tiles_for_bbox, fetch_tile

# Mount Rainier National Park
south, west, north, east = 46.75, -121.95, 46.95, -121.55

zoom = zoom_for_bbox(south, west, north, east, target_px=800)  # -> 11
tiles = tiles_for_bbox(south, west, north, east, zoom)         # covers bbox + 1-tile margin

elevations = fetch_tile(*tiles[0], zoom)   # (256, 256) float32 array of metres
```

Raw PNGs are cached under `.tile_cache/{z}/{x}/{y}.png`. A cached tile is never
re-downloaded; a corrupt one is discarded and fetched again.

The one-tile margin on `tiles_for_bbox` is deliberate — resampling near the edge
of the requested area needs real samples just outside it, otherwise the boundary
picks up seams and clamped gradients.

## Building a heightmap

```python
from terrframe.heightmap import build_heightmap

hm = build_heightmap(46.75, -121.95, 46.95, -121.55, target_px=800, exaggeration=1.5)

hm.elevation       # (427, 583) float32 metres, row 0 at the north edge
hm.meters_per_px   # 52.3 -- equal on both axes, so prints are not stretched
hm.bbox            # what the grid actually covers (a superset of the request)
hm.size_meters     # (30510.0, 22250.0)
```

The pipeline is zoom pick → fetch (with margin) → `stitch` → `crop_to_bbox` →
`resample_to_meters` → `fill_nodata` → `flatten_water` → `exaggerate`.

### On projections

Web Mercator is **conformal**: it already stretches the y axis by exactly
`1 / cos(lat)`, so a raw tile crop's pixels are *already* square on the ground.
A per-axis `cos(lat)` correction here would double-apply that stretch and squash
the terrain north–south.

What Mercator does *not* do is hold scale constant — metres per pixel shrinks
toward the poles. So `resample_to_meters` rebuilds rows at constant ground
spacing (equidistant in latitude) instead of constant Mercator spacing. Scale is
pinned at the bbox centre latitude; the residual error is roughly
`1 - cos(dlat/2) / cos(lat_center)`, under 0.1% for a 1° span at mid-latitudes.

Measured against WGS84 geodesics for the Rainier bbox: pixel aspect is within
0.50% of true, and pixels are square to the same 0.50%.

Cropping always rounds **outward**, so the result covers at least the requested
bbox — `hm.bbox` reports what was actually cut.

## Preview

```bash
python scripts/preview.py --bbox 46.75,-121.95,46.95,-121.55 --exaggeration 1.5 -o preview.png
```

Renders a north-west-lit hillshade blended with a hypsometric tint. Flattened
water is drawn as water rather than as the bottom of the land ramp. `--z-factor`
emphasises the shading without touching the data.

## Meshing

```python
from terrframe.heightmap import build_heightmap
from terrframe.mesh import heightmap_to_mesh, export

hm = build_heightmap(38.85, -120.25, 39.35, -119.85, exaggeration=3.5)
mesh = heightmap_to_mesh(hm, width_mm=200.0, base_mm=6.0)
export(mesh, "tahoe.stl")   # .stl (binary) or .3mf, by extension
```

The solid is a terrain surface on top, four vertical skirt walls, and a flat
bottom at `z = 0`, with the lowest valley floor sitting at `z = base_mm`.

**Watertightness comes from construction, not repair.** One vertex array and
one face array build a single `Trimesh`; the walls reuse the terrain's own edge
vertices, so there is nothing to weld. No boolean unions, no `trimesh.repair`,
and `process=False` so trimesh cannot silently "fix" anything behind the checks.
`heightmap_to_mesh` raises with the open-edge count rather than returning a mesh
that fails `is_watertight` or `is_winding_consistent`.

Vertical scale derives from `hm.meters_per_px`, so the model is geometrically
truthful — including any exaggeration baked in upstream. Past `max_vertices`
(default 4M) the grid is decimated with a bilinear zoom; footprint and vertical
scale are unaffected, because both are pinned to the original ground width.

### Auto exaggeration

`--exaggeration auto` targets printed relief at `TARGET_RELIEF_RATIO` (0.18) of
the print width, clamped to 1.0–5.0:

```
factor = clamp(0.18 * width_m / relief_m, 1.0, 5.0)
```

Flat country gets pushed up; terrain already steeper than 18% of its own width
prints as-is. The constants are module-level in `mesh.py` and meant to be tuned
by hand.

## Tests

```bash
.venv/Scripts/python -m pytest              # offline suite
.venv/Scripts/python -m pytest --network    # also hit the live tile server
```

Tests marked `@pytest.mark.network` are skipped unless `--network` is passed.
The Terrarium decoding is pinned offline against a synthetic PNG with known RGB
values, and separately validated against reality: the real zoom-10 tile over
Mt. Rainier must top out between 4300 and 4450 m.

## License

MIT
