# terrframe

Turn a geographic bounding box into a 3D-printable STL terrain model.

## Status

Early. The elevation tile layer is done; everything downstream is a stub.

| Stage | Module | State |
| --- | --- | --- |
| Elevation tiles | `terrframe.tiles` | ✅ implemented |
| Heightmap stitch / crop / resample | `terrframe.heightmap` | ✅ implemented |
| Preview renderer | `scripts/preview.py` | ✅ implemented |
| Mesh generation + STL export | `terrframe.mesh` | 🚧 stub |
| CLI | `terrframe.cli` | 🚧 stub |

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
