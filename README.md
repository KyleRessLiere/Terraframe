# terrframe

Turn a geographic bounding box into a 3D-printable STL terrain model.

## Status

Early. The elevation tile layer is done; everything downstream is a stub.

| Stage | Module | State |
| --- | --- | --- |
| Elevation tiles | `terrframe.tiles` | ✅ implemented |
| Heightmap stitch / crop / reproject | `terrframe.heightmap` | 🚧 stub |
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
