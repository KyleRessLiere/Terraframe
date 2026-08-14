"""OpenStreetMap vector features stamped into a heightmap.

Flat metro areas do not survive as elevation alone. Their visual identity is
water and linework: smoothing blurs riverbanks, and a river near sea level
barely registers as relief at all, so a terrain-only render of somewhere like
Washington DC is anonymous texture. OSM polygons put the crisp edges back.

Everything here is an array operation on the heightmap *before* meshing.
Mesh booleans are banned in this codebase; watertightness comes from
construction, and a stamp that only moves z-values cannot break it.

Geometry arrives from Overpass in lon/lat and is rasterised against the
heightmap's own affine, so no reprojection is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, distance_transform_edt
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union

from .heightmap import Heightmap, fill_nodata

__all__ = [
    "LAYER_QUERIES",
    "STYLE_LAYERS",
    "STYLES",
    "FeatureSet",
    "OverpassError",
    "apply_features",
    "building_remover",
    "fetch_osm",
    "fetch_osm_or_warn",
    "layers_for_style",
    "rasterize_mask",
    "remove_buildings",
    "should_remove_buildings",
    "stamp_shoreline",
    "stamp_water",
]

#: Feature layers each style stamps. Roads, trails and markers are not built
#: yet; ``detailed`` currently renders the same as ``natural``.
STYLE_LAYERS: dict[str, tuple[str, ...]] = {
    "minimal": (),
    "natural": ("water",),
    "detailed": ("water",),
}

STYLES = tuple(STYLE_LAYERS)

#: Overpass mirrors, tried in order. The main instance rate-limits hard.
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

#: Server-side query budget, in seconds.
OVERPASS_TIMEOUT_S = 25

#: Client-side HTTP timeout. Slightly longer, so the server's own timeout wins.
REQUEST_TIMEOUT_S = 40.0

#: One retry, after this pause, on the rate-limit and gateway-timeout codes.
RETRY_STATUS = (429, 504)
RETRY_BACKOFF_S = 5.0

USER_AGENT = "terrframe/0.1.0 (+https://github.com/KyleRessLiere/Terraframe)"

#: Overpass selectors per layer. Values are (selector, geometry kind).
LAYER_QUERIES: dict[str, tuple[tuple[str, ...], str]] = {
    "water": (('way["natural"="water"]', 'way["waterway"="riverbank"]', 'relation["natural"="water"]'), "polygon"),
    "buildings": (('way["building"]',), "polygon"),
    "roads": (('way["highway"]',), "line"),
    "trails": (('way["highway"~"^(path|footway|track)$"]',), "line"),
}

#: How far, in pixels, the building mask is grown before removal. Roof edges
#: bleed into neighbouring pixels during resampling, so the footprint alone
#: leaves a rim of roof height behind.
BUILDING_DILATION_PX = 1

#: Depth, in millimetres, that stamped water is recessed below its bank.
#:
#: Deep enough to read as water in a single-colour print. The product cannot
#: depend on colour-change printing, so depth and geometry do all the work; a
#: shallow recess reads as a tone shift and disappears in one filament.
WATER_DEPTH_MM = 1.0

#: Width of the raised bead outlining every water body, in millimetres.
SHORELINE_WIDTH_MM = 0.5

#: Height of that bead above local terrain, in millimetres.
SHORELINE_HEIGHT_MM = 0.4

#: Horizontal budget for the bank wall, in millimetres. The drop from bead
#: crown to water plane must happen within this distance to read as a hard
#: step rather than a slope. It is one pixel by construction; this is the
#: assertion that the grid is fine enough for that pixel to be small enough.
MAX_BANK_STEP_MM = 0.3


class OverpassError(RuntimeError):
    """Raised when OSM features cannot be retrieved from Overpass."""


@dataclass(frozen=True)
class FeatureSet:
    """Geometry returned for one bbox, by layer name."""

    layers: dict[str, list[BaseGeometry]]

    def __getitem__(self, layer: str) -> list[BaseGeometry]:
        return self.layers.get(layer, [])

    def __bool__(self) -> bool:
        return any(self.layers.values())

    def counts(self) -> dict[str, int]:
        return {name: len(geoms) for name, geoms in self.layers.items()}


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------


def _cache_key(bbox: tuple[float, float, float, float], layers: tuple[str, ...]) -> str:
    payload = json.dumps({"bbox": [round(v, 6) for v in bbox], "layers": sorted(layers)})
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def build_query(bbox: tuple[float, float, float, float], layers: tuple[str, ...]) -> str:
    """Build one combined Overpass QL query covering every requested layer.

    A single query per call matters: Overpass rate-limits aggressively, and we
    re-run these constantly while iterating on styling.
    """
    south, west, north, east = bbox
    area = f"({south},{west},{north},{east})"

    clauses: list[str] = []
    for layer in layers:
        if layer not in LAYER_QUERIES:
            raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(LAYER_QUERIES)}")
        selectors, _ = LAYER_QUERIES[layer]
        clauses.extend(f"  {selector}{area};" for selector in selectors)

    body = "\n".join(clauses)
    return f"[out:json][timeout:{OVERPASS_TIMEOUT_S}];\n(\n{body}\n);\nout geom;"


def _post(url: str, query: str) -> requests.Response:
    return requests.post(
        url,
        data={"data": query},
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_S,
    )


def _request_overpass(query: str) -> dict:
    """POST a query, retrying once on rate-limit/gateway codes, then failing over."""
    last: str = "no endpoints configured"

    for url in OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                response = _post(url, query)
            except requests.RequestException as exc:
                last = f"{url}: {type(exc).__name__}: {exc}"
                break  # transport failure: move to the next mirror

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    last = f"{url}: response was not JSON: {exc}"
                    break

            last = f"{url}: HTTP {response.status_code}"
            if response.status_code in RETRY_STATUS and attempt == 0:
                time.sleep(RETRY_BACKOFF_S)
                continue
            break

    raise OverpassError(f"Overpass query failed ({last})")


def _relation_geometry(element: dict) -> BaseGeometry | None:
    """Assemble a multipolygon relation from its member ways.

    Large rivers are relations, not ways -- the Potomac's main channel is one --
    so skipping these silently drops exactly the water that makes a flat metro
    area legible. Outer rings are frequently split across several member ways,
    so the members are merged and polygonized rather than read individually.
    """
    outer: list[LineString] = []
    for member in element.get("members") or []:
        if member.get("role") not in {"outer", ""}:
            continue
        points = [(node["lon"], node["lat"]) for node in member.get("geometry") or []]
        if len(points) >= 2:
            outer.append(LineString(points))

    if not outer:
        return None

    try:
        merged = linemerge(outer)
        polygons = list(polygonize(merged if merged.geom_type == "MultiLineString" else [merged]))
    except Exception:  # pragma: no cover - shapely is tolerant, OSM is not
        return None

    if not polygons:
        return None

    geometry = unary_union(polygons)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry if not geometry.is_empty else None


def _element_geometry(element: dict) -> BaseGeometry | None:
    """Convert one Overpass element with ``out geom`` coordinates to shapely."""
    if element.get("type") == "relation":
        return _relation_geometry(element)

    points = [(node["lon"], node["lat"]) for node in element.get("geometry") or []]
    if len(points) < 2:
        return None

    closed = len(points) >= 4 and points[0] == points[-1]
    if closed:
        try:
            polygon = Polygon(points)
        except Exception:  # pragma: no cover - shapely guards most cases itself
            return None
        # Self-intersecting OSM ways are common; buffer(0) repairs them.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon if not polygon.is_empty else None

    return LineString(points)


def _parse_elements(payload: dict, layers: tuple[str, ...]) -> dict[str, list[BaseGeometry]]:
    """Sort Overpass elements back into the layers that asked for them."""
    result: dict[str, list[BaseGeometry]] = {layer: [] for layer in layers}

    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        geometry = _element_geometry(element)
        if geometry is None:
            continue

        highway = tags.get("highway")
        is_polygon = geometry.geom_type in {"Polygon", "MultiPolygon"}

        if "water" in result and is_polygon and (
            tags.get("natural") == "water" or tags.get("waterway") == "riverbank"
        ):
            result["water"].append(geometry)
        if "buildings" in result and is_polygon and "building" in tags:
            result["buildings"].append(geometry)
        if highway and not is_polygon:
            if "trails" in result and highway in {"path", "footway", "track"}:
                result["trails"].append(geometry)
            elif "roads" in result:
                # Trails are handled as their own layer, so roads excludes them.
                result["roads"].append(geometry)

    return result


def fetch_osm(
    bbox: tuple[float, float, float, float],
    layers: tuple[str, ...] | list[str] = ("water",),
    cache_dir: str | os.PathLike[str] = ".osm_cache",
) -> FeatureSet:
    """Fetch OSM geometry for a bbox, one combined Overpass query, disk-cached.

    Args:
        bbox: ``(south, west, north, east)`` in degrees.
        layers: Any of ``water``, ``buildings``, ``roads``, ``trails``.
        cache_dir: Directory holding cached Overpass responses.

    Returns:
        A :class:`FeatureSet` of shapely geometry in lon/lat.

    Raises:
        OverpassError: The query could not be completed. Callers that must not
            fail should use :func:`fetch_osm_or_warn`.
    """
    layers = tuple(layers)
    for layer in layers:
        if layer not in LAYER_QUERIES:
            raise ValueError(f"unknown layer {layer!r}; expected one of {sorted(LAYER_QUERIES)}")

    path = Path(cache_dir) / f"{_cache_key(tuple(bbox), layers)}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            path.unlink(missing_ok=True)  # corrupt entry: refetch
        else:
            return FeatureSet(_parse_elements(payload, layers))

    payload = _request_overpass(build_query(tuple(bbox), layers))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)

    return FeatureSet(_parse_elements(payload, layers))


def fetch_osm_or_warn(
    bbox: tuple[float, float, float, float],
    layers: tuple[str, ...] | list[str] = ("water",),
    cache_dir: str | os.PathLike[str] = ".osm_cache",
) -> FeatureSet:
    """Like :func:`fetch_osm`, but degrades to no features instead of failing.

    Overpass going down must never break a build: the result is a terrain-only
    model, which is what the tool produced before features existed.
    """
    try:
        return fetch_osm(bbox, layers, cache_dir)
    except OverpassError as exc:
        warnings.warn(f"{exc}; continuing with terrain only", RuntimeWarning, stacklevel=2)
        return FeatureSet({layer: [] for layer in tuple(layers)})


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------


def rasterize_mask(
    geoms: list[BaseGeometry],
    hm: Heightmap,
    width_px: float | None = None,
) -> np.ndarray:
    """Burn geometry onto the heightmap's grid as a boolean mask.

    Args:
        geoms: Shapely geometry in lon/lat.
        hm: Heightmap supplying the grid and its affine.
        width_px: When given, line geometry is buffered to this width in pixels
            before burning. Polygons ignore it.

    Returns:
        A boolean array shaped like ``hm.elevation``.
    """
    rows, cols = hm.elevation.shape
    if not geoms:
        return np.zeros((rows, cols), dtype=bool)

    transform = hm.transform
    if width_px:
        # Buffer in degrees, since the geometry and affine are both lon/lat.
        degrees = abs(transform.a) * width_px / 2.0
        geoms = [g.buffer(degrees) for g in geoms]

    burned = rasterize(
        ((geom, 1) for geom in geoms if not geom.is_empty),
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=True,
        dtype="uint8",
    )
    return burned.astype(bool)


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------


def stamp_water(
    hm: Heightmap,
    geoms: list[BaseGeometry],
    depth_mm: float = WATER_DEPTH_MM,
    mm_per_meter: float | None = None,
) -> np.ndarray:
    """Flatten each water polygon to its bank height, then engrave it.

    Every polygon is levelled independently to the minimum elevation found on
    its own edge, so a river that drops along its length stays plausible
    instead of being forced to one global level. The polygon boundary *is* the
    bank, so the mask is never blurred -- a gaussian here would soften exactly
    the crisp edge that makes flat cities legible.

    Engraving rather than flushing gives a paint line for colour-change prints
    and reads better than a surface that sits level with its bank.

    Args:
        hm: Heightmap to stamp. Its ``elevation`` is not modified in place.
        geoms: Water polygons in lon/lat.
        depth_mm: How far below bank level the water sits, in printed mm.
        mm_per_meter: Vertical print scale. Defaults to the value implied by
            ``hm``, so the recess is a constant depth on the finished model
            regardless of exaggeration.

    Returns:
        A new float32 elevation array.
    """
    elevation = np.array(hm.elevation, dtype=np.float32)
    if not geoms:
        return elevation

    scale = _mm_per_meter(hm) if mm_per_meter is None else mm_per_meter
    if scale <= 0.0:
        raise ValueError(f"mm_per_meter must be positive, got {scale}")
    depth_m = depth_mm / scale

    # Bank levels are read from the pristine terrain, never from the array
    # being written. Water polygons frequently touch, and sampling a neighbour
    # that has already been engraved makes each one step down again -- on a
    # city like DC that cascaded to hundreds of metres below sea level.
    original = np.array(hm.elevation, dtype=np.float32)

    for geom in geoms:
        mask = rasterize_mask([geom], hm)
        if not mask.any():
            continue

        level = _edge_minimum(original, mask)
        if level is None:
            continue
        elevation[mask] = np.float32(level - depth_m)

    return elevation


def stamp_shoreline(
    hm: Heightmap,
    water_geoms: list[BaseGeometry],
    width_mm: float = SHORELINE_WIDTH_MM,
    height_mm: float = SHORELINE_HEIGHT_MM,
    elevation: np.ndarray | None = None,
    mm_per_meter: float | None = None,
) -> np.ndarray:
    """Raise a rounded bead along every shoreline, on the terrain side.

    The cross-section, walking inward from dry land, is:
    terrain, a gaussian-rounded shoulder rising to the crown, then a hard drop
    to the flat water plane. The water side is never touched, so the water
    stays dead flat right up to its edge and the bank stays a wall.

    Rather than extracting ``geom.boundary`` per polygon, the bead is driven by
    a distance field from the union water mask. That gets three things for
    free: overlapping or near-touching shorelines produce one bead instead of
    stacking, interior rings are outlined because an island is simply terrain
    adjacent to water, and nothing can bleed across the boundary because the
    field is zero wherever water is.

    Args:
        hm: Heightmap supplying grid, scale and the water geometry's frame.
        water_geoms: The same polygons passed to :func:`stamp_water`.
        width_mm: How far the bead's shoulder reaches onto the terrain.
        height_mm: Crown height above local terrain. Zero disables the bead.
        elevation: Array to raise; defaults to ``hm.elevation``. Pass the
            already-recessed array so the bead lands after the water stamp.
        mm_per_meter: Vertical print scale; defaults to the value implied by
            ``hm``.

    Returns:
        A new float32 elevation array.
    """
    base = hm.elevation if elevation is None else elevation
    out = np.array(base, dtype=np.float32)
    if not water_geoms or height_mm <= 0.0:
        return out

    scale = _mm_per_meter(hm) if mm_per_meter is None else mm_per_meter
    if scale <= 0.0:
        raise ValueError(f"mm_per_meter must be positive, got {scale}")

    water = rasterize_mask(water_geoms, hm)
    if not water.any() or water.all():
        return out

    mm_per_px = _mm_per_pixel(hm)
    width_px = max(width_mm / mm_per_px, 1.0)

    # Distance in pixels to the nearest water pixel; zero inside water.
    distance = distance_transform_edt(~water)

    # Crown sits one pixel out from the water, then falls off outward. sigma is
    # half the requested width so the shoulder has faded to ~14% by then.
    sigma = max(width_px / 2.0, 0.5)
    shoulder = np.exp(-(((distance - 1.0) / sigma) ** 2) / 2.0)
    bead = np.where(distance >= 1.0, shoulder, 0.0)
    bead[water] = 0.0  # hard edge: nothing crosses onto the water plane

    height_m = height_mm / scale
    return (out + bead.astype(np.float32) * np.float32(height_m)).astype(np.float32, copy=False)


def _mm_per_pixel(hm: Heightmap, width_mm: float = 200.0) -> float:
    """Printed millimetres spanned by one pixel, matching the mesh's pitch."""
    cols = hm.elevation.shape[1]
    return width_mm / (cols - 1) if cols > 1 else width_mm


def _edge_minimum(elevation: np.ndarray, mask: np.ndarray) -> float | None:
    """Lowest elevation on the ring of pixels just outside a mask.

    Using the outside ring rather than the interior means the level is set by
    the bank the water actually meets, which is what keeps a stamped river
    sitting properly in its valley.
    """
    ring = binary_dilation(mask, iterations=1) & ~mask
    if ring.any():
        return float(np.nanmin(elevation[ring]))
    # A polygon covering the whole grid has no outside; fall back to its own.
    return float(np.nanmin(elevation[mask])) if mask.any() else None


def _mm_per_meter(hm: Heightmap, width_mm: float = 200.0) -> float:
    """Printed millimetres per ground metre, matching mesh.heightmap_to_mesh."""
    rows, cols = hm.elevation.shape
    ground_width_m = hm.meters_per_px * (cols - 1)
    return width_mm / ground_width_m if ground_width_m > 0 else 0.0


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def should_remove_buildings(
    setting: str | bool,
    geoms: list[BaseGeometry],
    is_bare_earth: bool = False,
) -> bool:
    """Decide whether building removal runs.

    ``"auto"`` turns removal on only when there is something to remove and the
    source is not already bare earth. Wilderness returns no geometry, and a
    LiDAR bare-earth DTM has no buildings in it to begin with, so in both cases
    the work is skipped.

    Args:
        setting: ``"auto"``, or a bool / ``"on"`` / ``"off"`` to force it.
        geoms: Building polygons returned for the bbox.
        is_bare_earth: Whether the elevation source is already bare earth.

    Returns:
        Whether to run :func:`remove_buildings`.
    """
    if isinstance(setting, bool):
        return setting
    lowered = str(setting).strip().lower()
    if lowered in {"on", "true", "yes"}:
        return True
    if lowered in {"off", "false", "no", "none"}:
        return False
    if lowered != "auto":
        raise ValueError(f"--remove-buildings takes on, off or auto; got {setting!r}")

    return bool(geoms) and not is_bare_earth


def remove_buildings(
    hm: Heightmap,
    geoms: list[BaseGeometry],
    dilation_px: int = BUILDING_DILATION_PX,
) -> np.ndarray:
    """Cut building footprints out of the terrain and heal the holes.

    Footprints are grown by ``dilation_px`` before removal, because resampling
    bleeds roof height into neighbouring pixels and the bare footprint leaves a
    rim behind. The holes become NaN and are healed by the existing
    :func:`~terrframe.heightmap.fill_nodata`, so filled ground is always a
    convex combination of real surrounding samples.

    This runs before despike/smooth/exaggerate, so ``auto_exaggeration`` sizes
    the print off cleaned ground rather than off a rooftop.

    Args:
        hm: Heightmap to clean.
        geoms: Building polygons in lon/lat.
        dilation_px: Pixels to grow each footprint by.

    Returns:
        A new float32 elevation array with buildings replaced by ground.
    """
    elevation = np.array(hm.elevation, dtype=np.float32)
    if not geoms:
        return elevation

    mask = rasterize_mask(geoms, hm)
    if not mask.any():
        return elevation
    if dilation_px > 0:
        mask = binary_dilation(mask, iterations=dilation_px)

    if mask.all():
        warnings.warn(
            "building footprints cover the whole bbox; skipping removal",
            RuntimeWarning,
            stacklevel=2,
        )
        return elevation

    elevation[mask] = np.nan
    return fill_nodata(elevation)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def layers_for_style(style: str, remove_buildings_setting: str | bool = "auto") -> tuple[str, ...]:
    """Layers that must be fetched for a style, including buildings if needed."""
    if style not in STYLE_LAYERS:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")

    layers = list(STYLE_LAYERS[style])
    # "auto" still needs the query, since the decision depends on whether any
    # building geometry comes back at all.
    if remove_buildings_setting is not False and str(remove_buildings_setting).lower() != "off":
        layers.append("buildings")
    return tuple(dict.fromkeys(layers))


def building_remover(geoms: list[BaseGeometry]):
    """A ``pre_clean`` hook for :func:`~terrframe.heightmap.build_heightmap`."""

    def _hook(hm: Heightmap) -> np.ndarray:
        return remove_buildings(hm, geoms)

    return _hook


def apply_features(
    hm: Heightmap,
    style: str,
    feature_set: FeatureSet,
    water_depth_mm: float = WATER_DEPTH_MM,
    shoreline_mm: float = SHORELINE_HEIGHT_MM,
    shoreline_width_mm: float = SHORELINE_WIDTH_MM,
) -> Heightmap:
    """Stamp a style's features onto an already-exaggerated heightmap.

    Z-order is fixed: terrain is finished first (despike, smooth, exaggerate),
    then water, then linework, then markers. Stamps are computed in printed
    millimetres, so a bead or a recess is the same physical size no matter what
    exaggeration the terrain got.

    Args:
        hm: Finished, exaggerated heightmap.
        style: One of :data:`STYLES`.
        feature_set: Geometry already fetched for this bbox.
        water_depth_mm: Engrave depth for water.

    Returns:
        A new :class:`Heightmap`; the input is not modified.
    """
    if style not in STYLE_LAYERS:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")

    elevation = np.array(hm.elevation, dtype=np.float32)
    layers = STYLE_LAYERS[style]
    water_mask = hm.water_mask

    if "water" in layers and feature_set["water"]:
        elevation = stamp_water(hm, feature_set["water"], depth_mm=water_depth_mm)
        # Bead goes on after the recess, so it rides the finished bank.
        elevation = stamp_shoreline(
            hm,
            feature_set["water"],
            width_mm=shoreline_width_mm,
            height_mm=shoreline_mm,
            elevation=elevation,
        )
        water_mask = rasterize_mask(feature_set["water"], hm)

    # Linework and markers land here, in that order, once implemented. Roads
    # will ride over the bead through the same max-semantics.

    return Heightmap(
        elevation=elevation,
        meters_per_px=hm.meters_per_px,
        bbox=hm.bbox,
        zoom=hm.zoom,
        exaggeration=hm.exaggeration,
        requested_bbox=hm.requested_bbox,
        water_mask=water_mask,
    )
