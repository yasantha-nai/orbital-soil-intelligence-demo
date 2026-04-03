"""
GEE tile URL helpers for map visualisation.

Generates tile URLs for Sentinel-2, Sentinel-1, DEM overlays and
heavy metal risk surfaces, clipped to the area of interest.
"""

import json

import ee

from config import DEM_BAND, DEM_COLLECTION, DEM_VIS, MAX_CLOUD_COVER

# ── Layer definitions ─────────────────────────────────────────────────────────

S2_LAYERS = {
    "S2: True Color (RGB)": {
        "fn": lambda img: img,
        "vis": {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
    },
    "S2: False Color (NIR)": {
        "fn": lambda img: img,
        "vis": {"bands": ["B8", "B4", "B3"], "min": 0, "max": 5000},
    },
    "S2: NDVI": {
        "fn": lambda img: img.normalizedDifference(["B8", "B4"]).rename("NDVI"),
        "vis": {"min": 0, "max": 0.9, "palette": ["#d73027", "#fee08b", "#1a9850"]},
    },
    "S2: NDRE": {
        "fn": lambda img: img.normalizedDifference(["B8", "B5"]).rename("NDRE"),
        "vis": {"min": 0, "max": 0.9, "palette": ["#762a83", "#fee08b", "#1b7837"]},
    },
    "S2: EVI": {
        "fn": lambda img: img.expression(
            "2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)",
            {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
        ),
        "vis": {"min": 0, "max": 1, "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
    },
}

S1_LAYERS = {
    "S1: VH Backscatter": {
        "fn": lambda img: img.select("VH"),
        "vis": {"min": -25, "max": 0, "palette": ["#000000", "#ffffff"]},
    },
    "S1: VV Backscatter": {
        "fn": lambda img: img.select("VV"),
        "vis": {"min": -15, "max": 5, "palette": ["#000000", "#ffffff"]},
    },
    "S1: RVI (Vegetation)": {
        "fn": lambda img: img.select("VH").multiply(4)
            .divide(img.select("VV").add(img.select("VH"))).rename("RVI"),
        "vis": {"min": 0, "max": 2, "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
    },
}

DEM_LAYERS = {
    "DEM: Elevation": {"type": "dem", "vis": DEM_VIS},
    "DEM: Slope": {
        "type": "dem_slope",
        "vis": {"min": 0, "max": 60, "palette": ["#ffffcc", "#fd8d3c", "#800026"]},
    },
    "DEM: Hillshade": {
        "type": "dem_hillshade",
        "vis": {"min": 0, "max": 255, "palette": ["#000000", "#ffffff"]},
    },
}

ALL_OVERLAY_LAYERS = {**S2_LAYERS, **S1_LAYERS}


# ── Composite builders ────────────────────────────────────────────────────────

def _build_s2_composite(start_date: str, end_date: str, geometry: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .median()
    )


def _build_s1_composite(start_date: str, end_date: str, geometry: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .mean()
    )


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_overlay_tiles(
    layer_names: list[str],
    start_date: str,
    end_date: str,
    coords_json: str,
    palette_overrides: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """
    Fetch GEE tile URLs for the given layers, clipped to the boundary.

    Args:
        layer_names       : keys from ALL_OVERLAY_LAYERS
        start_date        : 'YYYY-MM-DD'
        end_date          : 'YYYY-MM-DD'
        coords_json       : JSON string of [[lon, lat], ...] boundary
        palette_overrides : optional {layer_name: [hex_low, ..., hex_high]}

    Returns:
        {layer_name: tile_url}
    """
    coords = json.loads(coords_json)
    geometry = ee.Geometry.Polygon([coords])

    s2_names = [n for n in layer_names if n.startswith("S2")]
    s1_names = [n for n in layer_names if n.startswith("S1")]

    s2_composite = _build_s2_composite(start_date, end_date, geometry) if s2_names else None
    s1_composite = _build_s1_composite(start_date, end_date, geometry) if s1_names else None

    results = {}
    for name in layer_names:
        cfg = ALL_OVERLAY_LAYERS[name]
        base = s2_composite if name.startswith("S2") else s1_composite
        img = cfg["fn"](base).clip(geometry)
        vis = dict(cfg["vis"])
        if palette_overrides and name in palette_overrides and "palette" in vis:
            vis["palette"] = palette_overrides[name]
        map_id = img.getMapId(vis)
        results[name] = map_id["tile_fetcher"].url_format
    return results


def fetch_dem_tiles(coords_json: str) -> dict[str, str]:
    """
    Fetch GEE tile URLs for DEM-derived layers (elevation, slope, hillshade).

    Returns:
        {layer_name: tile_url}
    """
    coords = json.loads(coords_json)
    geometry = ee.Geometry.Polygon([coords])

    dem = (
        ee.ImageCollection(DEM_COLLECTION)
        .select(DEM_BAND)
        .mosaic()
        .clip(geometry)
    )

    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").clip(geometry)
    hillshade = ee.Terrain.hillshade(dem).clip(geometry)

    results = {}
    for name, cfg in DEM_LAYERS.items():
        t = cfg["type"]
        if t == "dem":
            img = dem
        elif t == "dem_slope":
            img = slope
        else:
            img = hillshade
        map_id = img.getMapId(cfg["vis"])
        results[name] = map_id["tile_fetcher"].url_format
    return results


def fetch_single_date_tile(
    date_str: str,
    vis_type: str,
    coords_json: str,
    window_days: int = 15,
) -> str | None:
    """
    Fetch a tile URL for the best S2 image near a specific date.

    Returns tile URL string or None if no image found.
    """
    coords = json.loads(coords_json)
    geometry = ee.Geometry.Polygon([coords])

    date = ee.Date(date_str)
    end = date.advance(window_days, "day")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(date, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()
    if count == 0:
        return None

    img = collection.first()

    vis_map = {
        "True Color": (img, {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}),
        "NDVI": (
            img.normalizedDifference(["B8", "B4"]),
            {"min": 0, "max": 0.9, "palette": ["#d73027", "#fee08b", "#1a9850"]},
        ),
        "NDRE": (
            img.normalizedDifference(["B8", "B5"]),
            {"min": 0, "max": 0.9, "palette": ["#762a83", "#fee08b", "#1b7837"]},
        ),
        "EVI": (
            img.expression(
                "2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)",
                {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
            ),
            {"min": 0, "max": 1, "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
        ),
    }

    vis_img, vis_params = vis_map.get(vis_type, vis_map["NDVI"])
    vis_img = vis_img.clip(geometry)

    try:
        map_id = vis_img.getMapId(vis_params)
        return map_id["tile_fetcher"].url_format
    except Exception:
        return None
