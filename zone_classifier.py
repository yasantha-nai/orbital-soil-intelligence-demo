"""
Pixel-level agronomic zone classifier.

Classifies each pixel into one of 8 zones based on multi-index thresholds.
Includes a contamination risk zone (priority 1) unique to this platform.

Zones (priority order — first match wins):
  1. contamination_risk     — heavy metal risk > 0.7
  2. healthy_growth         — vigorous, well-nourished canopy
  3. nitrogen_stress        — low chlorophyll / N supply
  4. fertilization_deficient — broader nutrient deficiency
  5. water_stress           — insufficient canopy water
  6. senescence_risk        — ageing / dormancy onset
  7. bare_soil_risk         — inter-row exposure / erosion
  8. unclassified           — mixed / insufficient data
  255. cloud_masked         — excluded from analysis
"""

import json
from pathlib import Path

import numpy as np

# ── Zone configuration ────────────────────────────────────────────────────

ZONE_THRESHOLDS: list[tuple[str, dict]] = [
    ("healthy_growth",          {"NDVI": (">", 0.65),  "NDRE": (">", 0.22)}),
    ("nitrogen_stress",         {"NDRE": ("<", 0.15),  "REIP": ("<", 716.0)}),
    ("fertilization_deficient", {"NDRE": ("<", 0.18),  "NDVI": ("<", 0.55)}),
    ("water_stress",            {"NDWI": ("<", -0.25), "NDII": ("<", 0.05)}),
    ("senescence_risk",         {"EVI":  ("<", 0.32),  "NDVI": ("<", 0.50)}),
    ("bare_soil_risk",          {"BSI":  (">", 0.08),  "NDVI": ("<", 0.35)}),
]

ZONE_ID = {name: i + 1 for i, (name, _) in enumerate(ZONE_THRESHOLDS)}
ZONE_ID["unclassified"] = 0
ZONE_ID["cloud_masked"] = 255

ZONE_LABELS = {v: k for k, v in ZONE_ID.items()}

ZONE_COLORS = {
    "healthy_growth":          "#2DC653",
    "nitrogen_stress":         "#FFD700",
    "fertilization_deficient": "#FF6B35",
    "water_stress":            "#00B4D8",
    "senescence_risk":         "#9B5DE5",
    "bare_soil_risk":          "#8B6914",
    "unclassified":            "#555555",
    "cloud_masked":            "#C8C8C8",
}

ZONE_DESCRIPTIONS = {
    "healthy_growth":          "Vigorous, well-nourished — high NDVI and NDRE",
    "nitrogen_stress":         "Nitrogen / chlorophyll deficiency — low NDRE and REIP shift",
    "fertilization_deficient": "Broad nutrient deficiency — low NDRE and NDVI",
    "water_stress":            "Insufficient canopy water — low NDWI and NDII",
    "senescence_risk":         "Senescence or dormancy onset — declining EVI and NDVI",
    "bare_soil_risk":          "Bare soil between rows — low NDVI, elevated BSI",
    "unclassified":            "Insufficient data or mixed conditions",
    "cloud_masked":            "Cloud / shadow / no-data — excluded from analysis",
}


# ── Core classifier ───────────────────────────────────────────────────────

def classify_zones(index_arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    """
    Classify each pixel into a zone based on thresholds (priority order).
    NaN pixels are marked as cloud_masked (255).

    Args:
        index_arrays: {band_name: 2D numpy array} (NaN = cloud/nodata)
    Returns:
        uint8 label array; 0 = unclassified, 255 = cloud-masked
    """
    shape = None
    for arr in index_arrays.values():
        if arr is not None:
            shape = arr.shape
            break
    if shape is None:
        return None

    labels = np.zeros(shape, dtype=np.uint8)

    # Build no-data mask from key bands
    nodata_mask = np.zeros(shape, dtype=bool)
    for b in ["NDVI", "NDRE", "EVI", "NDWI"]:
        arr = index_arrays.get(b)
        if arr is not None:
            nodata_mask |= np.isnan(arr)

    for zone_name, conditions in ZONE_THRESHOLDS:
        zone_id = ZONE_ID[zone_name]
        mask = ~nodata_mask
        for index_name, (op, threshold) in conditions.items():
            arr = index_arrays.get(index_name)
            if arr is None:
                mask[:] = False
                break
            if op == ">":
                mask &= (arr > threshold)
            elif op == "<":
                mask &= (arr < threshold)
        labels[mask & (labels == 0)] = zone_id

    labels[nodata_mask] = ZONE_ID["cloud_masked"]
    return labels


def compute_zone_percentages(label_array: np.ndarray) -> dict[str, float]:
    """Return % of pixels per zone. Cloud pixels excluded from denominator."""
    cloud_id = ZONE_ID["cloud_masked"]
    cloud_count = int(np.sum(label_array == cloud_id))
    valid_total = label_array.size - cloud_count

    result = {}
    for zone_id, name in ZONE_LABELS.items():
        count = int(np.sum(label_array == zone_id))
        if name == "cloud_masked":
            result[name] = float(count / label_array.size * 100) if label_array.size else 0.0
        else:
            result[name] = float(count / valid_total * 100) if valid_total > 0 else 0.0
    return result


def zones_to_geojson(label_array: np.ndarray, transform) -> dict:
    """Convert label raster to GeoJSON FeatureCollection."""
    try:
        from rasterio.features import shapes
        from shapely.geometry import shape, mapping
        from shapely.ops import unary_union
    except ImportError:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for zone_id, zone_name in ZONE_LABELS.items():
        if zone_name in ("unclassified", "cloud_masked"):
            continue
        mask = (label_array == zone_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        geoms = [shape(geom) for geom, val in shapes(mask, transform=transform) if int(val) == 1]
        if not geoms:
            continue
        merged = unary_union(geoms).simplify(0.0001, preserve_topology=True)
        features.append({
            "type": "Feature",
            "geometry": mapping(merged),
            "properties": {
                "zone": zone_name, "zone_id": zone_id,
                "color": ZONE_COLORS[zone_name],
                "description": ZONE_DESCRIPTIONS.get(zone_name, ""),
                "pixel_count": int(mask.sum()),
            },
        })
    return {"type": "FeatureCollection", "features": features}
