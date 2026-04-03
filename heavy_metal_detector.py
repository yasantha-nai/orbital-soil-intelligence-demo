"""
Heavy metal contamination risk engine.

Computes proxy-based risk surfaces and time-series from Sentinel-2 imagery
using weighted vegetation/soil indices and metal-specific multipliers.

Detection modes:
  - Bimodal (recommended): combines vegetation stress + bare-soil signatures
  - Direct (bare soil): host-phase signatures on exposed soil
  - Indirect (canopy stress): crop stress as bio-sensors (red edge shift, etc.)
"""

import ee
import pandas as pd

from config import (
    MAX_CLOUD_COVER,
    METAL_MULTIPLIERS,
    PROXY_WEIGHTS,
    S2_SCALE,
    VIS_PALETTES,
)


def _compute_indices(img: ee.Image) -> dict[str, ee.Image]:
    """Compute core spectral indices from a Sentinel-2 image."""
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = img.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    bsi = img.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": img.select("B11"),
            "RED": img.select("B4"),
            "NIR": img.select("B8"),
            "BLUE": img.select("B2"),
        },
    ).rename("BSI")
    return {"NDVI": ndvi, "NDRE": ndre, "NDMI": ndmi, "BSI": bsi}


def _compute_risk(indices: dict[str, ee.Image], metal: str) -> ee.Image:
    """Compute normalised risk score from indices and metal multiplier."""
    base_risk = (
        indices["NDVI"].multiply(PROXY_WEIGHTS["NDVI"])
        .add(indices["NDRE"].multiply(PROXY_WEIGHTS["NDRE"]))
        .add(indices["NDMI"].multiply(PROXY_WEIGHTS["NDMI"]))
        .add(indices["BSI"].multiply(PROXY_WEIGHTS["BSI"]))
    )
    multiplier = METAL_MULTIPLIERS.get(metal, 1.0)
    return base_risk.multiply(multiplier).unitScale(-1, 1).clamp(0, 1).rename("risk")


def build_proxy_layers(
    lat: float,
    lon: float,
    radius_km: float,
    start_date: str,
    end_date: str,
    metal: str,
) -> dict:
    """
    Build proxy risk layers from Sentinel-2 composite.

    Returns dict with geometry info and tile URLs for map visualisation.
    """
    geom = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .median()
        .clip(geom)
    )

    indices = _compute_indices(s2)
    risk = _compute_risk(indices, metal)

    return {
        "geom": geom.getInfo(),
        "risk_tile": risk.getMapId(VIS_PALETTES["risk"])["tile_fetcher"].url_format,
        "ndvi_tile": indices["NDVI"].getMapId(VIS_PALETTES["ndvi"])["tile_fetcher"].url_format,
        "ndmi_tile": indices["NDMI"].getMapId(VIS_PALETTES["ndmi"])["tile_fetcher"].url_format,
        "bsi_tile":  indices["BSI"].getMapId(VIS_PALETTES["bsi"])["tile_fetcher"].url_format,
    }


def compute_timeseries(
    lat: float,
    lon: float,
    radius_km: float,
    start_date: str,
    end_date: str,
    metal: str,
) -> pd.DataFrame:
    """
    Extract per-image time-series of indices and risk score.

    Returns a DataFrame with columns: date, NDVI, NDRE, NDMI, BSI, Risk.
    """
    geom = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER + 5))
    )

    multiplier = METAL_MULTIPLIERS.get(metal, 1.0)

    def add_indices(img):
        idx = _compute_indices(img)
        risk = (
            idx["NDVI"].multiply(PROXY_WEIGHTS["NDVI"])
            .add(idx["NDRE"].multiply(PROXY_WEIGHTS["NDRE"]))
            .add(idx["NDMI"].multiply(PROXY_WEIGHTS["NDMI"]))
            .add(idx["BSI"].multiply(PROXY_WEIGHTS["BSI"]))
            .multiply(multiplier)
            .unitScale(-1, 1)
            .clamp(0, 1)
            .rename("Risk")
        )
        return img.addBands([idx["NDVI"], idx["NDRE"], idx["NDMI"], idx["BSI"], risk])

    coll = coll.map(add_indices)

    def summarize(img):
        stats = img.select(["NDVI", "NDRE", "NDMI", "BSI", "Risk"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=S2_SCALE,
            maxPixels=1e7,
        )
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM-dd"))

    feats = coll.map(summarize).filter(ee.Filter.notNull(["Risk"]))
    rows = ee.FeatureCollection(feats).getInfo().get("features", [])

    data = []
    for r in rows:
        p = r.get("properties", {})
        data.append({
            "date": p.get("date"),
            "NDVI": p.get("NDVI"),
            "NDRE": p.get("NDRE"),
            "NDMI": p.get("NDMI"),
            "BSI":  p.get("BSI"),
            "Risk": p.get("Risk"),
        })

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df
