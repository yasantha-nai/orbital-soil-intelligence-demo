"""
MODIS Terra Land Surface Temperature (LST) and Crop Water Stress Index (CWSI).

Dataset: MODIS/061/MOD11A1 — daily, 1 km resolution.
CWSI = (LST_day − T_air + 2) / 10, clipped to [0, 1].
"""

import ee
import pandas as pd


MODIS_LST_COLLECTION = "MODIS/061/MOD11A1"
MODIS_LST_SCALE = 1000

_LST_SCALE_FACTOR = 0.02
_KELVIN_OFFSET = 273.15
_CWSI_DT_WET = -2.0
_CWSI_DT_DRY = 8.0
_CWSI_RANGE = _CWSI_DT_DRY - _CWSI_DT_WET  # 10°C


def collect_modis_lst(
    coordinates: list[list[float]],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Collect daily MODIS Terra LST for a polygon.

    Returns DataFrame with: date, lst_day_c, lst_night_c, lst_diurnal_delta.
    """
    geometry = ee.Geometry.Polygon([coordinates])

    collection = (
        ee.ImageCollection(MODIS_LST_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .select(["LST_Day_1km", "LST_Night_1km", "QC_Day"])
    )

    def _extract_daily(image):
        date_str = image.date().format("YYYY-MM-dd")
        qc = image.select("QC_Day")
        good = qc.bitwiseAnd(3).eq(0)
        masked = image.updateMask(good)

        vals = masked.select(["LST_Day_1km", "LST_Night_1km"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geometry,
            scale=MODIS_LST_SCALE, maxPixels=1e6,
        )
        return ee.Feature(None, {
            "date": date_str,
            "lst_day_dn": vals.get("LST_Day_1km"),
            "lst_night_dn": vals.get("LST_Night_1km"),
        })

    features = collection.map(_extract_daily)
    rows = ee.FeatureCollection(features).getInfo()["features"]

    if not rows:
        return pd.DataFrame(columns=["date", "lst_day_c", "lst_night_c", "lst_diurnal_delta"])

    records = []
    for feat in rows:
        p = feat["properties"]
        day_dn = p.get("lst_day_dn")
        night_dn = p.get("lst_night_dn")
        day_c = (day_dn * _LST_SCALE_FACTOR - _KELVIN_OFFSET) if day_dn else None
        night_c = (night_dn * _LST_SCALE_FACTOR - _KELVIN_OFFSET) if night_dn else None
        records.append({
            "date": p.get("date"),
            "lst_day_c": round(day_c, 2) if day_c is not None else None,
            "lst_night_c": round(night_c, 2) if night_c is not None else None,
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["lst_diurnal_delta"] = (df["lst_day_c"] - df["lst_night_c"]).round(2)
    return df


def add_cwsi(df_modis: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """Add CWSI column by merging MODIS LST with NASA POWER air temperature."""
    out = df_modis.copy()

    wx = df_weather.copy()
    if "date" not in wx.columns:
        wx = wx.reset_index().rename(columns={"index": "date"})
    wx["date"] = pd.to_datetime(wx["date"])
    wx = wx[["date", "temp_avg_c"]].dropna()

    out = out.merge(wx, on="date", how="left")
    out["cwsi"] = (
        (out["lst_day_c"] - out["temp_avg_c"] - _CWSI_DT_WET) / _CWSI_RANGE
    ).clip(0, 1).round(3)
    out = out.drop(columns=["temp_avg_c"], errors="ignore")
    return out
