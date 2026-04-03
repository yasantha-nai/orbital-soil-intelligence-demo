"""
Collects historical weather data using NASA POWER API.
No API key required. Data is specific to agriculture (community=AG).
"""

import requests
import pandas as pd

from config import POWER_COMMUNITY, POWER_VARIABLES

POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def get_centroid(coordinates: list[list[float]]) -> tuple[float, float]:
    """Compute centroid (lat, lon) of a polygon from [[lon, lat], ...] pairs."""
    lons = [c[0] for c in coordinates]
    lats = [c[1] for c in coordinates]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def collect_weather(
    coordinates: list[list[float]],
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch daily weather data from NASA POWER for the centroid of the polygon.

    Returns DataFrame indexed by date with weather columns.
    """
    if variables is None:
        variables = POWER_VARIABLES

    lat, lon = get_centroid(coordinates)

    params = {
        "parameters": ",".join(variables),
        "community": POWER_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON",
    }

    response = requests.get(POWER_BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    data = response.json()
    daily = data["properties"]["parameter"]

    df = pd.DataFrame(daily)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.replace(-999.0, float("nan")).sort_index()

    rename_map = {
        "T2M": "temp_avg_c",
        "T2M_MAX": "temp_max_c",
        "T2M_MIN": "temp_min_c",
        "PRECTOTCORR": "rainfall_mm",
        "RH2M": "humidity_pct",
        "WS2M": "wind_speed_ms",
        "ALLSKY_SFC_SW_DWN": "solar_rad_mj",
        "ALLSKY_SFC_LW_DWN": "longwave_rad_wm2",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df
