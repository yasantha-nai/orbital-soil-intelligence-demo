"""
Session persistence for Orbital Soil Intelligence Platform.

Saves and loads session metadata (coordinates, date range, crop type,
collected data flags) so users do not have to re-run data collection
after restarting the app.

Directory structure:
    data/sessions/<label>/session.json   — metadata
    data/<label>_*.csv                   — collected dataframes
    data/sessions/<label>/dem/           — cached DEM GeoTIFFs
    data/sessions/<label>/rasters/       — per-date index GeoTIFFs
    data/sessions/<label>/zones/         — zone classification outputs
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

SESSIONS_DIR = Path("data/sessions")


def list_sessions() -> list[dict]:
    """Return saved sessions sorted by creation time (newest first)."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions
    for session_dir in sorted(SESSIONS_DIR.iterdir()):
        json_file = session_dir / "session.json"
        if json_file.exists():
            try:
                with open(json_file) as f:
                    meta = json.load(f)
                meta["has_dem"] = (session_dir / "dem" / "dem.tif").exists()
                meta["has_rasters"] = (
                    any((session_dir / "rasters").glob("*.tif"))
                    if (session_dir / "rasters").exists()
                    else False
                )
                meta["has_zones"] = (
                    any((session_dir / "zones").glob("*.tif"))
                    if (session_dir / "zones").exists()
                    else False
                )
                sessions.append(meta)
            except Exception:
                pass
    return sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)


def save_session(
    label: str,
    coordinates: list,
    start_date,
    end_date,
    results: dict,
    crop_type: str = "Corn/Maize",
) -> dict:
    """
    Persist session metadata. Returns the metadata dict that was saved.
    """
    session_dir = SESSIONS_DIR / label
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "label": label,
        "coordinates": coordinates,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "crop_type": crop_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "has_s2": bool(results.get("df_s2") is not None and not results["df_s2"].empty),
        "has_s1": bool(results.get("df_s1") is not None and not results["df_s1"].empty),
        "has_weather": bool(results.get("df_weather") is not None and not results["df_weather"].empty),
        "has_embeddings": "df_embeddings" in results,
        "has_modis": bool(results.get("df_modis") is not None and not results["df_modis"].empty),
        "has_planet": bool(results.get("df_planet") is not None and not results["df_planet"].empty),
        "has_merged": bool(results.get("df_merged") is not None and not results["df_merged"].empty),
        "has_heavy_metal": bool(results.get("df_heavy_metal") is not None),
    }

    with open(session_dir / "session.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_session(label: str) -> tuple[dict, dict] | tuple[None, None]:
    """
    Load a session by label.

    Returns:
        (metadata_dict, results_dict)  or  (None, None) if not found.
    """
    json_file = SESSIONS_DIR / label / "session.json"
    if not json_file.exists():
        return None, None

    with open(json_file) as f:
        meta = json.load(f)

    results = {"land_label": label}

    csv_map = {
        "df_s2":          f"data/sentinel2_{label}.csv",
        "df_s1":          f"data/sentinel1_{label}.csv",
        "df_weather":     f"data/weather_{label}.csv",
        "df_embeddings":  f"data/embeddings_{label}.csv",
        "df_modis":       f"data/modis_{label}.csv",
        "df_planet":      f"data/planetscope_{label}.csv",
        "df_merged":      f"data/merged_{label}.csv",
        "df_heavy_metal": f"data/heavymetal_{label}.csv",
    }

    for key, path in csv_map.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                results[key] = df
            except Exception:
                pass

    return meta, results


def get_session_dir(label: str) -> Path:
    """Return session directory Path (creates if absent)."""
    d = SESSIONS_DIR / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def raster_cache_dir(label: str) -> Path:
    d = SESSIONS_DIR / label / "rasters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dem_cache_dir(label: str) -> Path:
    d = SESSIONS_DIR / label / "dem"
    d.mkdir(parents=True, exist_ok=True)
    return d


def zone_cache_dir(label: str) -> Path:
    d = SESSIONS_DIR / label / "zones"
    d.mkdir(parents=True, exist_ok=True)
    return d
