"""
Copernicus GLO-30 DEM downloader and terrain analysis.

Downloads DEM via GEE, computes terrain derivatives (slope, aspect, hillshade,
flow direction) using scipy, and derives erosion risk via simplified RUSLE.

Cache layout: data/sessions/<label>/dem/{dem,slope,aspect,hillshade,erosion}.tif
"""

import io
import json
import zipfile
from pathlib import Path

import ee
import numpy as np
import requests
from scipy import ndimage

from config import DEM_BAND, DEM_COLLECTION, DEM_SCALE

# Optional rasterio import — only needed for GeoTIFF I/O
try:
    import rasterio
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_tif(path: Path, array: np.ndarray, transform, crs: str) -> None:
    if not HAS_RASTERIO:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32",
        width=array.shape[1], height=array.shape[0],
        count=1, crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(array.astype(np.float32), 1)


def _load_tif(path: Path):
    if not HAS_RASTERIO:
        return None, None, None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform, str(src.crs)


def _decode_gee_response(content: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if tifs:
                return z.read(tifs[0])
    except zipfile.BadZipFile:
        pass
    return content


# ── Public API ────────────────────────────────────────────────────────────────

def download_dem(coords_json: str, cache_dir: Path | None = None, scale: int = DEM_SCALE):
    """
    Download Copernicus GLO-30 DEM for the AOI.

    Returns (elevation_array, affine_transform, crs_string).
    """
    if cache_dir:
        cache_path = Path(cache_dir) / "dem.tif"
        if cache_path.exists() and HAS_RASTERIO:
            return _load_tif(cache_path)

    coords = json.loads(coords_json)
    geometry = ee.Geometry.Polygon([coords])

    dem_img = (
        ee.ImageCollection(DEM_COLLECTION)
        .select(DEM_BAND)
        .mosaic()
        .clip(geometry)
    )

    url = dem_img.getDownloadURL({
        "scale": scale, "crs": "EPSG:4326",
        "format": "GEO_TIFF", "region": geometry,
    })

    response = requests.get(url, timeout=180)
    response.raise_for_status()
    tif_bytes = _decode_gee_response(response.content)

    if HAS_RASTERIO:
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            array = src.read(1).astype(np.float32)
            transform = src.transform
            crs = str(src.crs)
    else:
        # Fallback: return raw bytes info
        return None, None, None

    array[array < -9000] = np.nan

    if cache_dir:
        _save_tif(Path(cache_dir) / "dem.tif", array, transform, crs)

    return array, transform, crs


def compute_terrain_derivatives(dem_array: np.ndarray, transform, cache_dir: Path | None = None):
    """Derive slope (°), aspect (°), hillshade, and flow direction vectors."""
    px_deg = abs(transform.a)
    px_m = px_deg * 111_320

    dem = np.where(np.isnan(dem_array), np.nanmedian(dem_array), dem_array)
    gy, gx = np.gradient(dem, px_m, px_m)

    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2))).astype(np.float32)

    aspect = np.degrees(np.arctan2(gy, -gx)).astype(np.float32)
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    sun_az = np.radians(315)
    sun_el = np.radians(45)
    hillshade = 255 * (
        np.cos(sun_el) * np.cos(np.radians(slope))
        + np.sin(sun_el) * np.sin(np.radians(slope))
        * np.cos(sun_az - np.radians(aspect))
    )
    hillshade = np.clip(hillshade, 0, 255).astype(np.float32)

    mag = np.sqrt(gx**2 + gy**2) + 1e-10
    flow_x = (-gx / mag).astype(np.float32)
    flow_y = (-gy / mag).astype(np.float32)

    result = {
        "slope": slope, "aspect": aspect, "hillshade": hillshade,
        "flow_x": flow_x, "flow_y": flow_y,
    }

    if cache_dir and HAS_RASTERIO:
        crs = "EPSG:4326"
        for name in ["slope", "aspect", "hillshade"]:
            _save_tif(Path(cache_dir) / f"{name}.tif", result[name], transform, crs)

    return result


def compute_erosion_risk(slope_array: np.ndarray, ndvi_array: np.ndarray,
                         cache_dir: Path | None = None, transform=None):
    """
    Simplified RUSLE erosion risk = LS-factor × C-factor.
    LS = (slope/45)^1.4, C = (1 − NDVI). Returns 0–1 normalised risk.
    """
    if ndvi_array.shape != slope_array.shape:
        zoom_y = slope_array.shape[0] / ndvi_array.shape[0]
        zoom_x = slope_array.shape[1] / ndvi_array.shape[1]
        ndvi_array = ndimage.zoom(ndvi_array, (zoom_y, zoom_x), order=1)

    ls = np.clip((slope_array / 45.0) ** 1.4, 0, 1)
    c = 1.0 - np.clip(ndvi_array, 0, 1)
    risk = ls * c
    rmin, rmax = np.nanmin(risk), np.nanmax(risk)
    if rmax > rmin:
        risk = (risk - rmin) / (rmax - rmin)
    risk = risk.astype(np.float32)

    if cache_dir and transform is not None and HAS_RASTERIO:
        _save_tif(Path(cache_dir) / "erosion.tif", risk, transform, "EPSG:4326")

    return risk
