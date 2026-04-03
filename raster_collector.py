"""
Per-pixel raster downloader for Sentinel-2 index data.

Downloads spectral indices per date as multi-band GeoTIFFs, cached under
data/sessions/<label>/rasters/<date>.tif. Used by 3D view and zone classifier.
"""

import io
import json
import zipfile
from pathlib import Path

import ee
import numpy as np
import requests

from config import MAX_CLOUD_COVER, S2_SCALE

try:
    import rasterio
    from rasterio.transform import Affine

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    Affine = None

# Import cloud masking and index functions from satellite_collector
from satellite_collector import _mask_s2_clouds, _add_vegetation_indices

RASTER_BANDS = ["NDVI", "NDRE", "EVI", "NDWI", "REIP", "NBR", "NDII", "BSI", "MSR"]


def _decode_gee_response(content: bytes) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
            if tifs:
                return z.read(tifs[0])
    except zipfile.BadZipFile:
        pass
    return content


def _save_multiband(path: Path, arrays: dict, transform, crs: str) -> None:
    if not HAS_RASTERIO:
        return
    bands = [b for b in RASTER_BANDS if b in arrays]
    if not bands:
        return
    h, w = next(iter(arrays.values())).shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32",
        width=w, height=h, count=len(bands),
        crs=crs, transform=transform, compress="lzw",
    ) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(arrays[band].astype(np.float32), i)
        dst.update_tags(band_names=json.dumps(bands))


def _load_multiband(path: Path):
    if not HAS_RASTERIO:
        return {}, None, None
    with rasterio.open(path) as src:
        tags = src.tags()
        band_names = json.loads(tags.get("band_names", "[]"))
        arrays = {}
        for i in range(src.count):
            name = band_names[i] if i < len(band_names) else f"band_{i+1}"
            band = src.read(i + 1).astype(np.float32)
            band[band < -9000] = np.nan
            arrays[name] = band
        return arrays, src.transform, str(src.crs)


def download_rasters_for_dates(
    coords_json: str,
    dates: list[str],
    cache_dir: Path,
    scale: int = S2_SCALE,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Download multi-band index GeoTIFF per date. Cached dates are skipped.

    Returns {date_str: {'arrays': {band: ndarray}, 'transform', 'crs'}}
    """
    if not HAS_RASTERIO:
        return {}

    coords = json.loads(coords_json)
    geometry = ee.Geometry.Polygon([coords])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total = len(dates)

    for idx, date_str in enumerate(dates):
        if progress_callback:
            progress_callback(idx / max(total, 1), f"Rasters — {date_str} ({idx+1}/{total})")

        cache_path = cache_dir / f"{date_str}.tif"

        # Load from cache
        if cache_path.exists():
            try:
                arrays, tr, crs = _load_multiband(cache_path)
                results[date_str] = {"arrays": arrays, "transform": tr, "crs": crs}
                continue
            except Exception:
                pass

        # Download from GEE
        try:
            date_ee = ee.Date(date_str)
            end_ee = date_ee.advance(6, "day")

            col = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(geometry)
                .filterDate(date_ee, end_ee)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
            )
            if col.size().getInfo() == 0:
                continue

            base_img = col.first()
            masked = _mask_s2_clouds(base_img)
            indexed = _add_vegetation_indices(masked)
            img = indexed.select(RASTER_BANDS).clip(geometry)

            url = img.getDownloadURL({
                "scale": scale, "crs": "EPSG:4326",
                "format": "GEO_TIFF", "region": geometry, "filePerBand": False,
            })

            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            tif_bytes = _decode_gee_response(resp.content)

            with rasterio.open(io.BytesIO(tif_bytes)) as src:
                n = min(src.count, len(RASTER_BANDS))
                transform = src.transform
                crs = str(src.crs)
                arrays = {}
                for i in range(n):
                    band = src.read(i + 1).astype(np.float32)
                    if src.nodata is not None:
                        band[band == src.nodata] = np.nan
                    band[band < -9000] = np.nan
                    arrays[RASTER_BANDS[i]] = band

            _save_multiband(cache_path, arrays, transform, crs)
            results[date_str] = {"arrays": arrays, "transform": transform, "crs": crs}

        except Exception as exc:
            if progress_callback:
                progress_callback(idx / max(total, 1), f"Error on {date_str}: {exc}")
            continue

    if progress_callback:
        progress_callback(1.0, "Raster download complete.")
    return results


def list_cached_dates(cache_dir: Path) -> list[str]:
    """Return sorted list of dates with cached rasters."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(p.stem for p in cache_dir.glob("*.tif"))
