"""
Orbital Soil Intelligence Platform — Heavy Metal Detection & Crop Analytics

Main Streamlit application with tabbed interface:
  Tab 1: Map & Boundary — draw/upload polygon, satellite overlays
  Tab 2: Heavy Metals — contamination risk analysis
  Tab 3: Timeline — multi-index time-series
  Tab 4: Data & Download — tabular data display, CSV export
"""

import io
import json
import os
import xml.etree.ElementTree as ET
from datetime import date

import ee
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from config import (
    CROP_NAMES,
    CROP_PROFILES,
    DETECTION_MODES,
    TARGET_METALS,
)
from heavy_metal_detector import build_proxy_layers, compute_timeseries
from satellite_collector import collect_sentinel1, collect_sentinel2
from weather_collector import collect_weather, get_centroid
from modis_collector import collect_modis_lst, add_cwsi
from session_manager import list_sessions, load_session, save_session

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Orbital Soil Intelligence", page_icon="🛰️", layout="wide")

st.title("🛰️ Orbital Soil Intelligence Platform")
st.caption(
    "Satellite-based screening of agricultural soil contamination & crop analytics"
)


# ── GEE initialisation ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Google Earth Engine...")
def init_gee():
    project = os.environ.get("GEE_PROJECT", "")
    sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    if not sa_json:
        raise RuntimeError("Missing GEE_SERVICE_ACCOUNT_JSON")
    info = json.loads(sa_json)
    sa_email = info.get("client_email")
    if not sa_email:
        raise RuntimeError("Service-account JSON missing client_email")
    creds = ee.ServiceAccountCredentials(sa_email, key_data=sa_json)
    ee.Initialize(creds, project=project or info.get("project_id"))
    return True


try:
    init_gee()
except Exception as exc:
    st.error(
        "Earth Engine credentials missing/invalid. "
        "Set GEE_SERVICE_ACCOUNT_JSON and GEE_PROJECT in environment."
    )
    with st.expander("Error details"):
        st.exception(exc)
    st.stop()


# ── Session state defaults ────────────────────────────────────────────────────

defaults = {
    "analysis_result": None,
    "coordinates": None,
    "map_center": [7.2906, 80.6337],
    "map_zoom": 12,
    "results": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helper: parse uploaded boundary files ─────────────────────────────────────

def _parse_geojson(data: dict) -> list[list[float]] | None:
    """Extract [[lon, lat], ...] from GeoJSON dict."""
    gtype = data.get("type", "")
    if gtype == "FeatureCollection":
        for feat in data.get("features", []):
            coords = _parse_geojson(feat)
            if coords:
                return coords
    elif gtype == "Feature":
        geom = data.get("geometry", {})
        return _parse_geojson(geom)
    elif gtype in ("Polygon", "MultiPolygon"):
        coords = data.get("coordinates", [])
        if gtype == "MultiPolygon":
            coords = coords[0]  # take first polygon
        if coords:
            return [[c[0], c[1]] for c in coords[0]]
    return None


def _parse_kml(content: str) -> list[list[float]] | None:
    """Extract polygon coordinates from KML."""
    try:
        root = ET.fromstring(content)
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        for coords_el in root.iter("{http://www.opengis.net/kml/2.2}coordinates"):
            text = coords_el.text.strip()
            pairs = []
            for part in text.split():
                vals = part.split(",")
                if len(vals) >= 2:
                    pairs.append([float(vals[0]), float(vals[1])])
            if len(pairs) >= 3:
                return pairs
    except Exception:
        pass
    return None


def _parse_csv_boundary(content: str) -> list[list[float]] | None:
    """Extract polygon from CSV with lon/lat columns."""
    try:
        df = pd.read_csv(io.StringIO(content))
        lon_col = next((c for c in df.columns if c.lower() in ("lon", "longitude", "lng", "x")), None)
        lat_col = next((c for c in df.columns if c.lower() in ("lat", "latitude", "y")), None)
        if lon_col and lat_col:
            coords = df[[lon_col, lat_col]].dropna().values.tolist()
            if len(coords) >= 3:
                return coords
    except Exception:
        pass
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    # Session management
    saved = list_sessions()
    if saved:
        session_labels = ["(New Analysis)"] + [s["label"] for s in saved]
        selected_session = st.selectbox("Load Session", session_labels, index=0)
        if selected_session != "(New Analysis)":
            meta, loaded_results = load_session(selected_session)
            if meta:
                st.session_state.results = loaded_results or {}
                st.success(f"Loaded: {selected_session}")
    else:
        selected_session = "(New Analysis)"

    st.divider()

    # Plot label & crop type
    session_label = st.text_input("Plot Label", value="plot_1")
    crop_type = st.selectbox("Crop Type", CROP_NAMES, index=CROP_NAMES.index("Corn/Maize"))
    crop_info = CROP_PROFILES[crop_type]
    st.caption(f"Priority metals: {', '.join(crop_info['priority_metals'])}")

    st.divider()

    # Date range
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date(2024, 1, 1))
    end = c2.date_input("To", value=date(2024, 12, 31))

    st.divider()

    # Boundary input method
    input_method = st.radio("Boundary Input", ["✏️ Draw on Map", "📁 Upload File"], index=0)

    if input_method == "📁 Upload File":
        uploaded = st.file_uploader(
            "Upload boundary",
            type=["geojson", "json", "kml", "csv"],
            help="GeoJSON, KML, or CSV with lon/lat columns",
        )
        if uploaded:
            content = uploaded.read().decode("utf-8")
            coords = None
            if uploaded.name.endswith((".geojson", ".json")):
                coords = _parse_geojson(json.loads(content))
            elif uploaded.name.endswith(".kml"):
                coords = _parse_kml(content)
            elif uploaded.name.endswith(".csv"):
                coords = _parse_csv_boundary(content)

            if coords:
                st.session_state.coordinates = coords
                lat_c, lon_c = get_centroid(coords)
                st.session_state.map_center = [lat_c, lon_c]
                st.success(f"Boundary loaded: {len(coords)} points")
            else:
                st.error("Could not parse boundary from file.")

    st.divider()

    # Detection settings (for Heavy Metals tab)
    mode = st.selectbox("Detection Mode", DETECTION_MODES, index=0)
    metal = st.selectbox("Target Metal", TARGET_METALS)

    st.divider()

    # ── Process All Data ──────────────────────────────────────────────────
    st.subheader("Data Collection")
    collect_s2 = st.checkbox("Sentinel-2 (Optical)", value=True)
    collect_s1 = st.checkbox("Sentinel-1 (SAR)", value=True)
    collect_wx = st.checkbox("Weather (NASA POWER)", value=True)
    collect_modis = st.checkbox("MODIS (Thermal/LST)", value=False)

    process_btn = st.button("▶ Process All Data", type="primary", use_container_width=True)


# ── Data collection pipeline ──────────────────────────────────────────────────

coords = st.session_state.coordinates

if process_btn:
    if coords is None or len(coords) < 3:
        st.error("Draw or upload a polygon boundary first.")
    else:
        geom = ee.Geometry.Polygon([coords])
        sd, ed = str(start), str(end)
        results = {"land_label": session_label}

        total_steps = sum([collect_s2, collect_s1, collect_wx, collect_modis])
        progress = st.progress(0, text="Starting data collection...")
        step = 0

        os.makedirs("data", exist_ok=True)

        if collect_s2:
            progress.progress(step / max(total_steps, 1), text="Collecting Sentinel-2...")
            df_s2 = collect_sentinel2(geom, sd, ed)
            results["df_s2"] = df_s2
            df_s2.to_csv(f"data/sentinel2_{session_label}.csv", index=False)
            step += 1

        if collect_s1:
            progress.progress(step / max(total_steps, 1), text="Collecting Sentinel-1 SAR...")
            df_s1 = collect_sentinel1(geom, sd, ed)
            results["df_s1"] = df_s1
            df_s1.to_csv(f"data/sentinel1_{session_label}.csv", index=False)
            step += 1

        if collect_wx:
            progress.progress(step / max(total_steps, 1), text="Fetching weather data...")
            df_wx = collect_weather(coords, sd, ed)
            results["df_weather"] = df_wx
            df_wx.to_csv(f"data/weather_{session_label}.csv")
            step += 1

        if collect_modis:
            progress.progress(step / max(total_steps, 1), text="Collecting MODIS LST...")
            df_modis = collect_modis_lst(coords, sd, ed)
            if collect_wx and "df_weather" in results:
                df_modis = add_cwsi(df_modis, results["df_weather"])
            results["df_modis"] = df_modis
            df_modis.to_csv(f"data/modis_{session_label}.csv", index=False)
            step += 1

        # Merge datasets on date
        merge_frames = []
        if "df_s2" in results and not results["df_s2"].empty:
            merge_frames.append(results["df_s2"])
        if "df_s1" in results and not results["df_s1"].empty:
            merge_frames.append(results["df_s1"])
        if "df_weather" in results and not results["df_weather"].empty:
            wx = results["df_weather"].reset_index()
            wx["date"] = pd.to_datetime(wx["date"])
            merge_frames.append(wx)

        if len(merge_frames) >= 2:
            merged = merge_frames[0]
            for other in merge_frames[1:]:
                merged = pd.merge(merged, other, on="date", how="outer", suffixes=("", "_dup"))
                merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
            merged = merged.sort_values("date").reset_index(drop=True)
            results["df_merged"] = merged
            merged.to_csv(f"data/merged_{session_label}.csv", index=False)

        progress.progress(1.0, text="Done!")
        st.session_state.results = results

        # Auto-save session
        save_session(
            label=session_label,
            coordinates=coords,
            start_date=start,
            end_date=end,
            results=results,
            crop_type=crop_type,
        )
        st.success(f"Session '{session_label}' saved with {len(results) - 1} datasets.")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_map, tab_metals, tab_timeline, tab_data = st.tabs([
    "🗺️ Map & Boundary",
    "☢️ Heavy Metals",
    "📈 Timeline",
    "📥 Data & Download",
])

res = st.session_state.results

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: MAP & BOUNDARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_map:
    st.subheader("🗺️ Map & Boundary")
    st.markdown("Draw a polygon on the map or upload a boundary file to define your area of interest.")

    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap",
    )

    # Add ESRI satellite basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="ESRI Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Drawing tools
    Draw(
        draw_options={
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
                "shapeOptions": {"color": "#FFD700", "weight": 3, "fill": False},
            },
            "rectangle": {
                "shapeOptions": {"color": "#FFD700", "weight": 3, "fill": False},
            },
            "circle": False,
            "circlemarker": False,
            "polyline": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # Show existing boundary
    if coords and len(coords) >= 3:
        folium.Polygon(
            locations=[[c[1], c[0]] for c in coords],
            color="#00d2ff",
            weight=2,
            fill=True,
            fill_opacity=0.1,
            tooltip="Current Boundary",
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    map_data = st_folium(
        m, key="main_map", height=600, use_container_width=True,
        returned_objects=["all_drawings", "last_active_drawing"],
    )

    # Capture drawn boundary
    if input_method == "✏️ Draw on Map" and map_data:
        drawings = map_data.get("all_drawings") or []
        if drawings:
            geom_data = drawings[-1].get("geometry", {})
            if geom_data.get("type") == "Polygon":
                new_coords = [[c[0], c[1]] for c in geom_data["coordinates"][0]]
                st.session_state.coordinates = new_coords
                lat_c, lon_c = get_centroid(new_coords)
                st.session_state.map_center = [lat_c, lon_c]

    # Boundary info
    if coords and len(coords) >= 3:
        lat_c, lon_c = get_centroid(coords)
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Centroid Lat", f"{lat_c:.6f}")
        bc2.metric("Centroid Lon", f"{lon_c:.6f}")
        bc3.metric("Points", len(coords))

        # Download boundary as GeoJSON
        geojson_str = json.dumps({
            "type": "Feature",
            "properties": {"label": session_label},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }, indent=2)
        st.download_button(
            "⬇️ Download Boundary as GeoJSON",
            data=geojson_str,
            file_name=f"{session_label}_boundary.geojson",
            mime="application/json",
        )
    else:
        st.info("Draw a polygon on the map or upload a boundary file to get started.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: HEAVY METALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_metals:
    st.subheader("☢️ Heavy Metal Contamination Analysis")
    st.markdown(
        """
**Detection modes:**
- **Direct:** host-phase signatures on bare soil
- **Indirect:** crop stress signatures as bio-sensors (red edge shift, chlorophyll depletion)
- **Bimodal:** combines both for year-round intelligence

> ⚠️ This is a **screening tool** — results guide targeted lab analysis, not replace it.
"""
    )

    run_metals = st.button("Run Heavy Metal Analysis", type="primary", key="run_metals")

    if run_metals:
        if coords and len(coords) >= 3:
            lat_c, lon_c = get_centroid(coords)
            radius_km = 5.0  # derive from boundary extent
            with st.spinner("Computing risk surfaces and time-series..."):
                layers = build_proxy_layers(lat_c, lon_c, radius_km, str(start), str(end), metal)
                ts = compute_timeseries(lat_c, lon_c, radius_km, str(start), str(end), metal)
            st.session_state.analysis_result = {
                "layers": layers, "ts": ts.to_json(date_format="iso", orient="split"),
                "lat": lat_c, "lon": lon_c, "radius_km": radius_km,
                "mode": mode, "metal": metal, "crop_type": crop_type,
            }
        else:
            st.error("Define a boundary first (Tab 1).")

    result = st.session_state.analysis_result
    if result is not None:
        layers = result["layers"]
        ts = pd.read_json(io.StringIO(result["ts"]), orient="split")
        ts["date"] = pd.to_datetime(ts["date"]) if not ts.empty else ts.get("date")
        lat_used, lon_used = float(result["lat"]), float(result["lon"])
        radius_used = float(result["radius_km"])
        mode_used, metal_used = result["mode"], result["metal"]

        # Risk map
        rm = folium.Map(location=[lat_used, lon_used], zoom_start=12, tiles="OpenStreetMap")
        folium.TileLayer(tiles=layers["risk_tile"], attr="GEE",
                         name=f"{metal_used} Risk Surface", overlay=True).add_to(rm)
        if mode_used != "Indirect (Canopy Stress)":
            folium.TileLayer(tiles=layers["bsi_tile"], attr="GEE",
                             name="Bare Soil Index (Direct)", overlay=True).add_to(rm)
        if mode_used != "Direct (Bare Soil)":
            folium.TileLayer(tiles=layers["ndvi_tile"], attr="GEE",
                             name="NDVI / Canopy (Indirect)", overlay=True).add_to(rm)
        folium.TileLayer(tiles=layers["ndmi_tile"], attr="GEE",
                         name="Moisture Proxy (NDMI)", overlay=True).add_to(rm)
        folium.Circle([lat_used, lon_used], radius=radius_used * 1000,
                       color="#00d2ff", fill=False, weight=2, tooltip="AOI").add_to(rm)
        folium.LayerControl(collapsed=False).add_to(rm)

        st.subheader("🗺️ Risk Map")
        st_folium(rm, height=500, use_container_width=True, key="risk_map")

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        latest_risk = float(ts["Risk"].dropna().iloc[-1]) if not ts.empty and ts["Risk"].notna().any() else np.nan
        mean_risk = float(ts["Risk"].mean()) if not ts.empty else np.nan
        max_risk = float(ts["Risk"].max()) if not ts.empty else np.nan
        mc1.metric("Latest Risk", f"{latest_risk:.2f}" if not np.isnan(latest_risk) else "N/A")
        mc2.metric("Mean Risk", f"{mean_risk:.2f}" if not np.isnan(mean_risk) else "N/A")
        mc3.metric("Peak Risk", f"{max_risk:.2f}" if not np.isnan(max_risk) else "N/A")

        # Crop profile
        profile = CROP_PROFILES.get(result.get("crop_type", "Corn/Maize"), {})
        if profile:
            with st.expander(f"📋 {result.get('crop_type', 'Corn/Maize')} Profile"):
                st.markdown(f"**Priority metals:** {', '.join(profile['priority_metals'])}")
                st.markdown(f"**Notes:** {profile['notes']}")

        # Timeline
        if not ts.empty:
            fig = px.line(ts, x="date", y=["Risk", "NDVI", "NDMI", "BSI"],
                          title=f"{metal_used} Proxy Timeline")
            fig.update_layout(legend_title_text="Signal")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ts.tail(24), use_container_width=True)
        else:
            st.warning("No valid scenes found. Try widening the date range.")
    else:
        st.info("Click **Run Heavy Metal Analysis** after defining a boundary.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: TIMELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_timeline:
    st.subheader("📈 Time-Series Analytics")

    df_s2 = res.get("df_s2")
    df_s1 = res.get("df_s1")
    df_wx = res.get("df_weather")
    df_modis_tl = res.get("df_modis")

    if df_s2 is not None and not df_s2.empty:
        # Vegetation indices
        vi_cols = [c for c in ["NDVI", "NDRE", "EVI", "REIP", "NDWI"] if c in df_s2.columns]
        if vi_cols:
            fig_vi = px.line(df_s2, x="date", y=vi_cols, title="Vegetation Indices")
            fig_vi.update_layout(legend_title_text="Index")
            st.plotly_chart(fig_vi, use_container_width=True)

        # Biophysical variables
        bio_cols = [c for c in ["LAI", "CCC", "CWC", "FAPAR", "FCOVER"] if c in df_s2.columns]
        if bio_cols:
            fig_bio = px.line(df_s2, x="date", y=bio_cols, title="Biophysical Variables (ESA SNAP)")
            st.plotly_chart(fig_bio, use_container_width=True)

        # SWIR indices
        swir_cols = [c for c in ["NBR", "NDII", "BSI", "MSR"] if c in df_s2.columns]
        if swir_cols:
            fig_swir = px.line(df_s2, x="date", y=swir_cols, title="SWIR-Enhanced Indices")
            st.plotly_chart(fig_swir, use_container_width=True)

    if df_s1 is not None and not df_s1.empty:
        fig_sar = px.line(df_s1, x="date", y=["VH", "VV", "RVI"], title="Sentinel-1 SAR Backscatter")
        st.plotly_chart(fig_sar, use_container_width=True)

    if df_wx is not None and not df_wx.empty:
        wx_plot = df_wx.reset_index() if "date" not in df_wx.columns else df_wx.copy()
        wx_plot["date"] = pd.to_datetime(wx_plot["date"])

        temp_cols = [c for c in ["temp_avg_c", "temp_max_c", "temp_min_c"] if c in wx_plot.columns]
        if temp_cols:
            fig_temp = px.line(wx_plot, x="date", y=temp_cols, title="Temperature (°C)")
            st.plotly_chart(fig_temp, use_container_width=True)

        if "rainfall_mm" in wx_plot.columns:
            fig_rain = px.bar(wx_plot, x="date", y="rainfall_mm", title="Rainfall (mm/day)")
            st.plotly_chart(fig_rain, use_container_width=True)

    if df_modis_tl is not None and not df_modis_tl.empty:
        lst_cols = [c for c in ["lst_day_c", "lst_night_c"] if c in df_modis_tl.columns]
        if lst_cols:
            fig_lst = px.line(df_modis_tl, x="date", y=lst_cols, title="MODIS Land Surface Temperature")
            st.plotly_chart(fig_lst, use_container_width=True)
        if "cwsi" in df_modis_tl.columns:
            fig_cwsi = px.line(df_modis_tl, x="date", y="cwsi", title="Crop Water Stress Index (CWSI)")
            st.plotly_chart(fig_cwsi, use_container_width=True)

    if not any([
        df_s2 is not None and not df_s2.empty,
        df_s1 is not None and not df_s1.empty,
        df_wx is not None and not df_wx.empty,
    ]):
        st.info("No data yet. Draw a boundary and click **▶ Process All Data** in the sidebar.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: DATA & DOWNLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_data:
    st.subheader("📥 Data & Download")

    lbl = res.get("land_label", session_label)

    datasets = {
        "📡 Sentinel-2 (Indices + Biophysical)": ("df_s2", f"sentinel2_{lbl}.csv", False),
        "📡 Sentinel-1 (SAR Backscatter)": ("df_s1", f"sentinel1_{lbl}.csv", False),
        "🌤️ Weather (NASA POWER)": ("df_weather", f"weather_{lbl}.csv", True),
        "🌡️ MODIS LST (Thermal / CWSI)": ("df_modis", f"modis_{lbl}.csv", False),
        "🔗 Merged Dataset": ("df_merged", f"merged_{lbl}.csv", False),
    }

    has_any = False
    for title, (key, filename, use_index) in datasets.items():
        df_show = res.get(key)
        if df_show is None:
            continue
        if isinstance(df_show, pd.DataFrame) and df_show.empty:
            continue
        has_any = True
        with st.expander(
            f"{title}  —  {len(df_show)} rows × {df_show.shape[1]} cols",
            expanded=(key == "df_s2"),
        ):
            st.dataframe(df_show, use_container_width=True, height=300)
            csv_bytes = df_show.to_csv(index=use_index).encode()
            st.download_button(
                f"⬇️ Download {filename}",
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
                key=f"dl_{key}",
            )

    if not has_any:
        st.info("No data collected yet. Draw a boundary and click **▶ Process All Data** in the sidebar.")
