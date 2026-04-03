"""
Orbital Soil Intelligence Platform — Heavy Metal Detection

Main Streamlit application. Provides satellite-based screening of heavy metal
contamination in agricultural soils using Sentinel-2 spectral proxies.
"""

import io
import json
import os
from datetime import date

import ee
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from config import (
    CROP_NAMES,
    CROP_PROFILES,
    DETECTION_MODES,
    RISK_PALETTE,
    TARGET_METALS,
)
from heavy_metal_detector import build_proxy_layers, compute_timeseries
from session_manager import list_sessions, load_session, save_session

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Orbital Soil Intelligence", page_icon="🛰️", layout="wide")

st.title("🛰️ Orbital Soil Intelligence — Heavy Metal Detection")
st.caption(
    "Satellite-based screening of agricultural soil contamination: "
    "Cd, As, Hg, Pb, Cr, Cu, Zn, Li, Te"
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

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Scenario Configuration")

    # Session management
    saved = list_sessions()
    if saved:
        session_labels = ["(New Analysis)"] + [s["label"] for s in saved]
        selected_session = st.selectbox("Load Session", session_labels, index=0)
        if selected_session != "(New Analysis)":
            meta, results = load_session(selected_session)
            if meta:
                st.success(f"Loaded session: {selected_session}")
    else:
        selected_session = "(New Analysis)"

    st.divider()

    # Crop type
    crop_type = st.selectbox("Crop Type", CROP_NAMES, index=CROP_NAMES.index("Corn/Maize"))
    crop_info = CROP_PROFILES[crop_type]
    st.caption(f"Priority metals: {', '.join(crop_info['priority_metals'])}")

    st.divider()

    # Location
    lat = st.number_input("Latitude", value=7.2906, format="%.6f")
    lon = st.number_input("Longitude", value=80.6337, format="%.6f")
    radius_km = st.slider("AOI Radius (km)", 1.0, 25.0, 5.0, 0.5)

    # Date range
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date(2024, 1, 1))
    end = c2.date_input("To", value=date(2024, 12, 31))

    st.divider()

    # Detection settings
    mode = st.selectbox("Detection Mode", DETECTION_MODES, index=0)
    metal = st.selectbox("Target Metal", TARGET_METALS)

    run = st.button("Run Orbital Analysis", type="primary", use_container_width=True)

    st.divider()

    # Session save
    session_label = st.text_input("Session Label", value="default")
    save_btn = st.button("Save Session", use_container_width=True)


# ── Product framing ───────────────────────────────────────────────────────────

st.markdown(
    """
### Detection Methodology
- **Direct detection:** host-phase signatures on bare soil
- **Indirect detection:** crop stress signatures as bio-sensors (red edge shift, chlorophyll depletion)
- **Bimodal strategy:** combines both for continuous year-round intelligence

> ⚠️ This is a **screening tool** — results guide targeted lab analysis, not replace it.
"""
)


# ── Run analysis ──────────────────────────────────────────────────────────────

if run:
    with st.spinner("Computing risk surfaces and time-series from Sentinel-2..."):
        layers = build_proxy_layers(lat, lon, radius_km, str(start), str(end), metal)
        ts = compute_timeseries(lat, lon, radius_km, str(start), str(end), metal)

    st.session_state.analysis_result = {
        "layers": layers,
        "ts": ts.to_json(date_format="iso", orient="split"),
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "mode": mode,
        "metal": metal,
        "crop_type": crop_type,
    }


# ── Save session handler ─────────────────────────────────────────────────────

if save_btn and st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    ts_df = pd.read_json(io.StringIO(result["ts"]), orient="split")
    save_session(
        label=session_label,
        coordinates=[result["lat"], result["lon"]],
        start_date=start,
        end_date=end,
        results={"df_heavy_metal": ts_df},
        crop_type=result.get("crop_type", "Corn/Maize"),
    )
    # Persist the time-series CSV
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    ts_df.to_csv(f"{data_dir}/heavymetal_{session_label}.csv", index=False)
    st.sidebar.success(f"Session '{session_label}' saved.")


# ── Display results ───────────────────────────────────────────────────────────

result = st.session_state.analysis_result
if result is not None:
    layers = result["layers"]
    ts = pd.read_json(io.StringIO(result["ts"]), orient="split")
    ts["date"] = pd.to_datetime(ts["date"]) if not ts.empty else ts.get("date")
    lat_used = float(result["lat"])
    lon_used = float(result["lon"])
    radius_used = float(result["radius_km"])
    mode_used = result["mode"]
    metal_used = result["metal"]

    # ── Map ────────────────────────────────────────────────────────────────

    st.subheader("🗺️ Orbital Risk Map")

    m = folium.Map(location=[lat_used, lon_used], zoom_start=12, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles=layers["risk_tile"],
        attr="Google Earth Engine",
        name=f"{metal_used} Risk Surface",
        overlay=True,
        control=True,
    ).add_to(m)

    if mode_used != "Indirect (Canopy Stress)":
        folium.TileLayer(
            tiles=layers["bsi_tile"],
            attr="GEE",
            name="Bare Soil Index (Direct)",
            overlay=True,
            control=True,
        ).add_to(m)

    if mode_used != "Direct (Bare Soil)":
        folium.TileLayer(
            tiles=layers["ndvi_tile"],
            attr="GEE",
            name="NDVI / Canopy (Indirect)",
            overlay=True,
            control=True,
        ).add_to(m)

    folium.TileLayer(
        tiles=layers["ndmi_tile"],
        attr="GEE",
        name="Moisture Proxy (NDMI)",
        overlay=True,
        control=True,
    ).add_to(m)

    folium.Circle(
        [lat_used, lon_used],
        radius=radius_used * 1000,
        color="#00d2ff",
        fill=False,
        weight=2,
        tooltip="AOI",
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=560, use_container_width=True)

    # ── Metrics ────────────────────────────────────────────────────────────

    c1, c2, c3 = st.columns(3)

    latest_risk = (
        float(ts["Risk"].dropna().iloc[-1])
        if not ts.empty and ts["Risk"].notna().any()
        else np.nan
    )
    mean_risk = float(ts["Risk"].mean()) if not ts.empty else np.nan
    max_risk = float(ts["Risk"].max()) if not ts.empty else np.nan

    c1.metric("Latest Risk Score", f"{latest_risk:.2f}" if not np.isnan(latest_risk) else "N/A")
    c2.metric("Mean Risk Score", f"{mean_risk:.2f}" if not np.isnan(mean_risk) else "N/A")
    c3.metric("Peak Risk Score", f"{max_risk:.2f}" if not np.isnan(max_risk) else "N/A")

    # ── Contamination context ──────────────────────────────────────────────

    crop_used = result.get("crop_type", "Corn/Maize")
    profile = CROP_PROFILES.get(crop_used, {})
    if profile:
        with st.expander(f"📋 {crop_used} Contamination Profile"):
            st.markdown(f"**Priority metals:** {', '.join(profile['priority_metals'])}")
            st.markdown(f"**Notes:** {profile['notes']}")

    # ── Temporal diagnostics ───────────────────────────────────────────────

    st.subheader("📈 Temporal Diagnostics")

    if ts.empty:
        st.warning("No valid scenes found in this date range / AOI. Try widening the date window.")
    else:
        fig = px.line(
            ts,
            x="date",
            y=["Risk", "NDVI", "NDMI", "BSI"],
            title=f"{metal_used} Proxy Timeline",
        )
        fig.update_layout(legend_title_text="Signal")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ts.tail(24), use_container_width=True)

else:
    st.info("Configure AOI + date range, then click **Run Orbital Analysis**.")
