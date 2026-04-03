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

st.set_page_config(page_title="Orbital Soil Intelligence", page_icon="🛰️", layout="wide")

st.title("🛰️ Orbital Soil Intelligence — Heavy Metal Detection Demo")
st.caption("AgroSat-aligned demo focused on Cadmium, Mercury, Lithium, and Tellurium risk intelligence")


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


@st.cache_data(show_spinner=False)
def build_proxy_layers(lat: float, lon: float, radius_km: float, start_date: str, end_date: str, metal: str):
    geom = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
        .clip(geom)
    )

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndre = s2.normalizedDifference(["B8", "B5"]).rename("NDRE")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")
    bsi = s2.expression(
        "((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))",
        {
            "SWIR": s2.select("B11"),
            "RED": s2.select("B4"),
            "NIR": s2.select("B8"),
            "BLUE": s2.select("B2"),
        },
    ).rename("BSI")

    # Bimodal proxy: vegetation stress + bare-soil signature
    base_risk = (
        ndvi.multiply(-0.35)
        .add(ndre.multiply(-0.25))
        .add(ndmi.multiply(-0.20))
        .add(bsi.multiply(0.45))
    )

    weights = {
        "Cadmium (Cd)": 1.00,
        "Mercury (Hg)": 1.12,
        "Lithium (Li)": 0.86,
        "Tellurium (Te)": 1.22,
    }
    risk = base_risk.multiply(weights.get(metal, 1.0)).unitScale(-1, 1).clamp(0, 1).rename("risk")

    vis = {
        "risk": {"min": 0, "max": 1, "palette": ["#00a65a", "#f1c40f", "#e67e22", "#c0392b"]},
        "ndvi": {"min": -0.2, "max": 0.8, "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
        "ndmi": {"min": -0.5, "max": 0.5, "palette": ["#7f3b08", "#f7f7f7", "#2d004b"]},
        "bsi": {"min": -0.3, "max": 0.5, "palette": ["#1a9850", "#fee08b", "#d73027"]},
    }

    return {
        "geom": geom.getInfo(),
        "risk_tile": risk.getMapId(vis["risk"])["tile_fetcher"].url_format,
        "ndvi_tile": ndvi.getMapId(vis["ndvi"])["tile_fetcher"].url_format,
        "ndmi_tile": ndmi.getMapId(vis["ndmi"])["tile_fetcher"].url_format,
        "bsi_tile": bsi.getMapId(vis["bsi"])["tile_fetcher"].url_format,
    }


@st.cache_data(show_spinner=False)
def monthly_proxy_timeseries(lat: float, lon: float, radius_km: float, start_date: str, end_date: str, metal: str):
    geom = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
    )

    def add_indices(img):
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
        risk = (
            ndvi.multiply(-0.35)
            .add(ndre.multiply(-0.25))
            .add(ndmi.multiply(-0.20))
            .add(bsi.multiply(0.45))
            .multiply({"Cadmium (Cd)": 1.00, "Mercury (Hg)": 1.12, "Lithium (Li)": 0.86, "Tellurium (Te)": 1.22}.get(metal, 1.0))
            .unitScale(-1, 1)
            .clamp(0, 1)
            .rename("Risk")
        )
        return img.addBands([ndvi, ndre, ndmi, bsi, risk])

    coll = coll.map(add_indices)

    def summarize(img):
        stats = img.select(["NDVI", "NDRE", "NDMI", "BSI", "Risk"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=20,
            maxPixels=1e7,
        )
        return ee.Feature(None, stats).set("date", img.date().format("YYYY-MM-dd"))

    feats = coll.map(summarize).filter(ee.Filter.notNull(["Risk"]))
    fc = ee.FeatureCollection(feats)
    rows = fc.getInfo().get("features", [])

    data = []
    for r in rows:
        p = r.get("properties", {})
        data.append(
            {
                "date": p.get("date"),
                "NDVI": p.get("NDVI"),
                "NDRE": p.get("NDRE"),
                "NDMI": p.get("NDMI"),
                "BSI": p.get("BSI"),
                "Risk": p.get("Risk"),
            }
        )

    df = pd.DataFrame(data)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


with st.sidebar:
    st.header("Scenario")
    lat = st.number_input("Latitude", value=7.2906, format="%.6f")
    lon = st.number_input("Longitude", value=80.6337, format="%.6f")
    radius_km = st.slider("AOI Radius (km)", 1.0, 25.0, 5.0, 0.5)

    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date(2024, 1, 1))
    end = c2.date_input("To", value=date(2024, 12, 31))

    mode = st.selectbox(
        "Detection Mode",
        ["Bimodal (Recommended)", "Direct (Bare Soil)", "Indirect (Canopy Stress)"],
        index=0,
    )
    metal = st.selectbox("Target Metal", ["Cadmium (Cd)", "Mercury (Hg)", "Lithium (Li)", "Tellurium (Te)"])

    run = st.button("Run Orbital Analysis", type="primary", use_container_width=True)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


try:
    init_gee()
except Exception as exc:
    st.error("Earth Engine credentials missing/invalid. Set GEE_SERVICE_ACCOUNT_JSON and GEE_PROJECT in environment.")
    with st.expander("Error details"):
        st.exception(exc)
    st.stop()

st.markdown(
    """
### Product framing (from Orbital Soil Intelligence deck)
- Direct detection: host-phase signatures on bare soil
- Indirect detection: crop stress signatures as bio-sensors
- Bimodal strategy: continuous year-round intelligence
"""
)

if run:
    with st.spinner("Computing risk surfaces and time-series from Sentinel-2..."):
        layers = build_proxy_layers(lat, lon, radius_km, str(start), str(end), metal)
        ts = monthly_proxy_timeseries(lat, lon, radius_km, str(start), str(end), metal)
    st.session_state.analysis_result = {
        "layers": layers,
        "ts": ts.to_json(date_format="iso", orient="split"),
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "mode": mode,
        "metal": metal,
    }

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

    m = folium.Map(location=[lat_used, lon_used], zoom_start=12, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles=layers["risk_tile"],
        attr="Google Earth Engine",
        name=f"{metal_used} Risk Surface",
        overlay=True,
        control=True,
    ).add_to(m)
    if mode_used != "Indirect (Canopy Stress)":
        folium.TileLayer(tiles=layers["bsi_tile"], attr="GEE", name="Bare Soil Index (Direct)", overlay=True, control=True).add_to(m)
    if mode_used != "Direct (Bare Soil)":
        folium.TileLayer(tiles=layers["ndvi_tile"], attr="GEE", name="NDVI/Canopy (Indirect)", overlay=True, control=True).add_to(m)
    folium.TileLayer(tiles=layers["ndmi_tile"], attr="GEE", name="Moisture Proxy (NDMI)", overlay=True, control=True).add_to(m)

    folium.Circle(
        [lat_used, lon_used],
        radius=radius_used * 1000,
        color="#00d2ff",
        fill=False,
        weight=2,
        tooltip="AOI",
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    st.subheader("🗺️ Orbital Risk Map")
    st_folium(m, height=560, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    latest_risk = float(ts["Risk"].dropna().iloc[-1]) if not ts.empty and ts["Risk"].notna().any() else np.nan
    mean_risk = float(ts["Risk"].mean()) if not ts.empty else np.nan
    max_risk = float(ts["Risk"].max()) if not ts.empty else np.nan
    c1.metric("Latest Risk Score", f"{latest_risk:.2f}" if not np.isnan(latest_risk) else "N/A")
    c2.metric("Mean Risk Score", f"{mean_risk:.2f}" if not np.isnan(mean_risk) else "N/A")
    c3.metric("Peak Risk Score", f"{max_risk:.2f}" if not np.isnan(max_risk) else "N/A")

    st.subheader("📈 Temporal Diagnostics")
    if ts.empty:
        st.warning("No valid scenes found in this date range/AOI. Try widening the date window.")
    else:
        fig = px.line(ts, x="date", y=["Risk", "NDVI", "NDMI", "BSI"], title=f"{metal_used} Proxy Timeline")
        fig.update_layout(legend_title_text="Signal")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(ts.tail(24), use_container_width=True)
else:
    st.info("Configure AOI + date range, then click 'Run Orbital Analysis'.")
