"""
Centralised configuration for Orbital Soil Intelligence Platform.

Bands, thresholds, spectral signatures, crop profiles, and API settings.
"""

import os

# ── Google Earth Engine ───────────────────────────────────────────────────────
GEE_PROJECT = os.environ.get("GEE_PROJECT", "")

# ── Sentinel-2 band definitions ──────────────────────────────────────────────
# 10 m native: B2, B3, B4, B8
# 20 m native: B5, B6, B7, B8A, B11, B12
S2_BANDS = {
    "Blue":      "B2",   # 490 nm
    "Green":     "B3",   # 560 nm
    "Red":       "B4",   # 665 nm
    "RedEdge1":  "B5",   # 705 nm
    "RedEdge2":  "B6",   # 740 nm
    "RedEdge3":  "B7",   # 783 nm
    "NIR":       "B8",   # 842 nm
    "NIR2":      "B8A",  # 865 nm
    "SWIR1":     "B11",  # 1610 nm
    "SWIR2":     "B12",  # 2190 nm
}

S2_SELECT_BANDS = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "QA60",
]

# Max cloud cover % for S2 scene filter
MAX_CLOUD_COVER = 50

# Scale (metres) for S2 spatial aggregation — 20 m matches biophysical processor
S2_SCALE = 20

# ── Spectral indices ─────────────────────────────────────────────────────────
# Standard vegetation indices
S2_SPECTRAL_INDICES = [
    "NDVI", "NDRE", "EVI", "REIP", "NDWI", "NBR", "NDII", "BSI", "MSR",
]

# Metal-sensitive indices (from White Paper §4.3)
METAL_SENSITIVE_INDICES = [
    "PRI",    # Photochemical Reflectance Index — early stress, high metal sensitivity
    "MCARI",  # Modified Chlorophyll Absorption Ratio — chlorophyll concentration
    "CRI",    # Carotenoid Reflectance Index — carotenoid/chlorophyll ratio
    "WBI",    # Water Band Index — leaf water content
]

# ── Sentinel-1 SAR ───────────────────────────────────────────────────────────
S1_BANDS = ["VV", "VH"]

# ── Copernicus DEM ───────────────────────────────────────────────────────────
DEM_COLLECTION = "COPERNICUS/DEM/GLO30"
DEM_BAND = "DEM"
DEM_SCALE = 30
DEM_VIS = {
    "min": 0, "max": 2500,
    "palette": ["#1a4466", "#2d6a4f", "#52b788", "#b5e48c", "#ffffff"],
}

# ── NASA POWER weather variables ─────────────────────────────────────────────
POWER_VARIABLES = [
    "T2M",               # Temperature at 2 m (°C)
    "T2M_MAX",           # Max temperature
    "T2M_MIN",           # Min temperature
    "PRECTOTCORR",       # Precipitation (mm/day)
    "RH2M",              # Relative humidity at 2 m (%)
    "WS2M",              # Wind speed at 2 m (m/s)
    "ALLSKY_SFC_SW_DWN", # Shortwave radiation (kWh/m²/day)
    "ALLSKY_SFC_LW_DWN", # Longwave radiation
]
POWER_COMMUNITY = "AG"

# ── AlphaEarth Foundation embeddings ─────────────────────────────────────────
AEF_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
AEF_BANDS = [f"A{i:02d}" for i in range(64)]  # A00 … A63
AEF_SCALE = 10

# ── Planet Labs / PlanetScope ────────────────────────────────────────────────
PLANET_API_KEY = os.environ.get("PL_API_KEY", "")
PS_ITEM_TYPE = "PSScene"
PS_BUNDLE_8B = "analytic_8b_sr_udm2"
PS_BUNDLE_4B = "analytic_sr_udm2"
PS_MAX_CLOUD_COVER = 0.50
PS_SR_SCALE = 10_000.0

# ── Heavy metal detection ────────────────────────────────────────────────────

# Proxy weights for bimodal risk scoring
# Negative = lower value → higher risk; positive = higher value → higher risk
PROXY_WEIGHTS = {
    "NDVI": -0.35,
    "NDRE": -0.25,
    "NDMI": -0.20,
    "BSI":   0.45,
}

# Metal-specific multipliers (adjust base risk per element)
METAL_MULTIPLIERS = {
    "Cadmium (Cd)":   1.00,
    "Mercury (Hg)":   1.12,
    "Lithium (Li)":   0.86,
    "Tellurium (Te)": 1.22,
    "Arsenic (As)":   1.05,
    "Lead (Pb)":      0.95,
    "Chromium (Cr)":  0.90,
    "Copper (Cu)":    0.88,
    "Zinc (Zn)":      0.82,
}

# All supported target metals (display names)
TARGET_METALS = list(METAL_MULTIPLIERS.keys())

# Detection modes
DETECTION_MODES = [
    "Bimodal (Recommended)",
    "Direct (Bare Soil)",
    "Indirect (Canopy Stress)",
]

# Spectral signatures per contaminant (key wavelengths in nm)
# From White Paper §3.1
CONTAMINANT_SPECTRAL_SIGNATURES = {
    "Cadmium (Cd)":   {"wavelengths": [410, 581, 626, 670, 690], "mechanism": "organic matter / Fe-Mn oxides"},
    "Arsenic (As)":   {"wavelengths": [480, 580, 1100], "mechanism": "Fe hydroxide co-precipitation"},
    "Lead (Pb)":      {"wavelengths": [480, 530, 600, 680, 1400, 1600], "mechanism": "clay minerals / organic matter"},
    "Mercury (Hg)":   {"wavelengths": [550, 680], "mechanism": "organic matter co-variation (weak)"},
    "Chromium (Cr)":  {"wavelengths": [450, 600, 650, 1000, 1100], "mechanism": "Cr(III) d-d transitions"},
    "Copper (Cu)":    {"wavelengths": [480, 780, 830], "mechanism": "Cu(II) d-d + Fe oxide association"},
    "Zinc (Zn)":      {"wavelengths": [350, 450, 2200, 2350], "mechanism": "clay minerals / carbonates"},
    "Lithium (Li)":   {"wavelengths": [1400, 1900, 2200], "mechanism": "spodumene / clay mineralogy"},
    "Tellurium (Te)": {"wavelengths": [], "mechanism": "proxy mineral mapping (MWIR 3.5 µm — beyond S2)"},
}

# Regulatory contamination thresholds (mg/kg)
CONTAMINATION_THRESHOLDS = {
    "Cadmium (Cd)":   {"min": 0.3,  "max": 0.6},
    "Arsenic (As)":   {"min": 5.0,  "max": 20.0},
    "Lead (Pb)":      {"min": 50.0, "max": 300.0},
    "Mercury (Hg)":   {"min": 0.05, "max": 0.3},
    "Chromium (Cr)":  {"min": 100.0, "max": 100.0},
    "Copper (Cu)":    {"min": 60.0, "max": 150.0},
    "Zinc (Zn)":      {"min": 200.0, "max": 300.0},
    "Lithium (Li)":   {"min": None, "max": None},
    "Tellurium (Te)": {"min": None, "max": None},
}

# Risk colour palette (green → yellow → orange → red)
RISK_PALETTE = ["#00a65a", "#f1c40f", "#e67e22", "#c0392b"]

# ── Crop-specific contamination profiles ─────────────────────────────────────
# From White Paper §2.2 — each crop has priority metals and bioaccumulation notes
CROP_PROFILES = {
    "Tea": {
        "priority_metals": ["Lead (Pb)", "Cadmium (Cd)", "Copper (Cu)"],
        "notes": "Perennial accumulation; Cu fungicide use common",
    },
    "Rice": {
        "priority_metals": ["Cadmium (Cd)", "Arsenic (As)", "Lead (Pb)"],
        "notes": "Exceptionally efficient Cd/As uptake under flooded conditions; BAF 0.5–2.0",
    },
    "Wheat": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Chromium (Cr)"],
        "notes": "Grain Cd correlates with soil pH and organic matter",
    },
    "Corn/Maize": {
        "priority_metals": ["Lead (Pb)", "Cadmium (Cd)", "Zinc (Zn)"],
        "notes": "Deep roots access contaminated subsoils; source of HFCS",
    },
    "Cocoa": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)"],
        "notes": "EU Cd limits for chocolate; South American volcanic soils naturally high in Cd",
    },
    "Coffee": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Copper (Cu)"],
        "notes": "Drying/processing can introduce post-harvest Pb contamination",
    },
    "Sugarcane": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Chromium (Cr)"],
        "notes": "Global sweetener; refining may concentrate metals",
    },
    "Citrus": {
        "priority_metals": ["Copper (Cu)", "Cadmium (Cd)", "Lead (Pb)"],
        "notes": "Cu fungicide use adds contamination; perennial accumulation",
    },
    "Tomato": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Copper (Cu)"],
        "notes": "Fruit accumulates metals from leaves",
    },
    "Soybean": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Zinc (Zn)"],
        "notes": "N-fixing alters soil pH affecting metal bioavailability",
    },
    "Palm Oil": {
        "priority_metals": ["Cadmium (Cd)", "Chromium (Cr)", "Lead (Pb)"],
        "notes": "Bioaccumulation from soil to oil demonstrated",
    },
    "Potato": {
        "priority_metals": ["Cadmium (Cd)", "Lead (Pb)", "Arsenic (As)"],
        "notes": "Tuber accumulation from direct soil contact; peeling reduces but does not eliminate",
    },
}

CROP_NAMES = list(CROP_PROFILES.keys())

# ── Visualisation palettes ───────────────────────────────────────────────────
VIS_PALETTES = {
    "risk":  {"min": 0,    "max": 1,   "palette": RISK_PALETTE},
    "ndvi":  {"min": -0.2, "max": 0.8, "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
    "ndre":  {"min": 0,    "max": 0.9, "palette": ["#762a83", "#fee08b", "#1b7837"]},
    "ndmi":  {"min": -0.5, "max": 0.5, "palette": ["#7f3b08", "#f7f7f7", "#2d004b"]},
    "bsi":   {"min": -0.3, "max": 0.5, "palette": ["#1a9850", "#fee08b", "#d73027"]},
    "evi":   {"min": 0,    "max": 1,   "palette": ["#8c510a", "#f6e8c3", "#01665e"]},
}
