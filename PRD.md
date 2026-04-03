# Product Requirements Document (PRD)
## Orbital Soil Intelligence Platform — "Clean-Origin" Agricultural Monitoring

**Version:** 2.0
**Date:** 2026-04-02
**Status:** Draft (White Paper Integrated)
**Working Repo:** yasanthar (orbital-soil-intelligence-demo)
**Reference Repo:** agrosat
**Source Documents:**
- White Paper: "Satellite-Based Hyperspectral Detection of Heavy Metals in Agricultural Soils" (v1.0, April 2026)
- Technical Brief: "Orbital Hyperspectral Monitoring of Global Agricultural Contaminants"
- Reference Codebase: AgroSat (tea estate satellite analytics platform)

---

## 1. Executive Summary

14–17% of global cropland is contaminated with heavy metals. Traditional soil testing costs USD 20–50 per sample, takes 1–2 weeks, and provides only point-sample coverage. This platform combines **orbital remote sensing** with **agronomic intelligence** to deliver continuous, spatial contamination screening at scale.

The current demo (278-line Streamlit app) detects 4 metals via Sentinel-2 proxy indices. The target state is a comprehensive **"Soil-to-Shelf" transparency platform** that:

1. **Screens agricultural land** for heavy metal contamination (Cd, As, Hg, Pb, Cr, Cu, Zn) using satellite-derived spectral proxies
2. **Monitors crop health** via multi-sensor data fusion (Sentinel-1/2, MODIS, Planet, weather)
3. **Provides agronomic analytics** (GPP, crop quality, disease risk, nutrient status, water stress)
4. **Guides remediation** (phytoremediation, biochar, lime — matched to contamination type)
5. **Serves corporate ESG compliance** with supply chain assurance and audit capabilities

**Target market:** Precision agriculture (USD 15B+), heavy metal remediation (USD 8.1B → 12.3B by 2032), Earth observation analytics (USD 7–10B).

---

## 2. Current State (yasanthar)

### What Exists
- Single `app.py` (278 lines) with all logic in one file
- Sentinel-2 proxy layers: NDVI, NDRE, NDMI, BSI
- Heavy metal risk scoring (Cd, Hg, Li, Te) with bimodal/direct/indirect detection
- Basic Folium map with risk surface overlay
- Monthly time-series (risk, NDVI, NDMI, BSI)
- Streamlit-based UI with sidebar controls
- GEE service account authentication
- Render deployment config

### Critical Gaps (vs. White Paper Vision)
- Only 4 metals (white paper targets 8+ including As, Pb, Cr, Cu, Zn)
- No ML inversion models (PLSR, SVR, RF, 1D-CNN) — only weighted index scoring
- No vegetation stress indices for indirect detection (PRI, MCARI, CRI, WBI)
- No NPK nutrient monitoring
- No remediation intelligence
- No crop-specific contamination profiles (12 crops in white paper)
- No multi-sensor fusion (S1, MODIS, hyperspectral sources)
- No business-tier delivery (ESG dashboard, supply chain audit)
- No terrain/erosion correlation with contamination
- Monolithic architecture — no modularity

---

## 3. Scientific Foundation (from White Paper)

### 3.1 Contaminant Detection — Dual-Mode Approach

#### Direct Detection (Bare Soil Spectral Signatures)
Metals modify soil reflectance through association with Fe oxides, clay minerals, and organic matter.

| Contaminant | Key Wavelengths | Detection Mechanism | Lab R² | Field R² |
|-------------|----------------|---------------------|--------|----------|
| Cadmium (Cd) | 410, 581–626, 670–690 nm | Co-varies with organic matter, Fe/Mn oxides | 0.65–0.82 | 0.55–0.72 |
| Arsenic (As) | ~1100 nm; 480, 580 nm | Co-precipitates with Fe hydroxides | 0.71–0.80 | Ongoing |
| Lead (Pb) | 480–530, 600–680, 1400–1600 nm | Binds to clay minerals and organic matter | 0.60–0.78 | With ML |
| Mercury (Hg) | 550–680 nm (visible) | Weak direct; co-variation with organic matter | 0.45–0.65 | Hardest |
| Chromium (Cr) | 450, 600–650, 1000–1100 nm | Cr(III) d-d transitions; Cr(VI) co-precipitates | 0.55–0.70 | — |
| Copper (Cu) | 480, 780, 830 nm | Cu(II) d-d transitions + Fe oxide association | 0.65–0.80 | — |
| Zinc (Zn) | 350–450, 2200–2350 nm | Associated with clay minerals and carbonates | 0.60–0.75 | — |

**Spectral transformations:** First-order derivatives, continuum removal, reciprocal logarithmic transforms.

#### Indirect Detection (Plant Stress / Chlorophyll Depletion)
Heavy metals cause measurable vegetation stress:
- Chlorophyll-a/b reduction (660–680 nm red, 430–450 nm blue)
- Photosynthetic electron transport disruption (fluorescence changes)
- Carotenoid/xanthophyll ratio alteration (500–560 nm)
- **Red edge shift** (680–750 nm): blue-shift correlates with contamination severity (r = −0.76 for Cr/Cu/Ni)
- Leaf water content changes (NIR/SWIR)
- LAI reduction, altered canopy architecture

#### Key Vegetation Indices for Metal Detection

| Index | Formula | Target | Metal Sensitivity |
|-------|---------|--------|-------------------|
| NDVI | (NIR−Red)/(NIR+Red) | General vigor | Moderate |
| Red Edge Position (REP) | Inflection 680–750 nm | Chlorophyll/stress | High |
| PRI | (R531−R570)/(R531+R570) | Light use efficiency | High (early stress) |
| MCARI | [(R700−R670)−0.2×(R700−R550)]×(R700/R670) | Chlorophyll conc. | High |
| CRI | 1/R510 − 1/R550 | Carotenoid/chlorophyll ratio | Moderate-High |
| WBI | R900/R970 | Leaf water content | Moderate |
| NDRE | (NIR−RedEdge)/(NIR+RedEdge) | Crop stress | High |
| BSI | (SWIR+Red−NIR−Blue)/(SWIR+Red+NIR+Blue) | Bare soil ID | Direct mode |

### 3.2 ML Inversion Models (White Paper §4.4)

The platform must implement ensemble ML for spectral-to-concentration inversion:

| Method | Use Case | Notes |
|--------|----------|-------|
| PLSR (Partial Least Squares) | Robust baseline | Handles multicollinearity in spectral data |
| SVR (Support Vector Regression) | Strong for Pb, Cd | With spectral preprocessing |
| Random Forest / XGBoost | Feature importance ranking | Identifies diagnostic wavelengths |
| 1D-CNN | Highest accuracy | Deep learning on raw spectral signatures |
| Transfer Learning | Scale ground-truth to satellite | Ground hyperspectral → satellite multispectral |

**Implementation note:** Initially implement weighted proxy scoring (current approach) enhanced with the full index suite. ML models added in Phase 6 when ground-truth calibration data is available.

### 3.3 Regulatory Contamination Thresholds

| Metal | Threshold (mg/kg) | Priority Crops |
|-------|-------------------|----------------|
| Cd | 0.3–0.6 | Rice, Cocoa, Leafy greens |
| Pb | 50–300 | Citrus, Tea, Potato |
| As | 5–20 | Rice, Wheat, Sugar cane |
| Hg | 0.05–0.3 | Sugar cane, Fruit trees |
| Cr | 100 | Tomato, Soybean |
| Cu | 60–150 | Orange/Citrus, Coffee |
| Zn | 200–300 | Wheat, Soybean |

### 3.4 Crop-Specific Contamination Profiles (12 Crops)

| Crop | Key Metals | Notes |
|------|-----------|-------|
| Rice | Cd, As, Pb | Exceptionally efficient Cd/As uptake under flooded conditions; BAF 0.5–2.0 |
| Wheat | Cd, Pb, Cr | Grain Cd correlates with soil pH and organic matter |
| Corn/Maize | Pb, Cd, Zn | Deep roots access contaminated subsoils; source of HFCS |
| Cocoa | Cd, Pb | EU Cd limits for chocolate; volcanic soils naturally high in Cd |
| Coffee | Cd, Pb, Cu | Drying/processing can introduce post-harvest Pb |
| Tea | Pb, Cd, Cu | Perennial accumulation; existing AgroSat quality model |
| Sugarcane | Cd, Pb, Cr | Global sweetener; refining may concentrate metals |
| Citrus | Cu, Cd, Pb | Cu fungicide use adds contamination; perennial accumulation |
| Tomato | Cd, Pb, Cu | Fruit accumulates from leaves |
| Soybean | Cd, Pb, Zn | N-fixing alters soil pH affecting metal bioavailability |
| Palm Oil | Cd, Cr, Pb | Bioaccumulation from soil to oil demonstrated |
| Potato | Cd, Pb, As | Tuber accumulation from direct soil contact |

---

## 4. Target Architecture

### 4.1 Module Structure

```
yasanthar/
├── app.py                      # Main Streamlit app (tabbed UI)
├── config.py                   # Constants, band defs, thresholds, crop profiles
├── satellite_collector.py      # Sentinel-2 collection & processing
├── sar_collector.py            # Sentinel-1 SAR data
├── weather_collector.py        # NASA POWER weather API
├── modis_collector.py          # MODIS LST & CWSI
├── dem_collector.py            # Terrain analysis (DEM derivatives)
├── raster_collector.py         # Per-pixel raster downloads
├── embeddings_collector.py     # AlphaEarth Foundation embeddings
├── planetscope_collector.py    # Planet Labs integration (optional)
├── heavy_metal_detector.py     # Heavy metal risk engine (expanded)
├── contamination_profiles.py   # Crop-specific thresholds & spectral signatures
├── remediation_engine.py       # Remediation recommendations
├── analytics.py                # GPP, quality scoring, disease risk
├── zone_classifier.py          # Pixel-level agronomic zone classification
├── gee_layers.py               # GEE tile layer URL generation
├── session_manager.py          # Session persistence & caching
├── requirements.txt            # Python dependencies
├── render.yaml                 # Deployment config
└── data/                       # Session data directory
    └── sessions/
```

### 4.2 5-Subsystem Architecture (White Paper §9)

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA ACQUISITION LAYER                                   │
│  GEE (S1, S2, MODIS, DEM, AEF) │ NASA POWER │ Planet (opt) │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  2. PREPROCESSING PIPELINE                                   │
│  Atmospheric correction │ 3-layer cloud mask │ Co-registration│
│  Spatial enhancement │ Quality assessment                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  3. ANALYSIS ENGINE                                          │
│  Bare-soil detection │ Spectral transforms │ Proxy scoring  │
│  Vegetation stress │ Biophysicals │ Data fusion              │
│  Heavy metal risk │ NPK estimation │ Erosion risk            │
│  [Future: ML inversion (PLSR/SVR/RF/CNN)]                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  4. DECISION SUPPORT SYSTEM                                  │
│  Risk classification maps │ Crop-specific advisory          │
│  Remediation recommendations │ NPK variable-rate maps       │
│  Irrigation scheduling │ Temporal change alerts             │
│  Zone classification │ Disease risk │ Quality scoring        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  5. DELIVERY PLATFORM                                        │
│  Streamlit web dashboard │ CSV/GeoTIFF export               │
│  Session persistence │ [Future: REST API, PDF reports]      │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.40+ |
| Maps | Folium + streamlit-folium |
| Charts | Plotly (3D, time-series), Matplotlib (dark-themed) |
| Satellite | Google Earth Engine (Sentinel-1/2, MODIS, DEM, AEF) |
| High-Res Imagery | Planet Labs PlanetScope (optional) |
| Weather | NASA POWER API |
| Raster I/O | Rasterio, NumPy |
| Geospatial | Shapely, PyProj, GeoPy |
| Terrain | SciPy (derivatives) |
| Deployment | Render / Streamlit Community Cloud |

---

## 5. Feature Requirements

### Phase 1: Foundation & Architecture (Priority: Critical)

#### F1.1 — Modular Codebase Refactor
**Description:** Extract the monolithic `app.py` into the module structure defined in §4.1.
**Acceptance Criteria:**
- Each module has a single responsibility
- Heavy metal detection logic extracted to `heavy_metal_detector.py`
- Configuration constants moved to `config.py`
- GEE initialization shared across modules
- All existing functionality preserved

#### F1.2 — Configuration Module
**Description:** Centralized configuration for bands, thresholds, API settings, crop profiles.
**Includes:**
- Sentinel-2 band definitions (12 bands: B2–B12 + SCL + cloud probability)
- Sentinel-1 band definitions (VV, VH)
- Spectral index formulas and ranges (9 vegetation + 6 metal-sensitive indices)
- Cloud cover thresholds (MAX_CLOUD_COVER = 50%)
- Scale/resolution settings (S2_SCALE = 20m)
- NASA POWER variable list (8 variables)
- DEM settings (Copernicus GLO-30, 30m resolution)
- Heavy metal spectral signatures table (from White Paper §3.1)
- Contamination thresholds per metal (from White Paper §3.1)
- Crop-specific contamination profiles (12 crops from White Paper §2.2)
- Metal-specific proxy weights and multipliers
- AlphaEarth embedding dimensions (64D)
- Planet Labs config (SuperDove 8-band, 3m)

#### F1.3 — Enhanced Cloud Masking
**Description:** Upgrade from basic QA60 filtering to 3-layer cloud masking.
**Implementation:**
- Layer 1: QA60 (ESA bands 10 & 11 — clouds, cirrus)
- Layer 2: SCL (Scene Classification — cloud shadows, thin cirrus, classes 0–11)
- Layer 3: MSK_CLDPRB (s2cloudless probability threshold)
- Cloud-masked pixels flagged as NaN in downstream processing
- Cloud coverage reporting (warnings at >5% and >30%)

#### F1.4 — Session Management
**Description:** Persistent session save/load with multi-level caching.
**Implementation:**
- Session metadata (JSON): label, coordinates, dates, crop type, data flags
- CSV persistence for all time-series datasets
- GeoTIFF caching for rasters, DEM, zones
- Auto-save after data collection
- Session listing and restore UI in sidebar
- Directory structure: `data/sessions/<label>/`

---

### Phase 2: Multi-Sensor Data Integration (Priority: High)

#### F2.1 — Sentinel-2 Enhanced Collection
**Description:** Upgrade S2 processing with biophysical variables, extended indices, and metal-sensitive indices.

**Standard Vegetation Indices:** NDVI, NDRE, EVI, REIP, NDWI, NBR, NDII, BSI, MSR

**Metal-Sensitive Indices (NEW from White Paper):**
- PRI (Photochemical Reflectance Index) — early stress detection, high metal sensitivity
- MCARI (Modified Chlorophyll Absorption Ratio Index) — chlorophyll concentration
- CRI (Carotenoid Reflectance Index) — carotenoid/chlorophyll ratio
- WBI (Water Band Index) — leaf water content

**Biophysical Variables (ESA SNAP S2 Processor v2.1):**
- LAI (Leaf Area Index)
- CCC (Canopy Chlorophyll Content)
- CWC (Canopy Water Content)
- FAPAR (Fraction Absorbed PAR)
- FCOVER (Vegetation Cover Fraction)
- REIP (Red Edge Inflection Point — nm)

**Implementation:** Neural network model with 11 inputs (8 S2 bands + 3 geometry angles), normalization/denormalization per variable. Per-pixel spatial stats (mean + std across plot).

#### F2.2 — Sentinel-1 SAR Integration
**Description:** Add radar backscatter data for all-weather monitoring.
**Outputs:**
- VH polarization (dB)
- VV polarization (dB)
- RVI (Radar Vegetation Index): 4 × VH / (VV + VH)
- 10m scale, mean over plot
**Value:** Cloud-penetrating data fills gaps during monsoon seasons — critical for tropical crop regions (cocoa, rice, palm oil, sugarcane).

#### F2.3 — Weather Data (NASA POWER)
**Description:** Integrate daily agroclimatology data.
**Variables:**
- Temperature: avg, max, min (°C) — needed for disease risk, quality scoring, CWSI
- Rainfall (mm/day) — disease risk trigger
- Relative humidity (%) — disease risk factor
- Wind speed (m/s) — contaminant dispersal proxy
- Solar radiation — shortwave & longwave (MJ/m²/day) — GPP calculation
**API:** NASA POWER, community=AG, no auth required.

#### F2.4 — MODIS Thermal & CWSI
**Description:** Land surface temperature and crop water stress monitoring.
**Data:** MOD11A1 daily, 1km resolution. QC masking (bits 0–1 = mandatory good quality).
**Outputs:**
- Daytime LST (°C): DN × 0.02 − 273.15
- Nighttime LST (°C)
- Diurnal delta (proxy for ET activity)
- CWSI: (LST_day − T_air + 2) / 10, clipped [0, 1]
**Caveat:** 1km resolution — estate-scale indicator, not plot-level.

#### F2.5 — AlphaEarth Foundation Embeddings
**Description:** Annual 64-dimensional learned representations from Google DeepMind.
**Source:** GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL (2017–2025)
**Value:**
- Encodes full-year phenology from multi-sensor data
- Unsupervised change detection via year-over-year cosine similarity
- Potential for detecting long-term contamination trends

#### F2.6 — Planet Labs Integration (Optional)
**Description:** High-resolution 3m imagery from PlanetScope SuperDove.
**Features:**
- Data API scene catalog search
- Orders API AOI-clipped delivery
- 8-band surface reflectance + UDM2 quality masking
- NDVI, NDRE, EVI, GNDVI indices
- Near-daily revisit enables rapid change detection
**Requirement:** PL_API_KEY environment variable
**White Paper note:** 3m target resolution achievable via Planet + Sentinel-2 fusion techniques.

#### F2.7 — Data Merge & Export
**Description:** Join all datasets on date, provide CSV downloads.
**Merged columns:** All S2 indices + biophysicals + metal-sensitive indices + S1 + weather + embeddings + MODIS + analytics
**Downloads:** Per-source CSV + merged CSV + GeoTIFF rasters, accessible from Data tab.

---

### Phase 3: Terrain & Spatial Analysis (Priority: High)

#### F3.1 — DEM & Terrain Derivatives
**Description:** Copernicus GLO-30 DEM processing for terrain analysis.
**Derivatives:**
- Elevation contour map (20-level terrain colormap)
- Slope (0–60°)
- Aspect (0–360°, clockwise from N)
- Hillshade (NW sun 315°, 45° elevation)
- Flow direction vectors (drainage patterns)
- **Erosion risk** (RUSLE LS × C factor: slope × (1 − NDVI))
**Heavy Metal Correlation:** Erosion risk × contamination = metal transport risk. Drainage flow direction indicates downstream contamination spread.
**Cache:** `data/sessions/<label>/dem/`

#### F3.2 — Per-Pixel Raster Downloads
**Description:** Multi-band GeoTIFF downloads for each S2 date.
**Bands:** NDVI, NDRE, EVI, NDWI, REIP, NBR, NDII, BSI, MSR (+ PRI, MCARI, CRI, WBI for metal detection)
**Resolution:** 20m
**Cache:** `data/sessions/<label>/rasters/<date>.tif`
**Uses:** 3D visualization, zone classification, pixel-level contamination mapping

#### F3.3 — 3D Visualization
**Description:** Interactive 3D surface plot combining DEM with vegetation/contamination indices.
**Features:**
- DEM mesh colored by selectable index (vegetation OR heavy metal risk)
- Plot boundary overlay (gold line, elevated)
- Temporal slider across cached dates
- Configurable lighting (ambient, diffuse, fresnel)
- Index statistics panel with time-series

#### F3.4 — Zone Classification (8 Zones — expanded)
**Description:** Pixel-level agronomic zone mapping.
**Zones (priority order):**
1. **Contamination Risk** (heavy metal risk > 0.7) — NEW
2. Healthy Growth (NDVI > 0.65, NDRE > 0.22)
3. Nitrogen Stress (NDRE < 0.15, REIP < 716nm)
4. Fertilization Deficient (NDRE < 0.18, NDVI < 0.55)
5. Water Stress (NDWI < −0.25, NDII < 0.05)
6. Senescence Risk (EVI < 0.32, NDVI < 0.50)
7. Bare Soil Risk (BSI > 0.08, NDVI < 0.35)
8. Unclassified / Cloud-Masked

**Outputs:** Zone map (Folium GeoJSON overlay), distribution chart, time-series of zone areas
**Multi-session comparison:** Overlay zones from different sessions/locations

---

### Phase 4: Analytics Engine (Priority: High)

#### F4.1 — Heavy Metal Detection Engine (Expanded)
**Description:** Comprehensive contamination risk assessment — the platform's core differentiator.

**Expanded Contaminant Coverage (8 metals):**
- Cadmium (Cd), Arsenic (As), Lead (Pb), Mercury (Hg)
- Chromium (Cr), Copper (Cu), Zinc (Zn)
- Tellurium (Te) — via proxy mineral mapping (white paper §4: infrared outlier)

**Detection Modes:**
- **Bimodal** (recommended): Combines vegetation stress + bare soil signatures
- **Direct** (bare soil): Host-phase signatures on exposed soil — uses BSI to identify bare-soil pixels
- **Indirect** (canopy stress): Crop stress as bio-sensors — red edge shift, PRI, MCARI, CRI

**Enhanced Risk Scoring:**
- Weighted combination of metal-sensitive indices (NDVI, NDRE, NDMI, BSI, PRI, MCARI, CRI, WBI, REIP)
- Metal-specific multipliers (from spectral signature table)
- Crop-specific bioaccumulation factor (BAF) adjustment
- Terrain correlation: erosion risk × contamination = transport risk
- Integration with biophysical variables (LAI, CCC as stress amplifiers)

**Contamination Map Layers:**
- Per-metal risk surface (0–1 normalized, color-coded)
- Composite contamination risk (max across metals)
- "At-Risk Pixels" flagging (30m × 30m per white paper)
- Downstream contamination flow (DEM drainage × risk)

**Temporal Analysis:**
- Multi-temporal risk trend analysis
- Seasonal contamination patterns (bare soil in fallow vs. crop stress in growing)
- Year-over-year change detection (via AEF embeddings cosine similarity)

#### F4.2 — Crop-Specific Contamination Profiles
**Description:** Configurable crop type selection adjusts detection parameters.
**Per-crop configuration:**
- Target metals and priority ranking
- Bioaccumulation factors
- Contamination thresholds (regulatory limits)
- Optimal detection season (fallow for direct, growing for indirect)
- Crop-specific vegetation index baselines
**Default:** Tea (Sri Lanka). Extensible to all 12 crops in white paper.

#### F4.3 — NPK Nutrient Monitoring (from White Paper §6)
**Description:** Remote sensing-based nutrient status estimation.
**Nitrogen:**
- Deficiency via reduced chlorophyll: increased 550nm reflectance, decreased 680nm absorption
- Red edge shift (710–740nm) for N status
- CCC (from biophysical processor) as direct N proxy
- **Target:** R² = 0.70–0.85 with ML + ground truth
**Phosphorus:**
- Indirect: co-varies with organic matter and Fe oxide mineralogy
- Plant-level: P deficiency → increased anthocyanin (~550nm)
- **Target:** R² = 0.50–0.70
**Potassium:**
- Weakest spectral correlations
- K deficiency → leaf margin necrosis, altered water regulation
- **Target:** R² = 0.40–0.60
**Value:** Satellite-guided variable-rate fertilization can reduce N use by 15–30% while maintaining yields.

#### F4.4 — Remediation Intelligence (from White Paper §7)
**Description:** Automated remediation recommendations based on contamination type.

**Remediation Methods:**
| Method | Applicable Metals | Mechanism | Monitoring |
|--------|-------------------|-----------|------------|
| **Biochar** | Cd (−52%), Pb (−46%), Cu (−29%), Zn (−36%) | pH increase, CEC, surface complexation | NDVI recovery tracking |
| **Lime** | Pb, Cd, Zn | pH to 6.5–7.0, insoluble hydroxides | NOT for As (increases mobility) |
| **Phytoextraction** | Zn/Cd (Thlaspi), As (Pteris vittata) | Hyperaccumulator plant uptake | Satellite harvest timing |
| **Phytostabilization** | General | Root exudates immobilize metals | Long-term NDVI stability |
| **Iron-based** | As-specific | Fe precipitation | LST/moisture monitoring |
| **EDTA Chelation** | Pb (enhanced phytoextraction) | Increases metal mobility | Groundwater risk flag |

**Platform delivers:**
- Risk-matched remediation recommendations (per metal, soil type, crop)
- Progress monitoring (multi-temporal satellite tracking of remediation effectiveness)
- Cost-effectiveness comparison
- Groundwater risk warnings (e.g., lime + As, EDTA leaching)

#### F4.5 — Gross Primary Production (GPP)
**Description:** Carbon fixation model for productivity estimation.
**Formula:** `GPP = FAPAR × (solar_rad_mj × 0.45) × LUE`
**LUE:** Configurable per crop type (default: 1.0 g C/MJ PAR for tea)
**Outputs:** Daily GPP, cumulative GPP, productivity classification
**Productivity labels:** Sparse < 1.5, Recovery 1.5–3, Normal 3–6, Peak > 6 g C/m²/day

#### F4.6 — Crop Quality Score
**Description:** Composite quality index (0–100) from satellite + weather signals.
**Default (Tea) Components:**
- CCC (35%): Bell curve at 300 µg/cm² — optimal N balance
- CWC (20%): Canopy water 0.05–0.22 g/cm²
- REIP (20%): Chlorophyll quality 710–726nm
- Diurnal range (25%): Temperature swing 5–15°C (Nuwara Eliya effect)
**Grades:** Premium (80+), Standard (60–79), Low (40–59), Poor (<40)
**Extensibility:** Component weights configurable per crop type.

#### F4.7 — Disease Risk Engine
**Description:** Multi-pathogen risk assessment from meteorological + canopy data.
**Pathogens:**
- **Blister Blight** (Exobasidium vexans): RH > 85%, 20–26°C, FCOVER density amplifier. Weights: 50% RH, 30% rain, 20% temp. Rolling 3-day avg.
- **Anthracnose** (Colletotrichum): RH > 80%, 22–28°C, rain > 2mm. Weights: 45% RH, 35% rain, 20% temp.
- **Grey Blight** (Pestalotiopsis): Drought-then-wet mechanism — NDWI < −0.15 for 5+ days, then RH > 75% or rain > 1mm. 80% drought exposure × wet trigger + 20% RH.

**Composite Alert:** ✅ None (<0.25), 👁 Watch (0.25–0.45), ⚠️ Warning (0.45–0.70), 🚨 Alert (≥0.70)

---

### Phase 5: UI/UX Enhancement (Priority: Medium)

#### F5.1 — Tabbed Interface (10 Tabs)
1. **🗺️ Map & Boundary** — Interactive map, boundary drawing/upload, satellite overlays
2. **📈 Timeline** — Multi-index time-series, biophysical trends, SAR, weather
3. **📥 Data & Download** — Tabular data display, CSV/GeoTIFF export
4. **🏔️ Terrain Analysis** — DEM derivatives, erosion risk, flow direction
5. **🧊 3D View** — Interactive 3D surface with index overlay
6. **🌱 Zone Analysis** — Agronomic + contamination zone classification
7. **📊 Analytics** — GPP, quality, disease risk, thermal/CWSI (sub-tabs)
8. **☢️ Heavy Metals** — Contamination risk maps, per-metal analysis, temporal trends
9. **🧪 Nutrients & Remediation** — NPK status, remediation recommendations, progress tracking
10. **📖 Reference** — Band definitions, formulas, crop profiles, contaminant guide

#### F5.2 — Boundary Management
**Description:** Replace lat/lon + radius with proper polygon boundaries.
**Input Methods:**
- Draw polygon/rectangle on map (Folium drawing tools)
- Upload GeoJSON, KML, or CSV (lon/lat columns)
- Location search via geocoding (Nominatim)
**Metrics:** Centroid, point count, estimated area (hectares)
**Export:** Download boundary as GeoJSON

#### F5.3 — Crop Type Selector
**Description:** Dropdown selector for target crop, adjusting detection parameters.
**Options:** Tea (default), Rice, Wheat, Corn, Cocoa, Coffee, Sugarcane, Citrus, Tomato, Soybean, Palm Oil, Potato
**Effect:** Adjusts metal priorities, contamination thresholds, quality model weights, disease pathogens, BAFs

#### F5.4 — Satellite Overlay Controls
**Layers:**
- S2: True Color, False Color (NIR), NDVI, NDRE, EVI
- S1: VH, VV, RVI
- DEM: Elevation, Slope, Hillshade
- Heavy Metal Risk Surface (per-metal selectable)
- Contamination Zone overlay
**Features:** Layer toggle, custom colour palette editor, date-based image viewer (locked viewport)

#### F5.5 — Sidebar Enhancement
**Controls:**
- Plot label input + crop type selector
- Date range selector
- Location search (geocoding)
- Boundary input method selector
- Satellite overlay picker + colour customizer
- Data processing checklist (S2, S1, Weather, Embeddings, MODIS)
- Session save/load
- "Process All Data" button

---

### Phase 6: ML & Advanced Analytics (Priority: Future)

#### F6.1 — ML Inversion Models
**Description:** Replace proxy scoring with trained ML models for concentration estimation.
**Prerequisite:** Ground-truth calibration data from 3–5 regions.
**Models:** PLSR baseline → SVR/RF ensemble → 1D-CNN for highest accuracy.
**Target regions (from white paper):** Southern China (rice), West Africa (cocoa), India (sugarcane), Brazil (soybean), Europe (wheat).

#### F6.2 — Hyperspectral Data Integration
**Description:** Integrate data from dedicated hyperspectral satellites when available.
**Sources:** PRISMA (free, 30m, 250 bands), EnMAP (free, 30m, 224 bands), EMIT (free, 60m, 285 bands), Pixxel Honeybee (commercial, 8m, 450 bands, 2026+).
**Value:** Full VSWIR coverage (400–2500nm) enables direct mineral detection that Sentinel-2's 13 bands cannot achieve.

#### F6.3 — ESG Dashboard & Supply Chain Audit
**Description:** Corporate-facing reporting features.
**Features:**
- Dynamic ESG Score integrating soil health, contamination, carbon sequestration
- Supply chain risk mapping across multiple sourcing regions
- "At-Risk Pixel" flagging (30m × 30m)
- Automated PDF report generation
- Regenerative agriculture validation (fertilizer reduction via N/chlorophyll signatures)

#### F6.4 — REST API
**Description:** Programmatic access for third-party integration.
**Endpoints:** Risk query by AOI, time-series retrieval, zone classification, batch processing.
**Format:** GeoJSON, Cloud Optimized GeoTIFF.

---

## 6. Non-Functional Requirements

### 6.1 Performance
- GEE queries cached via `@st.cache_data` and `@st.cache_resource`
- Raster and DEM data cached to disk
- Session auto-save to prevent data loss on page refresh
- Lazy loading of expensive computations (terrain, 3D, zones)

### 6.2 Scalability
- Support for any lat/lon worldwide (not limited to Sri Lanka)
- Configurable crop type with 12 crop profiles
- Multi-session comparison for estate/regional analysis
- Multi-source satellite architecture (no single-point-of-failure per white paper)

### 6.3 Reliability
- Graceful degradation when optional APIs unavailable (Planet, MODIS)
- Empty result set handling with user-friendly warnings
- Cloud coverage warnings (>5%, >30% thresholds)
- Error handling for GEE quota limits
- Position as **screening tool** guiding targeted lab analysis — NOT a replacement (per white paper risk analysis)

### 6.4 Security
- No credentials in code (all via environment variables)
- GEE service account authentication
- Planet API key optional
- `.gitignore` excludes `.env`, secrets, `__pycache__`

### 6.5 Deployment
- **Primary:** Render.com (via render.yaml blueprint)
- **Alternative:** Streamlit Community Cloud
- **Environment Variables Required:**
  - `GEE_PROJECT` — Google Earth Engine project ID
  - `GEE_SERVICE_ACCOUNT_JSON` — Service account credentials
  - `PL_API_KEY` — Planet Labs API key (optional)

---

## 7. Implementation Phases & Priority

| Phase | Scope | Priority | Complexity |
|-------|-------|----------|------------|
| **Phase 1** | Architecture refactor, config, cloud masking, sessions | Critical | Medium |
| **Phase 2** | Multi-sensor data (S2+, S1, Weather, MODIS, AEF, Planet) | High | High |
| **Phase 3** | Terrain, rasters, 3D view, zone classification | High | High |
| **Phase 4** | Analytics (heavy metals expanded, NPK, remediation, GPP, quality, disease) | High | High |
| **Phase 5** | UI/UX (10-tab interface, boundary mgmt, crop selector, overlays) | Medium | Medium |
| **Phase 6** | ML inversion, hyperspectral integration, ESG dashboard, API | Future | Very High |

**Note:** Phases 2–4 can be partially parallelized. Phase 1 is prerequisite for all others. Phase 6 requires ground-truth data partnerships.

---

## 8. Business Model Alignment (White Paper §8)

### Service Tiers (Future Revenue Model)

| Tier | Target Customer | Pricing | Description |
|------|----------------|---------|-------------|
| **Tier 1: Regional Screening** | Govts, intl orgs (FAO, World Bank) | USD 0.10–0.50/ha/year | Annual updates, quarterly hotspot monitoring |
| **Tier 2: Farm Intelligence** | Large agribusiness, cooperatives | USD 2–10/ha/year | Multi-sensor analysis, zone classification |
| **Tier 3: Supply Chain Assurance** | Food manufacturers, retailers | USD 5K–50K/audit | Project-based contamination audits |
| **Tier 4: Remediation Intelligence** | Environmental consultants | USD 10–50/ha/year | Detection + remediation + monitoring |
| **Tier 5: Data API** | Ag-tech companies, ESG analysts | Usage-based per query | Developer platform for third-party apps |

### Corporate Engagement Strategy ("Clean-Origin" Platform)
**Phase 1 — Pilot Basin Monitoring:** Target high-priority water basins where major corps have existing partnerships.
**Phase 2 — Tier 1 Supplier Certification:** Satellite-verified soil health as requirement for "Clean Sourced" ingredient tiers.
**Phase 3 — Automated Remediation:** AI-detected contamination alerts linked to local agronomists for targeted soil remediation.

---

## 9. Dependencies & External Services

| Service | Required? | Purpose | Cost |
|---------|-----------|---------|------|
| Google Earth Engine | Yes | All satellite data processing | Free (research) |
| NASA POWER API | Yes | Weather/agroclimatology | Free (public) |
| Planet Labs API | Optional | High-res 3m imagery | $0.50–5/km² |
| Nominatim (OSM) | Optional | Location geocoding | Free (public) |
| PRISMA / EnMAP | Future | Hyperspectral data | Free (research proposal) |
| Pixxel | Future | Commercial hyperspectral | $5–50/km² est. |

---

## 10. Success Metrics

### Phase 1–5 (Implementation)
- All 13+ spectral indices computed and visualized (9 vegetation + 4 metal-sensitive)
- 5 biophysical variables (LAI, CCC, CWC, FAPAR, FCOVER) operational
- 8 heavy metals with per-metal risk surfaces
- 12 crop profiles with configurable detection parameters
- Multi-sensor data collection (S2, S1, Weather, MODIS) working end-to-end
- Remediation recommendations generated per contamination type
- NPK nutrient status estimation functional
- Session persistence with disk caching
- 10-tab UI with analytics sub-tabs
- Zone classification with 8 zones (including contamination)
- Disease risk engine with 3 pathogens
- Data export (CSV, GeoTIFF) for all datasets
- Deployment successful on Render

### Phase 6 (Future)
- ML models achieving published R² targets per metal
- Hyperspectral data integration operational
- ESG dashboard with automated reporting
- REST API with documented endpoints

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Indirect detection specificity** | Multiple stressors produce similar symptoms | Multi-index ensemble + crop profiles + temporal patterns |
| **Mercury low R²** (0.45–0.65) | Unreliable Hg screening | Flag as "low confidence"; emphasize in UI; recommend lab confirmation |
| **Vegetation cover blocks direct detection** | Can't see soil under canopy | Multi-temporal bare-soil composites from fallow periods; indirect mode |
| **Tropical cloud cover** | Sparse optical time-series | S1 SAR fills gaps; 3-layer cloud mask; MODIS thermal |
| **GEE quota limits** | Processing bottlenecks | Aggressive caching; reduce scale where possible |
| **MODIS 1km resolution** | Inaccurate thermal per-plot | Use as estate-scale indicator; document caveat |
| **Model region-specificity** | Models don't transfer between regions | Crop profiles as first step; ground-truth calibration for ML phase |
| **Data continuity** (PRISMA past design life) | Gap before CHIME/SBG | Multi-source architecture; Pixxel/Wyvern as commercial bridge |
| **Regulatory positioning** | Mistaken as replacement for lab testing | Clear disclaimer: "screening tool, not a certified measurement" |

---

## 12. References

- White Paper: "Satellite-Based Hyperspectral Detection of Heavy Metals in Agricultural Soils" v1.0 (April 2026)
- Technical Brief: "Orbital Hyperspectral Monitoring of Global Agricultural Contaminants" (April 2026)
- AgroSat reference implementation (github.com/yasantha-nai/agrosat)
- ESA SNAP S2 Biophysical Processor ATBD v2.1
- Guyot & Baret 1988 (REIP formula)
- RUSLE (Revised Universal Soil Loss Equation)
- NASA POWER API documentation
- Google Earth Engine API documentation
- 30 academic/industry references cited in white paper (2017–2025)
