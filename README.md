# Orbital Soil Intelligence Demo

Heavy-metal detection demo web app inspired by the `Orbital_Soil_Intelligence` deck and aligned with AgroSat workflows.

## Focus
- Cadmium (Cd)
- Mercury (Hg)
- Lithium (Li)
- Tellurium (Te)

## Detection framework
- Direct (bare-soil host-phase signatures)
- Indirect (crop/canopy stress signatures)
- Bimodal fusion for year-round coverage

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required env vars
- `GEE_PROJECT`
- `GEE_SERVICE_ACCOUNT_JSON`
- `PL_API_KEY` (optional)

## Render
This repo includes `render.yaml` for Blueprint deployment.
