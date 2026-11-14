# Mars Marswalk Planner

An interactive slope-aware and thermal-inertia route planner for Martian surface traversal, built on a HiRISE Digital Terrain Model and THEMIS Global Thermal Inertia mosaic of Gale Crater.

## Data Credits

| Data | Source |
|---|---|
| HiRISE ortho image (`PSP_001752_1750_RED_C_01_ORTHO.JP2`) | NASA/JPL/University of Arizona |
| HiRISE DTM (`DTEEC_001488_1750_001752_1750_U02.IMG`) | NASA/JPL/University of Arizona & USGS Astrogeology Science Center |
| Thermal Inertia Mosaic (100 m/px) | NASA/JPL/Arizona State University (THEMIS), mosaic by USGS Astrogeology |
| Coverage | Gale Crater, Mars (~1 m/px DTM/ortho, 100 m/px THEMIS) |

Data accessed via the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/), [HiRISE Data Archive](https://www.uahirise.org/), and [USGS Astropedia](https://astrogeology.usgs.gov/).

## Pipeline

```
run_all.py
 ├─ step1_crop.py   → Crops 2×2 km window from DTM + ortho
 ├─ step2_slope.py  → Computes slope, hillshade, slope-class PNG
 ├─ step3_route.py  → Least-cost route via skimage.graph (dilated impassable no-go)
 └─ step4_themis.py → Fetches remote THEMIS window via /vsicurl/, resamples & extracts TI
```

All thresholds and coordinates are in **`config.py`**.

## Quick Start

```bash
# 1. Run pipeline
python run_all.py

# 2. Serve the map
cd web
python -m http.server 8000

# 3. Open http://localhost:8000
```

## Configuration (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CROP_ROW`, `CROP_COL` | 2000, 2000 | Top-left of 2 km crop |
| `CROP_SIZE` | 1977 | Pixels (~2 km at 1.0117 m/px) |
| `SLOPE_SAFE` | 15° | ≤15° → safe (green) |
| `SLOPE_CAUTION` | 20° | 15–20° → caution (yellow) |
| `START` | (200, 200) | Route start in crop coordinates |
| `THEMIS_TI_URL` | S3 Astropedia URL | Global THEMIS 100 m/px TI mosaic (.cub) |

> **⚠️ Disclaimer:** Slope thresholds are **lunar figures used as assumptions for Mars**:
> - 20° hard limit: NASA HLS suited-walking requirement (NTRS 20210026296)
> - 15° preferred cap: lunar south-pole traverse study (phys.org, Oct 2023)
> Validate against rover/suit hardware specs before mission use.

## Leaflet Map Features

- **Layer control**: toggle Ortho / Hillshade / Slope classes / Thermal Inertia (THEMIS) / Route / Markers
- **Click** anywhere on the map to read `row, col` coordinates
- **Side panel**: route length (m), max slope (°), % caution, waypoints, Start Thermal Inertia (tiu), Target Thermal Inertia (tiu)
- **Thermal Inertia legend**: plasma gradient bar showing min–max TI in $J\ m^{-2}\ K^{-1}\ s^{-1/2}$
