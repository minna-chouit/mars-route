# Mars Marswalk Planner

An interactive slope-aware route planner for Martian surface traversal, built on a HiRISE Digital Terrain Model of Gale Crater.

## Data Credits

| Data | Source |
|---|---|
| HiRISE ortho image (`PSP_001752_1750_RED_C_01_ORTHO.JP2`) | NASA/JPL/University of Arizona |
| HiRISE DTM (`DTEEC_001488_1750_001752_1750_U02.IMG`) | NASA/JPL/University of Arizona & USGS Astrogeology Science Center |
| Coverage | Gale Crater, Mars (HiRISE stereo pair, ~1 m/px) |

Data accessed via the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/) and [HiRISE Data Archive](https://www.uahirise.org/).

## Pipeline

```
run_all.py
 ├─ step1_crop.py   → Crops 2×2 km window from DTM + ortho
 ├─ step2_slope.py  → Computes slope, hillshade, slope-class PNG
 └─ step3_route.py  → Least-cost route via skimage.graph
```

All thresholds and coordinates are in **`config.py`**.

## Quick Start

```bash
# 1. Run pipeline (takes ~1-2 min for 2000×2000 routing)
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
| `TARGET` | (1700, 1700) | Route target in crop coordinates |

> **⚠️ Disclaimer:** Slope thresholds are **assumptions** adapted from lunar/Mars analogue studies (Wilcox et al. 2007, NASA-TM-2007-214593). They must be validated against specific hardware specifications (rover chassis, EVA suit) before operational use.

## Leaflet Map Features

- **Layer control**: toggle ortho / hillshade / slope-class / route
- **Click** anywhere on the map to read `row, col` coordinates (copy into `config.py` to move start/target)
- **Side panel**: route length (m), max slope (°), % of route in caution zone

## File Outputs

```
web/
  index.html         ← Leaflet map
  ortho.png          ← 8-bit HiRISE orthoimage crop
  hillshade.png      ← Computed hillshade
  slope_class.png    ← RGBA slope-class overlay
  route.json         ← Route waypoints + stats
  route_debug.png    ← Quick-look debug image
  crop_dtm.tif       ← Float32 DTM crop (GeoTIFF)
  dtm_crop.npy       ← DTM as numpy array
  slope.npy          ← Slope array (degrees)
  ortho_crop.npy     ← Ortho as numpy array
  preview_dtm.png    ← DTM preview (choose crop)
  preview_ortho.png  ← Ortho preview (choose crop)
```
