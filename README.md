# Mars Marswalk Planner

A web-based, slope-aware and rock-hazard traverse planner for Martian surface exploration in Gale Crater, Mars. The application integrates high-resolution terrain elevation (HiRISE DTM), context and ultra-high-resolution surface imagery (HiRISE ortho), remote global thermal inertia (THEMIS), and automated rock hazard detection to plan a continuous, safety-constrained loop traverse with a live Marswalk budget and elevation profile.

## How to Run

1. **Download data**:
   Download the two primary data files into the `data/` directory:
   - `DTEEC_001488_1750_001752_1750_U02.IMG` (HiRISE DTM, ~1 m/px, PDS3 float32)
   - `PSP_001752_1750_RED_C_01_ORTHO.JP2` (HiRISE context ortho, 8-bit image)
   
   *(Optional high-resolution rock hazard source: `PSP_001752_1750_RED_A_01_ORTHO.JP2` at 0.25 m/px for boulder extraction).*

2. **Run the processing pipeline**:
   ```bash
   python run_all.py
   ```
   This executes:
   - `step1_crop.py`: Extracts the 2 km × 2 km crop window from the DTM and context ortho.
   - `step2_slope.py`: Calculates terrain slope, hillshade, and classified slope overlay.
   - `step4_themis.py`: Queries the global THEMIS mosaic remotely via `/vsicurl/` for the crop window and resamples to grid.
   - `step5_boulders.py`: Extracts rock hazards from 0.25 m/px ortho via windowed shadow blob analysis.
   - `step3_route.py`: Computes least-cost loop routes across safety thresholds with boulder density penalties.

3. **Serve the web map locally**:
   ```bash
   cd web
   python -m http.server 8000 --bind 127.0.0.1
   ```
   Open `http://127.0.0.1:8000` in your web browser.

## Layers

All map layers are rendered using Leaflet (`L.CRS.Simple`) with coordinate flip correction and toggleable layer controls:

- **Base canvas**: Neutral dark background.
- **Ortho image**: 8-bit HiRISE visible surface reflectance (`PSP_001752_1750_RED_C_01_ORTHO`, unchecked by default).
- **Hillshade**: Shaded relief computed from DTM gradients with solar azimuth 315° (NW) and altitude 45° (unchecked by default).
- **Slope classes**: Semi-transparent colored risk overlay (checked by default):
  - Green: Safe (≤15°)
  - Yellow: Caution (15°–20°)
  - Red: No-Go (>20° with a 1-pixel morphological dilation safety buffer)
- **Thermal Inertia (THEMIS 50%)**: Quantitative thermal inertia from the 100 m/px THEMIS global mosaic upsampled smoothly via bilinear interpolation, displayed at 50% opacity with a plasma color scale (unchecked by default).
- **Possible boulders**: Detected candidate rocks (2.0–20 m² shadow blobs, eccentricity <0.9) rendered via Leaflet canvas as small magenta circle markers (radius 3, fill #ff2bd6, 1px white outline, unchecked by default).
- **Traverse Route**: Continuous multi-leg loop polyline connecting the habitat base and all science stops, colored orange (checked by default).
- **Science Stops & Base**: Color-coded markers for the Landing Site and science waypoints, with interactive popups showing target descriptions, elevations, slopes, and local thermal inertia (checked by default).

## Safety Assumptions

- **Lunar slope thresholds are assumptions**: The 20° hard cutoff is derived from NASA Human Landing System (HLS) suited-walking requirements (NTRS 20210026296), and the 15° preferred limit comes from a lunar south-pole traverse study (phys.org, Oct 2023). Both are lunar figures adopted as engineering assumptions for Martian suited EVA traverse planning and must be re-validated against specific rover and EVA suit mobility specifications.
- **Walking speed is an assumption**: The walking time budget defaults to an assumed suited walking speed, not validated (default 0.8 m/s, user-editable in the panel).
- **Science stops are illustrative**: The science stops (*Landing Site / Base*, *Hollow*, *Layered outcrop*, and *Boxwork ridge base*) are illustrative targets selected to demonstrate multi-stop routing through representative geological features in Gale Crater.
- **Rock hazards are unvalidated**: Boulders are identified through classical morphological processing of high-contrast optical shadows from 0.25 m/px imagery; they represent potential boulder obstacles and are not field-validated ground truth.

## Data Credits

- **HiRISE**: NASA/JPL/University of Arizona
- **DTM**: USGS Astrogeology
- **THEMIS**: NASA/JPL/ASU

---

Code was developed with AI assistance (Google Antigravity/Gemini and Claude).
