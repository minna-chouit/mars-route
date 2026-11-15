"""
Mars Marswalk Planner — Configuration
All thresholds and coordinates live here.

ASSUMPTION NOTE: Slope thresholds are lunar figures applied as assumptions for Mars.
  • 20° hard limit  → NASA HLS suited-walking requirement (NTRS 20210026296)
  • 15° preferred cap → lunar south-pole traverse study (phys.org, Oct 2023)
They must be validated against specific hardware specs before operational use.
"""

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DTM_PATH   = "data/DTEEC_001488_1750_001752_1750_U02.IMG"
ORTHO_PATH = "data/PSP_001752_1750_RED_C_01_ORTHO.JP2"
BOULDER_ORTHO_PATH = "data/PSP_001752_1750_RED_A_01_ORTHO.JP2"
THEMIS_TI_URL = (
    "https://asc-astropedia.s3.us-west-2.amazonaws.com/Mars/Odyssey/"
    "THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/"
    "THEMIS_TI_Mosaic_Quant_30S120E_100mpp.cub"
)

# ---------------------------------------------------------------------------
# Pixel size (metres) — from PDS label
# ---------------------------------------------------------------------------
PIXEL_SIZE_M = 1.0117  # metres per pixel

# ---------------------------------------------------------------------------
# Crop window: top-left pixel (row, col) + size in pixels
# Change these to explore a different 2 km × 2 km area.
# ---------------------------------------------------------------------------
CROP_ROW  = 2000   # top-left row in the full DTM
CROP_COL  = 2000   # top-left col in the full DTM
CROP_SIZE = 1977   # ~2 km / 1.0117 m/px ≈ 1977 px

# ---------------------------------------------------------------------------
# Slope thresholds (degrees)
# ASSUMPTION: lunar figures used as proxies for Mars (see note above).
# ---------------------------------------------------------------------------
SLOPE_SAFE    = 15.0   # ≤ 15° → safe (green)   [lunar south-pole traverse cap]
SLOPE_CAUTION = 20.0   # 15–20° → caution (yellow)
# > SLOPE_CAUTION → no-go (hard block after dilation)
#   source: NASA HLS suited-walking requirement (NTRS 20210026296)

# ---------------------------------------------------------------------------
# Hillshade sun direction
# ---------------------------------------------------------------------------
HILLSHADE_AZIMUTH  = 315.0  # degrees, NW sun
HILLSHADE_ALTITUDE =  45.0  # degrees above horizon

# ---------------------------------------------------------------------------
# Route start (row, col) within the CROP — top-left quadrant
# ---------------------------------------------------------------------------
START = (200, 200)   # (row, col) in crop coordinates

# ---------------------------------------------------------------------------
# Cost function & Rock Hazard parameters
# ---------------------------------------------------------------------------
# Passable cells: cost = 1 + 10*(slope/15)**2 + BOULDER_PENALTY_SCALE * boulder_density
# No-go cells (after 1-px dilation): cost = NOGO_COST (truly impassable)
NOGO_COST = 1e9
NOGO_DILATE_PX = 1
BOULDER_PENALTY_SCALE = 1.0  # penalty per 20 m cell boulder count
BOULDER_MIN_AREA_M2 = 2.0
BOULDER_MAX_AREA_M2 = 20.0
BOULDER_MAX_ECCENTRICITY = 0.9
