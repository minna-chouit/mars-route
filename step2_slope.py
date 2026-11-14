"""
Step 2: Slope, hillshade, and slope-class overlay.
Reads:  web/dtm_crop.npy
Writes:
  web/slope.npy          — slope in degrees (float32)
  web/slope_class.png    — RGBA PNG: green/yellow/red + transparent
  web/hillshade.png      — 8-bit hillshade
  web/ortho.png          — ortho (copy of preview_ortho.png, renamed for Leaflet)
"""
import numpy as np
from PIL import Image
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import PIXEL_SIZE_M, SLOPE_SAFE, SLOPE_CAUTION, HILLSHADE_AZIMUTH, HILLSHADE_ALTITUDE

os.makedirs("web", exist_ok=True)

# ── Load DTM crop ─────────────────────────────────────────────────────────────
dtm = np.load("web/dtm_crop.npy").astype("float64")
# Fill NaN with nearest-valid for gradient (avoid edge artefacts)
from scipy.ndimage import generic_filter
nan_mask = np.isnan(dtm)
if nan_mask.any():
    # Simple fill: replace NaN with local mean
    filled = dtm.copy()
    filled[nan_mask] = np.nanmean(dtm)
    dtm = filled

# ── Slope ─────────────────────────────────────────────────────────────────────
# np.gradient returns [dy, dx] i.e. [row-direction, col-direction]
dy, dx = np.gradient(dtm, PIXEL_SIZE_M)
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad).astype("float32")

np.save("web/slope.npy", slope_deg)
print(f"Slope: min={slope_deg.min():.2f}°  max={slope_deg.max():.2f}°  mean={slope_deg.mean():.2f}°")
print(f"  Safe (≤{SLOPE_SAFE}°):    {(slope_deg <= SLOPE_SAFE).mean()*100:.1f}%")
print(f"  Caution ({SLOPE_SAFE}–{SLOPE_CAUTION}°):  {((slope_deg > SLOPE_SAFE) & (slope_deg <= SLOPE_CAUTION)).mean()*100:.1f}%")
print(f"  No-go (>{SLOPE_CAUTION}°):   {(slope_deg > SLOPE_CAUTION).mean()*100:.1f}%")

# ── Slope-class RGBA PNG ──────────────────────────────────────────────────────
H, W = slope_deg.shape
rgba = np.zeros((H, W, 4), dtype="uint8")

safe    = slope_deg <= SLOPE_SAFE
caution = (slope_deg > SLOPE_SAFE) & (slope_deg <= SLOPE_CAUTION)
nogo    = slope_deg > SLOPE_CAUTION

# Green (safe) — semi-transparent
rgba[safe]    = [  0, 180,   0, 100]
# Yellow (caution)
rgba[caution] = [255, 220,   0, 140]
# Red (no-go)
rgba[nogo]    = [220,   0,   0, 160]

Image.fromarray(rgba, "RGBA").save("web/slope_class.png")
print("Saved web/slope_class.png")

# ── Hillshade ─────────────────────────────────────────────────────────────────
# Standard hillshade formula
az_rad  = np.radians(360.0 - HILLSHADE_AZIMUTH + 90.0)  # convert to math convention
alt_rad = np.radians(HILLSHADE_ALTITUDE)

# Surface normals from gradient
slope_rad_arr = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dx, dy)   # aspect (north=0, clockwise)

hs = (
    np.sin(alt_rad) * np.cos(slope_rad_arr)
  + np.cos(alt_rad) * np.sin(slope_rad_arr) * np.cos(az_rad - aspect)
)
hs = np.clip(hs, 0, 1)
hs_uint8 = (hs * 255).astype("uint8")
Image.fromarray(hs_uint8).save("web/hillshade.png")
print("Saved web/hillshade.png")

# Copy ortho
import shutil
if os.path.exists("web/preview_ortho.png"):
    shutil.copy("web/preview_ortho.png", "web/ortho.png")
    print("Saved web/ortho.png")

print("Done.")
