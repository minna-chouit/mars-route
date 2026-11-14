"""
Step 4: Fetch remote THEMIS Thermal Inertia layer via /vsicurl/.
Reads only the crop's lat/lon window from the global 100 m/px mosaic.
Resamples smoothly to crop grid (1977x1977).
Generates web/themis.png (colored overlay) and records TI at start/target in web/route.json.
"""
import os, sys, json
import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from PIL import Image
from matplotlib import colormaps

sys.path.insert(0, os.path.dirname(__file__))
from config import THEMIS_TI_URL

os.makedirs("web", exist_ok=True)

# 1. Coordinate translation from HiRISE DTM crop
# HiRISE DTM uses central meridian 137.3, center lat 0, Mars sphere R=3396190.
# THEMIS uses equirectangular with central meridian 0, same radius R.
R_DEG = 3396190.0 * np.pi / 180.0  # ~59274.6975 m / deg

print("Reading DTM crop bounds...")
with rasterio.open("web/crop_dtm.tif") as dtm:
    b = dtm.bounds
    H, W = dtm.height, dtm.width
    lon_min = 137.3 + b.left / R_DEG
    lon_max = 137.3 + b.right / R_DEG
    lat_min = b.bottom / R_DEG
    lat_max = b.top / R_DEG

print(f"Crop bounding box: Lon [{lon_min:.5f}°, {lon_max:.5f}°], Lat [{lat_min:.5f}°, {lat_max:.5f}°]")

themis_left   = lon_min * R_DEG
themis_right  = lon_max * R_DEG
themis_bottom = lat_min * R_DEG
themis_top    = lat_max * R_DEG

# 2. Fetch window via /vsicurl/
print("Reading remote THEMIS window via /vsicurl/...")
vsicurl_url = "/vsicurl/" + THEMIS_TI_URL

with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    with rasterio.open(vsicurl_url) as src:
        win = rasterio.windows.from_bounds(themis_left, themis_bottom, themis_right, themis_top, src.transform)
        print(f"THEMIS native window: {win.width:.1f}x{win.height:.1f} px -> upsampling smoothly to {W}x{H}...")
        ti = src.read(1, window=win, out_shape=(H, W), resampling=Resampling.bilinear)

# Mask NoData (< -1e30 or non-positive)
valid_mask = (ti > 0) & (ti > -1e30)
np.save("web/themis.npy", ti)

ti_valid = ti[valid_mask]
ti_min = float(np.percentile(ti_valid, 1)) if len(ti_valid) else 0.0
ti_max = float(np.percentile(ti_valid, 99)) if len(ti_valid) else 1000.0
ti_mean = float(np.mean(ti_valid)) if len(ti_valid) else 0.0

print(f"Thermal Inertia: min={ti_min:.1f}, max={ti_max:.1f}, mean={ti_mean:.1f} J m^-2 K^-1 s^-1/2")

# 3. Create colored overlay RGBA PNG
# Using 'plasma' colormap
cmap = colormaps["plasma"]
norm_ti = np.clip((ti - ti_min) / max(1.0, (ti_max - ti_min)), 0.0, 1.0)
rgba = (cmap(norm_ti) * 255).astype(np.uint8)
rgba[..., 3] = np.where(valid_mask, 160, 0)  # ~63% opacity where valid, transparent otherwise

Image.fromarray(rgba, "RGBA").save("web/themis.png")
print("Saved web/themis.png")

# 4. Update web/route.json with thermal inertia values at start & target
route_json_path = "web/route.json"
if os.path.exists(route_json_path):
    with open(route_json_path, "r") as f:
        data = json.load(f)

    start = data.get("start", [200, 200])
    target = data.get("target", [1714, 1708])

    ti_start = float(ti[start[0], start[1]]) if valid_mask[start[0], start[1]] else None
    ti_target = float(ti[target[0], target[1]]) if valid_mask[target[0], target[1]] else None

    data["ti_min"] = round(ti_min, 1)
    data["ti_max"] = round(ti_max, 1)
    data["ti_mean"] = round(ti_mean, 1)
    data["ti_start"] = round(ti_start, 1) if ti_start is not None else None
    data["ti_target"] = round(ti_target, 1) if ti_target is not None else None

    with open(route_json_path, "w") as f:
        json.dump(data, f)
    print(f"Updated {route_json_path} (Start TI: {data['ti_start']} tiu, Target TI: {data['ti_target']} tiu)")

print("Done step 4.")
