"""
Step 5: Rock hazard (boulder) detection using classical image processing on high-res RED_A ortho (0.25 m/px).
Windowed read matching DTM crop bounds.
Gaussian smooth -> adaptive threshold for dark shadow blobs -> morphological opening -> label & regionprops.
Filters: 2.0 - 20 m² area, eccentricity < 0.9.
Saves:
  web/boulders.json          - list of detected boulders with centroid (r, c), diameter, area
  web/boulder_density.npy    - 20 m cell boulder density grid (1977x1977)
  web/boulders.png           - transparent overlay with small circles
"""
import os, sys, time, json
import numpy as np
import rasterio
import rasterio.windows
from scipy.ndimage import gaussian_filter, uniform_filter, binary_opening
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PIXEL_SIZE_M, BOULDER_MIN_AREA_M2, BOULDER_MAX_AREA_M2, BOULDER_MAX_ECCENTRICITY
)

os.makedirs("web", exist_ok=True)

# 1. Read window matching DTM crop bounds
print("Opening crop_dtm.tif to get bounds...")
with rasterio.open("web/crop_dtm.tif") as dtm:
    b = dtm.bounds
    H_dtm, W_dtm = dtm.height, dtm.width

print("Reading window from PSP_001752_1750_RED_A_01_ORTHO.JP2...")
t0 = time.time()
with rasterio.open("data/PSP_001752_1750_RED_A_01_ORTHO.JP2") as ds:
    win = rasterio.windows.from_bounds(b.left, b.bottom, b.right, b.top, ds.transform)
    ortho_a = ds.read(1, window=win)

H_a, W_a = ortho_a.shape
print(f"Read window: {W_a}x{H_a} px in {time.time()-t0:.1f}s")

# 2. Classical Image Processing
print("Applying Gaussian smoothing & adaptive thresholding...")
t1 = time.time()
smoothed = gaussian_filter(ortho_a.astype(np.float32), sigma=1.0)
local_mean = uniform_filter(smoothed, size=35)
shadow_mask = smoothed < (local_mean - 20.0)
opened = binary_opening(shadow_mask, structure=np.ones((2, 2), dtype=bool))
print(f"Filtering complete in {time.time()-t1:.1f}s")

print(f"Labeling connected components & filtering rock blobs ({BOULDER_MIN_AREA_M2} - {BOULDER_MAX_AREA_M2} m²)...")
t2 = time.time()
lbl = label(opened)
props = regionprops(lbl)

px_res = 0.2529285
px_area = px_res ** 2

boulders = []
scale_r = H_dtm / float(H_a)
scale_c = W_dtm / float(W_a)

for p in props:
    area_m2 = p.area * px_area
    if BOULDER_MIN_AREA_M2 <= area_m2 <= BOULDER_MAX_AREA_M2 and p.eccentricity < BOULDER_MAX_ECCENTRICITY:
        r_a, c_a = p.centroid
        r_dtm = r_a * scale_r
        c_dtm = c_a * scale_c
        eq_diam = getattr(p, "equivalent_diameter_area", getattr(p, "equivalent_diameter", 0))
        diam_m = eq_diam * px_res
        boulders.append({
            "r": round(float(r_dtm), 1),
            "c": round(float(c_dtm), 1),
            "d": round(float(diam_m), 2),
            "a": round(float(area_m2), 2)
        })

print(f"Extracted {len(boulders)} valid boulder candidates (min area {BOULDER_MIN_AREA_M2} m²) in {time.time()-t2:.1f}s")

# 3. Compute 20 m cell boulder density map (1977 x 1977)
print("Computing 20 m cell boulder density grid...")
boulder_grid = np.zeros((H_dtm, W_dtm), dtype=np.float32)
for bld in boulders:
    r = int(np.clip(round(bld["r"]), 0, H_dtm - 1))
    c = int(np.clip(round(bld["c"]), 0, W_dtm - 1))
    boulder_grid[r, c] += 1.0

cell_size_px = int(round(20.0 / PIXEL_SIZE_M))  # ~20 px
density_20m = uniform_filter(boulder_grid, size=cell_size_px) * (cell_size_px * cell_size_px)
np.save("web/boulder_density.npy", density_20m)
print(f"Boulder density (count / 20m cell): min={density_20m.min():.1f}, max={density_20m.max():.1f}, mean={density_20m.mean():.2f}")

# 4. Save boulders.json
with open("web/boulders.json", "w") as f:
    json.dump({
        "count": len(boulders),
        "boulders": boulders,
        "label": "possible boulders from shadows, not validated"
    }, f)
print(f"Saved web/boulders.json (total: {len(boulders)})")

# 5. Generate transparent overlay PNG with magenta circles
print("Generating web/boulders.png overlay...")
img = Image.new("RGBA", (W_dtm, H_dtm), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Magenta #ff2bd6 with white outline
for bld in boulders:
    r, c = bld["r"], bld["c"]
    rad = 3.0
    draw.ellipse([c - rad - 1, r - rad - 1, c + rad + 1, r + rad + 1], fill=(255, 255, 255, 255))
    draw.ellipse([c - rad, r - rad, c + rad, r + rad], fill=(255, 43, 214, 230))

img.save("web/boulders.png")
print("Saved web/boulders.png")
print("Done step 5.")
