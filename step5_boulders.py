"""
Step 5: Rock hazard (boulder) detection using classical image processing on high-res RED_A ortho (0.25 m/px).
Windowed read matching DTM crop bounds.
Gaussian smooth -> adaptive threshold for dark shadow blobs -> morphological opening -> label & regionprops.
Filters: 0.5 - 20 m² area, eccentricity < 0.9.
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
from config import PIXEL_SIZE_M

os.makedirs("web", exist_ok=True)

# 1. Read window matching DTM crop bounds
print("Opening crop_dtm.tif to get bounds...")
with rasterio.open("web/crop_dtm.tif") as dtm:
    b = dtm.bounds
    H_dtm, W_dtm = dtm.height, dtm.width

print(f"Reading window from PSP_001752_1750_RED_A_01_ORTHO.JP2...")
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
# Shadows are significantly darker than local background
shadow_mask = smoothed < (local_mean - 20.0)
opened = binary_opening(shadow_mask, structure=np.ones((2, 2), dtype=bool))
print(f"Filtering complete in {time.time()-t1:.1f}s")

print("Labeling connected components & filtering rock blobs...")
t2 = time.time()
lbl = label(opened)
props = regionprops(lbl)

px_res = 0.2529285
px_area = px_res ** 2

boulders = []
# Grid mapping factor from RED_A (7908) to DTM (1977)
scale_r = H_dtm / float(H_a)
scale_c = W_dtm / float(W_a)

for p in props:
    area_m2 = p.area * px_area
    if 0.5 <= area_m2 <= 20.0 and p.eccentricity < 0.9:
        r_a, c_a = p.centroid
        r_dtm = r_a * scale_r
        c_dtm = c_a * scale_c
        # Equivalent diameter in meters
        eq_diam = getattr(p, "equivalent_diameter_area", getattr(p, "equivalent_diameter", 0))
        diam_m = eq_diam * px_res
        boulders.append({
            "r": round(float(r_dtm), 1),
            "c": round(float(c_dtm), 1),
            "d": round(float(diam_m), 2),
            "a": round(float(area_m2), 2)
        })

print(f"Extracted {len(boulders)} valid boulder candidates in {time.time()-t2:.1f}s")

# 3. Compute 20 m cell boulder density map (1977 x 1977)
print("Computing 20 m cell boulder density grid...")
boulder_grid = np.zeros((H_dtm, W_dtm), dtype=np.float32)
for bld in boulders:
    r = int(np.clip(round(bld["r"]), 0, H_dtm - 1))
    c = int(np.clip(round(bld["c"]), 0, W_dtm - 1))
    boulder_grid[r, c] += 1.0

# 20 m cell size in DTM pixels
cell_size_px = int(round(20.0 / PIXEL_SIZE_M))  # ~20 px
# Uniform filter gives the mean count per pixel in a (20x20) window.
# Multiplying by (20*20) gives the total count inside that 20m x 20m cell.
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
print("Saved web/boulders.json")

# 5. Generate transparent overlay PNG with small circles
print("Generating web/boulders.png overlay...")
img = Image.new("RGBA", (W_dtm, H_dtm), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw each boulder as a small circle (radius 1.5 - 2.5 px depending on diameter)
for bld in boulders:
    r, c = bld["r"], bld["c"]
    rad = max(1.2, min(3.5, bld["d"] / 2.0))
    # Cyan circle with subtle outline
    draw.ellipse([c - rad, r - rad, c + rad, r + rad],
                 fill=(0, 229, 255, 200),
                 outline=(0, 100, 180, 240))

img.save("web/boulders.png")
print("Saved web/boulders.png")
print("Done step 5.")
