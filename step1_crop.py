"""
Step 1: Crop a 2×2 km window from the DTM and ortho.
Saves:
  web/crop_dtm.tif   — float32 elevation crop
  web/crop_ortho.tif — 8-bit ortho crop
  web/preview_dtm.png
  web/preview_ortho.png
"""
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DTM_PATH, ORTHO_PATH,
    CROP_ROW, CROP_COL, CROP_SIZE,
    PIXEL_SIZE_M
)

os.makedirs("web", exist_ok=True)

NODATA = -3.4e38

def open_dtm(path):
    """Try rasterio auto-detect, then PDS driver."""
    try:
        ds = rasterio.open(path)
        arr = ds.read(1)
        if arr.max() == arr.min():
            raise ValueError("flat read — trying PDS driver")
        return ds, arr
    except Exception as e:
        print(f"  Auto-detect failed ({e}), trying PDS driver...")
        ds = rasterio.open(path, driver="PDS")
        return ds, ds.read(1)

print("Opening DTM …")
ds_dtm, full_dtm = open_dtm(DTM_PATH)
print(f"  DTM shape: {full_dtm.shape}, dtype: {full_dtm.dtype}")
print(f"  DTM range (excluding nodata): ", end="")

# Mask nodata
valid = full_dtm[full_dtm > NODATA + 1e30]
print(f"{valid.min():.1f} – {valid.max():.1f} m")

# ── Auto-pick window if defaults land on nodata ──────────────────────────────
row0, col0 = CROP_ROW, CROP_COL
sz = CROP_SIZE

def window_ok(arr, r, c, s, nodata=NODATA):
    patch = arr[r:r+s, c:c+s]
    bad = np.sum(patch <= nodata + 1e30)
    return bad / patch.size < 0.01   # <1 % nodata

if not window_ok(full_dtm, row0, col0, sz):
    print("  Default crop has nodata — scanning for a clean window …")
    H, W = full_dtm.shape
    found = False
    for r in range(0, H - sz, sz // 4):
        for c in range(0, W - sz, sz // 4):
            if window_ok(full_dtm, r, c, sz):
                # prefer area with real relief (std > 5 m)
                patch = full_dtm[r:r+sz, c:c+sz]
                patch = np.where(patch > NODATA + 1e30, patch, np.nan)
                if np.nanstd(patch) > 5.0:
                    row0, col0 = r, c
                    found = True
                    break
        if found:
            break
    if not found:
        print("  WARNING: could not find a clean window with high relief — using defaults")
        row0, col0 = CROP_ROW, CROP_COL

print(f"  Crop window: row={row0}, col={col0}, size={sz}")

# Crop DTM
dtm_crop = full_dtm[row0:row0+sz, col0:col0+sz].copy()
dtm_crop = np.where(dtm_crop > NODATA + 1e30, dtm_crop, np.nan)
print(f"  DTM crop elevation: {np.nanmin(dtm_crop):.1f} – {np.nanmax(dtm_crop):.1f} m, std={np.nanstd(dtm_crop):.2f} m")

# Save DTM crop as GeoTIFF (keeps transform for reference)
win = Window(col0, row0, sz, sz)
crop_transform = rasterio.windows.transform(win, ds_dtm.transform) if ds_dtm.transform else None
with rasterio.open(
    "web/crop_dtm.tif", "w",
    driver="GTiff", height=sz, width=sz, count=1,
    dtype="float32", crs=ds_dtm.crs,
    transform=crop_transform,
    nodata=np.nan
) as dst:
    dst.write(dtm_crop.astype("float32"), 1)
print("  Saved web/crop_dtm.tif")

# DTM preview: hillshade-style (simple normalised)
dtm_vis = dtm_crop.copy()
lo, hi = np.nanpercentile(dtm_vis, 2), np.nanpercentile(dtm_vis, 98)
dtm_vis = np.clip((dtm_vis - lo) / (hi - lo) * 255, 0, 255)
dtm_vis = np.where(np.isnan(dtm_vis), 0, dtm_vis).astype("uint8")
Image.fromarray(dtm_vis).save("web/preview_dtm.png")
print("  Saved web/preview_dtm.png")

# ── Ortho ────────────────────────────────────────────────────────────────────
print("Opening ortho …")
with rasterio.open(ORTHO_PATH) as ds_ortho:
    print(f"  Ortho size: {ds_ortho.width}×{ds_ortho.height}, bands: {ds_ortho.count}")
    # ortho may differ in shape; scale crop window
    sx = ds_ortho.width  / ds_dtm.width
    sy = ds_ortho.height / ds_dtm.height
    or0  = int(row0 * sy);  oc0  = int(col0 * sx)
    osz  = int(sz   * sy);  ocs  = int(sz   * sx)
    print(f"  Ortho crop: row={or0}, col={oc0}, h={osz}, w={ocs}")
    win_o = Window(oc0, or0, ocs, osz)
    ortho_crop = ds_ortho.read(1, window=win_o)

# Resize ortho to match DTM crop if needed
if ortho_crop.shape != (sz, sz):
    from PIL import Image as PILImage
    pil_o = PILImage.fromarray(ortho_crop).resize((sz, sz), PILImage.BILINEAR)
    ortho_crop = np.array(pil_o)

# Stretch to full 0-255
lo, hi = np.percentile(ortho_crop, 1), np.percentile(ortho_crop, 99)
ortho_vis = np.clip((ortho_crop.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
Image.fromarray(ortho_vis).save("web/preview_ortho.png")
print("  Saved web/preview_ortho.png")

# Save raw ortho crop as npy for later steps
np.save("web/ortho_crop.npy", ortho_vis)
np.save("web/dtm_crop.npy",   dtm_crop.astype("float32"))
print(f"\nActual crop: CROP_ROW={row0}, CROP_COL={col0}  (update config.py if needed)")
print("Done — check web/preview_dtm.png and web/preview_ortho.png")
