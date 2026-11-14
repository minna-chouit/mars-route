"""
Step 3 (v2): Truly-impassable no-go, smooth quadratic cost, component snapping,
             ridge-target selection, and full route validation.

Reads:  web/slope.npy
Writes:
  web/route.json  — {route, length_m, max_slope_deg, pct_caution, start, target,
                      target_label, no_route}
  web/route_debug.png
"""
import numpy as np
import json, os, sys
from scipy.ndimage import binary_dilation, label as cc_label
from skimage.graph import route_through_array
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SLOPE_SAFE, SLOPE_CAUTION, NOGO_COST, NOGO_DILATE_PX,
    START, PIXEL_SIZE_M
)

os.makedirs("web", exist_ok=True)

slope = np.load("web/slope.npy").astype("float64")
H, W  = slope.shape
print(f"Slope array: {H}×{W}")

# ── 1. Build dilated no-go mask & cost map ───────────────────────────────────
raw_nogo = slope > SLOPE_CAUTION
struct   = np.ones((3, 3), dtype=bool)                      # 8-connected 1-px dilation
nogo     = binary_dilation(raw_nogo, structure=struct,
                            iterations=NOGO_DILATE_PX)

# Smooth quadratic penalty inside passable cells
cost = np.where(
    nogo,
    NOGO_COST,
    1.0 + 10.0 * (slope / SLOPE_SAFE) ** 2
)

passable_pct = (~nogo).mean() * 100
print(f"Passable after dilation: {passable_pct:.1f}%  "
      f"(no-go from {raw_nogo.mean()*100:.1f}% → {nogo.mean()*100:.1f}% after dilation)")

# ── 2. Label connected components of passable mask ───────────────────────────
passable = ~nogo
labeled, n_comp = cc_label(passable, structure=np.ones((3,3)))
comp_sizes = np.bincount(labeled.ravel())
comp_sizes[0] = 0                          # background label
largest_label  = int(comp_sizes.argmax())
largest_mask   = labeled == largest_label
print(f"Connected components: {n_comp}  |  largest has {comp_sizes[largest_label]:,} cells "
      f"({comp_sizes[largest_label]/passable.size*100:.1f}% of grid)")

# Helper: snap a point to the nearest cell of `mask`
def snap_to_mask(r, c, mask, label_arr=None, label_id=None, search_r=200):
    target_mask = mask if label_arr is None else (label_arr == label_id)
    if target_mask[r, c]:
        return r, c
    # Build a small bounding box search
    r0, r1 = max(0, r-search_r), min(H, r+search_r+1)
    c0, c1 = max(0, c-search_r), min(W, c+search_r+1)
    sub     = target_mask[r0:r1, c0:c1]
    if not sub.any():
        # fall back to global search
        rows, cols = np.where(target_mask)
        dists = (rows - r)**2 + (cols - c)**2
        idx   = dists.argmin()
        return int(rows[idx]), int(cols[idx])
    rows_s, cols_s = np.where(sub)
    dists  = (rows_s - (r-r0))**2 + (cols_s - (c-c0))**2
    idx    = dists.argmin()
    return int(rows_s[idx] + r0), int(cols_s[idx] + c0)

# Clamp and snap START
def clamp(pt): return (max(1, min(pt[0], H-2)), max(1, min(pt[1], W-2)))

start_raw = clamp(START)
if labeled[start_raw] != largest_label:
    start = snap_to_mask(*start_raw, largest_mask, label_arr=labeled, label_id=largest_label)
    print(f"Start snapped: {start_raw} → {start}")
else:
    start = start_raw
    print(f"Start OK: {start}")

# ── 3. Find ridge-base target (item 5) ───────────────────────────────────────
# Strategy: scan the bottom-right quadrant for cells that border the no-go ridge
# on its gentle (safe-side) face.  Compare two candidates:
#   Candidate A: near col=1700 (original target area)
#   Candidate B: same row, 100+ px to the right (gentler slope side)
# Pick whichever gives lower total route cost.

def find_ridge_base(row_hint, col_hint, search=120):
    """Return nearest passable cell in largest component close to (row_hint, col_hint)."""
    r, c = clamp((row_hint, col_hint))
    return snap_to_mask(r, c, largest_mask, label_arr=labeled, label_id=largest_label)

def route_cost_and_stats(start_pt, target_pt):
    """Run routing; return (route_list, length_m, max_slope, pct_caution, total_cost) or None."""
    try:
        indices, total = route_through_array(cost, start_pt, target_pt, fully_connected=True)
    except Exception as e:
        print(f"  Routing failed: {e}")
        return None
    if total >= NOGO_COST:   # trapped — no real path
        return None
    route = [[int(r), int(c)] for r, c in indices]
    # Verify no cell is in the no-go mask
    nogo_hits = sum(1 for r, c in route if nogo[r, c])
    assert nogo_hits == 0, f"Route passes through {nogo_hits} no-go cells!"
    # Stats
    length_m = sum(
        PIXEL_SIZE_M * np.sqrt((route[i][0]-route[i-1][0])**2 + (route[i][1]-route[i-1][1])**2)
        for i in range(1, len(route))
    )
    rslopes = [slope[r, c] for r, c in route]
    max_sl  = max(rslopes)
    assert max_sl < SLOPE_CAUTION, f"Max slope on route {max_sl:.2f}° ≥ SLOPE_CAUTION {SLOPE_CAUTION}°"
    pct_cau = sum(1 for s in rslopes if SLOPE_SAFE < s <= SLOPE_CAUTION) / len(rslopes) * 100
    return route, length_m, max_sl, pct_cau, total

# Two candidate targets in the bottom-right quadrant
cand_A = find_ridge_base(1700, 1700)   # original bottom-right
cand_B = find_ridge_base(1700, 1820)   # right-shifted (gentler slope side)

print(f"\nCandidate A (bottom-right):  {cand_A}")
print(f"Candidate B (right-shifted): {cand_B}")

result_A = route_cost_and_stats(start, cand_A)
result_B = route_cost_and_stats(start, cand_B)

if result_A:
    print(f"  Route A: length={result_A[1]:.1f}m  max_slope={result_A[2]:.2f}°  total_cost={result_A[4]:.1f}")
else:
    print("  Route A: no safe path")

if result_B:
    print(f"  Route B: length={result_B[1]:.1f}m  max_slope={result_B[2]:.2f}°  total_cost={result_B[4]:.1f}")
else:
    print("  Route B: no safe path")

# Pick the lower-cost one
best = None
target_label_str = "Ridge base (science target)"
if result_A and result_B:
    if result_A[4] <= result_B[4]:
        best, target = result_A, cand_A
        print("→ Keeping Route A (lower total cost)")
    else:
        best, target = result_B, cand_B
        print("→ Keeping Route B (lower total cost)")
elif result_A:
    best, target = result_A, cand_A
    print("→ Route A only viable option")
elif result_B:
    best, target = result_B, cand_B
    print("→ Route B only viable option")
else:
    target = cand_A   # fallback label position

# ── 4. Build output ───────────────────────────────────────────────────────────
if best:
    route, length_m, max_sl, pct_cau, _ = best
    print(f"\n✓ Final route: {len(route)} waypoints  "
          f"{length_m:.1f}m  max={max_sl:.2f}°  caution={pct_cau:.1f}%")
    out = {
        "route":         route,
        "length_m":      round(length_m, 2),
        "max_slope_deg": round(float(max_sl), 2),
        "pct_caution":   round(pct_cau, 2),
        "start":         list(start),
        "target":        list(target),
        "target_label":  target_label_str,
        "no_route":      False,
    }
else:
    print("\n✗ No safe route found.")
    out = {
        "route": [], "length_m": 0, "max_slope_deg": 0, "pct_caution": 0,
        "start":  list(start),
        "target": list(target),
        "target_label": target_label_str,
        "no_route": True,
    }

with open("web/route.json", "w") as f:
    json.dump(out, f)
print("Saved web/route.json")

# ── 5. Debug overlay ──────────────────────────────────────────────────────────
slope_vis = slope.copy()
lo, hi = np.percentile(slope_vis, 1), np.percentile(slope_vis, 99)
img_arr = np.clip((slope_vis - lo) / (hi - lo) * 200, 0, 200).astype("uint8")
img = Image.fromarray(img_arr).convert("RGB")
draw = ImageDraw.Draw(img)

if best:
    pts = [(c, r) for r, c in route[::5]]
    if len(pts) > 1:
        draw.line(pts, fill=(255, 100, 0), width=2)

sr, sc = start
tr, tc = target
draw.ellipse([sc-6, sr-6, sc+6, sr+6], fill=(0, 220, 0),  outline=(0, 0, 0))
draw.ellipse([tc-6, tr-6, tc+6, tr+6], fill=(220, 0,   0), outline=(0, 0, 0))

img.save("web/route_debug.png")
print("Saved web/route_debug.png")
print("Done.")
