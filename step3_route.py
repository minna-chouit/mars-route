"""
Step 3: Multi-stop science loop routing with minimax gentlest loop for blocked thresholds.
Safety thresholds: 20°, 15°, 10°.
- When a threshold has no safe route, computes the gentlest loop via binary search (minimax over max-slope limit 10°–25°).
- Fixes contradictory messages: if 1-pixel safety buffer blocks route, reports "blocked by the safety buffer".
- Computes closed-loop elevation profile (start and end elevations verified to match), marks highest & lowest points.
- Prints "route cells inside the no-go mask: N" and asserts N == 0 for all slider settings.
"""
import os, sys, json
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.graph import route_through_array
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SLOPE_SAFE, SLOPE_CAUTION, NOGO_COST, NOGO_DILATE_PX,
    START, PIXEL_SIZE_M, BOULDER_PENALTY_SCALE
)

os.makedirs("web", exist_ok=True)

slope = np.load("web/slope.npy").astype("float64")
dtm   = np.load("web/dtm_crop.npy").astype("float64")
H, W  = slope.shape

# 1. Load boulder density and boulder coordinates
boulder_density = np.zeros((H, W), dtype=np.float32)
if os.path.exists("web/boulder_density.npy"):
    try:
        boulder_density = np.load("web/boulder_density.npy").astype("float64")
        print(f"Loaded boulder density grid (mean={boulder_density.mean():.2f}, max={boulder_density.max():.1f} / 20m cell)")
    except Exception as e:
        print(f"Warning: could not load boulder_density.npy: {e}")

boulders_list = []
boulder_coords_m = None
if os.path.exists("web/boulders.json"):
    try:
        with open("web/boulders.json", "r") as f:
            bld_data = json.load(f)
            boulders_list = bld_data.get("boulders", [])
            if len(boulders_list) > 0:
                boulder_coords_m = np.array([[b["r"], b["c"]] for b in boulders_list]) * PIXEL_SIZE_M
                print(f"Loaded {len(boulders_list)} boulder coordinates for proximity checks.")
    except Exception as e:
        print(f"Warning: could not load boulders.json: {e}")

# 2. Science Stops
STOPS = [
    {
        "id": "start",
        "name": "Landing Site / Base",
        "rc": [int(START[0]), int(START[1])],
        "role": "start",
        "desc": "Primary habitat and landing zone located on flat, safe crater plains."
    },
    {
        "id": "stop1",
        "name": "Hollow",
        "rc": [650, 750],
        "role": "stop",
        "desc": "Sedimentary depression trapping wind-blown eolian grains and fine dust; primary target to sample active surface transport and shallow volatiles."
    },
    {
        "id": "stop2",
        "name": "Layered outcrop",
        "rc": [1200, 1300],
        "role": "stop",
        "desc": "Exposed stratigraphic sequence of rhythmically bedded lacustrine mudstones; vital for deciphering ancient aqueous depositional history."
    },
    {
        "id": "stop3",
        "name": "Boxwork ridge base",
        "rc": [1714, 1708],
        "role": "stop",
        "desc": "Fractured, mineralized veins formed by deep groundwater circulation through Gale Crater bedrock; critical target for past subsurface habitability."
    }
]

# Hard no-go mask (> 20° dilated by 1 px)
hard_nogo = binary_dilation(slope > SLOPE_CAUTION, structure=np.ones((3, 3)), iterations=NOGO_DILATE_PX)

# Loop leg definitions
LEG_SEQUENCE = [
    (0, 1, "Leg 1: Base → Hollow"),
    (1, 2, "Leg 2: Hollow → Layered outcrop"),
    (2, 3, "Leg 3: Layered outcrop → Boxwork ridge"),
    (3, 0, "Leg 4: Boxwork ridge → Base Return")
]

def plan_loop(cost_arr, impassable_mask):
    """Attempt to route loop Start -> 1 -> 2 -> 3 -> Start. Returns (success, legs, full_route, failure_reason)"""
    legs = []
    full_route = []
    for from_idx, to_idx, leg_name in LEG_SEQUENCE:
        pt_from = tuple(STOPS[from_idx]["rc"])
        pt_to   = tuple(STOPS[to_idx]["rc"])
        if impassable_mask[pt_from[0], pt_from[1]] or impassable_mask[pt_to[0], pt_to[1]]:
            return False, [], [], f"{leg_name} endpoints blocked by mask"
        try:
            indices, total_cost = route_through_array(cost_arr, pt_from, pt_to, fully_connected=True)
            leg_pts = [[int(r), int(c)] for r, c in indices]
            hits = sum(1 for r, c in leg_pts if impassable_mask[r, c])
            if hits > 0 or total_cost >= NOGO_COST / 2:
                return False, [], [], f"{leg_name} hits impassable mask"
            legs.append((leg_name, STOPS[from_idx]["name"], STOPS[to_idx]["name"], leg_pts))
            if len(full_route) == 0:
                full_route.extend(leg_pts)
            else:
                full_route.extend(leg_pts[1:])
        except Exception as e:
            return False, [], [], f"{leg_name} error: {e}"
    return True, legs, full_route, ""

# 3. Minimax Binary Search for Gentlest Loop (10° to 25°)
print("\nComputing gentlest loop via minimax binary search (10° to 25°)...")
low_s, high_s = 10.0, 25.0
gentlest_route_pts = None
gentlest_legs_raw = None
gentlest_cap = 25.0

for iter_idx in range(9):  # precision < 0.05°
    mid_s = (low_s + high_s) / 2.0
    test_mask = (slope > mid_s) | hard_nogo
    test_cost = np.where(test_mask, NOGO_COST, 1.0 + 10.0 * (slope / 15.0)**2 + BOULDER_PENALTY_SCALE * boulder_density)
    ok, legs, route, _ = plan_loop(test_cost, test_mask)
    if ok:
        gentlest_cap = mid_s
        gentlest_route_pts = route
        gentlest_legs_raw = legs
        high_s = mid_s
    else:
        low_s = mid_s

print(f"Minimax optimal slope cap: {gentlest_cap:.2f}°")

def build_route_stats(full_route, legs_raw, th_val, is_gentlest, label_override=None):
    """Compile comprehensive stats, closed-loop elevation profile, and boulder proximity."""
    # Distance
    total_dist_m = sum(
        PIXEL_SIZE_M * np.sqrt((full_route[i][0] - full_route[i-1][0])**2 + (full_route[i][1] - full_route[i-1][1])**2)
        for i in range(1, len(full_route))
    )
    route_slopes = [float(slope[r, c]) for r, c in full_route]
    max_slope_route = max(route_slopes)
    pct_caution_route = sum(1 for s in route_slopes if 15.0 < s <= 20.0) / len(route_slopes) * 100.0

    # Max radius from start
    start_r, start_c = STOPS[0]["rc"]
    max_dist_from_start_m = max(
        PIXEL_SIZE_M * np.sqrt((r - start_r)**2 + (c - start_c)**2)
        for r, c in full_route
    )

    # Boulder proximity within 5m
    total_boulders_5m = 0
    full_tree = None
    if boulder_coords_m is not None and len(boulder_coords_m) > 0:
        full_tree = cKDTree(np.array(full_route) * PIXEL_SIZE_M)
        dists, _ = full_tree.query(boulder_coords_m, distance_upper_bound=5.0)
        total_boulders_5m = int(np.sum(dists <= 5.0))

    # Per-leg info
    legs_info = []
    for leg_name, f_name, t_name, leg_pts in legs_raw:
        leg_len = sum(
            PIXEL_SIZE_M * np.sqrt((leg_pts[i][0] - leg_pts[i-1][0])**2 + (leg_pts[i][1] - leg_pts[i-1][1])**2)
            for i in range(1, len(leg_pts))
        )
        l_slopes = [float(slope[r, c]) for r, c in leg_pts]
        l_bld = 0
        if boulder_coords_m is not None and len(boulder_coords_m) > 0:
            ltree = cKDTree(np.array(leg_pts) * PIXEL_SIZE_M)
            ldists, _ = ltree.query(boulder_coords_m, distance_upper_bound=5.0)
            l_bld = int(np.sum(ldists <= 5.0))
        legs_info.append({
            "name": leg_name,
            "from_stop": f_name,
            "to_stop": t_name,
            "length_m": round(leg_len, 1),
            "max_slope_deg": round(max(l_slopes), 1),
            "boulders_within_5m": l_bld,
            "waypoints": len(leg_pts)
        })

    # Elevation Profile along full closed loop (Start and End must match)
    step = max(1, len(full_route) // 250)
    profile_pts = []
    cum_dist = 0.0
    for i in range(len(full_route)):
        if i > 0:
            dr = full_route[i][0] - full_route[i-1][0]
            dc = full_route[i][1] - full_route[i-1][1]
            cum_dist += PIXEL_SIZE_M * np.sqrt(dr*dr + dc*dc)
        if i % step == 0 or i == len(full_route) - 1:
            r, c = full_route[i]
            profile_pts.append({
                "d_km": round(cum_dist / 1000.0, 3),
                "elev": round(float(dtm[r, c]), 1),
                "slope": round(float(slope[r, c]), 1),
                "rc": [r, c]
            })

    # Verify start and end elevation match for closed loop
    elev_start = profile_pts[0]["elev"]
    elev_end   = profile_pts[-1]["elev"]
    assert abs(elev_start - elev_end) < 0.1, f"Start ({elev_start}m) and end ({elev_end}m) elevation mismatch!"

    # Mark highest and lowest points
    elevs = [p["elev"] for p in profile_pts]
    min_idx = int(np.argmin(elevs))
    max_idx = int(np.argmax(elevs))
    lowest_pt  = {"d_km": profile_pts[min_idx]["d_km"], "elev": profile_pts[min_idx]["elev"]}
    highest_pt = {"d_km": profile_pts[max_idx]["d_km"], "elev": profile_pts[max_idx]["elev"]}

    lbl = label_override if label_override else (
        f"gentlest possible: {max_slope_route:.1f}° max slope, {round(total_dist_m)} m, exceeds selected limit"
        if is_gentlest else f"Safe Traverse ({th_val}° max slope)"
    )

    return {
        "safe": not is_gentlest,
        "is_gentlest": is_gentlest,
        "label": lbl,
        "route": full_route,
        "legs": legs_info,
        "total_distance_m": round(total_dist_m, 1),
        "max_slope_deg": round(max_slope_route, 1),
        "pct_caution": round(pct_caution_route, 1),
        "max_distance_from_start_m": round(max_dist_from_start_m, 1),
        "boulders_within_5m": total_boulders_5m,
        "profile": profile_pts,
        "elevation_summary": {
            "start_elev": elev_start,
            "end_elev": elev_end,
            "lowest": lowest_pt,
            "highest": highest_pt
        },
        "waypoints": len(full_route)
    }

# 4. Evaluate Thresholds: 20°, 15°, 10°
THRESHOLDS = [20.0, 15.0, 10.0]
results_by_thresh = {}

for th in THRESHOLDS:
    th_key = str(int(th))
    print(f"\n================ Evaluating Slider Setting: {th}° ================")
    
    # Threshold mask: slope > th with 1px dilation safety buffer
    raw_nogo_th = slope > th
    nogo_th = binary_dilation(raw_nogo_th, structure=np.ones((3, 3)), iterations=NOGO_DILATE_PX)
    cost_th = np.where(
        nogo_th,
        NOGO_COST,
        1.0 + 10.0 * (slope / 15.0)**2 + BOULDER_PENALTY_SCALE * boulder_density
    )
    
    # Try normal routing first
    safe_ok, legs_th, route_th, fail_reason = plan_loop(cost_th, nogo_th)
    
    if safe_ok:
        data_th = build_route_stats(route_th, legs_th, int(th), is_gentlest=False)
        data_th["status_message"] = f"Safe loop verified under {int(th)}° limit."
        print(f"--> Threshold {th}°: SAFE LOOP! Dist: {data_th['total_distance_m']} m, Max slope: {data_th['max_slope_deg']}°")
    else:
        # Determine precise non-contradictory failure reason
        # Check if route attempted without safety buffer would have slope < th
        raw_cost = np.where(raw_nogo_th, NOGO_COST, 1.0 + 10.0 * (slope / 15.0)**2 + BOULDER_PENALTY_SCALE * boulder_density)
        raw_ok, _, raw_route, _ = plan_loop(raw_cost, raw_nogo_th)
        
        if raw_ok:
            max_raw = max(slope[r, c] for r, c in raw_route)
            if max_raw <= th:
                reason = "blocked by the safety buffer"
            else:
                reason = f"blocked by terrain steeper than {int(th)}°"
        else:
            reason = f"blocked by terrain steeper than {int(th)}°"
            
        print(f"--> Threshold {th}°: No strict route ({reason}). Using gentlest loop.")
        gentlest_lbl = f"gentlest possible: {max(slope[r, c] for r, c in gentlest_route_pts):.1f}° max slope, {round(sum(PIXEL_SIZE_M * np.sqrt((gentlest_route_pts[i][0]-gentlest_route_pts[i-1][0])**2 + (gentlest_route_pts[i][1]-gentlest_route_pts[i-1][1])**2) for i in range(1, len(gentlest_route_pts))))} m, exceeds selected limit"
        data_th = build_route_stats(gentlest_route_pts, gentlest_legs_raw, int(th), is_gentlest=True, label_override=gentlest_lbl)
        data_th["status_message"] = f"No safe route at {int(th)}° limit ({reason}). Displaying gentlest possible loop."
    
    # Requirement 5: Print "route cells inside the no-go mask: N" and assert N == 0
    route_to_check = data_th["route"]
    nogo_cells = sum(1 for r, c in route_to_check if hard_nogo[r, c])
    print(f"route cells inside the no-go mask: {nogo_cells}")
    assert nogo_cells == 0, f"Error: route cells inside no-go mask is {nogo_cells} != 0!"
    
    results_by_thresh[th_key] = data_th

# Default threshold = 20
active_th = "20"

# Master JSON
output_data = {
    "stops": STOPS,
    "start": STOPS[0]["rc"],
    "target": STOPS[3]["rc"],
    "target_label": STOPS[3]["name"],
    "thresholds": results_by_thresh,
    "active_threshold": active_th,
    "route": results_by_thresh[active_th]["route"],
    "length_m": results_by_thresh[active_th]["total_distance_m"],
    "max_slope_deg": results_by_thresh[active_th]["max_slope_deg"],
    "pct_caution": results_by_thresh[active_th]["pct_caution"],
    "legs": results_by_thresh[active_th]["legs"],
    "max_distance_from_start_m": results_by_thresh[active_th]["max_distance_from_start_m"],
    "boulders_within_5m": results_by_thresh[active_th]["boulders_within_5m"],
    "boulder_hazard_label": "possible boulders from shadows, not validated"
}

# Preserve THEMIS info
if os.path.exists("web/route.json"):
    try:
        with open("web/route.json", "r") as f:
            old = json.load(f)
            for k in ["ti_start", "ti_target", "ti_min", "ti_max", "ti_mean"]:
                if k in old: output_data[k] = old[k]
    except Exception:
        pass

with open("web/route.json", "w") as f:
    json.dump(output_data, f)
print("\nSaved web/route.json successfully.")

# Debug PNG
lo, hi = np.percentile(slope, 1), np.percentile(slope, 99)
img = Image.fromarray(np.clip((slope - lo) / (hi - lo) * 200, 0, 200).astype("uint8")).convert("RGB")
draw = ImageDraw.Draw(img)
pts = [(c, r) for r, c in results_by_thresh[active_th]["route"][::5]]
if len(pts) > 1:
    draw.line(pts, fill=(255, 100, 0), width=2)
colors = [(0, 220, 0), (0, 180, 255), (255, 200, 0), (220, 0, 0)]
for idx, s in enumerate(STOPS):
    r, c = s["rc"]
    draw.ellipse([c-7, r-7, c+7, r+7], fill=colors[idx % len(colors)], outline=(0, 0, 0))
img.save("web/route_debug.png")
print("Saved web/route_debug.png")
print("Done step 3.")
