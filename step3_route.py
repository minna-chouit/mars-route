"""
Step 3: Multi-stop science loop routing with safety thresholds (10°, 15°, 20°)
        and boulder-density penalty to avoid rock hazards.
Continuous route: Start -> Stop 1 -> Stop 2 -> Stop 3 -> Start.
Computes per-leg stats, total budget, max distance from start, elevation profile data,
and counts boulders within 5 m of the route.
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

# Load boulder density and boulder coordinates if available
boulder_density = np.zeros((H, W), dtype=np.float32)
if os.path.exists("web/boulder_density.npy"):
    try:
        boulder_density = np.load("web/boulder_density.npy").astype("float64")
        print(f"Loaded boulder density grid (mean={boulder_density.mean():.2f}, max={boulder_density.max():.1f} / 20m cell)")
    except Exception as e:
        print(f"Warning: could not load boulder_density.npy: {e}")

boulders_list = []
boulder_tree = None
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

# ── Science Stops Definition ─────────────────────────────────────────────────
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

# Preserve existing THEMIS TI values if present in route.json
existing_ti = {}
if os.path.exists("web/route.json"):
    try:
        with open("web/route.json", "r") as f:
            old = json.load(f)
            for k in ["ti_start", "ti_target", "ti_min", "ti_max", "ti_mean"]:
                if k in old: existing_ti[k] = old[k]
    except Exception:
        pass

# Also sample TI at each stop if available
themis_arr = None
if os.path.exists("web/themis.npy"):
    try:
        themis_arr = np.load("web/themis.npy")
    except Exception:
        pass

for stop in STOPS:
    r, c = stop["rc"]
    stop["elevation_m"] = round(float(dtm[r, c]), 1)
    stop["slope_deg"]   = round(float(slope[r, c]), 1)
    if themis_arr is not None and 0 <= r < themis_arr.shape[0] and 0 <= c < themis_arr.shape[1]:
        stop["ti_val"] = round(float(themis_arr[r, c]), 1)

# Waypoint sequence for loop: Start -> 1 -> 2 -> 3 -> Start
LEG_SEQUENCE = [
    (0, 1, "Leg 1: Base → Hollow"),
    (1, 2, "Leg 2: Hollow → Layered outcrop"),
    (2, 3, "Leg 3: Layered outcrop → Boxwork ridge"),
    (3, 0, "Leg 4: Boxwork ridge → Base Return")
]

# ── Precompute Routes for Thresholds 10°, 15°, 20° ───────────────────────────
THRESHOLDS = [20.0, 15.0, 10.0]
results_by_thresh = {}

for th in THRESHOLDS:
    th_key = str(int(th))
    print(f"\n================ Evaluating Max Slope Threshold: {th}° ================")
    raw_nogo = slope > th
    struct = np.ones((3, 3), dtype=bool)
    nogo = binary_dilation(raw_nogo, structure=struct, iterations=NOGO_DILATE_PX)
    
    # Smooth quadratic slope penalty + boulder density penalty inside passable ground
    # No-go cells remain blocked with NOGO_COST
    cost = np.where(
        nogo,
        NOGO_COST,
        1.0 + 10.0 * (slope / 15.0) ** 2 + BOULDER_PENALTY_SCALE * boulder_density
    )
    
    legs_info = []
    combined_route = []
    overall_safe = True
    failure_reason = ""
    
    for from_idx, to_idx, leg_name in LEG_SEQUENCE:
        pt_from = tuple(STOPS[from_idx]["rc"])
        pt_to   = tuple(STOPS[to_idx]["rc"])
        
        # Check if endpoints themselves are blocked
        if nogo[pt_from[0], pt_from[1]]:
            overall_safe = False
            failure_reason = f"{STOPS[from_idx]['name']} is on slope > {int(th)}°."
            break
        if nogo[pt_to[0], pt_to[1]]:
            overall_safe = False
            failure_reason = f"{STOPS[to_idx]['name']} is on slope > {int(th)}°."
            break
            
        try:
            indices, total_cost = route_through_array(cost, pt_from, pt_to, fully_connected=True)
            leg_pts = [[int(r), int(c)] for r, c in indices]
            
            # Verify every point is safe
            hits = sum(1 for r, c in leg_pts if nogo[r, c])
            leg_slopes = [float(slope[r, c]) for r, c in leg_pts]
            max_sl = max(leg_slopes)
            
            if hits > 0 or max_sl > th:
                overall_safe = False
                failure_reason = f"{leg_name} blocked by terrain steeper than {int(th)}° (max {max_sl:.1f}°)."
                print(f"  {leg_name}: BLOCKED (hits={hits}, max_slope={max_sl:.1f}°)")
                break
                
            leg_len_m = sum(
                PIXEL_SIZE_M * np.sqrt((leg_pts[i][0] - leg_pts[i-1][0])**2 + (leg_pts[i][1] - leg_pts[i-1][1])**2)
                for i in range(1, len(leg_pts))
            )
            pct_cau = sum(1 for s in leg_slopes if 15.0 < s <= 20.0) / len(leg_slopes) * 100.0
            
            # Count boulders within 5m of this leg
            leg_boulders_5m = 0
            if boulder_coords_m is not None and len(boulder_coords_m) > 0:
                leg_tree = cKDTree(np.array(leg_pts) * PIXEL_SIZE_M)
                dists, _ = leg_tree.query(boulder_coords_m, distance_upper_bound=5.0)
                leg_boulders_5m = int(np.sum(dists <= 5.0))
            
            legs_info.append({
                "name": leg_name,
                "from_stop": STOPS[from_idx]["name"],
                "to_stop": STOPS[to_idx]["name"],
                "length_m": round(leg_len_m, 1),
                "max_slope_deg": round(max_sl, 1),
                "pct_caution": round(pct_cau, 1),
                "boulders_within_5m": leg_boulders_5m,
                "waypoints": len(leg_pts)
            })
            print(f"  {leg_name}: {leg_len_m:.1f} m, max slope {max_sl:.1f}°, {pct_cau:.1f}% caution, {leg_boulders_5m} bld ≤5m")
            
            if len(combined_route) == 0:
                combined_route.extend(leg_pts)
            else:
                combined_route.extend(leg_pts[1:])
                
        except Exception as e:
            overall_safe = False
            failure_reason = f"Routing error on {leg_name}: {e}"
            print(f"  {leg_name}: Failed ({e})")
            break
            
    if overall_safe:
        # Full loop statistics
        total_dist_m = sum(
            PIXEL_SIZE_M * np.sqrt((combined_route[i][0] - combined_route[i-1][0])**2 + (combined_route[i][1] - combined_route[i-1][1])**2)
            for i in range(1, len(combined_route))
        )
        route_slopes = [float(slope[r, c]) for r, c in combined_route]
        max_slope_route = max(route_slopes)
        pct_caution_route = sum(1 for s in route_slopes if 15.0 < s <= 20.0) / len(route_slopes) * 100.0
        
        # Max distance from start (straight-line radius)
        start_r, start_c = STOPS[0]["rc"]
        distances_from_start = [
            PIXEL_SIZE_M * np.sqrt((r - start_r)**2 + (c - start_c)**2)
            for r, c in combined_route
        ]
        max_dist_from_start_m = max(distances_from_start)
        
        # Total boulders within 5m of continuous route
        total_boulders_5m = 0
        if boulder_coords_m is not None and len(boulder_coords_m) > 0:
            full_tree = cKDTree(np.array(combined_route) * PIXEL_SIZE_M)
            dists, _ = full_tree.query(boulder_coords_m, distance_upper_bound=5.0)
            total_boulders_5m = int(np.sum(dists <= 5.0))
        
        # Profile samples along route (~250 points)
        step = max(1, len(combined_route) // 250)
        profile_pts = []
        cum_dist = 0.0
        
        for i in range(len(combined_route)):
            if i > 0:
                dr = combined_route[i][0] - combined_route[i-1][0]
                dc = combined_route[i][1] - combined_route[i-1][1]
                cum_dist += PIXEL_SIZE_M * np.sqrt(dr*dr + dc*dc)
            if i % step == 0 or i == len(combined_route) - 1:
                r, c = combined_route[i]
                profile_pts.append({
                    "d": round(cum_dist, 1),
                    "elev": round(float(dtm[r, c]), 1),
                    "slope": round(float(slope[r, c]), 1),
                    "rc": [r, c]
                })
                
        results_by_thresh[th_key] = {
            "safe": True,
            "route": combined_route,
            "legs": legs_info,
            "total_distance_m": round(total_dist_m, 1),
            "max_slope_deg": round(max_slope_route, 1),
            "pct_caution": round(pct_caution_route, 1),
            "max_distance_from_start_m": round(max_dist_from_start_m, 1),
            "boulders_within_5m": total_boulders_5m,
            "profile": profile_pts,
            "waypoints": len(combined_route)
        }
        print(f"--> Threshold {th}°: SAFE LOOP! Dist: {total_dist_m:.1f} m, Max radius: {max_dist_from_start_m:.1f} m, Boulders ≤5m: {total_boulders_5m}")
    else:
        results_by_thresh[th_key] = {
            "safe": False,
            "reason": f"No safe route found for {int(th)}° threshold. ({failure_reason})"
        }
        print(f"--> Threshold {th}°: NO SAFE ROUTE ({failure_reason})")

# Determine default active threshold (prefer 20)
active_th = "20" if results_by_thresh.get("20", {}).get("safe") else "15"

# Combine into master route.json
output_data = {
    "stops": STOPS,
    "start": STOPS[0]["rc"],
    "target": STOPS[3]["rc"],
    "target_label": STOPS[3]["name"],
    "thresholds": results_by_thresh,
    "active_threshold": active_th,
    # Flatten primary active route for direct backwards compatibility
    "route": results_by_thresh[active_th].get("route", []),
    "length_m": results_by_thresh[active_th].get("total_distance_m", 0),
    "max_slope_deg": results_by_thresh[active_th].get("max_slope_deg", 0),
    "pct_caution": results_by_thresh[active_th].get("pct_caution", 0),
    "legs": results_by_thresh[active_th].get("legs", []),
    "max_distance_from_start_m": results_by_thresh[active_th].get("max_distance_from_start_m", 0),
    "boulders_within_5m": results_by_thresh[active_th].get("boulders_within_5m", 0),
    "boulder_hazard_label": "possible boulders from shadows, not validated",
    "no_route": not results_by_thresh[active_th].get("safe", False)
}

# Add back THEMIS values
output_data.update(existing_ti)

with open("web/route.json", "w") as f:
    json.dump(output_data, f)
print("\nSaved web/route.json with rock hazard avoidance & boulder proximity stats.")

# Save debug preview PNG
lo, hi = np.percentile(slope, 1), np.percentile(slope, 99)
img = Image.fromarray(np.clip((slope - lo) / (hi - lo) * 200, 0, 200).astype("uint8")).convert("RGB")
draw = ImageDraw.Draw(img)

if results_by_thresh[active_th].get("safe"):
    pts = [(c, r) for r, c in results_by_thresh[active_th]["route"][::5]]
    if len(pts) > 1:
        draw.line(pts, fill=(255, 100, 0), width=2)

colors = [(0, 220, 0), (0, 180, 255), (255, 200, 0), (220, 0, 0)]
for idx, s in enumerate(STOPS):
    r, c = s["rc"]
    col = colors[idx % len(colors)]
    draw.ellipse([c-7, r-7, c+7, r+7], fill=col, outline=(0, 0, 0))

img.save("web/route_debug.png")
print("Saved web/route_debug.png")
