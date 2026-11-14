"""
Step 3 (v4): Single target — Ridge base (1714, 1708).
Reads:  web/slope.npy
Writes: web/route.json, web/route_debug.png
"""
import numpy as np, json, os, sys
from scipy.ndimage import binary_dilation, label as cc_label
from skimage.graph import route_through_array
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from config import SLOPE_SAFE, SLOPE_CAUTION, NOGO_COST, NOGO_DILATE_PX, START, PIXEL_SIZE_M

os.makedirs("web", exist_ok=True)
slope = np.load("web/slope.npy").astype("float64")
H, W  = slope.shape

nogo = binary_dilation(slope > SLOPE_CAUTION, structure=np.ones((3,3)), iterations=NOGO_DILATE_PX)
cost = np.where(nogo, NOGO_COST, 1.0 + 10.0 * (slope / SLOPE_SAFE)**2)

labeled, _ = cc_label(~nogo, structure=np.ones((3,3)))
sizes = np.bincount(labeled.ravel()); sizes[0] = 0
largest = labeled == int(sizes.argmax())

def snap(r, c, search=200):
    r, c = max(1,min(r,H-2)), max(1,min(c,W-2))
    if largest[r,c]: return r, c
    r0,r1 = max(0,r-search),min(H,r+search+1)
    c0,c1 = max(0,c-search),min(W,c+search+1)
    sub = largest[r0:r1,c0:c1]
    rs,cs = np.where(sub if sub.any() else largest)
    if not sub.any(): r0=c0=0
    d = (rs-(r-r0))**2+(cs-(c-c0))**2
    i = d.argmin()
    return int(rs[i]+r0), int(cs[i]+c0)

start  = snap(*START)
target = snap(1714, 1708)
print(f"Start: {start}  Target: {target}")

indices, total = route_through_array(cost, start, target, fully_connected=True)
route = [[int(r),int(c)] for r,c in indices]
assert all(not nogo[r,c] for r,c in route), "Route hits no-go cell!"
rslopes = [slope[r,c] for r,c in route]
max_sl  = float(max(rslopes))
assert max_sl < SLOPE_CAUTION
length_m = sum(PIXEL_SIZE_M*np.sqrt((route[i][0]-route[i-1][0])**2+(route[i][1]-route[i-1][1])**2)
               for i in range(1,len(route)))
pct_cau = sum(1 for s in rslopes if SLOPE_SAFE < s <= SLOPE_CAUTION)/len(rslopes)*100

print(f"Route: {len(route)} pts  {length_m:.1f} m  max={max_sl:.2f}°  caution={pct_cau:.1f}%")

out = dict(route=route, length_m=round(length_m,2), max_slope_deg=round(max_sl,2),
           pct_caution=round(pct_cau,2), start=list(start), target=list(target),
           target_label="Ridge base (science target)", no_route=False)
json.dump(out, open("web/route.json","w"))
print("Saved web/route.json")

lo,hi = np.percentile(slope,1),np.percentile(slope,99)
img = Image.fromarray(np.clip((slope-lo)/(hi-lo)*200,0,200).astype("uint8")).convert("RGB")
draw = ImageDraw.Draw(img)
pts = [(c,r) for r,c in route[::5]]
if len(pts)>1: draw.line(pts, fill=(255,100,0), width=2)
sr,sc=start; tr,tc=target
draw.ellipse([sc-6,sr-6,sc+6,sr+6], fill=(0,220,0),  outline=(0,0,0))
draw.ellipse([tc-6,tr-6,tc+6,tr+6], fill=(220,0,0),  outline=(0,0,0))
img.save("web/route_debug.png")
print("Done.")
