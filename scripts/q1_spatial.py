"""
q1_spatial.py - Spatial Reasoning & Data Filtering
====================================================
Q1a: Plot Delhi-NCR shapefile and overlay a 60x60 km uniform grid
Q1b: Filter satellite images whose center coordinates fall inside the region
Q1c: Report total images before and after filtering

Usage:
    python q1_spatial.py

Outputs:
    - outputs/q1_grid_map.png         : map with Delhi-NCR boundary + grid
    - outputs/filtered_images.txt     : list of image filenames inside region
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import Transformer

# ─────────────────────────────────────────────
# Paths
# scripts/q1_spatial.py  ->  data is one level up
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # AiforSustainability/
RGB_DIR       = os.path.join(BASE_DIR, "rgb")
NCR_GEOJSON   = os.path.join(BASE_DIR, "delhi_ncr_region.geojson")
OUT_DIR       = os.path.join(BASE_DIR, "scripts", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load Delhi-NCR boundary
# ─────────────────────────────────────────────
print("Loading Delhi-NCR boundary...")
ncr = gpd.read_file(NCR_GEOJSON)
ncr = ncr.to_crs(epsg=4326)   # ensure WGS84 (lat/lon)
print(f"  CRS: {ncr.crs}")
print(f"  Bounds: {ncr.total_bounds}")   # [minx, miny, maxx, maxy]

# ─────────────────────────────────────────────
# 2. Build 60×60 km grid in EPSG:32644 (UTM),
#    then reproject back to WGS84 for plotting
# ─────────────────────────────────────────────
print("\nBuilding 60×60 km grid...")

# Project NCR to UTM zone 44N (EPSG:32644) as specified in task
ncr_utm = ncr.to_crs(epsg=32644)
minx, miny, maxx, maxy = ncr_utm.total_bounds

GRID_SIZE = 60_000   # 60 km in metres

# Create grid cells
grid_cells = []
x = minx
while x < maxx:
    y = miny
    while y < maxy:
        cell = box(x, y, x + GRID_SIZE, y + GRID_SIZE)
        grid_cells.append(cell)
        y += GRID_SIZE
    x += GRID_SIZE

grid_utm = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32644")
grid_wgs84 = grid_utm.to_crs(epsg=4326)

print(f"  Total grid cells created: {len(grid_cells)}")

# ─────────────────────────────────────────────
# 3. Plot Delhi-NCR boundary + grid
# ─────────────────────────────────────────────
print("\nPlotting map...")
fig, ax = plt.subplots(figsize=(10, 10))

# Plot grid first (background)
grid_wgs84.boundary.plot(ax=ax, color='steelblue', linewidth=0.8,
                          linestyle='--', alpha=0.7, label='60×60 km Grid')

# Plot NCR boundary on top
ncr.plot(ax=ax, facecolor='none', edgecolor='red',
         linewidth=2.5, label='Delhi-NCR Boundary')

ax.set_title("Delhi-NCR Region with 60×60 km Uniform Grid", fontsize=14, fontweight='bold')
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "q1_grid_map.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Map saved: {out_path}")

# ─────────────────────────────────────────────
# 4. Load all image filenames and parse coords
# ─────────────────────────────────────────────
print("\nLoading image filenames...")
all_images = glob.glob(os.path.join(RGB_DIR, "*.png"))
total_before = len(all_images)
print(f"  Total images BEFORE filtering: {total_before}")

# Parse lat/lon from filename: "28.9090_77.2734.png"
def parse_coords(filepath):
    fname = os.path.basename(filepath).replace(".png", "")
    parts = fname.split("_")
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon

# ─────────────────────────────────────────────
# 5. Filter images inside Delhi-NCR boundary
# ─────────────────────────────────────────────
print("Filtering images inside Delhi-NCR region...")

# Union of NCR geometry for point-in-polygon test
ncr_union = ncr.union_all() if hasattr(ncr, 'union_all') else ncr.geometry.unary_union

filtered = []
for img_path in all_images:
    lat, lon = parse_coords(img_path)
    pt = Point(lon, lat)   # shapely uses (x=lon, y=lat)
    if ncr_union.contains(pt):
        filtered.append(img_path)

total_after = len(filtered)
print(f"  Total images AFTER filtering:  {total_after}")
print(f"  Images removed (outside NCR):  {total_before - total_after}")

# Save filtered list
filtered_path = os.path.join(OUT_DIR, "filtered_images.txt")
with open(filtered_path, "w") as f:
    for p in sorted(filtered):
        f.write(os.path.basename(p) + "\n")
print(f"  -> Filtered list saved: {filtered_path}")

# ─────────────────────────────────────────────
# 6. Plot filtered image locations on map
# ─────────────────────────────────────────────
print("\nPlotting filtered image locations...")
fig, ax = plt.subplots(figsize=(10, 10))

grid_wgs84.boundary.plot(ax=ax, color='steelblue', linewidth=0.8,
                          linestyle='--', alpha=0.5)
ncr.plot(ax=ax, facecolor='lightyellow', edgecolor='red',
         linewidth=2.5, alpha=0.5)

# Plot filtered points
if filtered:
    coords = [parse_coords(p) for p in filtered]
    lats, lons = zip(*coords)
    ax.scatter(lons, lats, s=2, c='green', alpha=0.5, label=f'Inside NCR ({total_after})')

# Plot all points (outside)
all_coords = [parse_coords(p) for p in all_images]
all_lats, all_lons = zip(*all_coords)
ax.scatter(all_lons, all_lats, s=1, c='gray', alpha=0.2, label=f'All images ({total_before})')

ax.set_title(f"Satellite Image Locations\nBefore: {total_before}  |  After filtering: {total_after}",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path2 = os.path.join(OUT_DIR, "q1_filtered_points.png")
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Points map saved: {out_path2}")

print("\n✓ Q1 Complete!")
print(f"  Images before filtering : {total_before}")
print(f"  Images after filtering  : {total_after}")