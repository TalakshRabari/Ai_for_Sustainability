"""
q2_labels.py - Label Construction & Dataset Preparation
=========================================================
For each filtered image:
  - Extract 128x128 land-cover patch from land_cover.tif
  - Assign label using dominant (mode) land-cover class
  - Map ESA codes to: Built-up, Vegetation, Water, Cropland, Others
  - 60/40 train-test split
  - Visualize class distribution

Usage:
    python q2_labels.py

Outputs:
    - outputs/dataset.csv             : image path, label, split
    - outputs/q2_class_distribution.png
    - outputs/q2_sample_patches.png   : sample images per class
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from scipy import stats
from sklearn.model_selection import train_test_split
from collections import Counter

# ─────────────────────────────────────────────
# Paths
# scripts/q2_labels.py  ->  data is one level up
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # AiforSustainability/
RGB_DIR       = os.path.join(BASE_DIR, "rgb")
TIFF_PATH     = os.path.join(BASE_DIR, "worldcover_bbox_delhi_ncr_2021.tif")
FILTERED_LIST = os.path.join(BASE_DIR, "scripts", "outputs", "filtered_images.txt")
OUT_DIR       = os.path.join(BASE_DIR, "scripts", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# ESA WorldCover class mapping
# Code -> simplified category
# ─────────────────────────────────────────────
ESA_CLASS_MAP = {
    10:  "Vegetation",   # Tree cover
    20:  "Vegetation",   # Shrubland
    30:  "Vegetation",   # Grassland
    40:  "Cropland",     # Cropland
    50:  "Built-up",     # Built-up
    60:  "Others",       # Bare/sparse vegetation
    70:  "Others",       # Snow and ice
    80:  "Water",        # Permanent water bodies
    90:  "Others",       # Herbaceous wetland
    95:  "Others",       # Mangroves
    100: "Others",       # Moss and lichen
}

def esa_to_category(code):
    return ESA_CLASS_MAP.get(int(code), "Others")

# ─────────────────────────────────────────────
# Load filtered image list
# ─────────────────────────────────────────────
print("Loading filtered image list...")

if os.path.exists(FILTERED_LIST):
    with open(FILTERED_LIST) as f:
        image_files = [os.path.join(RGB_DIR, line.strip())
                       for line in f if line.strip()]
else:
    # Fall back to all images if q1 hasn't been run
    print("  WARNING: filtered_images.txt not found, using all images")
    image_files = glob.glob(os.path.join(RGB_DIR, "*.png"))

print(f"  {len(image_files)} images to process")

# ─────────────────────────────────────────────
# Parse lat/lon from filename
# ─────────────────────────────────────────────
def parse_coords(filepath):
    fname = os.path.basename(filepath).replace(".png", "")
    parts = fname.split("_")
    return float(parts[0]), float(parts[1])   # lat, lon

# ─────────────────────────────────────────────
# Extract 128x128 land-cover patch for each image
# ─────────────────────────────────────────────
print("\nExtracting land-cover patches from TIF...")
print(f"  TIF path: {TIFF_PATH}")

IMG_SIZE = 128   # pixels

records = []
skipped = 0

with rasterio.open(TIFF_PATH) as src:
    print(f"  TIF CRS    : {src.crs}")
    print(f"  TIF bounds : {src.bounds}")
    print(f"  TIF shape  : {src.height} x {src.width}")
    print(f"  Resolution : {src.res}")

    for i, img_path in enumerate(image_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(image_files)}...")

        lat, lon = parse_coords(img_path)

        # Convert lat/lon to pixel row/col in the TIF
        row, col = src.index(lon, lat)

        # Calculate window: 128x128 centered on the pixel
        half = IMG_SIZE // 2
        row_off = row - half
        col_off = col - half

        # Check bounds
        if (row_off < 0 or col_off < 0 or
                row_off + IMG_SIZE > src.height or
                col_off + IMG_SIZE > src.width):
            skipped += 1
            continue

        # Read the patch
        window = Window(col_off, row_off, IMG_SIZE, IMG_SIZE)
        patch = src.read(1, window=window)   # band 1 = land cover class

        if patch.size == 0:
            skipped += 1
            continue

        # Dominant class = mode
        flat = patch.flatten()
        flat = flat[flat != src.nodata] if src.nodata is not None else flat
        if len(flat) == 0:
            skipped += 1
            continue

        dominant_code = int(stats.mode(flat, keepdims=True).mode[0])
        category = esa_to_category(dominant_code)

        records.append({
            'filepath':      img_path,
            'filename':      os.path.basename(img_path),
            'lat':           lat,
            'lon':           lon,
            'esa_code':      dominant_code,
            'label':         category,
        })

print(f"  Done. {len(records)} labeled, {skipped} skipped (out of bounds)")

# ─────────────────────────────────────────────
# Build DataFrame
# ─────────────────────────────────────────────
df = pd.DataFrame(records)
print(f"\nLabel distribution:")
print(df['label'].value_counts().to_string())

# ─────────────────────────────────────────────
# 60/40 train-test split (stratified)
# ─────────────────────────────────────────────
print("\nSplitting dataset 60/40...")

# Only stratify if all classes have >= 2 samples
label_counts = df['label'].value_counts()
rare = label_counts[label_counts < 2].index.tolist()
if rare:
    print(f"  WARNING: removing rare classes with <2 samples: {rare}")
    df = df[~df['label'].isin(rare)]

train_df, test_df = train_test_split(
    df, test_size=0.4, random_state=42, stratify=df['label'])

df['split'] = 'test'
df.loc[train_df.index, 'split'] = 'train'

print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")

# Save CSV
csv_path = os.path.join(OUT_DIR, "dataset.csv")
df.to_csv(csv_path, index=False)
print(f"  -> Dataset saved: {csv_path}")

# ─────────────────────────────────────────────
# Visualize class distribution
# ─────────────────────────────────────────────
print("\nVisualizing class distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall distribution
colors = {'Built-up': '#e74c3c', 'Vegetation': '#2ecc71',
          'Water': '#3498db', 'Cropland': '#f39c12', 'Others': '#95a5a6'}

labels = df['label'].value_counts().index
counts = df['label'].value_counts().values
bar_colors = [colors.get(l, 'gray') for l in labels]

axes[0].bar(labels, counts, color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('Overall Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Land Use Class')
axes[0].set_ylabel('Number of Images')
for i, v in enumerate(counts):
    axes[0].text(i, v + 5, str(v), ha='center', fontsize=10)
axes[0].tick_params(axis='x', rotation=15)

# Train vs Test distribution
train_counts = train_df['label'].value_counts()
test_counts  = test_df['label'].value_counts()
all_labels   = sorted(df['label'].unique())
x = np.arange(len(all_labels))
w = 0.35

axes[1].bar(x - w/2, [train_counts.get(l, 0) for l in all_labels],
            w, label='Train (60%)', color='steelblue', edgecolor='black', lw=0.5)
axes[1].bar(x + w/2, [test_counts.get(l, 0) for l in all_labels],
            w, label='Test (40%)',  color='coral',     edgecolor='black', lw=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(all_labels, rotation=15)
axes[1].set_title('Train / Test Split by Class', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Land Use Class')
axes[1].set_ylabel('Number of Images')
axes[1].legend()

plt.tight_layout()
dist_path = os.path.join(OUT_DIR, "q2_class_distribution.png")
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Distribution plot saved: {dist_path}")

# ─────────────────────────────────────────────
# Visualize sample patches per class
# ─────────────────────────────────────────────
print("Visualizing sample patches per class...")
from PIL import Image

unique_labels = df['label'].unique()
n_classes = len(unique_labels)
n_samples = 4

fig, axes = plt.subplots(n_classes, n_samples,
                          figsize=(n_samples * 2.5, n_classes * 2.5))
if n_classes == 1:
    axes = [axes]

for row_idx, label in enumerate(sorted(unique_labels)):
    subset = df[df['label'] == label].sample(
        min(n_samples, len(df[df['label'] == label])), random_state=42)
    for col_idx in range(n_samples):
        ax = axes[row_idx][col_idx]
        if col_idx < len(subset):
            try:
                img = Image.open(subset.iloc[col_idx]['filepath'])
                ax.imshow(img)
            except Exception:
                ax.set_facecolor('lightgray')
        else:
            ax.set_facecolor('lightgray')
        ax.axis('off')
        if col_idx == 0:
            ax.set_ylabel(label, fontsize=10, fontweight='bold', rotation=0,
                          labelpad=60, va='center')

plt.suptitle('Sample Satellite Patches per Land Use Class', fontsize=13, fontweight='bold')
plt.tight_layout()
sample_path = os.path.join(OUT_DIR, "q2_sample_patches.png")
plt.savefig(sample_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Sample patches saved: {sample_path}")

print("\n✓ Q2 Complete!")
print(f"  Total labeled images : {len(df)}")
print(f"  Train set size       : {len(train_df)}")
print(f"  Test set size        : {len(test_df)}")
print(f"  Classes              : {sorted(df['label'].unique())}")