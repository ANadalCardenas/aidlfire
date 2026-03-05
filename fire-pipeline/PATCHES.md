# Understanding Patches

This document explains what patches are, how they're generated, what data they contain, and how to use them for training.

---

## Table of Contents

- [What Are Patches?](#what-are-patches)
- [Why Numpy Instead of Images?](#why-numpy-instead-of-images)
- [The 7 spectral bands](#the-7-spectral-bands)
- [The 8th channel (NDVI)](#the-8th-channel-ndvi)
- [Patch File Structure](#patch-file-structure)
- [The metadata.csv File](#the-metadatacsv-file)
- [Loading Patches for Training](#loading-patches-for-training)
- [Adding Transforms](#adding-transforms)
- [Generating Patches](#generating-patches)

---

## What Are Patches?

Patches are small (256×256 pixel) chunks cut from large satellite images. We cut them because:

1. **Neural networks need consistent sizes** - can't feed 1500×1200 one time and 800×900 another
2. **Memory limits** - huge images don't fit in GPU memory
3. **More training examples** - one big image becomes many small patches

```
Original Image (1500 × 1500)
┌─────────────────────────────────────┐
│                                     │
│    ┌─────┬─────┬─────┬─────┐       │
│    │  1  │  2  │  3  │  4  │       │
│    ├─────┼─────┼─────┼─────┤       │
│    │  5  │  6  │  7  │  8  │  ...  │
│    ├─────┼─────┼─────┼─────┤       │
│    │  9  │ 10  │ 11  │ 12  │       │
│    └─────┴─────┴─────┴─────┘       │
│                                     │
└─────────────────────────────────────┘

Each patch = 256 × 256 pixels
```

---

## Why Numpy Instead of Images?

Each patch is saved as a **numpy array** (`.npy` file), not an image file like PNG or JPEG.

```
PNG/JPEG images:              Numpy arrays:
┌────────────────────┐       ┌────────────────────┐
│ • 3 channels (RGB) │       │ • 7+ channels      │
│ • 8-bit (0-255)    │       │ • 32-bit float     │
│ • Lossy (JPEG)     │       │ • No data loss     │
│ • For humans       │       │ • For computers    │
└────────────────────┘       └────────────────────┘
```

| Reason | Explanation |
|--------|-------------|
| **More channels** | Images have 3 (RGB), but we need 7 spectral bands |
| **Precision** | Images are 8-bit (0-255), we need float32 (0.0-1.0) |
| **No compression loss** | JPEG destroys data, numpy preserves exact values |
| **Direct to PyTorch** | `np.load()` → `torch.tensor()` is fast and easy |

---

## The 7 spectral bands

We select 7 useful bands from the original 12 Sentinel-2 bands:

```python
band_indices = (1, 2, 3, 7, 8, 10, 11)  # 0-indexed from original 12 bands
```

| Layer | Original Band | Wavelength | What It "Sees" |
|-------|---------------|------------|----------------|
| **0** | B02 (Blue) | 490 nm | Water, coastal areas, smoke |
| **1** | B03 (Green) | 560 nm | Vegetation vigor, green reflection |
| **2** | B04 (Red) | 665 nm | Chlorophyll absorption, burn scars |
| **3** | B08 (NIR) | 842 nm | Healthy vegetation (reflects strongly) |
| **4** | B8A (NIR narrow) | 865 nm | Vegetation / water boundary |
| **5** | B11 (SWIR1) | 1610 nm | **Fire & burn detection!** |
| **6** | B12 (SWIR2) | 2190 nm | **Fire & burn detection!** |

### What the Values Mean

```
Values: 0.0 to 1.0 (surface reflectance)

0.0 = absorbs all light at this wavelength
1.0 = reflects all light at this wavelength
```

### How Fire Shows Up

```
                    Healthy Forest    Burned Area
                    ──────────────    ───────────
Layer 0 (Blue)      low (~0.03)       low (~0.05)
Layer 1 (Green)     low (~0.05)       low (~0.06)
Layer 2 (Red)       low (~0.03)       higher (~0.10)    ← burn scar visible
Layer 3 (NIR)       HIGH (~0.40)      LOW (~0.15)       ← big drop!
Layer 4 (NIR narrow) HIGH (~0.35)     LOW (~0.12)       ← big drop!
Layer 5 (SWIR1)     medium (~0.20)    HIGH (~0.30)      ← increases!
Layer 6 (SWIR2)     low (~0.10)       HIGH (~0.25)      ← increases!
```

**The pattern**: Burned areas have LOW NIR (layers 3-4) and HIGH SWIR (layers 5-6). This is what your model learns to detect!

### Quick Inspection

```python
import numpy as np

image = np.load("patch_image.npy")

print("Shape:", image.shape)  # (256, 256, 7) or (256, 256, 8)

channel_names = ["Blue", "Green", "Red", "NIR", "NIR-narrow", "SWIR1", "SWIR2"]
if image.shape[2] == 8:
    channel_names.append("NDVI")
for i, name in enumerate(channel_names):
    layer = image[:, :, i]
    print(f"Layer {i} ({name}): min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")
```

---

## The 8th channel (NDVI)

When **NDVI** (Normalized Difference Vegetation Index) is enabled (default), patches have **8 channels**. The 8th channel helps separate burn scars from other dark surfaces (water, shadow, soil).

| Property | Value |
|----------|--------|
| **Formula** | NDVI = (NIR − Red) / (NIR + Red + ε) |
| **Source bands** | Red = channel 2 (B04), NIR = channel 3 (B08) |
| **Raw range** | [−1, 1] (vegetation high, water/soil low) |
| **Stored range** | [0, 1] (mapped: `(NDVI + 1) / 2`) |

- **Enable/disable**: Use `include_ndvi=True` (default) or `False` in `PatchConfig`, or `--no-ndvi` in `run_pipeline.py` to output 7 channels only.
- **Constants**: `constants.NUM_INPUT_CHANNELS` = 8, `RED_INDEX_7` = 2, `NIR_INDEX_7` = 3.

---

## Patch File Structure

After running the pipeline, you get:

```
patches/
├── train/
│   ├── EMSR230_AOI01_01_r0_c0_image.npy       # (256, 256, 7) or (256, 256, 8) float32
│   ├── EMSR230_AOI01_01_r0_c0_mask.npy        # (256, 256) uint8
│   ├── EMSR230_AOI01_01_r0_c128_image.npy
│   ├── EMSR230_AOI01_01_r0_c128_mask.npy
│   ├── EMSR230_AOI01_01_r128_c0_image.npy
│   ├── EMSR230_AOI01_01_r128_c0_mask.npy
│   ├── ...
│   └── metadata.csv
├── val/
│   ├── ...
│   └── metadata.csv
└── test/
    ├── ...
    └── metadata.csv
```

### File Naming Convention

```
EMSR230_AOI01_01_r128_c256_image.npy
─────────────── ──── ────
      │          │    │
      │          │    └── column offset in source image
      │          └── row offset in source image
      └── source image name
```

### What's in Each File

```python
# Image patch
image = np.load("*_image.npy")
print(image.shape)  # (256, 256, 7) or (256, 256, 8)
print(image.dtype)  # float32
print(image.min(), image.max())  # 0.0 to 1.0

# Mask patch
mask = np.load("*_mask.npy")
print(mask.shape)   # (256, 256)
print(mask.dtype)   # uint8
print(np.unique(mask))  # [0, 1] for DEL, or [0,1,2,3,4] for GRA
```

---

## The metadata.csv File

Each split folder contains a `metadata.csv` with info about every patch:

```csv
patch_id,source_image,row,col,x_min,y_min,x_max,y_max,cloud_cover,burn_fraction
EMSR230_AOI01_01_r0_c0,EMSR230_AOI01_01,0,0,-8.270614,39.762582,-8.154426,39.891082,0.0000,0.1523
EMSR230_AOI01_01_r0_c128,EMSR230_AOI01_01,0,128,-8.212520,39.762582,-8.096332,39.891082,0.0000,0.2847
...
```

| Column | Description | Example |
|--------|-------------|---------|
| `patch_id` | Unique patch identifier | `EMSR230_AOI01_01_r128_c256` |
| `source_image` | Original image name | `EMSR230_AOI01_01` |
| `row` | Pixel row in source image | `128` |
| `col` | Pixel column in source image | `256` |
| `x_min` | Left longitude (degrees) | `-8.270614` |
| `y_min` | Bottom latitude (degrees) | `39.826832` |
| `x_max` | Right longitude (degrees) | `-8.154426` |
| `y_max` | Top latitude (degrees) | `39.891082` |
| `cloud_cover` | Fraction of cloudy pixels (0-1) | `0.0312` |
| `burn_fraction` | Fraction of burned pixels (0-1) | `0.2845` |

### Visual Representation

```
┌────────────────────────────────────────────────────────────┐
│  patch_id = "EMSR230_AOI01_01_r128_c256"                   │
│              ───────────────── ─── ───                     │
│              source_image      row col                     │
│                                                            │
│  Geographic bounds (for mapping back to real world):       │
│  ┌──────────────────────────────┐                         │
│  │ (x_min, y_max)  (x_max, y_max)│                         │
│  │       ┌────────────┐         │                         │
│  │       │   PATCH    │         │                         │
│  │       │  256×256   │         │                         │
│  │       └────────────┘         │                         │
│  │ (x_min, y_min)  (x_max, y_min)│                         │
│  └──────────────────────────────┘                         │
│                                                            │
│  cloud_cover = 0.03  → 3% cloudy (good patch!)            │
│  burn_fraction = 0.28 → 28% of patch is burned            │
└────────────────────────────────────────────────────────────┘
```

### Useful For

- **Filtering**: Skip patches with high cloud cover or no fire
- **Geolocation**: Map predictions back to real-world coordinates
- **Analysis**: Statistics about your training data
- **Debugging**: Trace a patch back to its source image

---

## Loading Patches for Training

The `dataset.py` file provides ready-to-use PyTorch classes.

### Quick Usage

```python
from dataset import WildfirePatchDataset, WildfireDataModule

# Option 1: Single dataset
train_dataset = WildfirePatchDataset("./patches/train")

image, mask = train_dataset[0]
print(image.shape)  # torch.Size([7, 256, 256])
print(mask.shape)   # torch.Size([256, 256])

# Option 2: Full data module (train/val/test in one)
data_module = WildfireDataModule(
    patches_root="./patches",
    batch_size=32,
    num_workers=4,
)

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### What Happens Inside

```python
# The dataset does this automatically:

# 1. Find all patches
patch_files = glob("*_image.npy")

# 2. When you request a sample:
image = np.load("patch_image.npy")     # (256, 256, 7) or (256, 256, 8)
mask = np.load("patch_mask.npy")       # (256, 256)

# 3. Convert to PyTorch format
image = torch.from_numpy(image)        # still (256, 256, 7) or (256, 256, 8)
image = image.permute(2, 0, 1)         # now (7, 256, 256) or (8, 256, 256) ← channels first!
image = image.float()                  # float32

mask = torch.from_numpy(mask)          # (256, 256)
mask = mask.long()                     # int64 for CrossEntropyLoss
```

### Data Flow Diagram

```
patches/train/
├── EMSR230_r0_c0_image.npy      ┐
├── EMSR230_r0_c0_mask.npy       ┤
├── EMSR230_r0_c128_image.npy    ┤     WildfirePatchDataset
├── EMSR230_r0_c128_mask.npy     ┼───►  finds pairs & loads
├── ...                          ┤
└── metadata.csv                 ┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │ DataLoader  │
                                    │ batch_size  │
                                    │ shuffle     │
                                    │ num_workers │
                                    └─────────────┘
                                           │
                                           ▼
                                    ┌─────────────────────┐
                                    │ Training Loop       │
                                    │                                                         │
                                    │ images: (32,7,256,256) or (32,8,256,256)
                                    │ masks:  (32,256,256) │
                                    └─────────────────────┘
```

### Training Loop Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WildfirePatchDataset

# Load data
train_dataset = WildfirePatchDataset("./patches/train")
val_dataset = WildfirePatchDataset("./patches/val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Your model (e.g., U-Net with 7 or 8 input channels)
model = UNet(in_channels=8, out_channels=2)  # 8 with NDVI; 2 classes: fire / no fire
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        # images: (batch, 7 or 8, 256, 256)
        # masks:  (batch, 256, 256)

        optimizer.zero_grad()
        outputs = model(images)          # (batch, 2, 256, 256)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            # compute metrics...
```

---

## Data Augmentation

The dataset supports on-the-fly augmentation via [Albumentations](https://albumentations.ai/), which handles multi-channel images and applies transforms to both image and mask together.

### Quick Start

```python
from dataset import (
    WildfirePatchDataset,
    WildfireDataModule,
    get_training_augmentation,
    get_strong_augmentation,
)

# Option 1: Use default augmentation (automatically enabled)
dm = WildfireDataModule("./patches", batch_size=32)

# Option 2: Explicit augmentation control
augment = get_training_augmentation()
dataset = WildfirePatchDataset("./patches/train", augment=augment)

# Option 3: Fire-aware augmentation (stronger for fire patches)
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),
    fire_augment=get_strong_augmentation(),
    fire_threshold=0.01,  # 1% fire pixels triggers stronger augmentation
)
```

### Available Augmentations

**Standard Training Augmentation** (`get_training_augmentation()`):

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Horizontal flip | p=0.5 | Spatial invariance |
| Vertical flip | p=0.5 | Spatial invariance |
| Random 90° rotation | p=0.5 | Rotational invariance |
| Brightness/Contrast | ±10%, p=0.3 | Atmospheric variation |
| Gaussian noise | σ=0.01, p=0.3 | Noise robustness |

**Strong Augmentation** (`get_strong_augmentation()`):

All standard augmentations plus:
- Higher probabilities (p=0.5-0.7)
- Elastic deformation (α=100, σ=10, p=0.3)
- Stronger brightness/contrast (±15%)

### Fire-Aware Augmentation

Applies stronger augmentation to patches containing fire (class imbalance Strategy 5):

```python
# Patches with >1% fire pixels get strong augmentation
# Patches without fire get standard augmentation
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),      # For non-fire patches
    fire_augment=get_strong_augmentation(),   # For fire patches
    fire_threshold=0.01,
)
```

### Weighted Sampling

Oversample fire patches to address class imbalance:

```python
dm = WildfireDataModule(
    "./patches",
    batch_size=32,
    use_weighted_sampling=True,   # Enable oversampling
    fire_sample_weight=5.0,       # Fire patches sampled 5x more often
)
```

### Custom Augmentation

```python
import albumentations as A

custom_augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(0.0, 0.02), p=0.3),
    A.ElasticTransform(alpha=120, sigma=12, p=0.3),
])

dataset = WildfirePatchDataset("./patches/train", augment=custom_augment)
```

### Normalization

Compute dataset statistics and apply normalization:

```python
from torchvision import transforms
from dataset import compute_dataset_statistics

# Compute stats from your data
stats = compute_dataset_statistics("./patches/train")
print(f"Mean: {stats['mean']}")  # Shape: (7,) or (8,)
print(f"Std:  {stats['std']}")   # Shape: (7,) or (8,)

# Create normalizer (applied after augmentation)
normalize = transforms.Normalize(
    mean=stats['mean'].tolist(),
    std=stats['std'].tolist(),
)

# Use with augmentation
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),  # Applied to numpy arrays
    transform=normalize,                   # Applied to tensors
)
```

### Class Weights for Loss Function

Compute inverse-frequency weights for weighted loss:

```python
from dataset import compute_class_weights

# For binary segmentation (DEL mask)
weights = compute_class_weights("./patches/train", num_classes=2)
# Example output: [1.0, 15.3]  (fire class weighted 15x more)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

---

## Generating Patches

### Quick Start

```bash
cd fire-pipeline

# Generate patches (default: 8 channels = 7 bands + NDVI, DEL mask)
uv run python run_pipeline.py --skip-extraction

# 7 channels only (no NDVI, e.g. for backward compatibility)
uv run python run_pipeline.py --skip-extraction --no-ndvi

# Custom output location
uv run python run_pipeline.py --skip-extraction --output-dir /path/to/patches

# Use severity mask instead of binary
uv run python run_pipeline.py --skip-extraction --mask-type GRA
```

### Pipeline Options

```bash
uv run python run_pipeline.py --help

Options:
  --dataset-dir PATH      Where wildfires-cems is located
  --output-dir PATH       Where to save patches (default: ./patches)
  --mask-type {DEL,GRA}   DEL=binary, GRA=severity
  --splits LIST           Which splits to process (default: train val test)
  --max-cloud-cover FLOAT Reject patches cloudier than this (default: 0.5)
  --skip-extraction       Don't try to extract tar files
  --force                 Regenerate all patches (default: skip existing)
  --no-ndvi               Disable NDVI; output 7 channels only (default: include NDVI)
```

### What the Pipeline Does

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Load satellite image (12 bands)                                     │
│  2. Select 7 useful bands                                               │
│  3. Optionally compute NDVI and append as 8th channel (default: yes)    │
│  4. Load corresponding mask (DEL or GRA)                               │
│  5. Load cloud mask                                                     │
│  6. Slide 256×256 window across image                                   │
│  7. For each window position:                                           │
│     - Check cloud cover                                                 │
│     - If cloud_cover <= 50%: save patch                                 │
│     - Record metadata (coordinates, stats)                              │
│  8. Save metadata.csv                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Patch Extraction Settings

| Parameter | Training | Inference |
|-----------|----------|-----------|
| Patch size | 256×256 | 256×256 |
| Stride | 128 (50% overlap) | 256 (no overlap) |
| Purpose | More training samples | Clean tiling |

The 50% overlap during training means each pixel appears in multiple patches, giving the model more context variations.
