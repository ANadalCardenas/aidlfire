# Fire Detection Pipeline

A data processing pipeline for the CEMS Wildfire Dataset, designed to prepare Sentinel-2 satellite imagery for training deep learning models to detect and assess wildfire damage.

## Table of Contents

- [Quick Start](#quick-start)
- [Full workflow: CEMS + Sen2Fire (recommended)](#full-workflow-cems--sen2fire-recommended)
- [Overview](#overview)
- [Project Files](#project-files)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Downloading the Data](#downloading-the-data)
- [Pipeline Workflow](#pipeline-workflow)
- [Running the Pipeline](#running-the-pipeline)
- [Input channels and vegetation (NDVI)](#input-channels-and-vegetation-ndvi)
- [Region Filtering](#region-filtering)
- [Using the Patches for Training](#using-the-patches-for-training)
- [Class Imbalance Analysis](#class-imbalance-analysis)
- [Data Augmentation](#data-augmentation)
- [Model Training](#model-training)
- [Inference Pipeline](#inference-pipeline)
- [Streamlit Web App](#streamlit-web-app)
- [Visualization Tools](#visualization-tools)
- [Technical Reference](#technical-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
cd fire-pipeline

# Install dependencies (choose one based on your needs)
uv sync                  # Base only (data processing)
uv sync --extra train    # + PyTorch, segmentation-models-pytorch, wandb
uv sync --extra app      # + Streamlit, folium, planetary-computer
uv sync --extra all      # Everything included
```

### Train a Model

```bash
# 1. Install training dependencies
uv sync --extra train

# 2. Download data (can be interrupted and resumed)
uv run python download_dataset.py --output-dir ../wildfires-cems

# 3. Generate patches (can be interrupted and resumed)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches

# 4. (Optional) Analyze class distribution
uv run python analyze_patches.py ./patches

# 5. Train the model (--output-dir is required)
uv run python train.py --patches-dir ./patches --output-dir ./output/resnet34-unet --num-classes 2 --epochs 50

# Model saved to: ./output/resnet34-unet/checkpoints/best_model.pt
# Use --overwrite-output-dir to allow overwriting an existing output directory
```

**Both download and patch generation support resume**: If interrupted, just re-run the same command and it will skip already-completed work. Use `--force` to regenerate everything.

### Full workflow: CEMS + Sen2Fire (recommended)

Recommended defaults: **NDVI** (8 channels), **dual-head** (binary + severity), **s2cloudless** for Sen2Fire cloud filtering when available.

#### 1. Process the first dataset (CEMS)

```bash
uv sync --extra train

# Download CEMS data (optional: --splits train for faster testing)
uv run python download_dataset.py --output-dir ../wildfires-cems

# Generate patches (NDVI included by default; use --no-ndvi to disable)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches

# Optional: analyze class distribution
uv run python analyze_patches.py ./patches --plot
```

#### 2. Process the second dataset (Sen2Fire)

Sen2Fire patches are loaded directly from `.npz` files (no separate patch generation). Download the dataset (e.g. from [Zenodo 10881058](https://zenodo.org/records/10881058)) and place it so you have:

```
data-sen2fire/
├── scene1/   # .npz files (image 12×512×512, label 512×512)
├── scene2/
├── scene3/
└── scene4/
```

Cloud filtering uses **s2cloudless** by default when installed (`uv sync --extra train` includes it). Use `--no-s2cloudless` in the fine-tune script to use rule-based filtering only.

#### 3. Train dual-head model on CEMS (GRA severity)

Train with **dual heads** (binary fire + severity) so you can later fine-tune on Sen2Fire while keeping the severity head frozen:

```bash
# Requires GRA patches (num_classes=5). NDVI is used if patches have 8 channels.
uv run python train.py \
    --patches-dir ./patches \
    --output-dir ./output/dual_head \
    --num-classes 5 \
    --dual-head \
    --epochs 50

# Checkpoints: ./output/dual_head/checkpoints/best_model.pt
```

#### 4. Fine-tune on Sen2Fire (binary only, severity frozen)

```bash
uv run python train_sen2fire_finetune.py \
    --checkpoint ./output/dual_head/checkpoints/best_model.pt \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/sen2fire_finetune

# Uses s2cloudless for cloud filtering by default; --no-s2cloudless for rule-based only.
# Uses NDVI (8 ch) by default if the CEMS checkpoint has 8 channels; --no-ndvi for 7.
```

Result: a single model that outputs both **binary fire** and **severity** maps. In the app, use the "Show binary fire map" and "Show severity map" toggles to compare. History stores both maps for dual-head results.

### Alternative: Combined Binary → Severity workflow

For maximum geographic diversity, train binary on **CEMS + Sen2Fire combined**, then add severity on CEMS GRA only. See [docs/COMBINED_BINARY_SEVERITY_WORKFLOW.md](../docs/COMBINED_BINARY_SEVERITY_WORKFLOW.md) for details.

```bash
# Phase 1: Combined binary (CEMS DEL + Sen2Fire)
uv run python run_pipeline.py --dataset-dir ../wildfires-cems --output-dir ./patches --mask-type DEL
uv run python train_combined_binary.py \
    --patches-dir ./patches \
    --sen2fire-dir ../data-sen2fire \
    --output-dir ./output/combined_binary

# Phase 2: Severity fine-tune (CEMS GRA only)
uv run python run_pipeline.py --dataset-dir ../wildfires-cems --output-dir ./patches_gra --mask-type GRA
uv run python train_severity_finetune.py \
    --checkpoint ./output/combined_binary/checkpoints/best_model.pt \
    --patches-dir ./patches_gra \
    --output-dir ./output/severity_finetune
```

### Run the Web App

```bash
# 1. Install app dependencies
uv sync --extra app

# 2a. With a trained model (single model):
mkdir -p checkpoints
cp /path/to/best_model.pt ./checkpoints/
uv run streamlit run app.py

# 2b. With multiple models (dropdown in sidebar):
# Set FIRE_MODELS_DIR to your output dir (e.g. ./output); app lists subdirs as models
FIRE_MODELS_DIR=./output uv run streamlit run app.py

# 2c. Demo mode (no model needed, uses mock data):
uv run streamlit run app.py
```

### Run Inference (Python API)

```python
from inference import FireInferencePipeline

# Load trained model
pipeline = FireInferencePipeline("checkpoints/best_model.pt")

# Run on a GeoTIFF file
result = pipeline.predict_from_file("satellite_image.tif")

# Or on a numpy array (H, W, 7), (H, W, 8), or (H, W, 12)
result = pipeline.predict_from_array(image_array)

# Access results
print(f"Fire detected: {result.has_fire}")
print(f"Confidence: {result.fire_confidence:.1%}")
print(f"Coverage: {result.fire_fraction:.2%}")
print(f"Severity counts: {result.severity_counts}")
```

### Run Inference (Command Line)

```bash
uv run python inference.py checkpoints/best_model.pt satellite_image.tif
```

---

## Overview

This pipeline processes satellite imagery from the Copernicus Emergency Management Service (CEMS) Wildfire Dataset to create training-ready patches for semantic segmentation models. The goal is to train models that can:

1. **Fire Detection**: Binary classification of burned vs. unburned areas
2. **Severity Assessment**: Multi-class prediction of fire damage levels (0-4)

### What This Pipeline Does

```
Raw Sentinel-2 GeoTIFFs     →     256×256 Patches     →     PyTorch DataLoader
    (12 bands)                    (7 or 8 channels)            (ready for training)
```

1. Loads 12-band Sentinel-2 imagery and selects 7 key spectral bands
2. Optionally adds **NDVI** (vegetation index) as an 8th channel to improve burn vs. vegetation separation
3. Extracts overlapping patches using a sliding window
4. Filters out cloudy patches (>50% cloud cover)
5. Saves patches as NumPy arrays with metadata
6. Provides PyTorch Dataset classes for model training

---

## Project Files

### Scripts Included

| File | Purpose |
|------|---------|
| `patch_generator.py` | Core logic for cutting satellite images into 256×256 patches |
| `run_pipeline.py` | CLI to run the full patch generation pipeline |
| `download_dataset.py` | Download dataset from HuggingFace (for cloud deployment) |
| `dataset.py` | PyTorch Dataset/DataLoader with augmentation support |
| `analyze_patches.py` | Analyze class distribution and get imbalance recommendations |
| `model.py` | U-Net segmentation model with loss functions |
| `metrics.py` | Evaluation metrics for segmentation and detection |
| `train.py` | Training script with checkpointing and W&B logging |
| `visualize.py` | Tools for visualizing patches and source images |
| `remove_catalan_data.py` | Move Catalan fire data to separate folder for regional testing |
| `inference.py` | Inference pipeline for running trained models |
| `satellite_fetcher.py` | Fetch Sentinel-2 imagery from Planetary Computer |
| `storage.py` | Local storage/caching system for analysis results |
| `app.py` | Streamlit web app for interactive fire detection |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | This file - pipeline usage and reference |
| `PATCHES.md` | Detailed explanation of patch format, channels (7 or 8), and loading |
| `CHANGES.md` | Changelog and recent changes (e.g. NDVI extension, options) |

### What's Included

This pipeline provides a **complete system**:

- ✅ Data preparation (patching, augmentation)
- ✅ Model architecture (U-Net with pretrained encoders)
- ✅ Training script with checkpointing
- ✅ Loss functions (CrossEntropy, Dice, Focal)
- ✅ Evaluation metrics (IoU, Dice, Detection F1)
- ✅ W&B integration for experiment tracking
- ✅ Inference pipeline for new satellite imagery
- ✅ Streamlit web app for interactive analysis
- ✅ Satellite data fetching from Planetary Computer

**Not included** (bring your own):
- Model export (ONNX, TorchScript)

---

## Dataset Description

### Source: CEMS Wildfire Dataset

The [CEMS Wildfire Dataset](https://huggingface.co/datasets/links-ads/wildfires-cems) contains Sentinel-2 satellite imagery of wildfires across Europe (2017-2023), with expert-annotated burn masks from the Copernicus Emergency Management Service.

| Property | Value |
|----------|-------|
| **Images** | ~500+ wildfire events |
| **Time Period** | June 2017 - April 2023 |
| **Satellite** | Sentinel-2 L2A (atmospherically corrected) |
| **Resolution** | ~10 meters per pixel |
| **Coverage** | Primarily Europe (Portugal, Spain, Italy, Greece, etc.) |
| **License** | CC-BY-4.0 |

### File Structure

Each wildfire event (called an "activation") contains:

```
EMSR382/                          # Activation ID
└── AOI01/                        # Area of Interest
    └── EMSR382_AOI01_01/         # Tile within AOI
        ├── EMSR382_AOI01_01_S2L2A.tif    # 12-band satellite image
        ├── EMSR382_AOI01_01_DEL.tif      # Delineation mask (binary)
        ├── EMSR382_AOI01_01_GRA.tif      # Grading mask (severity)
        ├── EMSR382_AOI01_01_CM.tif       # Cloud mask
        ├── EMSR382_AOI01_01_ESA_LC.tif   # Land cover
        └── EMSR382_AOI01_01_S2L2A.json   # Metadata
```

### Naming Convention

```
EMSR382_AOI01_01
│     │     └── Tile number (large areas are split into tiles)
│     └── Area of Interest (some events have multiple AOIs)
└── Emergency activation ID (unique per fire event)
```

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Git LFS (for downloading the dataset)

### Setup

```bash
# Clone or navigate to this directory
cd fire-pipeline

# Install dependencies (handled automatically by uv)
uv sync

# For PyTorch support (optional, for training)
uv add torch
```

### Install Git LFS (required for dataset download)

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Initialize Git LFS
git lfs install
```

---

## Downloading the Data

### Option 1: Automated Download (Recommended)

Use `download_dataset.py` for automated download with resume support:

```bash
# Install download dependencies
uv sync --extra download

# Step 1: Download and extract (can be interrupted and resumed)
uv run python download_dataset.py --output-dir ../wildfires-cems

# Step 2: Generate patches separately (can be interrupted and resumed)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches

# OR: Do everything in one command (still resumable)
uv run python download_dataset.py \
    --output-dir ../wildfires-cems \
    --generate-patches \
    --patches-dir ./patches
```

**Resume Support:**
- **Download**: Automatically resumes interrupted downloads (via huggingface_hub)
- **Extraction**: Skips already-extracted splits (checks for EMSR* folders)
- **Patches**: Skips already-generated patches (checks for existing .npy files)

If you want to regenerate everything from scratch:
```bash
# Force regenerate all patches
uv run python run_pipeline.py --force ...

# Or with download_dataset.py
uv run python download_dataset.py --generate-patches --force-patches ...
```

**Additional options:**
```bash
# Download only training data (faster for testing)
uv run python download_dataset.py --splits train

# Skip extraction if already done
uv run python download_dataset.py --no-extract
```

### Option 2: Manual Git Clone

```bash
# Clone the dataset (will take 30-60 minutes depending on connection)
git lfs install
git clone https://huggingface.co/datasets/links-ads/wildfires-cems

# The data comes as compressed archives that need extraction
cd wildfires-cems
cat data/train/train.tar.*.gz.part | tar -xzvf - -i -C data/train/
cat data/val/val.tar.*.gz.part | tar -xzvf - -i -C data/val/
cat data/test/test.tar.*.gz.part | tar -xzvf - -i -C data/test/
```

**Note**: The `run_pipeline.py` script can handle extraction automatically.

### Option 3: Build from Source

See the [original CEMS repository](https://github.com/MatteoM95/CEMS-Wildfire-Dataset) for instructions on downloading directly from Copernicus and Sentinel Hub APIs.

### Cloud Deployment Workflow

For deploying to cloud environments (AWS, GCP, Azure, etc.):

#### Option A: Full Training Pipeline

Train a model from scratch on the cloud:

```bash
# 1. Clone your code repository (small, no data)
git clone https://github.com/your-repo/fire-detection.git
cd fire-detection/fire-pipeline

# 2. Install all dependencies
uv sync --extra all

# 3. Download data (~20-40 min, resumable if interrupted)
uv run python download_dataset.py --output-dir ./wildfires-cems

# 4. Generate patches (~10-20 min, resumable if interrupted)
uv run python run_pipeline.py \
    --dataset-dir ./wildfires-cems \
    --output-dir ./patches

# 5. Analyze class distribution (optional but recommended)
uv run python analyze_patches.py ./patches --plot

# 6. Train the model (--output-dir is required)
uv run python train.py \
    --patches-dir ./patches \
    --output-dir ./output/resnet34-unet \
    --num-classes 2 \
    --epochs 50 \
    --wandb  # Optional: enable W&B logging

# 7. Model checkpoint saved to: ./output/resnet34-unet/checkpoints/best_model.pt
```

**Note:** Both download and patch generation support resume. If interrupted (e.g., VM preemption), re-run the same command and it will continue from where it left off.

#### Option B: Run App with Pre-trained Model

Deploy the app using a model you've already trained:

```bash
# 1. Clone repository
git clone https://github.com/your-repo/fire-detection.git
cd fire-detection/fire-pipeline

# 2. Install app dependencies only
uv sync --extra app

# 3. Copy your trained model
mkdir -p checkpoints
cp /path/to/your/best_model.pt ./checkpoints/

# 4. Configure app (edit app.py)
#    - Set USE_MOCK_FETCHER = False for real satellite data
#    - Verify MODEL_PATH points to your checkpoint

# 5. Run the Streamlit app
streamlit run app.py
```

#### Option C: Demo Mode (No Training, No Data)

Run the app in mock mode for demos/testing:

```bash
# 1. Clone repository
git clone https://github.com/your-repo/fire-detection.git
cd fire-detection/fire-pipeline

# 2. Install app dependencies
uv sync --extra app

# 3. Run app (uses mock data and mock inference by default)
streamlit run app.py
```

#### What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `download_dataset.py` | Downloads raw satellite data from HuggingFace | `wildfires-cems/` directory |
| `download_dataset.py --generate-patches` | Downloads AND creates training patches | `wildfires-cems/` + `patches/` |
| `train.py` | Trains U-Net model on patches | `output/checkpoints/best_model.pt` |
| `app.py` | Runs Streamlit web interface | Web app at localhost:8501 |

---

## Pipeline Workflow

### Stage 1: Patch Extraction

The pipeline uses a **sliding window** approach to extract patches:

```
┌─────────────────────────────────────┐
│         Original Image              │
│         (e.g., 1380 × 1129)         │
│                                     │
│  ┌─────┬─────┬─────┐               │
│  │  1  │  2  │  3  │  ...          │  Stride = 128 (training)
│  ├─────┼─────┼─────┤               │  Stride = 256 (inference)
│  │  4  │  5  │  6  │  ...          │
│  ├─────┼─────┼─────┤               │  Patch size = 256×256
│  │  7  │  8  │  9  │  ...          │
│  └─────┴─────┴─────┘               │
│                                     │
└─────────────────────────────────────┘
```

| Parameter | Training | Inference |
|-----------|----------|-----------|
| Patch size | 256×256 | 256×256 |
| Stride | 128 (50% overlap) | 256 (no overlap) |
| Purpose | Data augmentation via overlap | Clean tiling for reconstruction |

### Stage 2: Band selection and vegetation index

From the original 12 Sentinel-2 bands, we select 7 bands. When NDVI is enabled (default), an 8th channel is added:

| Index | Band | Wavelength | Resolution | Use in Fire Detection |
|-------|------|------------|------------|----------------------|
| 0 | B02 | 490nm (Blue) | 10m | Smoke detection |
| 1 | B03 | 560nm (Green) | 10m | Vegetation health |
| 2 | B04 | 665nm (Red) | 10m | Burn scar visibility |
| 3 | B08 | 842nm (NIR) | 10m | Vegetation stress |
| 4 | B8A | 865nm (NIR narrow) | 20m | Vegetation/water |
| 5 | B11 | 1610nm (SWIR1) | 20m | Active fire, burn severity |
| 6 | B12 | 2190nm (SWIR2) | 20m | Active fire, burn severity |
| **7** | **NDVI** | (NIR−Red)/(NIR+Red) | — | Vegetation index (when enabled) |

**Why these bands?** SWIR bands (B11, B12) are particularly sensitive to fire and burn scars. NIR bands detect vegetation stress. NDVI (when used) helps separate burned vegetation from water/shadow/soil. The combination enables detection of both active fires and post-fire damage.

### Stage 3: Quality Filtering

Patches are filtered based on cloud cover:

```python
# Cloud mask values
0 = Clear sky
1 = Cloud (opaque)
2 = Light cloud / smoke / fog
3 = Cloud shadow

# Rejection criterion
if (pixels with value >= 1) / total_pixels > 0.5:
    reject_patch()
```

### Stage 4: Output format

```
patches/
├── train/
│   ├── EMSR230_AOI01_01_r0_c0_image.npy      # (256, 256, 7) or (256, 256, 8) float32
│   ├── EMSR230_AOI01_01_r0_c0_mask.npy       # (256, 256) uint8
│   ├── EMSR230_AOI01_01_r0_c128_image.npy
│   ├── EMSR230_AOI01_01_r0_c128_mask.npy
│   ├── ...
│   └── metadata.csv
├── val/
│   └── ...
└── test/
    └── ...
```

**Metadata CSV columns:**
- `patch_id`: Unique identifier
- `source_image`: Original image name
- `row`, `col`: Pixel coordinates in source image
- `x_min`, `y_min`, `x_max`, `y_max`: Geographic bounds (EPSG:4326)
- `cloud_cover`: Fraction of cloudy pixels (0-1)
- `burn_fraction`: Fraction of burned pixels (0-1)

---

## Running the Pipeline

### Quick Start

```bash
cd fire-pipeline

# Run the full pipeline (extract archives + generate patches)
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches \
    --mask-type DEL

# For severity grading instead of binary detection
uv run python run_pipeline.py \
    --dataset-dir ../wildfires-cems \
    --output-dir ./patches \
    --mask-type GRA
```

### Command Line Options

```bash
uv run python run_pipeline.py --help

Options:
  --dataset-dir PATH      Directory containing wildfires-cems clone
  --output-dir PATH       Output directory for patches (default: ./patches)
  --mask-type {DEL,GRA}   DEL=binary fire detection, GRA=severity grading
  --splits LIST           Splits to process (default: train val test)
  --max-cloud-cover FLOAT Maximum cloud cover fraction (default: 0.5)
  --skip-extraction       Skip tar extraction if already done
  --force                 Regenerate all patches (default: skip existing)
  --no-ndvi               Disable NDVI as 8th channel (output 7 channels only)
```

By default, patches include **8 channels** (7 spectral bands + NDVI). Use `--no-ndvi` to generate 7-channel patches only (e.g. for backward compatibility with older checkpoints).

### Using patch_generator.py Directly

For more control, use the `PatchGenerator` class:

```python
from pathlib import Path
from patch_generator import PatchGenerator, PatchConfig

# Custom configuration
config = PatchConfig(
    patch_size=256,
    stride_train=128,
    stride_inference=256,
    max_cloud_cover=0.5,
    include_ndvi=True,  # Add NDVI as 8th channel (default: True)
    band_indices=(1, 2, 3, 7, 8, 10, 11),  # 7 bands
    clip_min=0.0,
    clip_max=1.0,
)

generator = PatchGenerator(config)

# Process entire dataset
metadata = generator.process_dataset(
    dataset_root=Path("../wildfires-cems/data"),
    output_root=Path("./patches"),
    mask_type="DEL",
    splits=["train", "val", "test"],
)

# Or process a single image
metadata = generator.process_image(
    image_dir=Path("../wildfires-cems/data/train/EMSR230/AOI01/EMSR230_AOI01_01"),
    output_dir=Path("./patches/train"),
    mask_type="DEL",
    mode="train",
)
```

---

## Input channels and vegetation (NDVI)

### Default: 8 channels (7 bands + NDVI)

The pipeline adds a **vegetation index (NDVI)** as an 8th input channel by default. NDVI helps the model separate burn scars from other low-NIR surfaces (e.g. water, shadow, bare soil).

| Channel | Source | Description |
|---------|--------|-------------|
| 0–6 | Sentinel-2 | B02, B03, B04, B08, B8A, B11, B12 (same as before) |
| 7 | **NDVI** | (NIR − Red) / (NIR + Red), mapped from [−1, 1] to [0, 1] |

- **Constants** (`constants.py`): `NUM_INPUT_CHANNELS = 8`, `RED_INDEX_7 = 2`, `NIR_INDEX_7 = 3`, `INPUT_CHANNEL_NAMES`.
- **Patch generation**: Set `include_ndvi=False` in `PatchConfig` or use `--no-ndvi` in `run_pipeline.py` to output 7-channel patches only.
- **Training**: The training script uses `NUM_INPUT_CHANNELS` (8) when creating the model and saves `in_channels` in the checkpoint config.
- **Inference**: The pipeline reads `in_channels` from the checkpoint (default 7 for old checkpoints). If the model expects 8 channels and the input has 7 (e.g. 12-band GeoTIFF), NDVI is computed and appended automatically.

### Backward compatibility

- **Old 7-channel checkpoints**: Load as before; `in_channels` defaults to 7 if missing from config.
- **New 8-channel runs**: Regenerate patches without `--no-ndvi`, train as usual; checkpoints will have `in_channels: 8`.
- **Mixed usage**: You can generate 7-channel patches with `--no-ndvi` and train a 7-channel model; the code supports both.

---

## Region Filtering

### Separating Catalan Data

If you want to train a model on non-Catalan fires and test specifically on Catalonia (or vice versa), use the `remove_catalan_data.py` script to move Catalan fire events to a separate folder.

The script identifies Catalan fires based on geographic coordinates (bounding box: 0.15°-3.35°E longitude, 40.5°-42.9°N latitude) from the `satelliteData.csv` metadata file.

```bash
# Preview what will be moved (dry run - default)
uv run python remove_catalan_data.py

# Actually move the folders
uv run python remove_catalan_data.py --execute
```

**What happens:**
- Catalan EMSR folders are moved from `train/train/`, `val/val/`, `test/test/` to a new `cat/` folder
- The structure is preserved: `cat/train/`, `cat/val/`, `cat/test/`
- Original folder structure remains intact for non-Catalan data

**Example output:**
```
Found 4 Catalan EMSR codes:
  - EMSR259
  - EMSR365
  - EMSR578
  - EMSR581

SUMMARY
  train: 3 EMSR folders moved
  val: 0 EMSR folders moved
  test: 1 EMSR folders moved
  Total: 4 EMSR folders
```

**Resulting structure:**
```
wildfires-cems/data/
├── train/train/      # Non-Catalan training data
├── val/val/          # Non-Catalan validation data
├── test/test/        # Non-Catalan test data
└── cat/              # Catalan data (moved here)
    ├── train/        # Catalan training samples
    └── test/         # Catalan test samples
```

### Command Line Options

```bash
uv run python remove_catalan_data.py --help

Options:
  --dataset-root PATH   Path to wildfires-cems/data/ directory
  --csv-path PATH       Path to satelliteData.csv
  --dry-run             Only print what would be moved (default)
  --execute             Actually move the folders
```

### Customizing for Other Regions

To filter by a different region, modify the `CATALONIA_BOUNDS` in `remove_catalan_data.py`:

```python
# Example: Filter fires in Greece
CATALONIA_BOUNDS = {
    "lon_min": 19.0,
    "lon_max": 29.7,
    "lat_min": 34.8,
    "lat_max": 41.8,
}
```

You can also filter by country name directly using the `country` column in `satelliteData.csv`.

---

## Using the Patches for Training

### PyTorch Dataset

```python
from dataset import WildfirePatchDataset, WildfireDataModule

# Option 1: Single dataset
dataset = WildfirePatchDataset(
    patches_dir="./patches/train",
    transform=None,           # Optional: torchvision transforms
    target_transform=None,    # Optional: mask transforms
)

image, mask = dataset[0]
# image: torch.Tensor of shape (7, 256, 256) or (8, 256, 256), dtype float32
# mask: torch.Tensor of shape (256, 256), dtype int64

# Option 2: Full data module with all splits
data_module = WildfireDataModule(
    patches_root="./patches",
    batch_size=32,
    num_workers=4,
)

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

### Computing Normalization Statistics

```python
from dataset import compute_dataset_statistics

stats = compute_dataset_statistics("./patches/train")
print(f"Mean: {stats['mean']}")  # Shape: (7,) or (8,) depending on patches
print(f"Std:  {stats['std']}")   # Shape: (7,) or (8,)

# Use in transforms
from torchvision import transforms

normalize = transforms.Normalize(
    mean=stats['mean'].tolist(),
    std=stats['std'].tolist(),
)
```

### Example Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WildfirePatchDataset

# Load data
train_dataset = WildfirePatchDataset("./patches/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define model (example: simple U-Net)
model = YourSegmentationModel(in_channels=8, out_channels=2)  # 8 with NDVI; 2 classes for DEL
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for images, masks in train_loader:
        # images: (B, 7, 256, 256) or (B, 8, 256, 256)
        # masks: (B, 256, 256) with values 0 or 1

        optimizer.zero_grad()
        outputs = model(images)  # (B, 2, 256, 256)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
```

### Mask Values Reference

**DEL (Delineation) - Binary Fire Detection:**
| Value | Meaning |
|-------|---------|
| 0 | Not burned |
| 1 | Burned |

**GRA (Grading) - Severity Assessment:**
| Value | Meaning | Description |
|-------|---------|-------------|
| 0 | No damage | Unaffected area |
| 1 | Negligible | Minimal visible damage |
| 2 | Moderate | Possibly damaged |
| 3 | High | Clearly damaged |
| 4 | Destroyed | Complete destruction |

---

## Class Imbalance Analysis

After generating patches, analyze the class distribution to determine the right strategy for handling imbalanced data. Fire pixels are typically rare compared to background pixels.

### Running the Analysis

```bash
# Analyze all splits
uv run python analyze_patches.py ./patches

# Analyze only training set
uv run python analyze_patches.py ./patches --split train

# For GRA masks (5 severity classes)
uv run python analyze_patches.py ./patches --num-classes 5

# Generate distribution plots
uv run python analyze_patches.py ./patches --plot
uv run python analyze_patches.py ./patches --output-plot class_distribution.png
```

### What It Reports

**Patch-level statistics:**
- How many patches contain fire vs no fire
- Distribution of fire coverage per patch

**Pixel-level statistics:**
- Total pixels per class
- Imbalance ratio (background:fire)

**Recommendations:**
- Suggested class weights for loss function
- Whether to use weighted sampling
- Augmentation strategy suggestions

### Example Output

```
📦 PATCH-LEVEL STATISTICS (1,234 patches)
  🔥 Patches with ANY fire:    892 (72.3%)
  🔥 Patches with >1% fire:    756 (61.3%)
  ⬜ Patches with NO fire:     342 (27.7%)
  Mean fire fraction:   12.45%

🔢 PIXEL-LEVEL STATISTICS (80,936,960 pixels)
  Background:   72,843,264 pixels (90.00%)
  Fire:          8,093,696 pixels (10.00%)

  ⚖️  Imbalance ratio: 9.0:1 (background:fire)

⚖️  RECOMMENDED CLASS WEIGHTS
  Inverse frequency weights:
    Background:   1.00
    Fire:         9.00

📋 SUGGESTED STRATEGY:
  ✓ Use WEIGHTED SAMPLING (fire patches are minority)
  ✓ Use MODERATE LOSS WEIGHTING
  ✓ Use FIRE-AWARE AUGMENTATION
```

### Using the Recommendations

```python
import torch
from dataset import WildfireDataModule, compute_class_weights

# Get recommended class weights
weights = compute_class_weights("./patches/train", num_classes=2)

# Use in loss function
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))

# Configure data module based on recommendations
dm = WildfireDataModule(
    "./patches",
    batch_size=32,
    use_weighted_sampling=True,    # Oversample fire patches
    fire_sample_weight=5.0,        # Adjust based on analysis
)
```

---

## Data Augmentation

The dataset supports on-the-fly augmentation via [Albumentations](https://albumentations.ai/), which handles multi-channel (7-band) images and applies synchronized transforms to both image and mask.

### Default Augmentation

`WildfireDataModule` applies standard augmentation to training data by default:

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Horizontal flip | p=0.5 | Spatial invariance |
| Vertical flip | p=0.5 | Spatial invariance |
| Random 90° rotation | p=0.5 | Rotational invariance |
| Brightness/Contrast | ±10%, p=0.3 | Atmospheric variation |
| Gaussian noise | σ=0.01, p=0.3 | Noise robustness |

### Using Augmentation

```python
from dataset import (
    WildfireDataModule,
    WildfirePatchDataset,
    get_training_augmentation,
    get_strong_augmentation,
)

# Option 1: Default augmentation (automatically enabled)
dm = WildfireDataModule("./patches", batch_size=32)

# Option 2: Explicit control
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),
)

# Option 3: Disable augmentation
dm = WildfireDataModule("./patches", train_augment=None)
```

### Fire-Aware Augmentation

Apply stronger augmentation to patches containing fire (helps with class imbalance):

```python
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),      # Standard for non-fire patches
    fire_augment=get_strong_augmentation(),   # Stronger for fire patches
    fire_threshold=0.01,                      # 1% fire pixels triggers strong aug
)

# Or with the data module
dm = WildfireDataModule(
    "./patches",
    batch_size=32,
    fire_augment=get_strong_augmentation(),
    use_weighted_sampling=True,
    fire_sample_weight=5.0,
)
```

### Strong Augmentation

`get_strong_augmentation()` includes everything in standard augmentation plus:
- Higher probabilities (p=0.5-0.7)
- Elastic deformation (α=100, σ=10, p=0.3)
- Stronger brightness/contrast (±15%)

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

Augmentation and normalization can be combined:

```python
from torchvision import transforms
from dataset import WildfirePatchDataset, compute_dataset_statistics, get_training_augmentation

# Compute dataset statistics
stats = compute_dataset_statistics("./patches/train")

# Normalization transform (applied after augmentation, to tensors)
normalize = transforms.Normalize(
    mean=stats['mean'].tolist(),
    std=stats['std'].tolist(),
)

# Combined pipeline
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),  # Applied to numpy arrays
    transform=normalize,                   # Applied to tensors
)
```

---

## Model Training

A complete training pipeline is included for training fire segmentation models. For the **recommended workflow** (CEMS → Sen2Fire fine-tune with dual-head, NDVI, and s2cloudless), see [Full workflow: CEMS + Sen2Fire](#full-workflow-cems--sen2fire-recommended) above.

### Quick Start

```bash
# Install training dependencies
uv sync --extra train

# Train with binary fire detection (DEL mask)
uv run python train.py --patches-dir ./patches --num-classes 2

# Train with severity levels (GRA mask)
uv run python train.py --patches-dir ./patches --num-classes 5

# With W&B logging
uv run python train.py --patches-dir ./patches --wandb --project fire-detection
```

### Model Architecture

The model performs **both tasks with a single forward pass**:

```
                                    ┌─────────────────────┐
   256×256×7        U-Net           │  256×256 mask       │
   ─────────►    Segmentation  ───► │  (class per pixel)  │
                                    └─────────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                    Binary Detection                   Severity Map
                    (derived: any fire?)               (full output)
```

- **Encoder**: Pretrained ResNet34/EfficientNet (adapts to 7 input channels)
- **Decoder**: U-Net style with skip connections
- **Output**: (B, num_classes, 256, 256) segmentation logits

### Training Options

```bash
uv run python train.py --help

Key options:
  --patches-dir PATH      Patches directory (default: ./patches)
  --output-dir PATH       Output for checkpoints (required)
  --overwrite-output-dir  Allow overwriting if output dir exists
  --num-classes {2,5}     2=binary DEL, 5=severity GRA
  --encoder NAME          resnet34, efficientnet-b0, etc.
  --architecture ARCH     unet, unetplusplus, deeplabv3plus
  --batch-size N          Batch size (default: 16)
  --epochs N              Number of epochs (default: 50)
  --lr RATE               Learning rate (default: 1e-4)
  --focal-loss            Use focal loss for class imbalance
  --weighted-sampling     Oversample fire patches
  --wandb                 Enable W&B logging
  --resume PATH           Resume from checkpoint
```

### Using the Trained Model

```python
import torch
from model import FireSegmentationModel

# Load trained model
model = FireSegmentationModel(num_classes=5, encoder_name="resnet34")
checkpoint = torch.load("output/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    # Segmentation (pixel-wise severity)
    segmentation = model.predict_segmentation(images)  # (B, 256, 256)

    # Binary detection (derived from segmentation)
    has_fire = model.predict_fire_detection(images)  # (B,) boolean

    # Confidence score
    confidence = model.predict_fire_confidence(images)  # (B,) float 0-1
```

### Loss Functions

The training script supports multiple loss functions for handling class imbalance:

| Loss | Use Case |
|------|----------|
| **CrossEntropy + Dice** | Default, good balance |
| **Focal Loss** | Severe class imbalance |
| **Weighted CE** | Moderate imbalance |

```bash
# Use focal loss
uv run python train.py --focal-loss --focal-gamma 2.0

# Disable class weights
uv run python train.py --no-class-weights
```

### Metrics

Training reports both **segmentation** and **detection** metrics:

**Segmentation Metrics:**
- Mean IoU (Intersection over Union)
- Fire IoU (IoU for fire classes only)
- Fire Dice coefficient
- Fire Precision/Recall

**Detection Metrics (derived):**
- Detection Accuracy
- Detection Precision
- Detection Recall
- Detection F1

---

## Inference Pipeline

The inference pipeline allows you to run trained models on new satellite imagery.

### Using the Inference Pipeline

```python
from inference import FireInferencePipeline, create_visualization
import numpy as np

# Load trained model
pipeline = FireInferencePipeline("output/checkpoints/best_model.pt")

# Run inference on numpy array (H, W, 7) or (H, W, 12)
result = pipeline.predict_from_array(image_array)

# Or from a GeoTIFF file
result = pipeline.predict_from_file("satellite_image.tif")

# Access results
print(f"Fire detected: {result.has_fire}")
print(f"Confidence: {result.fire_confidence:.1%}")
print(f"Fire coverage: {result.fire_fraction:.2%}")
print(f"Severity counts: {result.severity_counts}")

# Get outputs
segmentation = result.segmentation      # (H, W) class indices
probabilities = result.probabilities    # (H, W, num_classes)
fire_mask = result.get_fire_mask()      # Binary mask
severity_map = result.get_severity_map()

# Create visualization overlay
visualization = create_visualization(image_array, result, alpha=0.5)
```

### Command Line Usage

```bash
uv run python inference.py checkpoints/best_model.pt satellite_image.tif
```

---

## Streamlit Web App

An interactive web application for fetching satellite imagery and running fire detection.

### Features

- **Interactive map** for selecting regions of interest
- **Satellite data fetching** from Microsoft Planetary Computer
- **Real-time inference** with fire detection and severity mapping
- **Analysis history** with filtering by date, location, and fire detection
- **Local caching** of imagery and results
- **Statistics dashboard** with detection summaries

### Running the App

```bash
# Install app dependencies
uv sync --extra app

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### App Configuration

Set via environment variables (or edit the top of `app.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FIRE_STORAGE_DIR` | `./cache` | Where to cache data |
| `FIRE_MODELS_DIR` | (unset) | Base dir for multiple models; when set, sidebar shows a dropdown of subdirs (e.g. `./output` → `./output/resnet34-unet/checkpoints/`) |
| `FIRE_MODEL_NAME` | `best_model` | Checkpoint filename without `.pt` (e.g. `best_model` → `best_model.pt`) |
| `FIRE_MODEL_PATH` | `./checkpoints/best_model.pt` | Fallback when `FIRE_MODELS_DIR` is not used |
| `FIRE_USE_MOCK` | `true` | Set `false` for real Sentinel-2 imagery |

### Using Real Satellite Data

To fetch actual Sentinel-2 imagery from Microsoft Planetary Computer:

1. Install the required packages:
   ```bash
   uv sync --extra app
   ```

2. Set `USE_MOCK_FETCHER = False` in `app.py`

3. The app will automatically handle authentication with Planetary Computer

### Mock Mode (for Development)

By default, the app runs in mock mode which:
- Generates synthetic satellite imagery
- Creates fake fire detections for UI testing
- Requires no API access or credentials

This is useful for developing and testing the UI without consuming API resources.

### Storage Structure

```
cache/
├── images/           # Raw satellite imagery (.npy)
├── results/          # Inference results (.npz)
├── visualizations/   # Generated overlays (.png)
└── database.db       # SQLite metadata
```

---

## Visualization Tools

### Command Line Interface

```bash
# View a single patch with all RGB composites and mask overlay
uv run python visualize.py patch EMSR230_AOI01_01_r128_c256 \
    --patches-dir ./patches/train \
    --mask-type DEL \
    --save patch_view.png

# Generate a grid of sample patches (great for quality checking)
uv run python visualize.py grid ./patches/train \
    --n-samples 16 \
    --mode true_color \
    --mask-type DEL \
    --save sample_grid.png

# Plot dataset statistics (burn distribution, cloud cover)
uv run python visualize.py stats ./patches/train \
    --save statistics.png

# Visualize a full source GeoTIFF (before patching)
uv run python visualize.py source \
    ../wildfires-cems/data/train/EMSR230/AOI01/EMSR230_AOI01_01/EMSR230_AOI01_01_S2L2A.tif \
    --mask ../wildfires-cems/data/train/EMSR230/AOI01/EMSR230_AOI01_01/EMSR230_AOI01_01_DEL.tif \
    --save source_view.png
```

### Python API

```python
from visualize import (
    plot_patch,
    plot_all_bands,
    plot_sample_grid,
    plot_dataset_statistics,
    plot_source_image,
    create_rgb_composite,
)
import numpy as np

# Load a patch
image = np.load("./patches/train/EMSR230_AOI01_01_r0_c0_image.npy")
mask = np.load("./patches/train/EMSR230_AOI01_01_r0_c0_mask.npy")

# Visualize single patch
plot_patch(image, mask, mask_type="DEL", title="Sample Patch")

# View all 7 bands individually
plot_all_bands(image, title="Spectral Bands")

# Create RGB composite for custom visualization
rgb = create_rgb_composite(image, mode="false_color")  # Returns (256, 256, 3)
```

### RGB Composite Modes

| Mode | Bands Used | Best For |
|------|------------|----------|
| `true_color` | B04-B03-B02 (R-G-B) | Natural appearance |
| `false_color` | B08-B04-B03 (NIR-R-G) | Vegetation (appears red) |
| `swir` | B12-B08-B04 (SWIR2-NIR-R) | Fire scars (appear cyan/blue) |

---

## Technical Reference

### Image Data Format

| Property | Value |
|----------|-------|
| **Shape** | (256, 256, 7) or (256, 256, 8) with NDVI |
| **Data type** | float32 |
| **Value range** | [0, 1] (clipped); NDVI raw [−1, 1] mapped to [0, 1] |
| **Channel order** | B02, B03, B04, B08, B8A, B11, B12 [, NDVI] |

### Checkpoint config (saved with model)

Training saves a `config` dict in each checkpoint. Inference uses it to rebuild the model:

| Key | Description | Default if missing |
|-----|-------------|--------------------|
| `num_classes` | 2 (DEL) or 5 (GRA) | 2 |
| `in_channels` | 7 or 8 (with NDVI) | 7 |
| `encoder_name` | Backbone (e.g. resnet34) | resnet34 |
| `architecture` | unet, unetplusplus, deeplabv3plus | unet |

Old checkpoints without `in_channels` are loaded with 7 channels. New training runs save `in_channels: 8` when using the default (NDVI enabled).

### Coordinate Reference System

- **CRS**: EPSG:4326 (WGS84 geographic coordinates)
- **Units**: Degrees latitude/longitude
- **Resolution**: ~10m × 10m per pixel (varies slightly by latitude)

### Patch Coverage

At 10m resolution:
- **256×256 patch** ≈ 2.56 km × 2.56 km ground coverage

### Expected Dataset Size

| Split | Estimated Patches | Estimated Size |
|-------|-------------------|----------------|
| Train | ~15,000-20,000 | ~25-35 GB |
| Val | ~2,000-3,000 | ~4-6 GB |
| Test | ~3,000-5,000 | ~6-10 GB |

*Actual numbers depend on cloud filtering and image sizes.*

---

## Troubleshooting

### "No patches found" error

1. Ensure the tar archives are extracted:
   ```bash
   ls ../wildfires-cems/data/train/EMSR*/
   ```
   You should see `AOI*` directories, not just `.tar.*.gz.part` files.

2. Run extraction manually:
   ```bash
   cd ../wildfires-cems
   cat data/train/train.tar.*.gz.part | tar -xzvf - -i -C data/train/
   ```

### Git LFS issues

If you see small text files instead of actual data:
```bash
cd ../wildfires-cems
git lfs pull
```

### Memory issues during patch generation

Process one split at a time:
```bash
uv run python run_pipeline.py --splits train --output-dir ./patches
uv run python run_pipeline.py --splits val --output-dir ./patches
uv run python run_pipeline.py --splits test --output-dir ./patches
```

### Cloud filtering removing too many patches

Adjust the threshold:
```bash
uv run python run_pipeline.py --max-cloud-cover 0.7  # Allow up to 70% cloud
```

---

## Testing

The project includes comprehensive unit and integration tests.

### Running Tests

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_inference.py

# Run with coverage report
uv run pytest --cov=. --cov-report=html
```

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_inference.py     # Inference pipeline tests
├── test_storage.py       # Storage system tests
├── test_satellite_fetcher.py  # Satellite fetcher tests
├── test_metrics.py       # Evaluation metrics tests
└── test_integration.py   # End-to-end integration tests
```

### What's Tested

| Component | Coverage |
|-----------|----------|
| Inference pipeline | Preprocessing, patch extraction, stitching |
| Storage system | Save/load, history, path validation |
| Satellite fetcher | Mock data generation, scene search |
| Metrics | Segmentation IoU/Dice, detection F1 |
| Model | Forward pass, loss functions |
| Dataset | Loading, augmentation |

---

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{arnaudo2023burned,
  title={Robust Burned Area Delineation through Multitask Learning},
  author={Arnaudo, Edoardo and Barco, Luca and Merlo, Matteo and Rossi, Claudio},
  booktitle={Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  year={2023}
}
```

---

## License

- **Dataset**: CC-BY-4.0 (Copernicus Emergency Management Service)
- **Pipeline code**: MIT
