# AI Agents Guide - CEMS Wildfire Detection Project

This document helps AI agents quickly understand and navigate this project.

## Quick Context

**What is this?** A satellite imagery dataset and processing pipeline for training wildfire detection/segmentation models.

**Goal**: Detect burned areas in Sentinel-2 satellite images using deep learning.

**Current state**: Complete end-to-end system including data prep, training, inference, and web app.

## Project Structure

```
CEMS-Wildfire-Dataset-main/
│
├── fire-pipeline/              # MAIN CODE DIRECTORY
│   ├── run_pipeline.py         # Entry point - generates patches
│   ├── patch_generator.py      # Core logic - cuts images into 256×256
│   ├── dataset.py              # PyTorch Dataset with augmentation
│   ├── model.py                # U-Net segmentation model
│   ├── train.py                # Training script with W&B
│   ├── metrics.py              # Evaluation metrics
│   ├── inference.py            # Inference pipeline
│   ├── app.py                  # Streamlit web app
│   ├── satellite_fetcher.py    # Fetch Sentinel-2 from Planetary Computer
│   ├── storage.py              # Local cache/storage system
│   ├── visualize.py            # Visualization tools
│   ├── analyze_patches.py      # Class imbalance analysis
│   ├── download_dataset.py     # Download from HuggingFace
│   ├── remove_catalan_data.py  # Regional data separation
│   ├── PATCHES.md              # Patch format documentation
│   └── README.md               # Usage instructions
│
├── wildfires-cems/data/        # RAW SATELLITE DATA
│   ├── train/train/EMSR*/      # ~400 training images
│   ├── val/val/EMSR*/          # ~80 validation images
│   ├── test/test/EMSR*/        # ~80 test images
│   └── cat/                    # Catalan regional data (held out)
│
├── csv_files/                  # METADATA
│   ├── satelliteData.csv       # Main metadata (560 rows)
│   ├── datasetPreConfigured.csv
│   └── cloudCoverage.csv
│
├── AIDL_PROJECT_GUIDE.md       # Beginner-friendly guide with diagrams
├── DATA_REFERENCE.md           # Technical data documentation
└── AGENTS.md                   # This file
```

## Key Concepts (Must Understand)

### 1. Input Data
- **Source**: Sentinel-2 satellite (12 spectral bands)
- **Selected bands**: 7 of 12 (indices 1,2,3,7,8,10,11); optional **NDVI** as 8th channel (default: on)
- **File format**: GeoTIFF (.tif)
- **Naming**: `EMSR{id}_AOI{area}_{tile}_S2L2A.tif`
- **Constants**: `NUM_INPUT_CHANNELS=8`, `include_ndvi=True` in `PatchConfig`; use `--no-ndvi` for 7 channels

### 2. Labels (Ground Truth)
- **DEL mask**: Binary (0=no fire, 1=fire) - use for segmentation
- **GRA mask**: Severity (0-4 levels) - optional, not always available
- **Created by**: Human experts at CEMS (not algorithms)

### 3. Patches
- **Size**: 256×256 pixels
- **Format**: NumPy arrays (.npy)
- **Image**: (256, 256, 7) or (256, 256, 8) float32, normalized 0-1 (8 = 7 bands + NDVI when `include_ndvi=True`)
- **Mask**: (256, 256) uint8

### 4. Fire Detection Signal
```
Burned areas show:
  - LOW values in NIR bands (3-4)
  - HIGH values in SWIR bands (5-6)
```

## Common Agent Tasks

### Task: Explore the dataset
```
Read: csv_files/satelliteData.csv
Key columns: EMSR, country, pixelBurned, koppen_subgroup, GRA, DEL
```

### Task: Understand data format
```
Read: fire-pipeline/PATCHES.md
Read: AIDL_PROJECT_GUIDE.md
```

### Task: Download dataset (cloud deployment)
```bash
cd fire-pipeline
uv sync --extra download
uv run python download_dataset.py --generate-patches --patches-dir ./patches
```

### Task: Generate training patches (if data already exists)
```bash
cd fire-pipeline
uv run python run_pipeline.py --skip-extraction --output-dir ./patches
# Default: 8 channels (7 bands + NDVI). Use --no-ndvi for 7 channels only.
```

### Task: Run inference on new images
```python
from inference import FireInferencePipeline

pipeline = FireInferencePipeline("checkpoints/best_model.pt")
result = pipeline.predict_from_file("satellite_image.tif")
print(f"Fire detected: {result.has_fire}")
print(f"Confidence: {result.fire_confidence:.1%}")
```

### Task: Run the Streamlit app
```bash
cd fire-pipeline
uv sync --extra app
streamlit run app.py
```

### Task: Use data augmentation
```python
from dataset import get_training_augmentation, get_strong_augmentation

# Augmentation is enabled by default in WildfireDataModule
# For fire-aware augmentation (stronger for fire patches):
dm = WildfireDataModule(
    "./patches",
    fire_augment=get_strong_augmentation(),
    use_weighted_sampling=True,
)
```

### Task: Analyze class imbalance
```bash
# Run after generating patches to get recommendations
uv run python analyze_patches.py ./patches --plot
```

### Task: Train fire detection model
```bash
cd fire-pipeline
uv sync --extra train
uv run python train.py --patches-dir ./patches --output-dir ./output/run1 --num-classes 2 --wandb
```

### Task: Combined binary + severity workflow (CEMS + Sen2Fire)
```bash
cd fire-pipeline
# Phase 1: Binary on CEMS DEL + Sen2Fire
uv run python train_combined_binary.py --patches-dir ./patches --sen2fire-dir ../data-sen2fire --output-dir ./output/combined_binary
# Phase 2: Severity on CEMS GRA only
uv run python train_severity_finetune.py --checkpoint ./output/combined_binary/checkpoints/best_model.pt --patches-dir ./patches_gra --output-dir ./output/severity_finetune
```
See `docs/COMBINED_BINARY_SEVERITY_WORKFLOW.md` for full details.

## File Quick Reference

| If you need... | Look at... |
|----------------|------------|
| Technical specs | `fire-pipeline/PATCHES.md` |
| Beginner explanations | `AIDL_PROJECT_GUIDE.md` |
| Data file formats | `DATA_REFERENCE.md` |
| Patch details | `fire-pipeline/PATCHES.md` |
| Class imbalance analysis | `fire-pipeline/analyze_patches.py` |
| Augmentation code | `fire-pipeline/dataset.py` |
| Model architecture | `fire-pipeline/model.py` |
| Training script | `fire-pipeline/train.py` |
| Evaluation metrics | `fire-pipeline/metrics.py` |
| Pipeline usage | `fire-pipeline/README.md` |
| Combined binary + severity workflow | `docs/COMBINED_BINARY_SEVERITY_WORKFLOW.md` |
| V3 pipeline architectures & training commands | `docs/V3_PIPELINE_ARCHITECTURES.md` |
| Dataset loading | `fire-pipeline/dataset.py` |
| Patch generation | `fire-pipeline/patch_generator.py` |
| Inference pipeline | `fire-pipeline/inference.py` |
| Web app | `fire-pipeline/app.py` |
| Satellite data fetching | `fire-pipeline/satellite_fetcher.py` |
| Local storage/caching | `fire-pipeline/storage.py` |

## Important Numbers

| Metric | Value |
|--------|-------|
| Total satellite images | ~560 |
| Training images | ~400 |
| Validation images | ~80 |
| Test images | ~80 |
| Patch size | 256×256 |
| Input channels | 7 or 8 (8 = 7 bands + NDVI by default) |
| DEL classes | 2 (binary) |
| GRA classes | 5 (severity) |
| Countries covered | 19 (mostly Mediterranean Europe) |
| Date range | 2017-2023 |

## What's Included

- ✅ Data preparation pipeline (patching, augmentation)
- ✅ U-Net segmentation model (`model.py`)
- ✅ Training script with checkpointing (`train.py`)
- ✅ Loss functions (CrossEntropy, Dice, Focal)
- ✅ Evaluation metrics (`metrics.py`)
- ✅ W&B integration for experiment tracking
- ✅ Inference pipeline (`inference.py`)
- ✅ Streamlit web app (`app.py`)
- ✅ Satellite data fetching from Planetary Computer
- ✅ Local storage/caching system

## What's NOT Included

Optional future improvements:
- [ ] Model export (ONNX, TorchScript)
- [ ] Cloud deployment configurations

## Regional Notes

**Catalonia** (Spain) data has been separated to `data/cat/` for:
- Regional model testing
- Transfer learning experiments
- Out-of-distribution evaluation

Catalan EMSR codes: EMSR259, EMSR365, EMSR578, EMSR581

## Dependencies

```
Python 3.12+
rasterio      # GeoTIFF reading
numpy         # Array operations
torch         # PyTorch (training/inference)
pandas        # CSV handling
streamlit     # Web app (optional)
```

Install with:
```bash
cd fire-pipeline
uv sync                    # Base dependencies
uv sync --extra train      # + training deps
uv sync --extra app        # + Streamlit app deps
uv sync --extra all        # Everything
```

## Warnings for Agents

1. **Check GRA availability** - Not all images have severity masks
2. **Mind the cloud mask** - Cloudy pixels are unreliable
3. **Patches are NumPy, not images** - 7 or 8 channels, float32
4. **Labels are TIF files** - DEL.tif and GRA.tif are the masks
5. **~4% images have no fire** - Edge tiles with pixelBurned=0
6. **Mock mode by default** - App uses mock fetcher, set `USE_MOCK_FETCHER=False` for real data
