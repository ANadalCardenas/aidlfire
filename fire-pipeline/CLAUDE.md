# Claude Context - Fire Pipeline Project

This file provides context for Claude Code and other AI assistants working on this project.

## Project Purpose

This is a **complete fire detection pipeline** for training and deploying deep learning models to detect/segment wildfires from Sentinel-2 satellite imagery. It processes the CEMS Wildfire Dataset into patches and includes:

- Data preparation and patch generation
- U-Net segmentation model with pretrained encoders
- Training script with loss functions and metrics
- Inference pipeline for new satellite imagery
- Streamlit web app for interactive analysis

## Key Technical Details

### Band Selection
From the original 12 Sentinel-2 bands, we select 7 useful ones. When NDVI is enabled (default), an 8th channel is added.

```python
band_indices = (1, 2, 3, 7, 8, 10, 11)  # 0-indexed
NUM_INPUT_CHANNELS = 8   # 7 bands + NDVI (use 7 if include_ndvi=False)
RED_INDEX_7, NIR_INDEX_7 = 2, 3  # For NDVI: (NIR - Red) / (NIR + Red)
```

| Index | Band | Wavelength | Purpose |
|-------|------|------------|---------|
| 0 | B02 (Blue) | 490nm | Water, smoke |
| 1 | B03 (Green) | 560nm | Vegetation |
| 2 | B04 (Red) | 665nm | Burn scars |
| 3 | B08 (NIR) | 842nm | Healthy vegetation (HIGH when healthy) |
| 4 | B8A (NIR narrow) | 865nm | Vegetation boundary |
| 5 | B11 (SWIR1) | 1610nm | **Fire detection** (HIGH when burned) |
| 6 | B12 (SWIR2) | 2190nm | **Fire detection** (HIGH when burned) |
| 7 | NDVI | (NIR−Red)/(NIR+Red) | Vegetation index (when enabled) |

**Fire signature**: LOW NIR (bands 3-4) + HIGH SWIR (bands 5-6) = burned area

### Patch Generation
- **Patch size**: 256×256 pixels
- **Training stride**: 128 (50% overlap for more samples)
- **Inference stride**: 256 (no overlap)
- **Output format**: NumPy arrays (.npy), NOT images
- **Image shape**: (256, 256, 7) or (256, 256, 8) float32, values 0-1 (NDVI stored as [0,1] from raw [−1,1])
- **Mask shape**: (256, 256) uint8, values 0-1 (DEL) or 0-4 (GRA)
- **NDVI**: Default on; disable with `PatchConfig(include_ndvi=False)` or `run_pipeline.py --no-ndvi`

### Mask Types
- **DEL (Delineation)**: Binary mask (0=no fire, 1=fire)
- **GRA (Grading)**: Severity levels (0=none, 1=minimal, 2=moderate, 3=high, 4=destroyed)
- Not all images have GRA masks - check `satelliteData.csv` column `GRA`

## File Locations

### Pipeline Scripts (fire-pipeline/)
| File | Purpose |
|------|---------|
| `run_pipeline.py` | CLI entry point for patch generation |
| `patch_generator.py` | Core patching logic |
| `dataset.py` | PyTorch Dataset/DataLoader classes |
| `visualize.py` | Visualization utilities |
| `remove_catalan_data.py` | Move Catalan fires to separate folder |

### Dataset Location
```
../wildfires-cems/data/
├── train/train/EMSR*/AOI*/   # Training images
├── val/val/EMSR*/AOI*/       # Validation images
├── test/test/EMSR*/AOI*/     # Test images
└── cat/                       # Catalan data (moved separately)
```

### Metadata CSVs (../csv_files/)
| File | Rows | Purpose |
|------|------|---------|
| `satelliteData.csv` | 560 | Per-image metadata (coords, country, bands, etc.) |
| `datasetPreConfigured.csv` | 274 | Pre/post fire image pairs |
| `cloudCoverage.csv` | 998 | Cloud coverage statistics |

### Documentation
| File | Content |
|------|---------|
| `../AIDL_PROJECT_GUIDE.md` | Beginner-friendly explanations with diagrams |
| `../DATA_REFERENCE.md` | Technical data reference |
| `PATCHES.md` | Patch format and loading details |
| `README.md` | Pipeline usage instructions |

## Common Tasks

### Download Dataset (for cloud deployment)
```bash
# Install download dependencies
uv sync --extra download

# Download, extract, and generate patches in one command
uv run python download_dataset.py --generate-patches --patches-dir ./patches

# Or download only
uv run python download_dataset.py --output-dir ../wildfires-cems
```

### Generate Patches (if data already downloaded)
```bash
cd fire-pipeline
uv run python run_pipeline.py --skip-extraction --output-dir ./patches
# 8 channels (7 bands + NDVI) by default; use --no-ndvi for 7 channels only
```

### Load Patches for Training
```python
from dataset import WildfirePatchDataset, WildfireDataModule

# Single dataset
dataset = WildfirePatchDataset("./patches/train")
image, mask = dataset[0]  # (7,256,256) or (8,256,256), (256,256)

# Full data module (with default augmentation)
dm = WildfireDataModule("./patches", batch_size=32)
train_loader = dm.train_dataloader()
```

### Data Augmentation
```python
from dataset import (
    WildfirePatchDataset,
    WildfireDataModule,
    get_training_augmentation,
    get_strong_augmentation,
    compute_class_weights,
)

# Standard augmentation (flips, rotations, brightness, noise)
augment = get_training_augmentation()
dataset = WildfirePatchDataset("./patches/train", augment=augment)

# Fire-aware augmentation (stronger for fire patches)
dataset = WildfirePatchDataset(
    "./patches/train",
    augment=get_training_augmentation(),
    fire_augment=get_strong_augmentation(),
    fire_threshold=0.01,
)

# Full setup with weighted sampling
dm = WildfireDataModule(
    "./patches",
    batch_size=32,
    train_augment=get_training_augmentation(),
    fire_augment=get_strong_augmentation(),
    use_weighted_sampling=True,
    fire_sample_weight=5.0,
)

# Compute class weights for loss function
weights = compute_class_weights("./patches/train", num_classes=2)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

### Check Dataset Statistics
```python
import pandas as pd
df = pd.read_csv("../csv_files/satelliteData.csv")
print(df['country'].value_counts())  # Fires by country
print(df['pixelBurned'].describe())  # Burn statistics
```

### Analyze Class Imbalance (after patching)
```bash
# Get recommendations for handling class imbalance
uv run python analyze_patches.py ./patches

# With plots
uv run python analyze_patches.py ./patches --plot --output-plot distribution.png
```

### Train Model
```bash
# Install training dependencies
uv sync --extra train

# Train with binary fire mask (DEL)
uv run python train.py --patches-dir ./patches --num-classes 2

# Train with severity mask (GRA)
uv run python train.py --patches-dir ./patches --num-classes 5 --focal-loss

# With W&B logging
uv run python train.py --patches-dir ./patches --wandb --project fire-detection
```

### Use Trained Model
```python
from model import FireSegmentationModel
import torch

model = FireSegmentationModel(num_classes=5)
checkpoint = torch.load("output/checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Both tasks in one pass
segmentation = model.predict_segmentation(images)  # (B, 256, 256)
has_fire = model.predict_fire_detection(images)    # (B,) boolean
confidence = model.predict_fire_confidence(images) # (B,) float
```

## Catalonia Region
Catalan fires were moved to `wildfires-cems/data/cat/` for regional testing:
- Bounding box: lon 0.15°-3.35°, lat 40.5°-42.9°
- EMSR codes: EMSR259, EMSR365, EMSR578, EMSR581

## Things to Remember

1. **Labels are masks** - DEL/GRA TIF files are the ground truth labels
2. **Labels are human-made** - CEMS analysts manually drew burn polygons
3. **Cloud mask matters** - Cloudy pixels are unreliable (CM files)
4. **~4% have no fire** - Some edge tiles have pixelBurned=0
5. **GRA looks black** - Values 0-4 appear black in viewers (scale to 0-255 to see)
6. **Model IS included** - `model.py` provides U-Net with pretrained encoders
7. **Shared constants** - Use `constants.py` for band indices, class names, device selection

## PyTorch Tensor Shapes

```python
# After DataLoader
images.shape  # (batch, 7, 256, 256) or (batch, 8, 256, 256) - channels first
masks.shape   # (batch, 256, 256) - class indices

# For segmentation model
output = model(images)  # (batch, num_classes, 256, 256)
loss = criterion(output, masks)  # CrossEntropyLoss expects this format
```
