# Training All Models - Summary

## What Was Updated

The `train_all_models.py` script and `TRAINING_ALL_MODELS.md` documentation have been updated to include baseline models training.

## New Features

### Added Support For:
- ✅ **YOLOv8 Baseline**: Detection/segmentation model (optional with `--include-yolo`)
- ✅ **Scratch Model**: Binary fire classifier from scratch (optional with `--include-scratch`)
- ✅ **Flexible Control**: Train subsets with `--segmentation-only` flag

### Total Models Available:
- **18 Segmentation models** (6 encoders × 3 architectures)
- **1 YOLOv8 baseline** (optional)
- **1 Scratch classifier** (optional)
- **Total: 20 models** when all options enabled

## Quick Start Examples

### Train All 20 Models
```bash
cd fire-pipeline
python train_all_models.py \
  --patches-dir ../patches \
  --num-classes 2 \
  --epochs 50 \
  --include-yolo \
  --include-scratch
```

### Train Only Segmentation (18 Models)
```bash
python train_all_models.py \
  --patches-dir ../patches \
  --num-classes 2 \
  --epochs 50
```

### Train Segmentation + YOLOv8 (19 Models)
```bash
python train_all_models.py \
  --patches-dir ../patches \
  --num-classes 2 \
  --epochs 50 \
  --include-yolo
```

### Dry Run (Preview All Commands)
```bash
python train_all_models.py \
  --patches-dir ../patches \
  --include-yolo \
  --include-scratch \
  --dry-run
```

## Output Structure

All models save to unique timestamped directories to prevent overwriting:

```
output/
└── training_run_20260207_124743/
    ├── encoder_resnet18-architecture_unet/
    ├── encoder_resnet18-architecture_unetplusplus/
    ├── encoder_resnet34-architecture_unet/
    ├── ... (18 total segmentation combinations)
    ├── yolo_seg/                    # Optional YOLOv8
    └── scratch_model/               # Optional Scratch model
```

## Expected Training Time

- **Segmentation only (18 models)**: ~15-20 GPU hours
- **+YOLOv8 (19 models)**: ~17-22 GPU hours
- **+Scratch (19 models)**: ~16-21 GPU hours
- **All (20 models)**: ~18-23 GPU hours

Use `--epochs 10` for quick testing (~3-4 GPU hours)

## New Command Line Arguments

```
--include-yolo          Train YOLOv8 detection baseline
--include-scratch       Train scratch model (binary classifier)
--segmentation-only     Skip baseline models (only segmentation)
```

## No Data Loss

Each training run uses a unique timestamp-based output directory. **No existing outputs will be overwritten**, so you can safely:
- Re-run the script multiple times
- Train different model subsets
- Compare results across runs

## Documentation

Full details available in:
- `TRAINING_ALL_MODELS.md` - Complete guide with examples
- `train_all_models.py --help` - Command line help
