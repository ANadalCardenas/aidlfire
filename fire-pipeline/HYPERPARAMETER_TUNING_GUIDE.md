# Hyperparameter Tuning Guide — Fire Detection Pipeline

## Table of Contents
1. [How the tuning flow works](#1-how-the-tuning-flow-works)
2. [Prerequisites](#2-prerequisites)
3. [General parameters](#3-general-parameters)
4. [CNN Scratch](#4-cnn-scratch)
5. [UNet Scratch](#5-unet-scratch)
6. [YOLO](#6-yolo)
7. [Where to find the results](#7-where-to-find-the-results)

---

## 1. How the tuning flow works

The tuning process runs in three phases:

```
PHASE 1 — Search (Ray Tune)
  └─ N trials (full training runs) are executed, each with a different
     hyperparameter configuration sampled randomly from the search space.
     These trials do NOT log to W&B or the CSV (avoids noise in logs).

PHASE 2 — Selection
  └─ Once the search finishes, all trials are ranked by the main metric
     for each model (val_loss, fire_iou, or mAP@0.5).
     The TOP K best configurations are selected.

PHASE 3 — Top-K re-training
  └─ Each of the K best configurations is re-trained once in full,
     with W&B and CSV logging enabled.
     Result: K well-documented runs in W&B and K rows in the CSV,
     all corresponding to the best configurations found during search.
```

**Why this flow?**
- Tuning involves dozens of training runs. Sending all of them to W&B
  would generate a lot of noise and make comparisons very difficult.
- Re-running only the best configurations ensures that W&B and the CSV
  contain only relevant, comparable results.

**Main metric per model:**

| Model        | Metric       | Direction |
|--------------|--------------|-----------|
| CNN Scratch  | `val_f1`     | higher is better |
| UNet Scratch | `fire_iou`   | higher is better |
| YOLO         | `mAP@0.5`    | higher is better |

---

## 2. Prerequisites

**Install dependencies:**
```bash
uv sync --extra train
pip install "ray[tune]" wandb
```

**Generate patches** at `./patches` (or the path you specify):
```bash
cd fire-pipeline
uv run python run_pipeline.py --dataset-dir ../wildfires-cems --output-dir ../patches --mask-type DEL
```

**Log in to W&B** (first time only):
```bash
wandb login
```

---

## 3. General parameters

All tuning commands share these parameters:

| Parameter | Description | Default |
|---|---|---|
| `--patches-dir` | Directory with train/val/test patches | `./patches` |
| `--output-dir` | Directory for checkpoints and logs | (required) |
| `--tune true` | Enables hyperparameter tuning mode | `false` |
| `--tune-target` | Model to tune: `scratch`, `unet_scratch`, `yolo` | `seg` |
| `--tune-samples` | Number of trials to run during the search | `20` |
| `--tune-top-k` | How many of the best trials are re-run with W&B + CSV | `3` |
| `--epochs` | Epochs per trial (and per final re-training) | `50` |
| `--batch-size` | Base batch size (may be overridden by the tuner) | `16` |
| `--wandb` | Enables Weights & Biases logging | disabled |
| `--project` | W&B project name | (required with --wandb) |
| `--results-csv` | Path to the CSV file where results are appended | `training_results.csv` |
| `--device` | Device: `auto`, `cpu`, `0` (GPU), `mps` | `auto` |

---

## 4. CNN Scratch

**Model description:**
Binary fire classification model (fire / no fire) built from scratch.
Lightweight architecture: 1×1 spectral conv → 3 conv blocks → global pooling → classifier.

**Tuned hyperparameters:**

| Hyperparameter | Search space | Description |
|---|---|---|
| `learning_rate` | loguniform(5e-5, 5e-4) | Learning rate |
| `weight_decay` | loguniform(1e-6, 1e-3) | L2 regularization |
| `dropout` | choice([0.1, 0.2, 0.3, 0.4]) | Dropout probability in the classifier head |

**Command:**
```bash
cd fire-pipeline

uv run python train.py \
  --patches-dir ../patches \
  --output-dir ./output/scratch \
  --tune true \
  --tune-target scratch \
  --tune-samples 20 \
  --tune-top-k 3 \
  --epochs 30 \
  --wandb \
  --project fire-detection \
  --results-csv ./training_results.csv
```

**Estimated time:** ~20 trials × time per trial. For 30 epochs on a mid-size dataset, expect 2–4 hours on GPU.

---

## 5. UNet Scratch

**Model description:**
Per-pixel fire segmentation model (fire mask) built from scratch.
Classic U-Net architecture with its own encoder and decoder, no pretrained weights.

**Tuned hyperparameters:**

| Hyperparameter | Search space | Description |
|---|---|---|
| `learning_rate` | loguniform(5e-5, 5e-4) | Learning rate |
| `weight_decay` | loguniform(1e-6, 1e-3) | L2 regularization |
| `batch_size` | choice([8, 16, 32]) | Batch size (affects BatchNorm statistics) |

**Command:**
```bash
cd fire-pipeline

uv run python train.py \
  --patches-dir ../patches \
  --output-dir ./output/unet_scratch \
  --tune true \
  --tune-target unet_scratch \
  --tune-samples 15 \
  --tune-top-k 3 \
  --epochs 30 \
  --num-classes 2 \
  --wandb \
  --project fire-detection \
  --results-csv ./training_results.csv
```

**Estimated time:** ~15 trials. Slower than CNN Scratch (segmentation task). Expect 4–6 hours on GPU.

---

## 6. YOLO

**Model description:**
YOLOv8 object detection model adapted for 8-channel multispectral imagery.
Detects bounding boxes around burned areas. Uses pretrained YOLOv8n weights (ImageNet/COCO).

**Important note:** Before training, the pipeline automatically exports patches
to Ultralytics format (TIFF images + YOLO-format labels). This is done once
and reused across all trials.

**Tuned hyperparameters:**

| Hyperparameter | Search space | Description |
|---|---|---|
| `lr0` | loguniform(5e-4, 1e-2) | Initial learning rate (with cosine annealing) |
| `weight_decay` | loguniform(1e-4, 1e-2) | L2 regularization |
| `batch` | choice([8, 16, 32]) | Batch size |

**Command:**
```bash
cd fire-pipeline

uv run python train.py \
  --patches-dir ../patches \
  --output-dir ./output/yolo \
  --tune true \
  --tune-target yolo \
  --tune-samples 10 \
  --tune-top-k 3 \
  --epochs 50 \
  --wandb \
  --project fire-detection \
  --results-csv ./training_results.csv
```

**Estimated time:** ~10 trials. YOLO is the slowest per trial (full Ultralytics pipeline). Expect 6–10 hours on GPU.

---

## 7. Where to find the results

### W&B
Go to [wandb.ai](https://wandb.ai) → project `fire-detection`.
You will see exactly `tune-top-k` runs per model (3 by default), with names like:
- `scratch-tune-top1-lr2.3e-04-wd5.1e-05-do0.20`
- `unet-scratch-tune-top1-lr1.8e-04-wd3.2e-05-bs16`
- `yolo-tune-top1-lr3.4e-03-wd2.1e-03-bs16`

### CSV (`training_results.csv`)
Each re-training appends one row to the CSV with the main metrics.
You can open it with Excel, Google Sheets, or pandas:

```python
import pandas as pd
df = pd.read_csv("fire-pipeline/training_results.csv")
print(df[["model_name", "wandb_run_name", "f1", "val_recall", "val_precision"]])
```

### Checkpoints
Best models are saved at:
```
output/
├── scratch/
│   ├── best_1/checkpoints/best_model.pt
│   ├── best_2/checkpoints/best_model.pt
│   └── best_3/checkpoints/best_model.pt
├── unet_scratch/
│   ├── best_1/checkpoints/best_model.pt
│   └── ...
└── yolo/
    ├── best_1/weights/best.pt
    └── ...
```

---

*Generated for the AIDLFire project — Fire Detection Pipeline*
